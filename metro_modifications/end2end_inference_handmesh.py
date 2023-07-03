"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as osp
import torch
import torchvision.models as models
import numpy as np
import cv2
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from metro.modeling.bert import BertConfig, METRO
from metro.modeling.bert import METRO_Hand_Network as METRO_Network
from metro.modeling._mano import MANO, Mesh
from metro.modeling.hrnet.hrnet_cls_net import get_cls_net
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.hrnet.config import update_config as hrnet_update_config
import metro.modeling.data.config as cfg

from metro.utils.renderer import Renderer, visualize_reconstruction_no_text, visualize_reconstruction_and_att_local
from metro.utils.geometric_layers import orthographic_projection
from metro.utils.logger import setup_logger
from metro.utils.miscellaneous import mkdir, set_seed

from PIL import Image
from torchvision import transforms
# Optimization and visualization utils
from metro.hand_utils.hand_utils import (optimize_for_mano_param, optimize_for_mano_arm_param, optimize_for_nimble_param,
                                         smooth_sequence, smooth_sequence_nimble, batch_to_tensor_device)

###### SMPLX with right arm only ######
SMPLX_PATH = "../../photometric_hand/arm/smplx/"
#######################################
sys.path.insert(0, SMPLX_PATH)
import smplx


transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

RESOLUTION = 448  # 224
transform_visualize = transforms.Compose([           
                    transforms.Resize(RESOLUTION),
                    transforms.CenterCrop(RESOLUTION),
                    transforms.ToTensor()])


def get_mano_arm_model(smplx_models_path):
    ### Get SMPLH
    model_folder = smplx_models_path
    model = smplx.create(model_folder, model_type='smplxarm',
                         gender='neutral', use_face_contour=False,
                         num_betas=10, use_pca=False,
                         num_expression_coeffs=10,
                         ext='npz')
    # global_orient, transl
    return model.to('cuda')


def get_nimble_model():
    nimble_dir = "../../photometric_hand/nimble/"
    sys.path.insert(0, nimble_dir)
    from NIMBLELayer import NIMBLELayer
    pm_dict_name = os.path.join(nimble_dir, r"assets/NIMBLE_DICT_9137.pkl")
    tex_dict_name = os.path.join(nimble_dir, r"assets/NIMBLE_TEX_DICT.pkl")
    device = 'cuda'
    # Load model
    pm_dict = np.load(pm_dict_name, allow_pickle=True)
    pm_dict = batch_to_tensor_device(pm_dict, device)
    tex_dict = np.load(tex_dict_name, allow_pickle=True)
    tex_dict = batch_to_tensor_device(tex_dict, device)
    textured_pkl = "assets/NIMBLE_TEX_FUV.pkl"
    f_uv = np.load(os.path.join(nimble_dir, textured_pkl), allow_pickle=True)

    nimble_mano_vreg = np.load(os.path.join(nimble_dir, "assets/NIMBLE_MANO_VREG.pkl"), allow_pickle=True)
    nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
    # import smplx
    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    return nlayer


def fill_img_background(in_img, mask=None):
    background = Image.new("RGB", in_img.size, (255, 255, 255))
    if mask is None:
        background.paste(in_img, mask=in_img.split()[3])
    else:
        background.paste(in_img, mask=mask)
    return background


def save_cropped_image(in_img, idx, cropped_img_dir, full_size_img_dir, mask_dir):
    mask = np.array(in_img)
    mask = mask[:, :, 3]
    mask = transforms.ToPILImage()(mask)
    mask_visual = transform_visualize(mask)
    mask_pil = transforms.ToPILImage()(mask_visual)
    mask_pil.save(osp.join(mask_dir, '%04d_mask.jpg' % idx))

    full_img = Image.open(osp.join(full_size_img_dir, "%04d.png" % idx))
    full_img = transform_visualize(full_img)
    full_img_mask = fill_img_background(transforms.ToPILImage()(full_img), mask_pil)
    full_img_mask.save(osp.join(cropped_img_dir, '%04d.jpg' % idx))


def get_mano_vert_colors(mano_layer):
    device = "cuda"
    mano_layer = mano_layer.to(device)
    batch_size = 1
    # Model para initialization
    shape = torch.zeros(batch_size, 10).to(device)
    rot = torch.zeros(batch_size, 3).to(device)
    pose = torch.zeros(batch_size, 45).to(device)
    trans = torch.zeros(batch_size, 3).to(device)

    mano_vertices, _ = mano_layer(torch.cat((rot, pose), 1), shape, trans)
    mano_vertices = mano_vertices[0].detach().cpu().numpy()
    scaler = MinMaxScaler()
    scaler.fit(mano_vertices)
    vertex_colors = scaler.transform(mano_vertices)
    aa = np.ones([778, 3]) * np.array([173, 216, 230]) / 255.
    vertex_colors = aa
    return vertex_colors


def run_inference(args, image_list, _metro_network, mano, renderer, mesh_sampler):
    args.fit_arm = not args.hand_only
    args.fit_nimble = False
    args.smooth_only = False
    if args.fit_arm:
        smplx_model_path = os.path.join(SMPLX_PATH, 'models')
        mano_arm_layer = get_mano_arm_model(smplx_model_path)
    elif args.fit_nimble:
        nimble_layer = get_nimble_model()

    prefix_dir = args.image_file_or_path
    # In case we want to run multiple subjects and multiple sequences at the same time
    subject_list = ["syn"]         # set to [""] for single subject
    sequence_list = ["wound_test"] # set to [""] for single sequence

    # Crop images if needed.
    # This is for images from Unscreen where the background is empty.
    # We obtain the segmentation mask from the empty pixels. The image is then resized to 448x448.
    if args.do_crop:
        for ii in subject_list:
            for jj in sequence_list:
                seq_prefix = osp.join(ii, jj)
                # Unscreen image with empty background
                image_dir = osp.join(prefix_dir, seq_prefix, "unscreen")
                # Image directory after cropping
                cropped_img_dir = osp.join(prefix_dir, seq_prefix, "unscreen_cropped")
                # Original image directory
                ori_img_dir = osp.join(prefix_dir, seq_prefix, "image")
                # Output mask directory
                mask_dir = osp.join(prefix_dir, seq_prefix, "mask")

                if os.path.isdir(cropped_img_dir):
                    print("- Skip %s. Images already cropped" % image_dir)
                    continue
                os.makedirs(cropped_img_dir, exist_ok=True)

                image_list = []
                # List images in the directory
                for filename in os.listdir(image_dir):
                    if ((filename.endswith(".png") or filename.endswith(".jpg"))
                        and 'pred' not in filename and 'mask' not in filename):
                        image_list.append(os.path.join(image_dir, filename))
                image_list.sort()

                for image_file in image_list:
                    # For Unscreen
                    idx = int(os.path.basename(image_file)[-8:-4])
                    img = Image.open(image_file)
                    save_cropped_image(img, idx, cropped_img_dir, ori_img_dir, mask_dir)

    for ii in subject_list:
        for jj in sequence_list:
            seq_prefix = osp.join(ii, jj)
            # Assume the image is already cropped
            image_dir = osp.join(prefix_dir, seq_prefix, "unscreen_cropped")
            image_list = []
            # List images in the directory
            for filename in os.listdir(image_dir):
                if ((filename.endswith(".png") or filename.endswith(".jpg"))
                    and 'pred' not in filename and 'mask' not in filename):
                    image_list.append(os.path.join(image_dir, filename))
            image_list.sort()
            
            output_prefix = "nimble_" if args.fit_nimble else ""
            cropped_img_dir =     osp.join(prefix_dir, seq_prefix, "unscreen_cropped")
            full_img_dir =        osp.join(prefix_dir, seq_prefix, "image")
            mask_dir =            osp.join(prefix_dir, seq_prefix, "mask")
            mano_fit_output_dir = osp.join(prefix_dir, seq_prefix, output_prefix + "metro_mano")
            mano_fit_smooth_dir = osp.join(prefix_dir, seq_prefix, output_prefix + "metro_mano_smooth")
            outdir =              osp.join(prefix_dir, seq_prefix, output_prefix + "metro_image")
            mano_fit_img_dir =    osp.join(prefix_dir, seq_prefix, output_prefix + "mano_fit_image")

            os.makedirs(outdir, exist_ok=True)
            os.makedirs(mano_fit_output_dir, exist_ok=True)
            os.makedirs(mano_fit_smooth_dir, exist_ok=True)
            os.makedirs(mano_fit_img_dir, exist_ok=True)
            os.makedirs(cropped_img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            if args.smooth_only:
                if args.fit_arm:
                    smooth_sequence(renderer, mano_arm_layer, mano_fit_output_dir, mano_fit_smooth_dir, use_smplx_arm=True, img_res=RESOLUTION)
                elif args.fit_nimble:
                    smooth_sequence_nimble(renderer, nimble_layer, mano_fit_output_dir, mano_fit_smooth_dir, mano.get_layer().th_J_regressor, img_res=RESOLUTION)
                else:
                    smooth_sequence(renderer, mano.get_layer(), mano_fit_output_dir, mano_fit_smooth_dir, use_smplx_arm=False, img_res=RESOLUTION)
                continue

            # switch to evaluate mode
            _metro_network.eval()
            # Get the vertex colors for visualization
            vert_rgb = get_mano_vert_colors(mano.get_layer())

            for image_file in image_list:
                idx = int(os.path.basename(image_file)[-8:-4])
                att_all = []
                img = Image.open(image_file)
                img_tensor = transform(img)
                img_visual = transform_visualize(img)

                # Prepare white image for mano visualization
                white_img = Image.new("RGB", img.size, (255, 255, 255))
                white_img_visual = transform_visualize(white_img)
                white_img_visual = torch.unsqueeze(white_img_visual, 0).cuda() 

                batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
                batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
                pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = _metro_network(batch_imgs, mano, mesh_sampler)       
                # obtain 3d joints from full mesh
                pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
                pred_3d_pelvis = pred_3d_joints_from_mesh[:,cfg.J_NAME.index('Wrist'),:]
                pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]
                pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]
                
                # Fit hand model to the predicted surfaces
                if args.fit_nimble:
                    pred_mano = optimize_for_nimble_param(pred_vertices, nimble_layer, mano.get_layer().th_J_regressor.numpy()) # , mano_faces=mano_faces)
                elif args.fit_arm:
                    pred_mano = optimize_for_mano_arm_param(pred_vertices, mano_arm_layer, idx, vert_rgb)
                else:
                    pred_mano = optimize_for_mano_param(pred_vertices, mano.get_layer(), idx, vert_rgb)

                pred_mano['cam'] = pred_camera.detach().cpu().numpy()
                pred_mano['joints_metro'] = pred_3d_joints_from_mesh.detach().cpu().numpy() * 1000.0
                pred_mano['verts_metro'] = pred_vertices.detach().cpu().numpy() * 1000.0
                
                with open(osp.join(mano_fit_output_dir, "%04d_mano.pkl" % idx), 'wb') as f: 
                    pickle.dump(pred_mano, f)

                # save attantion
                att_max_value = att[-1]
                att_cpu = np.asarray(att_max_value.cpu().detach())
                att_all.append(att_cpu)

                # obtain 3d joints, which are regressed from the full mesh
                pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
                # obtain 2d joints, which are projected from 3d joints of mesh
                pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
                pred_2d_coarse_vertices_from_mesh = orthographic_projection(pred_vertices_sub.contiguous(), pred_camera.contiguous())

                visual_imgs_att = visualize_mesh_and_attention( renderer, white_img_visual[0], # batch_visual_imgs[0],
                                                            pred_vertices[0].detach(), 
                                                            pred_vertices_sub[0].detach(), 
                                                            pred_2d_coarse_vertices_from_mesh[0].detach(),
                                                            pred_2d_joints_from_mesh[0].detach(),
                                                            pred_camera.detach(),
                                                            att[-1][0].detach(), 
                                                            vert_rgb)

                visual_imgs = visual_imgs_att.transpose(1,2,0)
                visual_imgs = np.asarray(visual_imgs)

                # show optimized MANO img
                visual_imgs_mano = visualize_mesh_and_attention( renderer, batch_visual_imgs[0],
                                                            torch.tensor(pred_mano['verts'][0]) / 1000.0,
                                                            pred_vertices_sub[0].detach(), 
                                                            pred_2d_coarse_vertices_from_mesh[0].detach(),
                                                            pred_2d_joints_from_mesh[0].detach(),
                                                            pred_camera.detach(),
                                                            att[-1][0].detach(),
                                                            vert_rgb)

                visual_imgs_mano = visual_imgs_mano.transpose(1,2,0)
                visual_imgs_mano = np.asarray(visual_imgs_mano)
                        
                vis_fname = image_file[:-4] + '_metro_pred.jpg'
                vis_fname = osp.join(outdir, "%04d" % idx + '_metro_pred.jpg')
                vis_fname_mano = osp.join(mano_fit_img_dir, "mano_fit_" + "%04d" % idx + '_metro_pred.jpg')
                print('save to ', vis_fname)
                cv2.imwrite(vis_fname, np.asarray(visual_imgs[:,:,::-1]*255))
                cv2.imwrite(vis_fname_mano, np.asarray(visual_imgs_mano[:,:,::-1]*255))
            
            # Done per-frame optimization, start sequence smoothing
            print(" --- Sequence Smoothing --- ")
            if args.fit_arm:
                smooth_sequence(renderer, mano_arm_layer, mano_fit_output_dir, mano_fit_smooth_dir, use_smplx_arm=True, img_res=RESOLUTION)
            elif args.fit_nimble:
                smooth_sequence_nimble(renderer, nimble_layer, mano_fit_output_dir, mano_fit_smooth_dir, mano.get_layer().th_J_regressor, img_res=RESOLUTION)
            else:
                smooth_sequence(renderer, mano.get_layer(), mano_fit_output_dir, mano_fit_smooth_dir, use_smplx_arm=False, img_res=RESOLUTION)
    return 


def visualize_mesh_and_attention( renderer, images,
                    pred_vertices_full,
                    pred_vertices, 
                    pred_2d_vertices,
                    pred_2d_joints,
                    pred_camera,
                    attention,
                    vert_colors):

    """Tensorboard logging."""
    
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full.cpu().numpy() 
    vertices = pred_vertices.cpu().numpy()
    vertices_2d = pred_2d_vertices.cpu().numpy()
    joints_2d = pred_2d_joints.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    att = attention.cpu().numpy()
    # Visualize reconstruction and attention
    rend_img = visualize_reconstruction_and_att_local(img, RESOLUTION, vertices_full, vertices, vertices_2d, cam, renderer, joints_2d, att, color=vert_colors) # color='pink')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img


def visualize_mesh_no_text( renderer,
                    images,
                    pred_vertices, 
                    pred_camera):
    """Tensorboard logging."""
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices = pred_vertices.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    # Visualize reconstruction only
    rend_img = visualize_reconstruction_no_text(img, RESOLUTION, vertices, cam, renderer, color='hand')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./test_images/hand', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                        help="The Image Feature Dimension.")   
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    
    parser.add_argument('--do_crop', action='store_true',
                        help="only crop the image from unscreen")
    parser.add_argument('--hand_only', action='store_true',
                        help="use model with hand only without arm mesh")
    parser.add_argument("--dataset", type=str, default='capture', 
                        help="capture or hanco or interhand")


    args = parser.parse_args()
    return args


def main(args):
    # python ./metro/tools/end2end_inference_handmesh.py --resume_checkpoint ./models/metro_release/metro_hand_state_dict.bin
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    mkdir(args.output_dir)
    logger = setup_logger("METRO Inference", args.output_dir, 0)
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.cuda()
    mesh_sampler = Mesh()
    # Renderer for visualization
    renderer = Renderer(faces=mano_model.face)

    # Load pretrained model    
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        # Build model from scratch, and load weights from state_dict.bin
        trans_encoder = []
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3]
        # init three transformer encoders in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, METRO
            config = config_class.from_pretrained(args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*4)

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Transformers total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        _metro_network = METRO_Network(args, config, backbone, trans_encoder)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict

    # update configs to enable attention outputs
    setattr(_metro_network.trans_encoder[-1].config,'output_attentions', True)
    setattr(_metro_network.trans_encoder[-1].config,'output_hidden_states', True)
    _metro_network.trans_encoder[-1].bert.encoder.output_attentions = True
    _metro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _metro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_metro_network.trans_encoder[-1].config,'device', args.device)

    _metro_network.to(args.device)
    logger.info("Run inference")

    image_list = []
    # Skip this part as the image paths will be defined in the function
    # if not args.image_file_or_path:
    #     raise ValueError("image_file_or_path not specified")
    # if op.isfile(args.image_file_or_path):
    #     image_list = [args.image_file_or_path]
    # elif op.isdir(args.image_file_or_path):
    #     # should be a path with images only
    #     for filename in os.listdir(args.image_file_or_path):
    #         if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
    #             image_list.append(args.image_file_or_path+'/'+filename) 
    # else:
    #     raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    run_inference(args, image_list, _metro_network, mano_model, renderer, mesh_sampler)    

if __name__ == "__main__":
    args = parse_args()
    main(args)
