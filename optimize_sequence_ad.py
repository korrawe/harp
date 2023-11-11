from json import dump
import os
import yaml
import shutil
from matplotlib import image
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # matplotlib.use('TkAgg')
# from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import renderer.renderer_helper as renderer_helper
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.ops import SubdivideMeshes, taubin_smoothing
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

from model.vgg import Vgg16Features
from loss.arap import arap_loss
from loss.kps_loss import kps_loss
from loss.texture_reg import albedo_reg, normal_reg

from utils.data_util import load_multiple_sequences
from utils.eval_util import image_eval, load_gt_vert, align_w_scale
from utils.visualize import render_360, concat_image_in_dir, render_360_light, render_image, prepare_mesh, prepare_materials, render_image_with_RT
from utils import file_utils, hand_model_utils, config_utils


def show_img_pair(ypred_np, ytrue_np, step=-1, silhouette=False, save_img_dir=None, prefix=""):
    fig = plt.figure(figsize=(10, 10)) 
    idx_list = [a for a in range(9)]

    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(3, 3, 3*i + j+1)
            ax.axis('off')
            idx = 3*i + j
            if idx_list[idx] >= len(ypred_np):
                break
            if silhouette:
                overlay = np.zeros([*ytrue_np[idx_list[idx]].shape[:2], 3])
                overlay[:, :, 0] = ytrue_np[idx_list[idx]]
                overlay[:, :, 2] = ypred_np[idx_list[idx]]
                ax.imshow(overlay)
            else:
                ax.imshow(ypred_np[idx_list[idx]])
    
    if save_img_dir is not None:
        if silhouette:
            fig_out_dir = save_img_dir + prefix + "sil_%04d.jpg" % (step)
        else:
            fig_out_dir = save_img_dir + prefix + "%04d.jpg" % (step)
        plt.savefig(fig_out_dir)
    else:
        plt.show()
    plt.close()


def get_mesh_subdivider(hand_layer, use_arm=False, device='cuda'):
    if use_arm:
        hand_verts, arm_joints = hand_layer(betas=torch.zeros([1, 10]).to(device), global_orient=torch.zeros([1, 3]).to(device),
            transl=torch.zeros([1, 3]).to(device), right_hand_pose=torch.zeros([1, 45]).to(device), return_type='mano_w_arm')
        faces = hand_layer.right_arm_faces_tensor
    else:
        hand_verts, hand_joints = hand_layer(torch.zeros([1, 48]).to(device),
            torch.zeros([1, 10]).to(device),
            torch.zeros([1, 3]).to(device))

        faces = hand_layer.th_faces
    
    mesh_mano = Meshes(hand_verts.to(device), faces.repeat(1, 1, 1).to(device))
    # use only the first mesh as template
    mesh_subdivider = SubdivideMeshes(meshes=mesh_mano)
    # # Sample code to export mesh
    # sub_mesh = mesh_subdivider(mesh_mano)
    # new_verts = sub_mesh.verts_padded()
    # new_faces = sub_mesh.faces_padded()
    # import trimesh
    # tri_m = trimesh.Trimesh(vertices=new_verts[0].detach().cpu().numpy(), faces=new_faces[0].detach().cpu().numpy())
    # tri_m.export("./data/template_mano_subdivide/template_mano.obj")
    return mesh_subdivider


def loss_l1_weighted(y_true, y_pred, weights):
    loss = torch.abs(y_true - y_pred)
    loss = loss * weights.unsqueeze(-1)
    return loss.mean()

def visualize_val(val_images_dataloader, epoch_id, device, params, val_params, configs, hand_layer,
                  mesh_subdivider, opt_app, use_verts_textures, GLOBAL_POSE, SHARED_TEXTURE):
    with torch.no_grad():
        for (fid, y_true, y_sil_true, _) in val_images_dataloader:

            print("epoch: %d" % epoch_id)
            for param_group in opt_app.param_groups:
                print("learning rate", param_group['lr'])
            # print("learning rate", lr)
            y_sil_true = y_sil_true.squeeze(-1).to(device)
            y_true = y_true.to(device)

            cur_batch_size = fid.shape[0]
            if configs["share_light_position"]:
                light_positions = params['light_positions'][0].repeat(cur_batch_size, 1)
            else:
                light_positions = params['light_positions'][fid]
            phong_renderer, silhouette_renderer, normal_renderer = renderer_helper.get_renderers(image_size=configs['img_size'], 
                light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, device=device)

            # Copy optimized params
            ### feature
            val_params['shape'] = params['shape']
            val_params['pose'] = params['pose']
            val_params['wrist_pose'] = params['wrist_pose']
            val_params['mesh_faces'] = params['mesh_faces']
            val_params['verts_rgb'] = params['verts_rgb']
            # UV MAP
            if not use_verts_textures:
                val_params['verts_uvs'] = params['verts_uvs']
                val_params['faces_uvs'] = params['faces_uvs']
                if configs['model_type'] == 'nimble':
                    val_params['nimble_tex'] = params['nimble_tex']
                else:
                    val_params['texture'] = params['texture']
                    if 'normal_map' in params:
                        val_params['normal_map'] = params['normal_map']

            val_params['verts_disps'] = params['verts_disps']            
            # Meshes
            hand_joints, hand_verts, faces, textures = prepare_mesh(val_params, fid, hand_layer, use_verts_textures, mesh_subdivider,
                global_pose=GLOBAL_POSE, configs=configs, shared_texture=SHARED_TEXTURE, use_arm=configs['use_arm'], device=device)
            meshes = Meshes(hand_verts, faces, textures)
            cam = val_params['cam'][fid]
            materials_properties = prepare_materials(val_params, fid.shape[0])
            # RGB UV
            if configs["self_shadow"]:
                light_R, light_T, cam_R, cam_T = renderer_helper.process_info_for_shadow(cam, light_positions, hand_verts.mean(1), 
                                                    image_size=configs['img_size'], focal_length=configs['focal_length'])
                shadow_renderer = renderer_helper.get_shadow_renderers(image_size=configs['img_size'], 
                    light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, 
                    amb_ratio=nn.Sigmoid()(params['amb_ratio']), device=device)
                y_pred = render_image_with_RT(meshes, light_T, light_R, cam_T, cam_R,
                            cur_batch_size, shadow_renderer, configs['img_size'], configs['focal_length'], silhouette=False, 
                            materials_properties=materials_properties)
            else:
                y_pred = render_image(meshes, cam, cur_batch_size, phong_renderer, configs['img_size'], configs['focal_length'],
                                        silhouette=False, materials_properties=materials_properties)
            
            show_img_pair(y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), save_img_dir=configs["base_output_dir"],
                step=epoch_id, silhouette=False, prefix="val_")

            if not use_verts_textures:
                fig_out_path = os.path.join(configs["base_output_dir"], "uv_%04d.jpg" % (epoch_id))
                normal_out_path = os.path.join(configs["base_output_dir"], "normal_%04d.jpg" % (epoch_id))

                pred_texture_out = params['texture'].detach().clone().cpu().numpy()[0].clip(0,1)
                Image.fromarray(np.uint8(pred_texture_out * 255)).save(fig_out_path)
                if not configs['model_type'] == 'nimble' and not configs['model_type'] == 'html':
                    opt_normal_out = torch.nn.functional.normalize(params["normal_map"], dim=-1)
                    opt_normal_out = (opt_normal_out.detach().cpu().numpy()[0] * 0.5 + 0.5).clip(0,1)
                    Image.fromarray(np.uint8(opt_normal_out * 255)).save(normal_out_path)

            del y_pred
            break


def load_uv_mask(configs, uv_size):
    uv_mask_path = configs["uv_mask"]
    uv_mask_pil = Image.open(uv_mask_path).convert('L').resize(uv_size)
    uv_mask = np.asarray(uv_mask_pil) / 255
    return torch.tensor(uv_mask)


def init_params(input_params, VERT_DISPS, VERT_DISPS_NORMALS, VERTS_COLOR, mano_faces, verts_textures, VERTS_UVS=None, FACES_UVS=None,
        model_type="harp", use_arm=False, configs=None, device='cuda'):
    params = {}
    # MANO
    params['trans'] = torch.nn.Parameter(input_params['trans'].detach().clone(), requires_grad=True)
    params['pose'] = torch.nn.Parameter(input_params['pose'].detach().clone(), requires_grad=True)
    params['rot'] = torch.nn.Parameter(input_params['rot'].detach().clone(), requires_grad=True)
    # Shape parameter must be the same for the entire sequence. Initialized with an average of the predicted shape
    params['shape'] = torch.nn.Parameter(input_params['shape'].mean(dim=0).detach().clone(), requires_grad=True)  # .to(device)
    # Use params['shape'].repeat([batch_size, 1]) during optimization
    # Arm
    params['wrist_pose'] = torch.nn.Parameter(torch.zeros([params['pose'].shape[0], 3]), requires_grad=True)
    # Initial joint prediction
    params['init_joints'] = input_params['joints']

    # Vertex displacement
    params['verts_disps'] = None
    if model_type == "html":
        N_MESH_VERTS = 778
    elif model_type == "nimble":
        N_MESH_VERTS = 5990
    elif use_arm:
        N_MESH_VERTS = 4083
    else:
        N_MESH_VERTS = 3093 # after 4-way subdivision from 778
    if VERT_DISPS:
        if VERT_DISPS_NORMALS:
            params['verts_disps'] = torch.nn.Parameter(torch.Tensor(np.zeros([N_MESH_VERTS, 1])).to(device), requires_grad=True)
        else:
            params['verts_disps'] = torch.nn.Parameter(torch.Tensor(np.zeros([N_MESH_VERTS, 3])).to(device), requires_grad=True)
    else:
        params['verts_disps'] = torch.nn.Parameter(torch.Tensor(np.zeros([N_MESH_VERTS, 1])).to(device), requires_grad=True)

    # Vertex colors
    if VERTS_COLOR is not None:
        verts_rgb_init = torch.from_numpy(VERTS_COLOR)
    else:
        verts_rgb_init = torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).tile([778, 1])
    params['verts_rgb'] = torch.nn.Parameter(verts_rgb_init.detach().clone(), requires_grad=True)
    # UV MAP
    if not verts_textures:
        params['verts_uvs'] = VERTS_UVS
        params['faces_uvs'] = FACES_UVS
        
        if model_type == "html":
            params['html_texture'] =  torch.nn.Parameter(torch.zeros([1, 101]), requires_grad=True)
            params['texture'] =  torch.nn.Parameter(torch.tensor([232, 190, 172]).repeat(1, 512, 512, 1) / 255., requires_grad=True)
            params['uv_mask'] = None
        elif model_type == "nimble":
            params['nimble_tex'] =  torch.nn.Parameter(torch.zeros([1, 10]), requires_grad=True)
            params['uv_mask'] = None
        else:
            # Initialized with Skin color
            params['texture'] =  torch.nn.Parameter(torch.tensor([232, 190, 172]).repeat(1, 512, 512, 1) / 255., requires_grad=True)               
            # Prepare UV mask according to texture size
            params['uv_mask'] = load_uv_mask(configs, params['texture'].shape[1:3])
            # Normal map. Initialize in normal space (not color space)
            params['normal_map'] =  torch.nn.Parameter(torch.tensor([0.0, 0.0, 1.0]).repeat(1, 512, 512, 1), requires_grad=True)

    # Light position for each frames
    total_frame = input_params['cam'].shape[0]
    starting_lights = torch.tensor(((-0.5, -0.5 , -0.5),)).repeat(total_frame, 1)
    params['light_positions'] = torch.nn.Parameter(starting_lights, requires_grad=True)
    # Ratio of ambient_light out of (ambient_light + diffuse_light). No specular. Before sigmoid. Need to pass through sigmoid before use.
    params['amb_ratio'] = torch.nn.Parameter(torch.tensor(0.4), requires_grad=True)  # Roughly 0.6 after sigmoid
    # Faces
    params['mesh_faces'] = mano_faces
    # Cameras
    params['cam'] = torch.nn.Parameter(input_params['cam'].detach().clone(), requires_grad=True)
    return params


def get_optimizers(params, configs):
    pose_params = [params['pose'], params['cam']]
    if configs["use_vert_disp"]:
        shape_params = [params['verts_disps'], params['shape']]
    else:
        shape_params = [params['shape']]

    if configs["model_type"] == "nimble":
        shape_params = [params['shape']]

    if configs["known_appearance"]:
        if configs["use_arm"] and configs["opt_arm_pose"]:
            opt_coarse = torch.optim.Adam([
                    {'params': pose_params, 'lr': 1.0e-3},
                    {'params': [params['wrist_pose'], params['rot']], 'lr': 1.0e-3},
                ])
        else:
            opt_coarse = torch.optim.Adam([
                    {'params': pose_params, 'lr': 1.0e-3},
                ])
    else:
        if configs['model_type'] == 'nimble':
            opt_coarse = torch.optim.Adam([
                    {'params': pose_params, 'lr': 1.0e-3},
                    {'params': [params['wrist_pose'], params['rot']], 'lr': 1.0e-2},
                ])
        elif configs["use_arm"] and configs["opt_arm_pose"]:
            opt_coarse = torch.optim.Adam([
                    {'params': pose_params, 'lr': 1.0e-3},
                    {'params': [params['wrist_pose'], params['rot']], 'lr': 1.0e-3},
                    {'params': shape_params, 'lr': 1.0e-3}
                ])
        else:
            opt_coarse = torch.optim.Adam([
                    {'params': pose_params, 'lr': 1.0e-3},
                    {'params': shape_params, 'lr': 1.0e-3}
                ])

    common_app_params = [params['light_positions'], params['amb_ratio']]

    opt_param_app = [*common_app_params]
    app_lr = 1.0e-2
    if configs["model_type"] == "html":
        opt_param_app.append(params['html_texture'])

    elif configs["model_type"] == "nimble":
        opt_param_app.append(params['nimble_tex'])

    else:
        opt_param_app = [*common_app_params, params['texture'], params['normal_map']]
        app_lr = 1.0e-2

    # If the appearance is given, we don't optimize it
    if configs["known_appearance"]:
        opt_param_app = [params['light_positions'], params['amb_ratio']]
    opt_app = torch.optim.Adam(opt_param_app, lr=app_lr)
    sched_coarse = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_coarse, patience=40, verbose=True) # 30 is also good
    return opt_coarse, opt_app, sched_coarse

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 4, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        x5 = torch.sigmoid(self.conv5(x4))
        return x5



def optimize_hand_sequence(configs, input_params, images_dataset, val_params, val_images_dataset,
        hand_layer, 
        VERTS_UVS=None, FACES_UVS=None, VERTS_COLOR=None, device='cuda'):

    tf_writer = SummaryWriter(log_dir=configs["base_output_dir"])
    # Coarse optimization, including translation, rotation, pose, shape
    COARSE_OPT = True
    # Appearance optimization, including texture, lighting
    APP_OPT = True
    # Whether all frames share the same hand pose
    GLOBAL_POSE = False
    IMAGE_EVAL = True
    VERT_DISPS = configs["use_vert_disp"]
    VERT_DISPS_NORMALS = True    
    # If vertex texutere is false, use UV map
    use_verts_textures = False
    SHARED_TEXTURE = True

    LOG_IMGAGE = True
    # Get mesh faces
    # if configs["model_type"] == "html":
    #     SUB_DIV_MESH = False
    #     html_layer = configs["html_layer"]
    #     mano_faces = hand_layer.th_faces
    # elif configs["model_type"] == "nimble":
    #     SUB_DIV_MESH = False
    #     mano_faces = hand_layer.skin_f
    if configs["use_arm"]:
        SUB_DIV_MESH = True
        mano_faces = hand_layer.right_arm_faces_tensor
    else:
        SUB_DIV_MESH = True
        mano_faces = hand_layer.th_faces

    if SUB_DIV_MESH:
        mesh_subdivider = get_mesh_subdivider(hand_layer, use_arm=configs['use_arm'], device=device)
    else:
        mesh_subdivider = None

    img_size = configs['img_size']  # 448
    base_output_dir = configs["base_output_dir"]

    if len(configs["start_from"]) > 0:
        if configs["known_appearance"] and configs["pose_already_opt"]:
            params = file_utils.load_result(configs["start_from"], test=configs["pose_already_opt"])
        else:
            params = file_utils.load_result(configs["start_from"])
            if configs["known_appearance"]:
                # For known appearance but new pose. Init pose parameters for optimization
                params['trans'] = torch.nn.Parameter(input_params['trans'].detach().clone(), requires_grad=True)
                params['pose'] = torch.nn.Parameter(input_params['pose'].detach().clone(), requires_grad=True)
                params['rot'] = torch.nn.Parameter(input_params['rot'].detach().clone(), requires_grad=True)
                params['cam'] = torch.nn.Parameter(input_params['cam'].detach().clone(), requires_grad=True)

        # smooth poses by interpolating every 5 frame
        temp_pose = params['pose'].detach().clone()
        for i in range(params['pose'].shape[0] // 30 - 1):
            for j in range(30):
                temp_pose[i * 30 + j] = ((30-j) * params['pose'][i*30] + j*params['pose'][i*30 + 30]) / 30.0
        params['pose'] = torch.nn.Parameter(temp_pose, requires_grad=True) 

        # temp_trans = params['trans'].detach().clone()
        temp_trans = torch.zeros_like(params['trans']) + params['trans'].mean(0)
        params['trans'] = torch.nn.Parameter(temp_trans, requires_grad=True) 

        # temp_rot = params['rot'].detach().clone()
        temp_rot = torch.zeros_like(params['rot']) + params['rot'].mean(0)
        params['rot'] = torch.nn.Parameter(temp_rot, requires_grad=True)

        if not 'wrist_pose' in params:
            params['wrist_pose'] = torch.nn.Parameter(torch.zeros([params['pose'].shape[0], 3]), requires_grad=True)
        # Ambient to diffuse ratio
        if not 'amb_ratio' in params:
            params['amb_ratio'] = torch.nn.Parameter(torch.tensor(0.4), requires_grad=True)  # Roughly 0.6 after sigmoid

        if not "normal_map" in params:
            params["normal_map"] =  torch.nn.Parameter(torch.tensor([0.0, 0.0, 1.0]).repeat(1, 512, 512, 1), requires_grad=True)
        
    else:
        params = init_params(input_params, VERT_DISPS, VERT_DISPS_NORMALS, VERTS_COLOR,  mano_faces, use_verts_textures, 
                    VERTS_UVS, FACES_UVS, model_type=configs["model_type"], use_arm=configs["use_arm"], configs=configs, device=device)
    #### End initialization ####

    batch_size = 18 # 19 # 10 # 30 # 2 # 16
    val_batch = 9
    
    images_dataloader = DataLoader(images_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_images_dataloader = DataLoader(val_images_dataset, batch_size=val_batch, shuffle=True, num_workers=20)  # Shuffle val

    ### Define loss and optimizer
    l1_loss = torch.nn.L1Loss()
    # For vgg_loss
    vgg = Vgg16Features(layers_weights = [1, 1/16, 1/8, 1/4, 1]).to(device)

    # Get optimizers
    opt_coarse, opt_app, sched_coarse = get_optimizers(params, configs)

    discriminator=Discriminator().to(device)
    discriminator_optimizer=torch.optim.Adam(discriminator.parameters(),lr=0.001)
    discriminator_scheduler=torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=2000, gamma=0.9, last_epoch=-1)
    # Loss weights
    losses = {"silhouette":  {"weight": 7.0, "values": []}, ## 7.0 # 5.0 ## 4.0
            "kps_anchor":    {"weight": 10.0, "values": []}, # 10.0 # 15.0 ## 4.0 # 5.0 (for 1,4,6)
            "vert_disp_reg": {"weight": 2.0, "values": []}, # 2.0 ## 4.0
            "normal":        {"weight": 0.1, "values": []}, # 0.1 ## 0.01
            "laplacian":     {"weight": 4.0, "values": []}, # 4.0 ## 4.0
            "arap":          {"weight": 0.2, "values": []}, # 0.2 ## 0.2 # 0.5
            # Appearance
            "photo" :        {"weight": 1.0, "values": []}, # 1.0 ## 1.0
            "vgg" :          {"weight": 1.0, "values": []}, # 1.0  ##
            "albedo" :       {"weight": 0.5, "values": []}, # 0.1 # 1.0 # 0.5 ##
            "normal_reg" :   {"weight": 0.1, "values": []}, # 0.5 ##
         }

    # Get renderers
    phong_renderer, silhouette_renderer, normal_renderer = renderer_helper.get_renderers(image_size=img_size, 
        device=device, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50)
    
    # Get reference mesh. Use the mesh from the first frame.
    (fid, y_true, y_sil_true, _) = images_dataset[0]

    with torch.no_grad():
        # NOTE: The reference mesh is from the first step, it should be the mesh with mean pose instead
        hand_joints, hand_verts, faces, textures = prepare_mesh(params, torch.tensor([fid]), hand_layer, use_verts_textures, mesh_subdivider, 
        global_pose=GLOBAL_POSE, configs=configs, shared_texture=SHARED_TEXTURE, use_arm=configs['use_arm'], device=device)
        ref_meshes = Meshes(hand_verts, faces, textures)

    ### Training Loop ###
    epoch_id = 0
    n_iter = 0
    for epoch_id in tqdm(range(0, configs["total_epoch"])): # 311 # 1501):  # 201
        frame_count = 0

        # print("Epoch: %d" % epoch_id)
        epoch_loss = 0.0
        mini_batch_count = 0
        for (fid, y_true, y_sil_true, y_sil_true_col) in images_dataloader:
            cur_batch_size = fid.shape[0]
            y_sil_true = y_sil_true.squeeze(-1).to(device)
            y_sil_true_col = y_sil_true_col.squeeze(-1).to(device)
            y_true = y_true.to(device)

            # Get new shader with updated light position
            if configs["share_light_position"]:
                light_positions = params['light_positions'][0].repeat(cur_batch_size, 1)
            else:
                light_positions = params['light_positions'][fid]
            phong_renderer, silhouette_renderer, normal_renderer = renderer_helper.get_renderers(image_size=img_size, 
                light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, device=device)

            # Meshes
            hand_joints, hand_verts, faces, textures = prepare_mesh(params, fid, hand_layer, use_verts_textures, mesh_subdivider,
                global_pose=GLOBAL_POSE, configs=configs, shared_texture=SHARED_TEXTURE, device=device, use_arm=configs['use_arm'])
            meshes = Meshes(hand_verts, faces, textures)
            cam = params['cam'][fid]

            # Material properties
            materials_properties = prepare_materials(params, fid.shape[0])

            # Shihouette
            # Stop computing and updating silhouette when learning texture model
            y_sil_pred = render_image(meshes, cam, cur_batch_size, silhouette_renderer, configs['img_size'], configs['focal_length'], silhouette=True)
            
            # RGB UV
            if configs["self_shadow"]:
                # Render with self-shadow
                light_R, light_T, cam_R, cam_T = renderer_helper.process_info_for_shadow(cam, light_positions, hand_verts.mean(1), 
                                                    image_size=configs['img_size'], focal_length=configs['focal_length'])
                shadow_renderer = renderer_helper.get_shadow_renderers(image_size=img_size, 
                    light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, 
                    amb_ratio=nn.Sigmoid()(params['amb_ratio']), device=device)

                y_pred = render_image_with_RT(meshes, light_T, light_R, cam_T, cam_R,
                            cur_batch_size, shadow_renderer, configs['img_size'], configs['focal_length'], silhouette=False, 
                            materials_properties=materials_properties)
            else:
                # Render without self-shadow
                y_pred = render_image(meshes, cam, cur_batch_size, phong_renderer, configs['img_size'], configs['focal_length'], 
                                      silhouette=False, materials_properties=materials_properties)

            if LOG_IMGAGE and epoch_id % 10 == 0 and mini_batch_count == 0:
                # Value range 0 (black) -> 1 (white)
                # Log silhouette
                show_img_pair(y_sil_pred.detach().cpu().numpy(), y_sil_true.detach().cpu().numpy(), save_img_dir=base_output_dir,
                        step=epoch_id, silhouette=True, prefix="")
                # Log RGB
                show_img_pair(y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), save_img_dir=base_output_dir,
                        step=epoch_id, silhouette=False, prefix="")
                # Loss visulization
                loss_image = torch.abs(y_true * y_sil_true_col.unsqueeze(-1) - y_pred * y_sil_true_col.unsqueeze(-1))
                show_img_pair(loss_image.detach().cpu().numpy(), y_true.detach().cpu().numpy(), save_img_dir=base_output_dir,
                        step=epoch_id, silhouette=False, prefix="loss_")
            mini_batch_count += 1

            loss = {}
            # Check training stage
            # training stage: [shape only, shape and appearance, appearance]
            if epoch_id < configs["training_stage"][0]:
                COARSE_OPT = True
                APP_OPT    = False
            elif epoch_id < configs["training_stage"][0] + configs["training_stage"][1]:
                COARSE_OPT = True
                APP_OPT    = True
            else:
                COARSE_OPT = False
                APP_OPT    = True

            if COARSE_OPT:
                # Silhouette only
                loss["silhouette"] = l1_loss(y_sil_true, y_sil_pred)
                
                # Keypoint anchor
                # Anchor keypoints to the initial prediction
                if not configs["known_appearance"] and not configs["model_type"] == "nimble":
                    loss["kps_anchor"] = kps_loss(params['init_joints'][fid], hand_joints, use_arm=configs['use_arm'])
                    if torch.isnan(loss["kps_anchor"]):
                        print("Anchor loss is nan")
                        import pdb; pdb.set_trace()

                # Mesh regularizer
                # Do not apply mesh loss of test sequence (when appearance is given).
                if VERT_DISPS and (not configs["known_appearance"]):
                    if VERT_DISPS_NORMALS:
                        loss["vert_disp_reg"] = torch.sum(params['verts_disps'] ** 2.0)
                    else:
                        loss["vert_disp_reg"] = torch.sum(torch.norm(params['verts_disps'], dim=1) ** 2.0)
                    loss["laplacian"] = mesh_laplacian_smoothing(meshes)
                    loss["normal"] = mesh_normal_consistency(meshes)
                    # As rigid as possible compare to the reference mesh (currently use mesh from the first frame)
                    loss["arap"] = arap_loss(meshes, ref_meshes)
                
            if APP_OPT:
                # Photometric loss
                loss["photo"] = l1_loss(y_true * y_sil_true_col.unsqueeze(-1), y_pred * y_sil_true_col.unsqueeze(-1))

                # VGG loss
                loss["vgg"] = l1_loss(vgg((y_pred * y_sil_true_col.unsqueeze(-1)).permute(0, 3, 1, 2)), 
                                    vgg((y_true * y_sil_true_col.unsqueeze(-1)).permute(0, 3, 1, 2)))
                
                # Texture regularization
                if not configs['model_type'] == 'nimble' and not configs['model_type'] == 'html':
                    # Smooth local texture
                    loss["albedo"] = albedo_reg(params['texture'], uv_mask=params["uv_mask"], std=1.0)
                    loss["normal_reg"] = normal_reg(params['normal_map'], uv_mask=params["uv_mask"])
            
            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
                tf_writer.add_scalar(k, l * losses[k]["weight"], n_iter)
                # print("%s: %.6f" % (k, l * losses[k]["weight"]))
            
            epoch_loss += sum_loss.detach().cpu()
            tf_writer.add_scalar('total_loss', sum_loss, n_iter)
            print("total_loss = %.6f" % sum_loss)

            opt_coarse.zero_grad()
            opt_app.zero_grad()
            sum_loss.backward()
            if COARSE_OPT:
                opt_coarse.step()
            if APP_OPT:
                opt_app.step()
            
            frame_count += batch_size
            n_iter += 1
            # Delete the variables to free up memory
            del y_pred
            del loss
        
        if COARSE_OPT:
            sched_coarse.step(epoch_loss / mini_batch_count)

        print(" Epoch loss = %.6f" % (epoch_loss / mini_batch_count))
        tf_writer.add_scalar('total_loss_epoch', (epoch_loss / mini_batch_count), epoch_id)

        if epoch_id % 20 == 0:
            visualize_val(val_images_dataloader, epoch_id, device, params, val_params, configs, hand_layer,
                          mesh_subdivider, opt_app, use_verts_textures, GLOBAL_POSE, SHARED_TEXTURE)

        if epoch_id % 200 == 0 and epoch_id > 0:
            file_utils.save_result(params, base_output_dir, test=configs["known_appearance"])
    #### Done Optimization ####

    # Save results
    file_utils.save_result(params, base_output_dir, test=configs["known_appearance"])

    # Output render images after optimization
    images_dataloader = DataLoader(images_dataset, batch_size=1, shuffle=False, num_workers=20)

    # val_params['shape'] = params['shape']
    # val_params['pose'] = params['pose']
    # val_params['mesh_faces'] = params['mesh_faces']
    # val_params['verts_rgb'] = params['verts_rgb']

    # Copy appearance parameters
    if not use_verts_textures:
        val_params['verts_uvs'] = params['verts_uvs']
        val_params['faces_uvs'] = params['faces_uvs']
        val_params['texture'] = params['texture']
    if VERT_DISPS:
        val_params['verts_disps'] = params['verts_disps']

    # Result dict
    images_for_eval = {
        "ref_image": [],
        "ref_mask": [],
        "pred_image": [],
        "pred_mask": []
    }
    image_stat_list = {
        "Silhouette IoU": [],
        "L1": [],
        "LPIPS": [],
        "MS_SSIM": []
    }
    SAVE_TEXTURE = True
    if SAVE_TEXTURE:
        uv_out_dir = os.path.join(base_output_dir, "uv_out")
        os.makedirs(uv_out_dir, exist_ok=True)
        uv_mask = params["uv_mask"]
        if isinstance(uv_mask, torch.Tensor):
            uv_mask = uv_mask.cpu().numpy()

        pred_texture_out = params["texture"]
        if isinstance(pred_texture_out, torch.Tensor):
            pred_texture_out = pred_texture_out.detach().cpu().numpy()[0]
        if uv_mask is None:
            uv_mask = np.ones(pred_texture_out.shape[:2])

        texture_out_path = os.path.join(uv_out_dir, "texture.png")
        texture_out = pred_texture_out.clip(0,1) * np.expand_dims(uv_mask, 2)
        texture_out_pil = Image.fromarray(np.uint8(texture_out*255))
        texture_out_pil.save(texture_out_path)

        if 'normal_map' in params:
            opt_normal_out = torch.nn.functional.normalize(params["normal_map"], dim=-1)
            opt_normal_out = opt_normal_out.detach().cpu().numpy()
            if not uv_mask is None:
                opt_normal_out = (opt_normal_out / 2.0 + 0.5) * np.expand_dims(uv_mask, 2)
            else:
                opt_normal_out = (opt_normal_out / 2.0 + 0.5)
            normal_out_pil = Image.fromarray(np.uint8(opt_normal_out[0].clip(0,1) * 255))
            normal_out_pil.save(os.path.join(uv_out_dir, "normal_map.png"))

    joint_err_list = []
    vert_err_list = []
    batch_size = 1
    test_name = "_test" if configs["known_appearance"] else ""
    os.makedirs(base_output_dir + "rendered_after_opt" + test_name, exist_ok=True)
    # Eval
    def clear_and_create_new_eval_dict(images_for_eval):
        del images_for_eval
        images_for_eval = {
            "ref_image": [],
            "ref_mask": [],
            "pred_image": [],
            "pred_mask": []
        }
        return images_for_eval

    with torch.no_grad():
        # Loop through the dataset
        for (fid, y_true, y_sil_true, _) in tqdm(images_dataloader):
            y_sil_true = y_sil_true.squeeze(-1).to(device)
            cur_batch_size = fid.shape[0]
            if configs["share_light_position"]:
                light_positions = params['light_positions'][0].repeat(cur_batch_size, 1)
            else:
                light_positions = params['light_positions'][fid]
            phong_renderer, silhouette_renderer, normal_renderer = renderer_helper.get_renderers(image_size=img_size, 
                light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, device=device)

            # Meshes
            hand_joints, hand_verts, faces, textures = prepare_mesh(params, fid, hand_layer, use_verts_textures, mesh_subdivider,
                global_pose=GLOBAL_POSE, configs=configs, shared_texture=SHARED_TEXTURE, use_arm=configs['use_arm'], device=device)
            # Material properties
            materials_properties = prepare_materials(params, fid.shape[0])

            meshes = Meshes(hand_verts, faces, textures)

            cam = params['cam'][fid]
            cur_batch_size = fid.shape[0]
            # Render Shihouette
            y_sil_pred = render_image(meshes, cam, cur_batch_size, silhouette_renderer, configs['img_size'], configs['focal_length'], silhouette=True)
            # Render RGB UV
            if configs["self_shadow"]:
                light_R, light_T, cam_R, cam_T = renderer_helper.process_info_for_shadow(cam, light_positions, hand_verts.mean(1), 
                                                    image_size=configs['img_size'], focal_length=configs['focal_length'])
                shadow_renderer = renderer_helper.get_shadow_renderers(image_size=img_size, 
                    light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50,
                    amb_ratio=nn.Sigmoid()(params['amb_ratio']), device=device)
                y_pred = render_image_with_RT(meshes, light_T, light_R, cam_T, cam_R,
                            cur_batch_size, shadow_renderer, configs['img_size'], configs['focal_length'], silhouette=False,
                            materials_properties=materials_properties)
            else:
                y_pred = render_image(meshes, cam, cur_batch_size, phong_renderer, configs['img_size'], configs['focal_length'], silhouette=False, 
                                      materials_properties=materials_properties)
            # Render Normal
            hand_joints, hand_verts, faces, textures_normal = prepare_mesh(params, fid, hand_layer, use_verts_textures, 
                  mesh_subdivider, global_pose=GLOBAL_POSE, configs=configs, device=device, vis_normal=True, use_arm=configs['use_arm'])
            meshes_normal = Meshes(hand_verts, faces, textures_normal)
            y_pred_normal = render_image(meshes_normal, cam, cur_batch_size, normal_renderer, configs['img_size'], 
                                        configs['focal_length'], silhouette=False, materials_properties=materials_properties)

            # Select one frame to render 360 degree
            if fid[0] == 0: # 0:
                with torch.no_grad():
                    render_360(params, fid, phong_renderer, configs['img_size'], configs['focal_length'], hand_layer, configs=configs, use_arm=configs['use_arm'],
                        verts_textures=use_verts_textures, mesh_subdivider=mesh_subdivider, global_pose=GLOBAL_POSE, save_img_dir=base_output_dir)
                    
                    render_360(params, fid, normal_renderer, configs['img_size'], configs['focal_length'], hand_layer, configs=configs, render_normal=True, use_arm=configs['use_arm'],
                        verts_textures=use_verts_textures, mesh_subdivider=mesh_subdivider, global_pose=GLOBAL_POSE, save_img_dir=base_output_dir)
                    
                    concat_image_in_dir(base_output_dir + "render_360", base_output_dir + "render_360_normal", base_output_dir + "render_360_combine")

                    render_360_light(params, fid, hand_verts, faces, textures, configs['img_size'], configs['focal_length'], save_img_dir=base_output_dir)

            if IMAGE_EVAL:
                # Eval in batch
                images_for_eval["ref_image"].append(y_true.detach().cpu())
                images_for_eval["ref_mask"].append(y_sil_true.detach().cpu())
                images_for_eval["pred_image"].append(y_pred.detach().cpu())
                images_for_eval["pred_mask"].append(y_sil_pred.detach().cpu())
                eval_batch_size = 64
                if len(images_for_eval["ref_image"]) >= eval_batch_size:
                    image_stats = image_eval(images_for_eval)
                    images_for_eval = clear_and_create_new_eval_dict(images_for_eval)
                    for k,v in image_stats.items():
                        image_stat_list[k].append(v)

            # Save image comparison
            fig_out_dir = base_output_dir + "rendered_after_opt" + test_name + "/" + "%04d.jpg" % (fid)
            img_true = y_true.detach().cpu().numpy()[0].clip(0,1) * 255
            img_array = y_pred.detach().cpu().numpy()[0].clip(0,1) * 255 # .transpose(2, 0, 1)
            img_array_normal = y_pred_normal.detach().cpu().numpy()[0].clip(0,1) * 255 # .transpose(2, 0, 1)
            
            ypred_np, ytrue_np = y_sil_pred.detach().cpu().numpy(), y_sil_true.detach().cpu().numpy()
            overlay = np.zeros([*ytrue_np[0].shape[:2], 3])
            overlay[:, :, 0] = ytrue_np[0]
            overlay[:, :, 2] = ypred_np[0]
            overlay = overlay * 225

            img_array = np.concatenate([img_true, img_array, img_array_normal, overlay], axis=1)
            img_array = img_array.astype(np.uint8)
            out_img = Image.fromarray(img_array)
            out_img.save(fig_out_dir)

            # Eval mesh vertices - for data where we have GT
            if configs["eval_mesh"]:
                gt_mano_verts = load_gt_vert(fid, configs["gt_mesh_dir"], dataset="synthetic", start_from_one=True, idx_offset=500)
                hand_joints, hand_verts, faces, textures = prepare_mesh(params, fid, hand_layer, use_verts_textures, mesh_subdivider, 
                    global_pose=GLOBAL_POSE, configs=configs, shared_texture=SHARED_TEXTURE, use_arm=configs['use_arm'], device=device)
                if configs["use_arm"]:
                    pred_mano_verts = hand_verts[0, hand_layer.right_mano_idx].detach().cpu().numpy()
                else:
                    pred_mano_verts = hand_verts[0, :778].detach().cpu().numpy()

                xyz_pred_aligned = align_w_scale(gt_mano_verts, pred_mano_verts)
                err = gt_mano_verts - xyz_pred_aligned

                mean_verts_err = np.linalg.norm(err, axis=1).mean()
                # print("mean joint err: %.3f mm" % (mean_verts_err * 1000.0))
                vert_err_list.append(mean_verts_err * 1000.0)

            # export mesh
            EXPORT_MESH = False
            if EXPORT_MESH:
                with torch.no_grad():
                    meshes_2 = taubin_smoothing(meshes)
                from pytorch3d.io import save_obj
                mesh_test_path =  os.path.join(base_output_dir, 'mesh', "%04d.obj" % (fid[0]))
                os.makedirs(os.path.join(base_output_dir, 'mesh'), exist_ok=True)
                # hand_joints
                save_obj(mesh_test_path,
                        verts=meshes_2.verts_padded()[0].detach().cpu(),
                        faces=meshes.faces_padded()[0].detach().cpu(),
                        verts_uvs=meshes.textures.verts_uvs_padded()[0].detach().cpu(),
                        faces_uvs=meshes.textures.faces_uvs_padded()[0].detach().cpu(),
                        texture_map=meshes.textures.maps_padded()[0].detach().cpu().clamp(0,1)
                        )

    if IMAGE_EVAL:
        if len(images_for_eval["ref_image"]) > 0:
            image_stats = image_eval(images_for_eval)
            for k,v in image_stats.items():
                image_stat_list[k].append(v)
        
        final_stats = {}
        for k, v in image_stat_list.items():
            final_stats[k] = np.mean(v)
        
        if len(vert_err_list) > 0:
            final_stats["Procrustes-aligned vertex error (mm)"] = np.mean(vert_err_list)
            np.savetxt(os.path.join(base_output_dir, "eval_vert_mm" + test_name + ".txt"), vert_err_list)
        # vert_err_list.append(mean_verts_err * 1000.0)

        print("  -- Evaluation --")
        # Update eval stat dict
        for (k, v) in final_stats.items():
            print(" %s: %.5f" % (k, v))

        out_result_file = os.path.join(base_output_dir, "eval_results" + test_name + ".txt")
        with open(out_result_file, "w") as f_out:
            for (k, v) in final_stats.items():
                f_out.write(" %s: %.5f\n" % (k, v))

def get_config(yaml_file=None):
    config_dict = {
        "use_arm": True,
        "opt_arm_pose": False,
        "use_smooth_seq": True,
        "average_cam_sequence": False,
        "img_size": 448,  # 224,
        "focal_length": 2000.0, # 1000.0, # need to be 1000.0 * img_size / 224
        "model_type": "harp",  # ["harp", "html", "nimble"]
        "test_seq": False ,
        "known_appearance": False,
        "load_siren": False,
        "self_shadow": True,
        "pose_already_opt": False,
        "share_light_position": True,
        "eval_mesh": False,
        "use_vert_disp": True,
        "total_epoch": 301,
        # [shape, shape and appearance, appearance only]
        "training_stage": [100, 100, 100],
        "metro_output_dir": "./data/sample_data/1/",
        "image_dir": "./data/sample_data/1/",
        "train_list": ["1", "2"],
        "val_list": ["1", "2"],
        "gt_mesh_dir": "",
        # Output directory
        "base_output_dir": "exp/out_test_ad/",
        "start_from": "",
    }
    if config_dict["use_arm"]:
        # Arm Template
        config_dict["MANO_TEMPLATE"] = "template/arm/arm_template.obj"
        config_dict["uv_mask"] = "template/arm/uv_mask.png"
    else:
        # Hand Template
        config_dict["MANO_TEMPLATE"] = "template/hand/textured_hand.obj"
        config_dict["uv_mask"] = "template/hand/uv_mask.png"

    os.makedirs(config_dict["base_output_dir"], exist_ok=True)
    with open(os.path.join(config_dict["base_output_dir"], "config.yaml"), 'w') as file:
        documents = yaml.dump(config_dict, file)

    return config_dict

def main():
    # Get config
    config_dict = get_config()
    config_dict["device"] = 'cuda' 
    hand_layer, VERTS_UVS, FACES_UVS, VERTS_COLOR = hand_model_utils.load_hand_model(config_dict)
    # Load data
    # Mask in the same dir as image with format "%04d.jpg" for image, "%04d_mask.jpg" for mask
    (mano_params, images_dataset,
     val_mano_params, val_images_dataset) = load_multiple_sequences(config_dict["metro_output_dir"], config_dict["image_dir"],
        train_list=config_dict["train_list"],
        val_list=config_dict["val_list"],
        average_cam_sequence=config_dict["average_cam_sequence"], # True,
        use_smooth_seq=config_dict["use_smooth_seq"],
        model_type=config_dict['model_type'])

    print("Training size:", len(images_dataset))
    print("Val size:", len(val_images_dataset))

    optimize_hand_sequence(config_dict, mano_params, images_dataset, val_mano_params, val_images_dataset,
        hand_layer, VERTS_UVS, FACES_UVS, VERTS_COLOR)


if __name__ == '__main__':
    main()
