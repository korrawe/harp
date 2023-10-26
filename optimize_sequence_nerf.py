from json import dump
import os
from typing import Any
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
from pytorch3d.structures import Meshes,Pointclouds
from pytorch3d.io import save_obj
from pytorch3d.ops import SubdivideMeshes, taubin_smoothing
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

from model.vgg import Vgg16Features
from loss.arap import arap_loss
from loss.kps_loss import kps_loss
from loss.texture_reg import albedo_reg, normal_reg

from utils.data_util import load_multiple_sequences
from utils.eval_util import image_eval, load_gt_vert, align_w_scale
from utils.visualize import render_360, concat_image_in_dir, render_360_light, render_image, prepare_mesh, prepare_materials, render_image_with_RT,prepare_mesh_NeRF
from utils import file_utils, hand_model_utils, config_utils

import trimesh

# add DS_NeRF to path
import sys
sys.path.append('DS_NeRF')

from DS_NeRF.run_nerf_helpers import *
from DS_NeRF.run_nerf import *
from pytorch3d.loss import point_mesh_face_distance

import lpips


# from DS_NeRF.run_nerf import config_parser

import point_mesh_distance_modified
import igl
import wandb



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
        hand_verts, arm_joints, _ = hand_layer(betas=torch.zeros([1, 10]).to(device), global_orient=torch.zeros([1, 3]).to(device),
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

@torch.no_grad()
def params_cam2transform(params,configs,device='cuda'):
    """given params,return transform from world to camera"""
    # X_cam = X_world R + T
    cams=params['cam'] # might be list
    focal_length=configs['focal_length']
    img_size=configs['img_size']
    camera_t = torch.stack([cams[:, 1], cams[:, 2], 2 * focal_length/(img_size * cams[:, 0] +1e-9)], dim=1).to(device)
    R_batch = torch.Tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).repeat(camera_t.shape[0], 1, 1).to(device)
    # combine rotation and translation into transformation matrix (B,4,4) 
    transform = torch.eye(4).unsqueeze(0).repeat(camera_t.shape[0], 1, 1).to(device)
    transform[:, :3, :3] = R_batch
    transform[:, :3, 3] = camera_t
    params['w2c']=transform
    return params


# Ray helpers
def get_rays(H, W, focal, c2w=None,device='cuda'):
    """Return coor in world"""
    if c2w is None:
        c2w = torch.eye(4).to(device)
    else:
        device=c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W * .5) / focal, (j - H * .5) / focal, torch.ones_like(i)], -1).to(device) #NOTE:probably wrong
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d 

@torch.no_grad()
def geometry_guided_near_far_torch(orig, dir, vert, geo_threshold=0.01):
    """Compute near and far for each ray"""
    num_vert = vert.shape[0]
    num_rays = orig.shape[0]
    orig_ = torch.repeat_interleave(orig[:, None, :], num_vert, 1)
    dir_ = torch.repeat_interleave(dir[:, None, :], num_vert, 1)
    vert_ = torch.repeat_interleave(vert[None, ...], num_rays, 0)
    orig_v = vert_ - orig_
    z0 = torch.einsum('ij,ij->i', orig_v.reshape(-1, 3), dir_.reshape(-1, 3)).reshape(num_rays, num_vert)
    dz = torch.sqrt(geo_threshold**2 - (torch.norm(orig_v, dim=2)**2 - z0**2))
    near = z0 - dz
    near[near != near] = float('inf')
    near = near.min(dim=1)[0]
    far = z0 + dz
    far[far != far] = float('-inf')
    far = far.max(dim=1)[0]
    return near, far # shape (num_rays,)

@torch.no_grad()
def geometry_guided_near_far_torch_dummy(orig, dir, vert, geo_threshold=0.01):
    """Compute near and far for each ray"""
    num_vert = vert.shape[0]
    num_rays = orig.shape[0]
    # 计算orig和vert的距离
    dist=((orig[:1]-vert)**2).sum(dim=-1).sqrt() # shape (num_vert,)
    min_dist=dist.min() # shape (1,)
    max_dist=dist.max() # shape (1,)
    near=min_dist-geo_threshold
    far=max_dist+geo_threshold

    # expand
    near=near.expand(num_rays)
    far=far.expand(num_rays)


    return near, far # shape (num_rays,)


@torch.no_grad()
def debug_ray_point_and_mesh(params,hand_layer,configs,mesh_subdivider):
    """测试世界系内的mesh是否与cam的光线对齐"""
    use_verts_textures = False
    GLOBAL_POSE=False
    SHARED_TEXTURE = True
    device=params['w2c'].device
    with torch.no_grad():
    # NOTE: The reference mesh is from the first step, it should be the mesh with mean pose instead
        hand_joints, hand_verts, faces, textures,_ = prepare_mesh_NeRF(params, torch.tensor([20]), hand_layer, use_verts_textures, mesh_subdivider, 
        global_pose=GLOBAL_POSE, configs=configs, shared_texture=SHARED_TEXTURE, use_arm=configs['use_arm'], device=device)
        ref_meshes = Meshes(hand_verts, faces, textures)

    # get rays
    H, W = configs['img_size'], configs['img_size']
    focal = configs['focal_length']
    rays_o, rays_d = get_rays(H, W, focal, torch.linalg.inv(params['w2c'][20]))
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    # choose 10000 rays using randperm
    rand_idx = torch.randperm(rays_o.shape[0])[:10000]
    rays_o = rays_o[rand_idx]
    rays_d = rays_d[rand_idx]

    temp_near, temp_far = geometry_guided_near_far_torch(rays_o, rays_d, hand_verts.squeeze(),geo_threshold=0.1)
    hit_mask=temp_near < temp_far

    temp_near[hit_mask] = temp_near[hit_mask]
    temp_far[hit_mask] = temp_far[hit_mask]
    rays_o = rays_o[hit_mask]
    rays_d = rays_d[hit_mask]
    near, far = torch.zeros_like(rays_d[...,1]), temp_far * torch.ones_like(rays_d[...,1])
    # get points between near and far with 10 intervals
    t_vals = torch.linspace(0., 1., steps=10, device=device)
    # expand 
    z_vals = near[...,None] * (1. - t_vals[None,...]) + far[...,None] * (t_vals[None,...])
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    # get mesh points
    mesh_pts = ref_meshes.verts_padded()[0]

    # 将 pts 和 mesh_pts 转换为 [N, 3] 的形式
    pts_np = pts.reshape(-1, 3).cpu().numpy()
    mesh_pts_np = mesh_pts.cpu().numpy()

    # 聚合所有的点
    all_points = np.concatenate([pts_np, mesh_pts_np], axis=0)

    # 设定颜色：蓝色用于 pts，红色用于 mesh_pts
    pts_colors = np.tile(np.array([[0.0, 0.0, 1.0]]), (pts_np.shape[0], 1))
    mesh_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (mesh_pts_np.shape[0], 1))

    all_colors = np.concatenate([pts_colors, mesh_colors], axis=0)

    point_cloud = trimesh.points.PointCloud(vertices=all_points, colors=all_colors)
    point_cloud.export('colored_points.ply')


def warp_samples_to_canonical_diff(pts_np,verts_np,faces_np, T):
    # f_id, unsigned_squared_dist=point_mesh_distance_modified.point_mesh_face_distance(pcl, meshes)
    signed_dist, f_id, closest = igl.signed_distance(pts_np, verts_np, faces_np[:, :3])
    # differentiable barycentric interpolation

    closest_face_np = faces_np[:, :3][f_id]
    closest_face_device_T=torch.from_numpy(closest_face_np).to(T.device) # (N, 3)

    closest_tri = verts_np[closest_face_np]
    closest_tri=torch.from_numpy(closest_tri).float().to(T.device)
    closest = torch.from_numpy(closest).float().to(T.device)
    v0v1 = closest_tri[:, 1] - closest_tri[:, 0]
    v0v2 = closest_tri[:, 2] - closest_tri[:, 0]
    v1v2 = closest_tri[:, 2] - closest_tri[:, 1]
    v2v0 = closest_tri[:, 0] - closest_tri[:, 2]
    v1p = closest - closest_tri[:, 1]
    v2p = closest - closest_tri[:, 2]
    N = torch.cross(v0v1, v0v2)
    denom = torch.bmm(N.unsqueeze(dim=1), N.unsqueeze(dim=2)).squeeze()
    C1 = torch.cross(v1v2, v1p)
    u = torch.bmm(N.unsqueeze(dim=1), C1.unsqueeze(dim=2)).squeeze() / denom
    C2 = torch.cross(v2v0, v2p)
    v = torch.bmm(N.unsqueeze(dim=1), C2.unsqueeze(dim=2)).squeeze() / denom
    w = 1 - u - v
    barycentric = torch.stack([u, v, w], dim=1) # (N, 3)
    T_interp = (T[closest_face_device_T] * barycentric[..., None, None]).sum(axis=1)
    T_interp_inv = torch.inverse(T_interp)
    return T_interp_inv, torch.from_numpy(f_id), torch.from_numpy(signed_dist)

def pts_w2canonical(pts,verts,faces,pts_np,verts_np,faces_np,hand_dict,return_dict=False):
    """ Transform pts from world to canonical space
    pts:(batch_size, N, 3)
    hand_dict: return by prepare_mesh_NeRF, with 
    'T':[batch_size, V, 4, 4],'W': [batch_size, V, J], 'transl': [batch_size, 3],
    'verts_offset':[batch_size, V, 3], 'joints_wrist':[batch_size, 3]"""
    assert len(pts.shape)==3, "pts should be (batch_size, N, 3),got {}".format(pts.shape)

    # warp_samples_to_canonical_diff
    T_interp_inv, f_id, signed_dist=warp_samples_to_canonical_diff(pts_np,verts_np,faces_np, hand_dict['T'].squeeze(0))

    T_interp_inv=T_interp_inv.to(pts.device) # (N, 4, 4)
    f_id=f_id.to(pts.device) # (N,)
    signed_dist=signed_dist.to(pts.device) # (N,)

    # 1. minus transl, please refer to hand_models/smplx/smplx/body_models.py: SMPLXARM_NeRF
    pts=pts-hand_dict['transl'][:,None,:]

    # 2. plus joints, after this pts are relative to template hand in world space
    pts=pts+hand_dict['joints_wrist'][:,None,:]

    # 3. transform pts to canonical space using T_interp_inv using einsum
    pts_canonical=torch.einsum('nij,bnj->bni',T_interp_inv[:,:3,:3],pts) + T_interp_inv[:,:3,3]# (batch_size, N, 3)

    if not return_dict:
        return pts_canonical
    else:
        ret_dict={}
        ret_dict['signed_dist']=signed_dist
        return pts_canonical,ret_dict

class CoordinateTransformer:
    def __init__(self, reference_points, scale):
        self.offset = -reference_points.mean(0)
        cetered_points = reference_points + self.offset
        cetered_points=cetered_points[cetered_points[:,0]<0]
        self.scale_factor = scale / (cetered_points).abs().max()
    
    def __call__(self, points, clamp=True):
        # 应用坐标转换
        transformed_points = (points+ self.offset) * self.scale_factor 
        
        # 将结果限制在 [-1, 1] 范围内
        if clamp: transformed_points = torch.clamp(transformed_points, -1, 1)
        
        return transformed_points

    
def save_points(points1,points2,save_path):
    """save points1 and points2 to save_path"""
    # point1, point2最多取50000个点
    if points1.shape[0] > 50000:
        idx = np.random.choice(points1.shape[0], 50000, replace=False)
        points1 = points1[idx]
    if points2.shape[0] > 50000:
        idx = np.random.choice(points2.shape[0], 50000, replace=False)
        points2 = points2[idx]
    # 聚合所有的点
    all_points = np.concatenate([points1, points2], axis=0)

    # 设定颜色：蓝色用于 pts，红色用于 mesh_pts
    point1_colors = np.tile(np.array([[0.0, 0.0, 1.0]]), (points1.shape[0], 1))
    point2_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (points2.shape[0], 1))

    all_colors = np.concatenate([point1_colors, point2_colors], axis=0)

    point_cloud = trimesh.points.PointCloud(vertices=all_points, colors=all_colors)

    if not save_path.endswith('.ply'):
        save_path+='.ply'

    point_cloud.export(save_path)


def test_pts_w2canonical(verts,faces,verts_np,faces_np,hand_dict):
    pts=verts.clone().detach()
    pts=pts+torch.randn_like(pts)*0.005
    pts_np=pts.detach().cpu().squeeze(0).numpy()
    # warp_samples_to_canonical_diff
    T_interp_inv, f_id, signed_dist=warp_samples_to_canonical_diff(pts_np,verts_np,faces_np, hand_dict['T'].squeeze(0))

    T_interp_inv=T_interp_inv.to(pts.device) # (N, 4, 4)
    f_id=f_id.to(pts.device) # (N,)
    signed_dist=signed_dist.to(pts.device) # (N,)

    # 1. minus transl, please refer to hand_models/smplx/smplx/body_models.py: SMPLXARM_NeRF
    pts=pts-hand_dict['transl'][:,None,:]

    # 2. plus joints, after this pts are relative to template hand in world space
    pts=pts+hand_dict['joints_wrist'][:,None,:]

    # 3. transform pts to canonical space using T_interp_inv using einsum
    pts_canonical=torch.einsum('nij,bnj->bni',T_interp_inv[:,:3,:3],pts) + T_interp_inv[:,:3,3]# (batch_size, N, 3)

    # convert to numpy and save (assume batch_size=1)
    pts_canonical_np=pts_canonical.detach().squeeze(0).cpu().numpy()
    verts_canonical_np=hand_dict['v_shaped'].detach().squeeze(0).cpu().numpy()

    # 聚合所有的点
    all_points = np.concatenate([pts_canonical_np, verts_canonical_np], axis=0)

    # 设定颜色：蓝色用于 pts，红色用于 mesh_pts
    pts_colors = np.tile(np.array([[0.0, 0.0, 1.0]]), (pts_canonical_np.shape[0], 1))
    mesh_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (verts_canonical_np.shape[0], 1))

    all_colors = np.concatenate([pts_colors, mesh_colors], axis=0)

    point_cloud = trimesh.points.PointCloud(vertices=all_points, colors=all_colors)

    point_cloud.export('canonical_points.ply')

    print('finished test with canonical_points.ply')


def sample_patch(mask, patch_size):
    H, W = mask.shape
    patch_mask = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
    
    # 计算可以作为中心点的坐标的有效范围
    y, x = torch.where(mask)
    valid_idx = torch.where((y >= patch_size // 2) & (y <= H - (patch_size + 1) // 2) & 
                            (x >= patch_size // 2) & (x <= W - (patch_size + 1) // 2))[0]
    
    if valid_idx.numel() == 0:
        # 如果没有有效的中心点，所有不靠近边缘的点都可以作为中心点
        y, x= torch.where(torch.ones_like(mask))
        valid_idx = torch.where((y >= patch_size // 2) & (y <= H - (patch_size + 1) // 2) & 
                                (x >= patch_size // 2) & (x <= W - (patch_size + 1) // 2))[0]
    
    # 随机选择一个中心点
    idx = valid_idx[torch.randint(0, valid_idx.numel(), (1,)).item()]
    center_y, center_x = y[idx].item(), x[idx].item()
    
    # 标记patch_mask中对应的区域
    patch_mask[center_y - patch_size // 2:center_y + (patch_size + 1) // 2,
               center_x - patch_size // 2:center_x + (patch_size + 1) // 2] = True
    
    return patch_mask


def render_whole_image(args,params,H,W,configs,
                       hand_verts,faces,fid,
                       render_kwargs,hand_dict,
                       canonical_coor_normalizer,hand_joints,initial_mask,
                       device='cuda'):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2]) 
    rays_o, rays_d = get_rays(H, W, configs['focal_length'], torch.linalg.inv(params['w2c'][fid].squeeze()))
    rays_o_reshape = rays_o.reshape(-1, 3)
    rays_d_reshape = rays_d.reshape(-1, 3)
    hit_mask=initial_mask.reshape(-1)
    rays_o_reshape=rays_o_reshape[hit_mask]
    rays_d_reshape=rays_d_reshape[hit_mask]
    if False:
        near, far = geometry_guided_near_far_torch_dummy(rays_o_reshape, 
                                                        rays_d_reshape/rays_d_reshape.norm(dim=-1,keepdim=True), 
                                                        hand_joints.squeeze(),geo_threshold=0.01)
    else:
        near, far = geometry_guided_near_far_torch(rays_o_reshape, rays_d_reshape/rays_d_reshape.norm(dim=-1,keepdim=True), hand_verts.squeeze(),geo_threshold=0.005)
    # hit_mask[hit_mask][near>far]=False
    hit_mask[torch.where(hit_mask)[0][near>far]]=False
    hit_mask_near_far=near<far
    near = near[hit_mask_near_far]
    far = far[hit_mask_near_far]
    rays_o_reshape = rays_o_reshape[hit_mask_near_far]
    rays_d_reshape = rays_d_reshape[hit_mask_near_far]
    if near.shape[0] < 100:
        raise Exception('too few rays, please check the camera pose')
    # reshape hit_mask into (H, W)
    hit_mask = hit_mask.reshape(H, W) 

    # get points between near and far with args.N_samples intervals
    t_vals = torch.linspace(0., 1., steps=args.N_samples, device=device)
    # expand
    z_vals = near[..., None] * (1. - t_vals[None, ...]) + far[..., None] * (t_vals[None, ...]) # (N_rays, N_samples)

    if render_kwargs['perturb'] > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)
        z_vals = lower + (upper - lower) * t_rand


    pts = rays_o_reshape[..., None, :] + rays_d_reshape[..., None, :] * z_vals[..., :, None] # (N_rays, N_samples, 3)

    pts=pts.reshape(1,-1,3)
    pts_np=pts.reshape(-1,3).detach().cpu().numpy()
    verts_np=hand_verts.reshape(-1,3).detach().cpu().numpy()
    faces_np=faces.detach().cpu().numpy().squeeze(0)

    # test_pts_w2canonical(hand_verts,faces,verts_np,faces_np,hand_dict)
    pts=pts_w2canonical(pts,hand_verts,faces,pts_np,verts_np,faces_np,hand_dict)
    pts=canonical_coor_normalizer(pts)

    pts=pts.reshape(-1,z_vals.shape[-1],3)

    all_ret_0=render_rays_given_pts(pts, rays_o_reshape, rays_d_reshape, z_vals, **render_kwargs)

    rgb_map_0, disp_map_0, acc_map_0,weights_0 = all_ret_0['rgb_map'], all_ret_0['disp_map'], all_ret_0['acc_map'], all_ret_0['weights']
    alpha_0=all_ret_0['alpha'] if 'alpha' in all_ret_0 else None

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid, weights_0[..., 1:-1], args.N_importance, det=(render_kwargs['perturb'] == 0.))
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = rays_o_reshape[..., None, :] + rays_d_reshape[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

    pts=pts.reshape(1,-1,3)
    pts_np=pts.reshape(-1,3).detach().cpu().numpy()

    # test_pts_w2canonical(hand_verts,faces,verts_np,faces_np,hand_dict)
    pts=pts_w2canonical(pts,hand_verts,faces,pts_np,verts_np,faces_np,hand_dict)
    pts=canonical_coor_normalizer(pts)

    pts=pts.reshape(-1,z_vals.shape[-1],3)

    all_ret=render_rays_given_pts(pts, rays_o_reshape, rays_d_reshape, z_vals,fine_stage=True, **render_kwargs)

    rgb_map, disp_map, acc_map, weights = all_ret['rgb_map'], all_ret['disp_map'], all_ret['acc_map'], all_ret['weights']

    rgb_all_0=torch.zeros((H,W,3)).to(device)
    idx_valid=torch.where(hit_mask)
    rgb_all_0[idx_valid]=rgb_map_0.reshape(-1,3)
    rgb_all=torch.zeros((H,W,3)).to(device)
    rgb_all[idx_valid]=rgb_map.reshape(-1,3)
    print(f'hit_mask.sum():{hit_mask.sum()}')
    print(f'idx_valid.sum():{idx_valid[0].reshape(-1).shape[0]+idx_valid[1].reshape(-1).shape[0]}')
    print(f'rgb_all_0.max():{rgb_all_0.max()}')
    print(f'rgb_all.max():{rgb_all.max()}')
    if rgb_all_0.max()<0.3:
        raise Exception('rgb_all_0.max()<0.3, wrong for all images,fid:{},got render max:{}'.format(fid,rgb_all_0.max()))


    return rgb_all_0,rgb_all
    



def optimize_hand_sequence(configs, input_params, images_dataset, val_params, val_images_dataset,
        hand_layer, 
        VERTS_UVS=None, FACES_UVS=None, VERTS_COLOR=None, device='cuda'):

    # tf_writer = SummaryWriter(log_dir=configs["base_output_dir"])
    DEBUG=False
    WANDB=True
    PATCH=True
    PATCH_SIZE=64
    if DEBUG:WANDB=False
    TIME=False
    HARD_SURFACE_OFFSET=0.31326165795326233
    SMLP_REG_DUMMY=True
    if WANDB:
        wandb.init(project="hand_nerf", config=configs, dir=configs["base_output_dir"])
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
    # NOTE:New for hand_nerf: init transform from world to camera into params
    params=params_cam2transform(params,configs,device=device)                                                               
    
    #### End initialization ####

    batch_size = 1 # 19 # 10 # 30 # 2 # 16
    val_batch = 1
    
    images_dataloader = DataLoader(images_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_images_dataloader = DataLoader(val_images_dataset, batch_size=val_batch, shuffle=True, num_workers=20)  # Shuffle val

    ### Define loss and optimizer
    l1_loss = torch.nn.L1Loss()
    # # For vgg_loss
    # vgg = Vgg16Features(layers_weights = [1, 1/16, 1/8, 1/4, 1]).to(device)

    # Get optimizers
    opt_coarse, opt_app, sched_coarse = get_optimizers(params, configs)

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
            # NeRF
            'rgb_0':         {"weight": 1.0, "values": []}, # 1.0 ## 1.0
            'rgb':         {"weight": 1.0, "values": []}, # 1.0 ## 1.0
            'acc_0':         {"weight": 0.001, "values": []}, 
            'acc':         {"weight": 0.001, "values": []},
            'sparsity_reg':  {"weight": 0.01, "values": []},
            'sparsity_reg_0':  {"weight": 0.01, "values": []},
            'smpl_reg':      {"weight": 0.1, "values": []},
            'smpl_reg_0':      {"weight": 0.1, "values": []},
            'smpl_reg_dummy':      {"weight": 0.1, "values": []},
            'smpl_reg_dummy_0':      {"weight": 0.1, "values": []},
            'lpips':         {"weight": 0.1, "values": []},
            'lpips_0':         {"weight": 0.1, "values": []},

            # 'acc_0':         {"weight": 0., "values": []}, 
            # 'acc':         {"weight": 0., "values": []},
            # 'sparsity_reg':  {"weight": 0., "values": []},
            # 'sparsity_reg_0':  {"weight": 0., "values": []},
            # 'smpl_reg':      {"weight": 0., "values": []},
            # 'smpl_reg_0':      {"weight": 0., "values": []},
            # 'smpl_reg_dummy':      {"weight": 0., "values": []},
            # 'smpl_reg_dummy_0':      {"weight": 0., "values": []},
            # 'lpips':         {"weight": 0., "values": []},
            # 'lpips_0':         {"weight": 0., "values": []},
         }
    # torch.autograd.set_detect_anomaly(True)

    lpips_fn=lpips.LPIPS(net='alex').to(device)
    for param in lpips_fn.parameters():
        param.requires_grad = False

    args=configs['nerf_args']
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf_tcnn(
            args)   
    
    # set lr of optimizer to be 0.005
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.005


    nerf_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9, last_epoch=-1)

    # debug_ray_point_and_mesh(params,hand_layer,configs,mesh_subdivider)
    H=W=configs['img_size']

    
    # Get reference mesh. Use the mesh from the first frame.
    (fid, y_true, y_sil_true, _, _) = images_dataset[0]

    with torch.no_grad():
        # NOTE: The reference mesh is from the first step, it should be the mesh with mean pose instead
        hand_joints, hand_verts, faces, textures, hand_dict = prepare_mesh_NeRF(params, torch.tensor([fid]), hand_layer, use_verts_textures, mesh_subdivider, 
        global_pose=GLOBAL_POSE, configs=configs, shared_texture=SHARED_TEXTURE, use_arm=configs['use_arm'], device=device)
        ref_meshes = Meshes(hand_verts, faces, textures)
        canonical_coor_normalizer=CoordinateTransformer(hand_dict['v_shaped'].squeeze(),scale=0.8)


    ### Training Loop ###
    epoch_id = 0
    n_iter = 0
    global_step=start
    epoch_start=start%(len(images_dataloader))

    PATCH_ORIGINAL=PATCH
    N_rand_original=args.N_rand

    for epoch_id in tqdm(range(epoch_start, configs["total_epoch"]),desc='Epoch:'): # 311 # 1501):  # 201
        frame_count = 0

        # print("Epoch: %d" % epoch_id)
        epoch_loss = 0.0
        mini_batch_count = 0
        # set process bar indicating image_dataloader
        bar=tqdm(total=len(images_dataloader),desc='Image:')
        for (fid, y_true, y_sil_true, y_sil_true_ero, y_sil_true_dil) in images_dataloader:
            bar.update(1)
            if TIME: t0 = time.time()
            cur_batch_size = fid.shape[0]
            y_sil_true = y_sil_true.squeeze(-1).to(device)
            y_sil_true_ero = y_sil_true_ero.squeeze(-1).to(device)
            y_sil_true_dil = y_sil_true_dil.squeeze(-1).to(device)
            y_true = y_true.to(device)
            if TIME:
                t1 = time.time()
                print(f't0-t1--preparing data:{t1-t0}')
            # Get new shader with updated light position
            if configs["share_light_position"]:
                light_positions = params['light_positions'][0].repeat(cur_batch_size, 1)
            else:
                light_positions = params['light_positions'][fid]
            # phong_renderer, silhouette_renderer, normal_renderer = renderer_helper.get_renderers(image_size=img_size, 
            #     light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, device=device)
            if TIME:
                t2=time.time()
                print(f't1-t2--get renderer:{t2-t1}')
            # Meshes
            hand_joints, hand_verts, faces, textures, hand_dict = prepare_mesh_NeRF(params, fid, hand_layer, use_verts_textures, mesh_subdivider,
                global_pose=GLOBAL_POSE, configs=configs, shared_texture=SHARED_TEXTURE, device=device, use_arm=configs['use_arm'])
            meshes = Meshes(hand_verts, faces, textures)
            cam = params['cam'][fid]
            if TIME:
                t3=time.time()
                print(f't2-t3--prepare mesh:{t3-t2}')

            # Material properties
            # materials_properties = prepare_materials(params, fid.shape[0])

            if TIME:
                t4=time.time()
                print(f't3-t4--prepare materials:{t4-t3}')
            
            #NOTE:New for hand_nerf: get rays in world frame

            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                        -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1, 2]) 
            rays_o, rays_d = get_rays(H, W, configs['focal_length'], torch.linalg.inv(params['w2c'][fid]).squeeze())
            rays_o_reshape = rays_o.reshape(-1, 3)
            rays_d_reshape = rays_d.reshape(-1, 3)
            if PATCH_ORIGINAL:
                args.N_rand=N_rand_original
                PATCH=mini_batch_count%3==0
            else:
                PATCH=False

            if TIME:
                t5=time.time()
                print(f't4-t5--get rays:{t5-t4}')
            if True:
                if not PATCH:
                    # 当不使用PATCH时，随机sampled的点是扩大后的mask
                    # use img mask as start
                    hit_mask=(y_sil_true_dil>0.).reshape(-1)
                    hit_mask_idx=torch.where((y_sil_true_dil>0.).reshape(-1))[0]
                    # choose 2*args.N_rand rays using randperm
                    hit_mask_choose_idx=hit_mask_idx[torch.randperm(hit_mask.sum())[:int(1.5*args.N_rand)]] if not DEBUG else hit_mask_idx[torch.arange(hit_mask.sum())]
                    hit_mask_choose_mask=torch.zeros_like(hit_mask)
                    hit_mask_choose_mask[hit_mask_choose_idx]=1
                    hit_mask=hit_mask*hit_mask_choose_mask # shape (H*W)
                    rays_o_reshape=rays_o_reshape[hit_mask] # shape (int(1.5*args.N_rand),3)
                    rays_d_reshape=rays_d_reshape[hit_mask] # shape (int(1.5*args.N_rand),3)
                    near, far = geometry_guided_near_far_torch(rays_o_reshape, 
                                                               rays_d_reshape/rays_d_reshape.norm(dim=-1,keepdim=True), 
                                                               hand_verts.squeeze(),geo_threshold=0.005) 
                    hit_mask_near_far=(near < far) # shape (int(1.5*args.N_rand))
                else:
                    # 当使用PATCH时，随机sample的点是缩小的的mask
                    args.N_rand=PATCH_SIZE**2
                    near=torch.ones(1)
                    while len(near)<args.N_rand:
                        hit_mask=(y_sil_true_ero>0.).squeeze(0) # shape (H,W)
                        patch_mask=sample_patch(hit_mask,patch_size=PATCH_SIZE)
                        hit_mask=patch_mask.reshape(-1)
                        rays_o_reshape_=rays_o_reshape[hit_mask]
                        rays_d_reshape_=rays_d_reshape[hit_mask]
                        near, far = geometry_guided_near_far_torch(rays_o_reshape_, 
                                                                   rays_d_reshape_/rays_d_reshape_.norm(dim=-1,keepdim=True), 
                                                                   hand_verts.squeeze(),geo_threshold=0.005)
                        if len(near)<args.N_rand or (near<far).sum()<1:
                            continue
                        near[torch.where(near>far)[0]]=near[near<far].mean()-0.01
                        far[torch.where(near>far)[0]]=far[near<far].mean()+0.01
                        hit_mask_near_far=(near < far)
                    rays_o_reshape = rays_o_reshape_
                    rays_d_reshape = rays_d_reshape_
                if TIME:
                    t6=time.time()
                    print(f't5-t6--get near far:{t6-t5}')
                
            else:
                near, far = geometry_guided_near_far_torch_dummy(rays_o_reshape, rays_d_reshape, hand_joints.squeeze(),geo_threshold=0.008)
                hit_mask=(y_sil_true_ero>0.).reshape(-1)
                hit_mask_near_far=(near < far)
                if TIME:
                    t6=time.time()
                    print(f't5-t6--get near far:{t6-t5}')
            # near=torch.zeros_like(near)
            try:
                near = near[hit_mask_near_far]
                far = far[hit_mask_near_far]
                rays_o_reshape = rays_o_reshape[hit_mask_near_far]
                rays_d_reshape = rays_d_reshape[hit_mask_near_far]
            except:
                # report near,far,rays_o_reshape,rays_d_reshape's shape and hit_mask_near_far's shape and sum
                print('index error')
                print(f'hit_mask_near_far.shape:{hit_mask_near_far.shape}')
                print(f'hit_mask_near_far.sum():{hit_mask_near_far.sum()}')
                print(f'near.shape:{near.shape}')
                print(f'far.shape:{far.shape}')
                print(f'rays_o_reshape.shape:{rays_o_reshape.shape}')
                print(f'rays_d_reshape.shape:{rays_d_reshape.shape}')
                continue
            # reshape hit_mask into (H, W)
            hit_mask = hit_mask.reshape(H, W)

            # choose args.N_rand rays using randperm
            rand_idx = torch.randperm(rays_o_reshape.shape[0])[:args.N_rand] 
            if DEBUG or PATCH:
                rand_idx=torch.arange(rays_o_reshape.shape[0])
            rays_o_reshape = rays_o_reshape[rand_idx]
            rays_d_reshape = rays_d_reshape[rand_idx]
            near = near[rand_idx]
            far = far[rand_idx]
            # get points between near and far with args.N_samples intervals
            t_vals = torch.linspace(0., 1., steps=args.N_samples, device=device)
            # expand
            z_vals = near[..., None] * (1. - t_vals[None, ...]) + far[..., None] * (t_vals[None, ...]) # (N_rays, N_samples)

            if near.shape[0] < 100:
                raise Exception('too few rays, please check the camera pose')

            if args.perturb > 0.:
                # get intervals between samples
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(device)
                z_vals = lower + (upper - lower) * t_rand

            pts = rays_o_reshape[..., None, :] + rays_d_reshape[..., None, :] * z_vals[..., :, None] # (N_rays, N_samples, 3)

            pts=pts.reshape(1,-1,3)
            pts_np=pts.reshape(-1,3).detach().cpu().numpy()
            verts_np=hand_verts.reshape(-1,3).detach().cpu().numpy()
            faces_np=faces.detach().cpu().numpy().squeeze(0)

            if TIME:
                t7=time.time()
                print(f't6-t7--get pts:{t7-t6}')

            if DEBUG:
                # 检查世界系下的点和手部顶点是否对齐
                pts_save=pts.clone().detach().reshape(-1,3).cpu().numpy()
                pts_save=pts_save
                hand_verts_save=hand_verts.detach().reshape(-1,3).cpu().numpy()
                save_points(pts_save,hand_verts_save,'pts_world_vs_handverts_world.ply')

            # test_pts_w2canonical(hand_verts,faces,verts_np,faces_np,hand_dict)
            try:
                pts,ret_dict_warp=pts_w2canonical(pts,hand_verts,faces,pts_np,verts_np,faces_np,hand_dict,return_dict=True)
            except:
                print('pts_w2canonical error')
                continue
            signed_dist_0=ret_dict_warp['signed_dist']
            
            near_min=near.min().item()
            far_max=far.max().item()

            if DEBUG:
                # 检查规范化后的手部顶点和手部顶点是否对齐
                hand_verts_canonicalize=pts_w2canonical(hand_verts,hand_verts,faces,
                                                     hand_verts.reshape(-1,3).detach().cpu().numpy(),
                                                     verts_np,faces_np,hand_dict)
                hand_verts_canonicalize_save=hand_verts_canonicalize.detach().reshape(-1,3).cpu().numpy()
                hand_verts_shaped_save=hand_dict['v_shaped'].detach().reshape(-1,3).cpu().numpy()
                save_points(hand_verts_canonicalize_save,hand_verts_shaped_save,'hand_verts_canonicalize_vs_hand_verts_shaped.ply')

            if DEBUG:
                # 检查规范化后的点和规范化前的点是否对齐
                pts_save=pts.clone().detach().reshape(-1,3).cpu().numpy()
                pts_save=pts_save
                pts_save_before_canonical_normalize=pts_save.copy()
                hand_verts_save=hand_dict['v_shaped'].detach().reshape(-1,3).cpu().numpy()
                save_points(pts_save,hand_verts_save,'pts_after_canonical_vs_handverts_shaped.ply')
                
            pts=canonical_coor_normalizer(pts)

            if DEBUG:
                # 检查正则化后的点和规范化后的点是否对齐
                pts_save=pts.clone().detach().reshape(-1,3).cpu().numpy()
                pts_save=pts_save
                hand_verts_save=pts_save_before_canonical_normalize
                save_points(pts_save,hand_verts_save,'pts_normalized_vs_pts_after_canonical.ply')

            if DEBUG:
                # 检查正则化后的点和bbox是否对齐
                hand_verts_shaped_normalize=canonical_coor_normalizer(hand_dict['v_shaped'].detach().reshape(-1,3)).detach().reshape(-1,3).cpu().numpy()
                # bbox at eight points [-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]
                bbox=np.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]])
                save_points(hand_verts_shaped_normalize,bbox,'hand_verts_shaped_normalized_vs_bbox.ply')
            
            pts=pts.reshape(args.N_rand,-1,3)

            if TIME:
                t8=time.time()
                print(f't7-t8--warp pts:{t8-t7}')

            all_ret_0=render_rays_given_pts(pts, rays_o_reshape, rays_d_reshape, z_vals,retraw=True, **render_kwargs_train)

            if TIME:
                t9=time.time()
                print(f't8-t9--first render rays:{t9-t8}')
            rgb_map_0, disp_map_0, acc_map_0,weights_0 = all_ret_0['rgb_map'], all_ret_0['disp_map'], all_ret_0['acc_map'], all_ret_0['weights']
            alpha_0=all_ret_0['alpha'] if 'alpha' in all_ret_0 else None

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights_0[..., 1:-1], args.N_importance, det=(args.perturb == 0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o_reshape[..., None, :] + rays_d_reshape[..., None, :] * z_vals[..., :,
                                                                       None]  # [N_rays, N_samples + N_importance, 3]

            pts=pts.reshape(1,-1,3)
            pts_np=pts.reshape(-1,3).detach().cpu().numpy()
            if TIME:
                t10=time.time()
                print(f't9-t10--sample pdf:{t10-t9}')

            # test_pts_w2canonical(hand_verts,faces,verts_np,faces_np,hand_dict)
            pts,ret_dict_warp=pts_w2canonical(pts,hand_verts,faces,pts_np,verts_np,faces_np,hand_dict,return_dict=True)
            signed_dist=ret_dict_warp['signed_dist']
            pts=canonical_coor_normalizer(pts)

            pts=pts.reshape(args.N_rand,-1,3)
            if TIME:
                t11=time.time()
                print(f't10-t11--warp pts:{t11-t10}')

            all_ret=render_rays_given_pts(pts, rays_o_reshape, rays_d_reshape, z_vals,retraw=True,fine_stage=True, **render_kwargs_train)

            if TIME:
                t12=time.time()
                print(f't11-t12--second render rays:{t12-t11}')
            y_gt_choose=y_true.squeeze(0)[hit_mask][hit_mask_near_far][rand_idx]

            rgb_map, disp_map, acc_map, weights = all_ret['rgb_map'], all_ret['disp_map'], all_ret['acc_map'], all_ret['weights']

            y_sil_true_choose=y_sil_true.squeeze(0)[hit_mask][hit_mask_near_far][rand_idx]

    

            # # Shihouette
            # # Stop computing and updating silhouette when learning texture model
            # y_sil_pred = render_image(meshes, cam, cur_batch_size, silhouette_renderer, configs['img_size'], configs['focal_length'], silhouette=True)
            
            # # RGB UV
            # if configs["self_shadow"]:
            #     # Render with self-shadow
            #     light_R, light_T, cam_R, cam_T = renderer_helper.process_info_for_shadow(cam, light_positions, hand_verts.mean(1), 
            #                                         image_size=configs['img_size'], focal_length=configs['focal_length'])
            #     shadow_renderer = renderer_helper.get_shadow_renderers(image_size=img_size, 
            #         light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, 
            #         amb_ratio=nn.Sigmoid()(params['amb_ratio']), device=device)

            #     y_pred = render_image_with_RT(meshes, light_T, light_R, cam_T, cam_R,
            #                 cur_batch_size, shadow_renderer, configs['img_size'], configs['focal_length'], silhouette=False, 
            #                 materials_properties=materials_properties)
            # else:
            #     # Render without self-shadow
            #     y_pred = render_image(meshes, cam, cur_batch_size, phong_renderer, configs['img_size'], configs['focal_length'], 
            #                           silhouette=False, materials_properties=materials_properties)

            # if LOG_IMGAGE and epoch_id % 10 == 0 and mini_batch_count == 0:
            #     # Value range 0 (black) -> 1 (white)
            #     # Log silhouette
            #     show_img_pair(y_sil_pred.detach().cpu().numpy(), y_sil_true.detach().cpu().numpy(), save_img_dir=base_output_dir,
            #             step=epoch_id, silhouette=True, prefix="")
            #     # Log RGB
            #     show_img_pair(y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), save_img_dir=base_output_dir,
            #             step=epoch_id, silhouette=False, prefix="")
            #     # Loss visulization
            #     loss_image = torch.abs(y_true * y_sil_true_col.unsqueeze(-1) - y_pred * y_sil_true_col.unsqueeze(-1))
            #     show_img_pair(loss_image.detach().cpu().numpy(), y_true.detach().cpu().numpy(), save_img_dir=base_output_dir,
            #             step=epoch_id, silhouette=False, prefix="loss_")
            

            loss = {}

            # NOTE:New for hand_nerf: get rgb loss
            loss_rgb = ((rgb_map-y_gt_choose)**2).mean()
            loss_rgb_0 = ((rgb_map_0-y_gt_choose)**2).mean()
            loss['rgb_0'] = loss_rgb_0
            loss['rgb'] = loss_rgb

            if losses['acc']['weight'] > 0.:
                loss['acc']=((acc_map-y_sil_true_choose)**2).mean()
                loss['acc_0']=((acc_map_0-y_sil_true_choose)**2).mean()
            else:
                loss['acc']=torch.tensor(0.0, device=device)
                loss['acc_0']=torch.tensor(0.0, device=device)

            if losses['sparsity_reg']['weight'] > 0.:
                sparsity_reg = torch.mean(-torch.log(
                    torch.exp(-torch.abs(acc_map.clamp(0.,1.))) + torch.exp(-torch.abs(1-acc_map.clamp(0.,1.)))
                + 1e-5) + HARD_SURFACE_OFFSET)

                sparsity_reg=sparsity_reg+torch.mean(-torch.log(
                    torch.exp(-torch.abs(weights.clamp(0.,1.))) + torch.exp(-torch.abs(1-weights.clamp(0.,1.)))
                + 1e-5) + HARD_SURFACE_OFFSET)

                loss['sparsity_reg']=sparsity_reg

                sparsity_reg_0=torch.mean(-torch.log(
                    torch.exp(-torch.abs(acc_map_0.clamp(0.,1.))) + torch.exp(-torch.abs(1-acc_map_0.clamp(0.,1.)))
                + 1e-5) + HARD_SURFACE_OFFSET)

                sparsity_reg_0=sparsity_reg_0+torch.mean(-torch.log(
                    torch.exp(-torch.abs(weights_0.clamp(0.,1.))) + torch.exp(-torch.abs(1-weights_0.clamp(0.,1.)))
                + 1e-5) + HARD_SURFACE_OFFSET)

                loss['sparsity_reg_0']=sparsity_reg_0
            else:
                loss['sparsity_reg']=torch.tensor(0.0, device=device)
                loss['sparsity_reg_0']=torch.tensor(0.0, device=device)

            if losses['smpl_reg']['weight'] > 0.:

                inside_volume = signed_dist< 0
                raw_inside_volume = all_ret['raw'].reshape(-1, 4)[inside_volume]
                if inside_volume.sum() > 0:
                    smpl_reg =  F.mse_loss(
                        1 - torch.exp(-torch.relu(raw_inside_volume[:, 3])),
                        torch.ones_like(raw_inside_volume[:, 3])
                    ) 
                else:
                    smpl_reg = torch.tensor(0.0, device=device)
                loss['smpl_reg'] = smpl_reg

                inside_volume_0 = signed_dist_0< 0
                raw_inside_volume_0 = all_ret_0['raw'].reshape(-1, 4)[inside_volume_0]
                if inside_volume_0.sum() > 0:
                    smpl_reg_0 =  F.mse_loss(
                        1 - torch.exp(-torch.relu(raw_inside_volume_0[:, 3])),
                        torch.ones_like(raw_inside_volume_0[:, 3])
                    ) 
                else:
                    smpl_reg_0 = torch.tensor(0.0, device=device)
                loss['smpl_reg_0'] = smpl_reg_0
            else:
                loss['smpl_reg']=torch.tensor(0.0, device=device)
                loss['smpl_reg_0']=torch.tensor(0.0, device=device)

            if SMLP_REG_DUMMY and losses['smpl_reg_dummy']['weight'] > 0. and np.random.rand()<0.5:
                pts_dummy=torch.rand_like(pts)*2-1
                raw_smpl=render_kwargs_train['network_query_fn'](pts, None, render_kwargs_train['network_fn'])
                v_shaped_canonical_normalize=canonical_coor_normalizer(hand_dict['v_shaped'].detach().reshape(-1,3),clamp=False).detach().reshape(-1,3)
                mesh_v_shaped_canonical_normalize=Meshes(v_shaped_canonical_normalize.unsqueeze(0),params['mesh_faces'].unsqueeze(0))
                mesh_v_shaped_canonical_normalize=mesh_subdivider(mesh_v_shaped_canonical_normalize)
                # Meshes(v_shaped_canonical_normalize.unsqueeze(0),faces.unsqueeze(0))
                signed_dist_smpl_dummy,_,_=igl.signed_distance(pts_dummy.reshape(-1,3).detach().cpu().numpy(), 
                                                               mesh_v_shaped_canonical_normalize.verts_packed().squeeze(0).cpu().numpy(), 
                                                               mesh_v_shaped_canonical_normalize.faces_packed().squeeze(0).cpu().numpy())
                if DEBUG:
                    mask=torch.from_numpy(signed_dist_smpl_dummy).to(device)
                    mask=(mask<0.001)*(mask>-0.001)
                    pts_dummy_save=(pts_dummy.clone().detach().reshape(-1,3)[mask]).cpu().numpy()
                    hand_verts_save=mesh_v_shaped_canonical_normalize.verts_packed().squeeze(0).cpu().numpy()
                    save_points(pts_dummy_save,hand_verts_save,'pts_dummy_vs_hand_verts.ply')


                inside_volume_smpl_dummy = signed_dist_smpl_dummy< 0
                raw_inside_volume_smpl_dummy = raw_smpl.reshape(-1, 4)[inside_volume_smpl_dummy]
                raw_outside_volume_smpl_dummy = raw_smpl.reshape(-1, 4)[~inside_volume_smpl_dummy]

                if DEBUG:
                    mask=inside_volume_smpl_dummy
                    pts_dummy_save=(pts_dummy.clone().detach().reshape(-1,3)[mask]).cpu().numpy()
                    hand_verts_save=mesh_v_shaped_canonical_normalize.verts_packed().squeeze(0).cpu().numpy()
                    save_points(pts_dummy_save,hand_verts_save,'pts_inside_vs_hand_verts.ply')


                if inside_volume_smpl_dummy.sum() > 0:
                    smpl_reg_dummy =  F.mse_loss(
                        1 - torch.exp(-torch.relu(raw_inside_volume_smpl_dummy[:, 3])),
                        torch.ones_like(raw_inside_volume_smpl_dummy[:, 3])
                    )
                else:
                    smpl_reg_dummy = torch.tensor(0.0, device=device)
                if ~inside_volume_smpl_dummy.sum() > 0:
                    smpl_reg_dummy = smpl_reg_dummy + F.mse_loss(
                        torch.exp(-torch.relu(raw_outside_volume_smpl_dummy[:, 3])),
                        torch.ones_like(raw_outside_volume_smpl_dummy[:, 3])
                    )
                loss['smpl_reg_dummy'] = smpl_reg_dummy

                raw_smpl_0=render_kwargs_train['network_query_fn'](pts, None, render_kwargs_train['network_fine'])
                if inside_volume_smpl_dummy.sum() > 0:
                    smpl_reg_dummy_0 =  F.mse_loss(
                        1 - torch.exp(-torch.relu(raw_inside_volume_smpl_dummy[:, 3])),
                        torch.ones_like(raw_inside_volume_smpl_dummy[:, 3])
                    )
                else:
                    smpl_reg_dummy_0 = torch.tensor(0.0, device=device)
                if ~inside_volume_smpl_dummy.sum() > 0:
                    smpl_reg_dummy_0 = smpl_reg_dummy_0 + F.mse_loss(
                        torch.exp(-torch.relu(raw_outside_volume_smpl_dummy[:, 3])),
                        torch.ones_like(raw_outside_volume_smpl_dummy[:, 3])
                    )
                loss['smpl_reg_dummy_0'] = smpl_reg_dummy_0
            else:
                loss['smpl_reg_dummy']=torch.tensor(0.0, device=device)
                loss['smpl_reg_dummy_0']=torch.tensor(0.0, device=device)
            
            if losses['lpips']['weight'] > 0. and PATCH:
                # reshape rgb_map and y_true to (1,PATCH_SIZE,PATCH_SIZE,3)
                rgb_map_lpips=rgb_map.reshape(1,PATCH_SIZE,PATCH_SIZE,3)
                y_gt_choose_lpips=y_gt_choose.reshape(1,PATCH_SIZE,PATCH_SIZE,3)
                rgb_map_lpips=rgb_map_lpips.permute(0,3,1,2)
                y_gt_choose_lpips=y_gt_choose_lpips.permute(0,3,1,2)
                loss['lpips']=lpips_fn(rgb_map_lpips,y_gt_choose_lpips,normalize=True).mean()

                rgb_map_0_lpips=rgb_map_0.reshape(1,PATCH_SIZE,PATCH_SIZE,3).permute(0,3,1,2)
                loss['lpips_0']=lpips_fn(rgb_map_0_lpips,y_gt_choose_lpips,normalize=True).mean()
            else:
                loss['lpips']=torch.tensor(0.0, device=device)
                loss['lpips_0']=torch.tensor(0.0, device=device)



            

            """
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
            """
            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
                # tf_writer.add_scalar(k, l * losses[k]["weight"], n_iter)
                if WANDB:
                    wandb.log({k: l * losses[k]["weight"]})
                    if (mini_batch_count %100<4) and PATCH:
                        # save rgb_map,rgb_map_0 and y_gt_choose
                        rgb_map_save=rgb_map.reshape(PATCH_SIZE,PATCH_SIZE,3).detach().cpu().numpy()
                        rgb_map_0_save=rgb_map_0.reshape(PATCH_SIZE,PATCH_SIZE,3).detach().cpu().numpy()
                        y_gt_choose_save=y_gt_choose.reshape(PATCH_SIZE,PATCH_SIZE,3).detach().cpu().numpy()
                        wandb.log({"rgb_patch": wandb.Image(rgb_map_save, caption="rgb_patch")})
                        wandb.log({"rgb_patch_0": wandb.Image(rgb_map_0_save, caption="rgb_patch_0")})
                        wandb.log({"y_gt_choose_patch": wandb.Image(y_gt_choose_save, caption="y_gt_choose_patch")})
            
                # print("%s: %.6f" % (k, l * losses[k]["weight"]))
            
            epoch_loss += sum_loss.detach().cpu()
            # tf_writer.add_scalar('total_loss', sum_loss, n_iter)
            if WANDB:wandb.log({"total_loss": sum_loss})
            # report total_loss in bar
            bar.set_description(("total_loss = %.6f" % sum_loss)+f' global_step:{global_step}' )
            if TIME:
                t13=time.time()
                print(f't12-t13--compute loss:{t13-t12}')

            opt_coarse.zero_grad()
            opt_app.zero_grad()
            optimizer.zero_grad()
            if TIME:
                t14=time.time()
                print(f't13-t14--zero grad:{t14-t13}')
            sum_loss.backward()
            if TIME:
                t15=time.time()
                print(f't14-t15--backward:{t15-t14}')
            optimizer.step()
            nerf_scheduler.step()
            if TIME:
                t16=time.time()
                print(f't15-t16--step:{t16-t15}')

            # if COARSE_OPT:
            #     opt_coarse.step()
            # if APP_OPT:
            #     opt_app.step()
            
            frame_count += batch_size
            n_iter += 1
            # Delete the variables to free up memory
            # del y_pred
            del loss
            del rgb_map, disp_map, acc_map, weights, rgb_map_0, disp_map_0, acc_map_0, weights_0
            torch.cuda.empty_cache()

            if LOG_IMGAGE and epoch_id % 1 == 0 and mini_batch_count == 0:
            # if True:
                with torch.no_grad():
                    rgb_all_0,rgb_all=render_whole_image(args,params,H,W,
                                                         configs,hand_verts,faces,fid,
                                                         render_kwargs_test,hand_dict,
                                                         canonical_coor_normalizer,hand_joints,(y_sil_true_dil>0.).squeeze(0),device) # (H, W, 3) on cuda, range 0-1
                    # save to wandb, with gt
                    if WANDB:
                        wandb.log({"rgb_all_0": wandb.Image(rgb_all_0.detach().cpu().numpy(), caption="rgb_all_0")})
                        wandb.log({"rgb_all": wandb.Image(rgb_all.detach().cpu().numpy(), caption="rgb_all")})
                        # y_true
                        wandb.log({"y_true": wandb.Image(y_true.detach().cpu().numpy(), caption="y_true")})

                    # compute psnr,lpips
                    y_true_masked=y_true.squeeze(0)*y_sil_true.squeeze(0).unsqueeze(-1) # (H, W, 3) 
                    psnr_0=(10*torch.log10(1./((rgb_all_0-y_true_masked)**2).mean())).item()
                    psnr=(10*torch.log10(1./((rgb_all-y_true_masked)**2).mean())).item()
                    lpips_0=lpips_fn(rgb_all_0.unsqueeze(0).permute(0,3,1,2),y_true_masked.unsqueeze(0).permute(0,3,1,2),normalize=True).mean().item()
                    lpips_=lpips_fn(rgb_all.unsqueeze(0).permute(0,3,1,2),y_true_masked.unsqueeze(0).permute(0,3,1,2),normalize=True).mean().item()
                    # save to wandb
                    if WANDB:
                        wandb.log({"psnr_0_train": psnr_0})
                        wandb.log({"psnr_train": psnr})
                        wandb.log({"lpips_0_train": lpips_0})
                        wandb.log({"lpips_train": lpips_})


            global_step += 1
            if global_step % args.i_weights == 0:
                path = os.path.join(args.basedir, args.expname, '{:06d}.tar'.format(global_step))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train[
                        'network_fn'] is not None else None,
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train[
                        'network_fine'] is not None else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            
            mini_batch_count += 1
            
        print(" Epoch loss = %.6f" % (epoch_loss / mini_batch_count))
        # tf_writer.add_scalar('total_loss_epoch', (epoch_loss / mini_batch_count), epoch_id)
        wandb.log({"total_loss_epoch": (epoch_loss / mini_batch_count)})

        if epoch_id % 25==0 and epoch_id>0:
            # traverse val dataloader
            with torch.no_grad():
                average_psnr_0=0
                average_psnr=0
                average_lpips_0=0
                average_lpips=0
                idx_val=0
                for (fid, y_true, y_sil_true, 
                     y_sil_true_ero, y_sil_true_dil) in (tqdm(val_images_dataloader,desc='Running validation')):
                    y_sil_true = y_sil_true.squeeze(-1).squeeze(0).to(device)
                    y_true=y_true.squeeze(0).to(device)
                    y_sil_true_ero=y_sil_true_ero.squeeze(-1).squeeze(0).to(device)
                    y_sil_true_dil=y_sil_true_dil.squeeze(-1).squeeze(0).to(device)
                    rgb_all_0,rgb_all=render_whole_image(args,params,H,W,
                                                         configs,hand_verts,faces,fid,
                                                         render_kwargs_test,hand_dict,
                                                         canonical_coor_normalizer,hand_joints,(y_sil_true_dil>0.).squeeze(0),device)
                    # save to wandb, with gt
                    if WANDB and idx_val%20==0:
                        wandb.log({"rgb_all_0_val": wandb.Image(rgb_all_0.detach().cpu().numpy(), caption="rgb_all_0_val")})
                        wandb.log({"rgb_all_val": wandb.Image(rgb_all.detach().cpu().numpy(), caption="rgb_all_val")})
                        # y_true
                        wandb.log({"y_true_val": wandb.Image(y_true.detach().cpu().numpy(), caption="y_true_val")})
                    
                    # compute psnr,lpips
                    y_true_masked=y_true.squeeze(0)*y_sil_true.squeeze(0).unsqueeze(-1) # (H, W, 3)
                    psnr_0=(10*torch.log10(1./((rgb_all_0-y_true_masked)**2).mean())).item()
                    psnr=(10*torch.log10(1./((rgb_all-y_true_masked)**2).mean())).item()
                    lpips_0=lpips_fn(rgb_all_0.unsqueeze(0).permute(0,3,1,2),y_true_masked.unsqueeze(0).permute(0,3,1,2),normalize=True).mean().item()
                    lpips_=lpips_fn(rgb_all.unsqueeze(0).permute(0,3,1,2),y_true_masked.unsqueeze(0).permute(0,3,1,2),normalize=True).mean().item()
                    
                    # add to average
                    average_psnr_0+=psnr_0
                    average_psnr+=psnr
                    average_lpips_0+=lpips_0
                    average_lpips+=lpips_
                    idx_val+=1

                average_psnr_0=average_psnr_0/len(val_images_dataloader)
                average_psnr=average_psnr/len(val_images_dataloader)
                average_lpips_0=average_lpips_0/len(val_images_dataloader)
                average_lpips=average_lpips/len(val_images_dataloader)
                # save to wandb
                if WANDB:
                    wandb.log({"psnr_0_val": average_psnr_0})
                    wandb.log({"psnr_val": average_psnr})
                    wandb.log({"lpips_0_val": average_lpips_0})
                    wandb.log({"lpips_val": average_lpips})

        # if epoch_id % 20 == 0:
        #     visualize_val(val_images_dataloader, epoch_id, device, params, val_params, configs, hand_layer,
        #                   mesh_subdivider, opt_app, use_verts_textures, GLOBAL_POSE, SHARED_TEXTURE)

        # if epoch_id % 200 == 0 and epoch_id > 0:
        #     file_utils.save_result(params, base_output_dir, test=configs["known_appearance"])
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
            hand_joints, hand_verts, faces, textures = prepare_mesh_NeRF(params, fid, hand_layer, use_verts_textures, mesh_subdivider,
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


def main():
    # Get config
    config_dict = config_utils.get_config()
    config_dict["device"] = 'cuda' 
    hand_layer, VERTS_UVS, FACES_UVS, VERTS_COLOR = hand_model_utils.load_hand_model_NeRF(config_dict)
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

    # get nerf args
    parser = config_parser()
    args = parser.parse_args()
    args.basedir='./'
    args.expname=config_dict['base_output_dir']
    config_dict['nerf_args'] = args
    

    optimize_hand_sequence(config_dict, mano_params, images_dataset, val_mano_params, val_images_dataset,
        hand_layer, VERTS_UVS, FACES_UVS, VERTS_COLOR)


if __name__ == '__main__':
    main()
