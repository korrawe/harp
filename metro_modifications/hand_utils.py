import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import torch
import trimesh
import pickle
import imageio
import glob
import cv2
from PIL import Image


def optimize_for_mano_param(pred_vertices, mano_layer, idx=0, vert_rgb=None):
    # mano_params = {}
    # batch_size = 1  # 512  # 32
    batch_size = pred_vertices.shape[0]
    # w_pose = 1.0
    # w_shape = 1.0
    ncomps = 45
    epoch_coarse = 500
    epoch_fine = 700  # 1000

    SAVE_MESH_FOR_DEBUG = False  # True
    out_params = {
        # 'gen_joints': [],
        'joints': [],
        'verts': [],
        'rot': [],
        'pose': [],
        'shape': [],
        'trans': [],
        # 'camera': []
    }

    data_size = pred_vertices.shape[0]
    target_vertices = pred_vertices.detach() * 1000.0
    target_vertices_mean = target_vertices.mean(1) # .unsqueeze(1)
    # target_vertices = target_vertices - target_vertices_mean

    faces = mano_layer.th_faces.numpy()
    if SAVE_MESH_FOR_DEBUG:
        out_dir = "../debug/"
        mesh = trimesh.Trimesh(target_vertices.cpu().numpy()[0], faces, vertex_colors=vert_rgb)
        mesh.export(os.path.join(out_dir, "%03d_" % idx + "metro_vert_hand.ply"))
    device = "cuda"
    mano_layer = mano_layer.to(device)

    # Loop until the fitting error is less than 10.0 to avoid local minima
    # loop a few times, otherwise use the last optimization results
    for try_num in range(4):
    # while(True):
        # Model para initialization:
        shape = torch.zeros(batch_size, 10).to(device)
        shape.requires_grad_()
        rot = torch.zeros(batch_size, 3).to(device)  # torch.rand(batch_size, 3).to(device)
        rot.requires_grad_()
        pose = torch.zeros(batch_size, ncomps).to(device)  # torch.randn(batch_size, ncomps).to(device)
        pose.requires_grad_()
        trans = torch.zeros(batch_size, 3).to(device) + (target_vertices_mean / 1000.0) #  torch.rand(batch_size, 3).to(device)
        trans.requires_grad_()

        start_vertices, start_joints = mano_layer(torch.cat((rot, pose), 1), shape, trans)

        if SAVE_MESH_FOR_DEBUG:
            mesh = trimesh.Trimesh(start_vertices.detach().cpu().numpy()[0], faces)
            mesh.export(os.path.join(out_dir, "%03d_" % idx + "metro_vert_hand_start.ply"))

        criteria_loss = torch.nn.MSELoss().to(device)
        
        # Optimize for global orientation
        optimizer = torch.optim.Adam([rot, trans], lr=1e-1)  # lr=1e-2
        for i in range(0, epoch_coarse):
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose), 1), shape, trans)
            # mesh = trimesh.Trimesh(hand_verts.detach().cpu().numpy()[0], faces)
            # mesh.export(os.path.join(out_dir, "metro_vert_hand_coarse.ply"))
            # hand_verts_mean = hand_verts.mean(1).unsqueeze(1)
            # hand_verts - hand_verts_mean
            loss = criteria_loss(hand_verts, target_vertices)
            # print('Coarse alignment: %6f' % (loss.data))
            # print("trans", trans)
            # print("rot", rot)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('After coarse alignment: %6f' % (loss.data))

        if SAVE_MESH_FOR_DEBUG:
            mesh = trimesh.Trimesh(hand_verts.detach().cpu().numpy()[0], faces)
            mesh.export(os.path.join(out_dir, "%03d_" % idx + "metro_vert_hand_coarse.ply"))

        # Local optimization 
        optimizer = torch.optim.Adam([rot, pose, shape, trans], lr=1e-2)
        for i in range(0, epoch_fine):
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
            # hand_verts_mean = hand_verts.mean(1).unsqueeze(1)
            # hand_verts - hand_verts_mean
            loss = criteria_loss(hand_verts, target_vertices)  # + w_shape*(shape*shape).mean() + w_pose*(pose*pose).mean()
            # print('Fine alignment: %6f' % (loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('After fine alignment: %6f'%(loss.data))
        if loss.data > 10.0:
            print("<<<<<<<<<<<<<<<<<<< ERROR TOO HIGH >>>>>>>>>>>>>>>>>>>>>>>")
            if try_num == 3:
                print(("<<<<<<<<<<<<<<<<<<< ERROR Does not decrease, use the last one >>>>>>>>>>>>>>>>>>>>>>>"))

        if loss.data <= 10.0:
            break

    hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)

    # hand_verts = hand_verts - hand_verts.mean(1).unsqueeze(1) + target_vertices_mean
    # hand_joints = hand_joints - hand_joints.mean(1).unsqueeze(1) + target_vertices_mean

    if SAVE_MESH_FOR_DEBUG:
        mesh = trimesh.Trimesh(hand_verts.detach().cpu().numpy()[0], faces)
        mesh.export(os.path.join(out_dir, "%03d_" % idx + "metro_vert_hand_fine.obj"))

    out_params['joints'] = hand_joints.detach().cpu().numpy()
    out_params['verts'] = hand_verts.detach().cpu().numpy()
    out_params['rot'] = rot.detach().cpu().numpy()
    out_params['pose'] = pose.detach().cpu().numpy()
    out_params['shape'] = shape.detach().cpu().numpy()
    out_params['trans'] = trans.detach().cpu().numpy()
    return out_params


def optimize_for_mano_arm_param(pred_vertices, mano_arm_layer, idx=0, vert_rgb=None):
    batch_size = pred_vertices.shape[0]
    ncomps = 45
    epoch_coarse = 500
    epoch_fine = 700  # 1000

    SAVE_MESH_FOR_DEBUG = False  # True
    out_params = {
        # 'gen_joints': [],
        'joints': [],
        'verts': [],
        'rot': [],
        'pose': [],
        'shape': [],
        'trans': [],
        # 'camera': []
    }
    data_size = pred_vertices.shape[0]
    target_vertices = pred_vertices.detach() * 1000.0
    target_vertices_mean = target_vertices.mean(1) # .unsqueeze(1)
    # target_vertices = target_vertices - target_vertices_mean

    faces = mano_arm_layer.faces
    if SAVE_MESH_FOR_DEBUG:
        out_dir = "../debug/"
        mesh = trimesh.Trimesh(target_vertices.cpu().numpy()[0], mano_arm_layer.mano_faces, vertex_colors=vert_rgb)
        mesh.export(os.path.join(out_dir,"metro_vert_hand.ply"))

    device = "cuda"

    # Loop until the fitting error is less than 10.0 to avoid local minima
    # loop three-four times, otherwise use the thrid optimization results
    for try_num in range(4):
        # Model para initialization:
        shape = torch.zeros(batch_size, 10).to(device)
        shape.requires_grad_()
        rot = torch.zeros(batch_size, 3).to(device)  # torch.rand(batch_size, 3).to(device)
        rot.requires_grad_()
        pose = torch.zeros(batch_size, ncomps).to(device)  # torch.randn(batch_size, ncomps).to(device)
        pose.requires_grad_()
        trans = torch.zeros(batch_size, 3).to(device) # + (target_vertices_mean) #  torch.rand(batch_size, 3).to(device)
        trans.requires_grad_()

        start_vertices, start_joints = mano_arm_layer(betas=shape, global_orient=rot, transl=trans, right_hand_pose=pose, return_type='mano_w_arm')

        if SAVE_MESH_FOR_DEBUG:
            mesh = trimesh.Trimesh(start_vertices.detach().cpu().numpy()[0], mano_arm_layer.faces)
            mesh.export(os.path.join(out_dir, "start_arm.ply"))

        criteria_loss = torch.nn.MSELoss().to(device)

        # Optimize for global orientation
        optimizer = torch.optim.Adam([rot, trans], lr=1e-1)  # lr=1e-2
        for i in range(0, epoch_coarse):
            hand_verts, hand_joints = mano_arm_layer(betas=shape, global_orient=rot, transl=trans, right_hand_pose=pose, return_type='mano')
            
            loss = criteria_loss(hand_verts, target_vertices)
            # print('Coarse alignment: %6f' % (loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if SAVE_MESH_FOR_DEBUG:
            arm_vertices, arm_joints = mano_arm_layer(betas=shape, global_orient=rot, transl=trans, right_hand_pose=pose, return_type='mano_w_arm')
            mesh = trimesh.Trimesh(arm_vertices.detach().cpu().numpy()[0], faces)
            mesh.export(os.path.join(out_dir, "arm_coarse.ply"))
        print('After coarse alignment: %6f' % (loss.data))
        
        # Local optimization 
        optimizer = torch.optim.Adam([rot, pose, shape, trans], lr=1e-2)
        for i in range(0, epoch_fine):
            hand_verts, hand_joints = mano_arm_layer(betas=shape, global_orient=rot, transl=trans, right_hand_pose=pose, return_type='mano')

            # hand_verts_mean = hand_verts.mean(1).unsqueeze(1)
            # hand_verts - hand_verts_mean
            loss = criteria_loss(hand_verts, target_vertices)  # + w_shape*(shape*shape).mean() + w_pose*(pose*pose).mean()
            # print('Fine alignment: %6f' % (loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('After fine alignment: %6f'%(loss.data))
        if loss.data > 10.0:
            print("<<<<<<<<<<<<<<<<<<< ERROR TOO HIGH >>>>>>>>>>>>>>>>>>>>>>>")
            if try_num == 3:
                print(("<<<<<<<<<<<<<<<<<<< ERROR Does not decrease, use the last one >>>>>>>>>>>>>>>>>>>>>>>"))

        if loss.data  <= 10.0:
            break

    hand_verts, hand_joints = mano_arm_layer(betas=shape, global_orient=rot, transl=trans, right_hand_pose=pose, return_type='mano')

    # hand_verts = hand_verts - hand_verts.mean(1).unsqueeze(1) + target_vertices_mean
    # hand_joints = hand_joints - hand_joints.mean(1).unsqueeze(1) + target_vertices_mean

    if SAVE_MESH_FOR_DEBUG:
        arm_vertices, arm_joints = mano_arm_layer(betas=shape, global_orient=rot, transl=trans, right_hand_pose=pose, return_type='mano_w_arm')
        mesh = trimesh.Trimesh(arm_vertices.detach().cpu().numpy()[0], faces)
        mesh.export(os.path.join(out_dir, "arm_fine.obj"))

    out_params['joints'] = hand_joints.detach().cpu().numpy()
    out_params['verts'] = hand_verts.detach().cpu().numpy()
    out_params['rot'] = rot.detach().cpu().numpy()
    out_params['pose'] = pose.detach().cpu().numpy()
    out_params['shape'] = shape.detach().cpu().numpy()
    out_params['trans'] = trans.detach().cpu().numpy()

    return out_params



def optimize_for_nimble_param(target_vertices, nimble_layer, mano_j_regressor_np, mano_faces=None, idx=0, vert_rgb=None):

    batch_size = target_vertices.shape[0]

    ncomps = 30
    epoch_coarse = 200 # 500
    epoch_fine = 400

    SAVE_MESH_FOR_DEBUG = False # True
    shape_param = torch.zeros(1, 20)

    target_vertices = target_vertices.detach() * 1000.0
    target_vertices_mean = target_vertices.mean(1) # .unsqueeze(1)
    # target_vertices = target_vertices - target_vertices_mean

    # faces = mano_layer.th_faces.numpy()
    faces = nimble_layer.skin_f.cpu().numpy()
    if SAVE_MESH_FOR_DEBUG:
        out_dir = "../debug/nimble_fit/"
        mesh = trimesh.Trimesh(target_vertices.cpu().numpy()[0], mano_faces, vertex_colors=vert_rgb)
        mesh.export(os.path.join(out_dir, "%04d_" % idx + "target.ply"))

    device = "cuda"
    target_vertices = target_vertices.to(device)
    # mano_layer = mano_layer.to(device)

    # Loop until the fitting error is less than 10.0 to avoid local minima
    # loop three-four times, otherwise use the thrid optimization results
    for try_num in range(1):
        # Initialization params
        tex_param = torch.zeros(1, 10).to(device)
        cur_shape = torch.zeros(batch_size, 20).to(device)
        cur_rot = torch.zeros(batch_size, 3).to(device)
        cur_pose = torch.zeros(batch_size, ncomps).to(device)
        cur_trans = torch.zeros(batch_size, 3).to(device) # + (target_vertices_mean / 1000.0).to(device)
        cur_scale = torch.ones(1).to(device)
        
        cur_pose.requires_grad_()
        cur_rot.requires_grad_()
        cur_shape.requires_grad_()
        cur_trans.requires_grad_()
        # cur_scale.requires_grad_()

        skin_v, muscle_v, bone_v, bone_joints, tex_img = nimble_layer.forward(cur_pose, cur_shape, tex_param, cur_rot, cur_trans,
            global_scale=cur_scale, handle_collision=False, no_tex=True)
        mano_v = nimble_layer.nimble_to_mano(skin_v, is_surface=True)

        if SAVE_MESH_FOR_DEBUG:
            mesh = trimesh.Trimesh(mano_v.detach().cpu().numpy()[0], mano_faces)
            mesh.export(os.path.join(out_dir, "%04d_" % idx + "start.ply"))

        criteria_loss = torch.nn.MSELoss().to(device)
        # Optimize for global orientation
        optimizer = torch.optim.Adam([cur_rot, cur_trans], lr=1e-1)  # lr=1e-2 # 
        for i in range(0, epoch_coarse):
            # hand_verts, hand_joints = mano_layer(torch.cat((rot, pose), 1), shape, trans)
            skin_v, muscle_v, bone_v, bone_joints, tex_img = nimble_layer.forward(cur_pose, cur_shape, tex_param, cur_rot, cur_trans,
                global_scale=cur_scale, handle_collision=False, no_tex=True)
            mano_v = nimble_layer.nimble_to_mano(skin_v, is_surface=True)
            loss = criteria_loss(mano_v, target_vertices)
            # print('Coarse alignment: %6f' % (loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('After coarse alignment: %6f' % (loss.data))
        if SAVE_MESH_FOR_DEBUG:
            mesh = trimesh.Trimesh(mano_v.detach().cpu().numpy()[0], mano_faces)
            mesh.export(os.path.join(out_dir, "%04d_" % idx + "vert_hand_coarse.ply"))

        # Local optimization 
        optimizer = torch.optim.Adam([cur_rot, cur_pose, cur_shape, cur_trans], lr=1e-2)
        for i in range(0, epoch_fine):
            skin_v, muscle_v, bone_v, bone_joints, tex_img = nimble_layer.forward(cur_pose, cur_shape, tex_param, cur_rot, cur_trans,
                global_scale=cur_scale, handle_collision=False, no_tex=True)
            mano_v = nimble_layer.nimble_to_mano(skin_v, is_surface=True)
            loss = criteria_loss(mano_v, target_vertices)  # + w_shape*(shape*shape).mean() + w_pose*(pose*pose).mean()
            # print('Fine alignment: %6f' % (loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('After fine alignment: %6f'%(loss.data))
        if loss.data > 10.0:
            print("<<<<<<<<<<<<<<<<<<< ERROR TOO HIGH >>>>>>>>>>>>>>>>>>>>>>>")
            if try_num == 3:
                print(("<<<<<<<<<<<<<<<<<<< ERROR Does not decrease, use the last one >>>>>>>>>>>>>>>>>>>>>>>"))

        if loss.data <= 10.0:
            break

    # hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
    skin_v, muscle_v, bone_v, bone_joints, tex_img = nimble_layer.forward(cur_pose, cur_shape, tex_param, cur_rot, cur_trans,
        global_scale=cur_scale, handle_collision=False, no_tex=True)
    mano_v = nimble_layer.nimble_to_mano(skin_v, is_surface=True)

    # hand_verts = hand_verts - hand_verts.mean(1).unsqueeze(1) + target_vertices_mean
    # hand_joints = hand_joints - hand_joints.mean(1).unsqueeze(1) + target_vertices_mean

    if SAVE_MESH_FOR_DEBUG:
        mesh = trimesh.Trimesh(mano_v.detach().cpu().numpy()[0], mano_faces)
        mesh.export(os.path.join(out_dir, "%04d_" % idx + "vert_hand_fine.obj"))

    # Compute mano joints from vertices
    mano_v_np = mano_v.detach().cpu().numpy()
    mano_j = np.expand_dims(np.matmul(mano_j_regressor_np, mano_v_np[0]), 0)
    tips = mano_v_np[:, [745, 317, 444, 556, 673]]
    mano_j = np.concatenate([mano_j, tips], 1)
    mano_j = mano_j[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

    out_params = {}
    out_params['joints'] = mano_j
    out_params['verts'] = mano_v.detach().cpu().numpy()
    out_params['rot'] = cur_rot.detach().cpu().numpy()
    out_params['pose'] = cur_pose.detach().cpu().numpy()
    out_params['shape'] = cur_shape.detach().cpu().numpy()
    out_params['trans'] = cur_trans.detach().cpu().numpy()

    return out_params


def merge_dict(mano_dict):
    out = {}
    keys = mano_dict[0].keys()
    for k in keys:
        out[k] = []
    dict_size = len(mano_dict)
    for idx in range(dict_size):
        for k in keys:
            if k == 'cam':
                out[k].append(torch.from_numpy(mano_dict[idx][k]))
            elif k == 'seq':
                out[k].append(mano_dict[idx][k])
            else:
                out[k].append(torch.from_numpy(mano_dict[idx][k].squeeze(0)))

    for k in keys:
        if k != 'seq':
            out[k] = torch.stack(out[k])
    return out


def load_params(mano_dir):
    file_names = [] 
    for name in os.listdir(mano_dir):
        if name.endswith(".pkl"):
            file_names.append(os.path.join(mano_dir, name))
    
    params_list = []
    file_names.sort()
    for file_name in file_names:
        with open(file_name, 'rb') as f: 
            mano_param = pickle.load(f)
        params_list.append(mano_param)
    
    params = merge_dict(params_list)
    return params


# def normalize_cam(params):
#     average_cam = torch.mean(params['cam'], dim=0)
#     cam_diff = params['cam'] - average_cam
#     # params['cam'] = params['cam'] - average_cam

#     average_trans = torch.mean(params['trans'], dim=0)
#     trans_diff = params['trans'] - average_trans
#     return


def render_sequence(renderer, params, focal_length=1000, img_res=224, color='light_blue'):
    from PIL import Image
    from torchvision import transforms
    # res = 448
    # focal_length = 2000
    f_ratio = img_res / 224.0
    default_focal_length = 1000.0

    f_ratio = img_res / 224.
    focal_length = focal_length * f_ratio

    white_img = Image.new("RGB", [img_res,img_res], (255, 255, 255))
    white_img_visual = np.array(white_img) / 255.
    # white_img_visual = transforms.ToTensor()(white_img).unsqueeze(0)
    # res = 224
    
    rend_img_list = []
    for idx in range(len(params['verts'])):
        camera = params['cam'][idx]
        camera_t = np.array([camera[1], camera[2], 2*focal_length/(img_res * camera[0] +1e-9)])

        vertices_full = params['verts'][idx].cpu().numpy() / 1000.0
        rend_img = renderer.render_vertex_color(vertices_full, camera_t=camera_t,
                        img=white_img_visual, use_bg=False,
                        focal_length=focal_length) # , vertex_color=color)
        rend_img = np.asarray(rend_img)
        rend_img_list.append(rend_img)
    return rend_img_list


######### Loss Functions #########
class LossInit:
    def __init__(self, params, device='cuda') -> None:
        self.poses = torch.Tensor(params['poses']).to(device)
        self.shapes = torch.Tensor(params['shapes']).to(device)

    def init_poses(self, poses, **kwargs):
        "distance to poses_0"
        return torch.sum((poses - self.poses)**2)/poses.shape[0]
    
    def init_shapes(self, shapes, **kwargs):
        "distance to shapes_0"
        return torch.sum((shapes - self.shapes)**2)/shapes.shape[0]
    
    # def init_verts(self, shapes, **kwargs):
    #     "distance to shapes_0"
    #     return torch.sum((shapes - self.shapes)**2)/shapes.shape[0]


func_l2 = lambda x: torch.sum(x**2)
class LossKeypoints3D:
    def __init__(self, keypoints3d, device='cuda', norm='l2') -> None:
        keypoints3d = torch.Tensor(keypoints3d).to(device)
        keypoints3d = keypoints3d - keypoints3d[:, 0, :].unsqueeze(1)

        self.nJoints = keypoints3d.shape[1]
        self.keypoints3d = keypoints3d[..., :3]
        # self.conf = keypoints3d[..., 3:]
        self.nFrames = keypoints3d.shape[0]
        self.norm = norm

    def loss_func(self, kpts_est, **kwargs):
        "distance of keypoints3d"
        nJoints = min([kpts_est.shape[1], self.keypoints3d.shape[1], 21])
        diff_square = (kpts_est[:, :nJoints, :3] - self.keypoints3d[:, :nJoints, :3]) # *self.conf[:, :nJoints]
        loss_3d = func_l2(diff_square)

        return loss_3d/self.nFrames


class LossAnchor:
    def __init__(self, keypoints3d, device='cuda', norm='l2') -> None:
        keypoints3d = torch.Tensor(keypoints3d).to(device)
        self.nJoints = keypoints3d.shape[1]
        self.keypoints3d = keypoints3d[..., :3]
        # self.conf = keypoints3d[..., 3:]
        self.nFrames = keypoints3d.shape[0]
        self.norm = norm

    def loss_func(self, kpts_est, **kwargs):
        "distance of keypoints3d"
        nJoints = min([kpts_est.shape[1], self.keypoints3d.shape[1], 21])
        diff_square = (kpts_est[:, :nJoints, :3] - self.keypoints3d[:, :nJoints, :3]) # *self.conf[:, :nJoints]
        loss_3d = func_l2(diff_square)

        return loss_3d/self.nFrames


class LossSmoothPoses:
    def __init__(self, nViews, nFrames) -> None:
        self.nViews = nViews
        self.nFrames = nFrames
        self.norm = 'l2'
    
    def poses(self, poses, **kwargs):
        loss = 0
        for nv in range(self.nViews):
            poses_ = poses[nv*self.nFrames:(nv+1)*self.nFrames, ]

            poses_interp = poses_.clone().detach()
            poses_interp[1:-1] = (poses_interp[1:-1] + poses_interp[:-2] + poses_interp[2:])/3
            loss += func_l2(poses_[1:-1] - poses_interp[1:-1])
        return loss/(self.nFrames-2)/self.nViews


class LossSmoothBodyMean:
    def __init__(self) -> None:
        pass

    def body(self, kpts_est, **kwargs):
        kpts_interp = kpts_est.clone().detach()
        kpts_interp[1:-1] = (kpts_interp[:-2] + kpts_interp[2:])/2
        loss = func_l2(kpts_est[1:-1] - kpts_interp[1:-1])
        return loss/(kpts_est.shape[0] - 2)


class LossSmoothCam:
    def __init__(self) -> None:
        pass

    def body(self, kpts_est, **kwargs):
        kpts_interp = kpts_est.clone().detach()
        kpts_interp[1:-1] = (kpts_interp[:-2] + kpts_interp[2:])/2
        loss = func_l2(kpts_est[1:-1] - kpts_interp[1:-1])
        return loss/(kpts_est.shape[0] - 2)

###### End - Loss Functions ######


def optimize_smooth_seq(params_cpu, mano_layer, use_smplx_arm=False, nimble_hand=False, img_res=224, mano_j_regressor=None, total_iter_pose=1000) :
    device = "cuda"
    if nimble_hand:
        tex_param = torch.zeros(1, 10).to(device)
        mano_j_regressor = mano_j_regressor.to(device)
    else:
        mano_layer = mano_layer.to(device)

    params_cpu['cam'] = params_cpu['cam'].unsqueeze(1)
    params = {key:torch.Tensor(val).to(device) for key, val in params_cpu.items()}
    root_joints_original = params_cpu['joints'][:, 0, :].unsqueeze(1)

    print(params.keys())
    out_params = [
        params['rot'],
        params['pose'],
        params['shape'],
        params['cam'],
    ]
    for par in out_params:
        par.requires_grad = True 
    nFrames = len(params['pose'])
    # weight = {
    #         'k3d': 1e2, 'k2d': 2e-3,
    #         'reg_poses': 1e-3, 'smooth_body': 1e2,
    #         # 'collision': 1  # If the frame number is too large (more than 1000), then GPU oom
    #     }

    loss_funcs = {
        'k3d': {'func': LossKeypoints3D(params_cpu['joints']).loss_func, 'weight': 1e-2},
        'smooth_body': {'func': LossSmoothBodyMean().body, 'weight': 0}, # 1e2},
        'smooth_poses': {'func': LossSmoothPoses(1, nFrames).poses, 'weight': 1e-1},
        # 'smooth_cam': {'func': LossSmoothPoses(1, nFrames).poses, 'weight': 1e2}
        # 'reg_poses': {'func':LossRegPoses(cfg).reg_body, 'weight': 1e2},
        # 'init_poses': {'func':LossInit(params, cfg).init_poses, 'weight': 1e-3}
    }

    total_iter = total_iter_pose # 1000
    prev_loss = 999999.0
    optimizer = torch.optim.Adam(out_params, lr=1e-3)  # lr=1e-2
    for iter in range(0, total_iter):
        if nimble_hand:
            skin_v, muscle_v, bone_v, hand_joints, tex_img = mano_layer.forward(params['pose'], params['shape'], tex_param.repeat(params['pose'].shape[0], 1),
                params['rot'], params['trans'], handle_collision=False, no_tex=True)
            hand_verts = mano_layer.nimble_to_mano(skin_v, is_surface=True)

        elif use_smplx_arm:
            hand_verts, hand_joints = mano_layer(
                betas=params['shape'], global_orient=params['rot'], transl=params['trans'], right_hand_pose=params['pose'], return_type='mano')
        else:
            hand_verts, hand_joints = mano_layer(torch.cat((params['rot'], params['pose']), 1), params['shape'], params['trans'])
        root_joints = hand_joints[:, 0, :].unsqueeze(1)
        hand_joints = hand_joints - root_joints

        loss = 0.0
        for loss_k, loss_func in loss_funcs.items():
            if loss_k == 'smooth_poses':
                vv = loss_func['func'](hand_joints) * loss_func['weight']
                
            loss += loss_func['func'](hand_joints) * loss_func['weight']

        if iter % 10 == 0:
            print("smooth %6f"%(vv.data), "total: %6f"%(loss.data) )
        
        if iter > 0 and prev_loss - loss.item() < 0.00001:
            break
        # loss_rel_change = (prev_loss, loss.item())
        prev_loss = (prev_loss + loss.item()) / 2.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('Smooth loss: %6f'%(loss.data))
    
    if nimble_hand:
        skin_v, muscle_v, bone_v, hand_joints, tex_img = mano_layer.forward(params['pose'], params['shape'], tex_param.repeat(params['pose'].shape[0], 1),
                params['rot'], params['trans'], handle_collision=False, no_tex=True)
        hand_verts = mano_layer.nimble_to_mano(skin_v, is_surface=True)
        hand_joints = nimble_to_mano_joints(hand_verts, mano_j_regressor)

    elif use_smplx_arm:
        hand_verts, hand_joints = mano_layer(
            betas=params['shape'], global_orient=params['rot'], transl=params['trans'], right_hand_pose=params['pose'], return_type='mano')
    else:
        hand_verts, hand_joints = mano_layer(torch.cat((params['rot'], params['pose']), 1), params['shape'], params['trans'])
    params['joints'] = hand_joints # - hand_joints[:, 0, :].unsqueeze(1) + root_joints_original.to(device)
    params['verts'] = hand_verts   # - hand_joints[:, 0, :].unsqueeze(1) + root_joints_original.to(device)

    ### Compute original cam-relative position
    focal_length = 1000.0
    # img_res = 448 # 224
    f_ratio = img_res / 224.
    focal_length = focal_length * f_ratio
    
    if nimble_hand:
        skin_v, muscle_v, bone_v, hand_joints, tex_img = mano_layer.forward(params['pose'], params['shape'], tex_param.repeat(params['pose'].shape[0], 1),
                params['rot'], params['trans'], handle_collision=False, no_tex=True)
        hand_verts = mano_layer.nimble_to_mano(skin_v, is_surface=True)
        hand_joints = nimble_to_mano_joints(hand_verts, mano_j_regressor)
    elif use_smplx_arm:
        hand_verts, hand_joints = mano_layer(
            betas=params['shape'], global_orient=params['rot'], transl=params['trans'], right_hand_pose=params['pose'], return_type='mano')
    else:
        hand_verts, hand_joints = mano_layer(torch.cat((params['rot'], params['pose']), 1), params['shape'], params['trans'])
    camera_t = torch.stack([params['cam'][:, :, 1], params['cam'][:, :, 2], 2 * focal_length/(img_res * params['cam'][:, :, 0] +1e-9)], dim=2)
    cam_rel_root = camera_t + hand_joints[:, 0, :] / 1000.0
    cam_rel_root = cam_rel_root.detach()

    ##### Optimize cam ##############
    loss_funcs = {
        'k3d': {'func': LossAnchor(cam_rel_root).loss_func, 'weight': 1e-2},
        'smooth_poses': {'func': LossSmoothPoses(1, nFrames).poses, 'weight': 1e-2},
    }
    total_iter = 1000
    optimizer = torch.optim.Adam([params['cam']], lr=1e-3)
    sched_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    
    for iter in range(0, total_iter):
        if nimble_hand:
            skin_v, muscle_v, bone_v, hand_joints, tex_img = mano_layer.forward(params['pose'], params['shape'], tex_param.repeat(params['pose'].shape[0], 1),
                    params['rot'], params['trans'], handle_collision=False, no_tex=True)
            hand_verts = mano_layer.nimble_to_mano(skin_v, is_surface=True)
            hand_joints = nimble_to_mano_joints(hand_verts, mano_j_regressor)
        elif use_smplx_arm:
            hand_verts, hand_joints = mano_layer(
                betas=params['shape'], global_orient=params['rot'], transl=params['trans'], right_hand_pose=params['pose'], return_type='mano')
        else:
            hand_verts, hand_joints = mano_layer(torch.cat((params['rot'], params['pose']), 1), params['shape'], params['trans'])
        camera_t = torch.stack([params['cam'][:, :, 1], params['cam'][:, :, 2], 2 * focal_length/(img_res * params['cam'][:, :, 0] +1e-9)], dim=2)
        cam_rel_root = camera_t + hand_joints[:, 0, :] / 1000.0 # params['trans'][fid]

        loss = 0.0
        for loss_k, loss_func in loss_funcs.items():
            if loss_k == 'smooth_poses':
                vv = loss_func['func'](cam_rel_root) * loss_func['weight']
                
            loss += loss_func['func'](cam_rel_root) * loss_func['weight']

        if iter % 10 == 0:
            print("cam smooth %6f"%(vv.data), "total: %6f"%(loss.data) )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sched_optimizer.step(loss)

    params['cam'] = params['cam'].squeeze(1)
    params_out = {key: val.detach().cpu() for key, val in params.items()}
    return params_out


def smooth_sequence(renderer, mano_layer, mano_fit_output_dir, mano_fit_smooth_dir, use_smplx_arm=False, img_res=224):
    print(" --- Sequence Smoothing --- ")
    base_dir = os.path.dirname(os.path.dirname(mano_fit_smooth_dir))
    mano_fit_smooth_img_dir = os.path.join(base_dir, "smooth_image")
    mano_fit_img_dir = os.path.join(base_dir, "mano_fit_image")
    concat_dir = os.path.join(base_dir, "before_after_smooth")
    os.makedirs(mano_fit_smooth_img_dir, exist_ok=True)

    print(mano_fit_output_dir)

    params = load_params(mano_fit_output_dir)
    ##################################
    params = remove_spike(params)
    ##################################

    params_out = optimize_smooth_seq(params, mano_layer, use_smplx_arm=use_smplx_arm, img_res=img_res)

    write_pkl(params_out, mano_fit_smooth_dir)
    
    rend_img_list = render_sequence(renderer, params_out, img_res=img_res)

    # Write image
    for idx in range(len(rend_img_list)):

        out_name = os.path.join(mano_fit_smooth_img_dir, "%04d.jpg" % (idx)) # TODO danger
        cv2.imwrite(out_name, np.asarray(rend_img_list[idx][:,:,::-1]*255))
    
    # Save rendered images    
    save_gif(mano_fit_smooth_img_dir, os.path.join(base_dir, "smooth.gif"))
    concat_image_in_dir(mano_fit_img_dir, mano_fit_smooth_img_dir, concat_dir)



def nimble_to_mano_joints(mano_v, mano_j_regressor):
    mano_j = torch.matmul(mano_j_regressor.unsqueeze(0), mano_v)
    tips = mano_v[:, [745, 317, 444, 556, 673]]
    mano_j = torch.cat([mano_j, tips], 1)
    mano_j_out = mano_j[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
    return mano_j_out


def smooth_sequence_nimble(renderer, nimble_layer, mano_fit_output_dir, mano_fit_smooth_dir, mano_j_regressor, img_res=224):
    print(" --- Sequence Smoothing --- ")
    base_dir = os.path.dirname(os.path.dirname(mano_fit_smooth_dir))
    mano_fit_smooth_img_dir = os.path.join(base_dir, "nimble_smooth_image")
    mano_fit_img_dir = os.path.join(base_dir, "nimble_mano_fit_image")
    concat_dir = os.path.join(base_dir, "nimble_before_after_smooth")
    os.makedirs(mano_fit_smooth_img_dir, exist_ok=True)

    print(mano_fit_output_dir)

    params = load_params(mano_fit_output_dir)
    # normalize_cam(params)
    
    ############# implement remove spike
    params = remove_spike(params)
    ##################################

    params_out = optimize_smooth_seq(params, nimble_layer, img_res=img_res, nimble_hand=True, mano_j_regressor=mano_j_regressor, total_iter_pose=20)

    write_pkl(params_out, mano_fit_smooth_dir)
    
    rend_img_list = render_sequence(renderer, params_out, img_res=img_res)

    # Write image
    for idx in range(len(rend_img_list)):
        out_name = os.path.join(mano_fit_smooth_img_dir, "%04d.jpg" % (idx))
        cv2.imwrite(out_name, np.asarray(rend_img_list[idx][:,:,::-1]*255))
    
    # Save rendered images    
    save_gif(mano_fit_smooth_img_dir, os.path.join(base_dir, "nimble_smooth.gif"))
    concat_image_in_dir(mano_fit_img_dir, mano_fit_smooth_img_dir, concat_dir)



def write_pkl(params, pkl_out_dir, unscreen=True):
    data_len = len(params['cam'])
    
    for idx in range(data_len):
        # NOTE: if use unscreen, this will start with 1
        file_idx = idx
        if unscreen:
            file_idx += 1
        out = {}
        for k, v in params.items():
            if k != 'cam':
                out[k] = params[k][idx, None].detach().cpu().numpy()
            else:
                out[k] = params[k][idx].detach().cpu().numpy()

        with open(os.path.join(pkl_out_dir, "%04d_mano.pkl" % file_idx), 'wb') as f: 
            pickle.dump(out, f)


def remove_spike(params):
    poses_ = params['pose']

    poses_interp = poses_.clone().detach()
    poses_interp[1:] = (poses_interp[1:] - poses_interp[:-1])
    diff = torch.norm(poses_interp[1:], dim=1)

    new_poses = poses_.clone().detach()
    # new_poses = torch.where((diff > 1.1).unsqueeze(1), (poses_[2:] + poses_[:-2])/2.0, poses_interp[1:-1])

    for idx in range(1, len(poses_)-1):
        if diff[idx-1] > 1.0 and diff[idx] > 1.0:
            new_poses[idx] = (poses_[idx-1] + poses_[idx+1]) / 2.
    

    params['pose'] = new_poses
    return params


def concat_image_in_dir(dir1, dir2, out_dir):
    image_name_list_1 = []
    image_name_list_2 = []
    for filename in os.listdir(dir1):
        if filename.endswith(".png") or filename.endswith(".jpg") :
            image_name_list_1.append(os.path.join(dir1, filename))
    image_name_list_1.sort()

    for filename in os.listdir(dir2):
        if filename.endswith(".png") or filename.endswith(".jpg") :
            image_name_list_2.append(os.path.join(dir2, filename))
    image_name_list_2.sort()

    os.makedirs(out_dir, exist_ok=True)
    for idx, (f1, f2) in enumerate(zip(image_name_list_1, image_name_list_2)):
        image_1 = np.asarray(Image.open(f1).convert('RGB'))
        image_2 = np.asarray(Image.open(f2).convert('RGB'))

        img_array = np.concatenate([image_1, image_2], axis=1) 
        img_array = img_array.astype(np.uint8)
        out_img = Image.fromarray(img_array)
        out_img.save(os.path.join(out_dir, "%04d.jpg" % idx))

    save_gif(out_dir, os.path.join(out_dir, "out.gif"))


def save_gif(in_dir, outname):
    images = []
    for filename in sorted(glob.glob(os.path.join(in_dir, "*.jpg"))):
        images.append(imageio.imread(filename))
    imageio.mimsave(outname, images, duration=0.1)



def set_ax_limits(ax, lim=100.0): # , lim=10.0): # 0.1
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)


def batch_to_tensor_device(batch, device):
    def to_tensor(arr):
        if isinstance(arr, int):
            return arr
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        if arr.dtype == np.int64:
            arr = torch.from_numpy(arr)
        else:
            arr = torch.from_numpy(arr).float()
        return arr

    for key in batch:
        if isinstance(batch[key], np.ndarray):
            batch[key] = to_tensor(batch[key]).to(device)
        elif isinstance(batch[key], list):
            for i in range(len(batch[key])):
                if isinstance(batch[key][i], list):
                    for j in range(len(batch[key][i])):
                        if isinstance(batch[key][i][j], np.ndarray):
                            batch[key][i][j] = to_tensor(batch[key][i][j]).to(device)
                else:
                    batch[key][i] = to_tensor(batch[key][i]).to(device)
        elif isinstance(batch[key], dict):
            batch[key] = batch_to_tensor_device(batch[key], device)
        elif isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    return batch
