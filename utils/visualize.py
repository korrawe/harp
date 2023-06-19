import os
import torch
import numpy as np
import glob
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.transforms import RotateAxisAngle
import renderer.renderer_helper as renderer_helper
from renderer.pbr_materials import PBRMaterials


def prepare_mesh(params, fid, mano_layer, verts_textures, mesh_subdivider, global_pose, configs, device='cuda', 
        vis_normal=False, shared_texture=True, use_arm=False):
    batch_size = fid.shape[0]  # params['pose'].shape[0]

    global_pose = False
    if global_pose:
        # Use the first pose parameter if global_pose is true
        pose_batch = params['pose'][0].repeat(batch_size, 1)
        rot_batch = params['rot'][0].repeat(batch_size, 1)
    else:
        pose_batch = params['pose'][fid]
        rot_batch = params['rot'][fid]
    
    if configs['model_type'] == 'nimble':
        cur_shape = params['shape'].repeat([batch_size, 1]).to(device)
        tex_param = params['nimble_tex'].repeat([batch_size, 1]).to(device)
        # Note that the joints here is NIMBLE bone joint
        hand_verts, muscle_v, bone_v, hand_joints, tex_img = mano_layer.forward(pose_batch.to(device), cur_shape, tex_param, rot_batch.to(device),
            params['trans'][fid].to(device), handle_collision=False, no_tex=False)

    elif use_arm:
        hand_verts, hand_joints = mano_layer(betas=params['shape'].repeat([batch_size, 1]).to(device), 
            global_orient=rot_batch.to(device), transl=params['trans'][fid].to(device), right_hand_pose=pose_batch.to(device), 
            right_wrist_pose=params['wrist_pose'][fid].to(device),
            return_type='mano_w_arm')
    else:
        hand_verts, hand_joints = mano_layer(torch.cat((rot_batch, pose_batch), 1).to(device),
                params['shape'].repeat([batch_size, 1]).to(device),
                params['trans'][fid].to(device))
    hand_verts = hand_verts / 1000.0
    hand_joints = hand_joints / 1000.0
    faces = params['mesh_faces'].repeat(batch_size, 1, 1)

    if params['verts_disps'] is not None:
        mesh_mano = Meshes(hand_verts.to(device), faces.to(device))
        if mesh_subdivider is not None:
            mesh = mesh_subdivider(mesh_mano)
            # NOTE: cannot call subdivide_homogeneous(mesh_mano) directly, _N will be set to 1 instead of len(mesh)
            # mesh._compute_vertex_normals()
        else:
            mesh = mesh_mano
        # Displacement along normal
        if params['verts_disps'].shape[1] == 1:
            verts_offset = (mesh.verts_normals_padded() * params['verts_disps'].repeat(batch_size, 1, 1))
        # Displacement in any direction
        else:
            verts_offset = params['verts_disps'].repeat(batch_size, 1, 1)

        hand_verts = mesh.verts_padded() + verts_offset
        faces = mesh.faces_padded()

    if verts_textures:
        if vis_normal:
            mesh = Meshes(hand_verts.to(device), faces.to(device))
            verts_normal = mesh.verts_normals_padded()
            verts_color = (verts_normal + 1.0) / 2.0
        else:
            verts_color = params['verts_rgb']
        textures = TexturesVertex(verts_features=verts_color.repeat(batch_size, 1, 1).float())
        mesh = Meshes(hand_verts.to(device), faces.to(device), textures.to(device))
    else:
        if configs['model_type'] == 'nimble':
            uv_map = tex_img[..., [2,1,0]]
            params['normal_map'] = tex_img[..., [6,5,4]].detach()
        elif shared_texture:
            uv_map = params['texture'][None, 0].repeat(batch_size, 1, 1, 1).to(device)
        else:
            uv_map = params['texture'].to(device)
        textures = TexturesUV(maps=uv_map,
                                faces_uvs=params['faces_uvs'].repeat(batch_size, 1, 1).type(torch.long).to(device),
                                verts_uvs=params['verts_uvs'].repeat(batch_size, 1, 1).to(device))
        mesh = Meshes(hand_verts.to(device), faces.to(device), textures.to(device))
    return hand_joints.to(device), hand_verts.to(device), faces.to(device), textures.to(device) # mesh


def prepare_materials(params, batch_size, shared_texture=True, device='cuda'):
    if 'normal_map' in params:
        if shared_texture:
            normal_maps = params['normal_map'][None, 0].repeat(batch_size, 1, 1, 1).to(device)
        else:
            normal_maps = params['normal_map'].to(device)

        normal_maps = torch.nn.functional.normalize(normal_maps, dim=-1)
        normal_maps = TexturesUV(maps=normal_maps,
                                faces_uvs=params['faces_uvs'].repeat(batch_size, 1, 1).type(torch.long).to(device),
                                verts_uvs=params['verts_uvs'].repeat(batch_size, 1, 1).to(device))
    else:
        normal_maps = None

    materials_prop = {
        "normal_maps": normal_maps,
    }
    return materials_prop


def change_pose(params, idx):
    minn = -1.0
    maxx = 1.0
    sz = maxx - minn
    # idx / 36 * sz + minn
    # index
    params['pose'][0][0] = 0.0; params['pose'][0][1] = -0.3; params['pose'][0][2] = 0.7 # 0.9
    params['pose'][0][3] = 0.0; params['pose'][0][4] = 0.0; params['pose'][0][5] = -0.1 # 0.9
    params['pose'][0][6] = 0.0; params['pose'][0][7] = 0.0; params['pose'][0][8] = 0.0

    # middle
    params['pose'][0][9] = 0.0; params['pose'][0][10] = 0.0; params['pose'][0][11] = 0.0 # 0.9
    params['pose'][0][12] = 0.0; params['pose'][0][13] = 0.0; params['pose'][0][14] = 0.0 # 0.9
    params['pose'][0][15] = 0.0; params['pose'][0][16] = 0.0; params['pose'][0][17] = 0.0


    # pinky
    params['pose'][0][18] = 0.0; params['pose'][0][19] = 0.0; params['pose'][0][20] = 0.0 # 0.6
    params['pose'][0][21] = 0.2; params['pose'][0][22] = 0.0; params['pose'][0][23] = -0.6
    params['pose'][0][24] = -0.0; params['pose'][0][25] = 0.0; params['pose'][0][26] = 0.0

    # ring
    params['pose'][0][27] = 0.0; params['pose'][0][28] = -0.2; params['pose'][0][29] = 0.8
    params['pose'][0][30] = 0.0; params['pose'][0][31] = 0.0; params['pose'][0][32] = 0.8
    params['pose'][0][33] = 0.0; params['pose'][0][34] = 0.0; params['pose'][0][35] = 0.0

    # Thumb
    params['pose'][0][36] = 0.5; params['pose'][0][37] = 0.5; params['pose'][0][38] = 0.1
    params['pose'][0][39] = 0.6; params['pose'][0][40] = -0.7; params['pose'][0][41] = 1.0
    params['pose'][0][42] = 0.0; params['pose'][0][43] = -1.0; params['pose'][0][44] = 0.1

    return params


def render_360(params, fid, renderer, img_size, focal_length, mano_layer, configs, render_normal=False,
        verts_textures=True, mesh_subdivider=None, global_pose=False, global_betas=True,
        device='cuda', save_img_dir=None, use_arm=False):
    hand_joints, hand_verts, faces, textures = prepare_mesh(params, fid, mano_layer, verts_textures, mesh_subdivider, global_pose, 
        configs=configs, device=device, use_arm=use_arm)
    cam = params['cam'][fid]
    batch_size = fid.shape[0]
    # Material properties
    materials_properties = prepare_materials(params, batch_size)
    if render_normal:
        out_dir = os.path.join(save_img_dir, "render_360_normal")
    else:
        out_dir = os.path.join(save_img_dir, "render_360")
    os.makedirs(out_dir, exist_ok=True)

    mesh = Meshes(hand_verts, faces, textures)
    y_center = torch.mean(hand_verts[:, :, 1], dim=(1))
    center = torch.mean(hand_verts[:, :, :], dim=(1), keepdim=True)

    for idx in range(36):
        hand_verts = hand_verts - center
        rot_y_10 = RotateAxisAngle(10, 'Y', device=device)
        hand_verts = rot_y_10.transform_points(hand_verts)
        # rot_y_10 = RotateAxisAngle(10, 'Y', device=device)
        # hand_verts = rot_y_10.transform_points(hand_verts)
        hand_verts = hand_verts + center
        mesh = Meshes(hand_verts, faces, textures)
        rendered_img = render_image(mesh, cam, batch_size, renderer, img_size, focal_length, materials_properties=materials_properties)

        rendered_img = rendered_img.detach().cpu().numpy()[0].clip(0,1) * 255
        rendered_img = rendered_img.astype(np.uint8)
        fig_out_dir = os.path.join(out_dir, "%04d.jpg" % (idx))
        out_img = Image.fromarray(rendered_img)
        out_img.save(fig_out_dir)
    
    for idx in range(36):
        # hand_verts[:, :, 1] = hand_verts[:, :, 1] - y_center
        hand_verts = hand_verts - center
        rot_x_10 = RotateAxisAngle(10, 'X', device=device)
        hand_verts = rot_x_10.transform_points(hand_verts)
        hand_verts = hand_verts + center
        # hand_verts[:, :, 1] = hand_verts[:, :, 1] + y_center
        mesh = Meshes(hand_verts, faces, textures)
        rendered_img = render_image(mesh, cam, batch_size, renderer,  img_size, focal_length, materials_properties=materials_properties)

        rendered_img = rendered_img.detach().cpu().numpy()[0].clip(0,1) * 255
        rendered_img = rendered_img.astype(np.uint8)
        fig_out_dir = os.path.join(out_dir, "h_%04d.jpg" % (idx))
        out_img = Image.fromarray(rendered_img)
        out_img.save(fig_out_dir)
    
    save_gif(out_dir, os.path.join(out_dir, "out.gif"))


def render_360_light(params, fid, hand_verts, faces, textures, img_size, focal_length, save_img_dir=None, device='cuda'):
    cam = params['cam'][fid]
    batch_size = fid.shape[0]
    out_dir = os.path.join(save_img_dir, "render_360_light")
    os.makedirs(out_dir, exist_ok=True)

    mesh = Meshes(hand_verts, faces, textures)
    y_center = torch.mean(hand_verts[:, :, 1], dim=(1))
    center = torch.mean(hand_verts[:, :, :], dim=(1), keepdim=True)
    start = -5.0
    end = 5.0
    num = 40
    for idx in range(num):
        cur_z = start + (end - start)/num * idx
        # light_positions = params['light_positions'][fid]
        light_positions = torch.Tensor(((1.0, 1.0, cur_z)))
        light_positions = light_positions.repeat(batch_size, 1)

        phong_renderer, silhouette_renderer, normal_renderer = renderer_helper.get_renderers(image_size=img_size, 
            light_posi=light_positions, silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, device=device)
        
        rendered_img = render_image(mesh, cam, batch_size, phong_renderer, img_size, focal_length)

        rendered_img = rendered_img.detach().cpu().numpy()[0].clip(0,1) * 255
        rendered_img = rendered_img.astype(np.uint8)
        fig_out_dir = os.path.join(out_dir, "%04d.jpg" % (idx))
        out_img = Image.fromarray(rendered_img)
        out_img.save(fig_out_dir)
    
    save_gif(out_dir, os.path.join(out_dir, "out.gif"))


## Currently not used
def render_with_rotation(hand_verts, faces, textures, cam, x_rot, y_rot, z_rot,
                         renderer, img_size, focal_length,
                         device='cuda', save_img_dir=None):
    batch_size = 1 # fid.shape[0]
    os.makedirs(save_img_dir + "render_360", exist_ok=True)

    mesh = Meshes(hand_verts, faces, textures)
    center = torch.mean(hand_verts[:, :, :], dim=(1), keepdim=True)

    hand_verts = hand_verts - center
    rot_y_10 = RotateAxisAngle(x_rot, 'X', device=device)
    rot_y_10 = RotateAxisAngle(y_rot, 'Y', device=device)
    rot_y_10 = RotateAxisAngle(z_rot, 'Z', device=device)
    hand_verts = rot_y_10.transform_points(hand_verts)
    hand_verts = hand_verts + center

    mesh = Meshes(hand_verts, faces, textures)
    rendered_img = render_image(mesh, cam, batch_size, renderer, img_size, focal_length)

    rendered_img = rendered_img.detach().cpu().numpy()[0].clip(0,1) * 255
    rendered_img = rendered_img.astype(np.uint8)
    fig_out_dir = os.path.join(save_img_dir, "render_360", "%04d.jpg" % (idx))
    out_img = Image.fromarray(rendered_img)
    out_img.save(fig_out_dir)


def render_image(mesh, cam, batch_size, renderer, img_size, focal_length, silhouette=False, device='cuda', materials_properties=dict()):
    materials = PBRMaterials(
        device=device,
        shininess=0.0,
        **materials_properties,
    )
    fx = focal_length
    px = img_size / 2.
    py = img_size / 2.
    
    camera_t = torch.stack([-cam[:, 1], -cam[:, 2], 2 * focal_length/(img_size * cam[:, 0] +1e-9)], dim=1)
    camera_t = camera_t.to(device)
    # Flip X and Y to convert OpenCV camera to Pytorch3D camera
    R_batch = torch.Tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]).repeat(batch_size, 1, 1).to(device)
    rendered_img = renderer(mesh,
                    principal_point=torch.Tensor([(px, py)]), # pp_batch,
                    focal_length=fx, # f_batch,  # ((fx, fy),),
                    T=camera_t,
                    R=R_batch,
                    materials=materials,
                    image_size=torch.Tensor([(img_size, img_size)])  # ((img_width, img_height),)
                )
    if silhouette:
        rendered_img = rendered_img[:, :, :, 3]
    else:
        # color
        rendered_img = rendered_img[:, :, :, 0:3]
    return rendered_img


def render_image_with_RT(mesh, light_t, light_r, cam_t, cam_r, batch_size, renderer, img_size, focal_length,
    silhouette=False, materials_properties=dict(), device='cuda'): # cam_r

    materials = PBRMaterials(
        device=device,
        shininess=0.0,
        **materials_properties,
    )
    fx = focal_length
    px = img_size / 2.
    py = img_size / 2.
    cam_t = cam_t.to(device)
    cam_r = cam_r.to(device)
    light_t = light_t.to(device)
    light_r = light_r.to(device)

    rendered_img = renderer(mesh,
                    principal_point=torch.Tensor([(px, py)]), # pp_batch,
                    focal_length=fx, # f_batch,  # ((fx, fy),),
                    T=light_t,
                    R=light_r, # R_batch,
                    cam_T=cam_t,
                    cam_R=cam_r,
                    materials=materials, 
                    image_size=torch.Tensor([(img_size, img_size)]), # ((img_width, img_height),)
                )
    if silhouette:
        rendered_img = rendered_img[:, :, :, 3]
    else:
        # color
        rendered_img = rendered_img[:, :, :, 0:3]
    return rendered_img


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
    import imageio

    images = []
    for filename in sorted(glob.glob(os.path.join(in_dir, "*.jpg"))):
        images.append(imageio.imread(filename))
    imageio.mimsave(outname, images, duration=0.1)