import os
import sys
import torch
import numpy as np
from pytorch3d.io import load_obj
from manopth.manolayer import ManoLayer
from utils.data_util import batch_to_tensor_device
from hand_models.smplx import smplx


def load_hand_model(config_dict):
    device = config_dict["device"]
    VERTS_COLOR = None

    if config_dict["model_type"] == "html":
        from htmlpth.utils.HTML import MANO_SMPL_HTML
        base_path = "./htmlpth/"
        model_path = os.path.join(base_path,"MANO_RIGHT.pkl")
        uv_path = os.path.join(base_path, "TextureBasis/uvs_right.pkl")
        tex_path = os.path.join(base_path,"TextureBasis/model_sr/model.pkl")
        html_layer = MANO_SMPL_HTML(model_path, tex_path, uv_path)

        VERTS_UVS = torch.unsqueeze(html_layer.verts_uvs, 0).cuda()
        FACES_UVS = torch.unsqueeze(html_layer.faces_uvs, 0).cuda()
        FACES_IDX = torch.unsqueeze(html_layer.faces_idx, 0).cuda()
        config_dict["html_layer"] = html_layer

    elif config_dict["model_type"] == "nimble":
        sys.path.insert(0, "../nimble/")
        from NIMBLELayer import NIMBLELayer
        nimble_dir = "../nimble"
        pm_dict_name = os.path.join(nimble_dir, r"assets/NIMBLE_DICT_9137.pkl")
        tex_dict_name = os.path.join(nimble_dir, r"assets/NIMBLE_TEX_DICT.pkl")

        # Texture
        test_template = "../template/nimble/rand_1_skin.obj"
        # Load MANO template with UV map
        # 5990 verts
        verts, faces, properties = load_obj(test_template, load_textures=True)  # load_textures=True
        VERTS_UVS = properties.verts_uvs.unsqueeze(0)  # [1, 5330, 2] [1, verts, 2]
        FACES_UVS = faces.textures_idx.unsqueeze(0)  # [1, 9984, 3] [1, faces, 3]

        # Load model
        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)

        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)

        textured_pkl = "assets/NIMBLE_TEX_FUV.pkl"
        f_uv = np.load(os.path.join(nimble_dir, textured_pkl), allow_pickle=True)
        

        nimble_mano_vreg = np.load(os.path.join(nimble_dir, "assets/NIMBLE_MANO_VREG.pkl"), allow_pickle=True)
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
        hand_layer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    else:
        # Load MANO template with UV map
        verts, faces, properties = load_obj(config_dict["MANO_TEMPLATE"], load_textures=True)
        VERTS_UVS = properties.verts_uvs.unsqueeze(0)  # [1, 4381, 2] [1, verts, 2]
        FACES_UVS = faces.textures_idx.unsqueeze(0)  # [1, 8128, 3] [1, faces, 3]
    
    if not config_dict["model_type"] == 'nimble':
        if config_dict["use_arm"]:
            model_folder = "hand_models/smplx/models/"
            hand_layer = smplx.create(model_folder, model_type='smplxarm',
                                gender='neutral', use_face_contour=False,
                                num_betas=10, use_pca=False,
                                num_expression_coeffs=10,
                                ext='npz')
            hand_layer = hand_layer.to(device)
        else:
            # METRO compatible MANO
            hand_layer = ManoLayer(mano_root='mano/models', flat_hand_mean=False, use_pca=False)
            hand_layer = hand_layer.to(device)

            # Get MANO rainbow vertex color
            from utils.opt_utils import get_mano_vert_colors
            VERTS_COLOR = get_mano_vert_colors(hand_layer)

    return hand_layer, VERTS_UVS, FACES_UVS, VERTS_COLOR