import copy
import os
import sys
import threading

import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import tkinter as tk

import matplotlib.pyplot as plt

from load_nerd import load_nerd_data
from load_blender import load_blender_data
from run_nerf_helpers_tcnn import NeRF_TCNN
from run_nerf_helpers import *
from correspondence_utils import *

from load_llff import load_llff_data, load_colmap_depth
from load_dtu import load_dtu_data

from .loss import SigmaLoss

from .data import RayDataset
from torch.utils.data import DataLoader

from .utils.generate_renderpath import generate_renderpath
import cv2
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
np.random.seed(0)
DEBUG = False

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

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, need_alpha=False, detach_weights=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(
            rays_flat[i:i + chunk], need_alpha=need_alpha, detach_weights=detach_weights, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None, depths=None, need_alpha=False, detach_weights=False,
           patch=None,hand_infos=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
      hand_infos:dict,
        th_to_action_verts: (Verts,4,4),
        faces : (Faces,3)
        hand_verts: (Verts,3)

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    rays_o, rays_d = get_rays(H, W, focal, c2w)

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = geometry_guided_near_far_torch(rays_o, rays_d, hand_verts ,geo_threshold=0.1)

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # B x 8
    if depths is not None:
        rays = torch.cat([rays, depths.reshape(-1, 1)], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(
        rays, chunk, need_alpha=need_alpha, detach_weights=detach_weights, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0,
                disp_require_grad=False, need_alpha=False, rgb_require_grad=False, detach_weights=False,
                patch_len=None, masks=None):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    if savedir is not None:
        np.savetxt(
            os.path.join(savedir, 'intrinsics.txt'),
            K
        )

    rgbs = []
    disps = []

    Xs = []
    Ys = []

    for i, c2w in enumerate(render_poses):
        if disp_require_grad or rgb_require_grad:
            if patch_len is not None:
                masked = np.where(masks[i] != 0)
                masked = (masked[0] // render_factor,
                          masked[1] // render_factor)
                Xs.append(random.randint(
                    masked[0].min(),
                    max(masked[0].max() - patch_len[0], masked[0].min())
                ))
                Ys.append(random.randint(
                    masked[1].min(),
                    max(masked[1].max() - patch_len[1], masked[1].min())
                ))
                patch = (Xs[-1], Ys[-1], patch_len[0], patch_len[1])
            else:
                patch = None
            rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4],
                                                   retraw=True, need_alpha=need_alpha,
                                                   detach_weights=detach_weights, patch=patch,
                                                   **render_kwargs)
        else:
            with torch.no_grad():
                rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4],
                                                       retraw=True, need_alpha=need_alpha, **render_kwargs)

        if disp_require_grad:
            disps.append(disp)
        else:
            disps.append(disp.detach().cpu().numpy())

        if rgb_require_grad:
            rgbs.append(rgb)
        else:
            rgbs.append(rgb.detach().cpu().numpy())

        if savedir is not None:
            rgb_dir = os.path.join(savedir, 'rgb')
            depth_dir = os.path.join(savedir, 'depth')
            disp_dir = os.path.join(savedir, 'disp')
            weight_dir = os.path.join(savedir, 'weight')
            alpha_dir = os.path.join(savedir, 'alpha')
            gt_img_dir = os.path.join(savedir, 'images')
            z_dir = os.path.join(savedir, 'z')
            pose_dir = os.path.join(savedir, 'pose')
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(gt_img_dir, exist_ok=True)
            os.makedirs(weight_dir, exist_ok=True)
            os.makedirs(z_dir, exist_ok=True)
            os.makedirs(pose_dir, exist_ok=True)
            os.makedirs(disp_dir, exist_ok=True)
            if need_alpha:
                os.makedirs(alpha_dir, exist_ok=True)

            rgb8 = to8b(rgbs[-1])
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(rgb_dir, '{:06d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            if gt_imgs is not None:
                gt_filename = os.path.join(gt_img_dir, '{:06d}.png'.format(i))
                try:
                    imageio.imwrite(gt_filename, to8b(
                        gt_imgs[i].detach().cpu().numpy()))
                except:
                    imageio.imwrite(gt_filename, to8b(gt_imgs[i]))

            depth = depth.cpu().numpy()

            np.save(
                os.path.join(depth_dir, '{:06d}.npy'.format(i)),
                depth
            )
            np.save(
                os.path.join(disp_dir, '{:06d}.npy'.format(i)),
                disp.cpu().numpy()
            )
            np.save(
                os.path.join(weight_dir, '{:06d}.npy'.format(i)),
                extras['weights'].cpu().numpy()
            )
            np.save(
                os.path.join(z_dir, '{:06d}.npy'.format(i)),
                extras['z_vals'].cpu().numpy()
            )
            if need_alpha:
                np.save(
                    os.path.join(alpha_dir, '{:06d}.npy'.format(i)),
                    extras['alpha'].cpu().numpy()
                )

            render_pose = np.concatenate(
                [render_poses[i, :3, :4].detach().cpu().numpy(),
                 np.array([[0, 0, 0, 1]])],
                axis=0
            )
            np.savetxt(
                os.path.join(pose_dir, '{:06d}.txt'.format(i)),
                render_pose
            )

    if disp_require_grad:
        disps = torch.stack(disps, 0)
    else:
        disps = np.stack(disps, 0)

    if rgb_require_grad:
        rgbs = torch.stack(rgbs, 0)
    else:
        rgbs = np.stack(rgbs, 0)

    return rgbs, disps, (Xs, Ys)


def render_path_projection(render_poses, hwf, chunk, render_kwargs, render_factor=0):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])

    z_vals = []
    weights = []
    c2ws = []
    for i, c2w in enumerate(render_poses):
        with torch.no_grad():
            rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], retraw=True,
                                                   **render_kwargs)

        z_vals.append(extras['z_vals'].cpu().numpy())
        weights.append(extras['weights'].cpu().numpy())
        c2ws.append(convert_pose(np.concatenate(
            [render_poses[i, :3, :4].detach().cpu().numpy(), np.array([[0, 0, 0, 1]])], axis=0
        )))

    return z_vals, weights, c2ws, K


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def render_test_ray(rays_o, rays_d, hwf, ndc, near, far, use_viewdirs, N_samples, network, network_query_fn, **kwargs):
    H, W, focal = hwf
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    z_vals = near * (1. - t_vals) + far * (t_vals)

    z_vals = z_vals.reshape([rays_o.shape[0], N_samples])

    rgb, sigma, depth_maps = sample_sigma(
        rays_o, rays_d, viewdirs, network, z_vals, network_query_fn)

    return rgb, sigma, z_vals, depth_maps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if args.alpha_model_path is None:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars = list(model.parameters())
    else:
        alpha_model = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                           input_ch=input_ch, output_ch=output_ch, skips=skips,
                           input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        print('Alpha model reloading from', args.alpha_model_path)
        ckpt = torch.load(args.alpha_model_path)
        alpha_model.load_state_dict(ckpt['network_fine_state_dict'])
        if not args.no_coarse:
            model = NeRF_RGB(D=args.netdepth, W=args.netwidth,
                             input_ch=input_ch, output_ch=output_ch, skips=skips,
                             input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, alpha_model=alpha_model).to(
                device)
            grad_vars = list(model.parameters())
        else:
            model = None
            grad_vars = []

    model_fine = None
    if args.N_importance > 0:
        if args.alpha_model_path is None:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        else:
            model_fine = NeRF_RGB(D=args.netdepth_fine, W=args.netwidth_fine,
                                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                                  input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                                  alpha_model=alpha_model).to(device)
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(inputs, viewdirs, network_fn,
                                                                           embed_fn=embed_fn,
                                                                           embeddirs_fn=embeddirs_fn,
                                                                           netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if args.sigma_loss:
        render_kwargs_train['sigma_loss'] = SigmaLoss(
            args.N_samples, args.perturb, args.raw_noise_std)

    ##########################

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def create_nerf_tcnn(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = lambda inp: inp, 3

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = lambda inp: inp, 3
    output_ch = 5 if args.N_importance > 0 else 4
    
    
    model = NeRF_TCNN(
        encoding="hashgrid",
    )
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF_TCNN(
            encoding="hashgrid",
        )
        grad_vars += list(model_fine.parameters())
        
    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    basedir = args.basedir
    expname = args.expname

   ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None' :
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]
    # if args.alpha_model_path is not None:
    #     ckpts=[args.alpha_model_path] if args.alpha_model_path.endswith(".pth") or args.alpha_model_path.endswith(".tar") \
    #             else [os.path.join(args.alpha_model_path, f) for f in sorted(os.listdir(args.alpha_model_path)) if 'tar' in f]
    # else:
    #     ckpts = []

    if args.masked_NeRF or args.object_removal or args.no_reload:
        ckpts = []

    print(f'Found ckpts {ckpts} no_reload {args.no_reload},')
    
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(inputs, viewdirs, network_fn,
                                                                           embed_fn=embed_fn,
                                                                           embeddirs_fn=embeddirs_fn,
                                                                           netchunk=args.netchunk)

    start = 0 if not len(ckpts) else ckpt['global_step']

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False,
                sigma_loss=None,
                verbose=False,
                need_alpha=False,
                detach_weights=False
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    #     raw = run_network(pts)
    if network_fn is not None:
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                            white_bkgd, pytest=pytest,
                                                                            need_alpha=need_alpha,
                                                                            detach_weights=detach_weights)
    else:
        if network_fine.alpha_model is not None:
            raw = network_query_fn(pts, viewdirs, network_fine.alpha_model)
            rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                white_bkgd, pytest=pytest,
                                                                                need_alpha=need_alpha,
                                                                                detach_weights=detach_weights)
        else:
            raw = network_query_fn(pts, viewdirs, network_fine)
            rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                white_bkgd, pytest=pytest,
                                                                                need_alpha=need_alpha,
                                                                                detach_weights=detach_weights)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, alpha0 = rgb_map, disp_map, acc_map, alpha

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                   None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                            white_bkgd, pytest=pytest,
                                                                            need_alpha=need_alpha,
                                                                            detach_weights=detach_weights)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map,
           'weights': weights, 'z_vals': z_vals}
    if retraw:
        ret['raw'] = raw
    if need_alpha:
        ret['alpha'] = alpha
        ret['alpha0'] = alpha0
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    if sigma_loss is not None and ray_batch.shape[-1] > 11:
        depths = ray_batch[:, 8]
        ret['sigma_loss'] = sigma_loss.calculate_loss(rays_o, rays_d, viewdirs, near, far, depths, network_query_fn,
                                                      network_fine)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def render_rays_given_pts(pts,rays_o,rays_d,z_vals,
                network_fn,
                network_query_fn,
                retraw=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False,
                # sigma_loss=None,
                need_alpha=False,
                detach_weights=False
                ):
    """Volumetric rendering.
    Args:
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    viewdirs=rays_d/rays_d.norm(dim=-1,keepdim=True)

    if network_fn is not None:
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                            white_bkgd, pytest=pytest,
                                                                            need_alpha=need_alpha,
                                                                            detach_weights=detach_weights)
    else:
        if network_fine.alpha_model is not None:
            raw = network_query_fn(pts, viewdirs, network_fine.alpha_model)
            rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                white_bkgd, pytest=pytest,
                                                                                need_alpha=need_alpha,
                                                                                detach_weights=detach_weights)
        else:
            raw = network_query_fn(pts, viewdirs, network_fine)
            rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                white_bkgd, pytest=pytest,
                                                                                need_alpha=need_alpha,
                                                                                detach_weights=detach_weights)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, alpha0 = rgb_map, disp_map, acc_map, alpha

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                   None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                            white_bkgd, pytest=pytest,
                                                                            need_alpha=need_alpha,
                                                                            detach_weights=detach_weights)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map,
           'weights': weights, 'z_vals': z_vals}
    if retraw:
        ret['raw'] = raw
    if need_alpha:
        ret['alpha'] = alpha
        ret['alpha0'] = alpha0
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=0.01,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=10,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_ray", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true',
                        help='render the train set instead of render_poses path')
    parser.add_argument("--render_mypath", action='store_true',
                        help='render the test path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=1000000,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # debug
    parser.add_argument("--debug", action='store_true')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')
    parser.add_argument("--colmap_depth", action='store_true',
                        help="Use depth supervision by colmap.")
    parser.add_argument("--depth_loss", action='store_true',
                        help="Use depth supervision by colmap - depth loss.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help="Depth lambda used for loss.")
    parser.add_argument("--sigma_loss", action='store_true',
                        help="Use depth supervision by colmap - sigma loss.")
    parser.add_argument("--sigma_lambda", type=float, default=0.1,
                        help="Sigma lambda used for loss.")
    parser.add_argument("--weighted_loss", action='store_true',
                        help="Use weighted loss by reprojection error.")
    parser.add_argument("--relative_loss", action='store_true',
                        help="Use relative loss.")
    parser.add_argument("--depth_with_rgb", action='store_true',
                        help="single forward for both depth and rgb")
    parser.add_argument("--normalize_depth", action='store_true',
                        help="normalize depth before calculating loss")

    parser.add_argument("--no_tcnn", action='store_true',
                        help='set to not use tinycudann and use the original NeRF')

    parser.add_argument("--clf_weight", type=float, default=0.01,
                        help='The weight of the classification loss')
    parser.add_argument("--clf_reg_weight", type=float, default=0.01,
                        help='The weight of the classification regularizer')
    parser.add_argument("--feat_weight", type=float,
                        default=0.01, help='The weight of the feature loss')
    parser.add_argument("--i_feat", type=int, default=10,
                        help='frequency of calculating the feature loss')
    parser.add_argument("--prepare", action='store_true',
                        help='Prepare depths for inpainting')
    parser.add_argument("--lpips", action='store_true',
                        help='use perceptual loss for rgb inpainting')
    parser.add_argument("--N_gt", type=int, default=0,
                        help='Number of ground truth inpainted samples')
    parser.add_argument("--N_train", type=int, default=None,
                        help='Number of training images used for optimization')
    parser.add_argument("--train_gt", action='store_true',
                        help='Use the gt inpainted images to train a NeRF')
    parser.add_argument("--masked_NeRF", action='store_true',
                        help='Only train NeRF on unmasked pixels')
    parser.add_argument("--object_removal", action='store_true',
                        help='Remove the object and shrink the masks')
    parser.add_argument("--tmp_images", action='store_true',
                        help='Use images in lama_images_tmp for ablation studies')
    parser.add_argument("--no_geometry", action='store_true',
                        help='Stop using inpainted depths for training')

    parser.add_argument("--lpips_render_factor", type=int, default=2,
                        help='The stride (render factor) used for sampling patches for the perceptual loss')
    parser.add_argument("--patch_len_factor", type=int, default=8,
                        help='The resizing factor to obtain the side lengths of the patches for the perceptual loss')
    parser.add_argument("--lpips_batch_size", type=int, default=4,
                        help='The number of patches used in each iteration for the perceptual loss')

    # NEW
    # vis_triplane
    parser.add_argument("--vis_triplane", action='store_true',
                        help='Visualize triplane and exit')
    # tv_lambda
    parser.add_argument("--tv_lambda", type=float, default=0.,
                        help='The weight of the TV loss')
    # sparse_lambda
    parser.add_argument("--sparse_lambda", type=float, default=0.,
                        help='The weight of the sparse loss')
    # edr_lambda
    parser.add_argument("--edr_lambda", type=float, default=0.,
                        help='The weight of the EDR loss')


    return parser


def gui_application(args, render_kwargs_test):
    root = tk.Tk()
    root.geometry("300x520")

    def set_values():
        args.feat_weight = float(feat.get())
        args.i_video = int(i_video.get())
        args.render_factor = int(render_factor.get())
        if white_bkgd.get() == 1:
            args.white_bkgd = True
        else:
            args.white_bkgd = False
        render_kwargs_test['white_bkgd'] = args.white_bkgd

    tk.Label(root, text="Feature weight").pack()
    feat = tk.Entry(root, textvariable=tk.StringVar(
        root, value=str(args.feat_weight)))
    feat.pack()
    tk.Label(root, text="i_video").pack()
    i_video = tk.Entry(root, textvariable=tk.StringVar(
        root, value=str(args.i_video)))
    i_video.pack()
    tk.Label(root, text="render factor").pack()
    render_factor = tk.Entry(root, textvariable=tk.StringVar(
        root, value=str(args.render_factor)))
    render_factor.pack()

    white_bkgd = tk.IntVar()
    tk.Checkbutton(root, text='White BG', onvalue=1,
                   offvalue=0, variable=white_bkgd).pack()

    tk.Button(root, text='Submit', command=set_values).pack()
    root.mainloop()


def train():
    parser = config_parser()
    args = parser.parse_args()

    gnrt_losses = []
    disc_losses = []

    if args.lpips:
        LPIPS = lpips.LPIPS(net='vgg')
        # LPIPS.eval()
        for param in LPIPS.parameters():
            param.requires_grad = False

    # Load data

    if args.dataset_type == 'llff':
        if args.colmap_depth:
            depth_gts = load_colmap_depth(
                args.datadir, factor=args.factor, bd_factor=.75, prepare=args.prepare)
        images, poses, bds, render_poses, i_test, masks, inpainted_depths, mask_indices = load_llff_data(args.datadir,
                                                                                                         args.factor,
                                                                                                         recenter=True,
                                                                                                         bd_factor=.75,
                                                                                                         spherify=args.spherify,
                                                                                                         prepare=args.prepare,
                                                                                                         args=args)

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0]))])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        # if args.prepare:  # todo uncomment?
        #     masks = np.abs(masks)

        if args.object_removal:
            masks = np.abs(masks)

        if args.N_gt > 0:
            if not args.train_gt:
                i_test = i_train[:args.N_gt]
                if args.N_train is None:
                    i_train = i_train[args.N_gt:]
                else:
                    i_train = i_train[args.N_gt:args.N_gt + args.N_train]
            else:
                i_test = i_train
                i_train = i_train[:args.N_gt]

    elif args.dataset_type == 'dtu':
        images, poses, hwf = load_dtu_data(args.datadir)
        print('Loaded DTU', images.shape, poses.shape, hwf, args.datadir)
        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        near = 0.1
        far = 5.0
        if args.colmap_depth:
            depth_gts = load_colmap_depth(
                args.datadir, factor=args.factor, bd_factor=.75)
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, masks, objects = load_blender_data(args.datadir,
                                                                                      args.half_res,
                                                                                      args.testskip)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * \
                images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    elif args.dataset_type == 'nerd':
        images, poses, bds, render_poses, i_test, masks, objects = load_nerd_data(args.datadir, args.factor,
                                                                                  recenter=True, bd_factor=.75,
                                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        # plt.imshow(images[0])
        # plt.savefig('sample.png')
        # plt.clf()
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])
    elif args.render_mypath:
        # render_poses = generate_renderpath(np.array(poses[i_test]), focal)
        render_poses = generate_renderpath(
            np.array(poses[i_test])[3:4], focal, sc=1)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    if args.no_tcnn:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
            args)
    else:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf_tcnn(
            args)

    # gui = threading.Thread(target=gui_application,
    #                        args=(args, render_kwargs_test,))
    # gui.start()

    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)


    if args.vis_triplane:
        render_kwargs_train['network_fn'].vis_triplane(two_exp_reso=8,vis_dir=os.path.join(basedir, expname,'triplane'),
                                                       additional_message=['iter',str(global_step),'coarse'])
        if 'network_fine' in render_kwargs_train and render_kwargs_train['network_fine'] is not None:
            render_kwargs_train['network_fine'].vis_triplane(two_exp_reso=8,vis_dir=os.path.join(basedir, expname,'triplane'),
                                                    additional_message=['iter',str(global_step),'fine'])
        return


    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = images[i_train]

            if args.render_test:
                testsavedir = os.path.join(
                    basedir, expname, 'renderonly_{}_{:06d}'.format('test', start))
            elif args.render_train:
                testsavedir = os.path.join(
                    basedir, expname, 'renderonly_{}_{:06d}'.format('train', start))
            else:
                testsavedir = os.path.join(
                    basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            if args.render_test_ray:
                # rays_o, rays_d = get_rays(H, W, focal, render_poses[0])
                index_pose = i_train[0]
                rays_o, rays_d = get_rays_by_coord_np(H, W, focal, poses[index_pose, :3, :4],
                                                      depth_gts[index_pose]['coord'])
                rays_o, rays_d = torch.Tensor(rays_o).to(
                    device), torch.Tensor(rays_d).to(device)
                rgb, sigma, z_vals, depth_maps = render_test_ray(rays_o, rays_d, hwf,
                                                                 network=render_kwargs_test['network_fine'],
                                                                 **render_kwargs_test)
                # sigma = sigma.reshape(H, W, -1).cpu().numpy()
                # z_vals = z_vals.reshape(H, W, -1).cpu().numpy()
                # np.savez(os.path.join(testsavedir, 'rays.npz'), rgb=rgb.cpu().numpy(), sigma=sigma.cpu().numpy(), z_vals=z_vals.cpu().numpy())
                visualize_sigma(sigma[0, :].cpu().numpy(), z_vals[0, :].cpu().numpy(),
                                os.path.join(testsavedir, 'rays.png'))
                print("colmap depth:", depth_gts[index_pose]['depth'][0])
                print("Estimated depth:", depth_maps[0].cpu().numpy())
                print(depth_gts[index_pose]['coord'])
            else:
                rgbs, disps, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images,
                                             savedir=testsavedir, render_factor=args.render_factor, need_alpha=True)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(
                    testsavedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
                disps[np.isnan(disps)] = 0
                print('Depth stats', np.mean(disps), np.max(
                    disps), np.percentile(disps, 95))
                imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps / np.percentile(disps, 95)), fps=30,
                                 quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p)
                        for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        if args.debug:
            print('rays.shape:', rays.shape)
        print('done, concats')
        labels = np.expand_dims(masks, axis=-1)  # [N, H, W, 1]
        labels = np.repeat(labels[:, None], 3, axis=1)  # [N, 3, H, W, 1]
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        print('rays_rgb.shape and labels.shape:', rays_rgb.shape,
              labels.shape, images.shape, masks.shape, poses.shape)
        rays_rgb = np.concatenate([rays_rgb, labels], -1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                            for i in i_train], 0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 4])
        rays_rgb = rays_rgb.astype(np.float32)

        # for depth_inpainting rays
        rays = np.stack([get_rays_np(H, W, focal, p)
                        for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        labels = np.expand_dims(inpainted_depths, axis=-1)  # [N, H, W, 1]
        labels = np.repeat(labels[:, None], 3, axis=1)  # [N, 3, H, W, 1]
        # [N, ro+rd+rgb, H, W, 3]
        rays_inp = np.concatenate([rays, images[:, None]], 1)
        print("########################", images.shape,
              rays.shape, rays_inp.shape, labels.shape)
        rays_inp = np.concatenate([rays_inp, labels], -1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_inp = np.transpose(rays_inp, [0, 2, 3, 1, 4])
        rays_inp = np.stack([rays_inp[i]
                            for i in i_train], 0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_inp = np.reshape(rays_inp, [-1, 3, 4])
        rays_inp = rays_inp.astype(np.float32)

        rays_depth = None
        if args.colmap_depth:
            print('get depth rays')
            rays_depth_list = []
            for i in i_train:
                if not args.prepare:
                    indices = [_ for _ in range(len(depth_gts[i]['coord']))
                               if masks[i][
                                   min(int(depth_gts[i]['coord'][_]
                                       [1]), masks[i].shape[0] - 1)
                    ][
                                   min(int(depth_gts[i]['coord'][_]
                                       [0]), masks[i].shape[1] - 1)
                    ] == 0
                    ]
                    depth_gts[i]['coord'] = depth_gts[i]['coord'][indices]
                    depth_gts[i]['weight'] = depth_gts[i]['weight'][indices]
                    depth_gts[i]['depth'] = depth_gts[i]['depth'][indices]
                rays_depth = np.stack(get_rays_by_coord_np(H, W, focal, poses[i, :3, :4], depth_gts[i]['coord']),
                                      axis=0)  # 2 x N x 3
                # print(rays_depth.shape)
                rays_depth = np.transpose(rays_depth, [1, 0, 2])
                depth_value = np.repeat(
                    depth_gts[i]['depth'][:, None, None], 3, axis=2)  # N x 1 x 3
                weights = np.repeat(
                    depth_gts[i]['weight'][:, None, None], 3, axis=2)  # N x 1 x 3
                rays_depth = np.concatenate(
                    [rays_depth, depth_value, weights], axis=1)  # N x 4 x 3
                rays_depth_list.append(rays_depth)

            rays_depth = np.concatenate(rays_depth_list, axis=0)
            print('rays_weights mean:', np.mean(rays_depth[:, 3, 0]))
            print('rays_weights std:', np.std(rays_depth[:, 3, 0]))
            print('rays_weights max:', np.max(rays_depth[:, 3, 0]))
            print('rays_weights min:', np.min(rays_depth[:, 3, 0]))
            print('rays_depth.shape:', rays_depth.shape)
            rays_depth = rays_depth.astype(np.float32)
            print('shuffle depth rays')
            # np.random.shuffle(rays_depth)

        if rays_depth is not None:
            max_depth = np.max(rays_depth[:, 3, 0])
        print('done')
        i_batch = 0

        if args.train_gt or args.prepare:
            rays_rgb_clf = rays_rgb.reshape(-1, 3, 4)
        else:
            rays_rgb_clf = rays_rgb[rays_rgb[:, :, 3] == 0].reshape(-1, 3, 4)
        rays_inp = rays_inp[rays_rgb[:, :, 3] != 0].reshape(-1, 3, 4)

        if not args.prepare:
            rays_rgb = rays_rgb[rays_rgb[:, :, 3] == 1].reshape(-1, 3, 4)

        print('shuffle rays')
        # np.random.shuffle(rays_rgb)
        # np.random.shuffle(rays_rgb_clf)
        # np.random.shuffle(rays_inp)
        print(
            f'rays_rgb shape is {rays_rgb.shape} and rays_rgb_clf shape is {rays_rgb_clf.shape}')

    if args.debug:
        return

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        # rays_rgb = torch.Tensor(rays_rgb).to(device)
        # rays_depth = torch.Tensor(rays_depth).to(device) if rays_depth is not None else None
        raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), batch_size=N_rand, shuffle=True, num_workers=0,
                                       generator=torch.Generator(device=device)))
        raysINP_iter = iter(DataLoader(RayDataset(rays_inp), batch_size=N_rand, shuffle=True, num_workers=0,
                                       generator=torch.Generator(device=device)))
        raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), batch_size=N_rand, shuffle=True,
                                         num_workers=0,
                                         generator=torch.Generator(device=device))) if rays_depth is not None else None
        raysRGBCLF_iter = iter(DataLoader(RayDataset(rays_rgb_clf), batch_size=N_rand, shuffle=True, num_workers=0,
                                          generator=torch.Generator(device=device)))

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            # batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            try:
                batch_inp = next(raysINP_iter).to(device)
            except StopIteration:
                raysINP_iter = iter(DataLoader(RayDataset(rays_inp), batch_size=N_rand, shuffle=True, num_workers=0,
                                               generator=torch.Generator(device=device)))
                batch_inp = next(raysINP_iter).to(device)
            try:
                batch = next(raysRGB_iter).to(device)
            except StopIteration:
                raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), batch_size=N_rand, shuffle=True, num_workers=0,
                                               generator=torch.Generator(device=device)))
                batch = next(raysRGB_iter).to(device)
            try:
                batch_clf = next(raysRGBCLF_iter).to(device)
            except StopIteration:
                raysRGBCLF_iter = iter(
                    DataLoader(RayDataset(rays_rgb_clf), batch_size=N_rand, shuffle=True, num_workers=0,
                               generator=torch.Generator(device=device)))
                batch_clf = next(raysRGBCLF_iter).to(device)
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            batch_rays = batch_rays[:, :, :-1]
            target_s, label_s = target_s[:, :3], target_s[:, 3]

            batch_inp = torch.transpose(batch_inp, 0, 1)
            batch_inp, target_inp = batch_inp[:2], batch_inp[2]
            batch_inp = batch_inp[:, :, :-1]
            target_inp, depth_inp = target_inp[:, :3], target_inp[:, 3]

            batch_clf = torch.transpose(batch_clf, 0, 1)
            batch_rays_clf, target_clf = batch_clf[:2], batch_clf[2]
            batch_rays_clf = batch_rays_clf[:, :, :-1]
            target_clf, label_s_clf = target_clf[:, :3], target_clf[:, 3]

            if args.colmap_depth:
                # batch_depth = rays_depth[i_batch:i_batch+N_rand]
                try:
                    batch_depth = next(raysDepth_iter).to(device)
                except StopIteration:
                    raysDepth_iter = iter(
                        DataLoader(RayDataset(rays_depth), batch_size=N_rand, shuffle=True, num_workers=0,
                                   generator=torch.Generator(device=device)))
                    batch_depth = next(raysDepth_iter).to(device)
                batch_depth = torch.transpose(batch_depth, 0, 1)
                batch_rays_depth = batch_depth[:2]  # 2 x B x 3
                target_depth = batch_depth[2, :, 0]  # B
                ray_weights = batch_depth[3, :, 0]

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(
                    H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H //
                                           2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W //
                                           2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0],
                                  select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays_clf,
                                               verbose=i < 10, retraw=True,
                                               **render_kwargs_train)

        if args.object_removal:
            rgb_complete, _, acc_complete, _, extras_complete = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                                       verbose=i < 10, retraw=True,
                                                                       detach_weights=False,
                                                                       **render_kwargs_train)
        else:
            rgb_complete, _, _, _, extras_complete = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                            verbose=i < 10, retraw=True, detach_weights=True,
                                                            **render_kwargs_train)

        if not args.prepare:
            _, disp_inp, _, _, extras_inp = render(H, W, focal, chunk=args.chunk, rays=batch_inp,
                                                   verbose=i < 10, retraw=True,
                                                   **render_kwargs_train)
        if True:
            if args.colmap_depth and not args.depth_with_rgb:
                _, _, _, depth_col, extras_col = render(H, W, focal, chunk=args.chunk, rays=batch_rays_depth,
                                                        verbose=i < 10, retraw=True, depths=target_depth,
                                                        **render_kwargs_train)
            elif args.colmap_depth and args.depth_with_rgb:
                depth_col = depth

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_clf)
        psnr = mse2psnr(img_loss)

        if not args.masked_NeRF and not args.object_removal:
            img_loss += img2mse(rgb_complete, target_s)
            # img_loss += img2l1(rgb_complete, target_s)
            if 'rgb0' in extras_complete and not args.no_coarse:
                img_loss0 = img2mse(extras_complete['rgb0'], target_s)
                img_loss += img_loss0

        depth_loss = 0
        if args.depth_loss:
            # depth_loss = img2mse(depth_col, target_depth)
            if args.weighted_loss:
                if not args.normalize_depth:
                    depth_loss = torch.mean(
                        ((depth_col - target_depth) ** 2) * ray_weights)
                else:
                    depth_loss = torch.mean(
                        (((depth_col - target_depth) / max_depth) ** 2) * ray_weights)
            elif args.relative_loss:
                depth_loss = torch.mean(
                    ((depth_col - target_depth) / target_depth) ** 2)
            else:
                depth_loss = img2mse(depth_col, target_depth)
        loss = img_loss + args.depth_lambda * depth_loss

        if hasattr(render_kwargs_train['network_fn'], 'vis_triplane'):
            if args.tv_lambda>0.:
                tv_loss=render_kwargs_train['network_fn'].total_variance_loss()
                if 'network_fine' in render_kwargs_train and render_kwargs_train['network_fine'] is not None:
                    tv_loss=tv_loss+render_kwargs_train['network_fine'].total_variance_loss()
                loss=loss+tv_loss*args.tv_lambda
                print(f'add tv loss, which is {tv_loss.detach().clone().item()}')
            if args.sparse_lambda>0.:
                sparse_loss=render_kwargs_train['network_fn'].sparse_loss()
                if 'network_fine' in render_kwargs_train and render_kwargs_train['network_fine'] is not None:
                    sparse_loss=sparse_loss+render_kwargs_train['network_fine'].sparse_loss()
                loss=loss+sparse_loss*args.sparse_lambda
                print(f'add sparse loss, which is {sparse_loss.detach().clone().item()}')
            if args.edr_lambda>0.:
                edr_loss=render_kwargs_train['network_fn'].edr_loss()
                if 'network_fine' in render_kwargs_train and render_kwargs_train['network_fine'] is not None:
                    edr_loss=edr_loss+render_kwargs_train['network_fine'].edr_loss()
                loss=loss+edr_loss*args.edr_lambda
                print(f'add edr loss, which is {edr_loss.detach().clone().item()}')

        if args.object_removal:
            loss += 0.001 * acc_complete.mean()

        if 'rgb0' in extras and not args.no_coarse:
            img_loss0 = img2mse(extras['rgb0'], target_clf)
            loss = loss + img_loss0

        if not args.prepare and not args.object_removal and not args.no_geometry:
            inp_loss = nn.MSELoss()(disp_inp, depth_inp)
            if 'disp0' in extras_inp and not args.no_coarse:
                inp_loss += nn.MSELoss()(extras_inp['disp0'], depth_inp)
            if not inp_loss.isnan():
                loss += inp_loss

        if args.lpips and i > 300 and i % 1 == 0:
            lpips_loss = 0
            lpips_render_factor = args.lpips_render_factor  # 1 is reasonable
            patch_len_factor = args.patch_len_factor  # 4 is reasonable
            batch_size = args.lpips_batch_size  # 4 is reasonable

            idx = copy.deepcopy(i_train)
            np.random.shuffle(idx)
            idx = idx[:batch_size]
            random_poses = poses[idx]

            patch_len = (hwf[0] // lpips_render_factor // patch_len_factor,
                         hwf[1] // lpips_render_factor // patch_len_factor)
            transform = torchvision.transforms.Resize(
                (hwf[0] // lpips_render_factor, hwf[1] // lpips_render_factor)
            )

            rgbs, disps, (Xs, Ys) = render_path(random_poses, hwf,
                                                args.chunk,
                                                render_kwargs_test,
                                                render_factor=lpips_render_factor,
                                                rgb_require_grad=True,
                                                need_alpha=False,
                                                detach_weights=True,
                                                patch_len=patch_len,
                                                masks=masks[idx]
                                                )

            for _ in range(len(idx)):
                prediction = ((rgbs[_] - 0.5) * 2).permute(2, 0, 1)[None, ...]

                target = ((images[idx[_]] - 0.5) *
                          2).permute(2, 0, 1)[None, ...]
                target = transform(target)[
                    :, :, Xs[_]:Xs[_] + patch_len[0], Ys[_]:Ys[_] + patch_len[1]]

                # todo calculate for all rgbs at once
                lpips_loss += LPIPS(prediction, target).mean()
            loss += lpips_loss / batch_size / 100

        if i % args.i_feat == 0 and i > 0:  # calculate inpainted depths
            if args.prepare:
                idx = list(range(len(poses)))
                random_poses = poses
            else:
                idx = copy.deepcopy(i_train)
                np.random.shuffle(idx)
                idx = idx[:1]
                random_poses = poses[idx]
            with torch.no_grad():
                rgbs, disps, _ = render_path(random_poses, hwf,
                                             args.chunk,
                                             render_kwargs_test,
                                             render_factor=args.render_factor,
                                             disp_require_grad=True,
                                             need_alpha=False
                                             )

            plt.subplot(131)
            plt.imshow((rgbs[0] * 255).astype('uint8'))
            plt.gcf().set_dpi(300)

            plt.subplot(132)
            plt.imshow(inpainted_depths[idx[0]])
            # plt.colorbar()

            plt.subplot(133)
            plt.imshow(disps[0].detach().cpu().numpy())
            # plt.colorbar()

            os.makedirs('test_renders', exist_ok=True)
            plt.savefig('test_renders/{}_lpips_{}.png'.format(
                expname, str(args.lpips)
            ), format='png')
            plt.clf()

            if args.prepare:
                try:
                    os.makedirs('lama/LaMa_test_images', exist_ok=True)
                    os.mkdir('lama/LaMa_test_images/label')
                except:
                    pass
                for _ in range(len(poses)):
                    cv2.imwrite(
                        f'lama/LaMa_test_images/img{_:0>3}.png', disps[_].detach().cpu().numpy() * 255)
                    cv2.imwrite(f'lama/LaMa_test_images/label/img{_:0>3}.png',
                                masks[_][::args.render_factor, ::args.render_factor] * 255)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # if i % 100 == 0:
        #     print(new_lrate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train[
                    'network_fn'] is not None else None,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train[
                    'network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if args.i_video > 0 and i % args.i_video == 0 and i >= 0:  # todo replace i > 4000 with i > 0
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                                             render_factor=args.render_factor, need_alpha=True)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname,
                                     '{}_lpips_{}_prepare_{}_{:06d}{}{}{}_'.format(
                                         expname,
                                         str(args.lpips),
                                         str(args.prepare),
                                         i,
                                         '_masked_nerf' if args.masked_NeRF else '',
                                         '_N_train_' +
                                         str(args.N_train) if args.N_train is not None else '',
                                         '_no_geo' if args.no_geometry else ''
                                     ))
            if args.train_gt:
                moviebase = os.path.join(basedir, expname,
                                         '{}_gt_images_{:06d}_'.format(
                                             expname,
                                             i,
                                         ))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.nanmax(disps)), fps=30, quality=8)

            with torch.no_grad():
                rgbs, disps, _ = render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                                             render_factor=args.render_factor)
            print('Done, saving', rgbs.shape, disps.shape)
            imageio.mimwrite(moviebase + 'test.mp4',
                             to8b(rgbs), fps=30, quality=8)
            if hasattr(render_kwargs_train['network_fn'], 'vis_triplane'):
                render_kwargs_train['network_fn'].vis_triplane(two_exp_reso=8,vis_dir=os.path.join(basedir, expname,'triplane'),
                                                    additional_message=['iter',str(global_step),'coarse'])
                if 'network_fine' in render_kwargs_train and render_kwargs_train['network_fine'] is not None:
                    render_kwargs_train['network_fine'].vis_triplane(two_exp_reso=8,vis_dir=os.path.join(basedir, expname,'triplane'),
                                                            additional_message=['iter',str(global_step),'fine'])

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
            #     with torch.no_grad():
            #         rgbs_still, _, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
            #                                        render_factor=args.render_factor,
            #                                        )
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0 and len(i_test) > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                rgbs, disps, _ = render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk,
                                             render_kwargs_test,
                                             gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

            filenames = [os.path.join(
                testsavedir, '{:03d}.png'.format(k)) for k in range(len(i_test))]

            test_loss = img2mse(torch.Tensor(rgbs), images[i_test])
            test_psnr = mse2psnr(test_loss)

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
