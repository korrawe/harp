import torch
import numpy as np
import pickle
import os

def save_result(params, base_output_dir, test=False):
    save_params = {}
    for (k, v) in params.items():
        if isinstance(v, torch.Tensor):
            save_params[k] = v.detach().cpu().numpy()
        else:
            save_params[k] = v
    
    test_suffix = "_test" if test else ""
    with open(os.path.join(base_output_dir, "saved_params" + test_suffix + ".pkl"), 'wb') as outfile:
        pickle.dump(save_params, outfile)

def load_result(base_output_dir, device='cuda', test=False):
    test_suffix = "_test" if test else ""
    with open(os.path.join(base_output_dir, "saved_params" + test_suffix + ".pkl"), 'rb') as infile:
        params = pickle.load(infile)
    for k in params:
        if not params[k] is None:
            params[k] = torch.from_numpy(params[k])
    params = set_require_grad(params)
    return params

def set_require_grad(params, device='cuda'):
    grad_param_list = ['trans', 'pose', 'wrist_pose', 'rot', 'shape', 'verts_disps', 'verts_rgb', 
        'texture', 'light_positions', 'normal_map', 'nimble_tex']
    for k in grad_param_list:
        if k in params:
            if k == 'verts_disps':
                params[k] = torch.nn.Parameter(params[k].to(device), requires_grad=True)    
            else:
                params[k] = torch.nn.Parameter(params[k], requires_grad=True)
    return params


def save_model(model, base_output_dir, epoch, color_optimizer):
    save_path = os.path.join(base_output_dir, "saved_model.pt")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': color_optimizer.state_dict(),
            # 'loss': loss,
            }, save_path)


def load_model(model, base_output_dir):
    save_path = os.path.join(base_output_dir, "saved_model.pt")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model