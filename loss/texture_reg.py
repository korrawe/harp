import torch
l1_loss = torch.nn.L1Loss()


def albedo_reg(uv_texture, std=2.0, uv_mask=None):
    '''
    sd in pixel
    '''
    
    # l1_loss = torch.nn.L1Loss()
    # import pdb; pdb.set_trace()
    # text_gray = torch.mean(uv_texture, dim=3)
    uv_texture = uv_texture.squeeze(0)
    texture_shape = uv_texture.shape
    dist = torch.normal(mean=0, std=std, size=(*texture_shape[:2], 2)).to(torch.int)

    grid_x, grid_y = torch.meshgrid(torch.arange(texture_shape[0]), torch.arange(texture_shape[1]), indexing='ij')
    tar_x = grid_x + dist[:, :, 0]
    tar_y = grid_y + dist[:, :, 1]
    
    tar_x = torch.clamp(tar_x, 0, texture_shape[0] - 1)
    tar_y = torch.clamp(tar_y, 0, texture_shape[1] - 1)
    tar_img = uv_texture[tar_x, tar_y]
    # sum of abs()
    diff = torch.norm(uv_texture - tar_img, p=1, dim=2) / 3.0

    if uv_mask is not None:
        diff *= uv_mask

    return diff.mean()


def normal_reg(normal_map, std=2.0, uv_mask=None):
    '''
    sd in pixel
    '''
    return 0.2 * close_to_z_reg(normal_map) + smooth_texture_reg(normal_map, std=std, uv_mask=uv_mask)


def close_to_z_reg(normal_map):
    # l1_loss = torch.nn.L1Loss()
    # import pdb; pdb.set_trace()
    diff = torch.norm(normal_map - torch.tensor([0.0, 0.0, 1.0], device=normal_map.device), p=2, dim=2) / 3.0
    # diff = l1_loss(normal_map, torch.tensor([0.0, 0.0, 1.0], device=normal_map.device))
    return diff.mean()


def smooth_texture_reg(texture_map, std=2.0, uv_mask=None):
    texture_map = texture_map.squeeze(0)
    texture_shape = texture_map.shape
    dist = torch.normal(mean=0, std=std, size=(*texture_shape[:2], 2)).to(torch.int)

    grid_x, grid_y = torch.meshgrid(torch.arange(texture_shape[0]), torch.arange(texture_shape[1]), indexing='ij')
    tar_x = grid_x + dist[:, :, 0]
    tar_y = grid_y + dist[:, :, 1]
    
    tar_x = torch.clamp(tar_x, 0, texture_shape[0] - 1)
    tar_y = torch.clamp(tar_y, 0, texture_shape[1] - 1)

    tar_img = texture_map[tar_x, tar_y]
    # sum of abs()
    diff = torch.norm(texture_map - tar_img, p=1, dim=2) / 3.0
    if uv_mask is not None:
        diff *= uv_mask

    return diff.mean()