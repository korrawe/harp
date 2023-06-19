import torch


def kps_loss(gt_kps, pred_kps, use_arm=False, device='cuda'):

    # import pdb; pdb.set_trace()
    if use_arm:
        pred_kps = pred_kps[:, :21, :]
    # Compute root-aligned loss to not change the pose too much
    # GT in (mm)
    gt_kps = (gt_kps - gt_kps[:, 0, None, :]).to(device)
    # gt_kps = gt_kps.to(device)
    # pred in (m)
    pred_kps = (pred_kps - pred_kps[:, 0, None, :]) * 1000.0
    # pred_kps = pred_kps * 1000.0

    # scale the loss by divided by 100.0
    return torch.mean((torch.norm(gt_kps - pred_kps, dim=2) / 100.0) ** 2.0)