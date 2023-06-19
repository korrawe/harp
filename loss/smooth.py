import torch


def smooth_pose(params, fid, mano_layer):
    """
    Compute as-rigid-as-possible loss term. Trying to maintain mess edge length
    compared to the template.

    Args:
        params: Meshes object with a batch of meshes.
        fid: current batch.
        mano_layer: mano_layer.

    Returns:
        loss: Average loss across the batch.
    """

    N = len(fid)
    loss = 0.0
    # import pdb; pdb.set_trace()
    # loss = ((v0 - v1).norm(dim=1, p=2) * 1000.0 - (ref_v0 - ref_v1).norm(dim=1, p=2) * 1000.0) ** 2.0
    # loss = loss * weights

    return loss.sum() / N


func_l2 = lambda x: torch.sum(x**2)

class LossSmoothPoses:
    def __init__(self, nFrames, use_arm=False) -> None:
        # self.nViews = nViews
        self.nFrames = nFrames # 150
        self.norm = 'l2'
        self.use_arm = use_arm
    
    def smooth_pose(self, params, fid, mano_layer, device='cuda'):
        # This is in (mm)
        N = len(fid)
        fid_r = torch.where(fid % self.nFrames == self.nFrames-1, fid, fid + 1)
        fid_l = torch.where(fid % self.nFrames == 0, fid, fid - 1)
        
        if self.use_arm:
            # import pdb; pdb.set_trace()
            hand_verts, hand_joints = mano_layer(betas=params['shape'].repeat([N, 1]).to(device), 
                global_orient=params['rot'][fid].to(device), transl=params['trans'][fid].to(device), right_hand_pose=params['pose'][fid].to(device), return_type='mano_w_arm')
            hand_verts_r, hand_joints_r = mano_layer(betas=params['shape'].repeat([N, 1]).to(device), 
                global_orient=params['rot'][fid_r].to(device), transl=params['trans'][fid_r].to(device), right_hand_pose=params['pose'][fid_r].to(device), return_type='mano_w_arm')
            hand_verts_l, hand_joints_l = mano_layer(betas=params['shape'].repeat([N, 1]).to(device), 
                global_orient=params['rot'][fid_l].to(device), transl=params['trans'][fid_l].to(device), right_hand_pose=params['pose'][fid_l].to(device), return_type='mano_w_arm')
        else:
            hand_verts, hand_joints = mano_layer(torch.cat((params['rot'][fid], params['pose'][fid]), 1).to(device),
                    params['shape'].repeat([N, 1]).to(device), params['trans'][fid].to(device)) # hand_verts = hand_verts / 1000.0
            
            hand_verts_r, hand_joints_r = mano_layer(torch.cat((params['rot'][fid_r], params['pose'][fid_r]), 1).to(device),
                    params['shape'].repeat([N, 1]).to(device), params['trans'][fid_r].to(device))
            
            hand_verts_l, hand_joints_l = mano_layer(torch.cat((params['rot'][fid_l], params['pose'][fid_l]), 1).to(device),
                    params['shape'].repeat([N, 1]).to(device), params['trans'][fid_l].to(device))

        # root-align
        # import pdb; pdb.set_trace()
        hand_joints_l = hand_joints_l - hand_joints_l[:, 0, :].unsqueeze(1)
        hand_joints_r = hand_joints_r - hand_joints_r[:, 0, :].unsqueeze(1)
        hand_joints = hand_joints - hand_joints[:, 0, :].unsqueeze(1)

        # need to put it in the camera space

        joint_interp = (hand_joints_l + hand_joints + hand_joints_r) / 3.
        joint_interp = joint_interp.detach()

        loss = func_l2(hand_joints - joint_interp)
        return loss / N


class LossSmoothRoots:
    def __init__(self, nFrames, focal_length, res, use_arm=False) -> None:
        # self.nViews = nViews
        self.nFrames = nFrames # 150
        self.norm = 'l2'
        self.focal_length = focal_length
        self.res = res
        self.use_arm = use_arm
    
    def smooth_root(self, params, fid, mano_layer, device='cuda'):
        # maybe trans in in (m) but cam is in (mm)?
        N = len(fid)
        fid_l = torch.where(fid % self.nFrames == 0, fid, fid - 1)
        fid_r = torch.where(fid % self.nFrames == self.nFrames-1, fid, fid + 1)

        if self.use_arm:
            hand_verts, hand_joints = mano_layer(betas=params['shape'].repeat([N, 1]).to(device), 
                global_orient=params['rot'][fid].to(device), transl=params['trans'][fid].to(device), right_hand_pose=params['pose'][fid].to(device), return_type='mano_w_arm')
            hand_verts_r, hand_joints_r = mano_layer(betas=params['shape'].repeat([N, 1]).to(device), 
                global_orient=params['rot'][fid_r].to(device), transl=params['trans'][fid_r].to(device), right_hand_pose=params['pose'][fid_r].to(device), return_type='mano_w_arm')
            hand_verts_l, hand_joints_l = mano_layer(betas=params['shape'].repeat([N, 1]).to(device), 
                global_orient=params['rot'][fid_l].to(device), transl=params['trans'][fid_l].to(device), right_hand_pose=params['pose'][fid_l].to(device), return_type='mano_w_arm')
        else:
            hand_verts, hand_joints = mano_layer(torch.cat((params['rot'][fid], params['pose'][fid]), 1).to(device),
                    params['shape'].repeat([N, 1]).to(device), params['trans'][fid].to(device)) # hand_verts = hand_verts / 1000.0
            
            hand_verts_r, hand_joints_r = mano_layer(torch.cat((params['rot'][fid_r], params['pose'][fid_r]), 1).to(device),
                    params['shape'].repeat([N, 1]).to(device), params['trans'][fid_r].to(device))
            
            hand_verts_l, hand_joints_l = mano_layer(torch.cat((params['rot'][fid_l], params['pose'][fid_l]), 1).to(device),
                    params['shape'].repeat([N, 1]).to(device), params['trans'][fid_l].to(device))
        
        #
        cam = params['cam'][fid].to(device)
        cam_l = params['cam'][fid_l].to(device)
        cam_r = params['cam'][fid_r].to(device)
        
        #
        camera_t = torch.stack([cam[:, 1], cam[:, 2], 2 * self.focal_length/(self.res * cam[:, 0] +1e-9)], dim=1)
        camera_t_l = torch.stack([cam_l[:, 1], cam_l[:, 2], 2 * self.focal_length/(self.res * cam_l[:, 0] +1e-9)], dim=1)
        camera_t_r = torch.stack([cam_r[:, 1], cam_r[:, 2], 2 * self.focal_length/(self.res * cam_r[:, 0] +1e-9)], dim=1)

        cam_rel_root = camera_t + hand_joints[:, 0, :].detach() / 1000.0 # params['trans'][fid]
        cam_rel_root_l = camera_t_l + hand_joints_l[:, 0, :].detach() / 1000.0 # params['trans'][fid_l]
        cam_rel_root_r = camera_t_r +  hand_joints_r[:, 0, :].detach() / 1000.0 # params['trans'][fid_r]

        # camera_t is applied to the object, which means the sum(cam_T + hand_joint) need to be constant
        # import pdb; pdb.set_trace()

        root_interp = (cam_rel_root_l + cam_rel_root + cam_rel_root_r) / 3.
        root_interp = root_interp.detach()

        loss = func_l2(cam_rel_root - root_interp)
        # loss = func_l2(cam_rel_root - torch.tensor([[0.0, 0.05, 1.4]]).to(device))
        # loss = func_l2(camera_t - torch.tensor([[0.0, 0.05, 1.2]]).to(device))
        
        return loss / N
