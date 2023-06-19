import numpy as np
import torch
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy.linalg import orthogonal_procrustes

lpips_fn = lpips.LPIPS(net='alex').to('cuda')
ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)

def image_eval(images_for_eval):
    stat = {
        "Silhouette IoU": [],
        "L1": [],
        "LPIPS": [],
        "MS_SSIM": []
    }    
    for (k, v) in images_for_eval.items():
        images_for_eval[k] = torch.vstack(v)
    # Because of the mismatch around the area under the wrist, 
    # we will compare only the area that overlapped with the predicted hand
    stat["Silhouette IoU"] = sil_iou(images_for_eval["ref_mask"], images_for_eval["pred_mask"])
    stat["L1"] = l1_diff(images_for_eval["ref_image"], images_for_eval["ref_mask"], images_for_eval["pred_image"], images_for_eval["pred_mask"])
    stat["LPIPS"] = lpips_diff(images_for_eval["ref_image"], images_for_eval["pred_image"])
    stat["MS_SSIM"] = ms_ssim_diff(images_for_eval["ref_image"], images_for_eval["pred_image"])
    return stat


def fill_bg(img, mask):
    img = img * mask.unsqueeze(-1)
    img = img + (mask.unsqueeze(-1) - 1) * -1.
    return img


def l1_diff(ref_image, ref_mask, pred_image, pred_mask):
    # range [0, 1]. Not [0, 255]
    # diff = torch.abs(ref_image * pred_mask.unsqueeze(-1) - pred_image * pred_mask.unsqueeze(-1))
    diff = torch.abs(ref_image - pred_image)
    return torch.mean(diff).detach().numpy()


def sil_iou(ref_masks, pred_masks):
    ref_bools = (ref_masks >= 0.5)
    pred_bools = (pred_masks >= 0.5)
    union = torch.logical_or(ref_bools, pred_bools)
    intersect = torch.logical_and(ref_bools, pred_bools)
    union_sum = union.sum([1,2])
    intersect_sum = intersect.sum([1,2])
    iou = intersect_sum / union_sum
    return torch.mean(iou).detach().numpy()

def lpips_diff(ref_images, pred_images):
    diff = lpips_fn(ref_images.permute(0, 3, 1, 2).to('cuda'), pred_images.permute(0, 3, 1, 2).to('cuda'))
    return torch.mean(diff).detach().cpu().numpy()


def ms_ssim_diff(ref_images, pred_images):
    # https://github.com/VainF/pytorch-msssim
    diff = ms_ssim_module(ref_images.permute(0, 3, 1, 2), pred_images.permute(0, 3, 1, 2))
    
    return torch.mean(diff).numpy()


def load_gt_vert(fid, gt_mesh_dir, dataset="synthetic", start_from_one=False, idx_offset=0):
    if dataset == "synthetic":
        if start_from_one:
            num = idx_offset + fid[0] + 1
        else:
            num = idx_offset + fid[0]
        mano_verts = np.loadtxt("{:s}/{:d}_manov.xyz".format(gt_mesh_dir, num))
    return mano_verts / 1000.0


class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds


def eval_procrustes(images_dataset, params, input_params, mano_layer, global_pose=False, average_pose=False, device="cuda"):
    images_dataloader = DataLoader(images_dataset, batch_size=1, shuffle=False, num_workers=20)
    joint_err_list = []
    for (fid, y_true, y_sil_true) in images_dataloader:
        # eval joint error after optimization
        target_joint = input_params['gt_joints'][fid].numpy()
        valid_joint_gt = input_params['gt_joint_valid'][fid]

        batch_size = 1
        if global_pose:
            # Use the first pose parameter if global_pose is true
            pose_batch = params['pose'][0].repeat(batch_size, 1)
            # Use average of all pose params in the batch. For first step evaluation
        elif average_pose:
            # import pdb; pdb.set_trace()
            pose_batch = params['pose'].mean(dim=0).repeat(batch_size, 1)
        else:
            pose_batch = params['pose'][fid]

        # batch_size = fid.shape[0]  # params['pose'].shape[0]
        hand_verts, hand_joints = mano_layer(torch.cat((params['rot'][fid], pose_batch), 1).to(device),
                params['shape'].unsqueeze(0).to(device),
                params['trans'][fid].to(device))

        root_aligned_target = target_joint[:,:] - target_joint[:, None, 0]
        # plot_skeleton(root_aligned_target[0], joint_order='mano') # in (mm)
        # plot_skeleton(targets['joint_coord'].detach().cpu().numpy()[0, :21], joint_order='interhand') # in (mm)
        
        pred_clone = hand_joints.detach().cpu().numpy() # in (mm)
        # Get valid joints - only do Procrustes aligment with the valid joints
        # remove invalid joint from both GT and prediction. This remove the first dimention from [1, 21, 3] to [21, 3]
        root_aligned_target = root_aligned_target[valid_joint_gt == 1]
        root_aligned_pred = root_aligned_pred[valid_joint_gt == 1]
        xyz_pred_aligned = align_w_scale(root_aligned_target, root_aligned_pred)
        err = root_aligned_target - xyz_pred_aligned

        mean_joint_err = np.linalg.norm(err, axis=1).mean()
        print("mean joint err: %.3f mm" % mean_joint_err)
        if len(root_aligned_target) > 0:
            joint_err_list.append(mean_joint_err)
        else:
            print("invalid ground truth")
        
    print("Mean Procrustes-aligned joint error of %d samples: %.3f mm" % (len(joint_err_list), np.mean(joint_err_list)))


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(mtx, trafo):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, s, s1, t1 = trafo
    return np.dot(mtx_t, R.T) * s * s1 + t1 + t2