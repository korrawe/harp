import numpy as np


def fast_reprojection(uv_A, w_A, z_A, cw2_A, w_B, z_B, c2w_B, c2w_B_inv, K, K_inv):
    pt_w_A = w_A[uv_A[1], uv_A[0]]
    pt_z_A = z_A[uv_A[1], uv_A[0]]
    pt_z_A = pt_z_A[pt_w_A == np.max(pt_w_A)]
    if len(pt_z_A) > 1:
        return None
    pt_z_A = pt_z_A[None, :, None]  # (1, 1, 1)
    xyz_A_camera = (np.stack([uv_A[0], uv_A[1], 1])[None, None, :] * pt_z_A) @ K_inv.T
    # xyz_A_camera has shape (1, 1, 3).
    xyz_A_world = np.concatenate([xyz_A_camera, np.ones([1, 1, 1])], axis=2) @ cw2_A.T
    # xyz_A_world has shape (1, 1, 4).
    uv_B = (xyz_A_world @ c2w_B_inv.T)[:, :, :3] @ K.T
    uv_B = (uv_B[:, :, :2] / uv_B[:, :, 2:]).astype(np.int32)[0][0]

    H, W, _ = w_B.shape
    if not 0 <= uv_B[1] < H or not 0 <= uv_B[0] < W:
        return None
    pt_w_B = w_B[uv_B[1], uv_B[0]]
    pt_z_B = z_B[uv_B[1], uv_B[0]]
    pt_z_B = pt_z_B[pt_w_B == np.max(pt_w_B)]
    if len(pt_z_B) > 1:
        return None
    pt_z_B = pt_z_B[None, :, None]
    xyz_B_camera = (np.stack([uv_B[0], uv_B[1], 1])[None, None, :] * pt_z_B) @ K_inv.T
    xyz_B_world = np.concatenate([xyz_B_camera, np.ones([1, 1, 1])], axis=2) @ c2w_B.T

    error = np.sum((xyz_A_world - xyz_B_world) ** 2)
    if error < 1e-3:
        return uv_B
    else:
        return None


def fast_correspondence(src_z, src_weight, src_c2w, tgt_z, tgt_weight, tgt_c2w, K, K_inv, points):
    tgt_c2w_inv = np.linalg.inv(tgt_c2w)

    uvs = []
    for point in points:
        u_A, v_A = point[0], point[1]
        uvs_B = fast_reprojection([u_A, v_A], src_weight, src_z, src_c2w, tgt_weight, tgt_z, tgt_c2w, tgt_c2w_inv, K,
                                  K_inv)
        if uvs_B is not None:
            uvs.append(uvs_B)

    if len(uvs) == 0:
        return None
    return np.vstack(uvs)
