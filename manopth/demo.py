from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from manopth.manolayer import ManoLayer


def generate_random_hand(batch_size=1, ncomps=6, mano_root='mano/models'):
    nfull_comps = ncomps + 3  # Add global orientation dims to PCA
    random_pcapose = torch.rand(batch_size, nfull_comps)
    mano_layer = ManoLayer(mano_root=mano_root)
    verts, joints = mano_layer(random_pcapose)
    return {'verts': verts, 'joints': joints, 'faces': mano_layer.th_faces}


def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, show=True,
                 face_color=(141 / 255, 184 / 255, 226 / 255)):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'], hand_info['joints']
    # rest_joints = hand_info['rest_joints'] #
    # verts_joints_assoc = hand_info['verts_assoc']

    visualize_bone = 13
    # rest_verts = hand_info['rest_verts'] # 

    # import pdb; pdb.set_trace()
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        # face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    # print("Joints", joints)
    # print("joint shape", joints.shape)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # ax.scatter(joints[:16, 0], joints[:16, 1], joints[:16, 2], color='r')
    # ax.scatter(rest_joints[:4, 0], rest_joints[:4, 1], rest_joints[:4, 2], color='g')
    # ax.scatter(rest_joints[4:, 0], rest_joints[4:, 1], rest_joints[4:, 2], color='b')
    

    # visualize only some part
    # seleceted = verts_joints_assoc[:-1] == visualize_bone
    # ax.scatter(verts[seleceted, 0], verts[seleceted, 1], verts[seleceted, 2], color='black', alpha=0.5)

    # cam_equal_aspect_3d(ax, verts.numpy())
    cam_equal_aspect_3d(ax, verts)
    # cam_equal_aspect_3d(ax, rest_joints.numpy())
    if show:
        plt.show()


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
