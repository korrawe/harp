import torch

from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.utils import TensorProperties
from torch.functional import F

class PBRMaterials(TensorProperties):
    """
    A class for storing a batch of material properties. Currently only one
    material per batch element is supported.
    """

    def __init__(
        self,
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        normal_maps=None,
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB ambient reflectivity of the material
            diffuse_color: RGB diffuse reflectivity of the material
            specular_color: RGB specular reflectivity of the material
            shininess: The specular exponent for the material. This defines
                the focus of the specular highlight with a high value
                resulting in a concentrated highlight. Shininess values
                can range from 0-1000.
            normal_maps" TexturesUV holding normal maps
            device: Device (as str or torch.device) on which the tensors should be located

        ambient_color, diffuse_color and specular_color can be of shape
        (1, 3) or (N, 3). shininess can be of shape (1) or (N).

        The colors and shininess are broadcast against each other so need to
        have either the same batch dimension or batch dimension = 1.
        """
        super().__init__(
            device=device,
            diffuse_color=diffuse_color,
            ambient_color=ambient_color,
            specular_color=specular_color,
            shininess=shininess,
        )
        self.normal_maps = normal_maps
        self.use_normal_map = (normal_maps is not None)
        for n in ["ambient_color", "diffuse_color", "specular_color"]:
            t = getattr(self, n)
            if t.shape[-1] != 3:
                msg = "Expected %s to have shape (N, 3); got %r"
                raise ValueError(msg % (n, t.shape))
        if self.shininess.shape != torch.Size([self._N]):
            msg = "shininess should have shape (N); got %r"
            raise ValueError(msg % repr(self.shininess.shape))

    def compute_tangent(self, normals): # verts):
        # From https://github.com/FreyrS/dMaSIF/blob/master/geometry_processing.py
        """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].
              normals        ->             uv
        (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)
        This routine assumes that the 3D "normal" vectors are normalized.
        It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".
        Args:
            normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.
        Returns:
            (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
                the tangent coordinate systems `[n_i,u_i,v_i].
        """
        x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
        s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
        a = -1 / (s + z)
        b = x * y * a
        uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
        uv = uv.view(uv.shape[:-1] + (2, 3))
        return uv
    
    # def interpolate_face_average_attributes(self, tangent, fragments, verts, faces, batch_size):
    #     return

    def apply_normal_map(self, pixel_normals, fragments, faces, verts):
        """
        Apply normal map to fragments
        """
        pix_to_face = fragments.pix_to_face
        batch_size = pix_to_face.shape[0]

        # tangent = self.compute_tangent(verts[faces])
        # Smoothe the tangent map by interpolating per vertex tangent
        # tangent_map = self.interpolate_face_average_attributes(
        #     tangent, fragments, verts, faces, batch_size
        # )

        # pixel_normals = F.normalize(pixel_normals, dim=-1)
        # bitangent_map = torch.cross(pixel_normals, tangent_map, dim=-1)
        # bitangent_map = F.normalize(bitangent_map, dim=-1)
        # tangent_map = torch.cross(bitangent_map, pixel_normals, dim=-1)
        # tangent_map = F.normalize(tangent_map, dim=-1)

        tangent = self.compute_tangent(pixel_normals)
        pixel_normals_e = pixel_normals.unsqueeze(4)
        # Flip X and Y to match pytorch3D convention
        TBN = torch.cat([-tangent, pixel_normals_e], dim=4) 

        # # pixel-wise TBN matrix - flip to get correct direction
        # TBN = torch.stack(
        #     (-tangent_map, -bitangent_map, pixel_normals), dim=4
        # ) 
        nm = self.normal_maps.sample_textures(fragments)
        new_normals = F.normalize(
            torch.matmul(
                TBN.transpose(-1, -2).reshape(-1, 3, 3), nm.reshape(-1, 3, 1)
            ).reshape(pixel_normals.shape),
            dim=-1,
        )
        # import pdb; pdb.set_trace()
        # import matplotlib.pyplot as plt
        # tmp_normals = new_normals.clone()
        # tmp_normals[:, :, :, :, 1] = tmp_normals[:, :, :, :, 1] * -1.
        # tmp_normals[:, :, :, :, 2] = tmp_normals[:, :, :, :, 2] * -1.
        # plt.imshow((tmp_normals[0, :, :, 0].detach().cpu().numpy() + 1) / 2.)
        # plt.show()
        return new_normals

    def clone(self):
        other = PBRMaterials(device=self.device)
        return super().clone(other)


    # def tangent_vectors(normals):
    # From https://github.com/FreyrS/dMaSIF/blob/master/geometry_processing.py
    # """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].
    #       normals        ->             uv
    # (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)
    # This routine assumes that the 3D "normal" vectors are normalized.
    # It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".
    # Args:
    #     normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.
    # Returns:
    #     (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
    #         the tangent coordinate systems `[n_i,u_i,v_i].
    # """
    # x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    # s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    # a = -1 / (s + z)
    # b = x * y * a
    # uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    # uv = uv.view(uv.shape[:-1] + (2, 3))
    # return uv

    # From https://github.com/shunsukesaito/PIFu/blob/975331106479436356fe8fae9ca2b96a56926930/lib/renderer/mesh.py#L302
    # compute tangent and bitangent

    # def normalize_v3(arr):
    #     ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    #     lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    #     eps = 0.00000001
    #     lens[lens < eps] = eps
    #     arr[:, 0] /= lens
    #     arr[:, 1] /= lens
    #     arr[:, 2] /= lens
    #     return arr
    # def compute_tangent(vertices, faces, normals, uvs, faceuvs):    
    #     # NOTE: this could be numerically unstable around [0,0,1]
    #     # but other current solutions are pretty freaky somehow
    #     c1 = np.cross(normals, np.array([0,1,0.0]))
    #     tan = c1
    #     normalize_v3(tan)
    #     btan = np.cross(normals, tan)

    #     # NOTE: traditional version is below

    #     # pts_tris = vertices[faces]
    #     # uv_tris = uvs[faceuvs]

    #     # W = np.stack([pts_tris[::, 1] - pts_tris[::, 0], pts_tris[::, 2] - pts_tris[::, 0]],2)
    #     # UV = np.stack([uv_tris[::, 1] - uv_tris[::, 0], uv_tris[::, 2] - uv_tris[::, 0]], 1)
        
    #     # for i in range(W.shape[0]):
    #     #     W[i,::] = W[i,::].dot(np.linalg.inv(UV[i,::]))

    #     # tan = np.zeros(vertices.shape, dtype=vertices.dtype)
    #     # tan[faces[:,0]] += W[:,:,0]
    #     # tan[faces[:,1]] += W[:,:,0]
    #     # tan[faces[:,2]] += W[:,:,0]

    #     # btan = np.zeros(vertices.shape, dtype=vertices.dtype)
    #     # btan[faces[:,0]] += W[:,:,1]
    #     # btan[faces[:,1]] += W[:,:,1]    
    #     # btan[faces[:,2]] += W[:,:,1]

    #     # normalize_v3(tan)
        
    #     # ndott = np.sum(normals*tan, 1, keepdims=True)
    #     # tan = tan - ndott * normals

    #     # normalize_v3(btan)
    #     # normalize_v3(tan)

    #     # tan[np.sum(np.cross(normals, tan) * btan, 1) < 0,:] *= -1.0

    #     return tan, btan