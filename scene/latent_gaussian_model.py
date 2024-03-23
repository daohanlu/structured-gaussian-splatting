from typing import Optional

import numpy as np
import torch

from .autodecoder import Decoder, get_embedder
from .gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from simple_knn._C import distCUDA2


def _standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_normalize_then_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two un-normalized quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = _quaternion_raw_multiply(torch.nn.functional.normalize(a, dim=-1), torch.nn.functional.normalize(b, dim=-1))
    return _standardize_quaternion(ab)


class LatentGaussianModel(GaussianModel, torch.nn.Module):
    def __init__(self, sh_degree: int, structure_means_init, latent_size: int = 512,
                 hidden_size: int = 2048, gaussians_per_structure: int = 16,
                 use_positional_embedding=False, positional_embedding_multires=None):
        GaussianModel.__init__(self, sh_degree)
        torch.nn.Module.__init__(self)
        assert len(structure_means_init.shape) == 2 and structure_means_init.shape[1] == 3, \
            'structure_means_init must be N by 3!'
        self.sh_degree = sh_degree
        self.gaussian_parameters_size = 11 + 3 * (sh_degree + 1) ** 2  # mean, opacity, scale, quaternions | color
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_structures = len(structure_means_init)
        self.gaussians_per_structure = gaussians_per_structure
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding and positional_embedding_multires is None:
            positional_embedding_multires = 10
        self.positional_embedding_multires = positional_embedding_multires

        device = structure_means_init.device
        self.structure_means = torch.nn.Parameter(structure_means_init)
        self.structure_opacities = torch.nn.Parameter(
            self.inverse_opacity_activation(torch.ones((self.num_structures, 1), device=device) * 0.1))
        self.structure_scales = torch.nn.Parameter(torch.ones((self.num_structures, 3), device=device))
        self.structure_rotations = torch.nn.Parameter(torch.randn((self.num_structures, 4), device=device))
        self.structure_latents = torch.nn.Parameter(torch.randn((structure_means_init.shape[0], self.latent_size),
                                                                device=device))
        self.normalize_quaternion = torch.nn.functional.normalize
        if self.use_positional_embedding:
            pos_embed_fn, pos_embed_size = get_embedder()
            self.decoder = Decoder(latent_size, [hidden_size] * 2,
                                   self.gaussian_parameters_size * self.gaussians_per_structure,
                                   norm_layers=[],
                                   pos_emb_size=pos_embed_size, pos_embed_fn=pos_embed_fn).to(device)
        else:
            self.decoder = Decoder(latent_size, [hidden_size] * 2,
                                   self.gaussian_parameters_size * self.gaussians_per_structure,
                                   norm_layers=[]).to(device)
        self.register_buffers()

        self.freeze_structure_means = False
        self.freeze_structure_scales = False
        self.freeze_structure_rotations = False
        self.freeze_structure_opacities = False

    def register_buffers(self):
        # register buffers so they're included in self.state_dict() when saving
        max_sh_degree = torch.tensor(self.max_sh_degree, dtype=torch.int)
        del self.max_sh_degree
        self.register_buffer('max_sh_degree', max_sh_degree)

        active_sh_degree = torch.tensor(self.active_sh_degree, dtype=torch.int)
        del self.active_sh_degree
        self.register_buffer('active_sh_degree', active_sh_degree)

        max_radii2D = self.max_radii2D
        del self.max_radii2D
        self.register_buffer('max_radii2D', max_radii2D)

    def update_from_vector(self, gaussian_parameters):
        """
        Updates self's pre-activation parameters from batch of vectorized Gaussian parameters
        :param gaussian_parameters: Batch of pre-activation vectorized Gaussian parameters.
            (num_gaussians, 11 + color_params)
        """
        N = gaussian_parameters.shape[0]

        xyz = gaussian_parameters[:, 0:3]
        opacities = gaussian_parameters[:, 3:4]
        scale = gaussian_parameters[:, 4:7]
        rotation = gaussian_parameters[:, 7:11]

        features_dc = gaussian_parameters[:, 11:11 + 3].reshape(N, 3, -1).transpose(1, 2)  # -> N x 1 x 3
        features_rest = gaussian_parameters[:, 11 + 3:].reshape(N, 3, -1).transpose(1, 2)  # -> N x 15 x 3
        # assert features_rest.shape[1] * features_rest.shape[2] == 3 * (sh_degree + 1) ** 2 - 3

        self._xyz = xyz
        self._scaling = scale
        self._rotation = rotation
        self._opacity = opacities
        self._features_dc = features_dc
        self._features_rest = features_rest

    def set_freeze_structures_params(self, frozen: bool):
        self.freeze_structure_means = frozen
        self.freeze_structure_scales = frozen
        self.freeze_structure_rotations = frozen
        self.freeze_structure_opacities = frozen

    def forward(self, latent_noise: Optional[torch.Tensor] = None) -> torch.Tensor:

        def flatten_structures(params, B):
            return params.reshape(B * self.gaussians_per_structure, -1)

        # decode gaussian parameters. self.gaussians_per_structure for each latent.
        structure_latents = self.structure_latents
        if latent_noise is not None:
            structure_latents = structure_latents + latent_noise.detach()
        if self.use_positional_embedding:
            gaussian_parameters = self.decoder(structure_latents, xyz=self.structure_means)
        else:
            gaussian_parameters = self.decoder(structure_latents)

        # --- compose each cluster's shared mean, scale, rotation, and opacity with its constituents ---
        B, D = gaussian_parameters.shape
        assert B == self.num_structures
        # reshape into (num clusters, num Gaussians per cluster, individual gaussian parameters)
        gaussian_parameters = gaussian_parameters.reshape(B, self.gaussians_per_structure,
                                                          self.gaussian_parameters_size)

        structure_means = self.structure_means if not self.freeze_structure_means else self.structure_means.detach()
        structure_opacities = self.structure_opacities if not self.freeze_structure_opacities else self.structure_opacities.detach()
        structure_scales = self.structure_scales if not self.freeze_structure_opacities else self.structure_scales.detach()
        structure_rotations = self.structure_rotations if not self.freeze_structure_rotations else self.structure_rotations.detach()
        # add each cluster's mean to constituents
        self._xyz = flatten_structures(gaussian_parameters[:, :, 0:3] + structure_means.unsqueeze(1), B)
        self._opacity = flatten_structures(gaussian_parameters[:, :, 3:4] + structure_opacities.unsqueeze(1), B)
        self._scaling = flatten_structures(gaussian_parameters[:, :, 4:7] + structure_scales.unsqueeze(1), B)
        self._rotation = flatten_structures(quaternion_normalize_then_multiply(structure_rotations.unsqueeze(1),
                                                                               gaussian_parameters[:, :, 7:11]), B)
        self._features_dc = flatten_structures(gaussian_parameters[:, :, 11:11 + 3], B)
        self._features_dc = self._features_dc.reshape(self._features_dc.shape[0], 1, 3)
        self._features_rest = flatten_structures(gaussian_parameters[:, :, 11 + 3:11 + 3 + 45], B)
        self._features_rest = self._features_rest.reshape(self._features_rest.shape[0], 15, 3)
        # gaussian_parameters[:, :, 0:3] += structure_means.unsqueeze(1)
        # # add each cluster's pre-sigmoid-activation opacity with constituents. (equiv. to a pre-activation bias)
        # gaussian_parameters[:, :, 3:4] += structure_opacities.unsqueeze(1)
        # # mul each cluster's scale with constituents. we use addition here since this param is in log-space
        # gaussian_parameters[:, :, 4:7] += structure_scales.unsqueeze(1)
        # # multiply each cluster's quaternions with constituents
        # gaussian_parameters[:, :, 7:11] = quaternion_normalize_then_multiply(structure_rotations.unsqueeze(1),
        #                                                                      gaussian_parameters[:, :, 7:11])

        # reshape into (num clusters * num Gaussians per cluster, individual gaussian parameters)
        gaussian_parameters = flatten_structures(gaussian_parameters, B)
        # import pdb;
        # pdb.set_trace()
        return gaussian_parameters

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        # super().create_from_pcd(pcd, spatial_lr_scale)

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0
        print("Number of structures at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self.structure_means = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.structure_scales = torch.nn.Parameter(scales.requires_grad_(True))
        self.structure_rotations = torch.nn.Parameter(rots.requires_grad_(True))
        self.structure_opacities = torch.nn.Parameter(opacities.requires_grad_(True))
        self.structure_latents = torch.nn.Parameter(torch.randn((self.structure_means.shape[0], self.latent_size),
                                                               device="cuda"))
        self.num_structures = self.structure_means.shape[0]
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, eps=1e-15)

    def save_ply(self, path):
        with torch.no_grad():
            self.forward()
        # import pdb
        # pdb.set_trace()
        super().save_ply(path)


def test():
    device = torch.device('cuda')
    num_components = 16 * 100000
    torch.manual_seed(42)
    lgm = LatentGaussianModel(3, torch.randn((num_components // 16, 3), device=device))
    random_target = torch.randn((num_components // 16 * 16, lgm.gaussian_parameters_size), device=device) * 10
    optimizer = torch.optim.Adam(lgm.parameters(), lr=0.01)
    for name, param in lgm.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    for i in range(1000):
        optimizer.zero_grad()
        gaussian_params = lgm.forward()
        # lgm.update_from_vector(gaussian_params)
        loss = torch.mean((random_target - gaussian_params) ** 2)

        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'Latent Gaussian Model sanity test: iter {i}, loss {loss.item()}')
    lgm.save_ply('/tmp/test.ply')


if __name__ == '__main__':
    test()
