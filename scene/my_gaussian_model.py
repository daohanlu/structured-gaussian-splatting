import os
import matplotlib.pyplot as plt
import numpy as np

import torch

from gaussian_model import GaussianModel


class MyGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.normalization_mean = None
        self.normalization_var = None

    def vectorized(self, normalize=False, detach=True) -> torch.Tensor:
        xyz = self._xyz  # N x 3
        opacities = self.get_opacity  # N x 1
        scale = self.get_scaling  # N x 3
        rotation = self.get_rotation  # N x 4

        # _features_dc: N x 1 x 3, _features_rest: N x 15 x 3
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1)
        f_rest = self._features_rest.transpose(1, 2).flatten(start_dim=1)

        shape_params = torch.concatenate((xyz, opacities, scale, rotation), dim=1)
        appearance_params = torch.concatenate((f_dc, f_rest), dim=1)
        if detach:
            shape_params = shape_params.detach()
            appearance_params = appearance_params.detach()
        return shape_params, appearance_params

    def compute_normalization(self):
        shape_params, appearance_params = self.vectorized()
        self.normalization_mean = torch.mean(appearance_params, dim=0)
        self.normalization_var = torch.std(appearance_params, dim=0)
        return self.normalization_mean, self.normalization_var

    @classmethod
    def from_vector(cls, sh_degree: int, shape_params: torch.Tensor, appearance_params: torch.Tensor):
        xyz = shape_params[:, 0:3]
        opacities = shape_params[:, 3:4]
        scale = shape_params[:, 4:7]
        rotation = shape_params[:, 7:11]

        N = len(appearance_params)
        features_dc = appearance_params[:, :3].reshape(N, 3, -1).transpose(1, 2)  # -> N x 1 x 3
        features_rest = appearance_params[:, 3:].reshape(N, 3, -1).transpose(1, 2)  # -> N x 15 x 3
        assert features_rest.shape[1] * features_rest.shape[2] == 3 * (sh_degree + 1) ** 2 - 3

        new_gm = cls(sh_degree)
        new_gm._xyz = xyz
        new_gm._scaling = new_gm.scaling_inverse_activation(scale)
        new_gm._rotation = rotation
        new_gm._opacity = new_gm.inverse_opacity_activation(opacities)
        new_gm._features_dc = features_dc
        new_gm._features_rest = features_rest
        return new_gm


def test(save_dir):
    def plot_PCA(appearance_params, mean, std=None, return_pca_projection=-1):
        appearance_params -= mean.unsqueeze(0)
        is_normalized = 'normalized' if std is not None else 'unnormalized'
        if std is not None:
            appearance_params /= std.unsqueeze(0)
        appearance_cov = appearance_params.T @ appearance_params  # (D x N) @ (N x D) -> (D x D)
        L, Q = torch.linalg.eigh(appearance_cov)  # eigen-decomposition -- equiv to SVD
        L_sum = L.sum()
        L_ratio = L / L_sum
        L_ratio, indices = L_ratio.sort(descending=True)
        Q = Q[:, indices]
        L_cumulative = torch.cumsum(L_ratio, dim=0)

        plt.plot(np.arange(len(L)) + 1, L_ratio.cpu().numpy(), marker='.')
        plt.title(f'PCA of {is_normalized} appearance (SH) parameters')
        plt.xlabel('# of PCA dimensions')
        plt.ylabel('Ratio of variance explained')
        plt.savefig(os.path.join(save_dir, f'appearance_{is_normalized}_PCA_variance_explained.svg'))
        plt.close()

        plt.plot(np.arange(len(L)) + 1, L_cumulative.cpu().numpy(), marker='.')
        plt.title(f'PCA of {is_normalized} appearance (SH) parameters')
        plt.xlabel('# of PCA dimensions')
        plt.ylabel('Ratio of variance explained (cumulative)')
        plt.savefig(os.path.join(save_dir, f'appearance_{is_normalized}_PCA_variance_explained_cumulative.svg'))
        plt.close()

        print(f'{is_normalized} appearance: std: {torch.std(appearance_params, dim=0)}')
        print(f'{is_normalized} appearance: top 3 principal components: {Q[indices[:3]]}')

        if return_pca_projection > 0:
            Q_low_rank = Q[:, :return_pca_projection]
            appearance_params_pca = appearance_params @ Q_low_rank @ Q_low_rank.T
            if std is not None:
                appearance_params_pca *= std.unsqueeze(0)
            appearance_params_pca += mean.unsqueeze(0)
            return appearance_params_pca
        else:
            return None

    gm = MyGaussianModel(3)
    gm.load_ply(
        '/home/lol/GitHub/gaussian-splatting/output/100a5dde-b-baseline/point_cloud/iteration_30000/point_cloud.ply')
    shape_params, appearance_params = gm.vectorized()
    shape_params_prime, appearance_params_prime = MyGaussianModel.from_vector(3, shape_params,
                                                                              appearance_params).vectorized()
    print(shape_params.shape, shape_params_prime.shape)
    assert torch.allclose(shape_params, shape_params_prime) and torch.allclose(appearance_params,
                                                                               appearance_params_prime)
    mean, std = gm.compute_normalization()
    os.makedirs(save_dir, exist_ok=True)

    # ----------plot opacities histogram----------
    opacities = gm.get_opacity.detach().cpu().numpy()
    plt.hist(opacities, bins=20)
    plt.xlabel('Opacity')
    plt.ylabel('Number of components')
    plt.savefig(os.path.join(save_dir, 'opacities_hist.svg'))
    plt.close()

    plt.hist(opacities, bins=20, cumulative=True)
    plt.xlabel('Opacity')
    plt.ylabel('Number of components (cumulative)')
    plt.savefig(os.path.join(save_dir, 'opacities_hist_cumulative.svg'))
    plt.close()

    # ----------plot PCA of appearance params----------
    appearance_params_PCA_unnorm = plot_PCA(appearance_params, mean, std=None, return_pca_projection=3)
    appearance_params_PCA_norm = plot_PCA(appearance_params, mean, std=std, return_pca_projection=3)
    gm_PCA_unnorm = MyGaussianModel.from_vector(3, shape_params, appearance_params_PCA_unnorm)
    gm_PCA_norm = MyGaussianModel.from_vector(3, shape_params, appearance_params_PCA_norm)
    gm_PCA_unnorm.save_ply(save_dir + "_ply_unnorm/point_cloud.ply")
    gm_PCA_norm.save_ply(save_dir + "_ply_norm/point_cloud.ply")

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    test('my_tests')
