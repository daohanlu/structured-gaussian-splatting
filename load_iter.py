#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import copy
import math
import os
import pdb
from collections import defaultdict
from typing import List, Union, Tuple, Dict, Literal
import pprint

import numpy as np
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import matplotlib.pyplot as plt
import matplotlib

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_full_batch_gradients(gaussians: GaussianModel, viewpoint_stack, background, pipe, opt, grad_keys):
    gaussians.optimizer.zero_grad(set_to_none=True)
    for i in range(len(viewpoint_stack)):
        viewpoint_cam = viewpoint_stack[i]
        bg = background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
    ret = {k: getattr(gaussians, k).grad for k in grad_keys}
    gaussians.optimizer.zero_grad(set_to_none=True)
    return ret


def get_grad_stats(gaussians: GaussianModel, viewpoint_stack, background, pipe, opt, grad_keys,
                   sampling: str = "random",
                   accum_steps: int = 1, determininstic_index: int = None,
                   monitor_params: Tuple[str, List[int]] = None) -> Tuple[
    torch.Tensor, Dict[str, np.array], Dict[str, np.array], Dict[str, np.array], Dict[str, np.array]]:
    assert sampling in ["random", "random_wo_replacement", "nearby", ""]
    cam_indices = None

    if determininstic_index is not None:
        cam_indices = list(range(determininstic_index, determininstic_index + accum_steps))
    else:
        if sampling == "random":
            cam_indices = [randint(0, len(viewpoint_stack) - 1) for _ in range(accum_steps)]
        elif sampling == 'nearby':
            index = randint(0, len(viewpoint_stack) - accum_steps)
            cam_indices = list(range(index, index + accum_steps))
        elif sampling == "random_wo_replacement":
            assert accum_steps == len(viewpoint_stack)
            cam_indices = np.random.permutation(np.arange(len(viewpoint_stack)))
    grad_running_sum = {k: 0 for k in grad_keys}
    sparsities = {k: [] for k in grad_keys}
    variances = {k: [] for k in grad_keys}
    cosines = {k: [] for k in grad_keys}
    SNRs = {k: [] for k in grad_keys}
    target_grad = get_full_batch_gradients(gaussians, viewpoint_stack, background, pipe, opt, grad_keys)

    gaussians.optimizer.zero_grad(set_to_none=True)
    exp_avg: Dict[str, torch.Tensor] = {}
    for group in gaussians.optimizer.param_groups:
        group_name = '_' + group['name']
        group_name = group_name.replace('_f_dc', '_features_dc')
        group_name = group_name.replace('_f_rest', '_features_rest')
        assert len(group['params']) == 1
        state = gaussians.optimizer.state[group['params'][0]]
        exp_avg[group_name] = state['exp_avg']
        assert torch.count_nonzero(exp_avg[group_name]) > 0, f'ADAM exp avg for {group_name} is all 0! Did you reset ADAM states?'

    for i, cam_idx in enumerate(cam_indices):
        loss = backward_once(gaussians, viewpoint_stack[cam_idx], opt, pipe, background)

        if monitor_params is not None:
            raise NotImplementedError()
        else:
            for k in grad_keys:
                grad_running_sum[k] += getattr(gaussians, k).grad

                signal = target_grad[k].flatten() / len(cam_indices)
                sample = grad_running_sum[k].flatten() / (i + 1)
                noise = (sample - signal)
                SNRs[k].append(float(torch.inner(signal, signal) / torch.inner(noise, noise)))
                variances[k].append(float(get_variance(sample, mean=exp_avg[k].flatten()).mean()))  # get the variance of the mean gradient
                sparsities[k].append(float(get_sparsity(grad_running_sum[k])))
                cosines[k].append(
                    float(torch.nn.functional.cosine_similarity(sample, signal, dim=0)))

        gaussians.optimizer.zero_grad(set_to_none=True)
    sparsities_np = {k: np.array(v) for k, v in sparsities.items()}
    variances_np = {k: np.array(v) for k, v in variances.items()}
    cosines_np = {k: np.array(v) for k, v in cosines.items()}
    SNRs_np = {k: np.array(v) for k, v in SNRs.items()}
    return grad_running_sum, sparsities_np, variances_np, cosines_np, SNRs_np


def get_sparsity(grad: torch.Tensor) -> torch.Tensor:
    return (grad == 0).sum() / grad.numel()


def get_variance(grad: torch.Tensor, mean) -> torch.Tensor:
    return (grad - mean) ** 2


def restored_gaussians(model_params, dataset, opt, deepcopy=False) -> GaussianModel:
    if deepcopy:
        model_params = copy.deepcopy(model_params)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    gaussians.restore(model_params, opt)
    return gaussians


def fill_subplot(ax, title, xs, ys, xlabel, ylabel, xscale='linear', legend_labels: Union[str, List[str]] = "", markersize=4):
    if isinstance(ys[0], list):
        for i in range(len(ys)):
            if isinstance(xs[0], list):
                ax.plot(xs[i], ys[i], label=legend_labels[i].replace('_', ''), marker='.', markersize=markersize)
            else:
                ax.plot(xs, ys[i], label=legend_labels[i].replace('_', ''), marker='.', markersize=markersize)
    else:
        ax.plot(xs, ys, label=legend_labels.replace('_', ''), marker='.', markersize=markersize)
    ax.set_title(title)
    if xlabel != '':
        ax.set_xlabel(xlabel)
    ax.set_xscale(xscale)
    # if xscale == 'log':
    #     ax.set_xticks(xs)
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if ylabel != '':
        ax.set_ylabel(ylabel)
    if legend_labels != '':
        ax.legend()


def plot_histogram(grads, num_bins=1000):
    counts, bins = torch.histogram(grads, bins=num_bins)
    plt.hist(bins[:-1], bins, weights=counts)
    # plt.yscale('symlog')
    plt.ylim((0, counts.max()))
    plt.show()
    plt.close()


def plot_similarity_matrix(similarity_mats: List[torch.Tensor], param_groups: List[str], iteration_number: int, abs=True):
    assert len(similarity_mats) == len(param_groups)
    if len(param_groups) >= 4:
        nrows, ncols = 2, math.ceil(len(param_groups) / 2)
    else:
        nrows, ncols = 1, len(param_groups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows + 2), dpi=200)
    for i, ax in enumerate(fig.axes):
        if i >= len(param_groups):
            ax.axis('off')
            continue
        if abs:
            ax.imshow(torch.abs(similarity_mats[i]), cmap='gray')
        else:
            ax.imshow(similarity_mats[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Param group {param_groups[i]}')
        ax.set_xlabel('view #')
        ax.set_ylabel('view #')
    plt.suptitle(f'Absolute cosine similarity of gradients from different views: Iteration{iteration_number}\n')
    plt.tight_layout()
    plt.show()
    plt.close()


def get_variance_sparsity_cosine_SNR(gaussians, train_cameras, background, pipe, opt, keys, num_trials, accum_steps,
                                     sampling='random'):
    sparsities = {k: 0 for k in keys}
    variances = {k: 0 for k in keys}
    cosines = {k: 0 for k in keys}
    SNRs = {k: 0 for k in keys}
    for trial in tqdm(range(num_trials)):
        grads_runing_sum, s, v, c, snrs = get_grad_stats(gaussians, train_cameras, background, pipe, opt, keys,
                                                         sampling=sampling, accum_steps=accum_steps)
        # grads_runing_sum, s, v, c = get_grad_stats(gaussians, train_cameras, background, pipe, opt, keys,
        #                                         sampling="", accum_steps=accum_steps, determininstic_index=0)
        for k in keys:
            sparsities[k] += s[k] / num_trials
            variances[k] += v[k] / num_trials
            cosines[k] += c[k] / num_trials
            SNRs[k] += snrs[k] / num_trials
    return variances, sparsities, cosines, SNRs


def plot_variance_sparsity_cosine(dataset, opt, train_cameras, background, pipe, checkpoint, keys, num_trials=32,
                                  iters_list=[7000, 15000, 30000], sampling='random'):
    accum_steps = len(train_cameras)
    random.seed()
    scene_name = os.path.basename(args.source_path)
    variances: List[Dict[str: np.array]] = []
    sparsities: List[Dict[str: np.array]] = []
    cosines: List[Dict[str: np.array]] = []
    SNRs: List[Dict[str: np.array]] = []
    for iter in iters_list:
        chpt = checkpoint.rstrip(".pth").split("chkpnt")[1]
        print('loading checkpoint ', checkpoint.replace("chkpnt" + str(chpt), "chkpnt" + str(iter)))
        (model_params, first_iter) = torch.load(checkpoint.replace("chkpnt" + str(chpt), "chkpnt" + str(iter)))
        gaussians = restored_gaussians(model_params, dataset, opt)
        v, s, c, snrs = get_variance_sparsity_cosine_SNR(gaussians, train_cameras, background, pipe, opt, keys,
                                                         num_trials, accum_steps, sampling=sampling)
        variances.append(v)
        sparsities.append(s)
        cosines.append(c)
        SNRs.append(snrs)

    save_dict = {'scene_name': scene_name,
                 'cosines_checkpoint': cosines,
                 'sparsities_checkpoint': sparsities,
                 'variances_checkpoint': variances,
                 'SNRs_checkpoint': SNRs,
                 'keys': keys,
                 'num_trails': num_trials,
                 'iters_list': iters_list,
                 'sampling': sampling}

    scene_name = os.path.basename(args.source_path)
    for k in keys:
        ncols = 3
        fig, ax = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5), dpi=300)
        xs = np.arange(1, accum_steps + 1)
        for i, iter in enumerate(iters_list):
            fill_subplot(ax[0], 'Batch size vs Grad Sparsity', xs,
                         sparsities[i][k],
                         'Batch size (log scale)', 'Sparsity', xscale='log', legend_labels='iter ' + str(iter))
            fill_subplot(ax[1], 'Batch size vs Grad Variance', xs,
                         variances[i][k],
                         'Batch size', 'Avg Parameter Variance', xscale='linear', legend_labels='iter ' + str(iter))
            fill_subplot(ax[2], 'Batch size vs Grad Precision', xs,
                         1/variances[i][k],
                         'Batch size', '1/(Avg Parameter Variance)', xscale='linear', legend_labels='iter ' + str(iter))
        # fig.suptitle(f'Scene: {scene_name}. Param group: {k.replace("_", "")}.')
        fig.tight_layout()
        os.makedirs(os.path.join('plots_snr', scene_name), exist_ok=True)
        fig_save_path = os.path.join('plots_snr', scene_name, f'scene_{scene_name}_param_{k.replace("_", "")}_sampling_{sampling}_trials_{num_trials}.pdf')
        fig.savefig(fig_save_path)
        torch.save(save_dict, (fig_save_path.rstrip('.pdf') + '.pt'))
        fig.show()
        plt.close(fig)


def backward_once(gaussians: GaussianModel, viewpoint_cam, opt, pipe, background):
    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
        "viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
    # Loss
    gt_image = viewpoint_cam.original_image
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss.backward()
    return loss


def run_iterations(gaussians: GaussianModel, train_cameras, opt, camera_ids, batch_size, pipe, background,
                   discard_last=True):
    gaussians.optimizer.zero_grad(set_to_none=True)
    for i, camera_id in enumerate(camera_ids):
        backward_once(gaussians, train_cameras[camera_id], opt, pipe, background)

        # Update params every batch_size iterations
        if (i + 1) % batch_size == 0:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
        elif not discard_last and i == len(camera_ids) - 1:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.optimizer.zero_grad(set_to_none=True)
    return gaussians


def clear_adam_state(optimizer, batch_size, rescale_betas=True,
                     lr_scaling: str = 'sqrt',
                     disable_momentum: bool = False):
    for group in optimizer.param_groups:
        # clear ADAM state
        # effective_bses = [1.0, 1.5049873111431473, 2.7032357769224435, 4.359507132813812, 4.9381875790105365, 5.37417372027395, 6.163066933794536, 6.572855687883565, 7.578009795823763, 8.639301783737858, 9.142897465170346, 10.531347945681924, 11.451054537769194, 11.850305762724846, 11.31590084660537, 11.02668728849445, 11.572372261790177, 11.329266139074573, 11.46019140106616, 12.001660258735463, 12.728031443382296, 13.272023941036055, 12.977059320122164, 13.13261867972596, 13.641024817672943, 13.863450764797912, 14.112725337059102, 14.897364432803249, 15.038098707851267, 15.291073712932263, 15.594879672043568, 15.73044990073995, 15.769253705047083, 15.994006063120962, 16.265393216035044, 16.336609245161462, 16.567914108028834, 16.762598910248542, 16.751942597858555, 17.135993175111082, 17.34633401392654, 17.36684923046376, 17.578179823599697, 17.629964652825613, 17.80358989486209, 17.880135141696073, 17.893365409766343, 17.935315961936652, 17.780071022356157, 17.981475199145397, 17.879820084765228, 18.00647721701445, 18.007065294978354, 17.88691523478462, 17.820218701577197, 17.933964444132243, 18.26070565613837, 18.321030566294265, 18.593946721914502, 18.805305763587324, 18.848135248165782, 19.160055197061382, 19.225264338855407, 19.421792351502056, 19.344990643231107, 19.432703002303867, 19.567476459665038, 19.62953673057074, 19.7831592543389, 20.152570299432924, 20.3866609958896, 20.380649013296864, 20.550959363879038, 20.67827675694694, 20.71336314423314, 20.72956904434027, 20.693768471396822, 20.567774996806506, 20.657902509279047, 20.692330334166677, 20.768576234455857, 20.85520118005168, 20.909910264106053, 21.014815305371226, 21.07972637443553, 21.28139040950196, 21.22919965786919, 21.330730787581157, 21.520202713870997, 21.547064501763085, 21.639366841189506, 21.712038416352218, 21.718866065203915, 21.84990214910783, 21.849152721233253, 21.856460598371715, 22.003717443558745, 22.0886741159992, 22.076349955058873, 22.15019089246895, 22.117001640448258, 22.13917961027548, 22.146765179102314, 22.20826030071563, 22.323677713266267, 22.32421314020669, 22.485194918321717, 22.570085352377788, 22.526872874063994, 22.521910090495673, 22.538584329948783, 22.58696467605947, 22.60043272294547, 22.622832336597355, 22.635177727789088, 22.730132574825987, 22.811644851278572, 22.852522851147928, 22.90067688277864, 22.970394480760927, 22.94129375055572, 23.003269100455803, 23.049292888660588, 23.046074382547662, 23.175441768691183, 23.171503557055644, 23.289909339793983, 23.319119618096437, 23.340102749374612, 23.331992368733058, 23.339944371699136, 23.348534556022432, 23.361932118473774, 23.405903790577796, 23.50292094423613, 23.534112854204686, 23.57479723418123, 23.57624471272199, 23.62499149297192, 23.69120992091771, 23.776791611959517, 23.767258804473855, 23.82374188833559, 23.884781408753767, 23.873846130860326, 23.88346895461194, 23.912116631252175, 23.95605433495395, 23.912260996961724, 23.965874175438596, 23.971844546738577, 23.978796604863504, 23.9806648669871, 24.08468842821396, 24.12982988717037, 24.18392414314407, 24.179952812316337, 24.122455347042493, 24.164281914544603, 24.16658624603205, 24.2095758067061, 24.181303056305666, 24.219533219219848, 24.28376576944926, 24.28959050122461, 24.358193935613187, 24.367555627272523, 24.349283599098708, 24.393242108446856, 24.39535239543647, 24.470697708461067, 24.466522142614824, 24.40268744729673, 24.43494808812099, 24.447839521378842, 24.462202211059033, 24.55593462022006, 24.578429345398238, 24.565906469304764, 24.576965307261382, 24.599242634665178, 24.6904702987655, 24.690620608945444, 24.691845594451486, 24.692939801608873, 24.736526356952417, 24.7990201495918, 24.82763131019962, 24.829790426767463, 24.836258859765394, 24.855294327779717, 24.89008221773117, 24.90383092854074, 24.91560480136466]
        # effective_bs = effective_bses[batch_size - 1]
        if rescale_betas:
            group['betas'] = (group['betas'][0] ** batch_size, group['betas'][1] ** batch_size)
            if disable_momentum:
                group['betas'] = (group['betas'][0] * 0, group['betas'][1])
        for p in group['params']:
            state = optimizer.state[p]
            state['exp_avg'] *= 0
            state['exp_avg_sq'] *= 0
            state['step'] *= 0
        if lr_scaling == 'constant':
            coeff = 1
        elif lr_scaling == 'sqrt':
            coeff = float(batch_size) ** 0.5
        elif lr_scaling == 'linear':
            coeff = float(batch_size)
        elif lr_scaling.startswith('power-'):
            power = float(lr_scaling.lstrip('power-'))
            coeff = float(batch_size) ** power
        else:
            raise ValueError(f'Unknown lr_scaling {lr_scaling}')
        group['lr'] *= coeff


def print_learning_rate(optimizer):
    for group in optimizer.param_groups:
        print(group['name'], group['lr'])


def get_test_errors(gaussians: GaussianModel, test_cameras, pipe, background):
    with torch.no_grad():
        l1_test = 0.0
        # psnr_test = 0.0
        for idx, viewpoint in enumerate(test_cameras):
            image = torch.clamp(render(viewpoint, gaussians, pipe, background)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            l1_test += l1_loss(image, gt_image).mean().double()
            # psnr_test += psnr(image, gt_image).mean().double()
        l1_test /= len(test_cameras)
        # psnr_test /= len(test_cameras)
    return l1_test


def compute_batch_size_vs_weights_delta(dataset, opt, train_cameras, test_cameras, background, pipe, checkpoint_path, keys,
                                        num_trials=32,
                                        checkpoints_list=[15000, 30000],
                                        batch_sizes=[1, 4, 16, 64], warmup_epochs=1, run_epochs=5,
                                        rescale_betas=True,
                                        lr_scaling: str = 'sqrt',
                                        disable_momentum=False,
                                        iid_sampling=False):
    random.seed()
    cosines_for_checkpoint = []
    norms_for_checkpoint = []
    losses_for_checkpoint = []
    test_losses_for_checkpoint = []
    param_index_map = {'_xyz': 1, '_features_dc': 2, '_features_rest': 3, '_scaling': 4, '_rotation': 5, '_opacity': 6}

    for checkpoint_itr in checkpoints_list:
        cosines: Dict[str, List[Dict[int, float]]] = {k: [{} for _ in range(len(batch_sizes))] for k in keys}
        norms: Dict[str, List[Dict[int, float]]] = {k: [{} for _ in range(len(batch_sizes))] for k in keys}
        losses: List[Dict[int, float]] = [defaultdict(float) for _ in range(len(batch_sizes))]
        test_losses: List[Dict[int, float]] = [defaultdict(float) for _ in range(len(batch_sizes))]
        chpt = checkpoint_path.rstrip(".pth").split("chkpnt")[1]
        cur_checkpoint = checkpoint_path.replace("chkpnt" + str(chpt), "chkpnt" + str(checkpoint_itr))
        print('loading checkpoint ', cur_checkpoint)
        (model_params, first_iter) = torch.load(cur_checkpoint)

        original_params = {k: model_params[param_index_map[k]] for k in keys}
        if iid_sampling:
            camera_idx = np.random.choice(np.arange(len(train_cameras)),
                                          size=len(train_cameras) * (warmup_epochs + run_epochs), replace=True)
        else:
            camera_idx = np.concatenate(
                [np.random.permutation(np.arange(len(train_cameras))) for _ in range(warmup_epochs + run_epochs)])

        for batch_size in batch_sizes:
            if batch_size == 1:
                continue
            print('Running for batch size', batch_size)
            temp_batch_sizes = [1, batch_size]
            running_gaussians = [restored_gaussians(model_params, dataset, opt, deepcopy=True) for _ in
                                 temp_batch_sizes]
            # Readjust ADAM parameters for batch size > 1
            for i in range(len(temp_batch_sizes)):
                clear_adam_state(running_gaussians[i].optimizer, temp_batch_sizes[i], rescale_betas=rescale_betas,
                                 lr_scaling=lr_scaling, disable_momentum=disable_momentum)
                # warmup new ADAM state
                run_iterations(running_gaussians[i], train_cameras, opt,
                               camera_idx[:len(train_cameras) * warmup_epochs], temp_batch_sizes[i], pipe, background,
                               discard_last=True)
                running_gaussians[i].restore_parameters(model_params, opt)

            for i, camera_id in enumerate(tqdm(camera_idx[len(train_cameras) * warmup_epochs:])):
                for j, temp_batch_size in enumerate(temp_batch_sizes):
                    running_gaussian = running_gaussians[j]
                    loss = backward_once(running_gaussian, train_cameras[camera_id], opt, pipe, background)

                    # average gradients from views
                    next_accum_step = min(len(camera_idx), (i // temp_batch_size + 1) * temp_batch_size)
                    last_accum_step = (i // temp_batch_size) * temp_batch_size
                    # loss /= min(temp_batch_size, len(train_cameras) - last_accum_step)
                    if temp_batch_size == 1:
                        losses[batch_sizes.index(temp_batch_size)][next_accum_step] = float(
                            loss.item()) / temp_batch_size
                    else:
                        losses[batch_sizes.index(temp_batch_size)][next_accum_step] += float(
                            loss.item()) / temp_batch_size
                    # Update params every batch_size iterations
                    # if (i + 1) % temp_batch_size == 0 or i == len(camera_idx) - 1:
                    if (i + 1) % temp_batch_size == 0:
                        running_gaussian.optimizer.step()
                        running_gaussian.optimizer.zero_grad(set_to_none=True)
                        for k in keys:
                            # compare weight delta from batch-size 1 and that from the current batch-size
                            weight_delta = getattr(running_gaussian, k).detach() - original_params[k]
                            norms[k][batch_sizes.index(temp_batch_size)][i + 1] = float(torch.linalg.norm(weight_delta))
                            if len(test_cameras) > 0:
                                test_losses[batch_sizes.index(temp_batch_size)][i + 1] = float(get_test_errors(running_gaussian, test_cameras, pipe, background))
                            if temp_batch_size != 1:
                                reference_weight_delta = getattr(running_gaussians[0], k).detach() - original_params[k]
                                cosines[k][batch_sizes.index(temp_batch_size)][i + 1] = float(
                                    torch.nn.functional.cosine_similarity(reference_weight_delta.flatten(),
                                                                          weight_delta.flatten(), dim=0))
            del running_gaussians, running_gaussian, weight_delta, reference_weight_delta, loss
        cosines_for_checkpoint.append(cosines)
        losses_for_checkpoint.append(losses)
        test_losses_for_checkpoint.append(test_losses)
        norms_for_checkpoint.append(norms)
        del model_params, first_iter, original_params
    # pprint.pp(losses_for_checkpoint)
    # pprint.pp(norms_for_checkpoint)
    # pprint.pp(cosines_for_checkpoint)
    return cosines_for_checkpoint, losses_for_checkpoint, test_losses_for_checkpoint, norms_for_checkpoint


def plot_weight_deltas_cosine_norm_loss(cosines, losses, norms, keys, checkpoint_iter, batch_sizes, rescale_betas: bool, lr_scaling: str,
                                        warmup_epochs: int, disable_momentum: bool, iid_sampling: bool):
    scene_name = os.path.basename(args.source_path)
    save_dict = {'scene_name': scene_name,
                 'cosines_checkpoint': cosines,
                 'test_losses_checkpoint': losses,
                 'norms_checkpoint': norms,
                 'keys': keys,
                 'checkpoint_iter': checkpoint_iter,
                 'batch_sizes': batch_sizes,
                 'rescale_betas': rescale_betas,
                 'lr_scaling': lr_scaling,
                 'warmup_epochs': warmup_epochs,
                 'disable_momentum': disable_momentum,
                 'iid_sampling': iid_sampling}

    for k in keys:
        fig, ax = plt.subplots(1, 3, figsize=(6 * 3, 6), dpi=200)
        fill_subplot(ax[0], 'Batch size vs cosine(weight delta w.r.t bs=1)',
                     [list(cosines[k][j].keys()) for j in range(len(batch_sizes))],
                     [list(cosines[k][j].values()) for j in range(len(batch_sizes))],
                     'Iterations', 'Cosine Sim', xscale='linear', legend_labels=[f'BS {b}' for b in batch_sizes])
        ax[0].set_ylim([0.4, 1.0])
        fill_subplot(ax[1], 'Batch size vs Train Loss',
                     [list(losses[j].keys()) for j in range(len(batch_sizes))],
                     [list(losses[j].values()) for j in range(len(batch_sizes))],
                     'Iterations', 'loss', xscale='linear', legend_labels=[f'BS {b}' for b in batch_sizes])
        fill_subplot(ax[2], 'Batch size vs norm(weight delta)',
                     [list(norms[k][j].keys()) for j in range(len(batch_sizes))],
                     [list(norms[k][j].values()) for j in range(len(batch_sizes))],
                     'Iterations', 'norm', xscale='linear', legend_labels=[f'BS {b}' for b in batch_sizes])

        disable_momentum_str = '_disable momentum' if disable_momentum else ''
        fig.suptitle(
            f'Scene: {scene_name}. Checkpoint {checkpoint_iter}. Rescale betas: {rescale_betas}{disable_momentum_str}. LR scaling: {lr_scaling}. Warmup: {warmup_epochs} epochs. IID {iid_sampling}. Params: {k}')
        fig.tight_layout()
        os.makedirs(os.path.join('plots_grad_delta_new', scene_name), exist_ok=True)
        fig_save_path = os.path.join('plots_grad_delta_new', scene_name,
                                 f'scene_{scene_name}_checkpoint_{checkpoint_iter}_param_{k.replace("_", "")}'
                                 f'_rescale_betas_{rescale_betas}{disable_momentum_str.replace(" ", "_")}_lr_{lr_scaling}_warmup_{warmup_epochs}_iid_{iid_sampling}_test_losses.pdf')
        fig.savefig(fig_save_path)
        torch.save(save_dict, (fig_save_path.rstrip('.pdf') + '.pt'))
        if k == '_xyz':
            fig.show()
        plt.close(fig)


def plot_grad_similarity_between_views(dataset, opt, train_cameras, background, pipe, checkpoint_path, keys,
                                       checkpoints_list: List[int]):

    for checkpoint_itr in checkpoints_list:
        grads: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
        similarity_matrix: Dict[str, torch.Tensor] = {k: torch.eye(len(train_cameras)) for k in keys}
        chpt = checkpoint_path.rstrip(".pth").split("chkpnt")[1]
        cur_checkpoint = checkpoint_path.replace("chkpnt" + str(chpt), "chkpnt" + str(checkpoint_itr))
        print('loading checkpoint ', cur_checkpoint)
        (model_params, first_iter) = torch.load(cur_checkpoint)
        gaussians = restored_gaussians(model_params, dataset, opt)
        print('Getting gradients ', cur_checkpoint)
        for i in tqdm(range(len(train_cameras))):
            gaussians.optimizer.zero_grad(set_to_none=True)
            loss = backward_once(gaussians, train_cameras[i], opt, pipe, background)
            for k in keys:
                grads[k].append(getattr(gaussians, k).grad.clone().cpu())
        del gaussians
        print('Computing cosines ', cur_checkpoint)
        for k in keys:
            for i in tqdm(range(len(train_cameras))):
                grads_i = grads[k][i].flatten().cuda()
                for j in range(i + 1, len(train_cameras)):
                    grads_j = grads[k][j].flatten().cuda()
                    sim = torch.cosine_similarity(grads_i, grads_j, dim=0).cpu()
                    similarity_matrix[k][i][j] = sim
                    similarity_matrix[k][j][i] = sim
        plot_similarity_matrix(list(similarity_matrix.values()), list(similarity_matrix.keys()), checkpoint_itr)




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset, None)
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    random.seed()
    n_epochs = 0
    keys = ['_xyz', '_rotation', '_scaling', '_opacity', '_features_dc']
    num_views = len(scene.getTrainCameras())
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTrainCameras()[:5]
    del scene

    # plot_grad_similarity_between_views(dataset, opt, train_cameras, background, pipe, checkpoint, keys,
    #                                    [7000, 15000, 30000])
    # quit()

    #['random', 'random_wo_replacement']
    plot_variance_sparsity_cosine(dataset, opt, train_cameras, background, pipe, checkpoint, keys, num_trials=32, sampling='random_wo_replacement')
    # quit()

    # Run all experiments for first checkpoint first, then for second
    for checkpoints_list in [[7000], [15000], [30000]]:
        batch_sizes = [1, 4, 8, 16, 32, 64]
        run_epochs = 4
        # Optimal Defaults
        iid_sampling = False
        warmup_epochs = 2
        disable_momentum = False
        rescale_betas = True
        lr_scaling = 'sqrt'

        for lr_scaling in ['sqrt', 'constant', 'linear']:
            # for lr_scaling in ['power-0.67', 'power-0.75']:
            cosines_checkpoint, losses_checkpoint, test_losses_checkpoint, norms_checkpoint = compute_batch_size_vs_weights_delta(
                dataset, opt, train_cameras, test_cameras, background, pipe, checkpoint, keys,
                checkpoints_list=checkpoints_list, batch_sizes=batch_sizes,
                run_epochs=run_epochs, warmup_epochs=warmup_epochs,
                rescale_betas=rescale_betas,
                disable_momentum=disable_momentum,
                lr_scaling=lr_scaling,
                iid_sampling=iid_sampling)
            for i in range(len(checkpoints_list)):
                plot_weight_deltas_cosine_norm_loss(cosines_checkpoint[i], test_losses_checkpoint[i], norms_checkpoint[i], keys, checkpoints_list[i],
                                                    batch_sizes, rescale_betas, lr_scaling, warmup_epochs, disable_momentum, iid_sampling)
        lr_scaling = 'sqrt'

        for rescale_betas in [False]:
            cosines_checkpoint, losses_checkpoint, test_losses_checkpoint, norms_checkpoint = compute_batch_size_vs_weights_delta(
                dataset, opt, train_cameras, test_cameras, background, pipe, checkpoint, keys,
                checkpoints_list=checkpoints_list, batch_sizes=batch_sizes,
                run_epochs=run_epochs, warmup_epochs=warmup_epochs,
                rescale_betas=rescale_betas,
                disable_momentum=disable_momentum,
                lr_scaling=lr_scaling,
                iid_sampling=iid_sampling)
            for i in range(len(checkpoints_list)):
                plot_weight_deltas_cosine_norm_loss(cosines_checkpoint[i], test_losses_checkpoint[i], norms_checkpoint[i], keys, checkpoints_list[i],
                                                    batch_sizes, rescale_betas, lr_scaling, warmup_epochs, disable_momentum, iid_sampling)
        rescale_betas = True
        #
        # for warmup_epochs in [0, 1]:
        #     cosines_checkpoint, losses_checkpoint, test_losses_checkpoint, norms_checkpoint = compute_batch_size_vs_weights_delta(
        #         dataset, opt, train_cameras, test_cameras, background, pipe, checkpoint, keys,
        #         checkpoints_list=checkpoints_list, batch_sizes=batch_sizes,
        #         run_epochs=run_epochs, warmup_epochs=warmup_epochs,
        #         rescale_betas=rescale_betas,
        #         disable_momentum=disable_momentum,
        #         lr_scaling=lr_scaling,
        #         iid_sampling=iid_sampling)
        #     for i in range(len(checkpoints_list)):
        #         plot_weight_deltas_cosine_norm_loss(cosines_checkpoint[i], test_losses_checkpoint[i], norms_checkpoint[i], keys, checkpoints_list[i],
        #                                             batch_sizes, rescale_betas, lr_scaling, warmup_epochs, disable_momentum, iid_sampling)
        # warmup_epochs = 2
        #
        # for iid_sampling in [False]:
        #     cosines_checkpoint, losses_checkpoint, test_losses_checkpoint, norms_checkpoint = compute_batch_size_vs_weights_delta(
        #         dataset, opt, train_cameras, test_cameras, background, pipe, checkpoint, keys,
        #         checkpoints_list=checkpoints_list, batch_sizes=batch_sizes,
        #         run_epochs=run_epochs, warmup_epochs=warmup_epochs,
        #         rescale_betas=rescale_betas,
        #         disable_momentum=disable_momentum,
        #         lr_scaling=lr_scaling,
        #         iid_sampling=iid_sampling)
        #     for i in range(len(checkpoints_list)):
        #         plot_weight_deltas_cosine_norm_loss(cosines_checkpoint[i], test_losses_checkpoint[i], norms_checkpoint[i], keys, checkpoints_list[i],
        #                                             batch_sizes, rescale_betas, lr_scaling, warmup_epochs, disable_momentum, iid_sampling)
        # iid_sampling = True
        #
        # for warmup_epochs in [2]:
        #     for lr_scaling in ['sqrt', 'constant', 'linear']:
        #         disable_momentums = [False, True] if warmup_epochs == 2 else [False]
        #         for disable_momentum in disable_momentums:
        #             rescale_betas = True
        #             cosines_checkpoint, losses_checkpoint, test_losses_checkpoint, norms_checkpoint = compute_batch_size_vs_weights_delta(
        #                 dataset, opt, train_cameras, test_cameras, background, pipe, checkpoint, keys,
        #                 checkpoints_list=checkpoints_list, batch_sizes=batch_sizes,
        #                 run_epochs=run_epochs, warmup_epochs=warmup_epochs,
        #                 rescale_betas=rescale_betas,
        #                 disable_momentum=disable_momentum,
        #                 lr_scaling=lr_scaling,
        #                 iid_sampling=iid_sampling)
        #             for i in range(len(checkpoints_list)):
        #                 plot_weight_deltas_cosine_norm_loss(cosines_checkpoint[i], test_losses_checkpoint[i], norms_checkpoint[i], keys, checkpoints_list[i],
        #                                                     batch_sizes, rescale_betas, lr_scaling, warmup_epochs, disable_momentum, iid_sampling)
    quit()

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            epoch_change = True
            n_epochs += 1
        else:
            epoch_change = False
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if epoch_change and False:
            print('epoch ', n_epochs)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
            #                 testing_iterations, scene, render, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
            #                                                          radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            #
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            #
            #     if iteration % opt.opacity_reset_interval == 0 or (
            #             dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("/tmp/sparsity-output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    save_iters = [0, 1_000, 4_500, 7_000, 11_000, 15_000, 18_000, 21_000, 24_000, 27_000, 30_000]
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=save_iters)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=save_iters)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=save_iters)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")