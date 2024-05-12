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
import math
import os
import pdb
from collections import defaultdict
from typing import List, Union, Tuple

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


def get_grad_stats(gaussians: GaussianModel, viewpoint_stack, background, pipe, opt, grad_keys,
                   sampling: str = "random",
                   accum_steps: int = 1, determininstic_index: int = None,
                   monitor_params: Tuple[str, List[int]] = None):
    assert sampling in ["random", "nearby", ""]
    cam_indices = None

    if determininstic_index is not None:
        cam_indices = list(range(determininstic_index, determininstic_index + accum_steps))
    else:
        if sampling == "random":
            cam_indices = [randint(0, len(viewpoint_stack) - 1) for _ in range(accum_steps)]
        elif sampling == 'nearby':
            index = randint(0, len(viewpoint_stack) - accum_steps)
            cam_indices = list(range(index, index + accum_steps))
    gradients = defaultdict(list)

    for i, cam_idx in enumerate(cam_indices):
        viewpoint_cam = viewpoint_stack[cam_idx]
        bg = background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        if monitor_params is not None:
            group_name, params_list = monitor_params
            assert group_name in grad_keys
            gradients[group_name].append(
                getattr(gaussians, group_name).grad.view(-1)[params_list].clone().to('cpu', non_blocking=True))
        else:
            for k in grad_keys:
                gradients[k].append(getattr(gaussians, k).grad.clone().to('cpu', non_blocking=True))
        gaussians.optimizer.zero_grad(set_to_none=True)
    return gradients


def get_sparsity(grad: torch.Tensor):
    return (grad == 0).sum() / grad.numel()


def get_variance(grad: torch.Tensor, mean):
    return (grad - mean) ** 2


def fill_subplot(ax, title, xs, ys, xlabel, ylabel, xscale='linear', legend_labels: Union[str, List[str]] = ""):
    if isinstance(ys[0], list):
        for i in range(len(ys)):
            ax.plot(xs, ys[i], label=legend_labels[i].replace('_', ''), marker='.')
    else:
        ax.plot(xs, ys, label=legend_labels.replace('_', ''), marker='.')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xscale(xscale)
    if xscale == 'log':
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
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


def plot_covariance(cov, to_plot: List[int], sqrt=True):
    if len(to_plot) >= 4:
        nrows, ncols = 2, math.ceil(len(to_plot) / 2)
    else:
        nrows, ncols = 1, len(to_plot)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows + 2), dpi=200)
    for i, ax in enumerate(fig.axes):
        if i > len(to_plot):
            break
        if sqrt:
            ax.imshow(torch.sqrt(torch.abs(cov[to_plot[i]])), cmap='gray')
            ax.set_title(f'sqrt(covariance) for parameter {to_plot[i]}')
        else:
            ax.imshow(torch.abs(cov[to_plot[i]]), cmap='gray')
            ax.set_title(f'covariance for parameter {to_plot[i]}')
        ax.set_xlabel('view #')
        ax.set_ylabel('view #')
    plt.suptitle('Grad covariance by view')
    plt.show()
    plt.close()


def get_variance_sparsity(gaussians, scene, background, pipe, opt, keys, num_trials, accum_steps_list):
    sparsities = {k: defaultdict(float) for k in keys}
    variances = {k: defaultdict(float) for k in keys}
    for trial in range(num_trials):
        grads = get_grad_stats(gaussians, scene.getTrainCameras().copy(), background, pipe, opt, keys,
                               sampling="random", accum_steps=accum_steps_list[-1])
        for accum_steps in accum_steps_list:
            # grads = get_grad_stats(gaussians, scene.getTrainCameras().copy(), background, pipe, opt, keys,
            #                        sampling="random", accum_steps=accum_steps)
            for k in keys:
                sum_grads = sum(grads[k][:accum_steps])
                variance = get_variance(sum_grads, 0).flatten()
                # variance_argsort = torch.argsort(variance)
                sparsities[k][accum_steps] += float(get_sparsity(sum_grads)) / num_trials
                variances[k][accum_steps] += float(variance.mean()) / num_trials
                # variances_10th_percentile[k][accum_steps] += float(variance[variance_argsort[int(len(variance) * 0.1)]]) / num_trials
                # variances_90th_percentile[k][accum_steps] += float(variance[variance_argsort[int(len(variance) * 0.9)]]) / num_trials
        # gradients_nearby = get_grad_stats(gaussians, scene.getTrainCameras().copy(), background, pipe, opt, sampling="nearby", accum_steps=accum_steps)
    return variances, sparsities

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

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
    num_params = gaussians._xyz.numel()
    num_params_of_interest = 1024
    parameters_of_interest = ('_xyz', [randint(0, num_params - 1) for _ in range(num_params_of_interest)])
    print('monitoring params', parameters_of_interest)
    cov = torch.zeros((num_params_of_interest, num_views, num_views), dtype=torch.float32)

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
            assert gaussians._xyz.grad is None
            grads = get_grad_stats(gaussians, scene.getTrainCameras().copy(), background, pipe, opt, keys,
                                   sampling="", accum_steps=num_views, determininstic_index=0,
                                   monitor_params=parameters_of_interest)
            # for i in range(num_params_of_interest):
            #     grad_per_view = torch.tensor([grads['_xyz'][view][i] for view in range(num_views)])
            #     cov[i] += torch.outer(grad_per_view, grad_per_view)
            # plot_covariance(cov / n_epochs, to_plot=list(range(6)))
            grad_per_view = torch.stack([grads['_xyz'][view] for view in range(num_views)], dim=0)  # n_views, n_params
            cov[0] = grad_per_view @ grad_per_view.T
            plot_covariance(cov, to_plot=list(range(1)))
            assert gaussians._xyz.grad is None

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

        sparsities = {k: defaultdict(float) for k in keys}
        variances = {k: defaultdict(float) for k in keys}
        variances_10th_percentile = {k: defaultdict(float) for k in keys}
        variances_90th_percentile = {k: defaultdict(float) for k in keys}
        accum_steps_list = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]

        random.seed()
        num_trials = 64
        variances: List[Dict[str: DefaultDict[float]]] = []
        sparsities: List[Dict[str: DefaultDict[float]]] = []
        iters_list = [15000, 21000, 27000, 30000]
        for iter in iters_list:
            chpt = checkpoint.rstrip(".pth").split("chkpnt")[1]
            print('loading checkpoint ', checkpoint.replace("chkpnt" + str(chpt), "chkpnt" + str(iter)))
            (model_params, first_iter) = torch.load(checkpoint.replace("chkpnt" + str(chpt), "chkpnt" + str(iter)))
            gaussians.restore(model_params, opt)
            v, s = get_variance_sparsity(gaussians, scene, background, pipe, opt, keys, num_trials, accum_steps_list)
            variances.append(v)
            sparsities.append(s)

        scene_name = os.path.basename(args.source_path)
        for k in keys:
            fig, ax = plt.subplots(1, 2, figsize=(6 * 2, 6))
            for i, iter in enumerate(iters_list):
                fill_subplot(ax[0], 'Batch size (simulated) vs Sparsity', accum_steps_list,
                             [sparsities[i][k][s] for s in accum_steps_list],
                             'Batch size', 'Sparsity', xscale='log', legend_labels='iter ' + str(iter))
                fill_subplot(ax[1], 'Batch size (simulated) vs Variance', accum_steps_list,
                             [variances[i][k][s] for s in accum_steps_list],
                             'Batch size', 'Avg Parameter Variance', xscale='linear', legend_labels='iter ' + str(iter))
            fig.suptitle(f'Scene: {scene_name}. Param group: {k.replace("_", "")}')
            fig.tight_layout()
            os.makedirs(os.path.join('plots', scene_name), exist_ok=True)
            fig.savefig(os.path.join('plots', scene_name, f'scene_{scene_name}_param_{k.replace("_", "")}_4.png'))
            fig.show()
            plt.close(fig)
        quit()

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
