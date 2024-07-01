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

import os
import torch
from random import randint
import cv2

from scene.latent_gaussian_model import LatentGaussianModel
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

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, debug_latent):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, dummy=debug_latent)
    assert dataset.sh_degree == 0
    gaussians = LatentGaussianModel(dataset.sh_degree, torch.zeros((1, 3), device=torch.device('cuda')))
    # gaussians.set_freeze_structures_params(True)
    scene = Scene(dataset, gaussians, downsample_init=1.0)
    gaussians.training_setup(opt)
    if checkpoint:
        # raise NotImplementedError()
        (model_state_dict, first_iter) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt)
        try:
            gaussians.load_state_dict(model_state_dict)
        except RuntimeError as e:
            print(f'Warning: mismatched keys. Trying torch.load with strict=False\n{e}')
            gaussians.load_state_dict(model_state_dict, strict=False)
        gaussians.active_sh_degree = gaussians.max_sh_degree

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    print(f'First iteration: {first_iter}; debug: {debug_latent}')
    if debug_latent:
        debug_iters = 10000000
    else:
        debug_iters = 0
    cv2_key = -1
    for iteration in range(first_iter, max(opt.iterations + 1, debug_iters)):
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

        # gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        if cv2_key == ord('q'):
            break
        latent_noise = torch.randn((1, gaussians.latent_size), device=gaussians.structure_latents.device, dtype=gaussians.structure_latents.dtype) * 0

        gaussians.forward()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()

        if debug_latent:
            gt = gt_image.cpu().numpy().transpose(1, 2, 0)
            rendered = torch.clip(image.detach(), 0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
            cv2.imshow("render", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            cv2.imshow("ground truth", cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
            cv2_key = cv2.waitKey(0)
            while cv2_key != ord('c'):
                if cv2_key == ord('n'):
                    latent_noise = torch.randn((1, gaussians.latent_size), device=gaussians.structure_latents.device, dtype=gaussians.structure_latents.dtype)
                elif cv2_key == ord('p'):
                    latent_noise *= 1.414
                elif cv2_key == ord('m'):
                    latent_noise /= 1.414
                if torch.linalg.norm(latent_noise) > 0:
                    print('Noise norm', torch.linalg.norm(latent_noise))
                with torch.no_grad():
                    gaussians.forward(latent_noise=latent_noise)
                    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
                    image_2, _, _, _ = render_pkg["render"], render_pkg[
                        "viewspace_points"], \
                        render_pkg["visibility_filter"], render_pkg["radii"]
                gt = gt_image.cpu().numpy().transpose(1, 2, 0)
                rendered = torch.clip(image_2.detach(), 0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
                # import pdb
                # pdb.set_trace()
                cv2.imshow("render", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                cv2.imshow("ground truth", cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
                cv2_key = cv2.waitKey(0)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and False:
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                #                                                      radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                #
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                #
                # if iteration % opt.opacity_reset_interval == 0 or (
                #         dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()
                pass

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                # print('stepped', iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.state_dict(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args, dummy=False):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())

        if dummy:
            args.model_path = os.path.join("/tmp/3dgs_tmp/", unique_str[0:10])
        else:
            args.model_path = os.path.join("./output_lgm/", unique_str[0:10])
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
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 100, 500, 1_000, 3_000, 7_000, 30_000, 45_000, 60_000, 75_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 100, 500, 1_000, 3_000, 7_000, 30_000, 45_000, 60_000, 75_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1, 100, 500, 1_000, 3_000, 7_000, 30_000, 45_000, 60_000, 75_000, 90_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--debug_latent', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.debug_latent)

    # All done
    print("\nTraining complete.")
