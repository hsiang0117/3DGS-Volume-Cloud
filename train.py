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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from tqdm import tqdm
from utils.image_utils import psnr
from PIL import Image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

LPIPS_AVAILABLE = False

def build_timestamped_model_path(base_dir="./output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(base_dir, timestamp)

    if not os.path.exists(model_path):
        return model_path

    suffix = 1
    while True:
        candidate = os.path.join(base_dir, f"{timestamp}_{suffix:02d}")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1

def save_periodic_render(model_path, iteration, image_tensor, image_name=None):
    render_dir = os.path.join(model_path, "render_test")
    os.makedirs(render_dir, exist_ok=True)
    suffix = f"_{image_name}" if image_name else ""
    render_path = os.path.join(render_dir, f"iter_{iteration:06d}{suffix}.png")
    img8 = (torch.clamp(image_tensor, 0.0, 1.0) * 255.0).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    Image.fromarray(img8).save(render_path)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # Sky-board color used outside alpha mask (mask=0). Tunable constant; kept fixed for now.
    sky_board = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda") if not dataset.white_background else torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
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
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Grab diagnostic tensors injected by render()
        diag_T_light = render_pkg.get("T_light")
        diag_Lk = render_pkg.get("Lk")

        alpha_mask = None
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda().clamp(0.0, 1.0)

        # Composite sky-board outside mask for visualization and (masked) supervision.
        if alpha_mask is not None:
            image_for_loss = image * alpha_mask + sky_board[:, None, None] * (1.0 - alpha_mask)
        else:
            image_for_loss = image

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if alpha_mask is not None:
            gt_for_loss = gt_image * alpha_mask + sky_board[:, None, None] * (1.0 - alpha_mask)
        else:
            gt_for_loss = gt_image

        Ll1 = l1_loss(image_for_loss, gt_for_loss)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image_for_loss.unsqueeze(0), gt_for_loss.unsqueeze(0))
        else:
            ssim_value = ssim(image_for_loss, gt_for_loss)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Physical parameter statistics
            if iteration % 500 == 0:
                sigma_t = gaussians.get_extinction
                albedo = gaussians.get_albedo
                g = gaussians.get_g_factor
                sun_dir = gaussians.get_sun_dir

                def _ms(x):
                    x = x.detach()
                    return x.mean().item(), x.std(unbiased=False).item()

                m_sig, s_sig = _ms(sigma_t)
                m_alb, s_alb = _ms(albedo)
                m_g, s_g = _ms(g)

                print(
                    f"\n[ITER {iteration}] Physical stats | "
                    f"sigma_t mean/std: {m_sig:.4f}/{s_sig:.4f} | "
                    f"albedo mean/std: {m_alb:.4f}/{s_alb:.4f} | "
                    f"g mean/std: {m_g:.4f}/{s_g:.4f} | "
                    f"sun_dir: [{sun_dir[0].item():.3f}, {sun_dir[1].item():.3f}, {sun_dir[2].item():.3f}]"
                )
                # Diagnostic: T_light and Lk statistics to detect collapse
                if diag_T_light is not None:
                    m_tl, s_tl = _ms(diag_T_light)
                    print(f"  T_light mean/std: {m_tl:.4f}/{s_tl:.4f} | min: {diag_T_light.min().item():.6f}")
                if diag_Lk is not None:
                    m_lk, s_lk = _ms(diag_Lk)
                    print(f"  Lk mean/std: {m_lk:.4f}/{s_lk:.4f} | max: {diag_Lk.max().item():.4f}")

            if iteration % 1000 == 0:
                save_periodic_render(scene.model_path, iteration, image_for_loss, viewpoint_cam.image_name)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # After finishing training, compute metrics on test set and write run note.
            if iteration == opt.iterations:
                metrics = {}
                with torch.no_grad():
                    test_cams = scene.getTestCameras()
                    # Blender datasets merge test into train when eval=False. If so, explicitly load transforms_test.json.
                    if (test_cams is None) or (len(test_cams) == 0):
                        try:
                            depths_folder = os.path.join(dataset.source_path, dataset.depths) if dataset.depths != "" else ""
                            test_cam_infos = readCamerasFromTransforms(
                                dataset.source_path,
                                "transforms_test.json",
                                depths_folder,
                                dataset.white_background,
                                True,
                            )
                            test_cams = cameraList_from_camInfos(
                                test_cam_infos,
                                resolution_scale=1.0,
                                args=dataset,
                                is_nerf_synthetic=True,
                                is_test_dataset=True,
                            )
                        except Exception:
                            test_cams = []
                    if test_cams and len(test_cams) > 0:
                        ssims = []
                        psnrs = []
                        # LPIPS intentionally disabled (user request): avoid large backbone downloads.
                        for cam in tqdm(test_cams, desc="Final metric eval (test)"):
                            pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                            pred = pkg["render"].clamp(0.0, 1.0)
                            m = cam.alpha_mask.cuda().clamp(0.0, 1.0) if cam.alpha_mask is not None else None
                            if m is not None:
                                pred = pred * m + sky_board[:, None, None] * (1.0 - m)
                                gt = cam.original_image.cuda() * m + sky_board[:, None, None] * (1.0 - m)
                            else:
                                gt = cam.original_image.cuda()

                            ssims.append(ssim(pred, gt).item())
                            psnrs.append(psnr(pred, gt).mean().item())

                        metrics["test_psnr"] = float(sum(psnrs) / len(psnrs))
                        metrics["test_ssim"] = float(sum(ssims) / len(ssims))
                        metrics["test_lpips"] = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = build_timestamped_model_path()

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/sigma_t_histogram", scene.gaussians.get_extinction, iteration)
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
