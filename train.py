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
import json
import torch
import math
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.system_utils import build_timestamped_model_path
from utils.image_utils import psnr, save_periodic_render, get_lpips_fn
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos, CameraPrefetcher
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, pipe)
    gaussians = GaussianModel(opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    # Learnable output tonemap: a handful of global coeffs in their OWN optimizer
    # (isolated from densify/prune). The flag lives on the pipeline params.
    if getattr(pipe, "tonemap_learnable", False):
        gaussians.setup_tonemap(opt)
        print(f"[train] learnable tonemap enabled (lr={opt.tonemap_lr}, "
              f"init coeffs={gaussians.get_tonemap_coeffs.tolist()})")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    prefetcher = CameraPrefetcher(scene, queue_size=2)
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_tonemap_learning_rate(iteration)  # no-op unless learnable tonemap

        # Pick a random Camera (image already prefetched by background worker)
        viewpoint_cam = prefetcher.next()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Grab diagnostic tensors injected by render()
        diag_T_light = render_pkg.get("T_light")
        diag_Lk = render_pkg.get("Lk")

        image_for_loss = image
        gt_for_loss = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image_for_loss, gt_for_loss)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image_for_loss.unsqueeze(0), gt_for_loss.unsqueeze(0))
        else:
            ssim_value = ssim(image_for_loss, gt_for_loss)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        scales = gaussians.get_scaling
        L_vol = (scales.prod(dim=1)).mean()
        loss += opt.lambda_scale * L_vol

        # Anisotropy regularizer (log-ratio form, gentle).
        #
        # `_scaling` is in log-space (scaling_activation = exp), so
        # `log(s_max) − log(s_min) = raw_max − raw_min` is free.
        #
        # Note: bounding aniso aggressively kills PSNR (cloud needs some
        # elongation to fit wisps/layers at current capacity). λ is kept
        # small so the regulariser only nudges the worst tail.
        # Disabled after densify ends to avoid uniform shrinking under
        # combined L_vol pressure.
        if iteration < opt.aniso_until_iter:
            raw_scaling = gaussians._scaling
            log_ratio = raw_scaling.max(dim=1).values - raw_scaling.min(dim=1).values
            log_threshold = math.log(opt.aniso_ratio_max)
            L_aniso = (log_ratio - log_threshold).clamp(min=0).pow(2).mean()
            loss += opt.lambda_aniso * L_aniso

        # Tonemap monotonicity regulariser (only with --tonemap_learnable).
        # softplus already guarantees positivity / no poles, but does not by
        # itself forbid a non-monotone S-curve; this hinges on any negative
        # slope of f on [0,8] so highlights can never invert. Cheap insurance,
        # same hinge-squared form as L_aniso.
        if gaussians.tonemap_optimizer is not None and opt.lambda_tonemap_mono > 0:
            xs = torch.linspace(0.0, 8.0, 32, device="cuda")
            fs = gaussians.apply_tonemap(xs)
            dneg = (fs[:-1] - fs[1:]).clamp(min=0)   # >0 where f decreases
            L_mono = dneg.pow(2).mean()
            loss += opt.lambda_tonemap_mono * L_mono

        loss.backward()

        iter_end.record()

        # Release the just-decoded image/alpha tensors. Camera is lazy-loading
        # under our refactor (see scene/cameras.py); without this each cam
        # would hold ~12 MB of GPU memory until next reuse, which on
        # 2989-frame datasets matters again.
        if hasattr(viewpoint_cam, "release_loaded"):
            viewpoint_cam.release_loaded()

        with torch.no_grad():
            # Physical parameter statistics
            if iteration % 500 == 0:
                beta_peak = gaussians.get_extinction
                albedo = gaussians.get_albedo
                g = gaussians.get_g_factor
                scales = gaussians.get_scaling
                # Prefer the *current frame's* sun_dir (per-frame, from JSON)
                # over the model-level fallback so the diag reflects what the
                # renderer actually used this iteration.
                if hasattr(viewpoint_cam, "sun_dir") and viewpoint_cam.sun_dir is not None:
                    sun_dir = viewpoint_cam.sun_dir
                else:
                    sun_dir = gaussians.get_sun_dir

                def _ms(x):
                    x = x.detach()
                    return x.mean().item(), x.std(unbiased=False).item()

                m_bp, s_bp = _ms(beta_peak)
                m_alb, s_alb = _ms(albedo)
                m_g, s_g = _ms(g)
                m_scale, s_scale = _ms(scales)
                gscale = torch.pow(torch.prod(scales, dim=1) + 1e-8, 1.0 / 3.0)
                min_gscale = gscale.min().item()
                aniso_ratio = scales.max(dim=1).values / scales.min(dim=1).values.clamp(min=1e-6)
                m_an, s_an = aniso_ratio.mean().item(), aniso_ratio.std(unbiased=False).item()
                p99_an = torch.quantile(aniso_ratio, 0.99).item()

                print(
                    f"\n [ITER {iteration}] Physical stats | "
                    f"\n beta_peak mean/std: {m_bp:.4f}/{s_bp:.4f} | "
                    f"albedo mean/std: {m_alb:.4f}/{s_alb:.4f} | "
                    f"g mean/std: {m_g:.4f}/{s_g:.4f}"
                    f"\n scale mean/std: {m_scale:.4f}/{s_scale:.4f} | "
                    f"gscale min: {min_gscale:.6f} | "
                    f"aniso mean/std/p99: {m_an:.2f}/{s_an:.2f}/{p99_an:.2f} | "
                    f"sun_dir: [{sun_dir[0].item():.3f}, {sun_dir[1].item():.3f}, {sun_dir[2].item():.3f}]"
                )
                if gaussians.tonemap_optimizer is not None:
                    tm = gaussians.get_tonemap_coeffs.detach().tolist()
                    can = list(GaussianModel.TONEMAP_CANONICAL)
                    print(f" tonemap coeffs (a,b,c,d): "
                          f"[{tm[0]:.4f}, {tm[1]:.4f}, {tm[2]:.4f}, {tm[3]:.4f}] "
                          f"(canonical [{can[0]:.2f}, {can[1]:.2f}, {can[2]:.2f}, {can[3]:.2f}])")
                # Diagnostic: T_light and Lk statistics to detect collapse
                if diag_T_light is not None:
                    m_tl, s_tl = _ms(diag_T_light)
                    print(f"  T_light mean/std: {m_tl:.4f}/{s_tl:.4f} | min: {diag_T_light.min().item():.6f}")
                if diag_Lk is not None:
                    m_lk, s_lk = _ms(diag_Lk)
                    print(f"  Lk mean/std: {m_lk:.4f}/{s_lk:.4f} | max: {diag_Lk.max().item():.4f}")

                # Densify diagnostic: how many points actually qualify for
                # densification, and how many are in danger of being pruned.
                # If `n_above_grad` is consistently small (<<1% of n_points)
                # the densify_grad_threshold is too high for the current
                # parameterisation. If `n_below_prune` ≈ densify net gain per
                # round, the cloud is in equilibrium (every new point is
                # immediately pruned next round).
                denom_d = gaussians.denom.clamp(min=1)
                grads_xyz = (gaussians.xyz_gradient_accum / denom_d).squeeze()
                grads_xyz = torch.where(torch.isnan(grads_xyz), torch.zeros_like(grads_xyz), grads_xyz)
                n_points = gaussians.get_xyz.shape[0]
                n_above = int((grads_xyz >= opt.densify_grad_threshold).sum().item())
                op = gaussians.get_opacity.squeeze()
                # Split path: needs scale_max > percent_dense * scene_extent
                # Clone path: needs scale_max <= percent_dense * scene_extent
                scale_max = scales.max(dim=1).values
                scene_extent = scene.cameras_extent
                clone_gate = scale_max <= opt.percent_dense * scene_extent
                split_gate = scale_max > opt.percent_dense * scene_extent
                grad_pass = grads_xyz >= opt.densify_grad_threshold
                n_clone_eligible = int((grad_pass & clone_gate).sum().item())
                n_split_eligible = int((grad_pass & split_gate).sum().item())
                # Contribution-based prune signal (active under physical strategy)
                contrib_str = ""
                if hasattr(gaussians, "contribution_accum") and gaussians.contribution_accum.numel() == n_points:
                    mean_contrib = gaussians.get_mean_contribution()
                    n_visible = int((gaussians.contribution_denom >= getattr(opt, "prune_min_visible_frames", 5)).sum().item())
                    n_below_contrib = int(
                        ((mean_contrib < getattr(opt, "contribution_threshold", 1e-4)) &
                         (gaussians.contribution_denom >= getattr(opt, "prune_min_visible_frames", 5)) &
                         (gaussians.prune_grace == 0)).sum().item()
                    )
                    n_dead = int(
                        ((gaussians.contribution_denom == 0) &
                         (gaussians.prune_grace == 0)).sum().item()
                    )
                    contrib_str = (
                        f" | contrib mean/median: {mean_contrib.mean().item():.4f}/{mean_contrib.median().item():.4f} | "
                        f"visible frames p50: {gaussians.contribution_denom.median().item():.0f} | "
                        f"n_below_contrib: {n_below_contrib} | "
                        f"n_dead: {n_dead}"
                    )
                print(
                    f"  densify diag: n_points={n_points} | "
                    f"grad max/mean: {grads_xyz.max().item():.6f}/{grads_xyz.mean().item():.6f} | "
                    f"n_above_thresh: {n_above} ({100.0*n_above/max(n_points,1):.2f}%) | "
                    f"clone-eligible: {n_clone_eligible} | split-eligible: {n_split_eligible} | "
                    f"opacity min/median: {op.min().item():.6f}/{op.median().item():.6f}"
                    f"{contrib_str}"
                )

            if iteration % 1000 == 0:
                save_periodic_render(scene.model_path, iteration, image_for_loss, viewpoint_cam.image_name)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Per-Gaussian image-contribution accumulator. Always-on (decoupled
            # from densify_until_iter) so the diagnostic / future heuristics see
            # current state, not a frozen snapshot from when densify ended.
            contribution = render_pkg.get("contribution")
            if contribution is not None:
                gaussians.add_contribution_stats(contribution)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.physical_densify_and_prune(opt, iteration, radii, scene.cameras_extent)

            # Post-densify maintenance: resurrect + prune + accumulator reset.
            # Must come AFTER the densification block — otherwise prune in tick
            # shrinks P and the per-step indices (visibility_filter / radii)
            # from this iteration's forward pass become stale.
            gaussians.tick_post_densify_maintenance(opt, iteration)

            # Needle surgery: structural hard ceiling on the aniso tail.
            # Placed after all other structure changes in this tick so the
            # optimizer step below sees consistent tensors. Runs through the
            # whole schedule — needles regrow from the contrast-compression
            # pressure, so a densify-phase-only pass would unravel by 30k.
            needle_iv = getattr(opt, "needle_split_interval", 0)
            if (needle_iv > 0 and iteration % needle_iv == 0
                    and iteration <= getattr(opt, "needle_split_until_iter", opt.iterations)):
                n_split = gaussians.split_needles(
                    getattr(opt, "needle_split_ratio", 30.0), opt)
                if n_split > 0:
                    print(f"\n[ITER {iteration}] needle surgery: split {n_split} (ratio > {opt.needle_split_ratio})")

            # Optimizer step
            if iteration < opt.iterations:
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                # Step the standalone tonemap optimizer (gradients flowed in via
                # the shared loss.backward()). No-op unless --tonemap_learnable.
                if gaussians.tonemap_optimizer is not None:
                    gaussians.tonemap_optimizer.step()
                    gaussians.tonemap_optimizer.zero_grad(set_to_none = True)

            # After finishing training, compute metrics on test set and write run note.
            if iteration == opt.iterations:
                metrics = {}
                with torch.no_grad():
                    test_cams = scene.getTestCameras()
                    # Blender datasets merge test into train when eval=False. If so, explicitly load transforms_test.json.
                    if (test_cams is None) or (len(test_cams) == 0):
                        try:
                            test_cam_infos = readCamerasFromTransforms(
                                dataset.source_path,
                                "transforms_test.json",
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
                        lpips_vals = []
                        lpips_fn = get_lpips_fn()  # None if package missing
                        for cam in tqdm(test_cams, desc="Final metric eval (test)"):
                            pkg = render(cam, gaussians, pipe, background, separate_sh=SPARSE_ADAM_AVAILABLE)
                            pred = pkg["render"].clamp(0.0, 1.0)
                            gt = cam.original_image.cuda()

                            ssims.append(ssim(pred, gt).item())
                            psnrs.append(psnr(pred, gt).mean().item())
                            if lpips_fn is not None:
                                lpips_vals.append(lpips_fn(pred, gt))

                            if hasattr(cam, "release_loaded"):
                                cam.release_loaded()

                        metrics["test_psnr"] = float(sum(psnrs) / len(psnrs))
                        metrics["test_ssim"] = float(sum(ssims) / len(ssims))
                        metrics["test_lpips"] = (
                            float(sum(lpips_vals) / len(lpips_vals)) if lpips_vals else None
                        )

                        # Persist final metrics and surface them on stdout so
                        # an end-of-run grep can pick up results without
                        # parsing tqdm output.
                        metrics_path = os.path.join(scene.model_path, "metrics.json")
                        with open(metrics_path, "w") as f:
                            json.dump(metrics, f, indent=2)
                        lpips_str = (
                            f"{metrics['test_lpips']:.4f}"
                            if metrics["test_lpips"] is not None else "n/a"
                        )
                        print(
                            f"\n[Final] test PSNR {metrics['test_psnr']:.3f} | "
                            f"SSIM {metrics['test_ssim']:.4f} | "
                            f"LPIPS {lpips_str} → {metrics_path}"
                        )

# git checkout test
def prepare_output_and_logger(args, pipe=None):
    if not args.model_path:
        args.model_path = build_timestamped_model_path()

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # Persist pipeline params alongside model params: the viewer's
    # --tlight auto reads the tlight_* flags from cfg_args to pick the
    # matching T_light source, and ModelParams alone doesn't contain them.
    merged = dict(vars(args))
    if pipe is not None:
        merged.update(vars(pipe))
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**merged)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        lpips_fn = get_lpips_fn()  # None if package missing
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                lpips_count = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += float(ssim(image, gt_image).item())
                    if lpips_fn is not None:
                        lpips_test += lpips_fn(image, gt_image)
                        lpips_count += 1
                    if hasattr(viewpoint, "release_loaded"):
                        viewpoint.release_loaded()
                n_cams = len(config['cameras'])
                psnr_test /= n_cams
                l1_test /= n_cams
                ssim_test /= n_cams
                lpips_str = (
                    f" LPIPS {lpips_test / lpips_count:.4f}"
                    if lpips_count > 0 else ""
                )
                print("\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.3f} SSIM {:.4f}{}".format(
                    iteration, config['name'], float(l1_test), float(psnr_test), ssim_test, lpips_str))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    if lpips_count > 0:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips',
                                             lpips_test / lpips_count, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/beta_peak_histogram", scene.gaussians.get_extinction, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
