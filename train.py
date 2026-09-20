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
from utils.camera_utils import CameraPrefetcher
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, pipe, opt)
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    # Learnable output tonemap: global coeffs in their own optimizer, isolated
    # from densify/prune; flag lives on the pipeline params.
    if getattr(pipe, "tonemap_learnable", False):
        gaussians.setup_tonemap(opt)
        print(f"[train] learnable tonemap enabled (lr={opt.tonemap_lr}, "
              f"init coeffs={gaussians.get_tonemap_coeffs.tolist()})")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

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

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

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

        # Anisotropy regularizer (log-ratio hinge), active while
        # iteration < opt.aniso_until_iter. `_scaling` is log-space, so
        # log(s_max) − log(s_min) = raw_max − raw_min.
        if iteration < opt.aniso_until_iter:
            raw_scaling = gaussians._scaling
            log_ratio = raw_scaling.max(dim=1).values - raw_scaling.min(dim=1).values
            log_threshold = math.log(opt.aniso_ratio_max)
            L_aniso = (log_ratio - log_threshold).clamp(min=0).pow(2).mean()
            loss += opt.lambda_aniso * L_aniso

        # Tonemap monotonicity regulariser (only with --tonemap_learnable):
        # penalizes negative slope of f on [0,8]; same form as L_aniso.
        if gaussians.tonemap_optimizer is not None and opt.lambda_tonemap_mono > 0:
            xs = torch.linspace(0.0, 8.0, 32, device="cuda")
            fs = gaussians.apply_tonemap(xs)
            dneg = (fs[:-1] - fs[1:]).clamp(min=0)   # >0 where f decreases
            L_mono = dneg.pow(2).mean()
            loss += opt.lambda_tonemap_mono * L_mono

        loss.backward()

        iter_end.record()

        # Release the just-decoded image tensor; cameras lazy-load (see
        # scene/cameras.py) and would otherwise hold it until next reuse.
        # release_loaded() drops the CPU cache entry (_loaded_rgb) only.
        if hasattr(viewpoint_cam, "release_loaded"):
            viewpoint_cam.release_loaded()

        with torch.no_grad():
            if iteration % 500 == 0:
                sigma_t = gaussians.get_sigma_t
                omega = gaussians.get_omega
                g = gaussians.get_g
                scales = gaussians.get_scaling
                # Prefer the current frame's per-frame v_l (from JSON) over the model-level fallback.
                if hasattr(viewpoint_cam, "v_l") and viewpoint_cam.v_l is not None:
                    v_l = viewpoint_cam.v_l
                else:
                    v_l = gaussians.get_v_l

                def _ms(x):
                    x = x.detach()
                    return x.mean().item(), x.std(unbiased=False).item()

                m_bp, s_bp = _ms(sigma_t)
                m_alb, s_alb = _ms(omega)
                m_g, s_g = _ms(g)
                m_scale, s_scale = _ms(scales)
                gscale = torch.pow(torch.prod(scales, dim=1) + 1e-8, 1.0 / 3.0)
                min_gscale = gscale.min().item()
                aniso_ratio = scales.max(dim=1).values / scales.min(dim=1).values.clamp(min=1e-6)
                m_an, s_an = aniso_ratio.mean().item(), aniso_ratio.std(unbiased=False).item()
                p99_an = torch.quantile(aniso_ratio, 0.99).item()

                print(
                    f"\n [ITER {iteration}] Physical stats | "
                    f"\n sigma_t mean/std: {m_bp:.4f}/{s_bp:.4f} | "
                    f"omega mean/std: {m_alb:.4f}/{s_alb:.4f} | "
                    f"g mean/std: {m_g:.4f}/{s_g:.4f}"
                    f"\n scale mean/std: {m_scale:.4f}/{s_scale:.4f} | "
                    f"gscale min: {min_gscale:.6f} | "
                    f"aniso mean/std/p99: {m_an:.2f}/{s_an:.2f}/{p99_an:.2f} | "
                    f"v_l: [{v_l[0].item():.3f}, {v_l[1].item():.3f}, {v_l[2].item():.3f}]"
                )
                if gaussians.tonemap_optimizer is not None:
                    tm = gaussians.get_tonemap_coeffs.detach().tolist()
                    can = list(GaussianModel.TONEMAP_CANONICAL)
                    print(f" tonemap coeffs (a,b,c,d): "
                          f"[{tm[0]:.4f}, {tm[1]:.4f}, {tm[2]:.4f}, {tm[3]:.4f}] "
                          f"(canonical [{can[0]:.2f}, {can[1]:.2f}, {can[2]:.2f}, {can[3]:.2f}])")

            if iteration % 1000 == 0:
                save_periodic_render(scene.model_path, iteration, image_for_loss, viewpoint_cam.image_name)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), final_iteration=opt.iterations)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Per-Gaussian Σ(α·T) accumulator; always-on, not gated by densify_until_iter.
            contribution = render_pkg.get("contribution")
            if contribution is not None:
                gaussians.add_contribution_stats(contribution)

            if iteration < opt.densify_until_iter:
                # Track per-Gaussian max screen-space radius (radii) while densifying.
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.physical_densify_and_prune(opt, iteration, radii, scene.cameras_extent)

            # Densify-window maintenance: resurrect + prune + accumulator reset (no-op
            # once iteration >= densify_until_iter). Must run AFTER densification, or
            # prune shrinks P and this iteration's indices (visibility_filter / radii) go stale.
            gaussians.tick_post_densify_maintenance(opt, iteration)

            # Needle surgery: hard ceiling on the aniso tail. Placed after all other
            # structure changes so the optimizer step below sees consistent tensors.
            needle_iv = getattr(opt, "needle_split_interval", 0)
            if (needle_iv > 0 and iteration % needle_iv == 0
                    and iteration <= getattr(opt, "needle_split_until_iter", opt.iterations)):
                n_split = gaussians.split_needles(
                    getattr(opt, "needle_split_ratio", 30.0), opt)
                if n_split > 0:
                    print(f"\n[ITER {iteration}] needle surgery: split {n_split} (ratio > {opt.needle_split_ratio})")

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # Step the standalone tonemap optimizer (gradients flowed in
                # via the shared loss.backward()). No-op unless --tonemap_learnable.
                if gaussians.tonemap_optimizer is not None:
                    gaussians.tonemap_optimizer.step()
                    gaussians.tonemap_optimizer.zero_grad(set_to_none = True)

def _resolve_stage1_ply(stage1_model):
    """Resolve a Stage-1 model path (output dir, iteration dir, or .ply) to a point_cloud.ply."""
    from utils.system_utils import searchForMaxIteration
    if not stage1_model:
        sys.exit("--stage2 requires --stage1_model (Stage-1 output dir or point_cloud.ply).")
    if stage1_model.endswith(".ply"):
        return stage1_model
    pc_dir = os.path.join(stage1_model, "point_cloud")
    if os.path.isdir(pc_dir):
        it = searchForMaxIteration(pc_dir)
        cand = os.path.join(pc_dir, f"iteration_{it}", "point_cloud.ply")
        if os.path.exists(cand):
            return cand
    cand = os.path.join(stage1_model, "point_cloud.ply")
    if os.path.exists(cand):
        return cand
    sys.exit(f"Could not find a point_cloud.ply under {stage1_model}")


def _stage2_env_ref_dirs():
    """Reference sun directions (OpenGL world, up=+Y) for env diagnostics."""
    out = []
    for name, el, az in [("zen~80", 80, 0), ("low~15/0", 15, 0), ("low~15/120", 15, 120)]:
        e, a = math.radians(el), math.radians(az)
        out.append((name, torch.tensor([math.cos(e) * math.cos(a), math.sin(e),
                                        math.cos(e) * math.sin(a)], device="cuda")))
    return out


def _log_env_params(gaussians, tag=""):
    """Print the learned per-channel zenith optical depth τ (the 3 T_sun params) and
    T_sun / sky-DC at a high and two low sun elevations."""
    if gaussians.env_net is None:
        return
    with torch.no_grad():
        tau = gaussians.env_net.tau
        print(f"[stage2 env{(' ' + tag) if tag else ''}] "
              f"tau(R,G,B)=[{tau[0]:.4f},{tau[1]:.4f},{tau[2]:.4f}]")
        parts = []
        for name, d in _stage2_env_ref_dirs():
            t_sun, e_lm = gaussians.env_net(d)
            dc = e_lm[0]
            parts.append(f"{name} T_sun=[{t_sun[0]:.3f},{t_sun[1]:.3f},{t_sun[2]:.3f}] "
                         f"E00=[{dc[0]:.2f},{dc[1]:.2f},{dc[2]:.2f}]")
        print(f"[stage2 env{(' ' + tag) if tag else ''}] " + " || ".join(parts))


def _stage2_eval(scene, gaussians, pipe, background, source_path, env_on=True,
                 with_lpips=False, desc="eval"):
    """Eval the test split; split PSNR into held-out-sun vs seen-sun (relighting gap).
    Temporarily forces pipe.env_lighting=env_on so the env-off baseline can be measured.
    Held-out suns = test time_index values absent from train; fallback {7,22,37,52}."""
    time_by_key = {}
    test_suns = set()
    tj_path = os.path.join(source_path, "transforms_test.json")
    if os.path.exists(tj_path):
        with open(tj_path) as f:
            tj = json.load(f)
        for fr in tj.get("frames", []):
            parts = fr["file_path"].replace("\\", "/").split("/")
            ti = int(fr.get("time_index", -1))
            time_by_key[(parts[0], os.path.splitext(parts[-1])[0])] = ti
            test_suns.add(ti)
    train_path = os.path.join(source_path, "transforms_train.json")
    if os.path.exists(train_path):
        with open(train_path) as f:
            train_suns = {int(fr.get("time_index", -1)) for fr in json.load(f).get("frames", [])}
        HELDOUT = test_suns - train_suns
    else:
        HELDOUT = {7, 22, 37, 52}
    test_cams = scene.getTestCameras()
    if not test_cams or len(test_cams) == 0:
        return None
    prev = pipe.env_lighting
    pipe.env_lighting = env_on
    lpips_fn = get_lpips_fn() if with_lpips else None
    psnrs, ssims, lps = [], [], []
    grp = {"heldout": [], "seen": []}
    with torch.no_grad():
        for cam in tqdm(test_cams, desc=desc, leave=False):
            pred = render(cam, gaussians, pipe, background)["render"].clamp(0.0, 1.0)
            gt = cam.original_image.cuda()
            psnrs.append(psnr(pred, gt).mean().item())
            ssims.append(ssim(pred, gt).item())
            if lpips_fn is not None:
                lps.append(lpips_fn(pred, gt))
            ip = cam.image_path.replace("\\", "/").split("/")
            ti = time_by_key.get((ip[-3], os.path.splitext(ip[-1])[0]), -1)
            grp["heldout" if ti in HELDOUT else "seen"].append(psnrs[-1])
            if hasattr(cam, "release_loaded"):
                cam.release_loaded()
    pipe.env_lighting = prev
    avg = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {"psnr": avg(psnrs), "ssim": avg(ssims), "lpips": avg(lps),
            "heldout_sun_psnr": avg(grp["heldout"]), "seen_sun_psnr": avg(grp["seen"]),
            "n_heldout": len(grp["heldout"]), "n_seen": len(grp["seen"])}


def _fmt_eval(r):
    ho, se = r["heldout_sun_psnr"], r["seen_sun_psnr"]
    gap = f"{ho - se:+.3f}" if (ho is not None and se is not None) else "n/a"
    lp = f"{r['lpips']:.4f}" if r["lpips"] is not None else "n/a"
    ho_s = f"{ho:.3f}" if ho is not None else "n/a"
    se_s = f"{se:.3f}" if se is not None else "n/a"
    return (f"PSNR {r['psnr']:.3f} | SSIM {r['ssim']:.4f} | LPIPS {lp} | "
            f"held-out sun {ho_s} ({r['n_heldout']}) vs seen {se_s} ({r['n_seen']}) gap {gap}")


def training_stage2(dataset, opt, pipe, testing_iterations, saving_iterations, stage1_model):
    """Stage 2: load & FREEZE a Stage-1 model and train ONLY the global environment-
    lighting net (T_sun + E_lm of v_l) on the env-on dataset (-s). Pure-black
    background, full-image supervision (no mask: bg is 0 in both GT and render).
    The Stage-1 output is left untouched; results go to this run's -m / model_path."""
    pipe.env_lighting = True
    stage1_ply = _resolve_stage1_ply(stage1_model)
    print(f"[stage2] freezing Stage-1 model: {stage1_ply}")

    tb_writer = prepare_output_and_logger(dataset, pipe, opt,
                                          extra={"stage2": True, "stage1_model": stage1_model})

    gaussians = GaussianModel()
    gaussians.load_ply(stage1_ply)
    # Freeze ALL per-Gaussian params — Stage 2 optimises only the global EnvNet.
    for p in (gaussians._xyz, gaussians._sigma_t, gaussians._omega,
              gaussians._g, gaussians._w,
              gaussians._scaling, gaussians._rotation):
        p.requires_grad_(False)

    # env-on cameras + GT; keep the frozen gaussians (don't re-init from points3d).
    scene = Scene(dataset, gaussians, init_gaussians=False)

    # Precompute per-Gaussian sky-visibility transfer once, then build the env net.
    gaussians.precompute_sky_transfer(n_dirs=getattr(pipe, "env_transfer_dirs", 48),
                                      sh_order=getattr(pipe, "env_sh_order", 2))
    gaussians.setup_env(opt, sh_order=getattr(pipe, "env_sh_order", 2))
    print(f"[stage2] env net params: {sum(p.numel() for p in gaussians.env_net.parameters())} "
          f"(env_lr={opt.env_lr}); pure-black bg, full-image supervision (no mask)")

    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")  # pure black
    prefetcher = CameraPrefetcher(scene, queue_size=2)
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Stage2 (env) progress")
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_env_learning_rate(iteration)
        viewpoint_cam = prefetcher.next()

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        gt = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss.backward()
        gaussians.env_optimizer.step()
        gaussians.env_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.5f}"})
                progress_bar.update(10)
            if tb_writer:
                tb_writer.add_scalar("stage2/total_loss", loss.item(), iteration)
            if iteration in saving_iterations:
                print(f"\n[stage2 ITER {iteration}] Saving (PLY + env sidecar)")
                scene.save(iteration)
            if (iteration in testing_iterations) and (iteration != opt.iterations):
                r = _stage2_eval(scene, gaussians, pipe, background, dataset.source_path,
                                 env_on=True, with_lpips=False, desc=f"eval@{iteration}")
                if r:
                    print(f"\n[stage2 ITER {iteration}] {_fmt_eval(r)}")
                _log_env_params(gaussians, tag=f"@{iteration}")
        if hasattr(viewpoint_cam, "release_loaded"):
            viewpoint_cam.release_loaded()
    progress_bar.close()

    # Final eval: env-on (held-out vs seen split) + env-off baseline (env contribution).
    _log_env_params(gaussians, tag="final")
    on = _stage2_eval(scene, gaussians, pipe, background, dataset.source_path,
                      env_on=True, with_lpips=True, desc="Stage2 final eval (env-on)")
    off = _stage2_eval(scene, gaussians, pipe, background, dataset.source_path,
                       env_on=False, with_lpips=False, desc="Stage2 final eval (env-off)")
    if on is not None:
        gap = (on["heldout_sun_psnr"] - on["seen_sun_psnr"]) \
            if (on["heldout_sun_psnr"] is not None and on["seen_sun_psnr"] is not None) else None
        contrib = (on["psnr"] - off["psnr"]) if off else None
        metrics = {
            "test_psnr": on["psnr"], "test_ssim": on["ssim"], "test_lpips": on["lpips"],
            "heldout_sun_psnr": on["heldout_sun_psnr"], "seen_sun_psnr": on["seen_sun_psnr"],
            "relighting_gap_db": gap,
            "env_off_psnr": off["psnr"] if off else None,
            "env_contribution_db": contrib,
        }
        with open(os.path.join(scene.model_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[stage2 Final] env-on  {_fmt_eval(on)}")
        if off is not None:
            print(f"[stage2 Final] env-off PSNR {off['psnr']:.3f}  →  env contributes "
                  f"{contrib:+.3f} dB (env-on minus env-off; ~0 ⇒ env-on≈env-off / atmosphere effect tiny)")

def prepare_output_and_logger(args, pipe=None, opt=None, extra=None):
    if not args.model_path:
        args.model_path = build_timestamped_model_path()

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # Persist all params alongside the model: ModelParams + PipelineParams (the viewer
    # auto-reads these for --tlight/tonemap/env) + OptimizationParams + extras (e.g. stage2).
    merged = dict(vars(args))
    if pipe is not None:
        merged.update(vars(pipe))
    if opt is not None:
        merged.update(vars(opt))
    if extra is not None:
        merged.update(extra)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**merged)))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, final_iteration=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

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
                lpips_avg = (lpips_test / lpips_count) if lpips_count > 0 else None
                lpips_str = f" LPIPS {lpips_avg:.4f}" if lpips_avg is not None else ""
                print("\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.3f} SSIM {:.4f}{}".format(
                    iteration, config['name'], float(l1_test), float(psnr_test), ssim_test, lpips_str))
                # Persist the final test metrics to metrics.json (this pass is the final eval).
                if config['name'] == 'test' and iteration == final_iteration:
                    metrics = {"test_psnr": float(psnr_test), "test_ssim": float(ssim_test),
                               "test_lpips": lpips_avg}
                    with open(os.path.join(scene.model_path, "metrics.json"), "w") as f:
                        json.dump(metrics, f, indent=2)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    if lpips_count > 0:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips',
                                             lpips_test / lpips_count, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/sigma_t_histogram", scene.gaussians.get_sigma_t, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--stage2", action="store_true", default=False,
                        help="Stage 2: load & freeze a Stage-1 model and train ONLY the "
                             "environment-lighting net (global T_sun + E_lm of v_l). "
                             "-s points to the env-on dataset.")
    parser.add_argument("--stage1_model", type=str, default="",
                        help="Stage 2: path to the Stage-1 output dir (or a point_cloud.ply) "
                             "to load and freeze. Its result is left untouched.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.stage2:
        training_stage2(lp.extract(args), op.extract(args), pp.extract(args),
                        args.test_iterations, args.save_iterations, args.stage1_model)
    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    print("\nTraining complete.")

