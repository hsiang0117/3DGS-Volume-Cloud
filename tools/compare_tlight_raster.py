#!/usr/bin/env python
"""Smoke test: compare raster T_light (light-space shadow pass) against the
voxel-cache T_light on a trained ply, for several sun directions.

Checks:
  1. The new record_front_tau channel returns sane buffers (no NaN/inf).
  2. T_light_raster is positively correlated with T_light_voxel (they
     approximate the same physical quantity).
  3. No inverted shadows: deeply buried Gaussians (voxel says dark) must not
     read bright in the raster pass (the wsum==0 fallback works).
  4. Timing per call.

Usage: .venv/Scripts/python.exe tools/compare_tlight_raster.py [ply_path]
"""
import sys
import time
import math
import torch

sys.path.insert(0, ".")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import (
    compute_T_light, compute_T_light_raster, normalized_gaussian_line_integral,
)
from utils.general_utils import build_rotation


def main(ply_path):
    gaussians = GaussianModel("default")
    gaussians.load_ply(ply_path)
    means3D = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    rotations = gaussians.get_rotation.detach()
    beta_peak = gaussians.get_extinction.detach()
    P = means3D.shape[0]
    print(f"loaded {P} gaussians from {ply_path}")

    R_t = build_rotation(rotations).transpose(1, 2)
    mass = beta_peak * ((2.0 * math.pi) ** 1.5) * torch.prod(scales, dim=1, keepdim=True)

    sun_dirs = [
        [0.0, 1.0, 0.0],          # zenith
        [0.0, 0.669, 0.743],      # mid elevation (dataset-like)
        [0.0, 0.105, -0.995],     # near horizon
    ]
    for sd in sun_dirs:
        L_dir = torch.tensor(sd, device="cuda", dtype=means3D.dtype)
        L_dir = L_dir / L_dir.norm()
        l_local = torch.matmul(R_t, L_dir.view(3, 1)).squeeze(-1)
        tau_sun = mass * normalized_gaussian_line_integral(scales, l_local)

        torch.cuda.synchronize(); t0 = time.time()
        T_vox = compute_T_light(means3D, tau_sun, scales, L_dir, grid_res=128)
        torch.cuda.synchronize(); t_vox = time.time() - t0

        torch.cuda.synchronize(); t0 = time.time()
        T_ras = compute_T_light_raster(means3D, tau_sun, scales, rotations, L_dir)
        torch.cuda.synchronize(); t_ras = time.time() - t0

        T_vox = T_vox.squeeze(-1)
        T_ras = T_ras.squeeze(-1).squeeze(-1) if T_ras.dim() > 1 else T_ras.squeeze()
        T_ras = T_ras.view(-1)
        assert T_ras.shape[0] == P, f"shape mismatch {T_ras.shape}"
        assert torch.isfinite(T_ras).all(), "NaN/inf in raster T_light"

        corr = torch.corrcoef(torch.stack([T_vox, T_ras]))[0, 1].item()
        # Inverted-shadow check: of the darkest 5% per voxel path, how many
        # does the raster path call bright (>0.5)?
        k = max(1, P // 20)
        dark_idx = torch.topk(T_vox, k, largest=False).indices
        inverted = (T_ras[dark_idx] > 0.5).float().mean().item()

        def stats(name, T):
            print(f"  {name}: mean {T.mean():.4f} | std {T.std():.4f} | "
                  f"min {T.min():.6f} | frac<0.01 {(T < 0.01).float().mean():.4f} | "
                  f"frac>0.99 {(T > 0.99).float().mean():.4f}")

        print(f"\nsun_dir {sd}  (voxel {t_vox*1e3:.1f} ms, raster {t_ras*1e3:.1f} ms)")
        stats("voxel ", T_vox)
        stats("raster", T_ras)
        print(f"  corr(voxel, raster) = {corr:.4f} | "
              f"inverted-shadow frac (dark5% voxel -> T_ras>0.5) = {inverted:.4f}")


if __name__ == "__main__":
    ply = sys.argv[1] if len(sys.argv) > 1 else \
        "output/20260610_183436/point_cloud/iteration_30000/point_cloud.ply"
    main(ply)
