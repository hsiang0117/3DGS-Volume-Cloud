"""Standalone test-set eval for an existing run — applies train.py's exact
eval computation (training_report's test loop) to a saved checkpoint.

train.py only runs / persists the test metrics when the final iteration is in
--test_iterations; this script runs that same pass on demand and writes the
same metrics.json. Reads source_path / T_light / tonemap / env from the run's
cfg_args so rendering matches how the model was trained. Test cameras come from
the dataset's transforms_test.json.

Usage:
    python tools/eval_testset.py output/<run> [iteration]
"""
import sys, os, re, json, argparse
sys.path.insert(0, '.')
import torch
from argparse import Namespace

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim          # same funcs as train.py
from utils.image_utils import psnr, get_lpips_fn    # same funcs as train.py

ap = argparse.ArgumentParser(description="Standalone test-set eval (reuses train.py logic).")
ap.add_argument("run", help="training output dir (cfg_args + point_cloud/)")
ap.add_argument("iteration", nargs="?", type=int, default=None,
                help="checkpoint iteration; default = newest")
cli = ap.parse_args()

run = cli.run
pc_dir = os.path.join(run, "point_cloud")
iteration = cli.iteration if cli.iteration is not None else searchForMaxIteration(pc_dir)
ply = os.path.join(pc_dir, f"iteration_{iteration}", "point_cloud.ply")

# --- cfg_args: dataset + render config (match training) -----------------
cfg = open(os.path.join(run, "cfg_args")).read()
m = re.search(r"source_path=['\"]([^'\"]+)['\"]", cfg)
source_path = m.group(1) if m else r"data/CloudDatasetZenith"
use_raster = ("tlight_voxel=True" not in cfg) if "tlight_voxel" in cfg else ("tlight_raster=True" in cfg)
mres = re.search(r"tlight_raster_res=(\d+)", cfg)
raster_res = int(mres.group(1)) if mres else 512

g = GaussianModel()
g.load_ply(ply)
env_lighting = (g.env_net is not None) and (g._sky_transfer.numel() > 0)
pipe = Namespace(tlight_voxel=not use_raster, tlight_raster_res=raster_res,
                 tlight_tau_filter="tlight_tau_filter=True" in cfg,
                 tlight_full_grad="tlight_full_grad=True" in cfg,
                 tonemap_aces="tonemap_aces=True" in cfg,
                 tonemap_learnable="tonemap_learnable=True" in cfg,
                 env_lighting=env_lighting, env_sh_order=g.env_sh_order)
background = torch.zeros(3, device="cuda")
print(f"run={run} iter={iteration} | source={source_path}")
print(f"T_light={'raster' if use_raster else 'voxel'} | aces={pipe.tonemap_aces} | env={env_lighting}")

# --- test cameras from transforms_test.json -----------------------------
cam_infos = readCamerasFromTransforms(source_path, "transforms_test.json", False, True)
cams = cameraList_from_camInfos(cam_infos, 1.0, Namespace(resolution=-1, data_device="cuda"), True, True)
if not cams:
    print("no test cameras found (transforms_test.json empty/missing)"); sys.exit(1)

# --- SAME eval computation as training_report in train.py ----------------
lpips_fn = get_lpips_fn()
l1_test = psnr_test = ssim_test = 0.0
lpips_test = 0.0; lpips_count = 0
with torch.no_grad():
    for viewpoint in cams:
        image = torch.clamp(render(viewpoint, g, pipe, background)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        ssim_test += float(ssim(image, gt_image).item())
        if lpips_fn is not None:
            lpips_test += lpips_fn(image, gt_image); lpips_count += 1
        if hasattr(viewpoint, "release_loaded"):
            viewpoint.release_loaded()
n = len(cams)
l1_test /= n; psnr_test /= n; ssim_test /= n
lpips_avg = (lpips_test / lpips_count) if lpips_count > 0 else None
lpips_str = f" LPIPS {lpips_avg:.4f}" if lpips_avg is not None else ""
print("\n[ITER {}] Evaluating test ({} views): L1 {:.6f} PSNR {:.3f} SSIM {:.4f}{}".format(
    iteration, n, float(l1_test), float(psnr_test), ssim_test, lpips_str))

metrics = {"test_psnr": float(psnr_test), "test_ssim": float(ssim_test),
           "test_lpips": (float(lpips_avg) if lpips_avg is not None else None),
           "test_l1": float(l1_test), "n_test": n, "iteration": iteration}
with open(os.path.join(run, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"wrote {os.path.join(run, 'metrics.json')}")
