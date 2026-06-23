"""Grouped test-set PSNR: held-out-sun group vs seen-sun (new-view) group.

Reads source_path / T_light source / tonemap mode from the run's cfg_args (so it
matches how the model was trained), auto-detects a Stage-2 env model (env sidecars),
and reports PSNR for the held-out suns (relighting generalisation) vs the rest.
Held-out suns are inferred from the split itself (time_index present in test but
absent from train); iteration defaults to the newest checkpoint in the run.

Usage:
    python tools/eval_test_groups.py output/<run> [iteration]
"""
import sys, json, os, re
sys.path.insert(0, '.')
import argparse
import torch
from argparse import Namespace
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render
from utils.image_utils import psnr

ap = argparse.ArgumentParser(description="Grouped test-set PSNR (held-out vs seen sun).")
ap.add_argument("run", help="training output dir (contains cfg_args + point_cloud/)")
ap.add_argument("iteration", nargs="?", type=int, default=None,
                help="checkpoint iteration; default = newest in point_cloud/")
cli = ap.parse_args()
run = cli.run
pc_dir = os.path.join(run, "point_cloud")
iteration = cli.iteration if cli.iteration is not None else searchForMaxIteration(pc_dir)
ply = os.path.join(pc_dir, f"iteration_{iteration}", "point_cloud.ply")

# --- cfg_args: dataset path + T_light source + tonemap mode -------------
cfg = open(os.path.join(run, 'cfg_args')).read()
m = re.search(r"source_path=['\"]([^'\"]+)['\"]", cfg)
source_path = m.group(1) if m else r'D:\3DGS-Volume-Cloud\data\CloudDatasetUniform'
if 'tlight_voxel' in cfg:
    use_raster = 'tlight_voxel=True' not in cfg
else:
    use_raster = 'tlight_raster=True' in cfg
m = re.search(r'tlight_raster_res=(\d+)', cfg)
raster_res = int(m.group(1)) if m else 512
tonemap_aces = 'tonemap_aces=True' in cfg
tonemap_learnable = 'tonemap_learnable=True' in cfg

g = GaussianModel()
g.load_ply(ply)
# Stage-2 env model carries env sidecars (env_net.pt / sky_transfer.npy) restored
# by load_ply; turn on env_lighting so render reproduces the trained shading.
env_lighting = (g.env_net is not None) and (g._sky_transfer.numel() > 0)
print(f'run={run} iter={iteration} | source_path={source_path}')
print(f'T_light={"raster" if use_raster else "voxel"} | tonemap_aces={tonemap_aces} '
      f'| tonemap_learnable={tonemap_learnable} | env_lighting={env_lighting}')

pipe = Namespace(k_sigma=0.0, tlight_voxel=not use_raster, tlight_raster_res=raster_res,
                 tonemap_aces=tonemap_aces, tonemap_learnable=tonemap_learnable,
                 env_lighting=env_lighting, env_sh_order=g.env_sh_order)
bg = torch.zeros(3, device='cuda')

cam_infos = readCamerasFromTransforms(source_path, 'transforms_test.json', False, True)
args = Namespace(resolution=-1, data_device='cuda')
cams = cameraList_from_camInfos(cam_infos, 1.0, args, True, True)

# Map (camXX, file) -> time_index from the test json to identify held-out suns.
test_json = json.load(open(os.path.join(source_path, 'transforms_test.json')))
time_by_key = {}
test_suns = set()
for f in test_json['frames']:
    parts = f['file_path'].split('/')
    time_by_key[(parts[0], parts[-1])] = f['time_index']
    test_suns.add(f['time_index'])

# Held-out suns = inferred from the split: time_index present in test but absent
# from train (a whole sun direction kept out of training). Falls back to the known
# {7,22,37,52} if the train json is missing.
train_path = os.path.join(source_path, 'transforms_train.json')
if os.path.exists(train_path):
    train_suns = {f['time_index'] for f in json.load(open(train_path))['frames']}
    HELDOUT_SUNS = test_suns - train_suns
else:
    HELDOUT_SUNS = {7, 22, 37, 52}
print(f'held-out suns (in test, absent from train) = {sorted(HELDOUT_SUNS)}')

groups = {'heldout_sun': [], 'seen_sun': []}
with torch.no_grad():
    for c, info in zip(cams, cam_infos):
        ip = info.image_path.replace('\\', '/')
        parts = ip.split('/')
        ti = time_by_key.get((parts[-3], parts[-1]), -1)
        pkg = render(c, g, pipe, bg)
        p = psnr(pkg['render'].clamp(0, 1), c.original_image.cuda()).mean().item()
        groups['heldout_sun' if ti in HELDOUT_SUNS else 'seen_sun'].append(p)
        if hasattr(c, 'release_loaded'):
            c.release_loaded()

for k, v in groups.items():
    if not v:
        print(f'{k}: n=0')
        continue
    v.sort()
    print(f'{k}: n={len(v)} | mean {sum(v)/len(v):.3f} | min {v[0]:.2f} | '
          f'p50 {v[len(v)//2]:.2f} | max {v[-1]:.2f}')
if groups['heldout_sun'] and groups['seen_sun']:
    gap = (sum(groups['heldout_sun'])/len(groups['heldout_sun'])
           - sum(groups['seen_sun'])/len(groups['seen_sun']))
    print(f'held-out - seen gap = {gap:+.3f} dB')
