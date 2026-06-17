"""
Signed-residual-by-luminance diagnostic + held-out-sun PSNR for a trained run.

Renders the full test split through the same render() path training/eval use
(T_light source and tonemap mode auto-detected from cfg_args), then reports:

  * per-bucket signed residual mean(pred - gt) over cloud pixels (GT luminance
    > eps; pure-black background excluded), split into luminance buckets. Flat
    ~0 across buckets means the output space matches the GT.
  * compression metric (|deep| + |bright|) / 2.
  * held-out-sun vs seen-sun PSNR (relighting generalisation gap); held-out
    whole suns are time_index in {7,22,37,52}.
  * learned tonemap coeffs, if any.

Usage:
    python tools/residual_buckets.py output/<run> [iteration]
"""
import sys, os, re, json
sys.path.insert(0, '.')
import torch
from argparse import Namespace
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render
from utils.image_utils import psnr

run = sys.argv[1]
iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
ply = f'{run}/point_cloud/iteration_{iteration}/point_cloud.ply'

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
print(f'run={run} iter={iteration}')
print(f'source_path={source_path}')
print(f'T_light={"raster" if use_raster else "voxel"} | tonemap_aces={tonemap_aces} '
      f'| tonemap_learnable={tonemap_learnable}')

g = GaussianModel('default')
g.load_ply(ply)
coeffs = g.get_tonemap_coeffs
if coeffs is not None:
    print(f'learned tonemap coeffs (a,b,c,d) = {[round(c,4) for c in coeffs.tolist()]} '
          f'(canonical {list(GaussianModel.TONEMAP_CANONICAL)})')

pipe = Namespace(compute_cov3D_python=False, debug=False, antialiasing=False,
                 k_sigma=0.0, tlight_voxel=not use_raster, tlight_raster_res=raster_res,
                 tonemap_aces=tonemap_aces, tonemap_learnable=tonemap_learnable)
bg = torch.zeros(3, device='cuda')

cam_infos = readCamerasFromTransforms(source_path, 'transforms_test.json', False, True)
args = Namespace(resolution=-1, data_device='cuda')
cams = cameraList_from_camInfos(cam_infos, 1.0, args, True, True)

# Map (camXX, file) -> time_index from the test json to identify held-out suns.
test_json = json.load(open(os.path.join(source_path, 'transforms_test.json')))
time_by_key = {}
for f in test_json['frames']:
    parts = f['file_path'].split('/')
    time_by_key[(parts[0], parts[-1])] = f['time_index']
HELDOUT_SUNS = {7, 22, 37, 52}

# Luminance buckets over cloud pixels. Cloud = render coverage (depth > 0)
# UNION GT showing cloud (gt_lum > tiny), so deep self-shadow pixels (GT
# near-black but cloud present) are kept; the 'deep' bucket isolates them.
BG_EPS = 0.004           # below this AND no coverage => pure-black background
BUCKETS = (('deep', 0.0, 0.05), ('dark', 0.05, 0.25),
           ('mid', 0.25, 0.55), ('bright', 0.55, 1.01))

def lum(img):  # img (3,H,W) -> (H,W)
    return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]

# Accumulators: signed residual sums + counts per bucket.
res = {b[0]: [0.0, 0] for b in BUCKETS}
psnr_groups = {'heldout_sun': [], 'seen_sun': []}

with torch.no_grad():
    for c, info in zip(cams, cam_infos):
        ip = info.image_path.replace('\\', '/')
        parts = ip.split('/')
        ti = time_by_key.get((parts[-3], parts[-1]), -1)
        pkg = render(c, g, pipe, bg)
        pred = pkg['render'].clamp(0, 1)
        gt = c.original_image.cuda()
        psnr_groups['heldout_sun' if ti in HELDOUT_SUNS else 'seen_sun'].append(
            psnr(pred, gt).mean().item())

        gl = lum(gt)
        pl = lum(pred)
        depth = pkg.get('depth')
        covered = (depth.squeeze(0) > 0) if depth is not None else (gl > BG_EPS)
        cloud = covered | (gl > BG_EPS)     # exclude only true background
        diff = (pl - gl)
        for name, lo, hi in BUCKETS:
            mask = cloud & (gl >= lo) & (gl < hi)
            n = int(mask.sum().item())
            if n:
                res[name][0] += float(diff[mask].sum().item())
                res[name][1] += n
        if hasattr(c, 'release_loaded'):
            c.release_loaded()

print('\n--- signed residual mean(pred-gt) by GT-luminance bucket (cloud pixels) ---')
vals = {}
for b, _, _ in BUCKETS:
    s, n = res[b]
    v = s / n if n else float('nan')
    vals[b] = v
    print(f'  {b:6s}: {v:+.4f}  (n={n:,})')
comp = (abs(vals['deep']) + abs(vals['bright'])) / 2
allmean = sum(vals[b] for b, _, _ in BUCKETS) / len(BUCKETS)
print(f'  compression (|deep|+|bright|)/2 = {comp:.4f}')
print(f'  mean signed residual (all buckets) = {allmean:+.4f}')
print(f'  [octave false-floor target = deep/dark systematic + bias]')

print('\n--- PSNR by sun group ---')
for k, v in psnr_groups.items():
    if v:
        v = sorted(v)
        print(f'  {k:12s}: n={len(v)} | mean {sum(v)/len(v):.3f} | '
              f'min {v[0]:.2f} | p50 {v[len(v)//2]:.2f} | max {v[-1]:.2f}')
gap = (sum(psnr_groups['heldout_sun'])/len(psnr_groups['heldout_sun'])
       - sum(psnr_groups['seen_sun'])/len(psnr_groups['seen_sun']))
print(f'  held-out - seen gap = {gap:+.3f} dB')
