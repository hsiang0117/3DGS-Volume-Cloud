"""
Diagnostic: localize mid-tone over-brightness by binning the signed luminance
residual (pred-gt) against per-pixel rendered T_light (sun transmittance /
shadow depth), not GT luminance.

T_light bins span the penumbra axis: ~0 deep core shadow, ~0.2-0.7 penumbra
(terminator), ~1 fully lit. Cross-tabbed with GT luminance to correlate mid-tone
with mid-T_light. Tests whether the multiple-scattering octave approximation
over-fills the soft lit->shadow transition.

Read-only: renders RGB + a T_light image (override_color pass) per test frame;
no retraining or model mutation.

Usage:
    python tools/penumbra_residual.py output/<run> [iteration]
"""
import sys, os, re, json
sys.path.insert(0, '.')
import torch
from argparse import Namespace
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render

run = sys.argv[1]
iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
ply = f'{run}/point_cloud/iteration_{iteration}/point_cloud.ply'

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
print(f'run={run} iter={iteration} | source={source_path}')
print(f'T_light={"raster" if use_raster else "voxel"} | aces={tonemap_aces} '
      f'| learnable={tonemap_learnable}')

g = GaussianModel()
g.load_ply(ply)

pipe = Namespace(
                 k_sigma=0.0, tlight_voxel=not use_raster, tlight_raster_res=raster_res,
                 tonemap_aces=tonemap_aces, tonemap_learnable=tonemap_learnable)
bg = torch.zeros(3, device='cuda')

cam_infos = readCamerasFromTransforms(source_path, 'transforms_test.json', False, True)
args = Namespace(resolution=-1, data_device='cuda')
cams = cameraList_from_camInfos(cam_infos, 1.0, args, True, True)

def lum(img):
    return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]

# T_light buckets (penumbra axis). edges -> 6 bins.
TL_EDGES = [0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.001]
TL_NAMES = ['core<.05', '.05-.2', '.2-.5(pen)', '.5-.8(pen)', '.8-.95', 'lit>.95']
# GT-luminance buckets.
GL_EDGES = [0.0, 0.05, 0.25, 0.55, 1.01]
GL_NAMES = ['deep', 'dark', 'mid', 'bright']

# res_by_tl[i] = [sum_diff, count]; xtab[gl][tl] = [sum_diff, count]
res_by_tl = [[0.0, 0] for _ in TL_NAMES]
xtab = [[[0.0, 0] for _ in TL_NAMES] for _ in GL_NAMES]

def bucket(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2

with torch.no_grad():
    for c in cams:
        pkg = render(c, g, pipe, bg)
        pred = pkg['render'].clamp(0, 1)
        gt = c.original_image.cuda()
        depth = pkg['depth'].squeeze(0)
        # Per-pixel T_light image: render per-Gaussian T_light as colour;
        # override_color skips tonemap, giving alpha-weighted shadow depth.
        tl_pg = pkg['T_light']  # (P,1) detached
        tl_pkg = render(c, g, pipe, bg, override_color=tl_pg.expand(-1, 3).contiguous())
        tl_img = tl_pkg['render'][0]  # (H,W), all 3 channels identical

        gl = lum(gt)
        pl = lum(pred)
        diff = (pl - gl)
        cloud = (depth > 0) | (gl > 0.004)

        # Vectorised bucket assignment.
        gl_idx = torch.bucketize(gl, torch.tensor(GL_EDGES[1:-1], device='cuda'))
        tl_idx = torch.bucketize(tl_img, torch.tensor(TL_EDGES[1:-1], device='cuda'))
        gl_idx = gl_idx.clamp(0, len(GL_NAMES) - 1)
        tl_idx = tl_idx.clamp(0, len(TL_NAMES) - 1)

        for ti in range(len(TL_NAMES)):
            m_tl = cloud & (tl_idx == ti)
            n = int(m_tl.sum().item())
            if n:
                res_by_tl[ti][0] += float(diff[m_tl].sum().item())
                res_by_tl[ti][1] += n
            for gi in range(len(GL_NAMES)):
                m2 = m_tl & (gl_idx == gi)
                n2 = int(m2.sum().item())
                if n2:
                    xtab[gi][ti][0] += float(diff[m2].sum().item())
                    xtab[gi][ti][1] += n2
        if hasattr(c, 'release_loaded'):
            c.release_loaded()

print('\n=== signed residual mean(pred-gt) by per-pixel T_light (penumbra axis) ===')
tot = sum(n for _, n in res_by_tl)
for name, (s, n) in zip(TL_NAMES, res_by_tl):
    v = s / n if n else float('nan')
    print(f'  {name:12s}: {v:+.4f}  (n={n:,}, {100*n/tot:4.1f}%)')

print('\n=== cross-tab: signed residual by GT-luminance (rows) x T_light (cols) ===')
hdr = ' ' * 8 + ''.join(f'{nm:>12s}' for nm in TL_NAMES)
print(hdr)
for gi, gname in enumerate(GL_NAMES):
    cells = []
    for ti in range(len(TL_NAMES)):
        s, n = xtab[gi][ti]
        cells.append(f'{(s/n):+.4f}' if n else '     -  ')
    print(f'  {gname:6s}' + ''.join(f'{c:>12s}' for c in cells))
print('\n  (counts per cell, same layout)')
for gi, gname in enumerate(GL_NAMES):
    cells = [f'{xtab[gi][ti][1]//1000}k' if xtab[gi][ti][1] else '-' for ti in range(len(TL_NAMES))]
    print(f'  {gname:6s}' + ''.join(f'{c:>12s}' for c in cells))
