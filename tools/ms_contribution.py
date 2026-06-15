"""
Experiment-2 Step A disambiguation: is the penumbra over-brightness driven by
the multiple-scattering OCTAVES specifically, or by T_light softness / HG?

penumbra_residual.py showed: deep core shadow is calibrated (residual ~0), but
the lit side of the terminator (mid T_light 0.05-0.5) is rendered too bright.
That is consistent with octave over-scatter, T_light being too soft, OR the HG
phase. This isolates the OCTAVE contribution by re-rendering each test frame
twice from the SAME trained model:

  full   = all 6 octaves (the trained behaviour)
  single = octave 0 only (single scatter): higher octave energies zeroed by
           setting _octave_weights[:,1:] to softplus^-1(0) in-process

ms = full - single  (the multiple-scattering contribution, in render/tonemap
space). We then bin ms by per-pixel T_light. If ms is large exactly in the
mid-T_light penumbra where the residual is positive, the octaves are the lever
for experiment 2; if ms is flat / lives elsewhere, the cause is T_light/HG.

Read-only: parameter is restored after rendering; no file is written, no
training. Tonemap is left as the model trained (so the comparison is in the
space the residual was measured in).

Usage:
    python tools/ms_contribution.py output/<run> [iteration]
"""
import sys, os, re
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
print(f'run={run} iter={iteration} | T_light={"raster" if use_raster else "voxel"} '
      f'| aces={tonemap_aces} | learnable={tonemap_learnable}')

g = GaussianModel('default')
g.load_ply(ply)

pipe = Namespace(compute_cov3D_python=False, debug=False, antialiasing=False,
                 k_sigma=0.0, tlight_voxel=not use_raster, tlight_raster_res=raster_res,
                 tonemap_aces=tonemap_aces, tonemap_learnable=tonemap_learnable)
bg = torch.zeros(3, device='cuda')

cam_infos = readCamerasFromTransforms(source_path, 'transforms_test.json', False, True)
args = Namespace(resolution=-1, data_device='cuda')
cams = cameraList_from_camInfos(cam_infos, 1.0, args, True, True)

# How much of the trained octave energy lives above octave 0?
ow = g.get_octave_weights  # (P,6)
ow_share = ow.sum(0)
ow_share = (ow_share / ow_share.sum()).tolist()
print('octave energy share (n=0..5):', [round(x, 3) for x in ow_share])

def lum(img):
    return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]

TL_EDGES = [0.0, 0.05, 0.2, 0.5, 0.8, 1.001]
TL_NAMES = ['core<.05', '.05-.2', '.2-.5(pen)', '.5-.8(pen)', '.8-1']
# ms by T_light: contribution magnitude; also full & single means for context.
acc = {k: [[0.0, 0] for _ in TL_NAMES] for k in ('ms', 'full', 'single')}

saved = g._octave_weights.detach().clone()
NEG = -30.0  # softplus(-30) ~ 0

with torch.no_grad():
    tl_edges = torch.tensor(TL_EDGES[1:-1], device='cuda')
    for c in cams:
        # full render
        pkg_full = render(c, g, pipe, bg)
        full = pkg_full['render'].clamp(0, 1)
        depth = pkg_full['depth'].squeeze(0)
        tl_pg = pkg_full['T_light']
        tl_img = render(c, g, pipe, bg,
                        override_color=tl_pg.expand(-1, 3).contiguous())['render'][0]

        # single-scatter render: zero octaves 1..5
        g._octave_weights.data[:, 1:] = NEG
        single = render(c, g, pipe, bg)['render'].clamp(0, 1)
        g._octave_weights.data.copy_(saved)

        fl, sl = lum(full), lum(single)
        ms = fl - sl
        cloud = (depth > 0) | (fl > 0.004)
        tl_idx = torch.bucketize(tl_img, tl_edges).clamp(0, len(TL_NAMES) - 1)
        for ti in range(len(TL_NAMES)):
            mtl = cloud & (tl_idx == ti)
            n = int(mtl.sum().item())
            if n:
                acc['ms'][ti][0] += float(ms[mtl].sum().item())
                acc['full'][ti][0] += float(fl[mtl].sum().item())
                acc['single'][ti][0] += float(sl[mtl].sum().item())
                for k in acc:
                    acc[k][ti][1] += n
        if hasattr(c, 'release_loaded'):
            c.release_loaded()

print('\n=== mean luminance by per-pixel T_light: full / single(oct0) / MS(full-single) ===')
print(' ' * 12 + f'{"full":>10s}{"single":>10s}{"MS":>10s}{"MS/full":>9s}{"count":>11s}')
for ti, name in enumerate(TL_NAMES):
    f = acc['full'][ti][0] / acc['full'][ti][1] if acc['full'][ti][1] else float('nan')
    s = acc['single'][ti][0] / acc['single'][ti][1] if acc['single'][ti][1] else float('nan')
    msv = acc['ms'][ti][0] / acc['ms'][ti][1] if acc['ms'][ti][1] else float('nan')
    n = acc['full'][ti][1]
    frac = msv / f if f else float('nan')
    print(f'  {name:10s}{f:>10.4f}{s:>10.4f}{msv:>+10.4f}{frac:>8.1%}{n:>11,}')
print('\nIf MS (and MS/full) peaks in the .2-.5 / .5-.8 penumbra bins -> octaves '
      'are the lever; if MS is flat across T_light -> cause is T_light/HG, not octaves.')
