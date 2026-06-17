import sys, json, os, re
sys.path.insert(0, '.')
import torch
from argparse import Namespace
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render
from utils.image_utils import psnr

run = sys.argv[1]
ply = f'{run}/point_cloud/iteration_30000/point_cloud.ply'

# Use the same T_light source the model was trained with (raster is default).
# Infer from cfg_args: tlight_voxel=True -> voxel, else tlight_raster flag;
# absent both -> voxel. argv[2] ('raster'|'voxel') overrides.
use_raster = True
raster_res = 512
cfg_path = os.path.join(run, 'cfg_args')
if len(sys.argv) > 2:
    use_raster = sys.argv[2] == 'raster'
elif os.path.exists(cfg_path):
    cfg = open(cfg_path).read()
    if 'tlight_voxel' in cfg:
        use_raster = 'tlight_voxel=True' not in cfg
    else:
        use_raster = 'tlight_raster=True' in cfg
    m = re.search(r'tlight_raster_res=(\d+)', cfg)
    if m:
        raster_res = int(m.group(1))
print(f'T_light source: {"raster" if use_raster else "voxel"}')

g = GaussianModel('default')
g.load_ply(ply)

cam_infos = readCamerasFromTransforms(r'D:\3DGS-Volume-Cloud\data\CloudDataset', 'transforms_test.json', False, True)
args = Namespace(resolution=-1, data_device='cuda')
cams = cameraList_from_camInfos(cam_infos, 1.0, args, True, True)

test_json = json.load(open('data/CloudDataset/transforms_test.json'))
# Match on the unique cam/time suffix, tolerant of path separators.
time_by_key = {}
for f in test_json['frames']:
    parts = f['file_path'].split('/')          # camXX / images / TTTT.png
    time_by_key[(parts[0], parts[-1])] = f['time_index']

pipe = Namespace(compute_cov3D_python=False, debug=False, antialiasing=False,
                 k_sigma=0.0, tlight_voxel=not use_raster, tlight_raster_res=raster_res)
bg = torch.zeros(3, device='cuda')

groups = {'old_inplane': [], 'new_outplane': []}
with torch.no_grad():
    for c, info in zip(cams, cam_infos):
        ip = info.image_path.replace('\\', '/')
        parts = ip.split('/')
        ti = time_by_key.get((parts[-3], parts[-1]), 0)
        pkg = render(c, g, pipe, bg)
        p = psnr(pkg['render'].clamp(0, 1), c.original_image.cuda()).mean().item()
        groups['new_outplane' if ti >= 61 else 'old_inplane'].append(p)
        if hasattr(c, 'release_loaded'):
            c.release_loaded()

for k, v in groups.items():
    v.sort()
    print(f'{k}: n={len(v)} | mean {sum(v)/len(v):.3f} | min {v[0]:.2f} | p50 {v[len(v)//2]:.2f} | max {v[-1]:.2f}')
