"""Evaluate cloud images using a fixed, lighting-independent VDB mask.

The evaluator always reports the original mask, full image, and rectangular crop.
--dilate (default 16) adds separate expanded-mask and outer-ring diagnostics;
it never replaces the original foreground. The default crop is the ORIGINAL
mask's bbox plus --margin (16). --crop-base dilated uses the expanded-mask bbox.

All PSNR values use pooled RGB MSE, then dB, then equal-weight frame averaging.
Regional SSIM averages the full-image SSIM map at selected window centers;
its 11x11 windows may include pixels outside the region. LPIPS-VGG v0.1 uses
unaltered full/cropped RGB tensors mapped from [0,1] to [-1,1], never zero-masked
images. No resizing or per-image exposure fitting is performed by the metrics.

Run in the selected repository's root with its own Python environment:
    python tools/eval_cloud_region.py output/<run> --iteration 30000 --save
    python <cloud-repo>/tools/eval_cloud_region.py output/<run> --repo official --save

--save creates a timestamped JSON alongside the checkpoint run.
--output FILE chooses an explicit new file. Existing results are never replaced.
Results use explicit *_mask_raw / *_mask_dilated / *_ring metric names;
per_view keys are full frame paths, preserving multiple lights per camera.
"""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def read_config(path):
    """Read argparse's saved Namespace without executing its contents."""
    node = ast.parse(Path(path).read_text(encoding='utf-8').strip(), mode='eval').body
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'Namespace' and not node.args
            and all(k.arg is not None for k in node.keywords)):
        raise ValueError('cfg_args must contain a literal Namespace(...)')
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def sha256_file(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def frame_records(frames):
    records, seen = [], set()
    for index, frame in enumerate(frames):
        path = PurePosixPath(frame['file_path'].replace('\\', '/')).as_posix()
        candidates = [p for p in PurePosixPath(path).parts if re.fullmatch(r'cam\d+', p)]
        if len(candidates) != 1:
            raise ValueError(f'Cannot identify a unique camNNN mask key in {path!r}')
        if path in seen:
            raise ValueError(f'Duplicate frame path: {path!r}; use unique image paths')
        seen.add(path)
        records.append({'index': index, 'file_path': path, 'camera': candidates[0]})
    if not records:
        raise ValueError('No frames in the requested transforms file')
    return records


def load_mask(path):
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f'Mask must be single-channel: {path}')
    values = set(np.unique(array).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f'Mask must be binary 0/1 or 0/255: {path}')
    mask = torch.from_numpy(array > 0)
    if not mask.any():
        raise ValueError(f'Empty foreground mask: {path}')
    return mask


def dilate_bool(mask, radius):
    if radius < 0:
        raise ValueError('Dilation radius must be nonnegative')
    if radius == 0:
        return mask
    # Equivalent to one square (2*r+1)^2 max filter, with much less work.
    x = mask[None, None].float()
    x = F.max_pool2d(x, (2*radius+1, 1), stride=1, padding=(radius, 0))
    x = F.max_pool2d(x, (1, 2*radius+1), stride=1, padding=(0, radius))
    return x[0, 0] > 0.5


def bbox(mask, margin):
    if margin < 0:
        raise ValueError('Crop margin must be nonnegative')
    ys, xs = torch.where(mask)
    if not len(ys):
        raise ValueError('Cannot crop an empty mask')
    return (max(int(xs.min())-margin, 0), max(int(ys.min())-margin, 0),
            min(int(xs.max())+margin+1, mask.shape[1]),
            min(int(ys.max())+margin+1, mask.shape[0]))


def psnr_from_mse(mse):
    mse = float(mse)
    return -10.0*math.log10(mse) if mse > 0 else math.inf


def ssim_map_of(a, b):
    """CHW -> CHW; same Gaussian SSIM formula/padding as the two repos."""
    if a.ndim != 3 or a.shape != b.shape:
        raise ValueError('SSIM requires matching CHW tensors')
    # Build the normalized kernel in float32 exactly as the upstream helper.
    kernel = torch.tensor([math.exp(-(i-5)**2/(2*1.5**2)) for i in range(11)])
    kernel /= kernel.sum()
    window = (kernel[:, None] @ kernel[None, :]).expand(a.shape[0], 1, 11, 11)
    window = window.contiguous().to(device=a.device, dtype=a.dtype)
    a, b = a[None], b[None]
    def conv(x):
        return F.conv2d(x, window, padding=5, groups=a.shape[1])
    ma, mb = conv(a), conv(b)
    va, vb = conv(a*a)-ma.square(), conv(b*b)-mb.square()
    cov = conv(a*b)-ma*mb
    return (((2*ma*mb+0.01**2)*(2*cov+0.03**2)) /
            ((ma.square()+mb.square()+0.01**2)*(va+vb+0.03**2)))[0]


def region_metrics(error_map, similarity_map, mask):
    count = int(mask.sum())
    if not count:
        return {'pixel_count': 0, 'area_fraction': 0.0, 'mse': None,
                'psnr': None, 'ssim': None}
    mse = float(error_map[mask].mean())
    return {'pixel_count': count, 'area_fraction': count/mask.numel(),
            'mse': mse, 'psnr': psnr_from_mse(mse),
            'ssim': float(similarity_map[mask].mean())}


def lpips01(model, a, b):
    # Both repos can supply CHW views of HWC storage. Fix the layout and cuDNN
    # precision so identical pixels cannot select different TF32 paths.
    a = (a[None].clamp(0, 1)*2-1).contiguous()
    b = (b[None].clamp(0, 1)*2-1).contiguous()
    with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
        return float(model(a, b).mean())


def mean_metrics(rows):
    means, counts = {}, {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        counts[key] = len(values)
        means[key] = sum(values)/len(values) if values else None
    return means, counts


def json_safe(value):
    """Keep exact-match PSNR explicit while writing strict JSON."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and math.isinf(value):
        return 'Infinity' if value > 0 else '-Infinity'
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('--repo', choices=['cloud', 'official'], default='cloud')
    parser.add_argument('--masks', type=Path, default=Path(r'D:\dataset\CloudDatasetMasks'))
    parser.add_argument('--transforms', default='transforms_test.json')
    parser.add_argument('--source', type=Path, help='Override the dataset path in cfg_args')
    parser.add_argument('--iteration', type=int, default=-1, help='Checkpoint iteration; -1 selects latest (recorded in JSON)')
    parser.add_argument('--dilate', type=int, default=16, help='Additional square-dilated region and outer ring; 0 disables them')
    parser.add_argument('--margin', type=int, default=16, help='Crop margin in pixels')
    parser.add_argument('--crop-base', choices=['raw', 'dilated'], default='raw', help='Mask used only to define the crop bbox; default raw')
    parser.add_argument('--save', action='store_true', help='Save to a new timestamped JSON in the run')
    parser.add_argument('--output', type=Path, help='Save to this new JSON instead; implies --save')
    args = parser.parse_args()
    if args.dilate < 0 or args.margin < 0 or args.iteration < -1:
        parser.error('dilate/margin must be nonnegative, iteration >= -1')
    if args.output and args.output.exists():
        parser.error(f'Output already exists; choose a new filename: {args.output}')
    return args


def main():
    args = parse_args()
    root = Path.cwd().resolve()
    sys.path.insert(0, str(root))
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos
    from utils.system_utils import searchForMaxIteration
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render

    run, masks_dir = Path(args.run).resolve(), args.masks.resolve()
    cfg_path = run/'cfg_args'
    cfg = read_config(cfg_path)
    source = (args.source or Path(cfg['source_path'])).resolve()
    transform_path = source/args.transforms
    records = frame_records(json.loads(transform_path.read_text(encoding='utf-8'))['frames'])
    iteration = args.iteration if args.iteration != -1 else searchForMaxIteration(str(run/'point_cloud'))
    ply = run/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
    if not ply.is_file():
        raise FileNotFoundError(ply)
    if cfg.get('train_test_exp', False):
        raise ValueError('Half-image exposure evaluation is not supported by this full-frame mask protocol')
    if cfg.get('white_background', False):
        raise ValueError('This evaluator targets black-background cloud datasets')
    if cfg.get('eval') is False:
        warnings.warn('cfg_args has eval=False; requested test frames may have been used in training')

    mask_cache, mask_hashes = {}, {}
    for rec in records:
        name = rec['camera']
        if name not in mask_cache:
            path = masks_dir/(name+'.png')
            mask_cache[name], mask_hashes[name] = load_mask(path), sha256_file(path)

    if args.repo == 'cloud':
        model = GaussianModel()
        model.load_ply(str(ply))
        available_env = model.env_net is not None and model._sky_transfer.numel() > 0
        use_env = bool(cfg.get('env_lighting', available_env))
        if use_env and not available_env:
            raise ValueError('Environment lighting requested, but checkpoint sidecars failed to load')
        if cfg.get('tonemap_learnable', False) and model.get_tonemap_coeffs is None:
            raise ValueError('Learnable tonemap requested, but coefficients failed to load')
        pipe = argparse.Namespace(tlight_voxel=cfg.get('tlight_voxel', False),
                                  tlight_raster_res=cfg.get('tlight_raster_res', 512),
                                  tonemap_aces=cfg.get('tonemap_aces', False),
                                  tonemap_learnable=cfg.get('tonemap_learnable', False),
                                  env_lighting=use_env, env_sh_order=model.env_sh_order)
        infos = readCamerasFromTransforms(str(source), args.transforms, False, True)
    else:
        model = GaussianModel(cfg.get('sh_degree', 3))
        model.load_ply(str(ply))
        pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                                  debug=False, antialiasing=cfg.get('antialiasing', False))
        infos = readCamerasFromTransforms(str(source), args.transforms, '', False, True)
    cameras = cameraList_from_camInfos(infos, 1.0,
        argparse.Namespace(resolution=-1, data_device='cuda', train_test_exp=False), True, True)
    if len(records) != len(cameras):
        raise ValueError('Frame/camera count mismatch')
    for rec, info, cam in zip(records, infos, cameras):
        expected = source/rec['file_path']
        if expected.suffix.lower() != '.png':
            expected = Path(str(expected)+'.png')
        if expected.resolve() != Path(info.image_path).resolve():
            raise ValueError(f'Camera order mismatch at {rec["file_path"]}')
        if tuple(mask_cache[rec['camera']].shape) != (cam.image_height, cam.image_width):
            raise ValueError(f'Mask/image size mismatch for {rec["file_path"]}; masks are never resized')

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import lpips
        perceptual = lpips.LPIPS(net='vgg', version='0.1').cuda().eval()
    perceptual.requires_grad_(False)

    print(f'[{args.repo}] iteration={iteration} frames={len(records)}', flush=True)
    print(f'raw mask always included; additional dilation={args.dilate}px; '
          f'crop={args.crop_base} bbox + {args.margin}px', flush=True)
    per_view, rows = {}, []
    bg = torch.zeros(3, device='cuda')
    with torch.no_grad():
        for rec, cam in zip(records, cameras):
            pred = render(cam, model, pipe, bg)['render'].clamp(0, 1)
            gt = cam.original_image.to('cuda').clamp(0, 1)
            if hasattr(cam, 'release_loaded'):
                cam.release_loaded()
            if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[0] != 3:
                raise ValueError(f'Expected matching RGB tensors for {rec["file_path"]}')
            if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
                raise ValueError(f'Non-finite image values for {rec["file_path"]}')
            raw = mask_cache[rec['camera']].to('cuda')
            expanded = dilate_bool(raw, args.dilate)
            error = (pred-gt).square().mean(0)
            sm = ssim_map_of(pred, gt).mean(0)
            regions = {'mask_raw': region_metrics(error, sm, raw)}
            if args.dilate:
                regions['mask_dilated'] = region_metrics(error, sm, expanded)
                regions['ring'] = region_metrics(error, sm, expanded & ~raw)
            x0, y0, x1, y1 = bbox(raw if args.crop_base == 'raw' else expanded, args.margin)
            pc, gc = pred[:, y0:y1, x0:x1], gt[:, y0:y1, x0:x1]
            if min(pc.shape[-2:]) < 16:
                raise ValueError('Crop too small for VGG LPIPS; increase --margin to reach at least 16x16')
            mse_full, mse_crop = float(error.mean()), float((pc-gc).square().mean())
            r = {'mse_full': mse_full, 'psnr_full': psnr_from_mse(mse_full),
                 'ssim_full': float(sm.mean()), 'lpips_full': lpips01(perceptual, pred, gt),
                 'mse_crop': mse_crop, 'psnr_crop': psnr_from_mse(mse_crop),
                 'ssim_crop': float(ssim_map_of(pc, gc).mean()), 'lpips_crop': lpips01(perceptual, pc, gc),
                 'crop_area_share': (x1-x0)*(y1-y0)/raw.numel(),
                 'mask_raw_share_in_crop': float(raw[y0:y1, x0:x1].float().mean()),
                 'mask_dilated_share_in_crop': float(expanded[y0:y1, x0:x1].float().mean())}
            for name, stats in regions.items():
                for metric in ['mse', 'psnr', 'ssim']:
                    r[f'{metric}_{name}'] = stats[metric]
                r[f'{name}_share'] = stats['area_fraction']
            per_view[rec['file_path']] = {**rec, 'image_size_wh': [pred.shape[2], pred.shape[1]],
                                        'crop_xyxy_exclusive': [x0, y0, x1, y1], 'metrics': r}
            rows.append(r)
            if len(rows) % 6 == 0 or len(rows) == len(records):
                print(f'  evaluated {len(rows)}/{len(records)} frames', flush=True)
    means, valid_counts = mean_metrics(rows)
    print(f'\n{"region":24s} {"PSNR":>10s} {"SSIM":>10s} {"LPIPS":>10s}')
    for name in ['full', 'mask_raw', 'mask_dilated', 'ring', 'crop']:
        if f'psnr_{name}' not in means:
            continue
        values = [means.get(f'{metric}_{name}') for metric in ['psnr', 'ssim', 'lpips']]
        print(f'{name:24s} ' + ' '.join(f'{v:10.5f}' if v is not None else f'{"n/a":>10s}' for v in values))
    print(f'raw mask/full={means["mask_raw_share"]:.3%}; crop/full={means["crop_area_share"]:.3%}; '
          f'raw mask/crop={means["mask_raw_share_in_crop"]:.3%}')
    worst = sorted(per_view.items(), key=lambda item: item[1]['metrics']['psnr_mask_raw'])[:3]
    print('worst raw-mask PSNR:', [(key, round(v['metrics']['psnr_mask_raw'], 3)) for key, v in worst])

    if args.save or args.output:
        stamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
        path = args.output or run/(f'metrics_cloud_region_{Path(args.transforms).stem}_'
            f'iter{iteration}_d{args.dilate}_m{args.margin}_crop-{args.crop_base}_lpips-fp32_{stamp}.json')
        path = path.resolve()
        manifest = masks_dir/'_metadata'/'manifest.json'
        out = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'repo': args.repo, 'repo_root': str(root), 'run': str(run), 'source': str(source),
            'iteration': iteration, 'checkpoint': str(ply), 'checkpoint_sha256': sha256_file(ply),
            'config': cfg, 'cfg_sha256': sha256_file(cfg_path), 'render_pipeline': vars(pipe),
            'transforms': str(transform_path), 'transforms_sha256': sha256_file(transform_path),
            'masks': str(masks_dir), 'mask_sha256_by_camera': mask_hashes,
            'mask_manifest': json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None,
            'script_sha256': sha256_file(__file__),
            'versions': {'torch': torch.__version__, 'lpips': version('lpips'), 'numpy': np.__version__},
            'protocol': {
                'psnr': 'Per-frame RGB-pooled MSE, peak=1, then -10*log10(MSE)',
                'ssim': '11x11 Gaussian sigma=1.5, C1=0.01^2 C2=0.03^2, zero padding=5; RGB mean',
                'regional_ssim': 'Full-image SSIM map averaged at mask-selected window centers; windows may include outside pixels',
                'lpips': 'LPIPS v0.1, VGG, calibrated weights, RGB [0,1] mapped to [-1,1]; no zero masking or crop resizing',
                'lpips_backend': 'contiguous NCHW float32; cuDNN TF32=False, benchmark=False, deterministic=True',
                'aggregation': 'Arithmetic mean across frames, equal frame weights; null metrics excluded with valid_counts',
                'exact_match_psnr_json': 'Positive infinity is serialized as the string Infinity',
                'raw_mask': 'Binary PNG foreground; always reported independently of dilation',
                'dilate_px': args.dilate, 'dilation_kernel': 'square, side=2*dilate_px+1',
                'ring': 'dilated mask minus original mask; n/a if empty',
                'crop_base': args.crop_base, 'margin_px': args.margin,
                'crop': 'Axis-aligned bbox of selected mask + margin, clipped to image; x1/y1 exclusive'},
            'frame_count': len(rows), 'means': means, 'valid_counts': valid_counts, 'per_view': per_view}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            json.dump(json_safe(out), f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f'Wrote {path}', flush=True)


if __name__ == '__main__':
    main()
