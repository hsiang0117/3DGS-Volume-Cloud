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
import torch
from PIL import Image

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_periodic_render(model_path, iteration, image_tensor, image_name=None):
    """Dump a single rendered frame to <model_path>/render_test/iter_NNNNNN[_<name>].png.

    Intended for periodic visual sanity checks during long training runs,
    keyed by iteration so a directory listing reads as a timeline. Tensor
    is expected as (C, H, W) in [0, 1].
    """
    render_dir = os.path.join(model_path, "render_test")
    os.makedirs(render_dir, exist_ok=True)
    suffix = f"_{image_name}" if image_name else ""
    render_path = os.path.join(render_dir, f"iter_{iteration:06d}{suffix}.png")
    img8 = (torch.clamp(image_tensor, 0.0, 1.0) * 255.0).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    Image.fromarray(img8).save(render_path)


# Module-level cache: lpips weight download is ~100 MB and one-time, so we
# build the model once and reuse for every eval call. `None` after a failed
# attempt → silent skip on subsequent calls so training never blocks on a
# missing optional dep.
_LPIPS_FN = None
_LPIPS_TRIED = False

def get_lpips_fn(net: str = "vgg", device: str = "cuda"):
    """Return a callable `f(pred01, gt01) -> float` computing LPIPS, or
    None if the `lpips` package is unavailable. Inputs are (C, H, W) or
    (N, C, H, W) tensors in [0, 1]; the helper handles the [-1, 1] remap
    LPIPS expects.
    """
    global _LPIPS_FN, _LPIPS_TRIED
    if _LPIPS_TRIED:
        return _LPIPS_FN
    _LPIPS_TRIED = True
    try:
        import lpips  # noqa: F401  — lazy import; ~100 MB backbone download on first call
    except ImportError:
        print("[lpips] package not installed; skipping LPIPS metric. "
              "`pip install lpips` to enable.")
        _LPIPS_FN = None
        return None
    try:
        model = lpips.LPIPS(net=net).to(device).eval()
    except Exception as e:
        print(f"[lpips] failed to construct {net} model: {e}; skipping.")
        _LPIPS_FN = None
        return None
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def _lpips(pred01, gt01):
        if pred01.dim() == 3:
            pred01 = pred01.unsqueeze(0)
            gt01 = gt01.unsqueeze(0)
        pred = (pred01.to(device).clamp(0.0, 1.0) * 2.0 - 1.0)
        gt = (gt01.to(device).clamp(0.0, 1.0) * 2.0 - 1.0)
        return float(model(pred, gt).mean().item())

    _LPIPS_FN = _lpips
    return _lpips
