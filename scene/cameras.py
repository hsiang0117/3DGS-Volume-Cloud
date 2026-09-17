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

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import threading
from collections import OrderedDict
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# ---------------------------------------------------------------------------
# CPU-side decoded-image cache: the raw uint8 array is decoded once and kept in
# RAM, while the GPU-side float tensor is still created and released per step,
# so VRAM usage is unchanged. `image_cache_max` (ModelParams) caps the number
# of cached frames (0 = unlimited, LRU eviction).
# ---------------------------------------------------------------------------
_DECODED_CACHE = OrderedDict()
_DECODED_CACHE_LOCK = threading.Lock()

def _decoded_uint8(image_path, max_entries=0):
    with _DECODED_CACHE_LOCK:
        arr = _DECODED_CACHE.get(image_path)
        if arr is not None:
            _DECODED_CACHE.move_to_end(image_path)
            return arr

    # Decode outside the lock (expensive); duplicate decodes on a rare race
    # are harmless — the last writer wins.
    with Image.open(image_path) as image:
        arr = np.array(image.convert("RGB"), dtype=np.uint8)

    with _DECODED_CACHE_LOCK:
        _DECODED_CACHE[image_path] = arr
        _DECODED_CACHE.move_to_end(image_path)
        if max_entries and len(_DECODED_CACHE) > max_entries:
            _DECODED_CACHE.popitem(last=False)
    return arr

class Camera(nn.Module):
    """Camera with lazy image loading.

    Stores path, target resolution and flags; the image tensor is decoded on
    access (`original_image`) and cached until released.
    """

    def __init__(self, resolution, R, T, FoVx, FoVy, image_path,
                 image_name, uid,
                 image_size=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 is_test_dataset = False, is_test_view = False,
                 is_nerf_synthetic = False,
                 v_l=None,
                 image_cache_max=0,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_cache_max = int(image_cache_max or 0)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # ---- Lazy-load metadata ----
        self.image_path = image_path
        self.resolution = resolution                # (W, H) target after rescale
        self.is_test_dataset = is_test_dataset
        self.is_test_view = is_test_view

        # Width / height in target (post-resize) coords.
        self.image_width = int(resolution[0])
        self.image_height = int(resolution[1])

        self.is_nerf_synthetic = is_nerf_synthetic

        # ---- Pose / projection (cheap, do eagerly) ----
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Per-frame sun direction (OpenGL world coords, points toward the
        # sun). Falls back to [0,1,0] when the dataset supplies none.
        if v_l is None:
            v_l = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.v_l = torch.from_numpy(np.asarray(v_l, dtype=np.float32)).to(self.data_device)

    # ------------------------------------------------------------------
    # Lazy-loaded tensor. The GPU tensor is rebuilt per step and released via
    # `Camera.release_loaded()`, so VRAM stays bounded by one image; the PNG
    # decode is not repeated because `_decoded_uint8` keeps the uint8 array in
    # the CPU cache (bounded by ModelParams.image_cache_max, 0 = unlimited).
    #
    # Load pipeline:
    #   decoded uint8 array from the CPU cache
    #   torch.from_numpy(uint8).to(cuda)
    #   F.interpolate on GPU to target resolution
    #   uint8 → fp32 / 255 on GPU (fused)
    # ------------------------------------------------------------------
    def _load_image(self):
        cache = getattr(self, "_loaded_rgb", None)
        if cache is not None:
            return cache

        # Decoded once per frame and kept in the CPU cache (see module top);
        # only the uint8→GPU copy below is repeated per step.
        arr = _decoded_uint8(self.image_path, self.image_cache_max)

        # (H, W, 3) uint8 CPU → (1, 3, H, W) uint8 GPU
        gpu_uint8 = torch.from_numpy(arr).to(self.data_device, non_blocking=True)
        gpu_uint8 = gpu_uint8.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        target_w, target_h = self.resolution
        if gpu_uint8.shape[-1] != target_w or gpu_uint8.shape[-2] != target_h:
            # interpolate needs float; convert before resize so antialiasing
            # works correctly. /255 is folded in.
            gpu_f = gpu_uint8.float().mul_(1.0 / 255.0)
            rgb = F.interpolate(gpu_f, size=(target_h, target_w),
                                mode="bilinear", align_corners=False,
                                antialias=True)
            rgb = rgb.squeeze(0).clamp_(0.0, 1.0)
        else:
            rgb = gpu_uint8.squeeze(0).float().mul_(1.0 / 255.0).clamp_(0.0, 1.0)

        self._loaded_rgb = rgb
        return rgb

    def release_loaded(self):
        """Drop the same-step image cache. Call once per training iteration
        after all consumers have finished with this camera's tensors."""
        if hasattr(self, "_loaded_rgb"):
            del self._loaded_rgb

    @property
    def original_image(self):
        return self._load_image()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, v_l=None):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        # Optional per-frame sun direction (OpenGL world coords). If None,
        # the renderer falls back to pc.get_v_l.
        if v_l is not None and not torch.is_tensor(v_l):
            v_l = torch.as_tensor(v_l, dtype=torch.float32, device="cuda")
        self.v_l = v_l

