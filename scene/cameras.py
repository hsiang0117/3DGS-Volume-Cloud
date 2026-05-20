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
import numpy as np
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch

class Camera(nn.Module):
    """Camera with **lazy** image loading.

    Decoding 2989 × 1024² images eagerly costs ~48 GB GPU memory and
    saturates main RAM at scene-load time. This class keeps only the path,
    target resolution and a few flags; the actual tensor is decoded on
    attribute access (`original_image`) and discarded by Python's GC after
    the consumer drops the reference.

    Each access pays ~5 ms (PNG decode) + ~3 ms (CPU→GPU upload) at 1024².
    For our train loop that touches each Camera ~once per iter this is
    well under the rasterizer cost.
    """

    def __init__(self, resolution, R, T, FoVx, FoVy, image_path,
                 image_name, uid,
                 image_size=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 is_test_dataset = False, is_test_view = False,
                 is_nerf_synthetic = False,
                 sun_dir=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # ---- Lazy-load metadata -------------------------------------------
        self.image_path = image_path
        self.resolution = resolution                # (W, H) target after rescale
        self.is_test_dataset = is_test_dataset
        self.is_test_view = is_test_view

        # Width / height in *target* (post-resize) coords. These are needed
        # by camera_to_JSON / projection setup before any pixel access.
        self.image_width = int(resolution[0])
        self.image_height = int(resolution[1])

        self.is_nerf_synthetic = is_nerf_synthetic

        # ---- Pose / projection (cheap, do eagerly) ------------------------
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Per-frame sun direction (OpenGL world coords, "pointing toward the sun").
        # If the dataset didn't supply one, fall back to [0,1,0] so existing
        # static-sun datasets keep behaving exactly as before.
        if sun_dir is None:
            sun_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.sun_dir = torch.from_numpy(np.asarray(sun_dir, dtype=np.float32)).to(self.data_device)

    # ------------------------------------------------------------------
    # Lazy-loaded tensor. Same-step accesses share a single decode via a
    # tiny cache that the training loop is expected to release at end of
    # iteration (`Camera.release_loaded()`). We do NOT keep the cache
    # across iterations: that would re-introduce the OOM this whole
    # refactor was meant to avoid.
    # ------------------------------------------------------------------
    def _load_image(self):
        cache = getattr(self, "_loaded_rgb", None)
        if cache is not None:
            return cache
        image = Image.open(self.image_path)
        resized = PILtoTorch(image, self.resolution)
        rgb = resized[:3, ...].clamp(0.0, 1.0).to(self.data_device)
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
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, sun_dir=None):
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
        # the renderer falls back to pc.get_sun_dir.
        if sun_dir is not None and not torch.is_tensor(sun_dir):
            sun_dir = torch.as_tensor(sun_dir, dtype=torch.float32, device="cuda")
        self.sun_dir = sun_dir

