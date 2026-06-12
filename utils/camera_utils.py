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

import numpy as np
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from random import randint
from utils.graphics_utils import fov2focal
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    # Local import: scene.cameras → scene/__init__.py → utils.camera_utils
    # would otherwise circular-import on a cold start.
    from scene.cameras import Camera

    # Read only PNG header for size (no pixel decode → cheap).
    with Image.open(cam_info.image_path) as image:
        orig_w, orig_h = image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution


        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_path=cam_info.image_path,
                  image_size=(orig_w, orig_h),
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  is_nerf_synthetic=is_nerf_synthetic,
                  sun_dir=getattr(cam_info, "sun_dir", None))

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    # Each loadCam does a PIL header read + Camera() construction. Both are
    # pure CPU work; parallelising with a thread pool gives a near-linear
    # speedup at scene-load time for thousands of cameras.
    n_workers = min(32, (os.cpu_count() or 4) * 2)
    camera_list = [None] * len(cam_infos)

    def _load(args_tuple):
        id, c = args_tuple
        return id, loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for id, cam in ex.map(_load, enumerate(cam_infos)):
            camera_list[id] = cam

    return camera_list

def camera_to_JSON(id, camera : "Camera"):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


class CameraPrefetcher:
    """Background producer of (Camera, image-loaded) pairs.

    The training loop calls `next()` to get a camera ready for use; a
    daemon worker thread picks indices off a shared sampler and warms each
    camera's `original_image` cache (PIL decode + GPU upload) ahead of
    time. Queue size = 2 keeps the worker one step ahead without holding
    extra GPU memory.

    Sampling matches the inline policy of the original loop: sample without
    replacement until the stack is empty, then refill from the full
    training-camera list. Refill happens inside the worker so the producer
    never starves.

    Multi-threaded CUDA uploads are safe under PyTorch's default stream —
    operations queue serially and synchronize correctly with the consumer.
    """

    def __init__(self, scene, queue_size: int = 2):
        self.scene = scene
        self._refill()
        self._q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _refill(self):
        self._stack = self.scene.getTrainCameras().copy()
        self._indices = list(range(len(self._stack)))

    def _pick_one(self):
        with self._lock:
            if not self._stack:
                self._refill()
            rand_idx = randint(0, len(self._indices) - 1)
            cam = self._stack.pop(rand_idx)
            self._indices.pop(rand_idx)
        return cam

    def _run(self):
        while not self._stop.is_set():
            cam = self._pick_one()
            # Warm the cached RGB tensor so the consumer's first access
            # hits the cache instead of decoding.
            try:
                _ = cam.original_image
            except Exception as e:
                # Don't let a single bad frame kill the producer — surface
                # the cam in any case; the consumer will hit the same error
                # synchronously and fail loudly there.
                print(f"[prefetcher] warm failed for {cam.image_name}: {e}")
            try:
                self._q.put(cam, timeout=1.0)
            except queue.Full:
                # Loop back to check stop signal; producer was preempted.
                continue

    def next(self):
        return self._q.get()

    def shutdown(self):
        self._stop.set()
        # Drain queue so worker can exit fast.
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break