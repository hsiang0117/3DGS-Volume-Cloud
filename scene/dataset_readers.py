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
import sys
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool
    # Per-frame sun direction in OpenGL/Blender world coordinates
    # ("pointing toward the sun"). Defaults to [0,1,0] for legacy datasets
    # that don't include the field — matches the previous hard-coded behaviour.
    sun_dir: np.array = np.array([0.0, 1.0, 0.0], dtype=np.float32)

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def _parse_one_frame(args):
    """Parse a single transforms.json frame into a CameraInfo.

    Hoisted to a top-level function so a ThreadPoolExecutor can fan out the
    per-frame PIL header read + matrix work across cores. Each frame's I/O is
    a tiny header read (PIL doesn't decode pixels here), but with thousands
    of frames the sequential cost becomes minutes — parallelisation drops
    that to seconds.
    """
    idx, frame, path, is_test = args
    cam_name = os.path.join(path, frame["file_path"])

    # NeRF 'transform_matrix' is a camera-to-world transform
    c2w = np.array(frame["transform_matrix"])
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    image_path = os.path.join(path, cam_name)
    image_name = Path(cam_name).stem
    with Image.open(image_path) as image:
        width, height = image.size

    sun_dir_raw = frame.get("sun_direction", None)
    if sun_dir_raw is None:
        sun_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        sun_dir = np.array(sun_dir_raw, dtype=np.float32)
        norm = np.linalg.norm(sun_dir)
        if norm > 1e-8:
            sun_dir = sun_dir / norm

    return idx, CameraInfo(uid=idx, R=R, T=T, FovY=None, FovX=None,
                            image_path=image_path, image_name=image_name,
                            width=width, height=height, is_test=is_test,
                            sun_dir=sun_dir)


def readCamerasFromTransforms(path, transformsfile, white_background, is_test, extension=".png"):
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

    # Parse frames in parallel — the per-frame work is pure CPU (PIL header
    # read + matrix invert) with no shared state, so a thread pool sized to
    # the CPU count gives a near-linear speedup. For ~3000 frames this drops
    # from ~5 min sequential to <30 s.
    n_workers = min(32, (os.cpu_count() or 4) * 2)
    args_iter = [(idx, frame, path, is_test) for idx, frame in enumerate(frames)]
    results = [None] * len(args_iter)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for idx, cam_info in ex.map(_parse_one_frame, args_iter):
            # FovX is shared across all frames (single camera intrinsic);
            # FovY needs the per-frame image height, so fix both up here.
            fovy = focal2fov(fov2focal(fovx, cam_info.width), cam_info.height)
            results[idx] = cam_info._replace(FovX=fovx, FovY=fovy)

    return results

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # No init point cloud supplied — start from random points inside the scene bbox.
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Blender" : readNerfSyntheticInfo
}