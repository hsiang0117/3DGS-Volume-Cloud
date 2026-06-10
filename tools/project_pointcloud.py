#!/usr/bin/env python
"""Project a point cloud into each training camera and render it as white dots
on black, to visually check whether the init points3d.ply lands where the cloud
actually appears in the training images.

Overlay each output PNG against the matching training image (same camera_index,
same resolution) to see alignment / coverage — especially whether the cloud
bottom is seeded with points.

Convention: transforms_*.json stores OpenGL/Blender c2w matrices (camera looks
down -Z, +Y up, +X right), as produced by convert_transforms.py. We invert to
world->camera and project with a pinhole model using camera_angle_x.

Usage:
    python tools/project_pointcloud.py [ply_path] [transforms_json] [out_dir]
Defaults:
    ply        = data/CloudDataset/points3d.ply
    transforms = data/CloudDataset/transforms_train.json
    out_dir    = data/CloudDataset/_pc_projection
One PNG per unique camera_index (first frame seen for that camera).
"""
import os
import sys
import json
import numpy as np
from plyfile import PlyData
from PIL import Image


def main(ply_path, transforms_path, out_dir, dot_radius=1, max_cams=None):
    ply = PlyData.read(ply_path)
    el = ply.elements[0]
    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1).astype(np.float64)
    N = len(xyz)
    print(f"loaded {N} points from {ply_path}")

    with open(transforms_path) as f:
        meta = json.load(f)
    fovx = meta["camera_angle_x"]

    # One frame per camera_index (the point cloud is static; time_index doesn't
    # change geometry, only sun — so any frame for a given cam works).
    frames_by_cam = {}
    for fr in meta["frames"]:
        ci = fr.get("camera_index", len(frames_by_cam))
        if ci not in frames_by_cam:
            frames_by_cam[ci] = fr
    cam_ids = sorted(frames_by_cam)
    if max_cams:
        cam_ids = cam_ids[:max_cams]

    os.makedirs(out_dir, exist_ok=True)
    hom = np.concatenate([xyz, np.ones((N, 1))], axis=1)  # (N,4)

    coverage = []
    for ci in cam_ids:
        fr = frames_by_cam[ci]
        # Image size: read from the actual training image if available, else 1024.
        img_rel = fr["file_path"]
        img_path = os.path.join(os.path.dirname(transforms_path), img_rel)
        if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path += ".png"
        if os.path.exists(img_path):
            with Image.open(img_path) as im:
                W, H = im.size
        else:
            W = H = 1024

        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        cam = (w2c @ hom.T).T[:, :3]  # (N,3) OpenGL camera space, looks down -Z

        # Pinhole projection. focal from horizontal FoV; square pixels.
        focal = (W * 0.5) / np.tan(fovx * 0.5)
        z = cam[:, 2]
        in_front = z < -1e-6  # in front of camera (OpenGL: -Z forward)
        u = focal * (cam[:, 0] / -z) + W * 0.5
        v = -focal * (cam[:, 1] / -z) + H * 0.5
        uu = np.round(u).astype(np.int64)
        vv = np.round(v).astype(np.int64)
        on_img = in_front & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)

        img = np.zeros((H, W), dtype=np.uint8)
        ru, rv = uu[on_img], vv[on_img]
        if dot_radius <= 0:
            img[rv, ru] = 255
        else:
            r = dot_radius
            for du in range(-r, r + 1):
                for dv in range(-r, r + 1):
                    su = np.clip(ru + du, 0, W - 1)
                    sv = np.clip(rv + dv, 0, H - 1)
                    img[sv, su] = 255

        out_path = os.path.join(out_dir, f"cam{ci:02d}_points.png")
        Image.fromarray(img).save(out_path)
        frac = on_img.mean()
        coverage.append((ci, frac))
        print(f"  cam{ci:02d}: {int(on_img.sum())}/{N} points on image ({frac*100:.1f}%) -> {out_path}")

    fr_arr = np.array([f for _, f in coverage])
    print()
    print(f"rendered {len(cam_ids)} cameras to {out_dir}")
    print(f"on-image fraction: min {fr_arr.min()*100:.1f}%  mean {fr_arr.mean()*100:.1f}%  max {fr_arr.max()*100:.1f}%")
    print("Overlay camNN_points.png against camNN/images/<frame>.png to check alignment.")
    print("Low on-image fraction or points off the cloud => init/image mismatch.")


if __name__ == "__main__":
    ply = sys.argv[1] if len(sys.argv) > 1 else "data/CloudDataset/points3d.ply"
    tj = sys.argv[2] if len(sys.argv) > 2 else "data/CloudDataset/transforms_train.json"
    out = sys.argv[3] if len(sys.argv) > 3 else "data/CloudDataset/_pc_projection"
    main(ply, tj, out)
