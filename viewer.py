"""
Real-time interactive viewer for trained volumetric-cloud Gaussian splats.

Run:
    pip install viser
    python viewer.py --ply <path/to/point_cloud.ply>
then open http://localhost:8080 in a browser.

Features (v1):
    - PLY loading and free-fly camera (viser's built-in orbit / WASD).
    - FOV and per-Gaussian scaling_modifier sliders.
    - Visualisation modes: RGB | depth | T_light | beta_peak.
    - Static sun direction (hardcoded [0, 1, 0]); T_light is computed once at
      startup and reused every frame.

Not in v1:
    - Per-frame / interactive sun direction.
    - Custom lighting UI.
"""

import os
import argparse
import math
import re
import time
from dataclasses import dataclass

import numpy as np
import torch
import viser

from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render, compute_T_light, compute_T_light_raster, normalized_gaussian_line_integral
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import build_rotation


# --- Pipeline stub (mirrors arguments.PipelineParams, no argparse needed) ---
@dataclass
class _ViewerPipe:
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False
    k_sigma: float = 1.5
    # T_light is supplied via precomputed_T_light (compute_T_light_cache), so
    # render() never reaches its own T_light branch here; tlight_voxel only
    # guards against accidental in-render computation.
    tlight_voxel: bool = True


def _quat_wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert a viser (w, x, y, z) quaternion into a 3×3 rotation matrix."""
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def viser_to_minicam(cam, width: int, height: int, z_near: float = 0.01, z_far: float = 100.0, sun_dir=None) -> MiniCam:
    """Build a 3DGS MiniCam from a viser CameraHandle.

    Both viser and the 3DGS rasterizer use the same camera-local convention:
    +X right, +Y down, +Z forward (OpenCV / COLMAP style). `cam.wxyz` is
    world-from-camera rotation; `cam.fov` is vertical FOV in radians. So we can
    use viser's pose as-is, with no axis flipping.

    (Datasets read via `readCamerasFromTransforms` come from OpenGL/Blender
    transforms and DO need a Y/Z axis flip — but that's a property of the
    on-disk transform_matrix, not of any camera the renderer sees.)
    """
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _quat_wxyz_to_matrix(np.asarray(cam.wxyz, dtype=np.float32))
    c2w[:3, 3] = np.asarray(cam.position, dtype=np.float32)
    w2c = np.linalg.inv(c2w)

    world_view = torch.from_numpy(w2c).float().cuda().T

    fovy = float(cam.fov)
    aspect = float(cam.aspect) if cam.aspect > 0 else (width / max(1, height))
    fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * aspect)

    proj = getProjectionMatrix(znear=z_near, zfar=z_far, fovX=fovx, fovY=fovy).cuda().T
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    return MiniCam(width, height, fovy, fovx, z_near, z_far, world_view, full_proj, sun_dir=sun_dir)


@torch.no_grad()
def compute_T_light_cache(gaussians: GaussianModel, sun_dir: torch.Tensor,
                          use_raster: bool = False, raster_res: int = 512) -> torch.Tensor:
    """Compute T_light for the given sun direction.

    Mirrors the per-Gaussian τ derivation used inside `render()` so the cache
    matches what render() would produce for the same sun direction.

    use_raster selects the light-space rasterized shadow pass (the
    --tlight_raster training path) instead of the 128^3 voxel cache. View a
    model with the SAME T_light source it was trained with — β/albedo
    calibrate against their training-time shadow field, so mixing sources
    shows mis-lit results.
    """
    sun_dir = sun_dir.to(device="cuda", dtype=torch.float32)
    sun_dir = sun_dir / (torch.linalg.norm(sun_dir) + 1e-8)
    s = gaussians.get_scaling
    beta_peak = gaussians.get_extinction
    mass = beta_peak * ((2.0 * math.pi) ** 1.5) * torch.prod(s, dim=1, keepdim=True)
    R = build_rotation(gaussians.get_rotation)                              # (P,3,3)
    R_t = R.transpose(1, 2)
    l_local = torch.matmul(R_t, sun_dir.view(3, 1)).squeeze(-1)             # (P,3)
    line_int_sun = normalized_gaussian_line_integral(s, l_local)
    tau_sun_per_gauss = mass * line_int_sun
    if use_raster:
        T = compute_T_light_raster(
            gaussians.get_xyz,
            tau_sun_per_gauss,
            s,
            gaussians.get_rotation,
            sun_dir,
            image_size=raster_res,
        )
        return T.view(-1, 1)
    return compute_T_light(
        gaussians.get_xyz,
        tau_sun_per_gauss,
        s,
        sun_dir,
        grid_res=128,
    )


def _spherical_to_dir(altitude_deg: float, azimuth_deg: float) -> np.ndarray:
    """Convert (altitude, azimuth) in degrees to a unit "toward the sun" vector
    in OpenGL world coords (Y up, X right, -Z forward).
    altitude=90 → straight up [0,1,0]; altitude=0, azimuth=0 → +X.
    """
    alt = math.radians(altitude_deg)
    az = math.radians(azimuth_deg)
    cy = math.sin(alt)
    horizontal = math.cos(alt)
    cx = horizontal * math.cos(az)
    cz = horizontal * math.sin(az)
    v = np.array([cx, cy, cz], dtype=np.float32)
    v /= max(np.linalg.norm(v), 1e-8)
    return v


def _depth_to_image(depth: torch.Tensor, near: float | None = None, far: float | None = None) -> np.ndarray:
    """Turbo-less depth-to-grayscale conversion. depth is (1, H, W) or (H, W)."""
    depth = depth.detach()
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    valid = depth > 0
    if near is None:
        near = depth[valid].min().item() if valid.any() else 0.0
    if far is None:
        far = depth.max().item() if depth.numel() > 0 else 1.0
    norm = (depth - near) / max(far - near, 1e-6)
    norm = norm.clamp(0.0, 1.0)
    rgb = norm.unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()
    return (rgb * 255.0).astype(np.uint8)


def _tensor_to_image(img_chw: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor in [0, 1] → (H, W, 3) uint8 numpy."""
    img = img_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def _load_training_cameras(ply_path: str, explicit_path: str | None = None) -> tuple[list[dict], str] | tuple[None, None]:
    """Locate cameras.json next to the trained run (../../.. from PLY) and parse it.

    Returns (cam_list, path) or (None, None) if not found / unparseable.
    """
    import json
    if explicit_path:
        candidate = explicit_path
    else:
        candidate = os.path.normpath(os.path.join(os.path.dirname(ply_path), os.pardir, os.pardir, "cameras.json"))
    if not os.path.exists(candidate):
        return None, None
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f), candidate
    except Exception as e:
        print(f"[viewer] failed to parse {candidate}: {e}")
        return None, None


def _load_train_transforms(ply_path: str) -> tuple[dict | None, str | None]:
    """Walk back from PLY path to find the source dataset's transforms_train.json.

    cameras.json stores image_name = file_stem, which collapses multi-camera /
    multi-time datasets onto duplicate strings (e.g. all 49 cams sharing
    "0000".."0060"). We need the original transforms file to recover the
    (camera_index, time_index) double-key and per-frame c2w / sun_dir.

    Recognises the layout used by Scene.__init__: <model>/cfg_args records
    the dataset source path. If we can't find that, falls back to a few
    common adjacent locations.
    """
    import json
    candidates = []
    # 1. <model>/cfg_args points to the source_path string
    cfg = os.path.normpath(os.path.join(os.path.dirname(ply_path), os.pardir, os.pardir, "cfg_args"))
    if os.path.exists(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                txt = f.read()
            # cfg_args is `Namespace(source_path='C:/.../CloudDataset', ...)`
            import re
            m = re.search(r"source_path=['\"]([^'\"]+)['\"]", txt)
            if m:
                candidates.append(os.path.join(m.group(1), "transforms_train.json"))
        except Exception:
            pass
    # 2. cwd / data/<basename>/transforms_train.json
    candidates.append("data/CloudDataset/transforms_train.json")
    for p in candidates:
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f), p
            except Exception as e:
                print(f"[viewer] failed to parse {p}: {e}")
    return None, None


def _transforms_frame_to_viser_pose(frame: dict, fov_x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Convert a transforms.json frame to viser's (position, up, look_at, fov_y).

    `transform_matrix` is OpenGL c2w (Y up, -Z forward). viser uses OpenCV
    convention internally but accepts (position, up, look_at) world-space
    triples regardless, so we read the local axes from the c2w columns.
    """
    M = np.asarray(frame["transform_matrix"], dtype=np.float32)
    pos = M[:3, 3]
    # OpenGL local axes: col0=right (+X), col1=up (+Y), col2=back (+Z) → forward = -col2
    up = M[:3, 1]
    forward = -M[:3, 2]
    look_at = pos + forward * 10.0
    return pos.astype(np.float32), up.astype(np.float32), look_at.astype(np.float32), float(fov_x)


def _training_cam_to_viser_pose(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Convert a cameras.json entry to viser's (position, up_direction, look_at, fov_y).

    cameras.json stores `rotation` as the COLMAP-style camera-to-world rotation
    (columns are camera local axes in world coords; +X right, +Y down, +Z forward).
    viser also uses OpenCV camera, so we can read out forward/up directly.

    fov_y comes from focal length and image height: fov_y = 2 * atan(H / 2 / fy).
    """
    pos = np.asarray(cam["position"], dtype=np.float32)
    rot = np.asarray(cam["rotation"], dtype=np.float32)        # (3,3), c2w rotation
    forward = rot[:, 2]                                          # camera +Z in world
    down = rot[:, 1]                                             # camera +Y in world
    up = -down                                                   # world "up" for viser
    # Pick a look-at point in front of camera; distance is arbitrary for orientation.
    look_at = pos + forward * 10.0
    fov_y = 2.0 * math.atan(cam["height"] / (2.0 * cam["fy"]))
    return pos, up.astype(np.float32), look_at.astype(np.float32), float(fov_y)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True, help="Path to trained .ply (point_cloud.ply).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=1024, help="Render resolution width.")
    parser.add_argument("--height", type=int, default=768, help="Render resolution height.")
    parser.add_argument("--bg", choices=["black", "white"], default="black")
    parser.add_argument("--cameras_json", default=None,
                        help="Optional explicit path to cameras.json (auto-located next to PLY by default).")
    parser.add_argument("--tlight", choices=["auto", "voxel", "raster"], default="auto",
                        help="T_light source. 'auto' (default) reads the training run's cfg_args "
                             "next to the PLY and matches what the model was trained with; "
                             "'voxel' = 128^3 grid cache, 'raster' = light-space shadow pass.")
    args = parser.parse_args()

    # Resolve the T_light source. Models calibrate β/albedo against their
    # training-time shadow field, so the viewer must use the same source.
    # cfg_args from current runs carries tlight_voxel (raster is the default);
    # runs from the transition window carried tlight_raster=True; runs older
    # than the raster pass carry neither and are voxel-trained.
    use_raster_tlight = args.tlight == "raster"
    tlight_raster_res = 512
    if args.tlight == "auto":
        # .../<run>/point_cloud/iteration_N/point_cloud.ply -> <run>/cfg_args
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.ply))))
        cfg_path = os.path.join(run_dir, "cfg_args")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = f.read()
                if "tlight_voxel" in cfg:
                    use_raster_tlight = "tlight_voxel=True" not in cfg
                else:
                    use_raster_tlight = "tlight_raster=True" in cfg
                m = re.search(r"tlight_raster_res=(\d+)", cfg)
                if m:
                    tlight_raster_res = int(m.group(1))
            except Exception as e:
                print(f"[viewer] cfg_args unreadable ({e}); defaulting to voxel T_light.")
        else:
            print("[viewer] No cfg_args found next to PLY; defaulting to voxel T_light.")
    print(f"[viewer] T_light source: {'raster' if use_raster_tlight else 'voxel'}"
          f"{f' ({tlight_raster_res}^2)' if use_raster_tlight else ''}")

    # --- Load Gaussians -----------------------------------------------------
    print(f"[viewer] Loading {args.ply} ...")
    gaussians = GaussianModel()
    gaussians.load_ply(args.ply)
    P = gaussians.get_xyz.shape[0]
    print(f"[viewer] Loaded {P} Gaussians.")

    # --- Precompute T_light for the initial sun direction --------------------
    initial_sun = _spherical_to_dir(altitude_deg=90.0, azimuth_deg=0.0)  # straight up
    print(f"[viewer] Precomputing T_light (sun={initial_sun.tolist()}) ...")
    t0 = time.time()
    T_light = compute_T_light_cache(
        gaussians, torch.from_numpy(initial_sun).cuda(),
        use_raster=use_raster_tlight, raster_res=tlight_raster_res,
    ).detach()
    torch.cuda.synchronize()
    print(f"[viewer] T_light ready in {time.time() - t0:.2f}s. Shape = {tuple(T_light.shape)}")

    # Max β_peak for normalising the beta_peak visualisation channel.
    beta_max = max(gaussians.get_extinction.detach().max().item(), 1e-6)

    # --- Compute a sensible initial camera pose from cloud bounds ----------
    # Use 1st–99th percentile bounds to shrug off floater outliers, then sit the
    # camera a few radii away looking at the cloud centre.
    with torch.no_grad():
        xyz_np = gaussians.get_xyz.detach().cpu().numpy()
    lo = np.percentile(xyz_np, 1, axis=0)
    hi = np.percentile(xyz_np, 99, axis=0)
    cloud_center = ((lo + hi) * 0.5).astype(np.float32)
    cloud_radius = float(max(np.linalg.norm(hi - lo) * 0.5, 1e-3))
    # Offset direction in viser (OpenGL) world: slightly above + to the side, looking toward -Z.
    _offset_dir = np.array([0.8, 0.4, 1.0], dtype=np.float32)
    _offset_dir /= np.linalg.norm(_offset_dir)
    default_cam_pos = cloud_center + _offset_dir * (cloud_radius * 2.8)
    default_cam_lookat = cloud_center.copy()

    # Clipping planes scaled to cloud size. Fixed 0.01/100 breaks large scenes:
    # a camera sitting 2.8·radius from centre is well beyond zfar=100 when radius
    # exceeds ~35, and every Gaussian gets culled.
    z_near = max(0.01, 0.05 * cloud_radius)
    z_far = max(100.0, 20.0 * cloud_radius)

    print(
        f"[viewer] Cloud bounds: center={cloud_center.tolist()}, "
        f"radius={cloud_radius:.3f}. Initial camera pos={default_cam_pos.tolist()}.\n"
        f"[viewer] Clipping planes: znear={z_near:.3f}, zfar={z_far:.3f}."
    )

    # --- Optional: training cameras for snap-to-pose comparison --------------
    # Prefer the original transforms_train.json: it preserves the
    # (camera_index, time_index) double-key plus per-frame sun_dir, which the
    # collapsed cameras.json does not (img_name there is just the file stem,
    # so multi-time datasets get duplicate keys).
    train_transforms, train_transforms_path = _load_train_transforms(args.ply)
    # Bucket frames by camera_index. `cam_frames[c]` is a dict {time_idx: frame}
    cam_frames: dict[int, dict[int, dict]] = {}
    train_fov_x = math.pi / 2.0  # fallback if json is missing camera_angle_x
    if train_transforms is not None:
        train_fov_x = float(train_transforms.get("camera_angle_x", train_fov_x))
        for fr in train_transforms.get("frames", []):
            ci = int(fr.get("camera_index", -1))
            ti = int(fr.get("time_index", 0))
            cam_frames.setdefault(ci, {})[ti] = fr
        n_cams = len(cam_frames)
        n_times = max((max(d.keys()) for d in cam_frames.values()), default=-1) + 1
        print(f"[viewer] Loaded {n_cams} cameras × {n_times} times from {train_transforms_path}.")
    else:
        print("[viewer] No transforms_train.json found — 'Snap to training cam' will be disabled.")

    # Legacy fallback (cameras.json) — only used if transforms_train.json was missing.
    train_cams, train_cams_path = _load_training_cameras(args.ply, args.cameras_json)
    if not cam_frames and train_cams:
        train_cam_names = [c.get("img_name", str(c.get("id", i))) for i, c in enumerate(train_cams)]
        print(f"[viewer] Falling back to cameras.json ({len(train_cams)} entries) from {train_cams_path}.")
    else:
        train_cam_names = []

    pipe = _ViewerPipe()
    bg_color = torch.tensor(
        [1.0, 1.0, 1.0] if args.bg == "white" else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )

    # --- viser server + GUI -------------------------------------------------
    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    # Sun direction arrow. Drawn outside the cloud bbox, pointing toward the
    # cloud centre (i.e. along the *light propagation* direction). Updated by
    # mutating the handle's `position` + `wxyz`; the geometry itself stays
    # fixed in the arrow's local frame.
    arrow_len = max(cloud_radius * 0.8, 0.5)
    arrow_offset = cloud_radius * 1.6  # stand-off distance from cloud centre

    # Local arrow geometry: tail at origin, head at +arrow_len along local +X.
    # We then rotate the whole arrow so local +X aligns with the world
    # "light propagation" direction (= -sun_dir), and translate so the tail
    # sits at cloud_center + sun_dir * arrow_offset (outside the cloud,
    # opposite the sun).
    _local_arrow_points = np.array([[[0.0, 0.0, 0.0],
                                     [arrow_len, 0.0, 0.0]]], dtype=np.float32)

    def _quat_align_x_to(target: np.ndarray) -> np.ndarray:
        """Quaternion (w, x, y, z) that rotates local +X to `target` (unit)."""
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t = target.astype(np.float64)
        t = t / max(np.linalg.norm(t), 1e-8)
        dot = float(np.dot(x_axis, t))
        if dot > 0.9999:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if dot < -0.9999:
            # 180° around any axis perpendicular to X; pick Y.
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(x_axis, t)
        axis = axis / max(np.linalg.norm(axis), 1e-8)
        angle = math.acos(max(-1.0, min(1.0, dot)))
        s = math.sin(angle / 2.0)
        return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)

    def _sun_arrow_pose(sun_dir_np: np.ndarray):
        s = sun_dir_np / max(np.linalg.norm(sun_dir_np), 1e-8)
        position = (cloud_center + s * arrow_offset).astype(np.float32)
        # Light propagation = -sun_dir; rotate local +X → -s.
        wxyz = _quat_align_x_to(-s)
        return position, wxyz

    initial_pos, initial_wxyz = _sun_arrow_pose(initial_sun)
    sun_arrow = server.scene.add_arrows(
        "/sun_arrow",
        points=_local_arrow_points,
        colors=np.array([255, 200, 60], dtype=np.uint8),
        shaft_radius=arrow_len * 0.04,
        head_radius=arrow_len * 0.10,
        head_length=arrow_len * 0.20,
        position=initial_pos,
        wxyz=initial_wxyz,
    )

    gui_mode = server.gui.add_dropdown(
        "View mode",
        options=("rgb", "T_light", "beta_peak", "depth"),
        initial_value="rgb",
    )
    # Snap-to-training-cam controls. If transforms_train.json is available we
    # show one dropdown of unique camera_index values + a time slider. This
    # keeps the option list at ~49 instead of 49×61=2989 (which used to
    # disconnect the viser websocket on connect).
    gui_train_cam = None              # legacy cameras.json dropdown
    gui_train_cam_text = None         # fallback text input
    gui_train_cam_idx = None          # new: unique camera_index dropdown
    gui_train_time = None             # new: time slider
    if cam_frames:
        sorted_cams = sorted(cam_frames.keys())
        gui_train_cam_idx = server.gui.add_dropdown(
            "Snap: camera",
            options=tuple(["(free)"] + [f"cam{ci:02d}" for ci in sorted_cams]),
            initial_value="(free)",
            hint="Pick a viewpoint by camera index. Combine with the time slider "
                 "below to choose which TOD frame to snap to.",
        )
        n_times = max((max(d.keys()) for d in cam_frames.values()), default=0) + 1
        gui_train_time = server.gui.add_slider(
            "Snap: time index", min=0, max=max(n_times - 1, 0), step=1, initial_value=0,
            hint=f"Time-of-day frame within the chosen camera (0..{n_times-1}).",
        )
    elif train_cams:
        # Legacy cameras.json path (single-light datasets).
        DROPDOWN_LIMIT = 256
        if len(train_cam_names) <= DROPDOWN_LIMIT:
            gui_train_cam = server.gui.add_dropdown(
                "Snap to training cam",
                options=tuple(["(free)"] + train_cam_names),
                initial_value="(free)",
                hint="Pick a training camera by image name.",
            )
        else:
            gui_train_cam_text = server.gui.add_text(
                "Snap to training cam",
                initial_value="",
                hint=(f"Type a training camera img_name and press Enter. "
                      f"Dropdown disabled (have {len(train_cam_names)} cams)."),
            )
    gui_scaling = server.gui.add_slider(
        "Gaussian scale", min=0.1, max=2.0, step=0.05, initial_value=1.0,
        hint="Multiplies all Gaussian scales at render time (visual only, does not modify stored model).",
    )
    gui_ksigma = server.gui.add_slider(
        "Sort k·σ clamp", min=0.0, max=3.0, step=0.1, initial_value=0.0,
        hint="Per-tile max-response sort: how far t* may deviate from centre depth, "
             "in units of σ along the view ray. 0 = stock 3DGS centre sort (default; "
             "popping is controlled by the aniso regulariser instead). "
             ">0 re-enables the per-tile shift but can show blocky tile-edge artefacts.",
    )
    gui_sun_alt = server.gui.add_slider(
        "Sun altitude (°)", min=-90.0, max=90.0, step=1.0, initial_value=90.0,
        hint="Sun elevation above horizon. 90 = straight up (legacy default), "
             "0 = horizon, negative = below horizon (cloud back-lit / dark). "
             "Changes auto-recompute T_light (~0.5s lag).",
    )
    gui_sun_az = server.gui.add_slider(
        "Sun azimuth (°)", min=-180.0, max=180.0, step=5.0, initial_value=0.0,
        hint="Sun rotation around the up axis. Only meaningful when altitude < 90. "
             "Changes auto-recompute T_light (~0.5s lag).",
    )
    gui_res = server.gui.add_slider(
        "Render size", min=256, max=1920, step=32, initial_value=args.width,
        hint="Render resolution width; height is derived from client aspect.",
    )
    gui_bgcolor = server.gui.add_rgb(
        "Background color",
        initial_value=(255, 255, 255) if args.bg == "white" else (0, 0, 0),
        hint="Rasterizer background colour. Handy for inspecting cloud edges / "
             "discrete Gaussian ellipsoids against different backdrops.",
    )
    gui_reset = server.gui.add_button(
        "Reset camera",
        hint="Re-fit the camera to the cloud's bounding box.",
    )
    gui_lock = server.gui.add_checkbox(
        "Orbit-only (lock to centre)",
        initial_value=True,
        hint="When on, the camera always looks at the cloud centre — only orbit and zoom are allowed (no panning / free flight).",
    )
    gui_record = server.gui.add_checkbox(
        "🔴 Record video",
        initial_value=False,
        hint="Capture the rendered view to an MP4 (30 fps, wall-clock timing) "
             "while checked. Untick to stop and save into ./recordings/. "
             "Sun / view / mode changes are all captured; render size is "
             "locked to the first recorded frame.",
    )
    gui_record_status = server.gui.add_text("Recording", initial_value="-")
    gui_record_status.disabled = True
    gui_fps = server.gui.add_text("FPS", initial_value="-")
    gui_fps.disabled = True
    gui_ngauss = server.gui.add_text("# Gaussians", initial_value=f"{P:,}")
    gui_ngauss.disabled = True

    # Flag set when any input that affects the image changes.
    state = {
        "needs_render": True,
        "last_render_time": 0.0,
        "diag_done": False,
        "sun_dir": initial_sun.copy(),     # numpy (3,) — current cached sun
        "T_light": T_light,                # current cached T_light tensor
    }

    # --- Video recorder ------------------------------------------------------
    # Wall-clock-faithful capture of the FIRST client's rendered frames: the
    # render loop is event-driven (only draws on changes), so the recorder
    # duplicates the last frame to fill idle time. Frames are buffered in RAM
    # (HxWx3 uint8; ~2.7 MB each at 1080p — minutes of footage are fine) and
    # encoded once on stop, keeping the interactive loop light.
    REC_FPS = 30

    class _Recorder:
        def __init__(self):
            self.active = False
            self.frames = []        # list[np.ndarray HxWx3 uint8]
            self.size = None        # (w, h) locked at first frame
            self.t_start = 0.0

        def start(self):
            self.active = True
            self.frames = []
            self.size = None
            self.t_start = time.time()
            print("[viewer] recording started")

        def add(self, img_np):
            if not self.active:
                return
            h, w = img_np.shape[:2]
            if self.size is None:
                # Even dimensions required by H.264/mp4v chroma subsampling.
                self.size = (w - (w % 2), h - (h % 2))
            tw, th = self.size
            if (w, h) != (tw, th):
                img_np = img_np[:th, :tw] if (w >= tw and h >= th) else None
                if img_np is None:
                    return  # render size shrank mid-recording; skip frame
            # Fill wall-clock gaps so playback timing matches what you saw.
            target_n = max(1, int(round((time.time() - self.t_start) * REC_FPS)))
            last = img_np[:th, :tw]
            while len(self.frames) < target_n:
                self.frames.append(last)

        def stop_and_save(self):
            self.active = False
            n = len(self.frames)
            if n == 0 or self.size is None:
                print("[viewer] recording stopped: no frames captured")
                return None
            import cv2
            os.makedirs("recordings", exist_ok=True)
            path = os.path.join(
                "recordings", time.strftime("cloud_%Y%m%d_%H%M%S") + ".mp4")
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, self.size)
            for f in self.frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            dur = n / REC_FPS
            print(f"[viewer] recording saved: {path} ({n} frames, {dur:.1f}s)")
            self.frames = []
            return path, n, dur

    recorder = _Recorder()

    def _toggle_record(_):
        if gui_record.value:
            recorder.start()
            gui_record_status.value = "recording..."
        else:
            result = recorder.stop_and_save()
            if result:
                path, n, dur = result
                gui_record_status.value = f"saved {os.path.basename(path)} ({dur:.0f}s)"
            else:
                gui_record_status.value = "no frames"
        state["needs_render"] = True

    gui_record.on_update(_toggle_record)

    def _mark_dirty(*_):
        state["needs_render"] = True

    def _apply_sun(*_):
        new_sun = _spherical_to_dir(gui_sun_alt.value, gui_sun_az.value)
        with torch.no_grad():
            new_cache = compute_T_light_cache(
                gaussians, torch.from_numpy(new_sun).cuda(),
                use_raster=use_raster_tlight, raster_res=tlight_raster_res,
            ).detach()
        torch.cuda.synchronize()
        state["sun_dir"] = new_sun
        state["T_light"] = new_cache
        state["needs_render"] = True
        # Move the arrow to follow the new direction.
        new_pos, new_wxyz = _sun_arrow_pose(new_sun)
        sun_arrow.position = new_pos
        sun_arrow.wxyz = new_wxyz
        print(f"[viewer] sun_dir → {new_sun.tolist()} (alt={gui_sun_alt.value:.1f}°, az={gui_sun_az.value:.1f}°)")

    gui_sun_alt.on_update(_apply_sun)
    gui_sun_az.on_update(_apply_sun)

    gui_mode.on_update(_mark_dirty)
    gui_scaling.on_update(_mark_dirty)
    gui_ksigma.on_update(_mark_dirty)
    gui_res.on_update(_mark_dirty)
    gui_bgcolor.on_update(_mark_dirty)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Tell viser the world up axis BEFORE setting the pose: viser's
        # default is +Z up, but our dataset (UE → OpenGL via convert_transforms.py)
        # uses +Y up, matching the trained Gaussians. Without this, the camera
        # axis is computed against the wrong "up" and the rendered image is
        # tilted/flipped relative to train's periodic saves.
        client.camera.up_direction = (0.0, 1.0, 0.0)
        client.camera.position = default_cam_pos
        client.camera.look_at = default_cam_lookat

        @client.camera.on_update
        def _(_):
            # When "Orbit-only" is on, snap look_at back to the cloud centre so
            # the user can only orbit + zoom (no panning / free translation).
            # viser's look_at setter is a no-op when the new value matches the
            # current one within its tolerance, so assigning here converges in
            # at most one extra frame without recursion.
            if gui_lock.value:
                if not np.allclose(
                    np.asarray(client.camera.look_at, dtype=np.float32),
                    cloud_center,
                    atol=1e-3,
                ):
                    client.camera.look_at = cloud_center
            state["needs_render"] = True

    @gui_reset.on_click
    def _(_):
        for c in server.get_clients().values():
            c.camera.up_direction = (0.0, 1.0, 0.0)
            c.camera.position = default_cam_pos
            c.camera.look_at = default_cam_lookat
        state["needs_render"] = True

    def _snap_to_cam(sel: str):
        if not sel or sel == "(free)":
            return
        cam = next((c for c in train_cams if c.get("img_name") == sel), None)
        if cam is None:
            print(f"[viewer] training cam '{sel}' not found")
            return
        pos, up, look_at, fov_y = _training_cam_to_viser_pose(cam)
        # Snap-to needs the look_at to match the training camera's forward, so
        # temporarily disable orbit-lock — otherwise the on_update callback
        # would immediately drag look_at back to the cloud centre.
        gui_lock.value = False
        for c in server.get_clients().values():
            c.camera.up_direction = up
            c.camera.position = pos
            c.camera.look_at = look_at
            c.camera.fov = fov_y
        state["needs_render"] = True
        print(f"[viewer] Snapped to training cam '{sel}': pos={pos.tolist()} fov_y={math.degrees(fov_y):.1f}deg "
              f"(GT: {cam['width']}x{cam['height']})")

    def _snap_to_cam_time():
        """New path: snap by (camera_index, time_index) using transforms.json frames."""
        if not cam_frames or gui_train_cam_idx is None:
            return
        sel = gui_train_cam_idx.value
        if sel == "(free)":
            return
        try:
            ci = int(sel.replace("cam", ""))
        except ValueError:
            print(f"[viewer] bad camera selector '{sel}'")
            return
        ti = int(gui_train_time.value) if gui_train_time is not None else 0
        frame = cam_frames.get(ci, {}).get(ti)
        if frame is None:
            print(f"[viewer] no frame for cam{ci:02d}, time={ti}")
            return
        pos, up, look_at, fov_y = _transforms_frame_to_viser_pose(frame, train_fov_x)
        gui_lock.value = False
        for c in server.get_clients().values():
            c.camera.up_direction = up
            c.camera.position = pos
            c.camera.look_at = look_at
            c.camera.fov = fov_y
        # Also push the frame's sun_dir into the global sun state so T_light
        # matches what training saw. Falls back to the slider value if the
        # frame lacks sun_direction.
        sd = frame.get("sun_direction")
        if sd is not None:
            new_sun = np.asarray(sd, dtype=np.float32)
            new_sun /= max(np.linalg.norm(new_sun), 1e-8)
            with torch.no_grad():
                new_cache = compute_T_light_cache(
                    gaussians, torch.from_numpy(new_sun).cuda(),
                    use_raster=use_raster_tlight, raster_res=tlight_raster_res,
                ).detach()
            state["sun_dir"] = new_sun
            state["T_light"] = new_cache
            new_pos, new_wxyz = _sun_arrow_pose(new_sun)
            sun_arrow.position = new_pos
            sun_arrow.wxyz = new_wxyz
        state["needs_render"] = True
        print(f"[viewer] Snapped to cam{ci:02d}, time={ti}: pos={pos.tolist()} fov={math.degrees(fov_y):.1f}deg")

    if gui_train_cam is not None:
        @gui_train_cam.on_update
        def _(_):
            _snap_to_cam(gui_train_cam.value)

    if gui_train_cam_text is not None:
        @gui_train_cam_text.on_update
        def _(_):
            _snap_to_cam(gui_train_cam_text.value.strip())

    if gui_train_cam_idx is not None:
        gui_train_cam_idx.on_update(lambda _: _snap_to_cam_time())
        if gui_train_time is not None:
            gui_train_time.on_update(lambda _: _snap_to_cam_time())

    # --- Render loop --------------------------------------------------------
    print(f"[viewer] Serving at http://localhost:{args.port}")
    while True:
        if not state["needs_render"]:
            if recorder.active:
                # While recording, idle frames still advance wall-clock time;
                # duplicate-fill happens inside recorder.add() on next render.
                # Re-render at the capture cadence so slow changes (e.g. sun
                # recompute) appear smoothly instead of as a single jump.
                time.sleep(1.0 / REC_FPS)
                state["needs_render"] = True
                continue
            time.sleep(0.01)
            continue
        state["needs_render"] = False

        clients = server.get_clients()
        if not clients:
            time.sleep(0.1)
            continue

        t_frame = time.time()
        first_client = True
        for client in clients.values():
            cam = client.camera
            # Keep aspect consistent with client window to avoid distortion.
            render_w = int(gui_res.value)
            render_h = max(1, int(render_w / max(cam.aspect, 1e-3)))
            current_sun = state["sun_dir"]
            current_T_light = state["T_light"]
            mini = viser_to_minicam(cam, render_w, render_h, z_near=z_near, z_far=z_far, sun_dir=current_sun)
            scaling_mod = float(gui_scaling.value)
            pipe.k_sigma = float(gui_ksigma.value)
            mode = gui_mode.value

            # Update background colour in place (RGB 0-255 -> 0-1) so we don't
            # allocate a new GPU tensor every frame.
            _bg = gui_bgcolor.value
            bg_color[0] = _bg[0] / 255.0
            bg_color[1] = _bg[1] / 255.0
            bg_color[2] = _bg[2] / 255.0

            if mode == "T_light":
                # T_light is (P, 1); broadcast to RGB grayscale.
                override = current_T_light.expand(-1, 3).contiguous()
            elif mode == "beta_peak":
                beta = gaussians.get_extinction.detach() / beta_max
                override = beta.clamp(0.0, 1.0).expand(-1, 3).contiguous()
            else:
                override = None

            try:
                with torch.no_grad():
                    out = render(
                        mini, gaussians, pipe, bg_color,
                        scaling_modifier=scaling_mod,
                        override_color=override,
                        precomputed_T_light=current_T_light,
                    )
            except RuntimeError as e:
                print(f"[viewer] render failed: {e}")
                continue

            if not state["diag_done"]:
                rend = out["render"]
                dep = out.get("depth")
                vis = out.get("visibility_filter")
                n_visible = int(vis.shape[0]) if vis is not None else -1
                print(
                    f"[viewer] DIAG first frame: "
                    f"cam_pos={np.asarray(cam.position).tolist()}, "
                    f"cam_wxyz={np.asarray(cam.wxyz).tolist()}, "
                    f"cam_fov={float(cam.fov):.3f} rad, aspect={float(cam.aspect):.3f} | "
                    f"render min/max/mean={rend.min().item():.3f}/{rend.max().item():.3f}/{rend.mean().item():.3f} | "
                    f"visible Gaussians={n_visible}/{P}"
                )
                if dep is not None:
                    dep_valid = dep[dep > 0]
                    if dep_valid.numel() > 0:
                        print(
                            f"[viewer] DIAG depth: min={dep_valid.min().item():.3f}, "
                            f"max={dep.max().item():.3f}, non-zero pixels={dep_valid.numel()}"
                        )
                    else:
                        print("[viewer] DIAG depth: NO non-zero depth pixels (all Gaussians culled?)")
                state["diag_done"] = True

            if mode == "depth":
                img_np = _depth_to_image(out["depth"])
            else:
                img_np = _tensor_to_image(out["render"])

            if first_client:
                recorder.add(img_np)
                if recorder.active:
                    gui_record_status.value = (
                        f"recording {time.time() - recorder.t_start:.0f}s "
                        f"({len(recorder.frames)} frames)")
                first_client = False

            client.scene.set_background_image(img_np, format="jpeg")

        dt = time.time() - t_frame
        if dt > 0:
            gui_fps.value = f"{1.0 / dt:5.1f}"


if __name__ == "__main__":
    main()
