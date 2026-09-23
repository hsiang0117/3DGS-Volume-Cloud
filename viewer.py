"""
Real-time interactive viewer for trained volumetric-cloud Gaussian splats.

Run:
    pip install viser
    python viewer.py --ply <path/to/point_cloud.ply>
then open http://localhost:8080 in a browser.

Provides PLY loading, free-fly camera (viser orbit / WASD), visualisation modes
(RGB | depth | T_light | sigma_t), and interactive sun direction (T_light
recomputed on change).
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # enable cv2 EXR before cv2 is imported (sky_backdrop / recorder)
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
from utils.lightpass_config import saved_lightpass_settings
from gaussian_renderer import render, compute_T_light_voxel, compute_T_light_raster, normalized_gaussian_line_integral
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import build_rotation
from sky_backdrop import SkyBackdrop, camera_ray_dirs


# --- Pipeline stub (mirrors arguments.PipelineParams, no argparse needed) ---
@dataclass
class _ViewerPipe:
    # T_light comes in via precomputed_T_light (compute_T_light_cache), so render()
    # never reaches its own T_light branch.
    tlight_voxel: bool = True
    # When True, render() shades in HDR linear and applies a tonemap curve to the
    # final RGB. tonemap_aces = fixed Narkowicz; tonemap_learnable = per-model
    # learned coeffs restored by load_ply (pc.apply_tonemap). Mutated per frame.
    tonemap_aces: bool = False
    tonemap_learnable: bool = False
    # Stage-2 environment lighting: render() adds the learned global sky (T_sun sun
    # transmittance ⊙ sun_term + ω·Σ E_lm·V_lm in-scatter) on top of the frozen sun
    # shading; auto-set when the model carries env sidecars.
    env_lighting: bool = False


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


def viser_to_minicam(cam, width: int, height: int, z_near: float = 0.01, z_far: float = 100.0, v_l=None, aspect=None) -> MiniCam:
    """Build a 3DGS MiniCam from a viser CameraHandle.

    Both viser and the 3DGS rasterizer use the same camera-local convention:
    +X right, +Y down, +Z forward (OpenCV / COLMAP style). `cam.wxyz` is
    world-from-camera rotation; `cam.fov` is vertical FOV in radians. So we can
    use viser's pose as-is, with no axis flipping.

    `aspect` overrides the horizontal FOV's aspect ratio: None uses the live
    canvas aspect; pass an explicit value (e.g. 1.0 for a square still) so fovx
    matches the requested width/height — required for still capture, so the
    rasterizer FOV and the sky-backdrop rays (which derive their own aspect from
    width/height) agree.
    """
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _quat_wxyz_to_matrix(np.asarray(cam.wxyz, dtype=np.float32))
    c2w[:3, 3] = np.asarray(cam.position, dtype=np.float32)
    w2c = np.linalg.inv(c2w)

    world_view = torch.from_numpy(w2c).float().cuda().T

    fovy = float(cam.fov)
    if aspect is None:
        aspect = float(cam.aspect) if cam.aspect > 0 else (width / max(1, height))
    fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * aspect)

    proj = getProjectionMatrix(znear=z_near, zfar=z_far, fovX=fovx, fovY=fovy).cuda().T
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    return MiniCam(width, height, fovy, fovx, z_near, z_far, world_view, full_proj, v_l=v_l)


@torch.no_grad()
def compute_T_light_cache(gaussians: GaussianModel, v_l: torch.Tensor,
                          use_raster: bool = False, raster_res: int = 512, tau_filter: bool = False, filter_variance: float = 0.0) -> torch.Tensor:
    """Compute T_light for the given sun direction.

    Mirrors the per-Gaussian τ derivation inside render() so the cache matches
    what render() produces for the same sun direction.

    use_raster selects the light-space rasterized shadow pass instead of the
    128^3 voxel cache. Contract: view a model with the SAME T_light source it was
    trained with, since σ_t/albedo calibrate against their training-time shadow
    field.
    """
    v_l = v_l.to(device="cuda", dtype=torch.float32)
    v_l = v_l / (torch.linalg.norm(v_l) + 1e-8)
    s = gaussians.get_scaling
    sigma_t = gaussians.get_sigma_t
    mass = sigma_t * ((2.0 * math.pi) ** 1.5) * torch.prod(s, dim=1, keepdim=True)
    R = build_rotation(gaussians.get_rotation)                              # (P,3,3)
    R_t = R.transpose(1, 2)
    l_local = torch.matmul(R_t, v_l.view(3, 1)).squeeze(-1)             # (P,3)
    line_int_v_l = normalized_gaussian_line_integral(s, l_local)
    tau_v_l = mass * line_int_v_l
    if use_raster:
        T = compute_T_light_raster(
            gaussians.get_xyz,
            tau_v_l,
            s,
            gaussians.get_rotation,
            v_l,
            image_size=raster_res,
            tau_filter=tau_filter,
            filter_variance=filter_variance,
        )
        return T.view(-1, 1)
    return compute_T_light_voxel(
        gaussians.get_xyz,
        tau_v_l,
        s,
        v_l,
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
    multi-time datasets onto duplicate strings. The original transforms file is
    needed to recover the (camera_index, time_index) double-key and per-frame
    c2w / v_l.

    Reads the Scene.__init__ layout: <model>/cfg_args records the dataset source
    path; also tries a couple of common adjacent locations.
    """
    import json
    candidates = []
    cfg = os.path.normpath(os.path.join(os.path.dirname(ply_path), os.pardir, os.pardir, "cfg_args"))
    if os.path.exists(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                txt = f.read()
            import re
            m = re.search(r"source_path=['\"]([^'\"]+)['\"]", txt)
            if m:
                candidates.append(os.path.join(m.group(1), "transforms_train.json"))
        except Exception:
            pass
    # Last-resort fallback only; the cfg_args source_path above is tried first
    # and this repo always writes it. Point at the dataset actually in-repo.
    candidates.append("data/CloudDatasetUniform/transforms_train.json")
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
    # OpenGL c2w: col0=right, col1=up, col2=back, so forward = -col2.
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
    look_at = pos + forward * 10.0
    fov_y = 2.0 * math.atan(cam["height"] / (2.0 * cam["fy"]))
    return pos, up.astype(np.float32), look_at.astype(np.float32), float(fov_y)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True, help="Path to trained .ply (point_cloud.ply).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=None,
                        help="Initial value of the 'Max render size' slider (cap on the rendered "
                             "longer side). The render itself matches the browser canvas, so this "
                             "only seeds the slider; omit to use the built-in default.")
    parser.add_argument("--height", type=int, default=768,
                        help="Unused: the viewport follows the browser canvas aspect and is capped "
                             "by the 'Max render size' slider. Accepted for compatibility.")
    parser.add_argument("--bg", choices=["black", "white"], default="black")
    parser.add_argument("--cameras_json", default=None,
                        help="Optional explicit path to cameras.json. By default it is looked up "
                             "in the RUN directory — the third level above the PLY "
                             "(<run>/point_cloud/iteration_N/point_cloud.ply), which is where "
                             "Scene writes it — not beside the PLY itself.")
    parser.add_argument("--tlight", choices=["auto", "voxel", "raster"], default="auto",
                        help="T_light source. 'auto' (default) reads the training run's cfg_args "
                             "(same RUN directory as above) and matches what the model was "
                             "trained with; 'voxel' = 128^3 grid cache, "
                             "'raster' = light-space shadow pass.")
    parser.add_argument("--tonemap", choices=["auto", "on", "off"], default="auto",
                        help="Output tonemap. 'auto' (default) reads tonemap_aces / tonemap_learnable "
                             "from the run's cfg_args and uses the curve the model was trained with "
                             "(learnable coeffs from tonemap.json if present, else fixed Narkowicz "
                             "ACES); 'on'/'off' force tonemap on/off. Tonemapped models look too "
                             "bright / low-contrast without it.")
    parser.add_argument("--sky_dir", default=None,
                        help="Optional directory of captured HDR sky cubemaps "
                             "(sky.json manifest + sky_alt*_*.exr faces, linear "
                             "SceneColorHDR, captured from UE at one 6-face cube "
                             "per sun elevation). "
                             "Enables the 'Sky backdrop' checkbox: the flat background is replaced "
                             "by the per-sun-elevation sky, composited behind the cloud.")
    parser.add_argument("--sky_exposure", type=float, default=3.35,
                        help="Linear exposure multiplier applied to the (linear-HDR) sky before "
                             "tonemapping, so the captured SceneColorHDR (no UE exposure baked in) "
                             "matches the editor look. Default 3.35 (calibrated against the UE "
                             "viewport across sun elevations); live-adjustable via the 'Sky "
                             "exposure' slider.")
    args = parser.parse_args()

    # T_light source (must match the model's training-time shadow field), from
    # cfg_args: tlight_voxel=True -> voxel; else tlight_raster=True -> raster.
    # With NO cfg_args we must fall back to the PROJECT defaults, not to an
    # arbitrary legacy branch: arguments.PipelineParams has tlight_voxel=False
    # (light-space raster is the trained default path) and tonemap_aces=True.
    # Seed with the project default, then let cfg_args override it; `voxel` is
    # only ever selected by an explicit flag or by a cfg that says so.
    use_raster_tlight = True
    tlight_raster_res = 512
    tlight_tau_filter = False
    tlight_filter_variance = 0.0
    # Training metadata only: the viewer never calls backward.
    trained_tlight_full_grad = True
    # Tonemap: cfg flags say whether a curve was trained and which kind; learnable
    # coeffs are confirmed after load_ply (tonemap.json sidecar).
    forced_tonemap = {"on": True, "off": False}.get(args.tonemap, None)
    cfg_tonemap_aces = False
    cfg_tonemap_learnable = False
    cfg = None
    # Always load the trained raster filter/resolution, including explicit modes.
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.ply))))
    cfg_path = os.path.join(run_dir, "cfg_args")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = f.read()
            print(f"[viewer] Training config: {cfg_path}")
        except Exception as e:
            print(f"[viewer] cfg_args unreadable ({e}); falling back to the PROJECT "
                  f"defaults for T_light / tonemap — pass --tlight and --tonemap "
                  f"explicitly if this model was not trained with them.")
    else:
        print(f"[viewer] No cfg_args in {run_dir}; falling back to the PROJECT "
              f"defaults for T_light / tonemap — pass --tlight and --tonemap "
              f"explicitly if this model was not trained with them.")
    if args.tlight == "voxel":
        use_raster_tlight = False  # explicit flag always wins
    elif args.tlight == "auto":
        if cfg is not None:
            if "tlight_voxel" in cfg:
                use_raster_tlight = "tlight_voxel=True" not in cfg
            else:
                use_raster_tlight = "tlight_raster=True" in cfg
        # else: keep the project default already set above (raster). Do NOT fall
        # back to the voxel cache — that is a legacy path, not the default, and
        # silently choosing it renders a shadow field the model never saw.
    # Raster shadow resolution is a property of the SOURCE, not of "auto":
    # whenever the raster pass is used (default, cfg, or an explicit
    # --tlight raster), take the resolution the model was trained with.
    # Purely informational when raster is not used.
    if cfg is not None:
        light_settings = saved_lightpass_settings(cfg)
        tlight_tau_filter = light_settings['tlight_tau_filter']
        trained_tlight_full_grad = light_settings['tlight_full_grad']
        tlight_filter_variance = light_settings['tlight_filter_variance']
        m = re.search(r"tlight_raster_res=(\d+)", cfg)
        if m:
            v = int(m.group(1))
            if 64 <= v <= 4096:
                tlight_raster_res = v
            else:
                print(f"[viewer] ignoring implausible tlight_raster_res={v} from "
                      f"cfg_args; using {tlight_raster_res}^2.")
    if cfg is not None:
        # Every PipelineParams is persisted, so a linear-space run records
        # tonemap_aces=False rather than omitting the flag; only a legacy
        # cfg_args can predate them.
        cfg_tonemap_aces = "tonemap_aces=True" in cfg
        cfg_tonemap_learnable = "tonemap_learnable=True" in cfg
    elif args.tonemap == "auto":
        # No cfg: use the project default (ACES on), not "no tonemap".
        cfg_tonemap_aces = True
    print(f"[viewer] T_light source: {'raster' if use_raster_tlight else 'voxel'}"
          f"{f' ({tlight_raster_res}^2)' if use_raster_tlight else ''}")
    filter_label = ("ON" if tlight_tau_filter else "OFF") if use_raster_tlight else "N/A (voxel)"
    print(f"[viewer] T_light tau filter: {filter_label} (from cfg_args)")
    print(f"[viewer] T_light dilation variance: {tlight_filter_variance} pixel^2 (from cfg_args)")
    print(f"[viewer] Training full light gradients: {trained_tlight_full_grad} "
          "(training only; no effect on viewer forward rendering)")

    print(f"[viewer] Loading {args.ply} ...")
    gaussians = GaussianModel()
    gaussians.load_ply(args.ply)
    P = gaussians.get_xyz.shape[0]
    print(f"[viewer] Loaded {P} Gaussians.")

    # A learnable model is only usable if load_ply restored its tonemap.json
    # coeffs; otherwise fall back to the fixed-ACES curve — the config asked for
    # a curve, and ACES is a closer stand-in than no tonemap at all.
    has_learnable = gaussians.get_tonemap_coeffs is not None
    tonemap_learnable = cfg_tonemap_learnable and has_learnable
    if (cfg_tonemap_learnable and not has_learnable
            and forced_tonemap is not False):
        if not cfg_tonemap_aces:
            print("[viewer] cfg asks for the learnable tonemap but tonemap.json is "
                  "missing; falling back to fixed Narkowicz ACES "
                  "(override with --tonemap off / --tonemap on).")
        cfg_tonemap_aces = True
    if forced_tonemap is None:
        tonemap_on = tonemap_learnable or cfg_tonemap_aces
    else:
        tonemap_on = forced_tonemap
    tonemap_aces = tonemap_on and not (tonemap_learnable)
    if tonemap_on and tonemap_learnable:
        print(f"[viewer] Output tonemap: learnable "
              f"(coeffs={[round(c, 4) for c in gaussians.get_tonemap_coeffs.tolist()]})")
    elif tonemap_on:
        print(f"[viewer] Output tonemap: ACES (fixed Narkowicz)")
    else:
        print(f"[viewer] Output tonemap: linear (clamp)")

    # Stage-2 environment lighting: enabled iff load_ply restored ALL THREE env
    # sidecars (env.json + env_net.pt + sky_transfer.npy) — the restore in
    # GaussianModel.load_ply gates on all three. The sun slider then drives
    # T_sun + E_lm; V_lm is the sun-independent precomputed transfer.
    has_env = env_on = (gaussians.env_net is not None) and (gaussians._sky_transfer.numel() > 0)
    if has_env:
        print(f"[viewer] Environment lighting: ON (SH{gaussians.env_sh_order}, "
              f"V_lm {tuple(gaussians._sky_transfer.shape)})")
    else:
        # Do not assert "Stage-1" outright: a Stage-2 run whose sidecars are
        # incomplete lands here too, and calling that a Stage-1 model would
        # contradict what the run actually is.
        sidecar_dir = os.path.dirname(os.path.abspath(args.ply))
        present = [n for n in ("env.json", "env_net.pt", "sky_transfer.npy")
                   if os.path.exists(os.path.join(sidecar_dir, n))]
        partial = f" — but {len(present)}/3 env sidecars are present here " \
                  f"({', '.join(present)}); this looks like a Stage-2 run with " \
                  f"incomplete sidecars, so env is OFF." if present else ""
        print(f"[viewer] Environment lighting: none (Stage-1 model){partial}")

    initial_sun = _spherical_to_dir(altitude_deg=90.0, azimuth_deg=0.0)  # straight up
    print(f"[viewer] Precomputing T_light (sun={initial_sun.tolist()}) ...")
    t0 = time.time()
    T_light = compute_T_light_cache(
        gaussians, torch.from_numpy(initial_sun).cuda(),
        use_raster=use_raster_tlight, raster_res=tlight_raster_res, tau_filter=tlight_tau_filter, filter_variance=tlight_filter_variance,
    ).detach()
    torch.cuda.synchronize()
    print(f"[viewer] T_light ready in {time.time() - t0:.2f}s. Shape = {tuple(T_light.shape)}")

    # Max σ_t for normalising the sigma_t visualisation channel.
    sigma_t_max = max(gaussians.get_sigma_t.detach().max().item(), 1e-6)

    # --- Initial camera pose from cloud bounds ------------------------------
    # 1st-99th percentile bounds (floaters excluded).
    with torch.no_grad():
        xyz_np = gaussians.get_xyz.detach().cpu().numpy()
    lo = np.percentile(xyz_np, 1, axis=0)
    hi = np.percentile(xyz_np, 99, axis=0)
    cloud_center = ((lo + hi) * 0.5).astype(np.float32)
    cloud_radius = float(max(np.linalg.norm(hi - lo) * 0.5, 1e-3))
    _offset_dir = np.array([0.8, 0.4, 1.0], dtype=np.float32)
    _offset_dir /= np.linalg.norm(_offset_dir)
    default_cam_pos = cloud_center + _offset_dir * (cloud_radius * 2.8)
    default_cam_lookat = cloud_center.copy()

    # Clipping planes scaled to cloud size; fixed 0.01/100 culls large scenes.
    z_near = max(0.01, 0.05 * cloud_radius)
    z_far = max(100.0, 20.0 * cloud_radius)

    print(
        f"[viewer] Cloud bounds: center={cloud_center.tolist()}, "
        f"radius={cloud_radius:.3f}. Initial camera pos={default_cam_pos.tolist()}.\n"
        f"[viewer] Clipping planes: znear={z_near:.3f}, zfar={z_far:.3f}."
    )

    # --- Training cameras for snap-to-pose comparison -----------------------
    # Prefer transforms_train.json, which preserves the (camera_index,
    # time_index) double-key plus per-frame v_l.
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

    train_cams, train_cams_path = _load_training_cameras(args.ply, args.cameras_json)
    if not cam_frames and train_cams:
        train_cam_names = [c.get("img_name", str(c.get("id", i))) for i, c in enumerate(train_cams)]
        print(f"[viewer] Falling back to cameras.json ({len(train_cams)} entries) from {train_cams_path}.")
    else:
        train_cam_names = []

    # --- Optional sky backdrop ----------------------------------------------
    backdrop = None
    if args.sky_dir:
        try:
            backdrop = SkyBackdrop(args.sky_dir)
            print(f"[viewer] Sky backdrop: {args.sky_dir} "
                  f"(elevations {backdrop.alt_min}..{backdrop.alt_max}, {backdrop.ext}). "
                  f"Toggle 'Sky backdrop'; the sun sliders drive elevation/azimuth.")
        except Exception as e:
            print(f"[viewer] Sky backdrop disabled (load failed: {e})")
            backdrop = None

    pipe = _ViewerPipe()
    pipe.tonemap_aces = tonemap_aces
    pipe.tonemap_learnable = tonemap_learnable
    pipe.env_lighting = env_on
    bg_color = torch.tensor(
        [1.0, 1.0, 1.0] if args.bg == "white" else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True
    light_source_label = f"raster {tlight_raster_res} x {tlight_raster_res}" if use_raster_tlight else "voxel 128^3"
    server.gui.add_markdown(
        f"**Model:** `{os.path.basename(run_dir)}`  \n"
        f"**Light pass:** {light_source_label}  \n"
        f"**Tau footprint compensation:** {filter_label}  \n"
        f"**Light dilation variance:** {tlight_filter_variance} pixel²  \n"
        f"**Full light gradients during training:** {trained_tlight_full_grad}  \n"
        "The gradient setting affects training only. "
        "Light-pass settings are read from the model config on startup."
    )

    # Sun direction arrow: drawn outside the cloud bbox, pointing along the
    # *light propagation* direction. Updated via the handle's `position` + `wxyz`.
    arrow_len = max(cloud_radius * 0.8, 0.5)
    arrow_offset = cloud_radius * 1.6  # stand-off distance from cloud centre

    # Local arrow geometry: tail at origin, head at +arrow_len along local +X,
    # rotated so local +X aligns with the world light-propagation direction
    # (= -v_l) and translated so the tail sits at cloud_center + v_l*arrow_offset.
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

    def _sun_arrow_pose(v_l_np: np.ndarray):
        s = v_l_np / max(np.linalg.norm(v_l_np), 1e-8)
        position = (cloud_center + s * arrow_offset).astype(np.float32)
        # Rotate local +X to the light-propagation direction (-s = -v_l).
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
        options=("rgb", "T_light", "sigma_t", "depth"),
        initial_value="rgb",
    )
    # Snap-to-training-cam controls. With transforms_train.json: one dropdown of
    # unique camera_index values + a time slider, rather than one entry per frame
    # (thousands of dropdown entries break the viser websocket on connect).
    gui_train_cam = None              # legacy cameras.json dropdown
    gui_train_cam_text = None         # fallback text input
    gui_train_cam_idx = None          # unique camera_index dropdown
    gui_train_time = None             # time slider
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
        "Max render size", min=256, max=3840, step=32,
        initial_value=max(args.width, 1920) if args.width is not None else 1920,
        hint="Upper bound on the rendered longer-side resolution. The render matches "
             "the browser canvas's true pixel size (aspect follows the window), capped "
             "here to bound GPU cost on large windows. Lower it if the frame rate drops.",
    )
    gui_bgcolor = server.gui.add_rgb(
        "Background color",
        initial_value=(255, 255, 255) if args.bg == "white" else (0, 0, 0),
        hint="Rasterizer background colour. Handy for inspecting cloud edges / "
             "discrete Gaussian ellipsoids against different backdrops.",
    )
    _tm_label = "Tonemap (learned)" if tonemap_learnable else "Tonemap (ACES)"
    gui_tonemap = server.gui.add_checkbox(
        _tm_label,
        initial_value=tonemap_on,
        hint="Apply the model's output tonemap to the RGB output (HDR linear shading "
             "→ tonemapped display space). Uses the learned per-model curve if the run "
             "has one (tonemap.json), else the fixed Narkowicz ACES curve. Auto-set from "
             "the run's cfg_args. Toggle to A/B the model's native space; tonemapped "
             "models look too bright / washed-out with this off.",
    )
    gui_env = server.gui.add_checkbox(
        "Environment light",
        initial_value=env_on,
        hint="Stage-2 environment lighting: add the learned global sky (T_sun sun "
             "transmittance ⊙ sun_term + ω·Σ E_lm·V_lm in-scatter fill) on top of the "
             "frozen sun shading; tracks the sun slider for relighting. Only active for "
             "a --stage2 model (env sidecars present); a no-op otherwise.",
    )
    gui_sky = server.gui.add_checkbox(
        "Sky backdrop",
        initial_value=(backdrop is not None),
        hint="Replace the flat background with the captured HDR sky cubemap: picks the "
             "cube for the sun-altitude slider and rotates it to the azimuth slider, "
             "composited behind the cloud via its transmittance. RGB view only; needs "
             "--sky_dir. Viewer-only backdrop — does not affect cloud shading.",
    )
    gui_sky.disabled = backdrop is None
    gui_sky_exposure = server.gui.add_slider(
        "Sky exposure", min=0.25, max=8.0, step=0.05, initial_value=float(args.sky_exposure),
        hint="Linear exposure multiplier on the HDR sky before tonemapping. The capture "
             "is raw SceneColorHDR (no UE exposure), so this dials the sky brightness to "
             "match the editor / cloud. ~3.0 ≈ the UE viewport. Sky backdrop only.",
    )
    gui_sky_exposure.disabled = backdrop is None
    gui_sky_warmth = server.gui.add_slider(
        "Sky warmth", min=-0.3, max=0.3, step=0.01, initial_value=0.09,
        hint="White-balance the sky: R·(1+w), B·(1-w). Corrects the "
             "residual cool/blue cast of Narkowicz ACES vs UE's filmic+grading so the "
             "backdrop matches the editor. Default 0.09 (calibrated across sun elevations; "
             "RMSE≈0.024). + = warmer, - = cooler. Sky backdrop only.",
    )
    gui_sky_warmth.disabled = backdrop is None
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
    gui_shot_size = server.gui.add_slider(
        "Still size (px)", min=256, max=4096, step=128, initial_value=1024,
        hint="Side length of the square still saved by 'Capture still'. Fixed "
             "aspect 1:1, independent of the browser window.",
    )
    gui_shot = server.gui.add_button(
        "📷 Capture still",
        hint="Render a square PNG at the current camera pose (and all current "
             "settings: sun, tonemap, sky backdrop, view mode) into ./figures/. "
             "For paper figures — resolution is fixed by 'Still size', not the window.",
    )
    gui_shot_status = server.gui.add_text("Still", initial_value="-")
    gui_shot_status.disabled = True
    gui_fps = server.gui.add_text("FPS", initial_value="-")
    gui_fps.disabled = True
    gui_ngauss = server.gui.add_text("# Gaussians", initial_value=f"{P:,}")
    gui_ngauss.disabled = True

    state = {
        "needs_render": True,
        "last_render_time": 0.0,
        "diag_done": False,
        "v_l": initial_sun.copy(),     # numpy (3,) — current cached sun
        "T_light": T_light,                # current cached T_light tensor
    }

    # Wall-clock-faithful capture of the FIRST client's rendered frames (the
    # render loop only draws on changes, so idle time repeats the last frame).
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
                # H.264/mp4v chroma subsampling needs even dimensions.
                self.size = (w - (w % 2), h - (h % 2))
            tw, th = self.size
            if (w, h) != (tw, th):
                img_np = img_np[:th, :tw] if (w >= tw and h >= th) else None
                if img_np is None:
                    return  # render size shrank mid-recording; skip frame
            # Fill wall-clock gaps so playback timing matches reality.
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
            # Prefer H.264 (avc1, broadly playable); mp4v is desktop-only. A writer
            # can "open" yet produce a header-only file, so size is verified below.
            for codec in ("avc1", "mp4v"):
                writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*codec), REC_FPS, self.size)
                if not writer.isOpened():
                    writer.release()
                    continue
                for f in self.frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                if os.path.getsize(path) > 1000:
                    break
            else:
                print("[viewer] recording failed: no working mp4 encoder")
                return None
            dur = n / REC_FPS
            print(f"[viewer] recording saved: {path} ({n} frames, {dur:.1f}s, {codec})")
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
                use_raster=use_raster_tlight, raster_res=tlight_raster_res, tau_filter=tlight_tau_filter, filter_variance=tlight_filter_variance,
            ).detach()
        torch.cuda.synchronize()
        state["v_l"] = new_sun
        state["T_light"] = new_cache
        state["needs_render"] = True
        new_pos, new_wxyz = _sun_arrow_pose(new_sun)
        sun_arrow.position = new_pos
        sun_arrow.wxyz = new_wxyz
        print(f"[viewer] v_l → {new_sun.tolist()} (alt={gui_sun_alt.value:.1f}°, az={gui_sun_az.value:.1f}°)")

    gui_sun_alt.on_update(_apply_sun)
    gui_sun_az.on_update(_apply_sun)

    gui_mode.on_update(_mark_dirty)
    gui_res.on_update(_mark_dirty)
    gui_bgcolor.on_update(_mark_dirty)
    gui_tonemap.on_update(_mark_dirty)
    gui_env.on_update(_mark_dirty)
    gui_sky.on_update(_mark_dirty)
    gui_sky_exposure.on_update(_mark_dirty)
    gui_sky_warmth.on_update(_mark_dirty)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Set viser's world up axis BEFORE the pose: viser defaults to +Z up, but
        # the dataset (UE → OpenGL) and the trained Gaussians use +Y up (otherwise
        # the view comes out tilted/flipped).
        client.camera.up_direction = (0.0, 1.0, 0.0)
        client.camera.position = default_cam_pos
        client.camera.look_at = default_cam_lookat

        @client.camera.on_update
        def _(_):
            # When "Orbit-only" is on, snap look_at back to the cloud centre (orbit
            # + zoom only). viser's setter is a no-op when the value already
            # matches, so assigning here converges without recursion.
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
        # Disable orbit-lock first: otherwise the camera on_update callback would
        # immediately drag look_at back to the cloud centre, breaking the snap.
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
        """Snap by (camera_index, time_index) using transforms.json frames."""
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
        # Push the frame's v_l into the global sun state so T_light matches what
        # training saw; falls back to the slider value if the frame lacks it.
        sd = frame.get("sun_direction")
        if sd is not None:
            new_sun = np.asarray(sd, dtype=np.float32)
            new_sun /= max(np.linalg.norm(new_sun), 1e-8)
            with torch.no_grad():
                new_cache = compute_T_light_cache(
                    gaussians, torch.from_numpy(new_sun).cuda(),
                    use_raster=use_raster_tlight, raster_res=tlight_raster_res, tau_filter=tlight_tau_filter, filter_variance=tlight_filter_variance,
                ).detach()
            state["v_l"] = new_sun
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

    print(f"[viewer] Serving at http://localhost:{args.port}")

    def render_frame(cam, render_w, render_h, aspect=None):
        """Render one frame for `cam` at (render_w, render_h), honouring every
        live GUI control (mode, tonemap, env, sky backdrop, bg colour).

        Shared by the interactive loop and the still-capture button so their
        output is identical. `aspect` overrides the camera's horizontal FOV
        aspect (the still passes render_w/render_h so a square capture isn't
        stretched by the live canvas aspect). Returns (H, W, 3) uint8.
        """
        current_sun = state["v_l"]
        current_T_light = state["T_light"]
        if aspect is None:
            aspect = render_w / max(1, render_h)
        mini = viser_to_minicam(cam, render_w, render_h, z_near=z_near, z_far=z_far,
                                v_l=current_sun, aspect=aspect)
        _tm_on = bool(gui_tonemap.value)
        pipe.tonemap_learnable = _tm_on and tonemap_learnable
        pipe.tonemap_aces = _tm_on and not tonemap_learnable
        pipe.env_lighting = bool(gui_env.value) and has_env
        mode = gui_mode.value
        sky_active = bool(gui_sky.value) and (backdrop is not None) and mode == "rgb"

        _bg = gui_bgcolor.value
        bg_color[0] = _bg[0] / 255.0
        bg_color[1] = _bg[1] / 255.0
        bg_color[2] = _bg[2] / 255.0

        if mode == "T_light":
            override = current_T_light.expand(-1, 3).contiguous()
        elif mode == "sigma_t":
            sigma_t = gaussians.get_sigma_t.detach() / sigma_t_max
            override = sigma_t.clamp(0.0, 1.0).expand(-1, 3).contiguous()
        else:
            override = None

        # Sky backdrop: sample the HDR cube along this view's rays, as the
        # rasterizer's per-pixel background.
        sky_image = None
        if sky_active:
            c2w_rot = _quat_wxyz_to_matrix(np.asarray(cam.wxyz, dtype=np.float32))
            rays = camera_ray_dirs(c2w_rot, float(cam.fov), render_w, render_h)
            sky_image = backdrop.sample(
                rays, float(gui_sun_alt.value), float(gui_sun_az.value)
            ).permute(2, 0, 1) * float(gui_sky_exposure.value)
            w = float(gui_sky_warmth.value)
            if w != 0.0:
                sky_image = sky_image * torch.tensor(
                    [1.0 + w, 1.0, 1.0 - w], device=sky_image.device, dtype=sky_image.dtype
                ).view(3, 1, 1)
            sky_image = sky_image.contiguous()

        with torch.no_grad():
            out = render(
                mini, gaussians, pipe, bg_color,
                override_color=override,
                precomputed_T_light=current_T_light,
                bg_image=sky_image,
            )
        if mode == "depth":
            return _depth_to_image(out["depth"]), out
        # For the sky backdrop, out["render"] is already tonemap(cloud +
        # T_final·sky) — composited in the rasterizer, tonemapped once.
        return _tensor_to_image(out["render"]), out

    def _save_still(_):
        """Render a square still at the current camera for paper figures."""
        clients = server.get_clients()
        if not clients:
            gui_shot_status.value = "no client connected"
            return
        cam = next(iter(clients.values())).camera
        side = int(gui_shot_size.value)
        try:
            img_np, _ = render_frame(cam, side, side, aspect=1.0)
        except RuntimeError as e:
            gui_shot_status.value = f"render failed: {e}"
            print(f"[viewer] still capture failed: {e}")
            return
        os.makedirs("figures", exist_ok=True)
        path = os.path.join("figures", time.strftime("fig_%Y%m%d_%H%M%S") + ".png")
        # render_frame returns RGB uint8; cv2 writes BGR, so flip channels.
        import cv2
        cv2.imwrite(path, img_np[..., ::-1])
        gui_shot_status.value = f"saved {os.path.basename(path)} ({side}²)"
        print(f"[viewer] still saved: {path} ({side}x{side})")

    gui_shot.on_click(_save_still)

    _last_canvas = {}
    while True:
        if not state["needs_render"]:
            # Poll for canvas resizes (camera on_update doesn't always fire on
            # resize): re-render when a client's canvas pixel size changed.
            for _cid, _cl in server.get_clients().items():
                _cw, _ch = int(_cl.camera.image_width or 0), int(_cl.camera.image_height or 0)
                if _last_canvas.get(_cid) != (_cw, _ch):
                    _last_canvas[_cid] = (_cw, _ch)
                    state["needs_render"] = True
            if not state["needs_render"]:
                if recorder.active:
                    # While recording, re-render at the capture cadence so idle
                    # wall-clock time fills in steadily.
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
            # Render at the canvas's true pixel size (aspect follows the window);
            # gui_res caps the LONGER side.
            max_side = int(gui_res.value)
            cw, ch = int(cam.image_width or 0), int(cam.image_height or 0)
            if cw > 0 and ch > 0:
                scale = min(1.0, max_side / float(max(cw, ch)))
                render_w = max(1, int(round(cw * scale)))
                render_h = max(1, int(round(ch * scale)))
            else:
                render_w = max_side
                render_h = max(1, int(render_w / max(cam.aspect, 1e-3)))

            try:
                img_np, out = render_frame(cam, render_w, render_h)
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

            if first_client:
                recorder.add(img_np)
                if recorder.active:
                    gui_record_status.value = (
                        f"recording {time.time() - recorder.t_start:.0f}s "
                        f"({len(recorder.frames)} frames)")
                first_client = False

            client.scene.set_background_image(img_np, format="png")

        dt = time.time() - t_frame
        if dt > 0:
            gui_fps.value = f"{1.0 / dt:5.1f}"


if __name__ == "__main__":
    main()
