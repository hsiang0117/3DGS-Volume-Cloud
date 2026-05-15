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
import time
from dataclasses import dataclass

import numpy as np
import torch
import viser

from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render, compute_T_light, normalized_gaussian_line_integral
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import build_rotation


# --- Pipeline stub (mirrors arguments.PipelineParams, no argparse needed) ---
@dataclass
class _ViewerPipe:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False
    k_sigma: float = 1.5


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


def viser_to_minicam(cam, width: int, height: int, z_near: float = 0.01, z_far: float = 100.0) -> MiniCam:
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

    return MiniCam(width, height, fovy, fovx, z_near, z_far, world_view, full_proj)


@torch.no_grad()
def compute_T_light_cache(gaussians: GaussianModel) -> torch.Tensor:
    """Compute T_light once for static sun direction [0, 1, 0].

    Mirrors the per-Gaussian τ derivation used inside `render()` so the cache
    matches what render() would produce for the same sun direction.
    """
    sun_dir = torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=torch.float32)
    s = gaussians.get_scaling
    beta_peak = gaussians.get_extinction
    mass = beta_peak * ((2.0 * math.pi) ** 1.5) * torch.prod(s, dim=1, keepdim=True)
    R = build_rotation(gaussians.get_rotation)                              # (P,3,3)
    R_t = R.transpose(1, 2)
    l_local = torch.matmul(R_t, sun_dir.view(3, 1)).squeeze(-1)             # (P,3)
    line_int_sun = normalized_gaussian_line_integral(s, l_local)
    tau_sun_per_gauss = mass * line_int_sun
    return compute_T_light(
        gaussians.get_xyz,
        tau_sun_per_gauss,
        s,
        sun_dir,
        grid_res=128,
    )


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
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--bg", choices=["black", "white"], default="black")
    parser.add_argument("--cameras_json", default=None,
                        help="Optional explicit path to cameras.json (auto-located next to PLY by default).")
    args = parser.parse_args()

    # --- Load Gaussians -----------------------------------------------------
    print(f"[viewer] Loading {args.ply} ...")
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.ply, use_train_test_exp=False)
    P = gaussians.get_xyz.shape[0]
    print(f"[viewer] Loaded {P} Gaussians.")

    # --- Precompute T_light (static sun) ------------------------------------
    print("[viewer] Precomputing T_light (static sun [0,1,0]) ...")
    t0 = time.time()
    T_light = compute_T_light_cache(gaussians).detach()
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
    train_cams, train_cams_path = _load_training_cameras(args.ply, args.cameras_json)
    if train_cams:
        train_cam_names = [c.get("img_name", str(c.get("id", i))) for i, c in enumerate(train_cams)]
        print(f"[viewer] Loaded {len(train_cams)} training cameras from {train_cams_path}.")
    else:
        train_cam_names = []
        print("[viewer] No cameras.json found — 'Snap to training cam' will be disabled.")

    pipe = _ViewerPipe()
    bg_color = torch.tensor(
        [1.0, 1.0, 1.0] if args.bg == "white" else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )

    # --- viser server + GUI -------------------------------------------------
    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    gui_mode = server.gui.add_dropdown(
        "View mode",
        options=("rgb", "T_light", "beta_peak", "depth"),
        initial_value="rgb",
    )
    if train_cams:
        gui_train_cam = server.gui.add_dropdown(
            "Snap to training cam",
            options=tuple(["(free)"] + train_cam_names),
            initial_value="(free)",
            hint="Pick a training camera by image name; viewer pose+FOV will match it so you can compare with render_test/iter_*_<name>.png.",
        )
    else:
        gui_train_cam = None
    gui_scaling = server.gui.add_slider(
        "Gaussian scale", min=0.1, max=2.0, step=0.05, initial_value=1.0,
        hint="Multiplies all Gaussian scales at render time (visual only, does not modify stored model).",
    )
    gui_ksigma = server.gui.add_slider(
        "Sort k·σ clamp", min=0.0, max=3.0, step=0.1, initial_value=1.5,
        hint="Per-tile max-response sort: how far t* may deviate from centre depth, "
             "in units of σ along the view ray. 0 = stock 3DGS centre sort. "
             "1.5 = recommended (no tile artefacts, long-axis popping fixed). "
             "3.0 ≈ unclamped (cleanest long-axis order, may show tile-edge artefacts).",
    )
    gui_res = server.gui.add_slider(
        "Render size", min=256, max=1920, step=32, initial_value=args.width,
        hint="Render resolution width; height is derived from client aspect.",
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
    gui_fps = server.gui.add_text("FPS", initial_value="-")
    gui_fps.disabled = True
    gui_ngauss = server.gui.add_text("# Gaussians", initial_value=f"{P:,}")
    gui_ngauss.disabled = True

    # Flag set when any input that affects the image changes.
    state = {"needs_render": True, "last_render_time": 0.0, "diag_done": False}

    def _mark_dirty(*_):
        state["needs_render"] = True

    gui_mode.on_update(_mark_dirty)
    gui_scaling.on_update(_mark_dirty)
    gui_ksigma.on_update(_mark_dirty)
    gui_res.on_update(_mark_dirty)

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

    if gui_train_cam is not None:
        @gui_train_cam.on_update
        def _(_):
            sel = gui_train_cam.value
            if sel == "(free)":
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

    # --- Render loop --------------------------------------------------------
    print(f"[viewer] Serving at http://localhost:{args.port}")
    while True:
        if not state["needs_render"]:
            time.sleep(0.01)
            continue
        state["needs_render"] = False

        clients = server.get_clients()
        if not clients:
            time.sleep(0.1)
            continue

        t_frame = time.time()
        for client in clients.values():
            cam = client.camera
            # Keep aspect consistent with client window to avoid distortion.
            render_w = int(gui_res.value)
            render_h = max(1, int(render_w / max(cam.aspect, 1e-3)))
            mini = viser_to_minicam(cam, render_w, render_h, z_near=z_near, z_far=z_far)
            scaling_mod = float(gui_scaling.value)
            pipe.k_sigma = float(gui_ksigma.value)
            mode = gui_mode.value

            if mode == "T_light":
                # T_light is (P, 1); broadcast to RGB grayscale.
                override = T_light.expand(-1, 3).contiguous()
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
                        use_trained_exp=False,
                        precomputed_T_light=T_light,
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

            client.scene.set_background_image(img_np, format="jpeg")

        dt = time.time() - t_frame
        if dt > 0:
            gui_fps.value = f"{1.0 / dt:5.1f}"


if __name__ == "__main__":
    main()
