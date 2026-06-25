"""
Sky backdrop for the viewer: a per-sun-elevation HDR cubemap sampled per camera
ray to replace the flat background behind the cloud.

The cubemaps are captured in UE (tools/ue_capture_sky_backdrop.py): one 6-face
cube per sun elevation 0..90 deg, all at a FIXED sun azimuth. The sky is
rotationally symmetric about the zenith except for the sun, so azimuth is a free
runtime rotation about the up axis — we pick the cube for round(sun_altitude) and
rotate it so the captured sun lands at the viewer's sun azimuth.

Frames
  Capture (UE world): left-handed, +Z up, +X forward. Empirically the sun glow
    sits on the +X (px) face, so in this frame the sun's horizontal direction is
    +X and the zenith is +Z.
  Viewer world (OpenGL, matches the trained Gaussians / sun_dir): +Y up, +X right,
    -Z forward.
We map cube->viewer by aligning (zenith +Z -> +Y) and (sun horizontal +X -> the
viewer's sun-azimuth horizontal direction). A camera ray d_v is sampled by
rotating it into the cube frame (d_c = R^T d_v) and doing a standard cube lookup.

Values are linear HDR (SceneColorHDRNoAlpha). The caller applies the same tonemap
it uses for the cloud so sky and cloud share a display space.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # cv2 ships EXR off by default here

import json
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Per captured face: (forward, right, up) in UE world, derived from the
# SceneCapture2D rotations in sky.json (px=+X fwd, pz=+Z up, etc.).
_FACE_BASIS = {
    "px": ((1, 0, 0),  (0, 1, 0),  (0, 0, 1)),
    "nx": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "py": ((0, 1, 0),  (-1, 0, 0), (0, 0, 1)),
    "ny": ((0, -1, 0), (1, 0, 0),  (0, 0, 1)),
    "pz": ((0, 0, 1),  (0, 1, 0),  (-1, 0, 0)),
    "nz": ((0, 0, -1), (0, 1, 0),  (1, 0, 0)),
}


def camera_ray_dirs(c2w_rot: np.ndarray, fov_y: float, width: int, height: int,
                    device="cuda") -> torch.Tensor:
    """World-space unit ray directions for every pixel of a viser/3DGS camera.

    Camera-local convention (viser / OpenCV): +X right, +Y down, +Z forward.
    `c2w_rot` is the 3x3 camera-to-world rotation; returns (H, W, 3) in viewer
    world coords (+Y up).
    """
    tan_y = math.tan(0.5 * fov_y)
    tan_x = tan_y * (width / max(1, height))
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    px = (2.0 * (xs + 0.5) / width - 1.0) * tan_x
    py = (2.0 * (ys + 0.5) / height - 1.0) * tan_y
    dirs_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)   # (H,W,3), +Z fwd
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)
    R = torch.as_tensor(c2w_rot, dtype=torch.float32, device=device)
    dirs_world = dirs_cam @ R.T                                     # (H,W,3)
    return dirs_world


class SkyBackdrop:
    """Lazily-loaded HDR cube sky, sampled per camera ray with runtime sun rotation."""

    def __init__(self, sky_dir: str, device="cuda", lru: int = 4):
        self.dir = sky_dir
        self.device = device
        with open(os.path.join(sky_dir, "sky.json"), "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        if self.meta.get("format") != "cube_faces":
            raise ValueError(f"unsupported sky format: {self.meta.get('format')}")
        self.ext = self.meta["ext"]
        self.faces = self.meta["faces"]                 # e.g. ["px","nx","py","ny","pz","nz"]
        self.alt_min = int(self.meta["alt_min"])
        self.alt_max = int(self.meta["alt_max"])
        self.pattern = self.meta["file_pattern"]        # "sky_alt{alt:02d}_{face}.exr"
        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._lru = lru
        # Stack of per-face (forward, right, up) basis vectors, aligned with self.faces.
        b = np.array([_FACE_BASIS[f] for f in self.faces], dtype=np.float32)  # (6,3,3)
        self._F = torch.as_tensor(b[:, 0], device=device)   # (6,3)
        self._R = torch.as_tensor(b[:, 1], device=device)
        self._U = torch.as_tensor(b[:, 2], device=device)

    # --- face loading / cache -------------------------------------------------
    def _read_exr(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"could not read {path} (OpenEXR disabled?)")
        return img[..., :3][..., ::-1].astype(np.float32)   # BGR(A) -> RGB

    def _get_elevation(self, alt: int) -> torch.Tensor:
        alt = int(min(max(alt, self.alt_min), self.alt_max))
        cached = self._cache.get(alt)
        if cached is not None:
            self._cache.move_to_end(alt)
            return cached
        faces = []
        for f in self.faces:
            p = os.path.join(self.dir, self.pattern.format(alt=alt, face=f))
            faces.append(torch.from_numpy(np.ascontiguousarray(self._read_exr(p))).permute(2, 0, 1))
        tens = torch.stack(faces, dim=0).to(self.device)    # (6,3,Hf,Wf)
        self._cache[alt] = tens
        self._cache.move_to_end(alt)
        while len(self._cache) > self._lru:
            self._cache.popitem(last=False)
        return tens

    # --- frame mapping --------------------------------------------------------
    def _cube_to_viewer(self, sun_az_deg: float) -> torch.Tensor:
        """3x3 R mapping cube axes (sun-horizontal +X, +Y, zenith +Z) -> viewer world.

        Columns: [sun_horizontal_viewer, third, up_viewer]. Matches _spherical_to_dir:
        azimuth 0 -> +X, increasing toward +Z; up is +Y.
        """
        a = math.radians(sun_az_deg)
        sunh = (math.cos(a), 0.0, math.sin(a))     # viewer horizontal sun direction
        up = (0.0, 1.0, 0.0)
        third = (math.sin(a), 0.0, -math.cos(a))   # up x sunh
        R = torch.tensor([[sunh[0], third[0], up[0]],
                          [sunh[1], third[1], up[1]],
                          [sunh[2], third[2], up[2]]], dtype=torch.float32, device=self.device)
        return R

    # --- sampling -------------------------------------------------------------
    @torch.no_grad()
    def sample(self, dirs_world: torch.Tensor, sun_alt_deg: float, sun_az_deg: float) -> torch.Tensor:
        """Sample the sky along (..., 3) viewer-world ray directions. Returns (...,3) linear HDR."""
        shape = dirs_world.shape[:-1]
        d_v = dirs_world.reshape(-1, 3).to(self.device)
        d_v = d_v / d_v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        R = self._cube_to_viewer(sun_az_deg)
        d_c = d_v @ R                                  # R^T d_v: viewer -> cube frame
        faces = self._get_elevation(int(round(sun_alt_deg)))   # (6,3,H,W)

        out = torch.zeros_like(d_c)
        ax = d_c.abs().argmax(dim=1)                   # major axis 0/1/2
        for i in range(len(self.faces)):
            Fi, Ri, Ui = self._F[i], self._R[i], self._U[i]
            axis = int(torch.nonzero(Fi != 0)[0])
            fsign = float(Fi[axis])
            mask = (ax == axis) & (torch.sign(d_c[:, axis]) == fsign)
            if not mask.any():
                continue
            dm = d_c[mask]
            denom = (dm @ Fi).clamp_min(1e-6)
            gx = (dm @ Ri) / denom                     # in [-1,1] within the face
            gy = -(dm @ Ui) / denom
            grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
            samp = F.grid_sample(faces[i:i + 1], grid, mode="bilinear",
                                 align_corners=True, padding_mode="border")
            out[mask] = samp.view(3, -1).T
        return out.reshape(*shape, 3)


# --- offline verification ----------------------------------------------------
def _aces(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def _dump_equirect(sky: "SkyBackdrop", alt: int, out_png: str, w: int = 1024,
                   exposure: float = 1.0, cube_frame: bool = True):
    """Build an equirect panorama by sampling the cube and write a tonemapped PNG.

    cube_frame=True samples directly in the cube frame (sun should land at the
    image centre column, lon=0 -> +X); used to verify face stitching/orientation.
    """
    h = w // 2
    lon = (np.linspace(-math.pi, math.pi, w, dtype=np.float32))[None, :].repeat(h, 0)
    lat = (np.linspace(math.pi / 2, -math.pi / 2, h, dtype=np.float32))[:, None].repeat(w, 1)
    # cube frame: +Z up, +X at lon=0
    dx = np.cos(lat) * np.cos(lon)
    dy = np.cos(lat) * np.sin(lon)
    dz = np.sin(lat)
    d = torch.from_numpy(np.stack([dx, dy, dz], -1)).to(sky.device)
    if cube_frame:
        faces = sky._get_elevation(int(round(alt)))
        out = torch.zeros_like(d.reshape(-1, 3))
        dc = d.reshape(-1, 3)
        ax = dc.abs().argmax(1)
        for i in range(len(sky.faces)):
            Fi, Ri, Ui = sky._F[i], sky._R[i], sky._U[i]
            axis = int(torch.nonzero(Fi != 0)[0]); fsign = float(Fi[axis])
            mask = (ax == axis) & (torch.sign(dc[:, axis]) == fsign)
            if not mask.any():
                continue
            dm = dc[mask]; denom = (dm @ Fi).clamp_min(1e-6)
            grid = torch.stack([(dm @ Ri) / denom, -(dm @ Ui) / denom], -1).view(1, -1, 1, 2)
            samp = F.grid_sample(faces[i:i + 1], grid, mode="bilinear", align_corners=True,
                                 padding_mode="border")
            out[mask] = samp.view(3, -1).T
        rgb = out.reshape(h, w, 3).cpu().numpy()
    else:
        rgb = sky.sample(d, alt, 0.0).cpu().numpy()
    img = _aces(rgb * exposure)
    cv2.imwrite(out_png, (img[..., ::-1] * 255).astype(np.uint8))
    print(f"wrote {out_png}  (alt={alt}, lin max {rgb.max():.3f})")


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else r"D:\3DGS-Volume-Cloud\data\sky_backdrop"
    sky = SkyBackdrop(d)
    print("meta:", {k: sky.meta[k] for k in ("format", "ext", "alt_min", "alt_max", "sun_azimuth_deg")})
    for a, expo in [(2, 6.0), (20, 2.0), (60, 1.0), (88, 1.0)]:
        _dump_equirect(sky, a, f"_equirect_alt{a:02d}.png", exposure=expo)
