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

import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud


class EnvNet(nn.Module):
    """Stage-2 global environment-lighting net: v_l (3,) -> (T_sun RGB<=1, E_lm SH).

    GLOBAL — one sky for the whole scene, a function of sun direction ONLY; it adds
    no per-Gaussian colour DOF (per-Gaussian colour stays in the frozen albedo ω).

    T_sun is an analytic atmospheric transmittance with 3 learnable params, the
    per-channel zenith optical depths τ=(τ_R,τ_G,τ_B):

        T_sun(v_l) = exp( − m(θ) · softplus(raw_tau) ),   θ = sun zenith angle

    m(θ) is the Kasten-Young air mass (fixed geometric function, not learned) and
    softplus keeps τ≥0, so the exponential alone lies in (0,1]. The returned
    T_sun is that exponential MULTIPLIED by a smoothstep horizon gate, so its
    range is [0,1]: it reaches exactly 0 once the sun is at or below the horizon.
    It depends on the zenith angle only. E_lm (additive sky in-scatter SH) comes
    from a small MLP over v_l."""
    # Near-neutral zenith optical depth init (R,G,B): small, τ_B>τ_R (faint Rayleigh tilt).
    TAU_INIT = (0.02, 0.04, 0.07)

    @staticmethod
    def _softplus_inverse(y, eps=1e-8):
        y = torch.clamp(torch.as_tensor(y, dtype=torch.float), min=eps)
        return torch.log(torch.expm1(y) + eps)

    @staticmethod
    def _air_mass(v_l):
        """Kasten-Young (1989) relative air mass from a sun direction (up=+Y):
        m(θ)=1/(cosθ + 0.50572 (96.07995-θ_deg)^-1.6364), finite at the horizon.
        Upper hemisphere only; cosθ is clamped to ~horizon so it never goes
        negative/explosive — below-horizon dimming is the gate in forward()."""
        cos_theta = torch.clamp(v_l.reshape(3)[1], 0.0, 1.0)    # up = +Y, upper hemi only
        theta_deg = torch.rad2deg(torch.arccos(cos_theta))
        denom = cos_theta + 0.50572 * torch.clamp(96.07995 - theta_deg, min=1e-3) ** (-1.6364)
        return 1.0 / torch.clamp(denom, min=1e-3)

    # Horizon softness (cosθ units): the sun gate fades T_sun to 0 over cosθ ∈ [0, HORIZON_SOFT].
    HORIZON_SOFT = 0.05

    def __init__(self, n_sh, hidden=64):
        super().__init__()
        self.n_sh = n_sh
        # T_sun: 3 learnable per-channel zenith optical depths (raw -> softplus -> τ≥0).
        self.raw_tau = nn.Parameter(self._softplus_inverse(self.TAU_INIT))
        # E_lm: small MLP of v_l (additive sky in-scatter), zero-init -> neutral.
        self.backbone = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.e_head = nn.Linear(hidden, n_sh * 3)   # -> E_lm (n_sh,3) sky radiance SH
        nn.init.zeros_(self.e_head.weight); nn.init.zeros_(self.e_head.bias)

    def forward(self, v_l):
        tau = torch.nn.functional.softplus(self.raw_tau)            # (3,) ≥0
        m = self._air_mass(v_l)                                 # scalar (upper hemi)
        t_sun = torch.exp(-m * tau)                                 # (3,) ∈(0,1]
        # Below-horizon gate (cosθ≤0): T_sun fades to 0 over HORIZON_SOFT via a smoothstep.
        cos_theta = v_l.reshape(3)[1]
        t = torch.clamp(cos_theta / self.HORIZON_SOFT, 0.0, 1.0)
        gate = t * t * (3.0 - 2.0 * t)                              # smoothstep(0,HORIZON_SOFT)
        t_sun = t_sun * gate
        h = self.backbone(v_l.reshape(1, 3))
        e_lm = self.e_head(h).reshape(self.n_sh, 3)
        return t_sun, e_lm

    @property
    def tau(self):
        return torch.nn.functional.softplus(self.raw_tau)


def _fibonacci_hemisphere(n):
    """n directions ~uniform over the upper hemisphere (world up = +Y, OpenGL)."""
    ga = math.pi * (3.0 - math.sqrt(5.0))
    pts = []
    for i in range(n):
        y = (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - y * y))
        phi = i * ga
        pts.append([r * math.cos(phi), y, r * math.sin(phi)])
    return torch.tensor(pts, dtype=torch.float, device="cuda")


def _sh_basis_deg2(dirs):
    """Real orthonormal SH basis up to l=2 (9 coeffs) at unit dirs (M,3) -> (M,9). Matches
    utils.sh_utils.eval_sh sign/constant convention so V_lm stays consistent."""
    from utils.sh_utils import C0, C1, C2
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    out = torch.empty((dirs.shape[0], 9), device=dirs.device, dtype=dirs.dtype)
    out[:, 0] = C0
    out[:, 1] = -C1 * y
    out[:, 2] = C1 * z
    out[:, 3] = -C1 * x
    out[:, 4] = C2[0] * xy
    out[:, 5] = C2[1] * yz
    out[:, 6] = C2[2] * (2.0 * zz - xx - yy)
    out[:, 7] = C2[3] * xz
    out[:, 8] = C2[4] * (xx - yy)
    return out


class GaussianModel:

    SIGMA_T_MAX = 5.0
    SIGMA_T_RAW_MAX = math.log(math.expm1(SIGMA_T_MAX))
    PRUNE_GRACE_STEPS = 500

    # Learnable output tonemap (Narkowicz ACES rational form), shared across RGB:
    #     f(x) = (a x^2 + b x) / (c x^2 + d x + e)
    # (a,b,c,d) learned via softplus(raw) (positive ⇒ no poles for x>=0); `e` pinned
    # (removes the form's scale degeneracy). Canonical init = the fixed Narkowicz curve.
    TONEMAP_CANONICAL = (2.51, 0.03, 2.43, 0.59)   # (a, b, c, d)
    TONEMAP_E = 0.14                                # pinned denominator constant

    @staticmethod
    def _softplus_inverse(y, eps=1e-8):
        y = torch.clamp(y, min=eps)
        return torch.log(torch.expm1(y) + eps)

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.rotation_activation = torch.nn.functional.normalize
        self.sigma_t_activation = lambda x: torch.clamp(
            torch.nn.functional.softplus(x), max=self.SIGMA_T_MAX)
        # Inverse softplus (valid below the clamp): x = log(expm1(y)).
        self.sigma_t_inverse_activation = lambda y: torch.log(
            torch.expm1(torch.clamp(y, min=1e-6, max=4.999)))
        self.omega_activation = torch.sigmoid
        self.g_activation = lambda x: 0.8 * torch.tanh(x)
        # Per-Gaussian multiple-scattering octave weights: softplus keeps them
        # non-negative; each only rescales the physical basis (HG·T·ω), so chroma stays in ω.
        self.w_activation = torch.nn.functional.softplus


    def __init__(self):
        self._xyz = torch.empty(0)
        # Physical appearance parameters (raw, pre-activation). `_sigma_t` is the peak
        # extinction σ_t (intensive, 1/length, not mass; mass = σ_t·(2π)^(3/2)·|Σ|^(1/2)).
        self._sigma_t = torch.empty(0)   # (P,1) raw -> softplus(σ_t)
        self._omega = torch.empty(0)       # (P,3) raw -> sigmoid
        self._g = torch.empty(0)     # (P,1) raw -> tanh
        self._w = torch.empty(0)  # (P,6) raw -> softplus, MS octave energy
        # Global (per-scene) tonemap coeffs: (4,) raw -> softplus -> (a,b,c,d), own optimizer.
        self._tonemap = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.scale_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.tonemap_optimizer = None
        # --- Stage 2 environment lighting (frozen geometry; see EnvNet) ---
        # _sky_transfer: (P, n_sh) precomputed per-Gaussian sky-visibility SH transfer
        # V_lm. A BUFFER, NOT an nn.Parameter (geometry-derived, achromatic constant).
        self._sky_transfer = torch.empty(0)
        self.env_net = None                 # global EnvNet: v_l -> (T_sun, E_lm)
        self.env_optimizer = None
        self.env_sh_order = 2
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_sigma_t(self):
        return self.sigma_t_activation(self._sigma_t)

    @torch.no_grad()
    def project_sigma_t(self):
        """Keep raw density at or below softplus^-1(5), including after Adam steps.

        The forward cap stays in place. At its boundary the clamp has a nonzero
        derivative, so a saturated Gaussian can learn a lower density again.
        Project in place to preserve the Parameter identity and Adam state.
        """
        self._sigma_t.clamp_(max=self.SIGMA_T_RAW_MAX)

    @property
    def get_omega(self):
        return self.omega_activation(self._omega)

    @property
    def get_g(self):
        return self.g_activation(self._g)

    @property
    def get_w(self):
        # (P,6) non-negative per-Gaussian multiple-scattering octave weights.
        return self.w_activation(self._w)

    @property
    def get_tonemap_coeffs(self):
        """(a, b, c, d) positive tonemap coefficients (softplus of raw). `e` is
        the pinned constant TONEMAP_E. Returns None if no tonemap param exists
        (e.g. a model trained without --tonemap_learnable) — callers must test
        `is None`, since an empty tensor would be truthy-by-identity and would
        be taken for a valid coefficient set."""
        if self._tonemap.numel() == 0:
            return None
        return torch.nn.functional.softplus(self._tonemap)

    def apply_tonemap(self, img):
        """Apply the learnable Narkowicz-form rational curve to an image/tensor,
        elementwise and shared across channels. Differentiable in both `img` and
        the tonemap coeffs. x is clamped >=0 so the curve stays monotone-domain.
        Returns img unchanged if no tonemap param is present."""
        coeffs = self.get_tonemap_coeffs
        if coeffs is None:
            return img
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        e = self.TONEMAP_E
        x = img.clamp(min=0.0)
        return (x * (a * x + b)) / (x * (c * x + d) + e)

    @property
    def get_v_l(self):
        return torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=self._xyz.dtype)

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        P = fused_point_cloud.shape[0]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Physical parameter init (raw): σ_t = 0.1 gives τ_center = σ_t·√(2π)·s ≈ 0.25·s
        # (isotropic s; τ_center = mass·ℓ with mass = σ_t·(2π)^(3/2)·∏s and
        # ℓ = 1/(2π·∏s·√Σ(d_i/s_i)²) — the (2π)^(3/2) and ∏s cancel, leaving ∝ s).
        sigma_t_init = torch.full((P, 1), 0.1, dtype=torch.float, device="cuda")
        sigma_t_raw = self._softplus_inverse(sigma_t_init)
        omega_raw = inverse_sigmoid(torch.full((P, 3), 0.8, dtype=torch.float, device="cuda"))
        g_raw = torch.atanh(torch.full((P, 1), 0.7, dtype=torch.float, device="cuda"))
        # Octave weights init: softplus(raw) == 0.5^n (n=0..5), the fixed 6-octave schedule.
        w_target = torch.tensor([0.5 ** n for n in range(6)], dtype=torch.float, device="cuda")
        w_raw = self._softplus_inverse(w_target).unsqueeze(0).repeat(P, 1)  # (P,6)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._sigma_t = nn.Parameter(sigma_t_raw.requires_grad_(True))
        self.project_sigma_t()
        self._omega = nn.Parameter(omega_raw.requires_grad_(True))
        self._g = nn.Parameter(g_raw.requires_grad_(True))
        self._w = nn.Parameter(w_raw.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.project_sigma_t()
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Scale signal accumulates only the "growing-scale" direction.
        # In log-scale parameterization a negative grad on _scaling increases s.
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Per-Gaussian Σ(α·T) accumulator + visible-pass count, used by physical densify/prune.
        self.contribution_accum = torch.zeros((self.get_xyz.shape[0],), device="cuda")
        self.contribution_denom = torch.zeros((self.get_xyz.shape[0],), device="cuda")
        # Prune-immunity counter: freshly split / cloned / resurrected points get N settling steps.
        self.prune_grace = torch.zeros((self.get_xyz.shape[0],), dtype=torch.int32, device="cuda")
        self._prune_grace_iteration = 0

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._sigma_t], 'lr': training_args.sigma_t_lr, "name": "sigma_t"},
            {'params': [self._omega], 'lr': training_args.omega_lr, "name": "omega"},
            {'params': [self._g], 'lr': training_args.g_lr, "name": "g"},
            {'params': [self._w],
             'lr': getattr(training_args, "w_lr", training_args.g_lr),
             "name": "w"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr,
            lr_final=training_args.scaling_lr * 0.1,
            max_steps=training_args.iterations)

        decay_ratio = 0.1
        iters = training_args.iterations
        self.sigma_t_scheduler_args = get_expon_lr_func(
            lr_init=training_args.sigma_t_lr,
            lr_final=training_args.sigma_t_lr * decay_ratio,
            max_steps=iters)
        self.omega_scheduler_args = get_expon_lr_func(
            lr_init=training_args.omega_lr,
            lr_final=training_args.omega_lr * decay_ratio,
            max_steps=iters)
        self.g_scheduler_args = get_expon_lr_func(
            lr_init=training_args.g_lr,
            lr_final=training_args.g_lr * decay_ratio,
            max_steps=iters)
        _ow_lr = getattr(training_args, "w_lr", training_args.g_lr)
        self.w_scheduler_args = get_expon_lr_func(
            lr_init=_ow_lr,
            lr_final=_ow_lr * decay_ratio,
            max_steps=iters)

    def setup_tonemap(self, training_args):
        """Enable the learnable output tonemap (called only when --tonemap_learnable).
        The 4 global coeffs live in their OWN Adam: _prune_optimizer indexes every
        group with a per-Gaussian mask, which would crash on a global param. Gradients
        still flow from loss.backward() since _tonemap is a leaf in the render graph."""
        # Create the parameter at canonical init unless a checkpoint already populated it.
        if self._tonemap.numel() == 0:
            raw = self._softplus_inverse(
                torch.tensor(self.TONEMAP_CANONICAL, dtype=torch.float, device="cuda"))
            self._tonemap = nn.Parameter(raw.requires_grad_(True))
        else:
            self._tonemap = nn.Parameter(self._tonemap.detach().cuda().requires_grad_(True))

        tm_lr = getattr(training_args, "tonemap_lr", 1e-3)
        self.tonemap_optimizer = torch.optim.Adam(
            [{'params': [self._tonemap], 'lr': tm_lr, "name": "tonemap"}],
            lr=0.0, eps=1e-15)
        self.tonemap_scheduler_args = get_expon_lr_func(
            lr_init=tm_lr,
            lr_final=tm_lr * 0.1,
            max_steps=training_args.iterations)

    def update_tonemap_learning_rate(self, iteration):
        """Step the standalone tonemap optimizer's LR schedule (no-op if the
        learnable tonemap is disabled)."""
        if self.tonemap_optimizer is None:
            return
        lr = self.tonemap_scheduler_args(iteration)
        for param_group in self.tonemap_optimizer.param_groups:
            param_group['lr'] = lr

    # ---------------- Stage 2: environment lighting ----------------
    def setup_env(self, training_args, sh_order=None):
        """Build the global EnvNet + its OWN Adam (isolated from densify/prune, same
        reason as the tonemap optimizer). Skips creation if env_net is already loaded
        (load_ply). Per-Gaussian params are expected to be frozen by the caller."""
        if sh_order is None:
            sh_order = getattr(self, "env_sh_order", 2)
        assert sh_order == 2, "only SH2 env transfer supported"
        n_sh = (sh_order + 1) ** 2
        self.env_sh_order = sh_order
        if self.env_net is None:
            self.env_net = EnvNet(n_sh).cuda()
        env_lr = getattr(training_args, "env_lr", 1e-3)
        self.env_optimizer = torch.optim.Adam(self.env_net.parameters(), lr=env_lr, eps=1e-15)
        self.env_scheduler_args = get_expon_lr_func(
            lr_init=env_lr, lr_final=env_lr * 0.1, max_steps=training_args.iterations)

    def update_env_learning_rate(self, iteration):
        if self.env_optimizer is None:
            return
        lr = self.env_scheduler_args(iteration)
        for param_group in self.env_optimizer.param_groups:
            param_group['lr'] = lr

    def precompute_sky_transfer(self, n_dirs=48, sh_order=2):
        """Precompute the per-Gaussian sky-visibility transfer V_lm (SH2) over the upper
        hemisphere, reusing the light-space rasterizer for per-Gaussian transmittance
        toward each sky direction. Achromatic, geometry-only; stored in
        _sky_transfer (P, n_sh)."""
        from gaussian_renderer import compute_T_light_raster, normalized_gaussian_line_integral
        assert sh_order == 2, "only SH2 env transfer supported"
        self.env_sh_order = sh_order
        dirs = _fibonacci_hemisphere(n_dirs)          # (N,3) world, upper hemisphere (+Y)
        Y = _sh_basis_deg2(dirs)                      # (N,9)
        means3D = self.get_xyz.detach()
        s = self.get_scaling.detach()
        sigma_t = self.get_sigma_t.detach()
        rotation = self.get_rotation                  # normalized quats (compute_T_light_raster detaches)
        R_t = build_rotation(rotation).transpose(1, 2)
        P = means3D.shape[0]
        w = (2.0 * math.pi) / n_dirs                  # hemisphere Monte-Carlo solid-angle weight
        V = torch.zeros((P, (sh_order + 1) ** 2), device="cuda")
        with torch.no_grad():
            for j in range(n_dirs):
                d = dirs[j]
                l_local = torch.matmul(R_t, d.view(3, 1)).squeeze(-1)            # (P,3)
                line_int = normalized_gaussian_line_integral(s, l_local)         # (P,1)
                geom = ((2.0 * math.pi) ** 1.5) * torch.prod(s, dim=1, keepdim=True) * line_int
                tau = sigma_t * geom                                           # (P,1)
                T_sky = compute_T_light_raster(
                    means3D, tau, s, rotation, d).squeeze(-1)                   # (P,)
                V += (T_sky.unsqueeze(1) * Y[j].unsqueeze(0)) * w
        self._sky_transfer = V
        print(f"[gaussian_model] precomputed sky transfer V_lm {tuple(V.shape)} over {n_dirs} dirs")

    def apply_env(self, v_l):
        """Return (T_sun (3,), fill (P,3)) for the current sun direction, or (None, None)
        if env lighting is inactive. fill = ω · (V_lm · E_lm); T_sun is the global RGB
        sun transmittance. Differentiable in the EnvNet params only (V_lm, ω frozen)."""
        if self.env_net is None or self._sky_transfer.numel() == 0:
            return None, None
        v_l = v_l.to(self._sky_transfer.dtype)
        t_sun, e_lm = self.env_net(v_l)                 # (3,), (n_sh,3)
        fill = self.get_omega * (self._sky_transfer @ e_lm)  # (P,n_sh)@(n_sh,3) -> (P,3)
        return t_sun, fill

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        _sched_map = {
            "xyz": self.xyz_scheduler_args,
            "scaling": self.scaling_scheduler_args,
            "sigma_t": self.sigma_t_scheduler_args,
            "omega": self.omega_scheduler_args,
            "g": self.g_scheduler_args,
            "w": self.w_scheduler_args,
        }
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in _sched_map:
                param_group['lr'] = _sched_map[name](iteration)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('sigma_t')
        for i in range(3):
            l.append('omega_{}'.format(i))
        l.append('g')
        for i in range(self._w.shape[1]):
            l.append('w_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        sigma_t = self._sigma_t.detach().cpu().numpy()
        omega = self._omega.detach().cpu().numpy()
        g = self._g.detach().cpu().numpy()
        w = self._w.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, sigma_t, omega, g, w, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # The global tonemap coeffs are not per-vertex, so a tonemap.json sidecar next to
        # the PLY (when the param exists) carries them; load_ply / viewer read it back.
        if self._tonemap.numel() > 0:
            raw = self._tonemap.detach().cpu().numpy().tolist()
            coeffs = torch.nn.functional.softplus(self._tonemap).detach().cpu().numpy().tolist()
            sidecar = {
                "version": 1,
                "form": "narkowicz_pinned_e",   # f=(a x^2+b x)/(c x^2+d x+e)
                "e": self.TONEMAP_E,
                "raw": raw,                      # pre-softplus, the actual params
                "coeffs": coeffs,                # (a,b,c,d) = softplus(raw)
            }
            with open(os.path.join(os.path.dirname(path), "tonemap.json"), "w") as f:
                json.dump(sidecar, f, indent=2)

        # Env lighting: V_lm and the EnvNet weights are not PLY attributes -> sidecars next to the PLY.
        if self.env_net is not None and self._sky_transfer.numel() > 0:
            d = os.path.dirname(path)
            np.save(os.path.join(d, "sky_transfer.npy"), self._sky_transfer.detach().cpu().numpy())
            torch.save(self.env_net.state_dict(), os.path.join(d, "env_net.pt"))
            with open(os.path.join(d, "env.json"), "w") as f:
                json.dump({"version": 1, "sh_order": self.env_sh_order,
                           "n_sh": (self.env_sh_order + 1) ** 2}, f, indent=2)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        props = plydata.elements[0].properties
        names = [p.name for p in props]

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        # σ_t: prefer 'sigma_t'; fall back to the legacy 'extinction' column.
        st_col = "sigma_t" if "sigma_t" in names else "extinction"
        sigma_t = np.asarray(plydata.elements[0][st_col])[..., np.newaxis]

        # ω: prefer 'omega_{i}'; fall back to legacy 'albedo_{i}'.
        if "omega_0" in names:
            omega = np.stack([np.asarray(plydata.elements[0][f"omega_{i}"])
                              for i in range(3)], axis=1)
        else:
            omega = np.stack([np.asarray(plydata.elements[0][f"albedo_{i}"])
                              for i in range(3)], axis=1)
        g_col = "g" if "g" in names else "g_factor"
        g = np.asarray(plydata.elements[0][g_col])[..., np.newaxis]

        # Octave weights (6 cols): prefer 'w_{i}', then 'octave_weight_{i}', else 0.5^n softplus-inverse.
        ow_names = [p.name for p in props if p.name.startswith("w_")]
        if not ow_names:
            ow_names = [p.name for p in props if p.name.startswith("octave_weight_")]
        if ow_names:
            ow_names = sorted(ow_names, key=lambda x: int(x.split('_')[-1]))
            w = np.zeros((xyz.shape[0], len(ow_names)), dtype=np.float32)
            for idx, attr_name in enumerate(ow_names):
                w[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            target = np.array([0.5 ** n for n in range(6)], dtype=np.float32)
            raw = np.log(np.expm1(np.clip(target, 1e-8, None)) + 1e-8)  # softplus^-1
            w = np.tile(raw[None, :], (xyz.shape[0], 1))

        scale_names = [p.name for p in props if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in props if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sigma_t = nn.Parameter(torch.tensor(sigma_t, dtype=torch.float, device="cuda").requires_grad_(True))
        self.project_sigma_t()
        self._omega = nn.Parameter(torch.tensor(omega, dtype=torch.float, device="cuda").requires_grad_(True))
        self._g = nn.Parameter(torch.tensor(g, dtype=torch.float, device="cuda").requires_grad_(True))
        self._w = nn.Parameter(torch.tensor(w, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Restore tonemap coeffs from the sidecar if present; absent → apply_tonemap is a no-op.
        tm_path = os.path.join(os.path.dirname(path), "tonemap.json")
        if os.path.exists(tm_path):
            try:
                with open(tm_path) as f:
                    sidecar = json.load(f)
                raw = torch.tensor(sidecar["raw"], dtype=torch.float, device="cuda")
                self._tonemap = nn.Parameter(raw.requires_grad_(True))
                print(f"[gaussian_model] loaded learnable tonemap coeffs "
                      f"{sidecar.get('coeffs')} from {tm_path}")
            except Exception as ex:
                print(f"[gaussian_model] failed to read {tm_path}: {ex}; "
                      f"tonemap disabled.")
                self._tonemap = torch.empty(0)
        else:
            self._tonemap = torch.empty(0)

        # Restore Stage-2 environment lighting (transfer V_lm + EnvNet) if the sidecars exist.
        env_path = os.path.join(os.path.dirname(path), "env.json")
        st_path = os.path.join(os.path.dirname(path), "sky_transfer.npy")
        net_path = os.path.join(os.path.dirname(path), "env_net.pt")
        if os.path.exists(env_path) and os.path.exists(st_path) and os.path.exists(net_path):
            try:
                with open(env_path) as f:
                    meta = json.load(f)
                self.env_sh_order = int(meta.get("sh_order", 2))
                n_sh = (self.env_sh_order + 1) ** 2
                st = np.load(st_path)
                self._sky_transfer = torch.tensor(st, dtype=torch.float, device="cuda")
                self.env_net = EnvNet(n_sh).cuda()
                self.env_net.load_state_dict(torch.load(net_path, map_location="cuda"))
                print(f"[gaussian_model] loaded env lighting (V_lm {tuple(self._sky_transfer.shape)}, "
                      f"SH{self.env_sh_order}) from {os.path.dirname(path)}")
            except Exception as ex:
                print(f"[gaussian_model] failed to read env sidecars: {ex}; env disabled.")
                self._sky_transfer = torch.empty(0)
                self.env_net = None
        else:
            self._sky_transfer = torch.empty(0)
            self.env_net = None

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # Every group here is per-point, so the mask applies to all of them.
            # (Global params live in their own optimizers, e.g. tonemap / env.)
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._sigma_t = optimizable_tensors["sigma_t"]
        self._omega = optimizable_tensors["omega"]
        self._g = optimizable_tensors["g"]
        self._w = optimizable_tensors["w"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # tmp_radii is set only transiently inside physical_densify_and_prune, so it may be absent here.
        tmp_radii = getattr(self, "tmp_radii", None)
        if tmp_radii is not None:
            self.tmp_radii = tmp_radii[valid_points_mask]
        if hasattr(self, "contribution_accum") and self.contribution_accum.numel() > 0:
            self.contribution_accum = self.contribution_accum[valid_points_mask]
            self.contribution_denom = self.contribution_denom[valid_points_mask]
            self.prune_grace = self.prune_grace[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # Defensive only: every caller (clone / split)
            # supplies all seven per-point keys, and the optimizer holds exactly
            # those seven groups, so this branch does not currently trigger.
            # It guards a future global parameter added to this optimizer.
            if group["name"] not in tensors_dict:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # Alive-only gate: a never-visible point has no useful gradient; new-borns have grace.
        alive_or_grace = (self.contribution_denom > 0) | (self.prune_grace > 0)
        if alive_or_grace.numel() == n_init_points:
            selected_pts_mask = torch.logical_and(selected_pts_mask, alive_or_grace)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # σ_t is intensive (local density): children inherit unchanged; scale shrinks meaningfully.
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_sigma_t = self._sigma_t[selected_pts_mask].repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        new_g = self._g[selected_pts_mask].repeat(N,1)
        new_w = self._w[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        d = {
            "xyz": new_xyz,
            "sigma_t": new_sigma_t,
            "omega": new_omega,
            "g": new_g,
            "w": new_w,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._sigma_t = optimizable_tensors["sigma_t"]
        self._omega = optimizable_tensors["omega"]
        self._g = optimizable_tensors["g"]
        self._w = optimizable_tensors["w"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Append zero stats for new children; keep existing stats (the prune predicate needs them).
        n_new_children = N * int(selected_pts_mask.sum().item())
        n_kept = self.get_xyz.shape[0] - n_new_children
        self.contribution_accum = torch.cat([
            self.contribution_accum[:n_kept],
            torch.zeros((n_new_children,), device="cuda"),
        ])
        self.contribution_denom = torch.cat([
            self.contribution_denom[:n_kept],
            torch.zeros((n_new_children,), device="cuda"),
        ])
        new_grace = torch.full((n_new_children,), self.PRUNE_GRACE_STEPS, dtype=torch.int32, device="cuda")
        self.prune_grace = torch.cat([
            self.prune_grace[:n_kept],
            new_grace,
        ])

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        alive_or_grace = (self.contribution_denom > 0) | (self.prune_grace > 0)
        if alive_or_grace.numel() == self.get_xyz.shape[0]:
            selected_pts_mask = torch.logical_and(selected_pts_mask, alive_or_grace)

        # σ_t is intensive: clone inherits it as-is (no halving of the parent).
        new_xyz = self._xyz[selected_pts_mask]
        new_sigma_t = self._sigma_t[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_g = self._g[selected_pts_mask]
        new_w = self._w[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        d = {
            "xyz": new_xyz,
            "sigma_t": new_sigma_t,
            "omega": new_omega,
            "g": new_g,
            "w": new_w,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._sigma_t = optimizable_tensors["sigma_t"]
        self._omega = optimizable_tensors["omega"]
        self._g = optimizable_tensors["g"]
        self._w = optimizable_tensors["w"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        n_added = int(new_tmp_radii.shape[0])
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Append zero stats for clones; keep existing stats across densify rounds.
        n_kept = self.get_xyz.shape[0] - n_added
        self.contribution_accum = torch.cat([
            self.contribution_accum[:n_kept],
            torch.zeros((n_added,), device="cuda"),
        ])
        self.contribution_denom = torch.cat([
            self.contribution_denom[:n_kept],
            torch.zeros((n_added,), device="cuda"),
        ])
        new_grace = torch.full((n_added,), self.PRUNE_GRACE_STEPS, dtype=torch.int32, device="cuda")
        self.prune_grace = torch.cat([
            self.prune_grace[:n_kept],
            new_grace,
        ])

    # ---------- Physical densify / prune (cloud parameterisation) ----------

    @torch.no_grad()
    def advance_prune_grace(self, iteration):
        """Advance once per training iteration, BEFORE creating/resurrecting points.

        Repeated calls at the same iteration do nothing; skipped iterations use
        their actual elapsed distance. A point born at i stays protected until
        i + PRUNE_GRACE_STEPS, independent of the two pruning schedules.
        """
        elapsed = iteration - self._prune_grace_iteration
        if elapsed < 0:
            raise ValueError("Prune grace iteration must not move backwards")
        if elapsed > 0:
            self.prune_grace.sub_(elapsed).clamp_min_(0)
            self._prune_grace_iteration = iteration

    def add_contribution_stats(self, contribution):
        """Per-step accumulator for Σ(α·T) per Gaussian.

        contribution: (P,) tensor from the rasterizer's per-Gaussian accumulator,
        added after every forward pass during training.
        """
        if self.contribution_accum.numel() == 0:
            self.contribution_accum = torch.zeros((self.get_xyz.shape[0],), device="cuda")
            self.contribution_denom = torch.zeros((self.get_xyz.shape[0],), device="cuda")
            self.prune_grace = torch.zeros((self.get_xyz.shape[0],), dtype=torch.int32, device="cuda")
        # contribution may cover points since pruned/split; skip on shape mismatch (realigned on reset).
        if contribution.shape[0] != self.contribution_accum.shape[0]:
            return
        with torch.no_grad():
            self.contribution_accum += contribution
            # Count only visible Gaussians (non-zero contribution), so off-screen frames don't dilute.
            self.contribution_denom += (contribution > 0).float()

    def get_mean_contribution(self):
        """Return per-Gaussian average Σ(α·T) over the steps it was visible."""
        denom = self.contribution_denom.clamp(min=1.0)
        return self.contribution_accum / denom

    def physical_densify_and_prune(self, opt, iteration, radii, scene_extent):
        """Cloud-parameterisation-aware densify / prune.

        Density growth: stock xyz/scale-grad-driven clone+split.

        Pruning: image-contribution prune in place of the opacity threshold. Two
        channels are OR'd, both gated by the grace period (`prune_grace == 0`):
          A. contribution — visible at least `opt.prune_min_visible_frames` times
             (so the mean below is meaningful) AND mean contribution Σ(α·T) per
             visible frame below `opt.contribution_threshold`.
          B. dead point — not visible to a single camera in the current
             accumulator window. This channel is deliberately NOT gated by the
             visible-frames minimum: a never-visible Gaussian has no mean to
             judge, and requiring visibility of it would keep it forever.
        A Gaussian therefore does NOT have to have been visible to be removed.

        Resurrection lives in tick_post_densify_maintenance(), not here.
        """
        self.advance_prune_grace(iteration)
        # 1. Density growth (stock path, with adaptive threshold)
        denom_g = self.denom.clamp(min=1)
        grads = self.xyz_gradient_accum / denom_g
        grads[grads.isnan()] = 0.0

        max_grad_eff = opt.densify_grad_threshold
        if getattr(opt, "densify_adaptive", False):
            # Top-K% of gradients as the threshold this round (grads decay late in training).
            g = grads.squeeze().abs()
            valid = g > 0
            if valid.any():
                target_q = 1.0 - opt.densify_top_frac
                max_grad_eff = max(
                    torch.quantile(g[valid], target_q).item(),
                    opt.densify_grad_min,
                )

        if getattr(opt, "densify_scale_grad_threshold", -1.0) > 0:
            grads_scale = self.scale_gradient_accum / denom_g
            grads_scale[grads_scale.isnan()] = 0.0
            grads = torch.maximum(grads, grads_scale * (max_grad_eff / opt.densify_scale_grad_threshold))

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad_eff, scene_extent)
        self.densify_and_split(grads, max_grad_eff, scene_extent)

        # 2. Contribution-based prune (per-Gaussian prune_grace protects new-borns).
        self._prune_by_contribution(opt)

        self.tmp_radii = None
        torch.cuda.empty_cache()

    def _prune_by_contribution(self, opt):
        """Two-channel prune for the physical strategy; the channels are OR'd.

        Both are gated by `grace_expired` (newly born / resurrected points still get
        a settling window), and `visible_enough` gates ONLY the contribution channel
        — a point that is never seen has no opportunity to contribute.

        Channel A — contribution: visible enough times yet projecting virtually no
            light onto valid pixels → dead weight regardless of geometry.

        Channel B — dead point: grace expired and not visible to a single camera in
            the current accumulator window (outside every frustum, or culled by the
            rasterizer before it deposits a pixel). Unlike channel A it is NOT gated
            by `opt.prune_min_visible_frames` — a never-visible Gaussian has no mean
            contribution to threshold, so requiring visibility would never remove it.
        """
        if self.get_xyz.shape[0] == 0:
            return 0
        grace_expired = self.prune_grace == 0
        visible_enough = self.contribution_denom >= opt.prune_min_visible_frames
        mean_contrib = self.get_mean_contribution()

        # A. Contribution channel — visible-but-low.
        contrib_mask = (
            grace_expired
            & visible_enough
            & (mean_contrib < opt.contribution_threshold)
        )

        # B. Dead-point channel — never visible after settling.
        dead_mask = grace_expired & (self.contribution_denom == 0)

        prune_mask = contrib_mask | dead_mask
        n = int(prune_mask.sum().item())
        if n > 0:
            self.prune_points(prune_mask)
        return n

    def tick_post_densify_maintenance(self, opt, iteration):
        """Per-iteration housekeeping during the densify window (despite the name it runs
        ONLY while iteration < densify_until_iter and is a no-op afterwards; the early
        return below is intentional):

          - σ_t resurrect of bottom `opt.resurrect_fraction` Gaussians every
            `opt.resurrect_interval` iterations.
          - Contribution prune every `opt.post_densify_prune_interval` iterations.
          - Reset of the contribution accumulators so the running mean tracks current
            model state, every `opt.contribution_reset_interval` iterations.
        """
        if iteration <= 0 or iteration >= getattr(opt, "densify_until_iter", float("inf")):
            return
        self.advance_prune_grace(iteration)
        # Order matters: resurrect → prune → reset — the prune predicate gates on
        # `contribution_denom >= prune_min_visible_frames`, so zeroing first masks every point.
        # 1. Resurrect schedule
        if (
            opt.resurrect_interval > 0
            and iteration % opt.resurrect_interval == 0
        ):
            self._resurrect_low_contribution(opt.resurrect_fraction)

        # 2. Contribution prune: reclaim low-contribution / dead points between the regular passes.
        prune_iv = getattr(opt, "post_densify_prune_interval", 0)
        if prune_iv > 0 and iteration % prune_iv == 0:
            self._prune_by_contribution(opt)

        # 3. Accumulator reset (must come AFTER prune in this tick — see note above)
        reset_iv = getattr(opt, "contribution_reset_interval", 1000)
        if reset_iv > 0 and iteration % reset_iv == 0:
            with torch.no_grad():
                if self.contribution_accum.numel() > 0:
                    self.contribution_accum.zero_()
                    self.contribution_denom.zero_()

    def _resurrect_low_contribution(self, fraction):
        """Reset σ_t of the lowest-contribution `fraction` of Gaussians back toward the
        initial value (0.1), letting them rejoin gradient flow. Only σ_t is touched;
        xyz / scale / rotation / albedo / g stay put."""
        if fraction <= 0 or self.get_xyz.shape[0] == 0:
            return
        with torch.no_grad():
            mean_contrib = self.get_mean_contribution()
            P = mean_contrib.shape[0]
            k = max(1, int(P * fraction))
            _, low_idx = torch.topk(mean_contrib, k, largest=False)
            # Skip points that are already in grace (recently born).
            low_idx = low_idx[self.prune_grace[low_idx] == 0]
            if low_idx.numel() == 0:
                return
            init_sigma_t = torch.full((low_idx.numel(), 1), 0.1, device="cuda")
            new_sigma_t = self._sigma_t.detach().clone()
            new_sigma_t[low_idx] = self._softplus_inverse(init_sigma_t)
            optimizable_tensors = self.replace_tensor_to_optimizer(new_sigma_t, "sigma_t")
            self._sigma_t = optimizable_tensors["sigma_t"]
            # Grant grace so they don't get pruned before σ_t has time to grow.
            self.prune_grace[low_idx] = self.PRUNE_GRACE_STEPS

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # Only the scale-growing direction (negative raw grad → larger s in log-parameterization).
        if self._scaling.grad is not None:
            grow = (-self._scaling.grad[update_filter]).detach().clamp(min=0).sum(dim=-1, keepdim=True)
            self.scale_gradient_accum[update_filter] += grow
        self.denom[update_filter] += 1

