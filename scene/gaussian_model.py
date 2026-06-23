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
    """Stage-2 global environment-lighting net: sun_dir (3,) -> (T_sun RGB<=1, E_lm SH).

    GLOBAL — one sky for the whole scene, a function of sun direction ONLY. It adds
    NO per-Gaussian colour DOF (per-Gaussian colour stays in the frozen albedo ρ),
    so the env term cannot regress the model to vanilla 3DGS / break relighting.

    T_sun is an ANALYTIC atmospheric transmittance with just 3 learnable params —
    the per-channel zenith optical depths τ=(τ_R,τ_G,τ_B):

        T_sun(sun_dir) = exp( − m(θ) · softplus(raw_tau) ),   θ = sun zenith angle

    m(θ) is the Kasten-Young air mass (a fixed geometric function, not learned);
    softplus keeps τ≥0 so T_sun∈(0,1]. Reddening (τ_B>τ_G>τ_R, Rayleigh ∝λ^-4) and
    low-sun dimming (m grows toward the horizon) fall out automatically; azimuth
    independence is exact (depends on θ only) — both confirmed empirically. raw_tau
    is initialised to a near-neutral, slightly-Rayleigh τ so Stage 2 starts ≈ the
    frozen Stage-1 sun render and learns the atmosphere on top.

    E_lm (additive sky in-scatter SH) keeps a small MLP; in a sun-dominated scene it
    learns ≈0 (sky fill negligible) and the env reduces to T_sun, but the term stays
    for generality (sky-dominated scenes where fill matters)."""
    # Near-neutral zenith optical depth init (R,G,B): τ_B>τ_R gives a faint Rayleigh
    # tilt; small so T_sun(zenith)≈exp(-τ)≈[0.98,0.96,0.93], ~Stage-1 neutral.
    TAU_INIT = (0.02, 0.04, 0.07)

    @staticmethod
    def _softplus_inverse(y, eps=1e-8):
        y = torch.clamp(torch.as_tensor(y, dtype=torch.float), min=eps)
        return torch.log(torch.expm1(y) + eps)

    @staticmethod
    def _air_mass(sun_dir):
        """Kasten-Young (1989) relative air mass from a sun direction (up=+Y).
        m(θ)=1/(cosθ + 0.50572 (96.07995-θ_deg)^-1.6364); finite at the horizon.
        Only valid for the upper hemisphere; cosθ is clamped to ~horizon so the
        formula never goes negative/explosive — below-horizon dimming is handled
        separately by the horizon gate in forward()."""
        cos_theta = torch.clamp(sun_dir.reshape(3)[1], 0.0, 1.0)    # up = +Y, upper hemi only
        theta_deg = torch.rad2deg(torch.arccos(cos_theta))
        denom = cos_theta + 0.50572 * torch.clamp(96.07995 - theta_deg, min=1e-3) ** (-1.6364)
        return 1.0 / torch.clamp(denom, min=1e-3)

    # Horizon softness (in cosθ units) for the below-horizon sun gate: T_sun fades to 0
    # over roughly cosθ ∈ [0, HORIZON_SOFT] so the sun "sets" smoothly instead of
    # snapping off. ~3° band.
    HORIZON_SOFT = 0.05

    def __init__(self, n_sh, hidden=64):
        super().__init__()
        self.n_sh = n_sh
        # T_sun: 3 learnable per-channel zenith optical depths (raw -> softplus -> τ≥0).
        self.raw_tau = nn.Parameter(self._softplus_inverse(self.TAU_INIT))
        # E_lm: small MLP of sun_dir (additive sky in-scatter), zero-init -> neutral.
        self.backbone = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.e_head = nn.Linear(hidden, n_sh * 3)   # -> E_lm (n_sh,3) sky radiance SH
        nn.init.zeros_(self.e_head.weight); nn.init.zeros_(self.e_head.bias)

    def forward(self, sun_dir):
        tau = torch.nn.functional.softplus(self.raw_tau)            # (3,) ≥0
        m = self._air_mass(sun_dir)                                 # scalar (upper hemi)
        t_sun = torch.exp(-m * tau)                                 # (3,) ∈(0,1]
        # Below-horizon gate: the sun below the horizon (cosθ≤0) means no direct
        # sunlight (occluded by the planet / extreme atmospheric path), so T_sun
        # fades to 0 over a soft band. Smooth (smoothstep) so relighting the sun
        # past the horizon "sets" continuously instead of snapping to white.
        cos_theta = sun_dir.reshape(3)[1]
        t = torch.clamp(cos_theta / self.HORIZON_SOFT, 0.0, 1.0)
        gate = t * t * (3.0 - 2.0 * t)                              # smoothstep(0,HORIZON_SOFT)
        t_sun = t_sun * gate
        h = self.backbone(sun_dir.reshape(1, 3))
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
    """Real orthonormal SH basis up to l=2 (9 coeffs) at unit dirs (M,3) -> (M,9).
    Matches utils.sh_utils.eval_sh sign/constant convention so the V_lm projection
    and any reconstruction stay consistent."""
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

    # Learnable output tonemap (Narkowicz ACES rational form), shared across RGB:
    #     f(x) = (a x^2 + b x) / (c x^2 + d x + e)
    # (a,b,c,d) learned via softplus(raw): positivity gives no poles for x>=0 and
    # output >= 0, so the curve stays smooth and bounded. `e` is pinned to remove
    # the rational form's overall-scale degeneracy (scaling numerator and
    # denominator by k leaves f unchanged), leaving 4 well-posed DoF. Canonical
    # init reproduces the fixed Narkowicz curve at iteration 0.
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
        self.extinction_activation = lambda x: torch.clamp(torch.nn.functional.softplus(x), max=5.0)
        # Inverse softplus (valid below the clamp): x = log(expm1(y)).
        self.extinction_inverse_activation = lambda y: torch.log(
            torch.expm1(torch.clamp(y, min=1e-6, max=4.999)))
        self.albedo_activation = torch.sigmoid
        self.g_factor_activation = lambda x: 0.8 * torch.tanh(x)
        # Per-Gaussian multiple-scattering octave weights: softplus keeps them
        # non-negative (scattered energy cannot be negative). A scalar-per-octave
        # weight only rescales the physical basis functions (HG·T·ρ), so chroma
        # stays locked in the albedo ρ and lighting still flows through every
        # term — it cannot bypass the physical model.
        self.octave_weight_activation = torch.nn.functional.softplus


    def __init__(self):
        self._xyz = torch.empty(0)
        # Physical appearance parameters (raw, pre-activation)
        # `_extinction` stores the peak extinction coefficient β_peak (intensive, 1/length),
        # NOT total mass. Mass = β_peak · (2π)^(3/2) · |Σ|^(1/2) is derived in the renderer.
        self._extinction = torch.empty(0)   # (P,1) raw -> softplus(β_peak)
        self._albedo = torch.empty(0)       # (P,3) raw -> sigmoid
        self._g_factor = torch.empty(0)     # (P,1) raw -> tanh
        self._octave_weights = torch.empty(0)  # (P,6) raw -> softplus, MS octave energy
        # Global (per-scene, NOT per-Gaussian) learnable output-tonemap coeffs:
        # (4,) raw -> softplus -> (a,b,c,d). Lives in its own optimizer.
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
        self.env_net = None                 # global EnvNet: sun_dir -> (T_sun, E_lm)
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
    def get_extinction(self):
        return self.extinction_activation(self._extinction)

    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)

    @property
    def get_g_factor(self):
        return self.g_factor_activation(self._g_factor)

    @property
    def get_octave_weights(self):
        # (P,6) non-negative per-Gaussian multiple-scattering octave weights.
        return self.octave_weight_activation(self._octave_weights)

    @property
    def get_tonemap_coeffs(self):
        """(a, b, c, d) positive tonemap coefficients (softplus of raw). `e` is
        the pinned constant TONEMAP_E. Empty tensor if no tonemap param exists
        (e.g. a model trained without --tonemap_learnable)."""
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
    def get_sun_dir(self):
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

        # Physical parameter initialization (raw / pre-activation)
        # `_extinction` stores the peak extinction coefficient β_peak directly.
        # β_peak ≈ 0.1 gives initial τ_center = β_peak·√(2π)·s ≈ 0.25·s at unit scale.
        beta_peak_init = torch.full((P, 1), 0.1, dtype=torch.float, device="cuda")
        extinction_raw = self._softplus_inverse(beta_peak_init)
        albedo_raw = inverse_sigmoid(torch.full((P, 3), 0.8, dtype=torch.float, device="cuda"))
        g_factor_raw = torch.atanh(torch.full((P, 1), 0.7, dtype=torch.float, device="cuda"))
        # Octave weights initialised so softplus(raw) == 0.5^n for n=0..5, i.e.
        # iteration 0 reproduces the fixed 6-octave a^n=0.5^n schedule.
        octave_target = torch.tensor([0.5 ** n for n in range(6)], dtype=torch.float, device="cuda")
        octave_weights_raw = self._softplus_inverse(octave_target).unsqueeze(0).repeat(P, 1)  # (P,6)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._extinction = nn.Parameter(extinction_raw.requires_grad_(True))
        self._albedo = nn.Parameter(albedo_raw.requires_grad_(True))
        self._g_factor = nn.Parameter(g_factor_raw.requires_grad_(True))
        self._octave_weights = nn.Parameter(octave_weights_raw.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Scale signal accumulates only the "growing-scale" direction.
        # In log-scale parameterization a negative grad on _scaling increases s.
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Per-Gaussian Σ(α·T) accumulator + count of forward passes contributed.
        # Used by the physical densify_and_prune logic to spot Gaussians with
        # negligible image contribution (regardless of opacity).
        self.contribution_accum = torch.zeros((self.get_xyz.shape[0],), device="cuda")
        self.contribution_denom = torch.zeros((self.get_xyz.shape[0],), device="cuda")
        # Per-Gaussian "frozen" counter: when a Gaussian was just split / cloned
        # / resurrected, give it N steps of prune immunity so it has a chance to
        # absorb gradient before being judged.
        self.prune_grace = torch.zeros((self.get_xyz.shape[0],), dtype=torch.int32, device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._extinction], 'lr': training_args.extiction_lr, "name": "extinction"},
            {'params': [self._albedo], 'lr': training_args.feature_lr, "name": "albedo"},
            {'params': [self._g_factor], 'lr': training_args.g_factor_lr, "name": "g_factor"},
            {'params': [self._octave_weights],
             'lr': getattr(training_args, "octave_weights_lr", training_args.g_factor_lr),
             "name": "octave_weights"},
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

        # Physical parameter LR decay: decay to 1/10 of initial by end of training
        decay_ratio = 0.1
        iters = training_args.iterations
        self.extinction_scheduler_args = get_expon_lr_func(
            lr_init=training_args.extiction_lr,
            lr_final=training_args.extiction_lr * decay_ratio,
            max_steps=iters)
        self.albedo_scheduler_args = get_expon_lr_func(
            lr_init=training_args.feature_lr,
            lr_final=training_args.feature_lr * decay_ratio,
            max_steps=iters)
        self.g_factor_scheduler_args = get_expon_lr_func(
            lr_init=training_args.g_factor_lr,
            lr_final=training_args.g_factor_lr * decay_ratio,
            max_steps=iters)
        _ow_lr = getattr(training_args, "octave_weights_lr", training_args.g_factor_lr)
        self.octave_weights_scheduler_args = get_expon_lr_func(
            lr_init=_ow_lr,
            lr_final=_ow_lr * decay_ratio,
            max_steps=iters)

    def setup_tonemap(self, training_args):
        """Enable the learnable output tonemap (called only when
        --tonemap_learnable). The 4 global coeffs live in their OWN Adam,
        isolated from the main optimizer's densify/prune machinery
        (_prune_optimizer indexes every group with a per-Gaussian mask, which
        would crash on a global param). Gradients still flow from loss.backward()
        since _tonemap is a leaf in the render graph."""
        # Create the parameter at canonical init unless a checkpoint already
        # populated it (resume / load_ply).
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
        """Precompute the per-Gaussian sky-visibility transfer V_lm (SH2) over the
        upper hemisphere, reusing the light-space rasterizer for per-Gaussian
        transmittance toward each sky direction. Achromatic, geometry-only; run once
        on the frozen Stage-1 model. Stored in _sky_transfer (P, n_sh)."""
        from gaussian_renderer import compute_T_light_raster, normalized_gaussian_line_integral
        assert sh_order == 2, "only SH2 env transfer supported"
        self.env_sh_order = sh_order
        dirs = _fibonacci_hemisphere(n_dirs)          # (N,3) world, upper hemisphere (+Y)
        Y = _sh_basis_deg2(dirs)                      # (N,9)
        means3D = self.get_xyz.detach()
        s = self.get_scaling.detach()
        beta_peak = self.get_extinction.detach()
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
                tau = beta_peak * geom                                           # (P,1)
                T_sky = compute_T_light_raster(
                    means3D, tau, s, rotation, d).squeeze(-1)                   # (P,)
                V += (T_sky.unsqueeze(1) * Y[j].unsqueeze(0)) * w
        self._sky_transfer = V
        print(f"[gaussian_model] precomputed sky transfer V_lm {tuple(V.shape)} over {n_dirs} dirs")

    def apply_env(self, sun_dir):
        """Return (T_sun (3,), fill (P,3)) for the current sun direction, or (None,
        None) if env lighting is not active. fill = ρ · (V_lm · E_lm); T_sun is the
        global RGB sun transmittance. Differentiable in the EnvNet params only
        (V_lm and ρ are frozen)."""
        if self.env_net is None or self._sky_transfer.numel() == 0:
            return None, None
        sun_dir = sun_dir.to(self._sky_transfer.dtype)
        t_sun, e_lm = self.env_net(sun_dir)                 # (3,), (n_sh,3)
        fill = self.get_albedo * (self._sky_transfer @ e_lm)  # (P,n_sh)@(n_sh,3) -> (P,3)
        return t_sun, fill

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        _sched_map = {
            "xyz": self.xyz_scheduler_args,
            "scaling": self.scaling_scheduler_args,
            "extinction": self.extinction_scheduler_args,
            "albedo": self.albedo_scheduler_args,
            "g_factor": self.g_factor_scheduler_args,
            "octave_weights": self.octave_weights_scheduler_args,
        }
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in _sched_map:
                param_group['lr'] = _sched_map[name](iteration)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('extinction')
        for i in range(3):
            l.append('albedo_{}'.format(i))
        l.append('g_factor')
        for i in range(self._octave_weights.shape[1]):
            l.append('octave_weight_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        extinction = self._extinction.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        g_factor = self._g_factor.detach().cpu().numpy()
        octave_weights = self._octave_weights.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, extinction, albedo, g_factor, octave_weights, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # The global learnable tonemap coeffs are not per-vertex, so they can't
        # ride the PLY. Mirror the metrics.json sidecar convention: write a small
        # tonemap.json next to the PLY (only when the param exists). The viewer
        # and load_ply read it back to reproduce the exact output curve.
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

        # Stage-2 environment lighting: the per-Gaussian transfer V_lm and the global
        # EnvNet weights are not per-vertex PLY attributes -> write sidecars next to
        # the PLY (only when present). load_ply / viewer read them back.
        if self.env_net is not None and self._sky_transfer.numel() > 0:
            d = os.path.dirname(path)
            np.save(os.path.join(d, "sky_transfer.npy"), self._sky_transfer.detach().cpu().numpy())
            torch.save(self.env_net.state_dict(), os.path.join(d, "env_net.pt"))
            with open(os.path.join(d, "env.json"), "w") as f:
                json.dump({"version": 1, "sh_order": self.env_sh_order,
                           "n_sh": (self.env_sh_order + 1) ** 2}, f, indent=2)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        extinction = np.asarray(plydata.elements[0]["extinction"])[..., np.newaxis]
        albedo = np.stack((np.asarray(plydata.elements[0]["albedo_0"]),
                           np.asarray(plydata.elements[0]["albedo_1"]),
                           np.asarray(plydata.elements[0]["albedo_2"])), axis=1)
        g_factor = np.asarray(plydata.elements[0]["g_factor"])[..., np.newaxis]

        # Per-Gaussian octave weights (6 cols). Backward-compat: PLYs saved before
        # this parameter existed have no octave_weight_* columns — fall back to the
        # fixed 0.5^n schedule (softplus-inverse) so old checkpoints still load.
        ow_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("octave_weight_")]
        if ow_names:
            ow_names = sorted(ow_names, key=lambda x: int(x.split('_')[-1]))
            octave_weights = np.zeros((xyz.shape[0], len(ow_names)), dtype=np.float32)
            for idx, attr_name in enumerate(ow_names):
                octave_weights[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            target = np.array([0.5 ** n for n in range(6)], dtype=np.float32)
            raw = np.log(np.expm1(np.clip(target, 1e-8, None)) + 1e-8)  # softplus^-1
            octave_weights = np.tile(raw[None, :], (xyz.shape[0], 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._extinction = nn.Parameter(torch.tensor(extinction, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._g_factor = nn.Parameter(torch.tensor(g_factor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._octave_weights = nn.Parameter(torch.tensor(octave_weights, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Restore the learnable tonemap coeffs from the sidecar if present.
        # Absent → linear-space / fixed-ACES model: leave _tonemap empty so
        # get_tonemap_coeffs returns None and apply_tonemap is a no-op.
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

        # Restore Stage-2 environment lighting (transfer V_lm + EnvNet) if present.
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
            # Some parameter groups are global (not per-point) and should not be pruned.

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
        self._extinction = optimizable_tensors["extinction"]
        self._albedo = optimizable_tensors["albedo"]
        self._g_factor = optimizable_tensors["g_factor"]
        self._octave_weights = optimizable_tensors["octave_weights"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # tmp_radii is only set transiently inside physical_densify_and_prune;
        # post-densify prune paths call prune_points with tmp_radii=None (or
        # unset before the first densify round) and don't need it downstream.
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
            # Some parameter groups are global (not per-point) and should not be extended.
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
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # Alive-only gate: a Gaussian that has never been visible to any
        # camera in the current accumulator window has no useful gradient
        # to propagate; cloning/splitting it just produces more dead
        # points that pile up until the next prune. New-born points are
        # protected by their grace counter.
        alive_or_grace = (self.contribution_denom > 0) | (self.prune_grace > 0)
        if alive_or_grace.numel() == n_init_points:
            selected_pts_mask = torch.logical_and(selected_pts_mask, alive_or_grace)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # β_peak is intensive (local density): children inherit unchanged; scale shrinks meaningfully.
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_extinction = self._extinction[selected_pts_mask].repeat(N,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
        new_g_factor = self._g_factor[selected_pts_mask].repeat(N,1)
        new_octave_weights = self._octave_weights[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        d = {
            "xyz": new_xyz,
            "extinction": new_extinction,
            "albedo": new_albedo,
            "g_factor": new_g_factor,
            "octave_weights": new_octave_weights,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._extinction = optimizable_tensors["extinction"]
        self._albedo = optimizable_tensors["albedo"]
        self._g_factor = optimizable_tensors["g_factor"]
        self._octave_weights = optimizable_tensors["octave_weights"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Append zero stats for new children; preserve existing points' stats so
        # the prune predicate (visible_enough = denom ≥ 5) keeps working inside
        # the densify phase. Wholesale reset here stalls aniso / contribution
        # prune.
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
        new_grace = torch.full((n_new_children,), 500, dtype=torch.int32, device="cuda")
        self.prune_grace = torch.cat([
            self.prune_grace[:n_kept],
            new_grace,
        ])

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def split_needles(self, ratio_threshold, opt=None):
        """Surgical split of high-anisotropy Gaussians (ratio > threshold).

        The aniso tail is ~95% disks (two long axes, one thin axis), so this
        fattens the thin axis rather than splitting the major axis: children
        keep the parent's orientation and long axes, the min axis is doubled,
        and β_peak is reduced to conserve total extinction mass (β·∏s). The two
        children are offset by ±σ_major/2 along the major axis so the pair
        approximately covers the parent's footprint. Each pass halves the ratio
        of every offender; converges to the threshold in log2(max/threshold)
        passes.

        Returns the number of Gaussians split.
        """
        if self.get_xyz.shape[0] == 0:
            return 0
        scaling = self.get_scaling
        ratio = scaling.max(dim=1).values / scaling.min(dim=1).values.clamp(min=1e-6)
        mask = ratio > ratio_threshold
        n = int(mask.sum().item())
        if n == 0:
            return 0

        sel_scaling = scaling[mask]                                    # (n,3)
        major_idx = sel_scaling.argmax(dim=1)                          # (n,)
        minor_idx = sel_scaling.argmin(dim=1)                          # (n,)
        sigma_major = sel_scaling.gather(1, major_idx.unsqueeze(1))    # (n,1)
        sigma_minor = sel_scaling.gather(1, minor_idx.unsqueeze(1))    # (n,1)

        rots = build_rotation(self._rotation[mask])                    # (n,3,3)
        major_dir = rots.gather(
            2, major_idx.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)  # (n,3)

        offset = major_dir * (sigma_major * 0.5)
        parent_xyz = self.get_xyz[mask]
        new_xyz = torch.cat([parent_xyz + offset, parent_xyz - offset], dim=0)

        # Fatten the thin axis (x2) — ratio halves. Mass bookkeeping per child:
        # volume x2 (fattened axis), and TWO children replace one parent, so β
        # would drop x4 for exact total-mass conservation. Child footprints
        # overlap near the parent centre, so /3.2 is mass-neutral in practice.
        child_scaling = sel_scaling.clone()
        child_scaling.scatter_(1, minor_idx.unsqueeze(1), sigma_minor * 2.0)
        new_scaling = self.scaling_inverse_activation(child_scaling.repeat(2, 1))

        beta = self.get_extinction[mask]                               # activated (n,1)
        new_extinction = self.extinction_inverse_activation(
            (beta / 3.2).clamp(min=1e-6)).repeat(2, 1)

        new_rotation = self._rotation[mask].repeat(2, 1)
        new_albedo = self._albedo[mask].repeat(2, 1)
        new_g_factor = self._g_factor[mask].repeat(2, 1)
        new_octave_weights = self._octave_weights[mask].repeat(2, 1)

        d = {
            "xyz": new_xyz,
            "extinction": new_extinction,
            "albedo": new_albedo,
            "g_factor": new_g_factor,
            "octave_weights": new_octave_weights,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._extinction = optimizable_tensors["extinction"]
        self._albedo = optimizable_tensors["albedo"]
        self._g_factor = optimizable_tensors["g_factor"]
        self._octave_weights = optimizable_tensors["octave_weights"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        n_children = 2 * n
        device = "cuda"
        # tmp_radii only exists transiently inside the densify pass; when the
        # surgery runs from the maintenance tick it is absent — skip then.
        tmp = getattr(self, "tmp_radii", None)
        if tmp is not None and tmp.numel():
            self.tmp_radii = torch.cat([tmp, torch.zeros(n_children, device=device)])
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        n_kept = self.get_xyz.shape[0] - n_children
        self.contribution_accum = torch.cat([
            self.contribution_accum[:n_kept],
            torch.zeros((n_children,), device=device)])
        self.contribution_denom = torch.cat([
            self.contribution_denom[:n_kept],
            torch.zeros((n_children,), device=device)])
        grace = 500 if opt is None else int(getattr(opt, "densify_from_iter", 500))
        self.prune_grace = torch.cat([
            self.prune_grace[:n_kept],
            torch.full((n_children,), grace, dtype=torch.int32, device=device)])

        # Remove the parents (mask refers to pre-cat indices; children appended after).
        prune_filter = torch.cat(
            [mask, torch.zeros(n_children, device=device, dtype=bool)])
        self.prune_points(prune_filter)
        return n

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # Alive-only gate (see densify_and_split for rationale).
        alive_or_grace = (self.contribution_denom > 0) | (self.prune_grace > 0)
        if alive_or_grace.numel() == self.get_xyz.shape[0]:
            selected_pts_mask = torch.logical_and(selected_pts_mask, alive_or_grace)

        # β_peak is intensive: clone inherits it as-is (no halving of the parent).
        new_xyz = self._xyz[selected_pts_mask]
        new_extinction = self._extinction[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_g_factor = self._g_factor[selected_pts_mask]
        new_octave_weights = self._octave_weights[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        d = {
            "xyz": new_xyz,
            "extinction": new_extinction,
            "albedo": new_albedo,
            "g_factor": new_g_factor,
            "octave_weights": new_octave_weights,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._extinction = optimizable_tensors["extinction"]
        self._albedo = optimizable_tensors["albedo"]
        self._g_factor = optimizable_tensors["g_factor"]
        self._octave_weights = optimizable_tensors["octave_weights"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        n_added = int(new_tmp_radii.shape[0])
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Append zero stats for clones; preserve existing points' stats so the
        # prune predicate stays alive across densify rounds. See densify_and_split
        # for the rationale.
        n_kept = self.get_xyz.shape[0] - n_added
        self.contribution_accum = torch.cat([
            self.contribution_accum[:n_kept],
            torch.zeros((n_added,), device="cuda"),
        ])
        self.contribution_denom = torch.cat([
            self.contribution_denom[:n_kept],
            torch.zeros((n_added,), device="cuda"),
        ])
        new_grace = torch.full((n_added,), 500, dtype=torch.int32, device="cuda")
        self.prune_grace = torch.cat([
            self.prune_grace[:n_kept],
            new_grace,
        ])

    # ---------- Physical densify / prune (cloud parameterisation) ----------

    def add_contribution_stats(self, contribution):
        """Per-step accumulator for Σ(α·T) per Gaussian.

        contribution: (P,) tensor from the rasterizer's per-Gaussian
        accumulator. Called after every forward pass during training.
        """
        if self.contribution_accum.numel() == 0:
            self.contribution_accum = torch.zeros((self.get_xyz.shape[0],), device="cuda")
            self.contribution_denom = torch.zeros((self.get_xyz.shape[0],), device="cuda")
            self.prune_grace = torch.zeros((self.get_xyz.shape[0],), dtype=torch.int32, device="cuda")
        # contribution may have been recorded for points that have since been
        # pruned/split; if shapes don't match, just skip (next forward will
        # re-align after stats reset in densify).
        if contribution.shape[0] != self.contribution_accum.shape[0]:
            return
        with torch.no_grad():
            self.contribution_accum += contribution
            # Only count this forward pass for Gaussians that were actually
            # visible (had non-zero contribution). Prevents off-screen frames
            # from diluting the average.
            self.contribution_denom += (contribution > 0).float()

    def get_mean_contribution(self):
        """Return per-Gaussian average Σ(α·T) over the steps it was visible."""
        denom = self.contribution_denom.clamp(min=1.0)
        return self.contribution_accum / denom

    def physical_densify_and_prune(self, opt, iteration, radii, scene_extent):
        """Cloud-parameterisation-aware densify / prune.

        Density growth: keep stock xyz/scale-grad-driven clone+split.

        Pruning: replace the opacity-threshold prune with image-contribution
        prune. A Gaussian is removed iff:
          - it was visible at all (contribution_denom > some min), AND
          - its mean contribution Σ(α·T) per visible frame falls below
            `opt.contribution_threshold`, AND
          - it is not currently in a grace period (prune_grace == 0).

        Resurrection: every `opt.resurrect_interval` iterations, reset the
        β_peak of the bottom `opt.resurrect_fraction` of Gaussians (by mean
        contribution) back toward the initialisation value, and grant them
        another grace period. This restores the predicate flow stock 3DGS
        gets from `reset_opacity()` — but under our β_peak parametrisation
        that reset is meaningless (opacity is analytic from extinction +
        scale), so we resurrect β_peak directly instead.
        """
        # 1. Density growth (stock path, with adaptive threshold)
        denom_g = self.denom.clamp(min=1)
        grads = self.xyz_gradient_accum / denom_g
        grads[grads.isnan()] = 0.0

        max_grad_eff = opt.densify_grad_threshold
        if getattr(opt, "densify_adaptive", False):
            # Take the top-K% of gradients as the threshold this round so that
            # densify keeps firing even when grads decay late in training.
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

        # 2. Contribution-based prune (replaces opacity threshold).
        # Per-Gaussian prune_grace protects new-borns; no global warmup needed.
        self._prune_by_contribution(opt)

        # Decay grace counter; accumulator reset is handled separately by
        # tick_post_densify_maintenance() so it stays alive post-densify.
        with torch.no_grad():
            self.prune_grace = (self.prune_grace - opt.densification_interval).clamp(min=0)

        self.tmp_radii = None
        torch.cuda.empty_cache()

    def _prune_by_contribution(self, opt):
        """Two-channel prune for the physical strategy.

        Each channel produces its own mask, gated only by `grace_expired`
        (so newly born / resurrected points still get a settling window),
        and the channels are OR'd together. Crucially, `visible_enough` is
        a gate ONLY for the contribution channel — a point that is never
        seen has no opportunity to contribute, so the contribution
        threshold doesn't apply to it. A separate dead-point channel
        handles the "never visible after grace expired" case.

        Channel A — contribution: visible enough times yet projects
            virtually no light onto valid pixels → dead weight regardless
            of geometry.

        Channel B — dead point: grace has expired but the point hasn't
            been visible to a single camera in the current accumulator
            window. Either it sits outside every frustum, or its scale is
            so small that the rasterizer culls it before it can deposit a
            single pixel. Without this channel such points pile up
            indefinitely (visible-frame-count = 0 makes the contribution
            channel above silently bypass them).

        There is no aniso prune channel: the full-schedule aniso regulariser
        holds p99 at a ~30 plateau, far below any sane prune ratio.
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
        """Per-iteration housekeeping during the densify window (resurrect + prune +
        accumulator reset). Despite the name, this runs ONLY while
        iteration < densify_until_iter and is a no-op afterwards — the early return
        below is intentional:

          Running resurrect/prune during the post-densify settle phase forms a net-
          destruction loop (resurrect→prune) that cost -17% points and -0.7 dB in
          testing, so maintenance is gated off once densify stops. Popping in the
          settle phase is held by the full-schedule aniso regulariser + needle
          surgery, not by a geometric prune.

          - β_peak resurrect of bottom `opt.resurrect_fraction` Gaussians every
            `opt.resurrect_interval` iterations.
          - Periodic reset of the contribution accumulators so the running mean
            tracks current model state, every `opt.contribution_reset_interval` iters.
        """
        if iteration <= 0 or iteration >= getattr(opt, "densify_until_iter", float("inf")):
            return
        # Order matters: resurrect → prune → reset. The prune predicate uses
        # `contribution_denom >= prune_min_visible_frames` as a gate, so zeroing
        # the accumulator first would mask every point and nothing would ever be
        # reclaimed (n_points freezes + aniso p99 grows unbounded).
        # 1. Resurrect schedule
        if (
            opt.resurrect_interval > 0
            and iteration % opt.resurrect_interval == 0
        ):
            self._resurrect_low_contribution(opt.resurrect_fraction)

        # 2. Contribution prune: reclaim low-contribution / dead points that
        # accumulate within the densify window between the regular prune passes.
        prune_iv = getattr(opt, "post_densify_prune_interval", 0)
        if prune_iv > 0 and iteration % prune_iv == 0:
            self._prune_by_contribution(opt)
            with torch.no_grad():
                # Match the grace decay rhythm used inside physical_densify_and_prune
                self.prune_grace = (self.prune_grace - prune_iv).clamp(min=0)

        # 3. Accumulator reset (must come AFTER prune in this tick — see note above)
        reset_iv = getattr(opt, "contribution_reset_interval", 1000)
        if reset_iv > 0 and iteration % reset_iv == 0:
            with torch.no_grad():
                if self.contribution_accum.numel() > 0:
                    self.contribution_accum.zero_()
                    self.contribution_denom.zero_()

    def _resurrect_low_contribution(self, fraction):
        """Reset β_peak of the lowest-contribution `fraction` of Gaussians
        back toward the initial value (0.1), letting them rejoin gradient flow.
        Only β_peak is touched; xyz / scale / rotation / albedo / g stay put.
        """
        if fraction <= 0 or self.get_xyz.shape[0] == 0:
            return
        with torch.no_grad():
            mean_contrib = self.get_mean_contribution()
            P = mean_contrib.shape[0]
            k = max(1, int(P * fraction))
            # Lowest k by mean contribution.
            _, low_idx = torch.topk(mean_contrib, k, largest=False)
            # Skip points that are already in grace (recently born).
            low_idx = low_idx[self.prune_grace[low_idx] == 0]
            if low_idx.numel() == 0:
                return
            init_beta = torch.full((low_idx.numel(), 1), 0.1, device="cuda")
            new_extinction = self._extinction.detach().clone()
            new_extinction[low_idx] = self._softplus_inverse(init_beta)
            # Replace param tensor in the optimiser (uses existing helper).
            optimizable_tensors = self.replace_tensor_to_optimizer(new_extinction, "extinction")
            self._extinction = optimizable_tensors["extinction"]
            # Grant grace so they don't get pruned before β has time to grow.
            self.prune_grace[low_idx] = 500

    # ----------------------------------------------------------------------

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # Only the scale-growing direction (negative raw grad → larger s in log-parameterization).
        if self._scaling.grad is not None:
            grow = (-self._scaling.grad[update_filter]).detach().clamp(min=0).sum(dim=-1, keepdim=True)
            self.scale_gradient_accum[update_filter] += grow
        self.denom[update_filter] += 1
