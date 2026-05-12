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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation


def normalized_gaussian_line_integral(scales, dirs_local):
    """
    Center-line integral of a normalized 3D Gaussian along a unit direction.

    For N(x; μ, Σ) with Σ = R diag(s^2) R^T and unit ray direction d_local in the
    Gaussian local frame, the full-line integral through the Gaussian centre is

        ∫ N(μ + t d) dt = 1 / (2π |S| prod(s) sqrt(sum((d_i / s_i)^2)))

    which is the exact 1D Gaussian integral for a normalized 3D Gaussian.
    """
    denom = (2.0 * math.pi) * torch.prod(scales, dim=1, keepdim=True) * torch.sqrt(
        torch.sum((dirs_local / scales) ** 2, dim=1, keepdim=True) + 1e-8
    )
    return 1.0 / (denom + 1e-8)


def compute_T_light(means3D, beta_peak, scales, rotations, sun_dir,
                    grid_res=128, stencil_buckets=(3, 7, 15, 31), chunk_size=16384):
    """
    Per-Gaussian sun transmittance via adaptive-stencil ellipsoid deposit.

    Gaussians are bucketed by their spatial extent measured in cell units
    (max(s) / min(cell_size)); each bucket runs with an appropriately-sized
    deposit stencil that covers roughly ±3σ for the largest Gaussian in the
    bucket. Smaller Gaussians reuse the cheaper small-stencil path, so the
    common case stays fast while the long tail of large Gaussians is still
    captured with physical fidelity.

    Within each bucket the deposit is the same scheme as before: transform
    cell centres into the Gaussian's principal frame, evaluate the PDF shape
    exp(-|local|²/2), and rescale so Σ_cell = mass_k exactly (mass-conserving,
    absorbs stencil truncation).

    Args:
        means3D:         (P, 3) Gaussian centres
        beta_peak:       (P, 1) peak extinction coefficient β_k (1/length)
        scales:          (P, 3) principal-axis scales
        rotations:       (P, 4) rotation quaternions (wxyz)
        sun_dir:         (3,)   normalised sun direction (assumed +Y)
        grid_res:        int    voxel resolution per axis
        stencil_buckets: tuple  ascending odd stencil sizes; last one is the cap
        chunk_size:      int    number of Gaussians per batch within a bucket

    Returns:
        T_light: (P, 1) sun transmittance per Gaussian centre
    """
    import torch.nn.functional as F
    device = means3D.device
    dtype = means3D.dtype
    P = means3D.shape[0]
    G = grid_res
    assert all((s % 2 == 1) and s > 0 for s in stencil_buckets), "stencil sizes must be positive odd ints"
    assert list(stencil_buckets) == sorted(stencil_buckets), "stencil_buckets must be ascending"

    # --- 1. Scene bounding box with 3σ padding -----------------------------
    with torch.no_grad():
        max_extent = scales.max().item()
        pad = 3.0 * max_extent
        bbox_min = means3D.min(dim=0).values - pad
        bbox_max = means3D.max(dim=0).values + pad
        bbox_size = bbox_max - bbox_min                    # (3,)
        cell_size = bbox_size / G                          # (3,)
        cell_vol = cell_size.prod()                        # scalar
        min_cell = cell_size.min().item()

    # Analytic mass for each normalized Gaussian.
    mass = beta_peak * ((2.0 * math.pi) ** 1.5) * torch.prod(scales, dim=1, keepdim=True)  # (P,1)
    inv_s = 1.0 / (scales + 1e-8)                                                          # (P,3)
    R_all = build_rotation(rotations)                                                       # (P,3,3)

    # --- 2. Bucket assignment (detached — geometric routing, not differentiable) ---
    # A stencil of size S covers ±(S/2) cells around the Gaussian centre. To cover
    # ±3σ we need half_cells ≥ 3·max(s)/min_cell, i.e. S ≥ 2·⌈3·max(s)/min_cell⌉ + 1.
    with torch.no_grad():
        max_s = scales.max(dim=1).values                                  # (P,)
        half_needed = torch.ceil(3.0 * max_s / max(min_cell, 1e-8))       # (P,) in cell-half units
        # Find smallest bucket whose half-size ≥ half_needed.
        bucket_halves = torch.tensor([s // 2 for s in stencil_buckets],
                                     device=device, dtype=half_needed.dtype)   # (B,)
        # (P, B) → first index where bucket_halves[b] >= half_needed[p]
        cmp = bucket_halves.unsqueeze(0) >= half_needed.unsqueeze(1)      # (P,B)
        # argmax returns first True; if all False, fall through to last bucket.
        bucket_ids = cmp.float().argmax(dim=1)                            # (P,)
        any_fits = cmp.any(dim=1)
        bucket_ids = torch.where(any_fits, bucket_ids,
                                 torch.full_like(bucket_ids, len(stencil_buckets) - 1))

    # --- 3. Per-bucket deposit --------------------------------------------
    volume = torch.zeros(G * G * G, device=device, dtype=dtype)

    def _deposit_bucket(idx_mask_nonzero, S):
        """Deposit the Gaussians indexed by idx_mask_nonzero using stencil size S."""
        nonlocal volume
        if idx_mask_nonzero.numel() == 0:
            return
        half = S // 2
        shifts_1d = torch.arange(-half, half + 1, device=device)
        sx, sy, sz = torch.meshgrid(shifts_1d, shifts_1d, shifts_1d, indexing='ij')
        stencil_offs = torch.stack([sx.reshape(-1), sy.reshape(-1), sz.reshape(-1)], dim=-1)  # (Kc, 3)
        Kc = stencil_offs.shape[0]

        Nb = idx_mask_nonzero.shape[0]
        for c0 in range(0, Nb, chunk_size):
            c1 = min(Nb, c0 + chunk_size)
            sel = idx_mask_nonzero[c0:c1]                                # (Pc,) long
            Pc = sel.shape[0]
            mu = means3D[sel]                                            # (Pc,3)
            Rt = R_all[sel].transpose(1, 2)                              # (Pc,3,3)
            inv_sc = inv_s[sel]                                          # (Pc,3)
            mass_c = mass[sel]                                           # (Pc,1)

            with torch.no_grad():
                center_ijk = torch.floor((mu - bbox_min) / cell_size).long()      # (Pc,3)
                cell_ijk = center_ijk.unsqueeze(1) + stencil_offs.unsqueeze(0)    # (Pc,Kc,3)
                in_bounds = ((cell_ijk >= 0) & (cell_ijk < G)).all(dim=-1)        # (Pc,Kc)
                cell_ijk_clamped = cell_ijk.clamp(0, G - 1)
                flat_idx = (cell_ijk_clamped[..., 0] * (G * G)
                            + cell_ijk_clamped[..., 1] * G
                            + cell_ijk_clamped[..., 2])                            # (Pc,Kc)
                cell_center_world = bbox_min + (cell_ijk_clamped.float() + 0.5) * cell_size

            delta = cell_center_world - mu.unsqueeze(1)                  # (Pc,Kc,3)
            local = torch.bmm(Rt, delta.transpose(1, 2)).transpose(1, 2) # (Pc,Kc,3)
            local = local * inv_sc.unsqueeze(1)
            r2 = (local * local).sum(dim=-1)                             # (Pc,Kc)

            shape = torch.exp(-0.5 * r2) * in_bounds.to(dtype)           # (Pc,Kc)
            per_gauss_sum = shape.sum(dim=1, keepdim=True).clamp(min=1e-12)
            cell_mass = shape * (mass_c / per_gauss_sum)                 # (Pc,Kc)

            volume = volume.scatter_add(
                0, flat_idx.reshape(-1), cell_mass.reshape(-1)
            )

    for b, S in enumerate(stencil_buckets):
        idx = (bucket_ids == b).nonzero(as_tuple=False).squeeze(-1)
        _deposit_bucket(idx, S)

    volume = volume.view(G, G, G)                                         # [x,y,z]

    # --- 4. mass-per-cell → σ_t average, integrate τ along +Y -------------
    sigma_field = volume / cell_vol                                       # (G,G,G)
    flipped = torch.flip(sigma_field, [1])
    inclusive_cs = torch.cumsum(flipped, dim=1)
    exclusive_cs = inclusive_cs - flipped
    tau_above_field = torch.flip(exclusive_cs, [1]) * cell_size[1]

    # --- 5. Trilinear sample τ_above at each Gaussian centre --------------
    with torch.no_grad():
        coords_norm = 2.0 * (means3D - bbox_min) / bbox_size - 1.0        # (P,3) in [-1,1]
        grid_pts = torch.stack([coords_norm[:, 2],
                                coords_norm[:, 1],
                                coords_norm[:, 0]], dim=-1)
        grid_pts = grid_pts.view(1, 1, 1, P, 3)

    tau_sun = F.grid_sample(
        tau_above_field.unsqueeze(0).unsqueeze(0),
        grid_pts,
        mode='bilinear', padding_mode='border', align_corners=True
    ).view(P, 1)

    T_light = torch.exp(-tau_sun)
    return T_light

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # Peak extinction coefficient β_peak (intensive, 1/length). Mass is derived below
    # once we have scale with scaling_modifier applied.
    beta_peak = pc.get_extinction  # (P,1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # --- Physical cloud shading (static sun lighting) ---
    # Sun irradiance: 4π compensates the 1/(4π) in the normalized HG phase function,
    # so that an isotropic (g=0), unit-albedo medium scatters all incoming light uniformly.
    sun_intensity = 4.0 * math.pi
    L_sun = torch.tensor([sun_intensity, sun_intensity, sun_intensity], device="cuda", dtype=means3D.dtype)
    L_dir = pc.get_sun_dir.to(dtype=means3D.dtype)

    # View direction: from point to camera (normalized)
    dir_pc = (viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D)
    v = dir_pc / (torch.linalg.norm(dir_pc, dim=1, keepdim=True) + 1e-8)

    # Build per-Gaussian rotation matrix (world <- local).
    # We need it for projecting direction vectors into the Gaussian's local frame.
    R = build_rotation(pc.get_rotation)  # (P,3,3)
    R_t = R.transpose(1, 2)

    # Local directions
    v_local = torch.bmm(R_t, v.unsqueeze(-1)).squeeze(-1)               # (P,3)
    l_local = torch.matmul(R_t, L_dir.view(3, 1)).squeeze(-1)           # (P,3)

    s = pc.get_scaling * scaling_modifier  # (P,3)
    mass = beta_peak * ((2.0 * math.pi) ** 1.5) * torch.prod(s, dim=1, keepdim=True)

    # Exact full-line 1D Gaussian integral for a normalized 3D Gaussian, evaluated
    # on the centre ray. Rasterization later multiplies by the projected 2D Gaussian,
    # yielding the intended approximation τ(x') ≈ τ_center · G_2D(x').
    line_int_view = normalized_gaussian_line_integral(s, v_local)
    line_int_sun = normalized_gaussian_line_integral(s, l_local)

    tau_view = mass * line_int_view
    tau_precomp = tau_view
    opacity = 1.0 - torch.exp(-tau_view)

    # cos(theta) between view dir and light dir
    cos_theta = torch.clamp((v * L_dir[None, :]).sum(dim=1, keepdim=True), -1.0, 1.0)

    # Henyey-Greenstein phase function with 1/(4π) normalization.
    g = pc.get_g_factor  # (P,1) in (-1,1)
    eps = 1e-6
    inv_4pi = 1.0 / (4.0 * math.pi)

    # Multi-octave scattering approximation (Frostbite / Wrenninge 2015).
    # Simulates multiple scattering bounces using the same physical parameters.
    # Higher octaves: less energy (a^n), less attenuation (T^(b^n)), more isotropic (g·c^n).
    ms_a = 0.5    # energy attenuation per bounce
    ms_b = 0.5    # transmittance power decay
    ms_c = 0.5    # phase isotropization rate
    num_octaves = 6

    tau_sun_per_gauss = mass * line_int_sun  # kept for optional diagnostics; unused in T_light now
    T_light = compute_T_light(means3D, beta_peak, s, pc.get_rotation, L_dir,
                              grid_res=128, stencil_buckets=(3, 7, 15, 31))

    scatter_sum = torch.zeros_like(mass)  # (P,1)
    for n in range(num_octaves):
        energy   = ms_a ** n
        g_eff    = g * (ms_c ** n)
        T_eff    = torch.pow(T_light.clamp(min=1e-8), ms_b ** n)
        denom_hg = torch.pow(1.0 + g_eff * g_eff - 2.0 * g_eff * cos_theta, 1.5) + eps
        HG_n     = inv_4pi * (1.0 - g_eff * g_eff) / denom_hg
        scatter_sum = scatter_sum + energy * T_eff * HG_n

    rho = pc.get_albedo  # (P,3)

    Lk = rho * L_sun[None, :] * scatter_sum
    colors_precomp = torch.clamp(Lk, 0.0, 1.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # Physical cloud shading is always precomputed per Gaussian before rasterization,
    # so the legacy separate SH path is intentionally bypassed here.
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        tau_precomp = tau_precomp,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "T_light": T_light.detach(),
        "Lk": Lk.detach(),
        }
    
    return out
