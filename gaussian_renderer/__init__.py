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


def compute_T_light(means3D, tau_per_gauss, scales, sun_dir, grid_res=128):
    """
    Approximate per-Gaussian sun transmittance via voxel grid (point-scatter).
    Fully differentiable w.r.t. tau_per_gauss (and means3D through grid_sample).

    Works with an arbitrary sun direction. The grid is built in a *light-space*
    frame whose third axis is `sun_dir`, so a single 1D prefix sum along that
    axis gives "tau above this voxel along the ray to the sun".

    1. Build an orthonormal basis (e1, e2, sun_dir) and rotate Gaussian centres
       into it.
    2. Build optical-depth field on a 3D grid in light-space (point-deposit).
    3. Exclusive prefix sum along the +sun axis → cumulative optical depth
       between each cell and the sun.
    4. Trilinear sample at each Gaussian centre → T_light = exp(-tau_sun).

    Args:
        means3D:        (P, 3) Gaussian centres
        tau_per_gauss:  (P, 1) per-Gaussian optical depth along the sun direction
        scales:         (P, 3) Gaussian scales (used only for bbox padding)
        sun_dir:        (3,)   normalised sun direction (any unit vector)
        grid_res:       int    voxel resolution per axis

    Returns:
        T_light: (P, 1) sun transmittance per Gaussian
    """
    import torch.nn.functional as F
    device = means3D.device
    dtype = means3D.dtype
    P = means3D.shape[0]

    # --- 0. Light-space orthonormal basis (e1, e2, sun_dir) ----------------
    # Rotation matrix R_lw maps a world-space vector v_w to light-space:
    #     v_L = R_lw @ v_w,  with R_lw = [[e1; e2; sun_dir]].
    # Inverse rotation R_lw^T maps light-space → world.
    with torch.no_grad():
        s = sun_dir.to(device=device, dtype=dtype).reshape(3)
        s = s / (torch.linalg.norm(s) + 1e-8)
        # Pick a helper axis that's not parallel to s.
        helper = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        if abs(float(torch.dot(s, helper).item())) > 0.95:
            helper = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        e1 = torch.linalg.cross(s, helper)
        e1 = e1 / (torch.linalg.norm(e1) + 1e-8)
        e2 = torch.linalg.cross(s, e1)
        e2 = e2 / (torch.linalg.norm(e2) + 1e-8)
        R_lw = torch.stack([e1, e2, s], dim=0)              # (3,3) rows are basis

    # --- 1. Light-space bbox with 3-sigma padding (detached) ---------------
    # Project all Gaussian centres into light-space, take axis-aligned bbox there.
    with torch.no_grad():
        means_L = means3D @ R_lw.T                          # (P,3) in light-space
        max_extent = scales.max(dim=1).values.max().item()
        pad = 3.0 * max_extent
        bbox_min = means_L.min(dim=0).values - pad
        bbox_max = means_L.max(dim=0).values + pad
        bbox_size = bbox_max - bbox_min                     # (3,)
        cell_size = bbox_size / grid_res                    # (3,)
        gi = ((means_L - bbox_min) / cell_size).long().clamp(0, grid_res - 1)
        flat_idx = gi[:, 0] * (grid_res * grid_res) + gi[:, 1] * grid_res + gi[:, 2]

    # --- 2. Scatter per-Gaussian tau into voxel grid (differentiable) -------
    volume = torch.zeros(grid_res * grid_res * grid_res, device=device, dtype=dtype)
    volume = volume.scatter_add(0, flat_idx, tau_per_gauss.squeeze(-1))
    # Indexed [light_x, light_y, light_z=sun_axis]
    volume = volume.view(grid_res, grid_res, grid_res)

    # --- 3. Exclusive prefix sum along the +sun axis (light-Z) -------------
    # tau_above[i,j,k] = Σ_{k'>k} volume[i,j,k']  (cells closer to the sun)
    flipped = torch.flip(volume, [2])
    inclusive_cs = torch.cumsum(flipped, dim=2)
    exclusive_cs = inclusive_cs - flipped
    tau_above = torch.flip(exclusive_cs, [2])

    # --- 4. Trilinear sample at Gaussian centres (in light-space) -----------
    with torch.no_grad():
        coords_norm = 2.0 * (means_L - bbox_min) / bbox_size - 1.0  # (P,3)
        # grid_sample expects (D=light_z, H=light_y, W=light_x) order with
        # the per-point stack [x, y, z] of *normalised* coords, but PyTorch's
        # 5D grid_sample is documented as `grid` last-dim = (x,y,z) with x
        # indexing the last input dim (W). To match the original code's
        # convention we feed (z, y, x).
        grid_pts = torch.stack([coords_norm[:, 2],
                                coords_norm[:, 1],
                                coords_norm[:, 0]], dim=-1)
        grid_pts = grid_pts.view(1, 1, 1, P, 3)

    tau_sun = F.grid_sample(
        tau_above.unsqueeze(0).unsqueeze(0),
        grid_pts,
        mode='bilinear', padding_mode='border', align_corners=True
    ).view(P, 1)

    T_light = torch.exp(-tau_sun)
    return T_light

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, precomputed_T_light=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    Viewer-only hooks (defaults preserve training behaviour):
        override_color:        (P, 3) tensor; if provided, replaces the physical Lk
                               as `colors_precomp` at rasterisation time. Used by
                               the interactive viewer to render diagnostic channels
                               (T_light, β_peak, …) instead of RGB.
        precomputed_T_light:   (P, 1) tensor; if provided, skip the expensive
                               compute_T_light call. Useful when the sun is static
                               and T_light only needs to be computed once.
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
        antialiasing=pipe.antialiasing,
        k_sigma=getattr(pipe, "k_sigma", 1.5),
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

    # --- Physical cloud shading ---
    # Sun irradiance: 4π compensates the 1/(4π) in the normalized HG phase function,
    # so that an isotropic (g=0), unit-albedo medium scatters all incoming light uniformly.
    sun_intensity = 4.0 * math.pi
    L_sun = torch.tensor([sun_intensity, sun_intensity, sun_intensity], device="cuda", dtype=means3D.dtype)
    # Per-frame sun direction comes from the camera (set by dataset_readers from
    # the JSON's sun_direction field). Falls back to the model-level
    # `pc.get_sun_dir` (currently hard-coded [0,1,0]) for legacy datasets / viewer
    # paths that don't supply one.
    if hasattr(viewpoint_camera, "sun_dir") and viewpoint_camera.sun_dir is not None:
        L_dir = viewpoint_camera.sun_dir.to(dtype=means3D.dtype, device=means3D.device)
        # Re-normalise defensively; numerical drift from per-frame data is cheap
        # to fix here and avoids surprising the T_light light-space basis below.
        L_dir = L_dir / (torch.linalg.norm(L_dir) + 1e-8)
    else:
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

    tau_sun_per_gauss = mass * line_int_sun
    if precomputed_T_light is not None:
        T_light = precomputed_T_light
    else:
        T_light = compute_T_light(means3D, tau_sun_per_gauss, s, L_dir, grid_res=128)

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
    if override_color is not None:
        colors_precomp = override_color
    else:
        colors_precomp = torch.clamp(Lk, 0.0, 1.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # Physical cloud shading is always precomputed per Gaussian before rasterization,
    # so the legacy separate SH path is intentionally bypassed here.
    rendered_image, radii, depth_image, contribution = rasterizer(
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
        # Per-Gaussian Σ(α·T) over visible pixels — used by the physical
        # densify_and_prune logic to identify negligible-contribution points.
        "contribution": contribution.detach(),
        }
    
    return out
