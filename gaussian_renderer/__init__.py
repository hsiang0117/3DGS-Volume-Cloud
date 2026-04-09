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


def compute_T_light(means3D, sigma_t, scales, sun_dir, grid_res=128):
    """
    Approximate per-Gaussian sun transmittance via voxel grid.
    Fully differentiable w.r.t. sigma_t (and means3D through grid_sample).

    1. Build extinction field on a 3D grid (point-deposit at Gaussian centers).
    2. Exclusive prefix sum along sun direction → cumulative optical depth above each cell.
    3. Trilinear sample at each Gaussian center → T_light = exp(-tau_sun).

    Args:
        means3D: (P, 3)  Gaussian centres
        sigma_t: (P, 1)  extinction coefficient (activated)
        scales:  (P, 3)  Gaussian scales
        sun_dir: (3,)    normalised sun direction  (currently assumed [0,1,0])
        grid_res: int    voxel resolution per axis

    Returns:
        T_light: (P, 1)  sun transmittance per Gaussian
    """
    device = means3D.device
    dtype = means3D.dtype
    P = means3D.shape[0]

    # --- 1. Bounding box with 3-sigma padding (detach to keep grid fixed) ---
    with torch.no_grad():
        max_extent = scales.max(dim=1).values.max().item()
        pad = 3.0 * max_extent
        bbox_min = means3D.min(dim=0).values - pad
        bbox_max = means3D.max(dim=0).values + pad
        bbox_size = bbox_max - bbox_min                    # (3,)
        cell_size = bbox_size / grid_res                   # (3,)
        # Grid indices (not differentiable, integer)
        gi = ((means3D - bbox_min) / cell_size).long().clamp(0, grid_res - 1)  # (P,3)
        flat_idx = gi[:, 0] * (grid_res * grid_res) + gi[:, 1] * grid_res + gi[:, 2]

    # --- 2. Scatter sigma_t into voxel grid (differentiable w.r.t. sigma_t) ---
    volume = torch.zeros(grid_res * grid_res * grid_res, device=device, dtype=dtype)
    volume = volume.scatter_add(0, flat_idx, sigma_t.squeeze(-1))  # out-of-place → differentiable
    volume = volume.view(grid_res, grid_res, grid_res)  # indexed [x, y, z]

    # --- 3. Exclusive prefix sum along Y (sun direction = +Y) ---
    # tau_above[x,y,z] = Σ_{y'>y} volume[x,y',z] · cell_size_y
    flipped = torch.flip(volume, [1])
    inclusive_cs = torch.cumsum(flipped, dim=1)
    exclusive_cs = inclusive_cs - flipped               # exclude current cell
    tau_above = torch.flip(exclusive_cs, [1]) * cell_size[1]

    # --- 4. Trilinear sample at Gaussian centers ---
    import torch.nn.functional as F
    with torch.no_grad():
        coords_norm = 2.0 * (means3D - bbox_min) / bbox_size - 1.0  # (P,3)
        grid_pts = torch.stack([coords_norm[:, 2],
                                coords_norm[:, 1],
                                coords_norm[:, 0]], dim=-1)          # (P,3)
        grid_pts = grid_pts.view(1, 1, 1, P, 3)                     # (1,1,1,P,3)

    tau_sun = F.grid_sample(
        tau_above.unsqueeze(0).unsqueeze(0),                     # (1,1,X,Y,Z)
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
    # Analytic optical thickness τ_k = σ_t,k * Δt, with Δt approximated by the ellipsoid
    # extent along the integration direction (view / sun). This makes transmittance direction-aware.
    sigma_t = pc.get_extinction  # (P,1)

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
    # Fixed environment
    L_sun = torch.tensor([1.0, 1.0, 1.0], device="cuda", dtype=means3D.dtype)
    # Sun direction in world space (normalized): learned global parameter.
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
    l_local = torch.matmul(R_t, L_dir.view(3, 1)).squeeze(-1)           # (P,3) via broadcast matmul

    s = pc.get_scaling * scaling_modifier  # (P,3)

    # Analytic path length: for 3D Gaussian with Σ = R·diag(s²)·Rᵀ, the ray integral
    # ∫G(o+td)dt = √(2π / (d^T Σ^{-1} d)) · G_2D(pixel).  CUDA already evaluates
    # G_2D per pixel, so we precompute h = √(2π) / ||d_local / s||.
    dt_view = math.sqrt(2.0 * math.pi) / torch.sqrt(torch.sum((v_local / s) ** 2, dim=1, keepdim=True) + 1e-8)
    dt_sun  = math.sqrt(2.0 * math.pi) / torch.sqrt(torch.sum((l_local / s) ** 2, dim=1, keepdim=True) + 1e-8)

    tau_view = sigma_t * dt_view
    tau_precomp = tau_view
    opacity = 1.0 - torch.exp(-tau_view)

    # cos(theta) between view dir and light dir
    cos_theta = torch.clamp((v * L_dir[None, :]).sum(dim=1, keepdim=True), -1.0, 1.0)

    # Henyey-Greenstein phase function — no 1/(4π) normalization.
    g = pc.get_g_factor  # (P,1) in (-1,1)
    eps = 1e-6

    # Sun transmittance: how much light reaches each Gaussian from the sun,
    # computed via differentiable voxel-grid prefix sum along the sun direction.
    T_light = compute_T_light(means3D, sigma_t, s, L_dir, grid_res=128)

    # Multi-octave scattering approximation (Frostbite / Wrenninge 2015).
    # Simulates multiple scattering bounces using the same physical parameters.
    # Higher octaves: less energy (a^n), less attenuation (T^(b^n)), more isotropic (g·c^n).
    # NOTE: no 1/(4π) — single directional light, energy scale absorbed by rho.
    ms_a = 0.5    # energy attenuation per bounce
    ms_b = 0.5    # transmittance power decay
    ms_c = 0.5    # phase isotropization rate
    num_octaves = 6

    scatter_sum = torch.zeros_like(sigma_t)  # (P,1)
    for n in range(num_octaves):
        energy   = ms_a ** n
        g_eff    = g * (ms_c ** n)
        T_eff    = torch.pow(T_light.clamp(min=1e-8), ms_b ** n)
        denom_hg = torch.pow(1.0 + g_eff * g_eff - 2.0 * g_eff * cos_theta, 1.5) + eps
        HG_n     = (1.0 - g_eff * g_eff) / denom_hg
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
