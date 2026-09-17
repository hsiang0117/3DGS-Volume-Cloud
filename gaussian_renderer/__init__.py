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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, rasterize_lightpass
from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation
from utils.graphics_utils import getProjectionMatrix


def normalized_gaussian_line_integral(scales, dirs_local):
    """
    Center-line integral of a normalized 3D Gaussian along a unit direction.

    For N(x; μ, Σ) with Σ = R diag(s^2) R^T and unit ray direction d_local in the
    Gaussian local frame, the full-line integral through the centre is
        1 / (2π prod(s) sqrt(sum((d_i / s_i)^2))).
    """
    denom = (2.0 * math.pi) * torch.prod(scales, dim=1, keepdim=True) * torch.sqrt(
        torch.sum((dirs_local / scales) ** 2, dim=1, keepdim=True) + 1e-8
    )
    return 1.0 / (denom + 1e-8)


def compute_T_light_voxel(means3D, tau_v_l, scales, v_l, grid_res=128):
    """
    Approximate per-Gaussian sun transmittance via voxel grid (point-scatter).

    Differentiable w.r.t. tau_v_l (hence σ_t and scales), and w.r.t. means3D
    through the grid_sample sampling coordinate (step 4). NOT differentiable
    through the integer deposit index (step 2 hard nearest-voxel scatter), nor
    the light-space basis R_lw / bbox framing, treated as constants.

    Works with an arbitrary sun direction. The grid is a *light-space* frame
    whose third axis is `v_l`, so one 1D prefix sum along that axis gives
    "tau above this voxel along the ray to the sun":
      1. rotate centres into an orthonormal basis (e1, e2, v_l);
      2. scatter optical depth onto a 3D grid in that frame;
      3. exclusive prefix sum along the +sun axis;
      4. trilinear sample at each centre -> T_light = exp(-tau_sun).

    Args:
        means3D:        (P, 3) Gaussian centres
        tau_v_l:        (P, 1) per-Gaussian optical depth along the sun direction
        scales:         (P, 3) Gaussian scales (used only for bbox padding)
        v_l:            (3,)   normalised sun direction (any unit vector)
        grid_res:       int    voxel resolution per axis

    Returns:
        T_light: (P, 1) sun transmittance per Gaussian
    """
    import torch.nn.functional as F
    device = means3D.device
    dtype = means3D.dtype
    P = means3D.shape[0]

    # --- 0. Light-space orthonormal basis (e1, e2, v_l) ----------------
    # Rotation matrix R_lw maps a world-space vector v_w to light-space:
    #     v_L = R_lw @ v_w,  with R_lw = [[e1; e2; v_l]].
    with torch.no_grad():
        s = v_l.to(device=device, dtype=dtype).reshape(3)
        s = s / (torch.linalg.norm(s) + 1e-8)
        helper = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        if abs(float(torch.dot(s, helper).item())) > 0.95:
            helper = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        e1 = torch.linalg.cross(s, helper)
        e1 = e1 / (torch.linalg.norm(e1) + 1e-8)
        e2 = torch.linalg.cross(s, e1)
        e2 = e2 / (torch.linalg.norm(e2) + 1e-8)
        R_lw = torch.stack([e1, e2, s], dim=0)              # (3,3) rows are basis

    # --- 1. Light-space position (differentiable) + bbox framing (detached) -
    means_L = means3D @ R_lw.T                              # (P,3) in light-space
    # bbox: 3-sigma padding so Gaussians don't fall outside the volume.
    with torch.no_grad():
        max_extent = scales.max(dim=1).values.max().item()
        pad = 3.0 * max_extent
        bbox_min = means_L.min(dim=0).values - pad
        bbox_max = means_L.max(dim=0).values + pad
        bbox_size = bbox_max - bbox_min                     # (3,)
        cell_size = bbox_size / grid_res                    # (3,)
        gi = ((means_L - bbox_min) / cell_size).long().clamp(0, grid_res - 1)
        flat_idx = gi[:, 0] * (grid_res * grid_res) + gi[:, 1] * grid_res + gi[:, 2]

    # --- 2. Hard nearest-voxel scatter of per-Gaussian tau (differentiable) --
    volume = torch.zeros(grid_res * grid_res * grid_res, device=device, dtype=dtype)
    volume = volume.scatter_add(0, flat_idx, tau_v_l.squeeze(-1))
    # Indexed [light_x, light_y, light_z=sun_axis]
    volume = volume.view(grid_res, grid_res, grid_res)

    # --- 3. Exclusive prefix sum along the +sun axis (light-Z) -------------
    # tau_above[i,j,k] = Σ_{k'>k} volume[i,j,k']  (cells closer to the sun)
    flipped = torch.flip(volume, [2])
    inclusive_cs = torch.cumsum(flipped, dim=2)
    exclusive_cs = inclusive_cs - flipped
    tau_above = torch.flip(exclusive_cs, [2])

    # --- 4. Trilinear sample at Gaussian centres (in light-space) -----------
    # The grid_sample coordinate carries gradient to means3D (R_lw/bbox constant).
    coords_norm = 2.0 * (means_L - bbox_min) / bbox_size - 1.0       # (P,3)
    # grid_sample expects (D,H,W)=(light_z,light_y,light_x), so feed (z,y,x).
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


def compute_T_light_raster(means3D, tau_v_l, scales, rotations,
                           v_l, image_size=512):
    """
    Per-Gaussian sun transmittance via a light-space rasterization pass.

    Renders the cloud from a distant "sun camera" looking along -v_l with the
    analytic-tau rasterizer. The CUDA kernel records, for every Gaussian, the
    alpha*T-weighted mean of the optical depth accumulated IN FRONT of it over
    all pixels of its light-space footprint (record_front_tau); shadow
    resolution is therefore set by the light image.

    The sun is a DISTANT NARROW-FOV PERSPECTIVE camera (the rasterizer's EWA
    Jacobian is perspective-only).

    Differentiable in tau_v_l ONLY (hence σ_t/scales/rotations through its
    Python-side construction): the CUDA lightpass backward replays the sorted
    buffers and pushes each Gaussian's dL/d(tau_light) onto the taus of all
    occluders in front of it, with the blend weights frozen. Geometry inputs
    (means3D/scales/rotations as splat shapes) are consumed detached: the sun
    camera framing and footprints are treated as constants.

    Returns:
        T_light: (P, 1) sun transmittance per Gaussian.
    """
    device = means3D.device
    dtype = means3D.dtype
    P = means3D.shape[0]

    means3D = means3D.detach()
    scales = scales.detach()
    rotations = rotations.detach()

    with torch.no_grad():
        v_l = v_l.reshape(3)
        v_l = v_l / (torch.linalg.norm(v_l) + 1e-8)

        # --- Sun camera: distant perspective looking along -v_l ----------
        centre = 0.5 * (means3D.min(dim=0).values + means3D.max(dim=0).values)
        # Bounding radius + 3-sigma pad so every splat fits the frustum.
        radius = torch.linalg.norm(means3D - centre, dim=1).max()
        pad = 3.0 * scales.max()
        r_fit = (radius + pad).item()
        D = 60.0 * max(r_fit, 1e-6)

        campos = centre + v_l * D
        # COLMAP/3DGS camera convention: +Z is the viewing direction.
        z_cam = -v_l
        helper = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        if abs(float(torch.dot(z_cam, helper).item())) > 0.95:
            helper = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        x_cam = torch.linalg.cross(helper, z_cam)
        x_cam = x_cam / (torch.linalg.norm(x_cam) + 1e-8)
        y_cam = torch.linalg.cross(z_cam, x_cam)

        # World->view, already transposed the way the rasterizer expects.
        R_c2w = torch.stack([x_cam, y_cam, z_cam], dim=1)        # (3,3) cols
        t_w2c = -(R_c2w.T @ campos)
        world_view_T = torch.zeros(4, 4, device=device, dtype=dtype)
        world_view_T[:3, :3] = R_c2w                              # = R_w2c^T
        world_view_T[3, :3] = t_w2c
        world_view_T[3, 3] = 1.0

        # Frustum sized to the padded cloud at its nearest depth, +5% margin.
        tanfov = 1.05 * r_fit / (D - r_fit)
        fov = 2.0 * math.atan(tanfov)
        znear = D - 1.5 * r_fit
        zfar = D + 1.5 * r_fit
        proj_T = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov) \
            .transpose(0, 1).to(device=device, dtype=dtype)
        full_proj_T = world_view_T @ proj_T

        sun_settings = GaussianRasterizationSettings(
            image_height=image_size,
            image_width=image_size,
            tanfovx=tanfov,
            tanfovy=tanfov,
            bg=torch.zeros(3, device=device, dtype=dtype),
            scale_modifier=1.0,
            viewmatrix=world_view_T,
            projmatrix=full_proj_T,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
            # Keep tau unscaled (no AA rescaling); distance along the sun IS
            # light-space order, so stock centre-depth sort is correct.
            antialiasing=False,
        )

    # Outside no_grad: the lightpass autograd Function carries gradient from
    # tau_light_sum into tau_v_l (σ_t/scales/rotations).
    tau_light_sum, tau_light_wsum, sun_radii = rasterize_lightpass(
        means3D, tau_v_l.view(-1), scales, rotations, sun_settings)

    covered = tau_light_wsum > 1e-8
    tau_light = tau_light_sum / tau_light_wsum.clamp(min=1e-8)
    T_light = torch.exp(-tau_light)
    # wsum==0 with a valid on-screen footprint (radii>0) means every covering
    # pixel early-terminated (T < 1e-4) before reaching this Gaussian, so it is
    # fully shadowed; culled Gaussians stay unlit-neutral at T=1.
    buried = (~covered) & (sun_radii > 0)
    T_light = torch.where(buried, torch.full_like(T_light, 1e-4), T_light)
    T_light = torch.where(covered | buried, T_light, torch.ones_like(T_light))

    return T_light.unsqueeze(-1)


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, override_color = None, precomputed_T_light=None, bg_image=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    Viewer-only hooks (defaults preserve training behaviour):
        override_color:        (P, 3) tensor; if provided, replaces the physical Lk
                               as `colors_precomp` at rasterisation time. Used by
                               the interactive viewer to render diagnostic channels
                               (T_light, σ_t, …) instead of RGB.
        precomputed_T_light:   (P, 1) tensor; if provided, skip the expensive
                               compute_T_light call.
        bg_image:              (3, H, W) linear tensor; if provided, used per-pixel
                               in place of bg_color in the final alpha-over (cloud
                               over sky). Viewer sky backdrop only; None in training.
    """
 
    # Zero tensor used to get pytorch gradients of the 2D (screen-space) means.
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
        bg_image=bg_image,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # Peak extinction coefficient σ_t (intensive, 1/length). Mass derived below.
    sigma_t = pc.get_sigma_t  # (P,1)

    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None

    # Sun irradiance: 4π compensates the 1/(4π) in the normalized HG phase function,
    # so that an isotropic (g=0), unit-albedo medium scatters all incoming light uniformly.
    sun_intensity = 4.0 * math.pi
    L_light = torch.tensor([sun_intensity, sun_intensity, sun_intensity], device="cuda", dtype=means3D.dtype)
    # Per-frame sun direction from the camera (dataset_readers, JSON
    # sun_direction); falls back to model-level `pc.get_v_l` ([0,1,0]).
    if hasattr(viewpoint_camera, "v_l") and viewpoint_camera.v_l is not None:
        v_l = viewpoint_camera.v_l.to(dtype=means3D.dtype, device=means3D.device)
        # Re-normalise: drift would perturb the T_light light-space basis below.
        v_l = v_l / (torch.linalg.norm(v_l) + 1e-8)
    else:
        v_l = pc.get_v_l.to(dtype=means3D.dtype)

    dir_pc = (viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D)
    v_o = dir_pc / (torch.linalg.norm(dir_pc, dim=1, keepdim=True) + 1e-8)

    # Per-Gaussian rotation matrix (world <- local), for projecting directions
    # into the Gaussian's local frame.
    R = build_rotation(pc.get_rotation)  # (P,3,3)
    R_t = R.transpose(1, 2)

    # Local directions
    v_o_local = torch.bmm(R_t, v_o.unsqueeze(-1)).squeeze(-1)            # (P,3)
    v_l_local = torch.matmul(R_t, v_l.view(3, 1)).squeeze(-1)            # (P,3)

    s = pc.get_scaling  # (P,3)
    mass = sigma_t * ((2.0 * math.pi) ** 1.5) * torch.prod(s, dim=1, keepdim=True)

    # Centre-ray line integral; rasterization later multiplies by the projected 2D
    # Gaussian, giving τ(x') ≈ τ_center · G_2D(x').
    line_int_v_o = normalized_gaussian_line_integral(s, v_o_local)
    line_int_v_l = normalized_gaussian_line_integral(s, v_l_local)

    tau_view = mass * line_int_v_o
    tau_precomp = tau_view
    alpha = 1.0 - torch.exp(-tau_view)

    # HG scattering angle: between the INCOMING propagation direction v_i and the
    # OUTGOING (scattered) direction v_o; forward lobe (g>0) peaks at cosθ=+1.
    # v_l points TOWARD the sun, so v_i = −v_l; v_o points toward the camera.
    v_i = -v_l
    cos_theta = torch.clamp((v_o * v_i[None, :]).sum(dim=1, keepdim=True), -1.0, 1.0)

    # Henyey-Greenstein phase function with 1/(4π) normalization.
    g = pc.get_g  # (P,1) in (-0.8, 0.8)
    eps = 1e-6
    inv_4pi = 1.0 / (4.0 * math.pi)

    # Multi-octave scattering approximation (Frostbite / Wrenninge 2015): higher
    # octaves have less energy, less attenuation (T^(b^n)), more isotropic (g·c^n).
    # Per-octave ENERGY weight `w[:, n]` is learnable (softplus, >=0) and only
    # rescales each basis term (HG·T_eff).
    ms_b = 0.5    # transmittance power decay (fixed)
    ms_c = 0.5    # phase isotropization rate (fixed)
    num_octaves = 6
    w = pc.get_w  # (P,6), >=0

    tau_v_l = mass * line_int_v_l
    if precomputed_T_light is not None:
        T_light = precomputed_T_light
    elif getattr(pipe, "tlight_voxel", False):
        # 128^3 voxel cache (light-space rasterization is the default path).
        T_light = compute_T_light_voxel(means3D, tau_v_l, s, v_l, grid_res=128)
    else:
        T_light = compute_T_light_raster(
            means3D, tau_v_l, s, pc.get_rotation,
            v_l, image_size=int(getattr(pipe, "tlight_raster_res", 512)))

    scatter_sum = torch.zeros_like(mass)  # (P,1)
    for n in range(num_octaves):
        energy = w[:, n:n+1]                      # (P,1) learnable per-Gaussian
        g_eff = g * (ms_c ** n)
        T_eff = torch.pow(T_light.clamp(min=1e-8), ms_b ** n)
        denom_hg = torch.pow(1.0 + g_eff * g_eff - 2.0 * g_eff * cos_theta, 1.5) + eps
        HG_n = inv_4pi * (1.0 - g_eff * g_eff) / denom_hg
        scatter_sum = scatter_sum + energy * T_eff * HG_n

    omega = pc.get_omega  # (P,3)

    Lk = omega * L_light[None, :] * scatter_sum
    # --- Stage 2 environment lighting (frozen geometry) ---
    # Modulate the sun term by the atmospheric transmittance T_sun(v_l) (RGB ≤1,
    # low-sun dimming/reddening) and add the sky in-scatter fill ω·Σ E_lm(v_l)·V_lm.
    # T_sun/E_lm are GLOBAL functions of v_l (EnvNet); no per-Gaussian colour DOF.
    # Linear space, before the output tonemap. No-op unless --stage2.
    if getattr(pipe, "env_lighting", False) and override_color is None:
        t_sun, fill = pc.apply_env(v_l)
        if t_sun is not None:
            Lk = t_sun[None, :] * Lk + fill
    # Output tonemap gateway: tonemap_learnable -> pc.apply_tonemap (4 learned
    # coeffs, e pinned); tonemap_aces -> fixed Narkowicz constants. Either means
    # HDR-linear shading, so the per-Gaussian clamp is lifted to 16.
    tonemap_learn = bool(getattr(pipe, "tonemap_learnable", False)) and pc.get_tonemap_coeffs is not None
    tonemap_on = tonemap_learn or bool(getattr(pipe, "tonemap_aces", False))
    if override_color is not None:
        colors_precomp = override_color
    elif tonemap_on:
        # HDR mode: the tonemap soft-clips at the image level; 16 guards fp blowups.
        colors_precomp = torch.clamp(Lk, 0.0, 16.0)
    else:
        colors_precomp = torch.clamp(Lk, 0.0, 1.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # Physical cloud shading is always precomputed per Gaussian before rasterization,
    # so the SH path is bypassed here.
    rendered_image, radii, depth_image, contribution, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = alpha,
        tau_precomp = tau_precomp,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if tonemap_on and override_color is None:
        # Both branches use the Narkowicz rational form; the learnable branch
        # carries gradient into the tonemap coeffs (its own optimizer).
        if tonemap_learn:
            rendered_image = pc.apply_tonemap(rendered_image)
        else:
            # ACES filmic approximation (Narkowicz 2015). Differentiable; x>=0
            # keeps it monotonic.
            x = rendered_image.clamp(min=0.0)
            rendered_image = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
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
