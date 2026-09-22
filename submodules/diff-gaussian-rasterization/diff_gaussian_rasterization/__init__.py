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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    tau_precomp,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        tau_precomp,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        tau_precomp,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Per-pixel background (sky backdrop) or an empty tensor for the
        # constant-bg path. Must match the C++ arg order (right after `bg`).
        bg_image = raster_settings.bg_image
        if bg_image is None:
            bg_image = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            bg_image,
            means3D,
            colors_precomp,
            opacities,
            tau_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.record_front_tau,
            raster_settings.light_tau_filter,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer. The trailing light-pass probes are empty
        # on the camera pass (record_front_tau is off), so swallow them with the
        # rest of the tail rather than pinning the arity here.
        (num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer,
         invdepths, contribution, tau_light_sum, tau_light_wsum, *_rest) = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, tau_precomp, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths, contribution, tau_light_sum, tau_light_wsum

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth, _grad_contribution, _grad_tau_light_sum, _grad_tau_light_wsum):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, tau_precomp, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                tau_precomp,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_tau_precomp, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_tau_precomp,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

def rasterize_lightpass(means3D, tau_precomp, scales, rotations, raster_settings, full_grad=False):
    """Light-space shadow pass with a differentiable tau path.

    Runs the analytic-tau rasterizer from a sun camera with record_front_tau
    and returns
    (tau_light_sum, tau_light_wsum, radii, tau_front_TG_sum, tau_front_G_sum).
    Callers turn the first two into T_light = exp(-sum/wsum); `radii` are the
    per-Gaussian light-space screen radii. TG_sum/G_sum accumulate T*G and G
    over the pixels that reached each splat alive (before the alpha gate), so
    TG/G is the measured front transmittance for splats too faint for wsum,
    and G_sum == 0 with a nonzero radius means every covering pixel was
    terminated by an occluder in front. By default, only tau_precomp receives
    the partial front-tau derivative with blend weights frozen. full_grad=True
    differentiates S/W, TG/G, blend weights, tau, and projected geometry,
    holding framing, sorting, discrete support, and classification fixed.
    """
    if raster_settings.antialiasing:
        raise ValueError("Light-pass experiments require camera AA disabled")
    return _RasterizeLightpass.apply(means3D, tau_precomp, scales, rotations, raster_settings, full_grad)


class _RasterizeLightpass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, tau_precomp, scales, rotations, raster_settings, full_grad):
        device = means3D.device
        dtype = means3D.dtype
        P = means3D.shape[0]
        empty = torch.empty(0, device=device, dtype=dtype)
        dummy_colors = torch.zeros(P, 3, device=device, dtype=dtype)
        # Ignored by the analytic-tau kernel branch, required by the binding.
        dummy_opacity = torch.zeros(P, 1, device=device, dtype=dtype)

        args = (
            raster_settings.bg,
            empty,  # bg_image: lightpass uses the constant-bg path
            means3D,
            dummy_colors,
            dummy_opacity,
            tau_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            empty,  # cov3D_precomp
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            empty,  # sh
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            True,   # record_front_tau
            raster_settings.light_tau_filter,
            raster_settings.debug,
        )
        (num_rendered, _, radii, geomBuffer, binningBuffer, imgBuffer,
         _, _, tau_light_sum, tau_light_wsum,
         tau_front_TG_sum, tau_front_G_sum) = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.full_grad = bool(full_grad)
        ctx.save_for_backward(tau_precomp, geomBuffer, binningBuffer, imgBuffer, means3D, scales, rotations, radii)
        ctx.mark_non_differentiable(radii)
        if not full_grad:
            ctx.mark_non_differentiable(tau_light_wsum, tau_front_TG_sum, tau_front_G_sum)
        return tau_light_sum, tau_light_wsum, radii, tau_front_TG_sum, tau_front_G_sum

    @staticmethod
    def backward(ctx, grad_tau_light_sum, _grad_wsum, _grad_radii,
                 _grad_TG_sum, _grad_G_sum):
        tau_precomp, geomBuffer, binningBuffer, imgBuffer, means3D, scales, rotations, radii = ctx.saved_tensors
        raster_settings = ctx.raster_settings
        if ctx.full_grad:
            gradients = [grad_tau_light_sum, _grad_wsum, _grad_TG_sum, _grad_G_sum]
            gradients = [torch.zeros_like(tau_precomp) if g is None else g.contiguous() for g in gradients]
            gm, gt, gs, gr = _C.rasterize_lightpass_backward_full(
                means3D, tau_precomp, scales, rotations, radii,
                raster_settings.viewmatrix, raster_settings.projmatrix, raster_settings.campos,
                raster_settings.tanfovx, raster_settings.tanfovy,
                raster_settings.image_height, raster_settings.image_width,
                *gradients, geomBuffer, ctx.num_rendered, binningBuffer, imgBuffer,
                raster_settings.light_tau_filter, raster_settings.debug)
            return gm, gt, gs, gr, None, None
        dL_dtau = _C.rasterize_lightpass_backward(
            tau_precomp,
            grad_tau_light_sum.contiguous(),
            raster_settings.image_height,
            raster_settings.image_width,
            geomBuffer,
            ctx.num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )
        return None, dL_dtau, None, None, None, None


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    # When True (and tau_precomp is provided), the render kernel records, per
    # Gaussian, the alpha*T-weighted mean analytic optical depth in front of it
    # along this camera's rays: T_light = exp(-tau_front_sum/tau_front_wsum).
    record_front_tau : bool = False
    # Optional per-pixel background image (CHANNELS x H x W, planar, linear),
    # used in place of the constant `bg` in the final alpha-over. The viewer's
    # sky backdrop sets this so the rasterizer composites cloud-over-sky in one
    # pass. None -> constant `bg` (training path, unchanged). Forward-only.
    bg_image : torch.Tensor = None
    light_tau_filter : bool = False

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, tau_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if colors_precomp is None:
            colors_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        if tau_precomp is None:
            tau_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        if scales is None:
            scales = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if rotations is None:
            rotations = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if cov3D_precomp is None:
            cov3D_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            tau_precomp,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

