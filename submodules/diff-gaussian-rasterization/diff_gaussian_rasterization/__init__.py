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

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
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
            raster_settings.k_sigma,
            raster_settings.record_front_tau,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, contribution, tau_front_sum, tau_front_wsum = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, tau_precomp, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths, contribution, tau_front_sum, tau_front_wsum

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth, _grad_contribution, _grad_tau_front_sum, _grad_tau_front_wsum):

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

def rasterize_lightpass(means3D, tau_precomp, scales, rotations, raster_settings):
    """Light-space shadow pass with a differentiable tau path.

    Runs the analytic-tau rasterizer from a sun camera with record_front_tau
    and returns (tau_front_sum, tau_front_wsum). Gradient flows from
    tau_front_sum back into tau_precomp ONLY (through the dedicated CUDA
    lightpass backward, which replays the saved sorted buffers and
    distributes each Gaussian's incoming gradient onto the taus of all
    occluders in front of it, blend weights frozen). Geometry inputs
    (means3D/scales/rotations) receive no gradient from this pass — they are
    consumed pre-detached by the caller; tau_precomp itself still carries
    their contribution through its own Python-side construction.
    """
    return _RasterizeLightpass.apply(means3D, tau_precomp, scales, rotations, raster_settings)


class _RasterizeLightpass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, tau_precomp, scales, rotations, raster_settings):
        device = means3D.device
        dtype = means3D.dtype
        P = means3D.shape[0]
        empty = torch.empty(0, device=device, dtype=dtype)
        dummy_colors = torch.zeros(P, 3, device=device, dtype=dtype)
        # Ignored by the analytic-tau kernel branch, required by the binding.
        dummy_opacity = torch.zeros(P, 1, device=device, dtype=dtype)

        args = (
            raster_settings.bg,
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
            raster_settings.k_sigma,
            True,   # record_front_tau
            raster_settings.debug,
        )
        (num_rendered, _, radii, geomBuffer, binningBuffer, imgBuffer,
         _, _, tau_front_sum, tau_front_wsum) = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(tau_precomp, geomBuffer, binningBuffer, imgBuffer)
        ctx.mark_non_differentiable(tau_front_wsum)
        return tau_front_sum, tau_front_wsum, radii

    @staticmethod
    def backward(ctx, grad_tau_front_sum, _grad_wsum, _grad_radii):
        tau_precomp, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors
        raster_settings = ctx.raster_settings
        dL_dtau = _C.rasterize_lightpass_backward(
            tau_precomp,
            grad_tau_front_sum.contiguous(),
            raster_settings.image_height,
            raster_settings.image_width,
            geomBuffer,
            ctx.num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )
        return None, dL_dtau, None, None, None


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
    # k_sigma controls how far the per-tile max-response depth t* may shift
    # from the centre depth, in units of σ along the view ray. ≤0 disables
    # the shift (stock 3DGS centre-depth sort), which is the current default:
    # the per-tile sort produced blocky tile-boundary artefacts, and the aniso
    # prune/penalty controls popping on its own. Set >0 to re-enable the
    # per-tile shift (no rebuild needed; the CUDA path is retained).
    k_sigma : float = 0.0
    # When True (and tau_precomp is provided), the render kernel records, per
    # Gaussian, the alpha*T-weighted mean of the analytic optical depth
    # accumulated IN FRONT of it along this camera's rays. Used by the
    # light-space shadow pass: T_light = exp(-tau_front_sum/tau_front_wsum).
    # Forward-only (gradients ignored).
    record_front_tau : bool = False

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

