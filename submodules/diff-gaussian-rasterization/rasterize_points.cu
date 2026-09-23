/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& bg_image,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& tau_precomp,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool record_front_tau,
	const bool light_tau_filter,
	const float light_filter_variance,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  TORCH_CHECK(std::isfinite(light_filter_variance) && light_filter_variance >= 0.0f,
              "light_filter_variance must be finite and nonnegative");
  TORCH_CHECK(record_front_tau || light_filter_variance == 0.3f,
              "Only the light pass supports configurable dilation");
  TORCH_CHECK(!light_tau_filter || (record_front_tau && (tau_precomp.numel() > 0 || means3D.size(0) == 0)),
              "light_tau_filter requires an analytic-tau light pass");
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  // Per-Gaussian Σ(α·T) over all visible pixels, for densify/prune in train.py.
  torch::Tensor gauss_contribution = torch::zeros({P}, float_opts);

  // Light-space shadow pass buffers: per-Gaussian alpha*T-weighted sum of the
  // optical depth in front of it, plus the weight sum. Size P only when
  // record_front_tau is set (requires tau_precomp); T_light = exp(-sum/wsum).
  const bool do_front_tau = record_front_tau && tau_precomp.numel() > 0;
  torch::Tensor tau_front_sum = torch::zeros({do_front_tau ? P : 0}, float_opts);
  torch::Tensor tau_front_wsum = torch::zeros({do_front_tau ? P : 0}, float_opts);

  // Light-pass measurement probes (see forward.cu): per-Gaussian sums of T*G
  // and G over pixels that reached the splat alive. Host side turns them into
  // the measured front transmittance for splats too faint for tau_front_wsum,
  // and G_sum == 0 (with a nonzero radius) into "fully occluded from the sun".
  torch::Tensor tau_front_TG_sum = torch::zeros({do_front_tau ? P : 0}, float_opts);
  torch::Tensor tau_front_G_sum = torch::zeros({do_front_tau ? P : 0}, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		bg_image.numel() > 0 ? bg_image.contiguous().data<float>() : nullptr,
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		tau_precomp.numel() > 0 ? tau_precomp.contiguous().data<float>() : nullptr,
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_invdepthptr,
		gauss_contribution.contiguous().data<float>(),
		do_front_tau ? tau_front_sum.contiguous().data<float>() : nullptr,
		do_front_tau ? tau_front_wsum.contiguous().data<float>() : nullptr,
		antialiasing,
		do_front_tau ? tau_front_TG_sum.contiguous().data<float>() : nullptr,
		do_front_tau ? tau_front_G_sum.contiguous().data<float>() : nullptr,
		light_tau_filter,
		light_filter_variance,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth, gauss_contribution, tau_front_sum, tau_front_wsum, tau_front_TG_sum, tau_front_G_sum);
}

torch::Tensor
RasterizeLightpassBackwardCUDA(
	const torch::Tensor& tau_precomp,
	const torch::Tensor& grad_tau_front_sum,
	const int image_height,
	const int image_width,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug)
{
  const int P = tau_precomp.size(0);
  torch::Tensor dL_dtau = torch::zeros_like(tau_precomp);
  if (P != 0 && R != 0)
  {
	CudaRasterizer::Rasterizer::lightpassBackward(P, R,
	  image_width, image_height,
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  grad_tau_front_sum.contiguous().data<float>(),
	  dL_dtau.contiguous().data<float>(),
	  debug);
  }
  return dL_dtau;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& tau_precomp,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtau = torch::zeros({0, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(tau_precomp.numel() != 0)
  {
	dL_dtau = torch::zeros_like(tau_precomp);
  }

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  tau_precomp.numel() > 0 ? tau_precomp.contiguous().data<float>() : nullptr,
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  tau_precomp.numel() > 0 ? dL_dtau.contiguous().data<float>() : nullptr,
	  dL_dcolors.contiguous().data<float>(),
	  dL_dinvdepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dtau, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeLightpassFullBackwardCUDA(
    const torch::Tensor& means, const torch::Tensor& tau,
    const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& radii,
    const torch::Tensor& view, const torch::Tensor& proj, const torch::Tensor& campos,
    float tan_fovx, float tan_fovy, int height, int width,
    const torch::Tensor& gs, const torch::Tensor& gw, const torch::Tensor& gtg, const torch::Tensor& gg,
    const torch::Tensor& geom, int R, const torch::Tensor& binning, const torch::Tensor& image,
    bool light_tau_filter, float light_filter_variance, bool debug)
{
    const int P = means.size(0);
    auto opts = means.options();
    auto dm = torch::zeros_like(means), dt = torch::zeros_like(tau);
    auto ds = torch::zeros_like(scales), dr = torch::zeros_like(rotations);
    if (P != 0 && R != 0)
    {
        auto dm2 = torch::zeros({P, 3}, opts);
        auto dq = torch::zeros({P, 4}, opts);
        auto dcov = torch::zeros({P, 6}, opts);
        CudaRasterizer::Rasterizer::lightpassBackwardFull(P, R, width, height,
            means.contiguous().data_ptr<float>(), tau.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(), rotations.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>(), view.contiguous().data_ptr<float>(),
            proj.contiguous().data_ptr<float>(), campos.contiguous().data_ptr<float>(), tan_fovx, tan_fovy,
            reinterpret_cast<char*>(geom.contiguous().data_ptr()),
            reinterpret_cast<char*>(binning.contiguous().data_ptr()),
            reinterpret_cast<char*>(image.contiguous().data_ptr()),
            gs.contiguous().data_ptr<float>(), gw.contiguous().data_ptr<float>(),
            gtg.contiguous().data_ptr<float>(), gg.contiguous().data_ptr<float>(),
            dm2.data_ptr<float>(), dq.data_ptr<float>(), dt.data_ptr<float>(), dm.data_ptr<float>(),
            dcov.data_ptr<float>(), ds.data_ptr<float>(), dr.data_ptr<float>(), light_tau_filter, light_filter_variance, debug);
    }
    return std::make_tuple(dm, dt, ds, dr);
}
