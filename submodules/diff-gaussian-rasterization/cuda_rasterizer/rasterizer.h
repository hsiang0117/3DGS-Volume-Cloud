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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const float* bg_image,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* tau_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* depth,
			float* gauss_contribution,
			float* tau_front_sum,
			float* tau_front_wsum,
			bool antialiasing,
			// Light-pass measurement probes (see forward.cu / forward.h).
			// Placed after antialiasing so no defaulted parameter follows them.
			float* tau_front_TG_sum,
			float* tau_front_G_sum,
			bool light_tau_filter,
	float light_filter_variance,
			int* radii = nullptr,
			bool debug = false);

		// Backward of the record_front_tau light pass: dL/d(tau_front_sum) ->
		// dL/d(con_o.w), the packed scalar, of occluders; replays the saved
		// buffers. Equals dL/d(tau_precomp) only while antialiasing is off.
		static void lightpassBackward(
			const int P, const int R,
			const int width, const int height,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* grad_tau_front_sum,
			float* dL_dtau,
			bool debug = false);

		// Complete continuous VJP with framing, sorting and discrete gates fixed.
		static void lightpassBackwardFull(
			int P, int R, int width, int height,
			const float* means, const float* tau, const float* scales, const float* rotations,
			const int* radii, const float* view, const float* proj, const float* campos,
			float tan_fovx, float tan_fovy,
			char* geom_buffer, char* binning_buffer, char* image_buffer,
			const float* grad_sum, const float* grad_wsum, const float* grad_TG, const float* grad_G,
			float* dmean2D, float* dconic, float* dtau, float* dmeans,
			float* dcov, float* dscale, float* drot, bool light_tau_filter, float light_filter_variance, bool debug);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* tau_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_invdepths,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dtau,
			float* dL_dcolor,
			float* dL_dinvdepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool antialiasing,
			bool debug);
	};
};

#endif
