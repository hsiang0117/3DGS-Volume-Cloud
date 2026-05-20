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
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    @staticmethod
    def _softplus_inverse(y, eps=1e-8):
        y = torch.clamp(y, min=eps)
        return torch.log(torch.expm1(y) + eps)

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.rotation_activation = torch.nn.functional.normalize
        self.extinction_activation = lambda x: torch.clamp(torch.nn.functional.softplus(x), max=5.0)
        self.albedo_activation = torch.sigmoid
        self.g_factor_activation = lambda x: 0.8 * torch.tanh(x)


    def __init__(self, optimizer_type="default"):
        self.optimizer_type = optimizer_type
        self._xyz = torch.empty(0)
        # Physical appearance parameters (raw, pre-activation)
        # `_extinction` stores the peak extinction coefficient β_peak (intensive, 1/length),
        # NOT total mass. Mass = β_peak · (2π)^(3/2) · |Σ|^(1/2) is derived in the renderer.
        self._extinction = torch.empty(0)   # (P,1) raw -> softplus(β_peak)
        self._albedo = torch.empty(0)       # (P,3) raw -> sigmoid
        self._g_factor = torch.empty(0)     # (P,1) raw -> tanh

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.scale_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
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
    def get_opacity(self):
        """
        Compatibility proxy for pruning / logging: interpret opacity as the
        analytic center-line transmittance of a normalized Gaussian using an
        isotropic proxy based on the geometric mean scale.
        """
        if self._xyz.numel() == 0:
            return torch.empty(0, device="cuda")
        beta_peak = self.get_extinction
        gscale = torch.pow(torch.prod(self.get_scaling, dim=1, keepdim=True) + 1e-8, 1.0 / 3.0)
        tau_center = beta_peak * (2.0 * math.pi) ** 0.5 * gscale
        return 1.0 - torch.exp(-tau_center)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

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
        actual_scales = torch.exp(scales)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Physical parameter initialization (raw / pre-activation)
        # `_extinction` stores the peak extinction coefficient β_peak directly.
        # β_peak ≈ 0.1 gives initial τ_center = β_peak·√(2π)·s ≈ 0.25·s at unit scale.
        beta_peak_init = torch.full((P, 1), 0.1, dtype=torch.float, device="cuda")
        extinction_raw = self._softplus_inverse(beta_peak_init)
        albedo_raw = inverse_sigmoid(torch.full((P, 3), 0.8, dtype=torch.float, device="cuda"))
        g_factor_raw = torch.atanh(torch.full((P, 1), 0.7, dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._extinction = nn.Parameter(extinction_raw.requires_grad_(True))
        self._albedo = nn.Parameter(albedo_raw.requires_grad_(True))
        self._g_factor = nn.Parameter(g_factor_raw.requires_grad_(True))
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
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
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

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        _sched_map = {
            "xyz": self.xyz_scheduler_args,
            "scaling": self.scaling_scheduler_args,
            "extinction": self.extinction_scheduler_args,
            "albedo": self.albedo_scheduler_args,
            "g_factor": self.g_factor_scheduler_args,
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
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, extinction, albedo, g_factor, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

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
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

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
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        d = {
            "xyz": new_xyz,
            "extinction": new_extinction,
            "albedo": new_albedo,
            "g_factor": new_g_factor,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._extinction = optimizable_tensors["extinction"]
        self._albedo = optimizable_tensors["albedo"]
        self._g_factor = optimizable_tensors["g_factor"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Append zero stats for new children; preserve existing points' stats so
        # the prune predicate (visible_enough = denom ≥ 5) keeps working inside
        # the densify phase. Resetting wholesale here was the reason aniso /
        # contribution prune never fired before iter 15000.
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

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # β_peak is intensive: clone inherits it as-is (no halving of the parent).
        new_xyz = self._xyz[selected_pts_mask]
        new_extinction = self._extinction[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_g_factor = self._g_factor[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        d = {
            "xyz": new_xyz,
            "extinction": new_extinction,
            "albedo": new_albedo,
            "g_factor": new_g_factor,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._extinction = optimizable_tensors["extinction"]
        self._albedo = optimizable_tensors["albedo"]
        self._g_factor = optimizable_tensors["g_factor"]
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

        Density growth: keep stock xyz/scale-grad-driven clone+split (well
        understood, stable; not the real bottleneck).

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

        # 2. Contribution + aniso based prune (replaces opacity threshold)
        if iteration >= opt.prune_warmup:
            self._prune_by_contribution_and_aniso(opt)

        # Decay grace counter; accumulator reset is handled separately by
        # tick_post_densify_maintenance() so it stays alive post-densify.
        with torch.no_grad():
            self.prune_grace = (self.prune_grace - opt.densification_interval).clamp(min=0)

        self.tmp_radii = None
        torch.cuda.empty_cache()

    def _prune_by_contribution_and_aniso(self, opt):
        """Two-channel prune for the physical strategy.

        Channel A (contribution): a Gaussian that has been visible enough
        frames yet projects almost zero light onto valid pixels is dead
        weight regardless of its parameters.

        Channel B (aniso): a Gaussian whose s_max/s_min exceeds
        `opt.prune_aniso_ratio` is degenerate as a volumetric primitive —
        keeping it just feeds the depth-sort popping the k_sigma clamp is
        trying to suppress. Reclaim its budget so resurrect can place a
        well-shaped point elsewhere.

        Both channels require `visible_enough` and `grace_expired` so we
        don't pop newborn / resurrected points.
        """
        if self.get_xyz.shape[0] == 0:
            return 0
        visible_enough = self.contribution_denom >= opt.prune_min_visible_frames
        grace_expired = self.prune_grace == 0
        mean_contrib = self.get_mean_contribution()
        below_thresh = mean_contrib < opt.contribution_threshold

        prune_aniso_ratio = getattr(opt, "prune_aniso_ratio", -1.0)
        if prune_aniso_ratio > 0:
            s = self.get_scaling
            ratio = s.max(dim=1).values / s.min(dim=1).values.clamp(min=1e-6)
            aniso_bad = ratio > prune_aniso_ratio
        else:
            aniso_bad = torch.zeros_like(below_thresh)

        prune_mask = visible_enough & grace_expired & (below_thresh | aniso_bad)
        n = int(prune_mask.sum().item())
        if n > 0:
            self.prune_points(prune_mask)
        return n

    def tick_post_densify_maintenance(self, opt, iteration):
        """Iteration-driven housekeeping that must keep running even after
        densify_until_iter:

          - β_peak resurrect of bottom `opt.resurrect_fraction` Gaussians
            every `opt.resurrect_interval` iterations.
          - Periodic reset of the contribution accumulators so the running
            mean tracks the current model state, not stale early-training
            statistics. Reset every `opt.contribution_reset_interval` iters.
        """
        if iteration <= 0:
            return
        # Order matters: resurrect → prune → reset. The prune predicate uses
        # `contribution_denom >= prune_min_visible_frames` as a gate, so if we
        # zeroed the accumulator first the gate would mask every point and
        # nothing would ever be reclaimed (silent failure observed as
        # n_points frozen + aniso p99 unbounded after densify_until_iter).
        # 1. Resurrect schedule (independent of densify_until_iter)
        if (
            opt.resurrect_interval > 0
            and iteration % opt.resurrect_interval == 0
        ):
            self._resurrect_low_contribution(opt.resurrect_fraction)

        # 2. Aniso/contribution prune post-densify. Without this, long
        # ellipsoids accumulating in the second half of training never get
        # reclaimed (densify_until_iter has stopped the regular prune path),
        # which we've seen drive viewer popping back up.
        prune_iv = getattr(opt, "post_densify_prune_interval", 0)
        if prune_iv > 0 and iteration % prune_iv == 0 and iteration >= opt.prune_warmup:
            self._prune_by_contribution_and_aniso(opt)
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
