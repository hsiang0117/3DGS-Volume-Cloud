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


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        # Physical appearance parameters (raw, pre-activation)
        # `_extinction` stores the peak extinction coefficient β_peak (intensive, 1/length),
        # NOT total mass. Mass = β_peak · (2π)^(3/2) · |Σ|^(1/2) is derived in the renderer.
        self._extinction = torch.empty(0)   # (P,1) raw -> softplus(β_peak)
        self._albedo = torch.empty(0)       # (P,3) raw -> sigmoid
        self._g_factor = torch.empty(0)     # (P,1) raw -> tanh

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        # Global (scene-level) sun direction, learned. Stored as raw 3D vector and normalized on use.
        self._sun_dir = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._extinction,
            self._albedo,
            self._g_factor,
            self._scaling,
            self._rotation,
            self._sun_dir,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._extinction,
        self._albedo,
        self._g_factor,
        self._scaling, 
        self._rotation, 
        self._sun_dir,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_sun_dir(self):
        return torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=self._xyz.dtype)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

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
        self._sun_dir = torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=torch.float)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Supplementary trigger signals for densification. xyz gradient alone is too weak
        # in the volumetric-cloud regime (most residual flows into β_peak / scale / albedo).
        self.beta_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Scale signal accumulates only the "growing-scale" direction.
        # In log-scale parameterization a negative grad on _scaling increases s.
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

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

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
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
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

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

    def reset_opacity(self):
        # Kept for training-loop compatibility; opacity is now analytic (from extinction + scale).
        return

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

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

        self.active_sh_degree = self.max_sh_degree

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
        self.beta_gradient_accum = self.beta_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

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

    def densification_postfix(self, tensors_dict, new_tmp_radii):
        """
        Append newly created tensors into the optimizer-managed tensors.
        Expected keys match optimizer group names.
        """
        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)
        self._xyz = optimizable_tensors["xyz"]
        self._extinction = optimizable_tensors["extinction"]
        self._albedo = optimizable_tensors["albedo"]
        self._g_factor = optimizable_tensors["g_factor"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.beta_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

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
        self.beta_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

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
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.beta_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii,
                          max_beta_grad=None, max_scale_grad=None):
        denom = self.denom.clamp(min=1)
        grads = self.xyz_gradient_accum / denom
        grads[grads.isnan()] = 0.0

        # Optional: blend β_peak and scale-growth gradient signals into the trigger.
        # xyz-grad alone is too weak in cloud rendering (residual flows into β_peak/scale).
        # A point is densified if ANY of the three normalized signals crosses its threshold.
        if max_beta_grad is not None and max_beta_grad > 0:
            grads_beta = self.beta_gradient_accum / denom
            grads_beta[grads_beta.isnan()] = 0.0
            grads = torch.maximum(grads, grads_beta * (max_grad / max_beta_grad))
        if max_scale_grad is not None and max_scale_grad > 0:
            grads_scale = self.scale_gradient_accum / denom
            grads_scale[grads_scale.isnan()] = 0.0
            grads = torch.maximum(grads, grads_scale * (max_grad / max_scale_grad))

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # Prune breakdown diagnostics (printed if any source fires).
        n_opacity = int(prune_mask.sum().item())
        n_big_vs = 0
        n_big_ws = 0
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            n_big_vs = int((big_points_vs & ~prune_mask).sum().item())
            n_big_ws = int((big_points_ws & ~prune_mask & ~big_points_vs).sum().item())
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        n_total = int(prune_mask.sum().item())
        if n_total > 0:
            print(f"  [prune] total={n_total} | opacity<{min_opacity}: {n_opacity} | "
                  f"big_vs(px): {n_big_vs} | big_ws(world): {n_big_ws}")
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # |∂L/∂_extinction|: picks up "wants more/less density here" residual pressure.
        if self._extinction.grad is not None:
            self.beta_gradient_accum[update_filter] += self._extinction.grad[update_filter].detach().abs()
        # Only the scale-growing direction (negative raw grad → larger s in log-parameterization).
        if self._scaling.grad is not None:
            grow = (-self._scaling.grad[update_filter]).detach().clamp(min=0).sum(dim=-1, keepdim=True)
            self.scale_gradient_accum[update_filter] += grow
        self.denom[update_filter] += 1
