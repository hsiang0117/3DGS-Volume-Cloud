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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        # Per-tile max-response sort: how far t* may deviate from centre depth,
        # in units of σ along the view ray. ≤0 reverts to stock 3DGS centre
        # sort. 1.5 = current default (no tile artefacts, long-axis popping
        # fixed).
        self.k_sigma = 1.5
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.extiction_lr = 0.025
        self.g_factor_lr = 0.0025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_scale = 0.1
        # Anisotropy penalty (log-ratio form): zero below `aniso_ratio_max`,
        # quadratic in log-ratio above. Cloud is an isotropic medium; long
        # ellipsoids cause depth-sort popping.
        #
        # Keep λ small. Empirically:
        #   λ=0.05 → PSNR collapses to ~25 (regularizer dominates fit gradient)
        #   λ=0.001 → PSNR ~43, aniso largely unbounded
        # Bounding aniso meaningfully and keeping PSNR high are in tension —
        # cloud structure (wisps, layers) genuinely benefits from elongated
        # Gaussians at current capacity. For aggressive aniso bounding, prefer
        # split-on-densify (`densify_split_aniso_max`) over loss regularisation.
        self.lambda_aniso = 0.005
        self.aniso_ratio_max = 5.0
        # Disable aniso reg after densify ends, so post-densify L_vol pressure
        # doesn't combine with the regularizer to uniformly shrink the cloud.
        self.aniso_until_iter = 15_000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 1e-4
        self.densify_scale_grad_threshold = 1e-6
        self.optimizer_type = "default"
        # Adaptive density threshold: take top `densify_top_frac` of grads each
        # round so growth doesn't stall when grads decay late in training.
        self.densify_adaptive = True
        self.densify_top_frac = 0.005          # top 0.5%
        self.densify_grad_min = 5e-5           # absolute floor
        # A Gaussian is pruned iff mean Σ(α·T) over visible frames falls below
        # this threshold. 1e-4 ≈ 0.01% of one fully-opaque pixel, very lenient
        # — main role is to remove "ghost" Gaussians, not active ones.
        self.contribution_threshold = 1e-4
        self.prune_min_visible_frames = 5      # require at least 5 visible frames before judging
        self.resurrect_interval = 3000         # every N iters, reset bottom β_peak
        self.resurrect_fraction = 0.05         # 5% of points
        # How often to clear the contribution accumulator so the running mean
        # tracks current model state. Independent of densify_until_iter; keeps
        # working post-densify. Set ≤0 to never reset (not recommended).
        self.contribution_reset_interval = 1000
        # Aniso side-channel for prune: if a Gaussian's s_max/s_min exceeds
        # this ratio it is pruned (regardless of contribution). Targets the
        # truly degenerate needle/sheet shapes. Disabled if ≤0.
        #
        # Empirical sweet spot: 300. Lower (=100) reclaims too many moderately
        # elongated Gaussians that carry real PSNR (wisps/layers); 51K
        # one-shot prune at iter 15500 dropped train PSNR from 44 to 40 — the
        # killed points had been training for 12K steps and the schedule
        # didn't allow them to regrow. Ratio>300 still catches needles/sheets
        # that cause severe popping while preserving normal cloud structure.
        self.prune_aniso_ratio = 300.0
        # How often to run the prune pass after densify_until_iter has stopped
        # the regular path. 1000 = drop bad ellipsoids about as often as we
        # reset accumulator stats. 0 to disable.
        self.post_densify_prune_interval = 1000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
