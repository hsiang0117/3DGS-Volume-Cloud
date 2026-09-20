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

from argparse import ArgumentParser, Namespace, BooleanOptionalAction
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
                    # BooleanOptionalAction emits both --<flag> and --no-<flag>, so
                    # default-True flags stay disable-able from the CLI.
                    action = BooleanOptionalAction if value else "store_true"
                    group.add_argument("--" + key, default=value, action=action)
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
        # Max decoded frames kept in the CPU image cache (LRU); 0 = unlimited.
        self.image_cache_max = 0
        # eval=True keeps transforms_test.json out of training; --no-eval merges
        # the test frames back in for full-data training.
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # T_light source: light-space rasterization (sun-camera shadow pass,
        # record_front_tau + native lightpass backward) with the full shadow
        # gradient through σ_t and scales/rotation. --tlight_voxel selects the
        # 128^3 voxel cache instead (a fallback flag rather than
        # tlight_raster=True, so the DEFAULT is expressed by an absent flag and
        # old cfgs without the key still resolve to raster); the viewer matches
        # the source a model was trained with via cfg_args.
        self.tlight_voxel = False
        self.tlight_raster_res = 512
        # Apply the fixed Narkowicz ACES curve to the final image so loss and
        # metrics live in the GT's display space; render() lifts the per-Gaussian
        # radiance clamp to HDR in this mode. Disable with --no-tonemap_aces for a
        # truly-linear GT.
        self.tonemap_aces = True
        # Learnable variant of the same Narkowicz curve: (a,b,c,d) are optimised
        # (e pinned). Takes precedence over --tonemap_aces when on; for a
        # truly-linear GT turn tonemapping off instead, since this family cannot
        # represent identity.
        self.tonemap_learnable = False
        # Stage 2 environment lighting: adds a global atmospheric term on top of
        # the frozen Stage-1 sun shading,
        #     L = T_sun(v_l) ⊙ sun_term  +  ω · Σ_lm E_lm(v_l) · V_lm
        # T_sun and E_lm are global functions of v_l; V_lm is the precomputed
        # per-Gaussian sky-visibility transfer. Enabled by --stage2. NOTE: it is
        # written to cfg_args but no reader consumes it — the viewer and the eval
        # scripts detect a Stage-2 model from the PLY's env sidecars instead
        # (env.json + env_net.pt + sky_transfer.npy). Kept for provenance only.
        self.env_lighting = False
        self.env_sh_order = 2          # SH order for sky radiance E_lm and visibility V_lm (SH2 = 9 coeffs)
        self.env_transfer_dirs = 48    # # hemisphere directions sampled for the V_lm precompute
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.omega_lr = 0.0025
        self.sigma_t_lr = 0.025
        self.g_lr = 0.0025
        # LR for per-Gaussian multiple-scattering octave weights (softplus, >= 0).
        self.w_lr = 0.0025
        # LR for the 4 global learnable tonemap coeffs (only with
        # --tonemap_learnable); decays to 0.1x over the schedule.
        self.tonemap_lr = 1e-3
        # Monotonicity penalty on the learnable tonemap (only with
        # --tonemap_learnable): hinge on negative slope over [0,8].
        self.lambda_tonemap_mono = 1e-2
        # LR for the Stage-2 environment net (global T_sun + E_lm MLP of v_l),
        # only with --stage2; own Adam, decays to 0.1x over the schedule.
        self.env_lr = 1e-3
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_scale = 0.1
        # Anisotropy penalty (log-ratio form): zero below `aniso_ratio_max`,
        # quadratic in the log-ratio above.
        self.lambda_aniso = 0.001
        self.aniso_ratio_max = 5.0
        # Iteration up to which the aniso regulariser runs, EXCLUSIVE: the
        # consumer is `if iteration < aniso_until_iter`, so with the default
        # 30_000 the last regularised iteration is 29_999 and iteration 30_000
        # is NOT penalised. To cover an N-iteration run, set this to N + 1.
        self.aniso_until_iter = 30_000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 1e-4
        self.densify_scale_grad_threshold = 1e-6
        # Adaptive density threshold: take the top `densify_top_frac` of position
        # gradients as each round's threshold.
        self.densify_adaptive = True
        self.densify_top_frac = 0.005          # top 0.5%
        self.densify_grad_min = 5e-5           # absolute floor
        # Prune threshold on the mean Σ(α·T) a Gaussian contributes over the
        # frames it is visible in.
        self.contribution_threshold = 1e-4
        # Gates ONLY the contribution channel below: mean Σ(α·T) is judged after
        # this many visible frames. The dead-point channel (never visible in the
        # window) is deliberately not gated by it — such a Gaussian has no mean
        # to threshold, so requiring visibility would never remove it.
        self.prune_min_visible_frames = 5
        self.resurrect_interval = 3000         # every N iters, reset bottom σ_t
        self.resurrect_fraction = 0.05         # 5% of points
        # Clear the contribution accumulator every N iterations so the running
        # mean tracks the current model; ≤ 0 never resets.
        self.contribution_reset_interval = 1000
        # Interval of the extra prune pass run inside the densify window;
        # 0 disables.
        self.post_densify_prune_interval = 1000
        # Needle surgery: every `needle_split_interval` iterations, split Gaussians
        # whose max/min scale ratio exceeds `needle_split_ratio` into two children
        # along the major axis; 0 disables.
        self.needle_split_interval = 1000
        self.needle_split_ratio = 30.0
        self.needle_split_until_iter = 29_000   # last iteration surgery runs on
        super().__init__(parser, "Optimization Parameters")
