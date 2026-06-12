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
        # Default True: transforms_test.json now holds a real held-out split
        # (tools/split_test_set.py). With eval=False the Blender loader merges
        # test frames back into training, silently leaking the split and
        # inflating final metrics. (argparse store_true means this can no
        # longer be disabled from the CLI — flip it here if ever needed.)
        self.eval = True
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
        # depth sort (the kernel skips the t* shift entirely).
        #
        # Disabled (0.0). The per-tile max-response sort was introduced to fix
        # long-axis popping, but in practice it produced blocky tile-boundary
        # artefacts, while pruning/penalising elongated Gaussians (the aniso
        # channel) controls popping effectively on its own. So we keep the
        # aniso machinery and revert sorting to stock 3DGS. The CUDA path is
        # retained but dead at k_sigma=0; set >0 to re-enable without a rebuild.
        self.k_sigma = 0.0
        # T_light source. DEFAULT (v3, validated 2026-06-12 on the uniform-sun
        # dataset): light-space rasterization (sun-camera shadow pass,
        # record_front_tau CUDA channel + native lightpass backward) with the
        # full shadow gradient (β AND σ_d through scales/rotation). Fixes the
        # voxel cache's needle shadows / chord bias / self-leak / bbox
        # aliasing, and with uniform sun coverage + needle surgery holds
        # aniso p99 ~22 at no PSNR cost (held-out-sun 30.83).
        # --tlight_voxel falls back to the legacy 128^3 voxel cache (only
        # correct pairing for models trained pre-raster; the viewer reads
        # cfg_args to match). store_true defaults can't be disabled from the
        # CLI, hence a dedicated fallback flag instead of tlight_raster=True.
        self.tlight_voxel = False
        self.tlight_raster_res = 512
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
        # LR for per-Gaussian multiple-scattering octave weights (softplus, >=0).
        # Same order as g_factor; tune down if the weights overfit per-view.
        self.octave_weights_lr = 0.0025
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
        #
        # CONTINUOUS-CONSTRAINT TEST: aniso p99 was found NOT to converge — it
        # grows monotonically whenever unconstrained. The soft regulariser was
        # previously switched off at aniso_until_iter=15000, after which p99
        # climbed freely (32 -> 58 by 30k, still rising). And λ=0 throughout
        # gave p99=183. So here we keep λ=0.001 AND run the regulariser for the
        # FULL schedule (aniso_until_iter = iterations) to test whether a
        # persistent constraint drives p99 to a plateau instead of letting it
        # grow in the back half. aniso_ratio_max=5 unchanged.
        self.lambda_aniso = 0.001
        self.aniso_ratio_max = 5.0
        # Run the aniso regulariser for the whole schedule (was 15000). The old
        # early-off was meant to avoid L_vol+reg co-shrinking the cloud after
        # densify; we now test persistent constraint instead. Watch for uniform
        # shrinkage (scale mean dropping) as the side effect to rule out.
        self.aniso_until_iter = 30_000
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
        # How often to run the prune pass after densify_until_iter has stopped
        # the regular path. 1000 = drop bad ellipsoids about as often as we
        # reset accumulator stats. 0 to disable.
        self.post_densify_prune_interval = 1000
        # Oversampling factor for out-of-plane sun frames (|sun_x| > 0.1) in
        # the training sampler. The TOD arc supplies 61 in-plane suns vs 24
        # supplement suns; ~2.5 equalises per-direction gradient frequency.
        # 1.0 = uniform (default). Diagnostic knob for the v3 needle exploit.
        self.sun_balance_weight = 1.0
        # Needle surgery: every `needle_split_interval` iters, split Gaussians
        # with aniso ratio > `needle_split_ratio` into two children along the
        # major axis (appearance-conserving, ratio halves). A structural hard
        # ceiling on the aniso tail that the soft regulariser cannot hold —
        # the contrast-compression pressure (missing ambient light) keeps
        # feeding needles, and loss-side λ can only trade PSNR against them.
        # 0 disables. Runs through the whole schedule (also post-densify).
        self.needle_split_interval = 1000
        self.needle_split_ratio = 30.0
        self.needle_split_until_iter = 29_000   # stop before final settle/eval
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
