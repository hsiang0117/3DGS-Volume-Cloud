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
                    # default=True bools need --no-<flag> to be disable-able from the
                    # CLI; store_true would pin them True forever. BooleanOptionalAction
                    # (py3.9+) generates both --<flag> and --no-<flag>.
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
        # transforms_test.json holds a real held-out split (tools/split_test_set.py).
        # Keep True normally: eval=False makes the Blender loader merge test frames
        # back into training, leaking the split and inflating metrics. Disable from
        # the CLI with --no-eval if you really want full-data training.
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Per-tile max-response sort: max deviation of t* from centre depth, in
        # units of σ along the view ray. ≤0 reverts to stock 3DGS centre-depth sort
        # (kernel skips the t* shift). Disabled: it produced blocky tile-boundary
        # artefacts; aniso pruning/penalising controls popping instead. CUDA path
        # is dead at 0.0; set >0 to re-enable without a rebuild.
        self.k_sigma = 0.0
        # T_light source. Default: light-space rasterization (sun-camera shadow
        # pass, record_front_tau CUDA channel + native lightpass backward) with the
        # full shadow gradient (β AND σ_d through scales/rotation). Avoids the voxel
        # cache's needle shadows / chord bias / self-leak / bbox aliasing.
        # --tlight_voxel falls back to the legacy 128^3 voxel cache (correct only for
        # models trained pre-raster; the viewer reads cfg_args to match). store_true
        # can't be disabled from the CLI, hence a fallback flag not tlight_raster=True.
        self.tlight_voxel = False
        self.tlight_raster_res = 512
        # Output tonemap to match the GT's display space. UE's HighResScreenshot GT
        # is filmic-tonemapped LDR while our physical shading is linear; fitting a
        # nonlinear target with a linear model shows up as dynamic-range compression.
        # When on, render() lifts the per-Gaussian radiance clamp (HDR) and applies
        # the fixed Narkowicz ACES approximation to the final image, so loss and
        # metrics compare in the GT's own space. Default True (current baseline output
        # space). Disable from the CLI with --no-tonemap_aces for a truly-linear GT
        # (then tonemap must be OFF; see tonemap_learnable note).
        self.tonemap_aces = True
        # Learnable output tonemap (opt-in alternative to fixed ACES): same Narkowicz
        # rational form but its 4 coeffs (a,b,c,d) are optimised (e pinned), so the
        # model fits the GT's true display curve instead of a fixed approximation,
        # absorbing residual curve-mismatch for any filmic GT (not just UE). Implies
        # the HDR clamp like tonemap_aces and takes precedence over it when on.
        # Default OFF: Narkowicz is already a good enough fit for UE filmic, so the
        # extra DoF doesn't pay; kept as insurance for a different filmic engine whose
        # curve drifts from Narkowicz. (For a truly-linear HDR GT use tonemap OFF, NOT
        # learnable — the Narkowicz family cannot represent identity.)
        self.tonemap_learnable = False
        # --- Stage 2: environment lighting (frozen geometry + global sky) ---
        # When on (set by --stage2), render adds the atmospheric env term on top of
        # the frozen Stage-1 sun shading:
        #     L = T_sun(sun_dir) ⊙ sun_term  +  ρ · Σ_lm E_lm(sun_dir) · V_lm
        # T_sun (RGB ≤1, sun atmospheric transmittance — expresses low-sun dimming/
        # reddening) and E_lm (sky radiance SH) are GLOBAL functions of sun_dir (a
        # small MLP, no per-Gaussian colour DOF); V_lm is the precomputed per-Gaussian
        # achromatic sky-visibility transfer. Persists to cfg_args so the viewer
        # auto-detects. Default OFF — Stage 1 behaviour unchanged.
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
        self.feature_lr = 0.0025
        self.extiction_lr = 0.025
        self.g_factor_lr = 0.0025
        # LR for per-Gaussian multiple-scattering octave weights (softplus, >=0).
        # Same order as g_factor; tune down if the weights overfit per-view.
        self.octave_weights_lr = 0.0025
        # LR for the 4 global learnable tonemap coeffs (only used with
        # --tonemap_learnable). Higher than the per-Gaussian LRs because it's a
        # handful of scalars seen by every pixel of every frame; decays to 0.1x.
        self.tonemap_lr = 1e-3
        # Monotonicity penalty on the learnable tonemap (only with
        # --tonemap_learnable). softplus already guarantees positivity / no
        # poles; this is a cheap insurance that f stays non-decreasing on [0,8]
        # so highlights never invert. Hinge on negative slope, like lambda_aniso.
        self.lambda_tonemap_mono = 1e-2
        # LR for the Stage-2 environment net (global T_sun + E_lm MLP of sun_dir),
        # only used with --stage2. Lives in its own Adam (isolated from densify/prune
        # like the tonemap optimizer); decays to 0.1x over the schedule.
        self.env_lr = 1e-3
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_scale = 0.1
        # Anisotropy penalty (log-ratio form): zero below `aniso_ratio_max`,
        # quadratic in log-ratio above. Cloud is an isotropic medium; long
        # ellipsoids cause depth-sort popping.
        #
        # Keep λ small. Bounding aniso and keeping PSNR high are in tension: λ=0.05
        # collapses PSNR to ~25 (regularizer dominates fit), λ=0.001 gives PSNR ~43
        # with aniso largely unbounded — cloud structure (wisps, layers) genuinely
        # benefits from elongated Gaussians at current capacity. For aggressive aniso
        # bounding prefer split-on-densify (`densify_split_aniso_max`) over loss reg.
        #
        # aniso p99 does NOT converge — it grows monotonically while unconstrained
        # (λ=0 throughout gives p99=183). Hence λ=0.001 with the regulariser run for
        # the FULL schedule (aniso_until_iter = iterations) so a persistent constraint
        # drives p99 to a plateau rather than letting it grow in the back half.
        self.lambda_aniso = 0.001
        self.aniso_ratio_max = 5.0
        # Run the aniso regulariser for the whole schedule. A persistent constraint
        # is needed because aniso p99 grows monotonically once the reg switches off.
        # Watch for uniform shrinkage (scale mean dropping) as a side effect.
        self.aniso_until_iter = 30_000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 1e-4
        self.densify_scale_grad_threshold = 1e-6
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
        # Needle surgery: every `needle_split_interval` iters, split Gaussians with
        # aniso ratio > `needle_split_ratio` into two children along the major axis
        # (appearance-conserving, ratio halves). A structural hard ceiling on the
        # aniso tail that the soft regulariser cannot hold. 0 disables. Runs through
        # the whole schedule (also post-densify).
        self.needle_split_interval = 1000
        self.needle_split_ratio = 30.0
        self.needle_split_until_iter = 29_000   # stop before final settle/eval
        super().__init__(parser, "Optimization Parameters")
