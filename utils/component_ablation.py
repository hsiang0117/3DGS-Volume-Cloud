"""Branch training default and checkpoint-driven component ablation behavior.

Missing mode in a historical checkpoint/pipeline means full, never this branch's
training default. Mode metadata is persisted by utils.lightpass_config.
"""

DEFAULT_COMPONENT_ABLATION = "single_scattering"
VALID_COMPONENT_ABLATIONS = (
    "full", "no_tlight", "single_scattering", "no_tlight_single_scattering",
)


def validate_component_ablation(mode):
    if mode not in VALID_COMPONENT_ABLATIONS:
        raise ValueError(f"Invalid component_ablation {mode!r}; expected one of {VALID_COMPONENT_ABLATIONS}")
    return mode


def training_component_ablation():
    return validate_component_ablation(DEFAULT_COMPONENT_ABLATION)


def saved_component_ablation(config):
    # Legacy files were trained with the full formula, even on this branch.
    return validate_component_ablation(config.get("component_ablation", "full"))


def pipeline_component_ablation(pipe):
    return validate_component_ablation(getattr(pipe, "component_ablation", "full"))


def bypass_sun_transmittance(mode):
    mode = validate_component_ablation(mode)
    return mode in ("no_tlight", "no_tlight_single_scattering")


def use_single_scattering(mode):
    mode = validate_component_ablation(mode)
    return mode in ("single_scattering", "no_tlight_single_scattering")
