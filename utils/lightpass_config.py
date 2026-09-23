"""Fixed training behavior and read-only compatibility with saved light passes."""
import ast
import math


def training_lightpass_settings():
    # Saved automatically in cfg_args, never registered as training arguments.
    return dict(tlight_tau_filter=False, tlight_full_grad=True,
                tlight_filter_variance=0.0)


def saved_lightpass_settings(config):
    """Absent fields in an existing checkpoint mean the historical 0.3 pass.

    Accept a parsed mapping or literal argparse Namespace text. New training
    always saves all three fields; old experimental checkpoints remain readable.
    """
    if isinstance(config, str):
        node = ast.parse(config.strip(), mode='eval').body
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'Namespace' and not node.args
                and all(k.arg is not None for k in node.keywords)):
            raise ValueError('cfg_args must contain a literal Namespace(...)')
        config = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    variance = float(config.get('tlight_filter_variance', 0.3))
    if not math.isfinite(variance) or variance < 0:
        raise ValueError('Invalid tlight_filter_variance in cfg_args')
    return dict(tlight_tau_filter=bool(config.get('tlight_tau_filter', False)),
                tlight_full_grad=bool(config.get('tlight_full_grad', False)),
                tlight_filter_variance=variance)
