#!/usr/bin/env python
"""Reconstruct and plot the effective scattering phase function from a trained
checkpoint, to test the "sharp forward peak + isotropic base" hypothesis.

The renderer's angular term (stripping the transmittance factor T_eff, which is
a per-Gaussian/per-light transmittance, NOT part of the phase function) is

    Phi(cos t) = sum_n w_n * HG(g * 0.5^n, cos t),     HG normalized by 1/4pi

We evaluate this over cos(theta) in [-1, 1] using the population-mean learned
weights w_n and mean g, and compare against:
  (a) the fixed 0.5^n schedule with the same g  (what the model used before), and
  (b) a single HG(g)  (one lobe, no multi-octave).

If the learned curve has a sharper forward peak AND a fatter isotropic tail than
both baselines, that confirms octave 0 specializes the forward lobe while the
energy-loaded high octaves provide a near-isotropic base — explaining why the
learned g rose to ~0.63 without the effective phase becoming unrealistically
forward-peaked.

Outputs a PNG next to the PLY (and prints summary stats; no display needed).

Usage:
    python tools/plot_phase_function.py <path/to/point_cloud.ply>
"""
import os
import sys
import math
import numpy as np
from plyfile import PlyData

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def softplus(x):
    return np.logaddexp(0.0, x)


def hg(g, cos_t):
    # 1/4pi-normalized Henyey-Greenstein, matching the renderer.
    inv_4pi = 1.0 / (4.0 * math.pi)
    denom = np.power(1.0 + g * g - 2.0 * g * cos_t, 1.5) + 1e-6
    return inv_4pi * (1.0 - g * g) / denom


def phase_sum(weights, g, cos_t, n_oct):
    """sum_n w_n * HG(g*0.5^n, cos_t). weights: (n_oct,), cos_t: (M,)."""
    out = np.zeros_like(cos_t)
    for n in range(n_oct):
        out += weights[n] * hg(g * (0.5 ** n), cos_t)
    return out


def main(path):
    ply = PlyData.read(path)
    el = ply.elements[0]
    names = [p.name for p in el.properties]

    ow_names = sorted(
        [n for n in names if n.startswith("octave_weight_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not ow_names:
        print(f"ERROR: no octave_weight_* columns in {path}.")
        sys.exit(1)
    if "g_factor" not in names:
        print(f"ERROR: no g_factor column in {path}.")
        sys.exit(1)

    P = el.count
    n_oct = len(ow_names)
    raw = np.zeros((P, n_oct), dtype=np.float64)
    for i, nm in enumerate(ow_names):
        raw[:, i] = np.asarray(el[nm])
    w_learned = softplus(raw).mean(axis=0)             # (n_oct,)

    # g activation in the model is 0.8 * tanh(raw)
    g_raw = np.asarray(el["g_factor"]).astype(np.float64)
    g_learned = float(np.mean(0.8 * np.tanh(g_raw)))

    w_fixed = np.array([0.5 ** n for n in range(n_oct)])

    cos_t = np.linspace(-1.0, 1.0, 1024)

    phi_learned = phase_sum(w_learned, g_learned, cos_t, n_oct)
    phi_fixed = phase_sum(w_fixed, g_learned, cos_t, n_oct)  # same g, fixed weights
    phi_single = hg(g_learned, cos_t)                        # one lobe

    # Forward (cos=+1) vs back (cos=-1) vs side (cos=0) — the shape signature.
    def at(arr, c):
        idx = np.argmin(np.abs(cos_t - c))
        return arr[idx]

    print(f"\nLoaded {P} Gaussians from {path}")
    print(f"mean g = {g_learned:.4f}")
    print(f"learned octave weights (mean): {np.array2string(w_learned, precision=4)}")
    print(f"fixed   octave weights:        {np.array2string(w_fixed, precision=4)}\n")

    def peak_ratio(arr):
        return at(arr, 1.0) / max(at(arr, -1.0), 1e-9)

    print(f"{'curve':>16} | {'fwd(+1)':>9} | {'side(0)':>9} | {'back(-1)':>9} | {'fwd/back':>9}")
    print("-" * 64)
    for label, arr in [("learned multi", phi_learned),
                       ("fixed multi", phi_fixed),
                       ("single HG(g)", phi_single)]:
        print(f"{label:>16} | {at(arr,1.0):>9.4f} | {at(arr,0.0):>9.4f} | "
              f"{at(arr,-1.0):>9.4f} | {peak_ratio(arr):>9.2f}")

    print("\n--- shape test ---")
    print(f"forward/back ratio: learned {peak_ratio(phi_learned):.2f} vs "
          f"fixed {peak_ratio(phi_fixed):.2f} vs single {peak_ratio(phi_single):.2f}")
    print(f"side (isotropic-base proxy) at cos=0: learned {at(phi_learned,0.0):.4f} vs "
          f"fixed {at(phi_fixed,0.0):.4f}")
    if not HAVE_MPL:
        print("\n(matplotlib not installed; skipped PNG. `pip install matplotlib` to plot.)")
        return

    # --- plot: linear and semilog, angle in degrees ---
    theta_deg = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
    order = np.argsort(theta_deg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ylog in zip(axes, [False, True]):
        ax.plot(theta_deg[order], phi_learned[order], label=f"learned multi (g={g_learned:.2f})", lw=2)
        ax.plot(theta_deg[order], phi_fixed[order], label="fixed 0.5^n multi", lw=1.5, ls="--")
        ax.plot(theta_deg[order], phi_single[order], label="single HG(g)", lw=1.2, ls=":")
        ax.set_xlabel("scattering angle (deg); 0 = forward")
        ax.set_ylabel("phase value")
        if ylog:
            ax.set_yscale("log")
            ax.set_title("semilog")
        else:
            ax.set_title("linear")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Effective scattering phase function (transmittance factor stripped)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(path), "phase_function.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved plot: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
