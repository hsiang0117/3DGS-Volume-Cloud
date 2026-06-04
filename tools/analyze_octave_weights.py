#!/usr/bin/env python
"""Analyze the learned per-Gaussian multiple-scattering octave weights.

Reads a trained point_cloud.ply, applies the softplus activation to the raw
octave_weight_* columns (matching GaussianModel.get_octave_weights), and reports
the per-octave distribution against the fixed 0.5^n baseline schedule.

Key question this answers: did the model shift multiple-scattering energy toward
octave 0 (the un-diluted g_eff = g term)? If so, that explains why the learned
HG anisotropy g rose from ~0.24 (fixed schedule) to ~0.63 here — less energy in
the high octaves means less g-isotropization (g_eff = g * 0.5^n) diluting the
effective phase, so g itself can/must grow to represent the real forward lobe.

Usage:
    python tools/analyze_octave_weights.py <path/to/point_cloud.ply>
"""
import sys
import numpy as np
from plyfile import PlyData


def softplus(x):
    # numerically stable softplus, matches torch.nn.functional.softplus
    return np.logaddexp(0.0, x)


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
        print(f"  available columns: {names}")
        sys.exit(1)

    P = el.count
    n_oct = len(ow_names)
    raw = np.zeros((P, n_oct), dtype=np.float64)
    for i, nm in enumerate(ow_names):
        raw[:, i] = np.asarray(el[nm])

    w = softplus(raw)                       # (P, n_oct), activated weights >= 0
    fixed = np.array([0.5 ** n for n in range(n_oct)])

    # --- per-octave distribution ---
    mean = w.mean(axis=0)
    std = w.std(axis=0)
    p10 = np.percentile(w, 10, axis=0)
    p50 = np.percentile(w, 50, axis=0)
    p90 = np.percentile(w, 90, axis=0)

    print(f"\nLoaded {P} Gaussians from {path}")
    print(f"Octaves: {n_oct}\n")
    hdr = f"{'oct':>3} | {'fixed 0.5^n':>11} | {'learned mean':>12} | {'std':>7} | " \
          f"{'p10':>7} | {'p50':>7} | {'p90':>7} | {'mean/fixed':>10}"
    print(hdr)
    print("-" * len(hdr))
    for n in range(n_oct):
        ratio = mean[n] / fixed[n] if fixed[n] > 0 else float("nan")
        print(f"{n:>3} | {fixed[n]:>11.4f} | {mean[n]:>12.4f} | {std[n]:>7.4f} | "
              f"{p10[n]:>7.4f} | {p50[n]:>7.4f} | {p90[n]:>7.4f} | {ratio:>10.3f}")

    # --- energy distribution across octaves (normalized) ---
    # How is total scattering energy split across octaves? Compare the learned
    # split to the fixed schedule's split. Energy shifting toward octave 0 is the
    # signature that explains the g rise.
    learned_frac = mean / mean.sum()
    fixed_frac = fixed / fixed.sum()
    print("\nEnergy fraction per octave (mean-weight share of total):")
    print(f"{'oct':>3} | {'fixed':>8} | {'learned':>8} | {'shift':>8}")
    print("-" * 34)
    for n in range(n_oct):
        shift = learned_frac[n] - fixed_frac[n]
        print(f"{n:>3} | {fixed_frac[n]:>8.3f} | {learned_frac[n]:>8.3f} | {shift:>+8.3f}")

    # --- headline diagnostics ---
    print("\n--- diagnostics ---")
    print(f"octave-0 energy share: fixed {fixed_frac[0]:.3f} -> learned {learned_frac[0]:.3f} "
          f"({'MORE' if learned_frac[0] > fixed_frac[0] else 'LESS'} energy in the un-diluted term)")
    hi = slice(3, n_oct)
    print(f"high-octave (n>=3) energy share: fixed {fixed_frac[hi].sum():.3f} -> "
          f"learned {learned_frac[hi].sum():.3f}")
    # effective isotropization: weighted-average of 0.5^n under each energy split
    iso_fixed = (fixed_frac * np.array([0.5 ** n for n in range(n_oct)])).sum()
    iso_learned = (learned_frac * np.array([0.5 ** n for n in range(n_oct)])).sum()
    print(f"energy-weighted mean g-dilution factor (avg of 0.5^n):")
    print(f"  fixed   {iso_fixed:.4f}  -> learned {iso_learned:.4f}  "
          f"({'less' if iso_learned > iso_fixed else 'more'} dilution => "
          f"{'higher' if iso_learned > iso_fixed else 'lower'} effective g headroom)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
