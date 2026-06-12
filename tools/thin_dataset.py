#!/usr/bin/env python
"""Thin the training split with a rotating (camera x time) lattice.

Keeps a frame iff (camera_index + time_index) % stride == phase. This is the
same rotating-subset idea as the supplement-sun capture: both axes keep full
coverage (every camera retains ~n_times/stride evenly spaced times, every
time retains ~n_cams/stride evenly spaced cameras), only the dense product
is thinned. The test split is NOT touched, so metrics stay comparable.

Idempotent: first run backs up transforms_train.json to
transforms_train_dense.json; later runs re-thin from that backup, so
changing --stride/--phase or restoring (--stride 1) never loses frames.

Usage:
    python tools/thin_dataset.py [--data data/CloudDataset] [--stride 3] [--phase 0]
    python tools/thin_dataset.py --stride 1     # restore the dense set
"""
import argparse
import json
import os
import shutil
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CloudDataset")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--exempt-time-from", type=int, default=61,
                    help="frames with time_index >= this are kept unconditionally "
                         "(supplement suns are already a rotating 1/3 camera subset; "
                         "lattice-thinning them again would drop 2/3 of the new suns)")
    args = ap.parse_args()

    train_path = os.path.join(args.data, "transforms_train.json")
    dense_path = os.path.join(args.data, "transforms_train_dense.json")

    if not os.path.exists(dense_path):
        shutil.copy2(train_path, dense_path)
        print(f"backed up dense train set -> {dense_path}")

    with open(dense_path) as f:
        meta = json.load(f)
    frames = meta["frames"]

    if args.stride <= 1:
        kept = frames
    else:
        kept = [fr for fr in frames
                if fr.get("time_index", 0) >= args.exempt_time_from
                or (fr["camera_index"] + fr.get("time_index", 0)) % args.stride
                == args.phase % args.stride]

    header = {k: v for k, v in meta.items() if k != "frames"}
    with open(train_path, "w") as f:
        json.dump({**header, "frames": kept}, f, indent=2)

    per_cam = Counter(fr["camera_index"] for fr in kept)
    per_time = Counter(fr.get("time_index", 0) for fr in kept)
    print(f"dense: {len(frames)} -> kept: {len(kept)} "
          f"(stride {args.stride}, phase {args.phase})")
    if args.stride > 1:
        print(f"frames/cam: {min(per_cam.values())}-{max(per_cam.values())} | "
              f"cams/time: {min(per_time.values())}-{max(per_time.values())}")
    print(f"-> {train_path} (test split untouched)")


if __name__ == "__main__":
    main()
