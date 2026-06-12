#!/usr/bin/env python
"""Carve a held-out test split out of transforms_train.json, stratified by
camera: every camera contributes the same number of frames, chosen randomly
but spread over the time/sun axis.

Selection per camera: sort the camera's frames by time_index, cut the list
into `per_cam` equal chunks, draw one random frame from each chunk. This makes
the split random yet evenly distributed over both camera poses and sun
directions.

Idempotent: on first run the current transforms_train.json is backed up to
transforms_train_full.json; later runs always re-split from that backup, so
re-running (e.g. with a different --per-cam or --seed) never loses frames.

Usage:
    python tools/split_test_set.py [--data data/CloudDataset] [--per-cam 2] [--seed 0]

NOTE: train with --eval from now on. With eval=False the stock Blender loader
merges transforms_test.json back into the training set (dataset_readers.py),
which would silently leak the test frames into training.
"""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CloudDataset")
    ap.add_argument("--per-cam", type=int, default=2,
                    help="test frames drawn from each camera")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_path = os.path.join(args.data, "transforms_train.json")
    test_path = os.path.join(args.data, "transforms_test.json")
    full_path = os.path.join(args.data, "transforms_train_full.json")

    # Re-splits must start from the untouched full set, not from an
    # already-carved train json.
    if not os.path.exists(full_path):
        shutil.copy2(train_path, full_path)
        print(f"backed up full train set -> {full_path}")

    with open(full_path) as f:
        meta = json.load(f)
    frames = meta["frames"]
    print(f"full set: {len(frames)} frames")

    by_cam = defaultdict(list)
    for fr in frames:
        by_cam[fr["camera_index"]].append(fr)

    rng = random.Random(args.seed)
    test_keys = set()
    for ci in sorted(by_cam):
        cam_frames = sorted(by_cam[ci], key=lambda fr: fr.get("time_index", 0))
        n = len(cam_frames)
        k = min(args.per_cam, n)
        # One random draw per equal time chunk -> even spread over sun motion.
        for j in range(k):
            lo = j * n // k
            hi = (j + 1) * n // k
            pick = cam_frames[rng.randrange(lo, hi)]
            test_keys.add(pick["file_path"])

    train_frames = [fr for fr in frames if fr["file_path"] not in test_keys]
    test_frames = [fr for fr in frames if fr["file_path"] in test_keys]
    assert len(train_frames) + len(test_frames) == len(frames)

    header = {k: v for k, v in meta.items() if k != "frames"}
    with open(train_path, "w") as f:
        json.dump({**header, "frames": train_frames}, f, indent=2)
    with open(test_path, "w") as f:
        json.dump({**header, "frames": test_frames}, f, indent=2)

    per_cam_counts = defaultdict(int)
    times = []
    for fr in test_frames:
        per_cam_counts[fr["camera_index"]] += 1
        times.append(fr.get("time_index", 0))
    counts = sorted(per_cam_counts.values())
    print(f"train: {len(train_frames)} frames -> {train_path}")
    print(f"test:  {len(test_frames)} frames ({len(per_cam_counts)} cameras, "
          f"{counts[0]}-{counts[-1]} per cam) -> {test_path}")
    if times:
        print(f"test time_index spread: min {min(times)}  max {max(times)}")
    print("REMINDER: pass --eval to train.py, otherwise the loader merges the "
          "test frames back into training.")


if __name__ == "__main__":
    main()
