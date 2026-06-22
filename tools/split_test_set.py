#!/usr/bin/env python
"""Carve a held-out test split out of transforms_train.json.

Two modes:

* SUN-held-out (uniform-sun dataset; pass --held-out-suns): whole sun
  directions named by --held-out-suns (their every frame) go to test as a
  relighting-generalisation set, plus --per-sun random frames from each
  remaining sun (stratified over the sun axis). This is the split used by
  data/CloudDatasetUniform: --held-out-suns 7,22,37,52 --per-sun 1.

* PER-camera (legacy CloudDataset; default): every camera contributes
  --per-cam frames, drawn one per equal time-chunk so the split is random yet
  spread over both camera poses and sun directions.

Idempotent: on first run the current transforms_train.json is backed up to
transforms_train_full.json; later runs always re-split from that backup, so
re-running (different --held-out-suns / --per-cam / --seed) never loses frames.

Usage:
    # uniform-sun dataset (relighting held-out suns + 1 frame/other sun)
    python tools/split_test_set.py --data data/CloudDatasetUniform \
        --held-out-suns 7,22,37,52 --per-sun 1
    # legacy per-camera split
    python tools/split_test_set.py --data data/CloudDataset --per-cam 2

NOTE: train with --eval (default True). With eval=False the Blender loader
merges transforms_test.json back into training (dataset_readers.py), leaking
the test frames.
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
    ap.add_argument("--held-out-suns", default=None,
                    help="comma-separated time_index values held out ENTIRELY as a "
                         "relighting test (enables sun-stratified mode). e.g. 7,22,37,52")
    ap.add_argument("--per-sun", type=int, default=1,
                    help="sun mode: random test frames drawn from each non-held-out sun")
    ap.add_argument("--per-cam", type=int, default=2,
                    help="legacy mode: test frames drawn from each camera")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_path = os.path.join(args.data, "transforms_train.json")
    test_path = os.path.join(args.data, "transforms_test.json")
    full_path = os.path.join(args.data, "transforms_train_full.json")

    # Re-splits must start from the untouched full set, not an already-carved train json.
    if not os.path.exists(full_path):
        shutil.copy2(train_path, full_path)
        print(f"backed up full train set -> {full_path}")

    with open(full_path) as f:
        meta = json.load(f)
    frames = meta["frames"]
    print(f"full set: {len(frames)} frames")

    rng = random.Random(args.seed)
    test_keys = set()

    if args.held_out_suns:
        # Sun-stratified: whole held-out suns + per-sun random draws.
        held = {int(x) for x in args.held_out_suns.split(",") if x.strip() != ""}
        by_sun = defaultdict(list)
        for fr in frames:
            by_sun[fr.get("time_index", 0)].append(fr)
        for ti in sorted(by_sun):
            sun_frames = by_sun[ti]
            if ti in held:
                for fr in sun_frames:                     # whole sun -> test
                    test_keys.add(fr["file_path"])
            else:
                k = min(args.per_sun, len(sun_frames))
                for fr in rng.sample(sun_frames, k):       # per-sun random -> test
                    test_keys.add(fr["file_path"])
        print(f"sun mode: held-out suns {sorted(held)} (whole) + {args.per_sun}/other sun")
    else:
        # Legacy per-camera: one random draw per equal time chunk.
        by_cam = defaultdict(list)
        for fr in frames:
            by_cam[fr["camera_index"]].append(fr)
        for ci in sorted(by_cam):
            cam_frames = sorted(by_cam[ci], key=lambda fr: fr.get("time_index", 0))
            n = len(cam_frames)
            k = min(args.per_cam, n)
            for j in range(k):
                lo, hi = j * n // k, (j + 1) * n // k
                test_keys.add(cam_frames[rng.randrange(lo, hi)]["file_path"])
        print(f"per-cam mode: {args.per_cam}/camera")

    train_frames = [fr for fr in frames if fr["file_path"] not in test_keys]
    test_frames = [fr for fr in frames if fr["file_path"] in test_keys]
    assert len(train_frames) + len(test_frames) == len(frames)

    header = {k: v for k, v in meta.items() if k != "frames"}
    with open(train_path, "w") as f:
        json.dump({**header, "frames": train_frames}, f, indent=2)
    with open(test_path, "w") as f:
        json.dump({**header, "frames": test_frames}, f, indent=2)

    times = [fr.get("time_index", 0) for fr in test_frames]
    print(f"train: {len(train_frames)} frames -> {train_path}")
    print(f"test:  {len(test_frames)} frames -> {test_path}")
    if times:
        print(f"test time_index spread: min {min(times)}  max {max(times)}")
    print("REMINDER: pass --eval to train.py, otherwise the loader merges the "
          "test frames back into training.")


if __name__ == "__main__":
    main()
