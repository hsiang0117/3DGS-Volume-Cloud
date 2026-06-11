#!/usr/bin/env python
"""Merge transforms_supplement.json (out-of-plane sun frames from
cloud_dataset_generator.main_supplement) into transforms.json, then re-run
convert_transforms.py to produce the OpenGL training json.

Idempotent: frames already present in transforms.json (same file_path) are
skipped, so re-running after a partial capture is safe.

Usage:
    python tools/merge_supplement.py [dataset_dir]
Default dataset_dir = D:/CloudDataset (the UE capture output).

After merging, regenerate the training jsons:
    python convert_transforms.py <dataset_dir>/transforms.json <data/CloudDataset/transforms_train.json 的源>
    python tools/split_test_set.py        # re-split from the new full set
NOTE: split_test_set.py re-splits from transforms_train_full.json — delete
that backup first so it re-snapshots the MERGED full set:
    rm data/CloudDataset/transforms_train_full.json
"""
import json
import sys
import shutil
from pathlib import Path


def main(dataset_dir):
    d = Path(dataset_dir)
    main_path = d / "transforms.json"
    supp_path = d / "transforms_supplement.json"

    with open(main_path, encoding="utf-8") as f:
        main_data = json.load(f)
    with open(supp_path, encoding="utf-8") as f:
        supp_data = json.load(f)

    existing = {fr["file_path"] for fr in main_data["frames"]}
    added, skipped = 0, 0
    for fr in supp_data["frames"]:
        if fr["file_path"] in existing:
            skipped += 1
            continue
        main_data["frames"].append(fr)
        added += 1

    if added == 0:
        print(f"nothing to merge ({skipped} frames already present)")
        return

    backup = d / "transforms_premerge_backup.json"
    if not backup.exists():
        shutil.copy2(main_path, backup)
        print(f"backed up -> {backup}")

    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(main_data, f, indent=2)

    times = sorted({fr["time_index"] for fr in supp_data["frames"]})
    print(f"merged {added} frames ({skipped} skipped), total now {len(main_data['frames'])}")
    print(f"supplement time_index range: {times[0]}..{times[-1]} ({len(times)} suns)")
    print("next: re-run convert_transforms.py, then delete transforms_train_full.json "
          "and re-run tools/split_test_set.py")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "D:/CloudDataset")
