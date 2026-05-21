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

from datetime import datetime
from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def build_timestamped_model_path(base_dir="./output"):
    """Generate an unused output path of the form ./output/YYYYMMDD_HHMMSS,
    falling back to a _NN numeric suffix if the timestamp directory already
    exists (multiple runs started within the same second)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = path.join(base_dir, timestamp)

    if not path.exists(model_path):
        return model_path

    suffix = 1
    while True:
        candidate = path.join(base_dir, f"{timestamp}_{suffix:02d}")
        if not path.exists(candidate):
            return candidate
        suffix += 1
