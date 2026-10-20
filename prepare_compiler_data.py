#!/usr/bin/env python3
"""Export the test split as float32 .npy files for the Axon compiler executor."""

from __future__ import annotations

import argparse
import os

import numpy as np

from wildfire_data import load_config, load_split, to_model_input


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["paths"]["compiler_data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    x_raw, y = load_split(cfg["paths"]["data_dir"], "test", cfg["data"]["input_size"])
    x_path = os.path.join(out_dir, "x_test_wildfire.npy")
    y_path = os.path.join(out_dir, "y_test_wildfire.npy")
    np.save(x_path, to_model_input(x_raw))
    np.save(y_path, y.astype(np.int64))
    print(f"Saved {x_path} {x_raw.shape} and {y_path} ({np.bincount(y).tolist()} per class)")


if __name__ == "__main__":
    main()
