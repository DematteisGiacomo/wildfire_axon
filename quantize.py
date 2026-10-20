#!/usr/bin/env python3
"""Convert the trained Keras model to float32 and full-int8 TFLite."""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

from keras_model import load_model, representative_input
from wildfire_data import load_config, load_split, to_model_input


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir, data_dir = cfg["paths"]["model_dir"], cfg["paths"]["data_dir"]
    model = load_model(os.path.join(model_dir, "wildfire_classifier.keras"))

    x_train, _ = load_split(data_dir, "train", cfg["data"]["input_size"])
    rng = np.random.default_rng(cfg["data"]["seed"])
    picks = rng.choice(len(x_train), size=min(cfg["quantize"]["calibration_samples"],
                                              len(x_train)), replace=False)
    x_cal = to_model_input(x_train[picks])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32 = converter.convert()
    fp32_path = os.path.join(model_dir, "wildfire_classifier_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(fp32)
    print(f"Saved {fp32_path} ({len(fp32)} bytes)")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_input(x_cal)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8 = converter.convert()
    int8_path = os.path.join(model_dir, "wildfire_classifier_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(int8)
    print(f"Saved {int8_path} ({len(int8)} bytes)")


if __name__ == "__main__":
    main()
