#!/usr/bin/env python3
"""Convert the trained Keras model to float32 and full-int8 TFLite."""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

from keras_model import load_model, representative_input
from wildfire_data import load_config, load_split, to_model_input


def save(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
    print(f"Saved {path} ({len(data)} bytes)")


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

    save(os.path.join(model_dir, "wildfire_classifier_fp32.tflite"),
         tf.lite.TFLiteConverter.from_keras_model(model).convert())

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_input(x_cal)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    save(os.path.join(model_dir, "wildfire_classifier_int8.tflite"), converter.convert())


if __name__ == "__main__":
    main()
