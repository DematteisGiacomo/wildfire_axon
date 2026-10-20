#!/usr/bin/env python3
"""Evaluate the Keras and int8 TFLite models on the held-out test split."""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

from keras_model import load_model
from wildfire_data import load_config, load_split, to_model_input


def tflite_predict(path: str, x: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    preds = []
    for sample in x:
        q = np.clip(np.round(sample / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None])
        interp.invoke()
        preds.append(interp.get_tensor(out["index"])[0])
    return np.argmax(np.asarray(preds), axis=1)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> None:
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"\n{name}: accuracy {np.mean(y_true == y_pred):.4f} on {len(y_true)} test images")
    width = max(len(c) for c in classes)
    corner = "true/pred"
    print(f"{corner:>{width + 2}} " + " ".join(f"{c:>{width}}" for c in classes) + "   recall")
    for i, c in enumerate(classes):
        print(f"{c:>{width + 2}} " + " ".join(f"{v:>{width}}" for v in cm[i])
              + f"   {cm[i, i] / max(cm[i].sum(), 1):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir = cfg["paths"]["model_dir"]
    classes = cfg["model"]["classes"]
    x_raw, y = load_split(cfg["paths"]["data_dir"], "test", cfg["data"]["input_size"])
    x = to_model_input(x_raw)

    model = load_model(os.path.join(model_dir, "wildfire_classifier.keras"))
    report("Keras float", y, np.argmax(model.predict(x, verbose=0), axis=1), classes)
    report("TFLite int8", y,
           tflite_predict(os.path.join(model_dir, "wildfire_classifier_int8.tflite"), x), classes)


if __name__ == "__main__":
    main()
