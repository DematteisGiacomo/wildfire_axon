#!/usr/bin/env python3
"""Save a contact sheet of the test images the int8 model gets wrong."""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from evaluate import tflite_probs
from wildfire_data import load_config, load_split, to_model_input


def contact_sheet(images: list[np.ndarray], captions: list[str], tile: int, cols: int) -> Image.Image:
    rows = max(1, (len(images) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 16)), "black")
    draw = ImageDraw.Draw(sheet)
    for i, (img, cap) in enumerate(zip(images, captions)):
        x, y = (i % cols) * tile, (i // cols) * (tile + 16)
        sheet.paste(Image.fromarray(img).resize((tile, tile), Image.BICUBIC), (x, y))
        draw.text((x + 2, y + tile + 2), cap, fill="white")
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--cols", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir, classes = cfg["paths"]["model_dir"], cfg["model"]["classes"]
    size = cfg["data"]["input_size"]
    x_raw, y = load_split(cfg["paths"]["data_dir"], "test", size)
    probs = tflite_probs(os.path.join(model_dir, "wildfire_classifier_int8.tflite"),
                         to_model_input(x_raw))
    pred = probs.argmax(axis=1)

    for label, name in enumerate(classes):
        wrong = np.flatnonzero((y == label) & (pred != label))
        wrong = wrong[np.argsort(-probs[wrong, 1 - label])]
        caps = [f"#{i} p({classes[1 - label]})={probs[i, 1 - label]:.2f}" for i in wrong]
        path = os.path.join(model_dir, f"errors_true_{name}.png")
        contact_sheet([x_raw[i] for i in wrong], caps, args.tile, args.cols).save(path)
        print(f"{len(wrong)} {name} test images misclassified -> {path}")


if __name__ == "__main__":
    main()
