"""Shared dataset helpers for the wildfire image classifier."""

from __future__ import annotations

import os

import numpy as np
import yaml

SPLITS = ("train", "val", "test")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_path(data_dir: str, split: str, size: int) -> str:
    return os.path.join(data_dir, f"{split}_{size}.npz")


def load_split(data_dir: str, split: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path(data_dir, split, size)) as d:
        return d["x"], d["y"]


def rgb565_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """Quantize uint8 RGB the way the device sees it (RGB565 on the wire, expanded to 8 bit)."""
    rgb = rgb.astype(np.uint8)
    r5, g6, b5 = rgb[..., 0] >> 3, rgb[..., 1] >> 2, rgb[..., 2] >> 3
    out = np.empty_like(rgb)
    out[..., 0] = (r5 << 3) | (r5 >> 2)
    out[..., 1] = (g6 << 2) | (g6 >> 4)
    out[..., 2] = (b5 << 3) | (b5 >> 2)
    return out


def to_model_input(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (N, H, W, 3) -> float32 model input in [-1, 1], as on the device."""
    return rgb565_roundtrip(rgb).astype(np.float32) / 127.5 - 1.0


def class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    return {i: float(len(y) / (num_classes * c)) for i, c in enumerate(counts)}
