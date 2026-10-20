#!/usr/bin/env python3
"""Download The Wildfire Dataset and cache it as small square RGB arrays.

Images are center-cropped to a square and box-downscaled, the same way the UART
feeder (feed_images_uart.py) prepares frames for the device.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from wildfire_data import SPLITS, cache_path, load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_square(path: Path, size: int) -> np.ndarray:
    img = Image.open(path)
    img.draft("RGB", (size * 2, size * 2))  # fast JPEG DCT downscale, keeps >= 2x size
    img = ImageOps.exif_transpose(img).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return np.asarray(img.resize((size, size), Image.BOX), dtype=np.uint8)


def _load(job: tuple[Path, int]) -> np.ndarray:
    return load_square(*job)


def find_root(download_dir: Path) -> Path:
    for p in [download_dir, *download_dir.rglob("*")]:
        if p.is_dir() and all((p / s).is_dir() for s in SPLITS):
            return p
    raise SystemExit(f"no {'/'.join(SPLITS)} folders found under {download_dir}")


def list_split(root: Path, split: str, classes: list[str]) -> tuple[list[Path], np.ndarray]:
    files, labels = [], []
    for label, name in enumerate(classes):
        folder = root / split / name.replace("_", "")
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMAGE_SUFFIXES:
                files.append(f)
                labels.append(label)
    return files, np.asarray(labels, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset-dir", type=Path,
                    help="Already downloaded dataset (default: download with kagglehub)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["paths"]["data_dir"]
    classes = cfg["model"]["classes"]
    sizes = sorted({cfg["data"]["input_size"], cfg["data"]["train_cache_size"]})
    os.makedirs(data_dir, exist_ok=True)

    if args.dataset_dir is None:
        import kagglehub

        args.dataset_dir = Path(kagglehub.dataset_download(cfg["data"]["kaggle_dataset"]))
    root = find_root(args.dataset_dir)
    print(f"Dataset: {root}")

    with Pool(args.jobs) as pool:
        for split in SPLITS:
            files, y = list_split(root, split, classes)
            counts = ", ".join(f"{c}={int((y == i).sum())}" for i, c in enumerate(classes))
            for size in sizes:
                x = np.stack(pool.map(_load, [(f, size) for f in files], chunksize=8))
                np.savez_compressed(cache_path(data_dir, split, size), x=x, y=y)
                print(f"{split}: {len(y)} images ({counts}) -> {cache_path(data_dir, split, size)}")

            if split == "test":
                export_demo_images(files, y, classes, cfg, pool)


def export_demo_images(files: list[Path], y: np.ndarray, classes: list[str], cfg: dict,
                       pool: Pool) -> None:
    """Save a few held-out test images (384x384) to feed to the device over UART."""
    demo_dir = Path(cfg["paths"]["data_dir"]) / "demo"
    demo_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg["data"]["seed"])
    count = cfg["data"]["demo_images_per_class"]
    for label, name in enumerate(classes):
        picks = rng.choice(np.flatnonzero(y == label), size=count, replace=False)
        for i, img in zip(picks, pool.map(_load, [(files[i], 384) for i in picks])):
            Image.fromarray(img).save(demo_dir / f"{name}_{files[i].stem[:40]}.jpg", quality=92)
    print(f"Demo images: {demo_dir}")


if __name__ == "__main__":
    main()
