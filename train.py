#!/usr/bin/env python3
"""Train the wildfire classifier: frozen-backbone head training, then fine-tuning."""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

from keras_model import build_model
from wildfire_data import class_weights, load_config, load_split, to_model_input


def rgb565_roundtrip_tf(x: tf.Tensor) -> tf.Tensor:
    """Float 0..255 RGB -> the 8-bit values the device reconstructs from RGB565."""
    x = tf.clip_by_value(tf.round(x), 0.0, 255.0)
    bits = tf.constant([5.0, 6.0, 5.0])
    hi = tf.floor(x / tf.pow(2.0, 8.0 - bits))
    return hi * tf.pow(2.0, 8.0 - bits) + tf.floor(hi / tf.pow(2.0, 2.0 * bits - 8.0))


def make_train_dataset(x: np.ndarray, y: np.ndarray, size: int, batch: int, seed: int):
    def augment(img, label):
        img = tf.cast(img, tf.float32)
        src = tf.shape(img)[0]
        scale = tf.random.uniform([], 0.6, 1.0)
        side = tf.cast(tf.cast(src, tf.float32) * tf.sqrt(scale), tf.int32)
        img = tf.image.random_crop(img, [side, side, 3])
        img = tf.image.resize(img, [size, size], method="area")
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15 * 255.0)
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.75, 1.25)
        img = tf.image.random_hue(img, 0.02)
        return rgb565_roundtrip_tf(img) / 127.5 - 1.0, label

    return (
        tf.data.Dataset.from_tensor_slices((x, y))
        .shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch)
        .prefetch(tf.data.AUTOTUNE)
    )


def plot_history(histories: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = {k: sum((hist[k] for hist in histories), []) for k in histories[0]}
    head_epochs = len(histories[0]["loss"])
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in ((axes[0], "loss", "Cross-entropy"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(epochs, h[key], label="train")
        ax.plot(epochs, h[f"val_{key}"], label="val")
        ax.axvline(head_epochs + 0.5, color="gray", linestyle="--", label="start fine-tuning")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training plots to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir, model_dir = cfg["paths"]["data_dir"], cfg["paths"]["model_dir"]
    size, fit = cfg["data"]["input_size"], cfg["fit"]
    classes = cfg["model"]["classes"]
    seed = cfg["data"]["seed"]
    os.makedirs(model_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    x_train, y_train = load_split(data_dir, "train", cfg["data"]["train_cache_size"])
    x_val, y_val = load_split(data_dir, "val", size)
    train_ds = make_train_dataset(x_train, y_train, size, fit["batch_size"], seed)
    val_data = (to_model_input(x_val), y_val)
    weights = class_weights(y_train, len(classes))
    print(f"Train {len(y_train)}, val {len(y_val)}, class weights {weights}")

    model, backbone = build_model(
        input_size=size,
        num_classes=len(classes),
        alpha=cfg["model"]["alpha"],
        dropout=cfg["model"]["dropout"],
    )

    backbone.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(fit["head_learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    head = model.fit(train_ds, validation_data=val_data, epochs=fit["head_epochs"],
                     class_weight=weights, verbose=2)

    backbone.trainable = True
    steps = fit["finetune_epochs"] * len(train_ds)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.CosineDecay(fit["finetune_learning_rate"], steps)
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    keras_path = os.path.join(model_dir, "wildfire_classifier.keras")
    finetune = model.fit(
        train_ds,
        validation_data=val_data,
        epochs=fit["finetune_epochs"],
        class_weight=weights,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=fit["early_stopping_patience"],
                                             restore_best_weights=True),
        ],
    )
    model.save(keras_path)
    print(f"Saved {keras_path}")

    plot_history([head.history, finetune.history],
                 os.path.join(model_dir, "training_history.png"))
    _, val_acc = model.evaluate(*val_data, verbose=0)
    print(f"Best validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
