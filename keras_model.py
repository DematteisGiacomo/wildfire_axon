"""MobileNetV1 wildfire classifier (fire / no fire) for the nRF Axon NPU."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


def _freeze_batch_norm(model: Model) -> None:
    """Keep ImageNet batch norm statistics; 1,887 images are too few to re-estimate them."""
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def set_backbone_trainable(model: Model, trainable: bool) -> None:
    for layer in model.layers:
        if not layer.name.startswith("head_"):
            layer.trainable = trainable
    _freeze_batch_norm(model)


def build_model(
    input_size: int = 96,
    num_classes: int = 2,
    alpha: float = 0.25,
    dropout: float = 0.3,
) -> Model:
    """MobileNetV1 with ImageNet weights and a 2-class head.

    Input is RGB scaled to [-1, 1] (pixel / 127.5 - 1), as the ImageNet weights expect.
    """
    backbone = tf.keras.applications.MobileNet(
        input_shape=(input_size, input_size, 3), alpha=alpha, include_top=False,
        weights="imagenet")
    x = layers.GlobalAveragePooling2D(name="head_pool")(backbone.output)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="head_fire_class")(x)
    model = Model(backbone.input, outputs, name="wildfire_classifier")
    _freeze_batch_norm(model)
    return model


def load_model(path: str) -> Model:
    return tf.keras.models.load_model(path)


def representative_input(x: np.ndarray):
    def generator():
        for i in range(len(x)):
            yield [x[i : i + 1].astype(np.float32)]

    return generator
