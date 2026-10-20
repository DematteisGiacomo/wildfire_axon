"""MobileNetV1 wildfire classifier (fire / no fire) for the nRF Axon NPU."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


def build_model(
    input_size: int = 96,
    num_classes: int = 2,
    alpha: float = 0.25,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> tuple[Model, Model]:
    """Return (model, backbone); the backbone is exposed to freeze it for head training.

    Input is RGB scaled to [-1, 1] (pixel / 127.5 - 1), as the ImageNet weights expect.
    """
    backbone = tf.keras.applications.MobileNet(
        input_shape=(input_size, input_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
    )

    inputs = layers.Input(shape=(input_size, input_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="fire_class")(x)
    return Model(inputs, outputs, name="wildfire_classifier"), backbone


def load_model(path: str) -> Model:
    return tf.keras.models.load_model(path)


def representative_input(x: np.ndarray):
    def generator():
        for i in range(len(x)):
            yield [x[i : i + 1].astype(np.float32)]

    return generator
