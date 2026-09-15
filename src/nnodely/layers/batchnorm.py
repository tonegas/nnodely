"""Batch normalization layer wrapper for nnodely."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


class BatchNorm(Layer):
    """
    Wrapper for keras.layers.BatchNormalization.

    Input:
        [batch, *dim, time, *seq]

    Output:
        [batch, *dim, time, *seq]

    Statistics are accumulated per feature, so the normalization axis defaults
    to the dim axis and not to the last axis as in Keras.
    """

    def __init__(
        self,
        axis: int = 1,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        name=None,
    ):
        self.axis = int(axis)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.center = bool(center)
        self.scale = bool(scale)
        super().__init__(
            name=name,
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
        )

    def output_shape(self, *inputs):
        # Normalization preserves every axis, including a dynamic sequence axis
        # that cannot be probed with a dummy tensor.
        return inputs[0].shape.dimensions

    def build_layer(self):
        return keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
        }
