"""Concatenate layer."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


class Concatenate(Layer):
    """Concatenate multiple streams along a chosen axis of the dim shape."""

    def __init__(self, axis: int = 0, name=None):
        self.axis = axis
        super().__init__(name=name, axis=self.axis)

    def _resolve_dim_axis(self, dim_rank: int) -> int:
        axis = self.axis
        if axis < 0 or axis >= dim_rank:
            raise ValueError(
                f"{self.name}: axis {self.axis} is out of bounds for dim rank {dim_rank}."
            )
        return axis

    def build_layer(self):
        keras_axis = self._resolve_dim_axis(len(self.dim)) + 1
        return keras.layers.Concatenate(axis=keras_axis, name=self.name)


class TimeConcatenate(Layer):
    """Concatenate multiple streams the time dimension"""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        keras_axis = 1 + len(self.dim)
        return keras.layers.Concatenate(axis=keras_axis, name=self.name)
