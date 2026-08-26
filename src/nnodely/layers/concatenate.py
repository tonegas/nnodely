"""Concatenate layer."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class ConcatenateImpl(keras.layers.Layer):
    def __init__(self, axis: int, input_rank: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)
        self.input_rank = int(input_rank)

    def call(self, xs):
        target_rank = max(len(x.shape) for x in xs)
        values = list(xs)
        for index, value in enumerate(values):
            while len(value.shape) < target_rank:
                value = keras.ops.expand_dims(value, axis=0)
            values[index] = value

        batch_offset = target_rank - self.input_rank
        return keras.ops.concatenate(values, axis=batch_offset + self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "input_rank": self.input_rank})
        return config


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
        axis = self._resolve_dim_axis(len(self.dim))
        return ConcatenateImpl(
            axis=axis,
            input_rank=self.shape.rank,
            name=self.name,
        )


class TimeConcatenate(Layer):
    """Concatenate multiple streams the time dimension"""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return ConcatenateImpl(
            axis=len(self.dim),
            input_rank=self.shape.rank,
            name=self.name,
        )
