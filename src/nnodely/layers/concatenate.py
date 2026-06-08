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

    def output_shape(self, *inputs):
        if not inputs:
            raise ValueError(f"{self.name}: Concatenate requires at least one input.")

        ref_seq = inputs[0].seq
        ref_time = inputs[0].time
        ref_dim = inputs[0].dim

        dim_rank = len(ref_dim)
        axis = self._resolve_dim_axis(dim_rank)

        out_dim = list(ref_dim)

        for inp in inputs[1:]:
            seq = inp.seq
            time = inp.time
            dim = inp.dim

            if seq != ref_seq:
                raise ValueError(
                    f"{self.name}: all inputs must have the same seq shape, "
                    f"got {ref_seq} and {seq}."
                )

            if time != ref_time:
                raise ValueError(
                    f"{self.name}: all inputs must have the same time dimension, "
                    f"got {ref_time} and {time}."
                )

            if len(dim) != dim_rank:
                raise ValueError(
                    f"{self.name}: all inputs must have the same dim rank, "
                    f"got {dim_rank} and {len(dim)}."
                )

            out_dim[axis] += dim[axis]

        return tuple(out_dim), ref_time, ref_seq

    def build_layer(self):
        keras_axis = self._resolve_dim_axis(len(self.dim)) + 1
        return keras.layers.Concatenate(axis=keras_axis, name=self.name)


class TimeConcatenate(Layer):
    """Concatenate multiple streams the time dimension"""

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        if not inputs:
            raise ValueError(
                f"{self.name}: TimeConcatenate requires at least one input."
            )

        ref_seq = inputs[0].seq
        ref_dim = inputs[0].dim

        for inp in inputs[1:]:
            if inp.seq != ref_seq:
                raise ValueError(
                    f"{self.name}: all inputs must have the same seq shape, "
                    f"got {ref_seq} and {inp.seq}."
                )
            if inp.dim != ref_dim:
                raise ValueError(
                    f"{self.name}: all inputs must have the same dim shape, "
                    f"got {ref_dim} and {inp.dim}."
                )

        time_out = sum(inp.time for inp in inputs)
        return ref_dim, time_out, ref_seq

    def build_layer(self):
        keras_axis = 1 + len(self.dim)
        return keras.layers.Concatenate(axis=keras_axis, name=self.name)
