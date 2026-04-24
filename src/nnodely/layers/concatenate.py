"""Concatenate layer."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


class Concatenate(Layer):
    """Concatenate multiple streams along a chosen axis of stream shape."""

    def __init__(self, axis: int = -1, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def _resolve_axis(self, rank: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(
                f"{self.name}: axis {self.axis} is out of bounds for rank {rank}."
            )
        return axis

    def __call__(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = tuple(inputs[0])
        return super().__call__(*inputs)

    def output_shape(self, seqs, times, dims):
        shapes = [
            tuple(seq) + (int(time),) + tuple(dim)
            for seq, time, dim in zip(seqs, times, dims)
        ]
        ref = shapes[0]
        rank = len(ref)
        axis = self._resolve_axis(rank)

        out = list(ref)
        for shape in shapes[1:]:
            if len(shape) != rank:
                raise ValueError(
                    f"{self.name}: all inputs must have the same rank, got {rank} and {len(shape)}."
                )

            for idx, (left, right) in enumerate(zip(out, shape)):
                if idx == axis:
                    continue
                if left != right:
                    raise ValueError(
                        f"{self.name}: input shapes differ on axis {idx}: {left} vs {right}."
                    )

            out[axis] += shape[axis]

        seq_rank = len(seqs[0])
        out_seq = tuple(out[:seq_rank])
        out_time = out[seq_rank]
        out_dim = tuple(out[seq_rank + 1 :])
        return out_seq, out_time, out_dim

    def build_layer(self):
        self._layer = keras.layers.Concatenate(axis=self.axis, name=self.name)
        return self._layer
