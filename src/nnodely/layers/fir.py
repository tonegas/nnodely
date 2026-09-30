"""Finite impulse response layer."""

from __future__ import annotations

import math

import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class FirImpl(keras.layers.Layer):
    """One dense projection of a whole window, at every step of a sequence.

    The dim and time axes of a sample are projected together onto
    ``out_features``, separately at each position of the ``seq_rank`` trailing
    sequence axes: ``(batch, *dim, time, *seq)`` becomes
    ``(batch, out_features, 1, *seq)``.
    """

    def __init__(self, out_features, use_bias=True, seq_rank=0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.seq_rank = int(seq_rank)
        # Holds the kernel and bias: one projection of the flattened window.
        self.proj = keras.layers.Dense(self.out_features, use_bias=self.use_bias)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features,
                "use_bias": self.use_bias,
                "seq_rank": self.seq_rank,
            }
        )
        return config

    def build(self, input_shape):
        window = input_shape[1 : len(input_shape) - self.seq_rank]
        self.proj.build((None, math.prod(int(axis) for axis in window)))
        super().build(input_shape)

    def call(self, x):
        rank = len(x.shape)
        first_seq = rank - self.seq_rank
        window = tuple(int(axis) for axis in x.shape[1:first_seq])

        # [batch, *dim, time, *seq] -> [batch, *seq, *dim, time]: the window is
        # contracted with the kernel laid out like it, so neither the batch nor
        # a sequence left dynamic has to be reshaped.
        x = keras.ops.transpose(x, [0, *range(first_seq, rank), *range(1, first_seq)])
        kernel = keras.ops.reshape(self.proj.kernel, (*window, self.out_features))
        y = keras.ops.tensordot(x, kernel, axes=len(window))
        if self.use_bias:
            y = y + self.proj.bias

        # [batch, *seq, out] -> [batch, out, 1, *seq]
        last = len(y.shape) - 1
        y = keras.ops.transpose(y, [0, last, *range(1, last)])
        return keras.ops.expand_dims(y, axis=2)


class Fir(Layer):
    """
    Dense projection of a whole window: its dim and time axes are mixed into
    ``out_features`` values, one time step long. Sequence axes are kept, the
    projection applied at each of their steps::

        input:  (batch, *dim, time, *seq)
        output: (batch, out_features, 1, *seq)
    """

    def __init__(self, out_features: int, use_bias: bool = True, name=None):
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        super().__init__(
            name=name, out_features=self.out_features, use_bias=self.use_bias
        )

    def output_shape(self, *inputs):
        # Declared rather than probed with a dummy tensor, which a sequence
        # axis left dynamic cannot be given.
        return (self.out_features,), 1, tuple(inputs[0].seq)

    def build_layer(self):
        return FirImpl(
            out_features=self.out_features,
            use_bias=self.use_bias,
            seq_rank=len(self.preds[0].seq),  # type: ignore
            name=self.name,
        )

    @property
    def kernel(self):
        if self._layer is not None:
            return self._layer.proj.kernel
        return None

    @property
    def bias(self):
        if self._layer is not None and self._layer.use_bias:
            return self._layer.proj.bias
        return None
