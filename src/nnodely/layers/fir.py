"""Finite impulse response layer."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


class Fir(Layer):
    """
    Linear temporal mixing over the time axis followed by a dense projection.

    Input:  (batch, time, in_features)
    Output: (batch, 1, out_features)
    """

    node_type = "Fir"

    def __init__(self, out_features: int, use_bias: bool = True, name=None):
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        super().__init__(
            name=name, out_features=self.out_features, use_bias=self.use_bias
        )

    def output_shape(self, seqs, times, dims):
        seq = seqs[0]
        in_dim = dims[0]
        if len(seq) != 0:
            raise NotImplementedError(
                "This proposal handles seq=() streams first; extend here for nested sequence dims."
            )
        if len(in_dim) != 1:
            raise ValueError(
                f"Fir currently expects a single feature axis, got dim={in_dim}"
            )
        return seq, 1, (self.out_features,)

    def build_layer(self):
        out_features = self.out_features
        use_bias = self.use_bias

        class _FirImpl(keras.layers.Layer):
            def __init__(self, name=None):
                super().__init__(name=name)
                self.flatten = keras.layers.Flatten()
                self.proj = keras.layers.Dense(out_features, use_bias=use_bias)
                self.reshape = keras.layers.Reshape((1, out_features))

            def call(self, x):
                x = self.flatten(x)
                x = self.proj(x)
                return self.reshape(x)

        self._layer = _FirImpl(name=self.name)
        return self._layer

    def call(self, x):
        return self._layer(x)
