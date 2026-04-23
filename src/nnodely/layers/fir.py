"""Finite impulse response layer."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class FirImpl(keras.layers.Layer):
    def __init__(self, out_features, use_bias=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.flatten = keras.layers.Flatten()
        self.proj = keras.layers.Dense(self.out_features, use_bias=self.use_bias)
        self.reshape = keras.layers.Reshape((1, self.out_features))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features,
                "use_bias": self.use_bias,
            }
        )
        return config

    def build(self, input_shape):
        flat_shape = self.flatten.compute_output_shape(input_shape)
        self.proj.build(flat_shape)
        proj_shape = self.proj.compute_output_shape(flat_shape)
        self.reshape.build(proj_shape)
        super().build(input_shape)

    def call(self, x):
        x = self.flatten(x)
        x = self.proj(x)
        return self.reshape(x)


class Fir(Layer):
    """
    Linear temporal mixing over the time axis followed by a dense projection.

    Input:  (batch, time, in_features)
    Output: (batch, 1, out_features)
    """

    # node_type = "Fir"

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
        self._layer = FirImpl(
            out_features=self.out_features, use_bias=self.use_bias, name=self.name
        )
        return self._layer

    def call(self, *xs):
        return self._layer(xs)
