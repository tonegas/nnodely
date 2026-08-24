from __future__ import annotations

import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class LinearImpl(keras.layers.Layer):
    """
    Linear projection along the dim axis.

    Input convention:
        [batch, in_dim, time, *seq]

    Output convention:
        [batch, out_dim, time, *seq]
    """

    def __init__(
        self,
        out_features: int = 1,
        use_bias: bool = True,
        name=None,
        initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(name=name)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.proj = keras.layers.Dense(
            self.out_features,
            use_bias=self.use_bias,
            kernel_initializer=initializer,
            bias_initializer=bias_initializer,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features,
                "use_bias": self.use_bias,
                "initializer": self.proj.kernel_initializer,
                "bias_initializer": self.proj.bias_initializer,
            }
        )
        return config

    def call(self, x):
        rank = len(x.shape)

        # [batch, dim, time, *seq] -> [batch, time, *seq, dim]
        perm = [0] + list(range(2, rank)) + [1]
        x = keras.ops.transpose(x, perm)

        # Dense acts on last axis: dim -> out_features
        x = self.proj(x)

        # [batch, time, *seq, out_features] -> [batch, out_features, time, *seq]
        rank_y = len(x.shape)
        inv_perm = [0, rank_y - 1] + list(range(1, rank_y - 1))
        x = keras.ops.transpose(x, inv_perm)

        return x


class Linear(Layer):
    """
    Linear projection along the dim dimension.

    Input:
        [batch, in_features, time, *seq]

    Output:
        [batch, out_features, time, *seq]
    """

    def __init__(
        self,
        out_features: int = 1,
        use_bias: bool = True,
        name=None,
        initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
        **kwargs,
    ):
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        super().__init__(
            name=name,
            out_features=self.out_features,
            use_bias=self.use_bias,
            initializer=initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )

    def output_shape(self, *inputs):
        x = inputs[0]

        if len(x.dim) != 1:
            raise ValueError(
                f"Linear currently expects a single dim axis, got dim={x.dim}"
            )

        return (self.out_features,), x.time, x.seq

    def build_layer(self):
        return LinearImpl(
            out_features=self.out_features,
            use_bias=self.use_bias,
            name=self.name,
            initializer=self.initializer,
            bias_initializer=self.bias_initializer,
        )

    def get_config(self):
        return {
            "name": self.name,
            "out_features": self.out_features,
            "use_bias": self.use_bias,
        }

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
