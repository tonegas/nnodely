"""Finite impulse response layer."""

from __future__ import annotations

import math

import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class FirImpl(keras.layers.Layer):
    """A filter over the time axis of every element of a window, at every
    step of a sequence.

    Each element of the dim axes has the time axis projected onto
    ``out_features`` channels, with a kernel of its own or, with
    ``shared_kernel``, one kernel shared by all of them; elements are never
    mixed. ``(batch, *dim, time, *seq)`` becomes
    ``(batch, out_features, *dim, 1, *seq)``, the channel axis dropped when
    there is one channel and a scalar dim ``(1,)`` dropped when there are more.
    """

    def __init__(
        self,
        out_features,
        use_bias=True,
        shared_kernel=False,
        seq_rank=0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.shared_kernel = bool(shared_kernel)
        self.seq_rank = int(seq_rank)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features,
                "use_bias": self.use_bias,
                "shared_kernel": self.shared_kernel,
                "seq_rank": self.seq_rank,
            }
        )
        return config

    def build(self, input_shape):
        first_seq = len(input_shape) - self.seq_rank
        dim = tuple(int(axis) for axis in input_shape[1 : first_seq - 1])
        time = int(input_shape[first_seq - 1])
        # A scalar has a single element: its kernel is the shared one.
        elements = () if self.shared_kernel or dim == (1,) else dim
        # Glorot over the (time, out_features) map of one element.
        limit = math.sqrt(6.0 / (time + self.out_features))
        self.kernel = self.add_weight(
            shape=(*elements, time, self.out_features),
            initializer=keras.initializers.RandomUniform(-limit, limit),
            name="kernel",
        )
        self.bias = (
            self.add_weight(
                shape=(*elements, self.out_features), initializer="zeros", name="bias"
            )
            if self.use_bias
            else None
        )
        super().build(input_shape)

    def call(self, x):
        rank = len(x.shape)
        first_seq = rank - self.seq_rank
        dim = tuple(int(axis) for axis in x.shape[1 : first_seq - 1])

        # [batch, *dim, time, *seq] -> [batch, *seq, *dim, 1, time]: the time
        # axis of every element is multiplied by its kernel, so neither the
        # batch nor a sequence left dynamic has to be reshaped.
        x = keras.ops.transpose(x, [0, *range(first_seq, rank), *range(1, first_seq)])
        y = keras.ops.matmul(keras.ops.expand_dims(x, -2), self.kernel)
        y = keras.ops.squeeze(y, -2)
        if self.bias is not None:
            y = y + self.bias

        # [batch, *seq, *dim, out] -> [batch, out, *dim, 1, *seq]
        seq_end = 1 + self.seq_rank
        dim_end = seq_end + len(dim)
        y = keras.ops.transpose(
            y, [0, dim_end, *range(seq_end, dim_end), *range(1, seq_end)]
        )
        y = keras.ops.expand_dims(y, axis=2 + len(dim))
        if self.out_features == 1:
            return keras.ops.squeeze(y, axis=1)
        if dim == (1,):
            return keras.ops.squeeze(y, axis=2)
        return y


class Fir(Layer):
    """
    Filter over the time axis of every element of a window: each element is
    projected onto ``out_features`` channels, one time step long. Elements
    are never mixed: each has a kernel of its own, or with
    ``shared_kernel=True`` they all share one. Sequence axes are kept, the
    filter applied at each of their steps::

        input:  (batch, *dim, time, *seq)
        output: (batch, out_features, *dim, 1, *seq)

    The channel axis is added only for ``out_features > 1``, and a scalar dim
    ``(1,)`` gives way to it::

        Fir(out_features=1)  on D=(2, 3) -> D=(2, 3)
        Fir(out_features=4)  on D=(2, 3) -> D=(4, 2, 3)
        Fir(out_features=4)  on D=(1,)   -> D=(4,)
    """

    def __init__(
        self,
        out_features: int,
        use_bias: bool = True,
        shared_kernel: bool = False,
        name=None,
    ):
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.shared_kernel = bool(shared_kernel)
        super().__init__(
            name=name,
            out_features=self.out_features,
            use_bias=self.use_bias,
            shared_kernel=self.shared_kernel,
        )

    def output_shape(self, *inputs):
        # Declared rather than probed with a dummy tensor, which a sequence
        # axis left dynamic cannot be given.
        dim = tuple(inputs[0].dim)
        if self.out_features == 1:
            out_dim = dim
        elif dim == (1,):
            out_dim = (self.out_features,)
        else:
            out_dim = (self.out_features, *dim)
        return out_dim, 1, tuple(inputs[0].seq)

    def build_layer(self):
        return FirImpl(
            out_features=self.out_features,
            use_bias=self.use_bias,
            shared_kernel=self.shared_kernel,
            seq_rank=len(self.preds[0].seq),  # type: ignore
            name=self.name,
        )

    @property
    def kernel(self):
        """``[*dim, time, out_features]``, or ``[time, out_features]`` for a
        scalar input or a shared kernel."""
        if self._layer is not None:
            return self._layer.kernel
        return None

    @property
    def bias(self):
        """``[*dim, out_features]``, or ``[out_features]`` for a scalar input
        or a shared kernel."""
        if self._layer is not None and self._layer.use_bias:
            return self._layer.bias
        return None
