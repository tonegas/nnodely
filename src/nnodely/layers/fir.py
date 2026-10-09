"""Finite impulse response layer."""

from __future__ import annotations

import keras

from nnodely.layers.parameter import Parameter, _check_weight_size, _ParameterWeights


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

    It owns no weight: it is called on ``[x, kernel]`` or ``[x, kernel,
    bias]``, the values of the Parameters holding them, and reads their first
    sample.
    """

    def __init__(
        self, out_features, shared_kernel=False, seq_rank=0, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.out_features = int(out_features)
        self.shared_kernel = bool(shared_kernel)
        self.seq_rank = int(seq_rank)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features,
                "shared_kernel": self.shared_kernel,
                "seq_rank": self.seq_rank,
            }
        )
        return config

    def build(self, input_shape):
        x_shape = input_shape[0]
        first_seq = len(x_shape) - self.seq_rank
        dim = tuple(int(axis) for axis in x_shape[1 : first_seq - 1])
        time = int(x_shape[first_seq - 1])
        # A scalar has a single element: its kernel is the shared one.
        elements = () if self.shared_kernel or dim == (1,) else dim
        self.kernel_shape = (*elements, time, self.out_features)
        self.bias_shape = (*elements, self.out_features)
        for label, shape, expected in zip(
            ("kernel", "bias"), input_shape[1:], (self.kernel_shape, self.bias_shape)
        ):
            _check_weight_size(self.name, label, shape, expected)
        super().build(input_shape)

    def call(self, inputs):
        x, kernel, *bias = inputs
        kernel = keras.ops.reshape(kernel[0], self.kernel_shape)

        rank = len(x.shape)
        first_seq = rank - self.seq_rank
        dim = tuple(int(axis) for axis in x.shape[1 : first_seq - 1])

        # [batch, *dim, time, *seq] -> [batch, *seq, *dim, 1, time]: the time
        # axis of every element is multiplied by its kernel, so neither the
        # batch nor a sequence left dynamic has to be reshaped.
        x = keras.ops.transpose(x, [0, *range(first_seq, rank), *range(1, first_seq)])
        y = keras.ops.matmul(keras.ops.expand_dims(x, -2), kernel)
        y = keras.ops.squeeze(y, -2)
        if bias:
            y = y + keras.ops.reshape(bias[0][0], self.bias_shape)

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


class Fir(_ParameterWeights):
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

    ``kernel`` is ``[*dim, time, out_features]`` and ``bias``
    ``[*dim, out_features]`` (without ``*dim`` for a scalar input or a shared
    kernel). Both are Parameters, predecessors of the layer: assigning or
    training them changes the filter. Each is given as a :class:`Parameter`,
    of the shape of the weight (axes of size 1 aside), or as a Keras
    initializer, by name or object, from which the layer makes one, named
    ``"<name>_kernel"`` or ``"<name>_bias"``, drawing the filter and the bias
    of each element alone, as if it were a scalar. ``bias=True`` makes a bias
    of zeros, and ``bias=False`` leaves it out::

        W = Parameter("W", dim=(3, 5, 4))   # (*dim, time, out_features)
        Fir(out_features=4, kernel=W, bias=False)([x.sw(5)])  # x with dim=3
    """

    def __init__(
        self,
        out_features: int,
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        shared_kernel: bool = False,
        name=None,
    ):
        self.out_features = int(out_features)
        self.shared_kernel = bool(shared_kernel)
        super().__init__(
            name,
            kernel,
            bias,
            out_features=self.out_features,
            shared_kernel=self.shared_kernel,
        )

    def _weight_shapes(self, x):
        dim = tuple(x.dim)
        elements = () if self.shared_kernel or dim == (1,) else dim
        return (*elements, x.time, self.out_features), (*elements, self.out_features)

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
            shared_kernel=self.shared_kernel,
            seq_rank=len(self.preds[0].seq),  # type: ignore
            name=self.name,
        )
