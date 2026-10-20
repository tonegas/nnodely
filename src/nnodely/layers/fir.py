"""Finite impulse response layer."""

from __future__ import annotations

import keras

from nnodely.core.shape import BATCH_AXIS, Shape, transpose
from nnodely.layers.parameter import Parameter, _check_weight_size, _ParameterWeights


@keras.saving.register_keras_serializable(package="nnodely")
class FirImpl(keras.layers.Layer):
    """A filter over the inner time, the window, of every element, at every
    step of the outer time axes.

    Each element of the dim axes has the inner time projected onto
    ``features`` channels, with a kernel of its own or, with
    ``shared_kernel``, one kernel shared by all of them; elements are never
    mixed. ``(batch, *outer, window, *dim)`` becomes
    ``(batch, *outer, 1, *dim, features)``, the channel axis dropped when
    there is one channel and a scalar dim ``(1,)`` dropped when there are more.

    It owns no weight: it is called on ``[x, kernel]`` or ``[x, kernel,
    bias]``, the values of the Parameters holding them, and reads their first
    sample.
    """

    def __init__(self, features, shared_kernel=False, time_rank=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.features = int(features)
        self.shared_kernel = bool(shared_kernel)
        self.time_rank = int(time_rank)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "shared_kernel": self.shared_kernel,
                "time_rank": self.time_rank,
            }
        )
        return config

    def build(self, input_shape):
        shape = Shape.from_tensor_shape(input_shape[0][1:], time_rank=self.time_rank)
        dim = tuple(int(axis) for axis in shape.dim)
        # A scalar has a single element: its kernel is the shared one.
        elements = () if self.shared_kernel or dim == (1,) else dim
        self.kernel_shape = (*elements, shape.inner_time, self.features)
        self.bias_shape = (*elements, self.features)
        for label, shape, expected in zip(
            ("kernel", "bias"), input_shape[1:], (self.kernel_shape, self.bias_shape)
        ):
            _check_weight_size(self.name, label, shape, expected)
        super().build(input_shape)

    def call(self, inputs):
        x, kernel, *bias = inputs
        kernel = keras.ops.reshape(kernel[0], self.kernel_shape)

        shape = Shape.from_tensor_shape(x.shape[1:], time_rank=self.time_rank)
        dim = tuple(int(axis) for axis in shape.dim)
        outer = shape.time_axes[:-1]

        # (batch, *outer, *dim, 1, window): the inner time of every element is
        # multiplied by its kernel, so neither the batch nor an outer time
        # left dynamic has to be reshaped.
        x = transpose(x, [BATCH_AXIS, *outer, *shape.dim_axes, shape.inner_time_axis])
        y = keras.ops.matmul(keras.ops.expand_dims(x, -2), kernel)
        y = keras.ops.squeeze(y, -2)
        if bias:
            y = y + keras.ops.reshape(bias[0][0], self.bias_shape)

        # (batch, *outer, *dim, out) is laid out as the output already, which
        # drops the channel axis when there is one channel and a scalar dim
        # when there are more; it lacks only its inner time, one step long.
        if self.features == 1:
            y = keras.ops.squeeze(y, len(y.shape) - 1)
        elif dim == (1,):
            y = keras.ops.squeeze(y, 1 + len(outer))
        return keras.ops.expand_dims(y, 1 + len(outer))


class Fir(_ParameterWeights):
    """
    Filter over the inner time, the window, of every element: each element
    is projected onto ``features`` channels, one time step long. Elements
    are never mixed: each has a kernel of its own, or with
    ``shared_kernel=True`` they all share one. The outer time axes are kept,
    the filter applied at each of their steps::

        input:  (batch, *outer, window, *dim)
        output: (batch, *outer, 1, *dim, features)

    The channel axis, the last, is added only for ``features > 1``, and a
    scalar dim ``(1,)`` gives way to it::

        Fir(features=1)  on D=(2, 3) -> D=(2, 3)
        Fir(features=4)  on D=(2, 3) -> D=(2, 3, 4)
        Fir(features=4)  on D=(1,)   -> D=(4,)

    ``kernel`` is ``[*dim, window, features]`` and ``bias``
    ``[*dim, features]`` (without ``*dim`` for a scalar input or a shared
    kernel). Both are Parameters, predecessors of the layer: assigning or
    training them changes the filter. Each is given as a :class:`Parameter`,
    of the shape of the weight (axes of size 1 aside), or as a Keras
    initializer, by name or object, from which the layer makes one, named
    ``"<name>_kernel"`` or ``"<name>_bias"``, drawing the filter and the bias
    of each element alone, as if it were a scalar. ``bias=True`` makes a bias
    of zeros, and ``bias=False`` leaves it out::

        W = Parameter("W", dim=(3, 5, 4))   # (*dim, window, features)
        Fir(features=4, kernel=W, bias=False)([x.sw(5)])  # x with dim=3
    """

    def __init__(
        self,
        features: int,
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        shared_kernel: bool = False,
        name=None,
    ):
        self.features = int(features)
        self.shared_kernel = bool(shared_kernel)
        super().__init__(
            name,
            kernel,
            bias,
            features=self.features,
            shared_kernel=self.shared_kernel,
        )

    def _weight_shapes(self, x):
        if not x.shape.time:
            raise ValueError(
                f"{self.name}: a Fir filters a window of samples, and "
                f"{x.name!r} has no time axis; take one with sw() or last()."
            )
        dim = tuple(x.dim)
        elements = () if self.shared_kernel or dim == (1,) else dim
        return (*elements, x.shape.inner_time, self.features), (
            *elements,
            self.features,
        )

    def _draw_shapes(self, kernel_shape, bias_shape):
        # The filter and the bias of each element alone, as for a scalar.
        return kernel_shape[-2:], bias_shape[-1:]

    def output_shape(self, *inputs):
        # Declared rather than probed with a dummy tensor, which a time axis
        # left dynamic cannot be given.
        dim = tuple(inputs[0].dim)
        if self.features == 1:
            out_dim = dim
        elif dim == (1,):
            out_dim = (self.features,)
        else:
            out_dim = (*dim, self.features)
        # The window becomes one sample; the outer time axes are kept.
        shape = inputs[0].shape
        return shape.replace(dim=out_dim, time=(*shape.time[:-1], 1))

    def build_layer(self):
        return FirImpl(
            features=self.features,
            shared_kernel=self.shared_kernel,
            time_rank=self.preds[0].shape.time_rank,  # type: ignore[union-attr]
            name=self.name,
        )
