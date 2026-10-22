from __future__ import annotations

import keras

from nnodely.layers.parameter import Parameter, _check_weight_size, _ParameterWeights


@keras.saving.register_keras_serializable(package="nnodely")
class LinearImpl(keras.layers.Layer):
    """
    Linear projection of the dim axes, ``x @ kernel + bias``, at every step
    of every time axis.

    Input convention:
        [batch, *time, *dim]

    Output convention, with ``axis=None``:
        [batch, *time, features]

    With ``axis=k`` only dim axis ``k`` is projected, and ``features``
    takes its place among the dim axes: with a matrix of its own at every
    position of the other dim axes, or with ``shared_kernel`` one for all.

    It owns no weight: it is called on ``[x, kernel]`` or ``[x, kernel,
    bias]``, the values of the Parameters holding them, and reads their first
    sample.
    """

    def __init__(
        self,
        features: int = 1,
        dim_rank: int = 1,
        axis=None,
        shared_kernel: bool = False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.features = int(features)
        self.dim_rank = int(dim_rank)
        self.axis = None if axis is None else int(axis)
        self.shared_kernel = bool(shared_kernel)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "dim_rank": self.dim_rank,
                "axis": self.axis,
                "shared_kernel": self.shared_kernel,
            }
        )
        return config

    def build(self, input_shape):
        # The dim axes are the last ones.
        dim = tuple(int(axis) for axis in input_shape[0][-self.dim_rank :])
        self.kernel_shape, self.bias_shape = _linear_weight_shapes(
            dim, self.features, self.axis, self.shared_kernel
        )
        for label, shape, expected in zip(
            ("kernel", "bias"), input_shape[1:], (self.kernel_shape, self.bias_shape)
        ):
            _check_weight_size(self.name, label, shape, expected)
        super().build(input_shape)

    def call(self, inputs):
        x, kernel, *bias = inputs
        kernel = keras.ops.reshape(kernel[0], self.kernel_shape)

        if self.axis is None:
            # The dim, the last axes of (batch, *time, *dim), contracted with
            # the first axes of the kernel: (batch, *time, out).
            y = keras.ops.tensordot(x, kernel, axes=self.dim_rank)
            if bias:
                y = y + keras.ops.reshape(bias[0][0], self.bias_shape)
            return y

        # The projected axis moved last, (batch, *time, *kept, 1, projected):
        # every position of the kept axes times its matrix, or the shared one.
        # The axes are addressed from the front, which ONNX needs.
        last = len(x.shape) - 1
        projected = last + 1 - self.dim_rank + self.axis
        if projected != last:
            x = keras.ops.moveaxis(x, projected, last)
        y = keras.ops.matmul(keras.ops.expand_dims(x, -2), kernel)
        y = keras.ops.squeeze(y, -2)
        if bias:
            y = y + keras.ops.reshape(bias[0][0], self.bias_shape)
        # out back in place of the projected axis.
        return y if projected == last else keras.ops.moveaxis(y, last, projected)


def _linear_weight_shapes(dim, features, axis, shared_kernel):
    """The kernel and bias shapes of a Linear on ``dim``."""
    if axis is None:
        return (*dim, features), (features,)
    kept = () if shared_kernel else (*dim[:axis], *dim[axis + 1 :])
    return (*kept, dim[axis], features), (*kept, features)


class Linear(_ParameterWeights):
    """
    Linear projection of the dim axes, ``x @ kernel + bias``, applied at every
    step of every time axis.

    With ``axis=None`` the whole dim is projected: every feature of a sample
    feeds every output::

        input:  (batch, *time, *dim)
        output: (batch, *time, features)
        kernel: (*dim, features)
        bias:   (features,)

    With ``axis=k`` only dim axis ``k`` is projected, and ``features``
    takes its place. Like the elements of a :class:`Fir`, every position of
    the other dim axes has a matrix and a bias of its own, or with
    ``shared_kernel=True`` they all share one. On ``D=(2, 3)`` with
    ``axis=1``, the output is ``D=(2, features)`` with a kernel
    ``(2, 3, features)`` and a bias ``(2, features)``, or with a
    shared kernel ``(3, features)`` and ``(features,)``.
    ``shared_kernel`` needs an ``axis``: without one, the whole dim is
    projected at once.

    ``kernel`` and ``bias`` are Parameters, predecessors of the layer:
    assigning or training them changes the projection. Each is given as a
    :class:`Parameter`, of the shape of the weight (axes of size 1 aside), or
    as a Keras initializer, by name or object, from which the layer makes
    one, named ``"<name>_kernel"`` or ``"<name>_bias"``; each matrix is drawn
    as one of the projected inputs by ``features``. ``bias=True`` draws
    the bias with ``"glorot_uniform"``, and ``bias=False`` leaves it out::

        W = Parameter("W", value=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        Linear(features=2, kernel=W, bias=False)([v])  # v with dim=3
    """

    _bias_initializer = "glorot_uniform"

    def __init__(
        self,
        features: int = 1,
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        axis: int | None = None,
        shared_kernel: bool = False,
        name=None,
    ):
        if shared_kernel and axis is None:
            raise ValueError(
                "Linear shared_kernel shares the kernel along the dim axes it "
                "does not project, so it needs an axis: without one, the whole "
                "dim is projected at once."
            )
        self.features = int(features)
        self.axis = None if axis is None else int(axis)
        self.shared_kernel = bool(shared_kernel)
        super().__init__(
            name,
            kernel,
            bias,
            features=self.features,
            axis=self.axis,
            shared_kernel=self.shared_kernel,
        )

    def _axis(self, dim) -> int | None:
        """``axis`` counted from the first dim axis, checked against ``dim``."""
        if self.axis is None:
            return None
        if not -len(dim) <= self.axis < len(dim):
            raise ValueError(
                f"{self.name}: axis {self.axis} is not an axis of the dim {dim}."
            )
        return self.axis % len(dim)

    def _weight_shapes(self, x):
        dim = tuple(x.dim)
        return _linear_weight_shapes(
            dim, self.features, self._axis(dim), self.shared_kernel
        )

    def _draw_shapes(self, kernel_shape, bias_shape):
        # Along an axis, the matrix and the bias of each position alone; the
        # whole-dim kernel as the Weight's default, one matrix.
        if self.axis is None:
            return None, None
        return kernel_shape[-2:], bias_shape[-1:]

    def output_shape(self, *inputs):
        # Declared rather than probed with dummy tensors, which would have to
        # stand in for the weights too.
        dim = tuple(inputs[0].dim)
        axis = self._axis(dim)
        out_dim = (
            (self.features,)
            if axis is None
            else (*dim[:axis], self.features, *dim[axis + 1 :])
        )
        return inputs[0].shape.replace(dim=out_dim)

    def build_layer(self):
        dim = tuple(self.preds[0].dim)  # type: ignore[attr-defined]
        return LinearImpl(
            features=self.features,
            dim_rank=len(dim),
            axis=self._axis(dim),
            shared_kernel=self.shared_kernel,
            name=self.name,
        )
