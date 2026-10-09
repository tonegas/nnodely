from __future__ import annotations

import keras

from nnodely.layers.parameter import Parameter, _check_weight_size, _ParameterWeights


@keras.saving.register_keras_serializable(package="nnodely")
class LinearImpl(keras.layers.Layer):
    """
    Linear projection of the dim axes, ``x @ kernel + bias``, at every time
    and sequence step.

    Input convention:
        [batch, *dim, time, *seq]

    Output convention, with ``axis=None``:
        [batch, out_features, time, *seq]

    With ``axis=k`` only dim axis ``k`` is projected, and ``out_features``
    takes its place among the dim axes.

    It owns no weight: it is called on ``[x, kernel]`` or ``[x, kernel,
    bias]``, the values of the Parameters holding them, and reads their first
    sample.
    """

    def __init__(
        self, out_features: int = 1, dim_rank: int = 1, axis=None, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.out_features = int(out_features)
        self.dim_rank = int(dim_rank)
        self.axis = None if axis is None else int(axis)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features,
                "dim_rank": self.dim_rank,
                "axis": self.axis,
            }
        )
        return config

    def build(self, input_shape):
        dim = tuple(int(axis) for axis in input_shape[0][1 : 1 + self.dim_rank])
        projected = dim if self.axis is None else (dim[self.axis],)
        self.kernel_shape = (*projected, self.out_features)
        for label, shape, expected in zip(
            ("kernel", "bias"),
            input_shape[1:],
            (self.kernel_shape, (self.out_features,)),
        ):
            _check_weight_size(self.name, label, shape, expected)
        super().build(input_shape)

    def call(self, inputs):
        x, kernel, *bias = inputs
        rank = len(x.shape)
        dim_axes = list(range(1, 1 + self.dim_rank))
        projected = dim_axes if self.axis is None else [1 + self.axis]
        kept = [axis for axis in dim_axes if axis not in projected]

        # [batch, *dim, time, *seq] -> [batch, *kept, time, *seq, *projected]:
        # the projected axes last, contracted with the first of the kernel.
        x = keras.ops.transpose(
            x, [0, *kept, *range(1 + self.dim_rank, rank), *projected]
        )
        x = keras.ops.tensordot(
            x, keras.ops.reshape(kernel[0], self.kernel_shape), axes=len(projected)
        )
        if bias:
            x = x + keras.ops.reshape(bias[0][0], (self.out_features,))

        # out_features back among the dim axes: first, or where axis was.
        return keras.ops.moveaxis(x, -1, 1 + (self.axis or 0))


class Linear(_ParameterWeights):
    """
    Linear projection of the dim axes, ``x @ kernel + bias``, applied at every
    time and sequence step.

    With ``axis=None`` the whole dim is projected: every feature of a sample
    feeds every output::

        input:  (batch, *dim, time, *seq)
        output: (batch, out_features, time, *seq)
        kernel: (*dim, out_features)

    With ``axis=k`` only dim axis ``k`` is projected, the same matrix applied
    along it at every position of the other dim axes, and ``out_features``
    takes its place: on ``D=(2, 3)``, ``axis=1`` gives ``D=(2, out_features)``
    with a kernel ``(3, out_features)``.

    ``bias`` is ``(out_features,)``. ``kernel`` and ``bias`` are Parameters,
    predecessors of the layer: assigning or training them changes the
    projection. Each is given as a :class:`Parameter`, of the shape of the
    weight (axes of size 1 aside), or as a Keras initializer, by name or
    object, from which the layer makes one, named ``"<name>_kernel"`` or
    ``"<name>_bias"``; the kernel is drawn as a matrix of the projected
    features by ``out_features``. ``bias=True`` draws the bias with
    ``"glorot_uniform"``, and ``bias=False`` leaves it out::

        W = Parameter("W", value=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        Linear(out_features=2, kernel=W, bias=False)([v.last()])  # v with dim=3
    """

    _bias_initializer = "glorot_uniform"

    def __init__(
        self,
        out_features: int = 1,
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        axis: int | None = None,
        name=None,
    ):
        self.out_features = int(out_features)
        self.axis = None if axis is None else int(axis)
        super().__init__(
            name, kernel, bias, out_features=self.out_features, axis=self.axis
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
        axis = self._axis(dim)
        projected = dim if axis is None else (dim[axis],)
        return (*projected, self.out_features), (self.out_features,)

    def output_shape(self, *inputs):
        # Declared rather than probed with dummy tensors, which would have to
        # stand in for the weights too.
        dim = tuple(inputs[0].dim)
        axis = self._axis(dim)
        out_dim = (
            (self.out_features,)
            if axis is None
            else (*dim[:axis], self.out_features, *dim[axis + 1 :])
        )
        return out_dim, inputs[0].time, tuple(inputs[0].seq)

    def build_layer(self):
        dim = tuple(self.preds[0].dim)  # type: ignore[attr-defined]
        return LinearImpl(
            out_features=self.out_features,
            dim_rank=len(dim),
            axis=self._axis(dim),
            name=self.name,
        )
