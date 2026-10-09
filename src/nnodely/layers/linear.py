from __future__ import annotations

import keras

from nnodely.layers.parameter import Parameter, _check_weight_size, _ParameterWeights


@keras.saving.register_keras_serializable(package="nnodely")
class LinearImpl(keras.layers.Layer):
    """
    Linear projection along the dim axis, ``x @ kernel + bias``.

    Input convention:
        [batch, in_dim, time, *seq]

    Output convention:
        [batch, out_dim, time, *seq]

    It owns no weight: it is called on ``[x, kernel]`` or ``[x, kernel,
    bias]``, the values of the Parameters holding them, and reads their first
    sample.
    """

    def __init__(self, out_features: int = 1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_features = int(out_features)

    def get_config(self):
        config = super().get_config()
        config.update({"out_features": self.out_features})
        return config

    def build(self, input_shape):
        self.kernel_shape = (int(input_shape[0][1]), self.out_features)
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

        # [batch, dim, time, *seq] -> [batch, time, *seq, dim]
        perm = [0] + list(range(2, rank)) + [1]
        x = keras.ops.transpose(x, perm)

        # The last axis, dim -> out_features, as a Dense layer projects it.
        x = keras.ops.matmul(x, keras.ops.reshape(kernel[0], self.kernel_shape))
        if bias:
            x = x + keras.ops.reshape(bias[0][0], (self.out_features,))

        # [batch, time, *seq, out_features] -> [batch, out_features, time, *seq]
        rank_y = len(x.shape)
        inv_perm = [0, rank_y - 1] + list(range(1, rank_y - 1))
        x = keras.ops.transpose(x, inv_perm)

        return x


class Linear(_ParameterWeights):
    """
    Linear projection along the dim dimension, ``x @ kernel + bias``.

    Input::

        [batch, in_features, time, *seq]

    Output::

        [batch, out_features, time, *seq]

    ``kernel`` is ``[in_features, out_features]`` and ``bias``
    ``[out_features]``. Both are Parameters, predecessors of the layer:
    assigning or training them changes the projection. Each is given as a
    :class:`Parameter`, of the shape of the weight (axes of size 1 aside), or
    as a Keras initializer, by name or object, from which the layer makes
    one, named ``"<name>_kernel"`` or ``"<name>_bias"``. ``bias=True`` draws
    the bias with ``"glorot_uniform"``, and ``bias=False`` leaves it out::

        W = Parameter("W", value=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        Linear(out_features=2, kernel=W, bias=False)([v.last()])  # v with dim=3
    """

    _bias_initializer = "glorot_uniform"

    def __init__(
        self,
        out_features: int = 1,
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        name=None,
    ):
        self.out_features = int(out_features)
        super().__init__(name, kernel, bias, out_features=self.out_features)

    def _weight_shapes(self, x):
        return (x.dim[0], self.out_features), (self.out_features,)

    def output_shape(self, *inputs):
        # Declared rather than probed with dummy tensors, which would have to
        # stand in for the weights too.
        dim = tuple(inputs[0].dim)
        return (self.out_features, *dim[1:]), inputs[0].time, tuple(inputs[0].seq)

    def build_layer(self):
        return LinearImpl(out_features=self.out_features, name=self.name)
