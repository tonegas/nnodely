from __future__ import annotations

import keras

from nnodely.layers.parameter import Parameter, _ParameterWeights


@keras.saving.register_keras_serializable(package="nnodely")
class LinearImpl(keras.layers.Layer):
    """
    Linear projection along the dim axis, ``x @ kernel + bias``.

    Input convention:
        [batch, in_dim, time, *seq]

    Output convention:
        [batch, out_dim, time, *seq]

    With ``external_kernel`` or ``external_bias`` the layer is called on
    ``[x, kernel, bias]`` (those given), batched values of the kernel's and
    the bias's size, and owns no weight for them.
    """

    def __init__(
        self,
        out_features: int = 1,
        use_bias: bool = True,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        external_kernel: bool = False,
        external_bias: bool = False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.external_kernel = bool(external_kernel)
        self.external_bias = bool(external_bias)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features,
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "external_kernel": self.external_kernel,
                "external_bias": self.external_bias,
            }
        )
        return config

    def build(self, input_shape):
        x_shape = (
            input_shape[0]
            if self.external_kernel or self.external_bias
            else input_shape
        )
        self.kernel_shape = (int(x_shape[1]), self.out_features)
        self.kernel = (
            None
            if self.external_kernel
            else self.add_weight(
                name="kernel",
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
            )
        )
        self.bias = (
            self.add_weight(
                name="bias",
                shape=(self.out_features,),
                initializer=self.bias_initializer,
            )
            if self.use_bias and not self.external_bias
            else None
        )
        super().build(input_shape)

    def call(self, inputs):
        x, kernel, bias = inputs, self.kernel, self.bias
        if self.external_kernel or self.external_bias:
            x, *values = inputs
            # A Parameter comes batched, the same value for every sample.
            if self.external_kernel:
                kernel = keras.ops.reshape(values.pop(0)[0], self.kernel_shape)
            if self.external_bias:
                bias = keras.ops.reshape(values.pop(0)[0], (self.out_features,))

        rank = len(x.shape)

        # [batch, dim, time, *seq] -> [batch, time, *seq, dim]
        perm = [0] + list(range(2, rank)) + [1]
        x = keras.ops.transpose(x, perm)

        # The last axis, dim -> out_features, as a Dense layer projects it.
        x = keras.ops.matmul(x, kernel)
        if bias is not None:
            x = x + bias

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
    ``[out_features]``. Each is either a Keras initializer, by name or
    object, that draws a weight of the layer's own, or a :class:`Parameter`
    the layer computes with instead: it becomes a predecessor of the layer,
    and assigning or training it changes the projection. A Parameter must
    have the shape of the weight, axes of size 1 aside. ``bias=True`` draws
    a bias with ``"glorot_uniform"``, and ``bias=False`` leaves it out::

        W = Parameter("W", value=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        Linear(out_features=2, kernel=W, bias=False)([v.last()])  # v with dim=3
    """

    def __init__(
        self,
        out_features: int = 1,
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        name=None,
    ):
        self.out_features = int(out_features)
        self._kernel = kernel
        self._bias = bias
        super().__init__(
            name=name,
            out_features=self.out_features,
            kernel=kernel,
            bias=bias,
        )

    def output_shape(self, *inputs):
        # Declared rather than probed with dummy tensors, which would have to
        # stand in for the Parameters too.
        dim = tuple(inputs[0].dim)
        self._check_parameters((dim[0], self.out_features), (self.out_features,))
        return (self.out_features, *dim[1:]), inputs[0].time, tuple(inputs[0].seq)

    def build_layer(self):
        kernel, bias = self._kernel, self._bias
        return LinearImpl(
            out_features=self.out_features,
            use_bias=bias is not False,
            kernel_initializer=(
                "glorot_uniform" if isinstance(kernel, Parameter) else kernel
            ),
            bias_initializer=(
                "glorot_uniform" if isinstance(bias, (bool, Parameter)) else bias
            ),
            external_kernel=isinstance(kernel, Parameter),
            external_bias=isinstance(bias, Parameter),
            name=self.name,
        )
