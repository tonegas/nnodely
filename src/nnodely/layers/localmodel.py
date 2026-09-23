"""Local model layer."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import cast

import keras

from nnodely.core.layer import Add, Layer
from nnodely.layers.time_ops import Select


def _per_cell_initializer(initializer):
    """Draw every cell from ``initializer`` as if it owned its own matrix.

    Variance scaling reads the fans from the whole tensor, so initialising
    ``[cells, features, out_features]`` in one shot would divide the scale by
    the number of cells: a cell only ever sees ``features`` inputs.
    """
    base = cast(keras.initializers.Initializer, keras.initializers.get(initializer))

    def initialize(shape, dtype=None):
        # A random initializer draws its seed once and then repeats itself, so
        # reusing one instance would give every cell the same matrix.
        config = base.get_config()
        cells = []
        for index in range(shape[0]):
            cell_config = dict(config)
            if cell_config.get("seed") is not None:
                cell_config["seed"] = config["seed"] + index
            cells.append(type(base).from_config(cell_config)(shape[1:], dtype=dtype))
        return keras.ops.stack(cells, axis=0)

    return initialize


@keras.saving.register_keras_serializable(package="nnodely")
class LocalModelImpl(keras.layers.Layer):
    """
    Affine local models blended by their membership degrees.

    Inputs (one activation per input):
        x_j: [batch, dim..., time]
        a_j: [batch, cells_j, 1]

    Output:
        [batch, out_features, 1]

    Cell ``(j, i)`` owns an affine map of the flattened ``x_j``, so the block is
    ``sum_j sum_i a_j,i (W_j,i x_j + b_j,i)``. Blending the memberships into the
    input before the projection turns those cells into a single matmul per
    input, instead of one Keras layer per cell.
    """

    def __init__(
        self,
        n_inputs: int = 1,
        out_features: int = 1,
        use_bias: bool = True,
        initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.n_inputs = int(n_inputs)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.features: list[int] = []
        self.cells: list[int] = []
        self.kernels: list[keras.Variable] = []
        self.biases: list[keras.Variable] = []

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_inputs": self.n_inputs,
                "out_features": self.out_features,
                "use_bias": self.use_bias,
                "initializer": self.initializer,
                "bias_initializer": self.bias_initializer,
            }
        )
        return config

    def build(self, input_shape):
        data_shapes = input_shape[: self.n_inputs]
        activation_shapes = input_shape[self.n_inputs :]

        for index, (data_shape, activation_shape) in enumerate(
            zip(data_shapes, activation_shapes)
        ):
            if any(axis is None for axis in data_shape[1:]):
                raise ValueError(
                    f"{self.name}: input {index} has a dynamic shape "
                    f"{tuple(data_shape)}; a local model needs a fixed window."
                )
            if len(activation_shape) != 3 or activation_shape[1] is None:
                raise ValueError(
                    f"{self.name}: activation {index} must have shape "
                    f"[batch, cells, 1], got {tuple(activation_shape)}."
                )

            features = prod(int(axis) for axis in data_shape[1:])
            cells = int(activation_shape[1])
            self.features.append(features)
            self.cells.append(cells)
            self.kernels.append(
                self.add_weight(
                    shape=(cells, features, self.out_features),
                    initializer=_per_cell_initializer(self.initializer),
                    name=f"kernel_{index}",
                )
            )
            if self.use_bias:
                self.biases.append(
                    self.add_weight(
                        shape=(cells, self.out_features),
                        initializer=self.bias_initializer,
                        name=f"bias_{index}",
                    )
                )
        super().build(input_shape)

    def call(self, xs):
        result = None
        for index, (x, activation) in enumerate(
            zip(xs[: self.n_inputs], xs[self.n_inputs :])
        ):
            features = self.features[index]
            cells = self.cells[index]

            x = keras.ops.reshape(x, (-1, features))
            activation = keras.ops.reshape(activation, (-1, cells))

            # [batch, cells, features] flattened: one matmul then evaluates
            # every cell and sums them weighted by their membership degree.
            blended = keras.ops.reshape(
                keras.ops.expand_dims(activation, -1) * keras.ops.expand_dims(x, 1),
                (-1, cells * features),
            )
            term = keras.ops.matmul(
                blended,
                keras.ops.reshape(
                    self.kernels[index], (cells * features, self.out_features)
                ),
            )
            if self.use_bias:
                term = term + keras.ops.matmul(activation, self.biases[index])

            result = term if result is None else result + term

        return keras.ops.reshape(result, (-1, self.out_features, 1))


class LocalModel(Layer):
    """
    Local models blended by fuzzy membership degrees.

    Every input is paired with the activation that schedules it, and the pairs
    are summed::

        LocalModel(out_features=1)([torque.sw(25)], [fuzzified_gear])

    Input:
        x_j: [batch, dim..., time]
        a_j: [batch, cells_j, 1]

    Output:
        [batch, out_features, 1]

    Each cell is affine in the flattened input, which is what lets the whole
    block collapse into one matmul per input. ``input_function`` and
    ``output_function`` escape that: they build one explicit cell per
    membership out of ordinary nnodely layers, so an arbitrary user function
    gets its own parameters per cell at the price of the expanded graph. They
    replace the affine cell entirely, so ``out_features`` and ``use_bias`` are
    then unused.
    """

    def __init__(
        self,
        out_features: int = 1,
        use_bias: bool = True,
        initializer="glorot_uniform",
        bias_initializer="zeros",
        input_function: Callable | None = None,
        output_function: Callable | None = None,
        n_inputs: int = 1,
        name=None,
    ):
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.input_function = input_function
        self.output_function = output_function
        self.n_inputs = int(n_inputs)
        super().__init__(
            name=name,
            out_features=self.out_features,
            use_bias=self.use_bias,
            initializer=initializer,
            bias_initializer=bias_initializer,
            input_function=input_function,
            output_function=output_function,
            n_inputs=self.n_inputs,
        )

    def __call__(self, inputs, activations):  # type: ignore #TODO: control the multiple call signature
        inputs = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        activations = (
            list(activations)
            if isinstance(activations, (list, tuple))
            else [activations]
        )
        if len(inputs) != len(activations):
            raise ValueError(
                f"{self.name}: got {len(inputs)} inputs and {len(activations)} "
                "activations; every input is scheduled by its own activation."
            )

        self.n_inputs = len(inputs)
        self._properties["n_inputs"] = self.n_inputs

        if self.input_function is None and self.output_function is None:
            return super().__call__([*inputs, *activations])

        return self._expand(inputs, activations)

    def _expand(self, inputs, activations):
        """One explicit cell per membership, for functions no matmul can fold."""
        cells = []
        for x, activation in zip(inputs, activations):
            for index in range(activation.dim[0]):
                cell = x if self.input_function is None else self.input_function([x])
                cell = cell * Select(idx=index, axis=0)([activation])
                if self.output_function is not None:
                    cell = self.output_function([cell])
                cells.append(cell)
        return Add()(cells) if len(cells) > 1 else cells[0]

    def build_layer(self):
        for index, node in enumerate(self.preds[: self.n_inputs]):
            shape = getattr(node, "shape", None)
            if shape is None:
                raise ValueError(
                    f"{self.name}: input {index} has no shape; a local model "
                    "needs a fixed window."
                )
            if shape.seq_rank:
                raise ValueError(
                    f"{self.name}: input {index} carries a sequence axis "
                    f"{shape.seq}; a local model consumes one window at a time."
                )
        for index, node in enumerate(self.preds[self.n_inputs :]):
            shape = getattr(node, "shape", None)
            if shape is None:
                raise ValueError(
                    f"{self.name}: activation {index} has no shape; a local model "
                    "needs a fixed window."
                )
            if shape.seq_rank or shape.dim_rank != 1 or shape.time != 1:
                raise ValueError(
                    f"{self.name}: activation {index} must have shape "
                    f"[cells, 1], got {shape}."
                )

        return LocalModelImpl(
            n_inputs=self.n_inputs,
            out_features=self.out_features,
            use_bias=self.use_bias,
            initializer=self.initializer,
            bias_initializer=self.bias_initializer,
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "out_features": self.out_features,
            "use_bias": self.use_bias,
            "n_inputs": self.n_inputs,
        }

    @classmethod
    def from_config(cls, config: dict, preds=None):
        layer = cls(**config)

        if preds is None or len(preds) == 0:
            return layer

        return layer(preds[: layer.n_inputs], preds[layer.n_inputs :])

    @property
    def kernel(self):
        """The cell matrices, one ``[cells, features, out_features]`` per input."""
        if self._layer is None:
            return None
        return (
            self._layer.kernels[0]
            if len(self._layer.kernels) == 1
            else self._layer.kernels
        )

    @property
    def bias(self):
        """The cell biases, one ``[cells, out_features]`` per input."""
        if self._layer is None or not self._layer.use_bias:
            return None
        return (
            self._layer.biases[0]
            if len(self._layer.biases) == 1
            else self._layer.biases
        )


## HIGH LEVEL BLOCK FOR LOCAL MODEL ##
# class LocalModel:
#     """
#     High-level abstraction for a local model built using only nnodely blocks
#     """

#     def __init__(
#         self,
#         input_function,
#         output_function=None,
#         name: str | None = None,
#     ):
#         self.input_function = input_function
#         self.output_function = output_function
#         self.name = name

#     def __call__(self, activation):
#         ret = []
#         local = Input("local_input")
#         for i in range(activation.dim[0]):
#             x = self.input_function([local]) * Select(idx=i, axis=0)([activation])
#             if self.output_function is not None:
#                 x = self.output_function([x])
#             ret.append(x)
#         ret = Add()(ret)
#         out = Output("local_output", ret)
#         return Modely(name=f"{self.name}", inputs=[local], outputs=[out])
