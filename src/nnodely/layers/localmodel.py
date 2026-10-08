"""Local model layer."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import cast

import numpy as np
import keras

from nnodely.core.dag import next_name
from nnodely.core.layer import Add, Identity, Layer
from nnodely.core.stream import Stream
from nnodely.layers.activations import (
    ELU,
    GELU,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Softplus,
    Swish,
    Tanh,
)
from nnodely.layers.arithmetic import Arithmetic, Clamp
from nnodely.layers.fir import Fir
from nnodely.layers.time_ops import Select
from nnodely.layers.trigonometric import Trigonometric

# Weightless layers that act on every element alone: applying one of them to
# all the cells stacked together is the same as applying it cell by cell.
_ELEMENTWISE = (
    Identity,
    Arithmetic,
    Clamp,
    Trigonometric,
    ReLU,
    LeakyReLU,
    ELU,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
    Softplus,
)


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
class FuzzyProductImpl(keras.layers.Layer):
    """
    Joint membership of several fuzzifications.

    Inputs:  a_k: [batch, n_k, 1]
    Output:  [batch, n_1 * n_2 * ..., 1], row-major: the last input varies fastest.
    """

    def call(self, xs):
        result = keras.ops.reshape(xs[0], (-1, xs[0].shape[1]))
        for activation in xs[1:]:
            cells = activation.shape[1]
            result = keras.ops.expand_dims(result, -1) * keras.ops.reshape(
                activation, (-1, 1, cells)
            )
            result = keras.ops.reshape(result, (-1, result.shape[1] * cells))
        return keras.ops.expand_dims(result, -1)


class FuzzyProduct(Layer):
    """Outer product of membership vectors, flattened row-major into ``[N, 1]``."""

    def build_layer(self):
        return FuzzyProductImpl(name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class LocalFirImpl(keras.layers.Layer):
    """
    One affine map per cell, weighted by its membership degree.

    Inputs:
        x:  [batch, dim..., time]
        mu: [batch, cells, 1]

    Output:
        reduce=True:  [batch, out_features, 1]           sum_i mu_i (W_i x + b_i)
        reduce=False: [batch, cells, out_features, 1]    mu_i (W_i x + b_i)
    """

    def __init__(self, out_features=1, use_bias=True, reduce=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.reduce = bool(reduce)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features,
                "use_bias": self.use_bias,
                "reduce": self.reduce,
            }
        )
        return config

    def build(self, input_shape):
        x_shape, mu_shape = input_shape
        self.features = prod(int(axis) for axis in x_shape[1:])
        self.cells = int(mu_shape[1])
        self.kernel = self.add_weight(
            shape=(self.cells, self.features, self.out_features),
            initializer=_per_cell_initializer("glorot_uniform"),
            name="kernel",
        )
        self.bias = (
            self.add_weight(
                shape=(self.cells, self.out_features), initializer="zeros", name="bias"
            )
            if self.use_bias
            else None
        )
        super().build(input_shape)

    def call(self, xs):
        x = keras.ops.reshape(xs[0], (-1, self.features))
        mu = keras.ops.reshape(xs[1], (-1, self.cells))

        if self.reduce:
            # Blending the memberships into the input first evaluates and sums
            # every cell with a single matmul.
            blended = keras.ops.reshape(
                keras.ops.expand_dims(mu, -1) * keras.ops.expand_dims(x, 1),
                (-1, self.cells * self.features),
            )
            y = keras.ops.matmul(
                blended,
                keras.ops.reshape(
                    self.kernel, (self.cells * self.features, self.out_features)
                ),
            )
            if self.bias is not None:
                y = y + keras.ops.matmul(mu, self.bias)
            return keras.ops.reshape(y, (-1, self.out_features, 1))

        y = keras.ops.einsum("bf,nfo->bno", x, self.kernel)
        if self.bias is not None:
            y = y + self.bias
        y = y * keras.ops.expand_dims(mu, -1)
        return keras.ops.reshape(y, (-1, self.cells, self.out_features, 1))


class LocalFir(Layer):
    """
    ``cells`` independent :class:`Fir` maps evaluated at once.

    Input:  x: [dim..., time], mu: [cells, 1]
    Output: [out_features, 1] summed over the cells, or
            [cells, out_features, 1] with ``reduce=False``.
    """

    def __init__(
        self,
        out_features: int = 1,
        use_bias: bool = True,
        reduce: bool = True,
        name=None,
    ):
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.reduce = bool(reduce)
        super().__init__(
            name=name,
            out_features=self.out_features,
            use_bias=self.use_bias,
            reduce=self.reduce,
        )

    def build_layer(self):
        x = cast(Stream, self.preds[0])
        if x.shape.seq_rank:
            raise ValueError(
                f"{self.name}: the input carries a sequence axis {x.shape.seq}; "
                "a local model consumes one window at a time."
            )
        return LocalFirImpl(
            out_features=self.out_features,
            use_bias=self.use_bias,
            reduce=self.reduce,
            name=self.name,
        )

    @property
    def kernel(self):
        """The cell matrices, ``[cells, features, out_features]``."""
        return None if self._layer is None else self._layer.kernel

    @property
    def bias(self):
        """The cell biases, ``[cells, out_features]``."""
        return None if self._layer is None else self._layer.bias


@keras.saving.register_keras_serializable(package="nnodely")
class CellStackImpl(keras.layers.Layer):
    """
    Cells weighted by their membership degree.

    Inputs:  cell_0, ..., cell_{N-1}: [batch, *cell], mu: [batch, N, 1]
    Output:  [batch, *cell] summed, or [batch, N, *cell] with ``reduce=False``.
    """

    def __init__(self, reduce=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduce = bool(reduce)

    def get_config(self):
        config = super().get_config()
        config.update({"reduce": self.reduce})
        return config

    def call(self, xs):
        cells = keras.ops.stack(xs[:-1], axis=1)
        mu = keras.ops.reshape(
            xs[-1], (-1, len(xs) - 1) + (1,) * (len(cells.shape) - 2)
        )
        cells = cells * mu
        return keras.ops.sum(cells, axis=1) if self.reduce else cells


class CellStack(Layer):
    """Stack the cells weighted by the memberships given last, optionally summed."""

    def __init__(self, reduce: bool = True, name=None):
        self.reduce = bool(reduce)
        super().__init__(name=name, reduce=self.reduce)

    def build_layer(self):
        return CellStackImpl(reduce=self.reduce, name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class CellSumImpl(keras.layers.Layer):
    """[batch, N, *cell] -> [batch, *cell]."""

    def call(self, x):
        return keras.ops.sum(x, axis=1)


class CellSum(Layer):
    """Sum over the leading cell axis, dropping it."""

    def build_layer(self):
        return CellSumImpl(name=self.name)


class LocalModel:
    """
    Local models blended by fuzzy membership degrees.

    ::

        LocalModel(Fir(out_features=1))(torque.sw(25), [gear_mu, speed_mu])

    The activations ``a_k: [n_k, 1]`` are combined into ``N = n_1 * n_2 * ...``
    joint memberships ``mu`` (their outer product, flattened row-major: cell
    ``(i_1, i_2)`` is number ``i_1 * n_2 + i_2``), which still sum to one when
    every activation does. Each cell gets its own instance of
    ``input_function``, evaluated on ``inputs``, and the result is::

        sum_i output_function_i(mu_i * input_function_i(inputs))

    ``input_function`` and ``output_function`` take the list of input streams
    and are either one callable, instantiated anew for every cell, or a list of
    ``N`` callables used as given; every cell must return the same shape.
    ``input_function`` defaults to ``Fir(out_features=1)``. With
    ``pass_index=True`` a plain function is a factory instead: it receives the
    cell index ``(i_1, i_2, ...)`` and returns the callable of that cell.

    A :class:`Fir` instance as ``input_function`` evaluates every cell with a
    single matmul, and a weightless elementwise layer instance (``ReLU()``,
    ``Tanh()``, ``Sin()``...) as ``output_function`` is applied once to all
    the cells together. Anything else builds one explicit subgraph per cell.
    """

    def __init__(
        self,
        input_function: Callable | list[Callable] | None = None,
        output_function: Callable | list[Callable] | None = None,
        pass_index: bool = False,
        name: str | None = None,
    ):
        self.input_function = (
            Fir(out_features=1) if input_function is None else input_function
        )
        self.output_function = output_function
        self.pass_index = bool(pass_index)
        self.name = next_name("LocalModel") if name is None else name
        # The layers made for the cells, kept so every call of this LocalModel
        # applies the same ones - and so shares their weights.
        self._cells: dict[tuple[str, int], Layer] = {}

    def __call__(self, inputs, activations):
        inputs = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        activations = (
            list(activations)
            if isinstance(activations, (list, tuple))
            else [activations]
        )
        for index, activation in enumerate(activations):
            shape = activation.shape
            if shape.seq_rank or shape.dim_rank != 1 or shape.time != 1:
                raise ValueError(
                    f"{self.name}: activation {index} must have shape "
                    f"[cells, 1], got {shape}."
                )

        sizes = [activation.dim[0] for activation in activations]
        indices = list(np.ndindex(*sizes))
        for role, function in (
            ("input", self.input_function),
            ("output", self.output_function),
        ):
            if isinstance(function, (list, tuple)) and len(function) != len(indices):
                raise ValueError(
                    f"{self.name}: got {len(function)} {role} functions for "
                    f"{len(indices)} cells."
                )
        mu = (
            activations[0]
            if len(activations) == 1
            else FuzzyProduct(name=f"{self.name}_mu")(activations)
        )

        once = self.output_function is None or isinstance(
            self.output_function, _ELEMENTWISE
        )
        final_name = self.name if self.output_function is None else f"{self.name}_cells"

        # LocalFir mixes every element of its input: a Fir only on a scalar.
        if (
            once
            and type(self.input_function) is Fir
            and len(inputs) == 1
            and inputs[0].dim == (1,)
        ):
            fir = cast(Fir, self.input_function)
            cells = LocalFir(
                out_features=fir.out_features,
                use_bias=fir.use_bias,
                reduce=self.output_function is None,
                name=final_name,
            )([inputs[0], mu])
        else:
            outputs = [
                self._cell(self.input_function, i, index, "in")(inputs)
                for i, index in enumerate(indices)
            ]
            self._check_same_shape(outputs, "input")
            if not once:
                return self._per_cell_outputs(outputs, mu, indices)
            cells = CellStack(reduce=self.output_function is None, name=final_name)(
                [*outputs, mu]
            )

        if self.output_function is None:
            return cells
        output_function = cast(Layer, self.output_function)
        return CellSum(name=self.name)([output_function([cells])])

    def _per_cell_outputs(self, outputs, mu, indices):
        """One output function per cell, for functions that cannot be batched."""
        cells = [
            self._cell(self.output_function, i, index, "out")(
                [output * Select(idx=i, axis=0)([mu])]  # type: ignore
            )
            for i, (output, index) in enumerate(zip(outputs, indices))
        ]
        self._check_same_shape(cells, "output")
        return Add(name=self.name)(cells) if len(cells) > 1 else cells[0]

    def _check_same_shape(self, cells, role):
        shapes = {cell.shape.tuple for cell in cells}
        if len(shapes) > 1:
            raise ValueError(
                f"{self.name}: every {role} function must return the same "
                f"shape, got {sorted(shapes)}."
            )

    def _cell(self, function, i, index, role):
        """The callable of cell ``i``, whose multi-index is ``index``."""
        if isinstance(function, (list, tuple)):
            return function[i]
        if isinstance(function, Layer):
            # A single layer repeated over the cells is copied, so each cell
            # has weights of its own.
            if (role, i) not in self._cells:
                self._cells[(role, i)] = type(function)(
                    name=f"{self.name}_{role}{i}", **function._properties
                )
            return self._cells[(role, i)]
        if self.pass_index:
            return function(index)
        return function
