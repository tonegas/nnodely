"""Local model block: local functions blended by fuzzy memberships."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, TypeGuard

import keras
import numpy as np

from nnodely.core.dag import next_name
from nnodely.core.layer import Identity, Layer
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
from nnodely.layers.trigonometric import Trigonometric
from nnodely.utils.utils import _per_slice_initializer

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


def _cell_dim(dim) -> tuple[int, ...]:
    """The dim of one cell of a stack, whose first dim axis runs over the cells."""
    return tuple(dim[1:]) or (1,)


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

    def output_shape(self, *inputs):
        return (prod(activation.dim[0] for activation in inputs),), 1, ()

    def build_layer(self):
        return FuzzyProductImpl(name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class CellStackImpl(keras.layers.Layer):
    """
    Cells stacked on a new first dim axis.

    Inputs:  cell_0, ..., cell_{N-1}: [batch, *cell, time, *seq]
    Output:  [batch, N, *cell, time, *seq], or [batch, N, time, *seq] for
             scalar cells.
    """

    def __init__(self, scalar=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scalar = bool(scalar)

    def get_config(self):
        config = super().get_config()
        config.update({"scalar": self.scalar})
        return config

    def call(self, xs):
        xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
        if self.scalar:
            return keras.ops.concatenate(xs, axis=1)
        return keras.ops.stack(xs, axis=1)


class CellStack(Layer):
    """Stack equally shaped cells on a new first dim axis; scalar cells give ``(N,)``."""

    def output_shape(self, *inputs):
        shapes = {cell.shape.tuple for cell in inputs}
        if len(shapes) > 1:
            raise ValueError(
                f"{self.name}: every cell must have the same shape, got "
                f"{sorted(shapes, key=str)}."
            )
        cell = inputs[0]
        cells = len(inputs)
        dim = (cells,) if tuple(cell.dim) == (1,) else (cells, *cell.dim)
        return dim, cell.time, tuple(cell.seq)

    def build_layer(self):
        return CellStackImpl(scalar=len(self.dim) == 1, name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class CellSelectImpl(keras.layers.Layer):
    """[batch, N, *cell, time, *seq] -> [batch, *cell, time, *seq], cell ``idx``."""

    def __init__(self, idx, keepdims=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.idx = int(idx)
        self.keepdims = bool(keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"idx": self.idx, "keepdims": self.keepdims})
        return config

    def call(self, x):
        if self.keepdims:
            return x[:, self.idx : self.idx + 1]
        return x[:, self.idx]


class CellSelect(Layer):
    """One cell of a stack, without the cell axis."""

    def __init__(self, idx: int, name=None):
        self.idx = int(idx)
        super().__init__(name=name, idx=self.idx)

    def output_shape(self, *inputs):
        stack = inputs[0]
        return _cell_dim(stack.dim), stack.time, tuple(stack.seq)

    def build_layer(self):
        return CellSelectImpl(
            idx=self.idx,
            keepdims=len(self.preds[0].dim) == 1,  # type: ignore
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class CellSumImpl(keras.layers.Layer):
    """[batch, N, *cell, time, *seq] -> [batch, *cell, time, *seq]."""

    def __init__(self, keepdims=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.keepdims = bool(keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"keepdims": self.keepdims})
        return config

    def call(self, x):
        return keras.ops.sum(x, axis=1, keepdims=self.keepdims)


class CellSum(Layer):
    """Sum of the cells of a stack, without the cell axis."""

    def output_shape(self, *inputs):
        stack = inputs[0]
        return _cell_dim(stack.dim), stack.time, tuple(stack.seq)

    def build_layer(self):
        return CellSumImpl(
            keepdims=len(self.preds[0].dim) == 1,  # type: ignore
            name=self.name,
        )


class LocalModel:
    """
    Local models blended by fuzzy membership degrees.

    ::

        LocalModel(input_function, output_function, vectorized=False)(inputs, activations)

    The activations ``a_k: [n_k, 1]`` are combined into ``N = n_1 * n_2 * ...``
    joint memberships ``mu`` (their outer product, flattened row-major: cell
    ``(i_1, i_2)`` is number ``i_1 * n_2 + i_2``). The result is::

        sum_i output_function_i(mu_i * input_function_i(inputs))

    Either function is one of:

    * ``None``: no function. Without an input function ``inputs`` is a single
      stream, weighted as it is; without an output function the weighted
      cells are only summed.
    * A weightless elementwise layer (``Sin()``, ``Tanh()``, ``ReLU()``...):
      applied once - to the inputs, or to all the weighted cells together.
    * Any other layer (``Fir(out_features=1)``...): copied for every cell as
      ``"{name}_in{i}"`` / ``"{name}_out{i}"``, so each cell has its own weights.
    * A callable, which is a factory ``f((i_1, i_2, ...)) -> layer`` called once
      per cell; what it returns is applied to that cell alone.
    * With ``vectorized=True``, a callable is instead called once on all the
      cells: the input function on ``inputs``, the output function on the
      weighted cells. It returns all the cells at once, on a first dim axis of
      ``N`` entries: ``(N, *cell)``, or ``(N,)`` for scalar cells.

    Functions are called with a list of streams. Every cell must have the same
    shape; the result has the shape of one output cell.
    """

    def __init__(
        self,
        input_function=None,
        output_function=None,
        vectorized: bool = False,
        name: str | None = None,
    ):
        for role, function in (("input", input_function), ("output", output_function)):
            if function is not None and not callable(function):
                raise TypeError(
                    f"LocalModel: the {role} function must be a layer or a callable, "
                    f"got {type(function).__name__}. For one function per cell, pass "
                    "a factory f(index) -> layer."
                )
        self.input_function = input_function
        self.output_function = output_function
        self.vectorized = bool(vectorized)
        self.name = next_name("LocalModel") if name is None else name
        # The layers made for the cells, kept so every call of this LocalModel
        # applies the same ones - and so shares their weights.
        self._cells: dict[tuple[str, int], Callable] = {}

    def __call__(self, inputs, activations):
        inputs = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        activations = (
            list(activations)
            if isinstance(activations, (list, tuple))
            else [activations]
        )
        if not all(isinstance(stream, Stream) for stream in inputs + activations):
            raise TypeError(f"{self.name}: inputs and activations must be Streams.")
        for index, activation in enumerate(activations):
            shape = activation.shape
            if shape.seq_rank or shape.dim_rank != 1:
                raise ValueError(
                    f"{self.name}: activation {index} must have shape "
                    f"[cells, T], got {shape}."
                )

        indices = list(np.ndindex(*(activation.dim[0] for activation in activations)))
        mu = (
            activations[0]
            if len(activations) == 1
            else FuzzyProduct(name=f"{self.name}_mu")(activations)
        )

        # The input functions of every cell, stacked on the first dim axis.
        function = self.input_function
        if self._is_vectorized(function):
            stack = self._check_cells(function(inputs), len(indices), "input")
        else:
            if function is None or isinstance(function, _ELEMENTWISE):
                if function is None and len(inputs) != 1:
                    raise ValueError(
                        f"{self.name}: without an input function the input is "
                        f"weighted as it is, so there must be one, got {len(inputs)}."
                    )
                # The same for every cell: evaluated once.
                shared = inputs[0] if function is None else function(inputs)
                values = [shared] * len(indices)
            else:
                values = [
                    self._cell("in", function, i, index)(inputs)
                    for i, index in enumerate(indices)
                ]
            stack = CellStack(name=f"{self.name}_inputs")(values)
        weighted = stack * mu

        function = self.output_function
        if self._is_vectorized(function):
            weighted = self._check_cells(function([weighted]), len(indices), "output")
        elif isinstance(function, _ELEMENTWISE):
            weighted = function([weighted])
        elif function is not None:
            outputs = [
                self._cell("out", function, i, index)(
                    [CellSelect(idx=i, name=f"{self.name}_cell{i}")([weighted])]
                )
                for i, index in enumerate(indices)
            ]
            weighted = CellStack(name=f"{self.name}_outputs")(outputs)
        return CellSum(name=self.name)([weighted])

    def _is_vectorized(self, function) -> TypeGuard[Callable[..., Any]]:
        return (
            self.vectorized
            and function is not None
            and not isinstance(function, Layer)
        )

    def _check_cells(self, stack, cells, role):
        if not isinstance(stack, Stream) or stack.dim[0] != cells:
            got = stack.shape if isinstance(stack, Stream) else type(stack).__name__
            raise ValueError(
                f"{self.name}: a vectorized {role} function must return a stream "
                f"whose first dim axis has one entry per cell ({cells}), got {got}."
            )
        return stack

    def _cell(self, role, function, i, index):
        """The callable of cell ``i``, whose multi-index is ``index``."""
        if (role, i) not in self._cells:
            if isinstance(function, Layer):
                # # A single layer repeated over the cells is copied, so each
                # # cell has weights of its own.
                # self._cells[(role, i)] = type(function)(
                #     name=f"{self.name}_{role}{i}", **function._properties
                # )
                self._cells[(role, i)] = function
            else:
                self._cells[(role, i)] = function(index)
        return self._cells[(role, i)]
