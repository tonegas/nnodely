"""Local model block: local functions blended by fuzzy memberships."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, TypeGuard

import keras
import numpy as np

from nnodely.core.dag import next_name
from nnodely.core.layer import Identity, Layer, Multiply
from nnodely.core.shape import Shape, align, slice_axis
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
    """The dim of one cell of a stack, whose last dim axis runs over the cells."""
    return tuple(dim[:-1]) or (1,)


@keras.saving.register_keras_serializable(package="nnodely")
class FuzzyProductImpl(keras.layers.Layer):
    """
    Joint membership of several fuzzifications.

    Inputs:  a_k of dim (n_k,)
    Output:  dim (n_1 * n_2 * ...,), row-major: the last input varies
             fastest; the time axis is kept.
    """

    def call(self, xs):
        xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
        # Each activation (batch, *time, n_k), with at most one time axis,
        # given the time of the others: the product runs over the last axis.
        time_rank = max(len(activation.shape) - 2 for activation in xs)
        xs = [
            align(activation, len(activation.shape) - 2, 1, time_rank, 1)
            for activation in xs
        ]
        result = xs[0]
        for activation in xs[1:]:
            cells = result.shape[-1] * activation.shape[-1]
            result = keras.ops.expand_dims(result, -1) * keras.ops.expand_dims(
                activation, -2
            )
            result = keras.ops.reshape(result, (-1, *result.shape[1:-2], cells))
        return result


class FuzzyProduct(Layer):
    """Outer product of membership vectors, flattened row-major: dim
    ``(N,)``, time ``(T,)`` or none."""

    def output_shape(self, *inputs):
        cells = prod(activation.dim[0] for activation in inputs)
        time = max(tuple(activation.time) for activation in inputs)
        return Shape(dim=(cells,), time=time)

    def build_layer(self):
        return FuzzyProductImpl(name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class CellStackImpl(keras.layers.Layer):
    """
    Cells stacked on a new last dim axis, the tensor axis ``axis`` of the
    result: ``(*cell, N)``, or ``(N,)`` for scalar cells, which are joined on
    their one dim axis.
    """

    def __init__(self, axis, scalar=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.scalar = bool(scalar)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "scalar": self.scalar})
        return config

    def call(self, xs):
        xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
        if self.scalar:
            return keras.ops.concatenate(xs, axis=self.axis)
        return keras.ops.stack(xs, axis=self.axis)


class CellStack(Layer):
    """Stack equally shaped cells on a new last dim axis; scalar cells give ``(N,)``."""

    def output_shape(self, *inputs):
        shapes = {cell.shape.tuple for cell in inputs}
        if len(shapes) > 1:
            raise ValueError(
                f"{self.name}: every cell must have the same shape, got "
                f"{sorted(shapes, key=str)}."
            )
        cell = inputs[0]
        cells = len(inputs)
        dim = (cells,) if tuple(cell.dim) == (1,) else (*cell.dim, cells)
        return cell.shape.replace(dim=dim)

    def build_layer(self):
        return CellStackImpl(
            axis=self.shape.dim_axes[-1], scalar=len(self.dim) == 1, name=self.name
        )


@keras.saving.register_keras_serializable(package="nnodely")
class CellSelectImpl(keras.layers.Layer):
    """Cell ``idx`` of a stack whose cell axis is the tensor axis ``axis``,
    without that axis."""

    def __init__(self, idx, axis, keepdims=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.idx = int(idx)
        self.axis = int(axis)
        self.keepdims = bool(keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"idx": self.idx, "axis": self.axis, "keepdims": self.keepdims})
        return config

    def call(self, x):
        if self.keepdims:
            return slice_axis(x, self.axis, slice(self.idx, self.idx + 1))
        return slice_axis(x, self.axis, self.idx)


class CellSelect(Layer):
    """One cell of a stack, without the cell axis."""

    def __init__(self, idx: int, name=None):
        self.idx = int(idx)
        super().__init__(name=name, idx=self.idx)

    def output_shape(self, *inputs):
        stack = inputs[0].shape
        return stack.replace(dim=_cell_dim(stack.dim))

    def build_layer(self):
        stack = self.preds[0].shape  # type: ignore[union-attr]
        return CellSelectImpl(
            idx=self.idx,
            axis=stack.dim_axes[-1],
            keepdims=stack.dim_rank == 1,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class CellSumImpl(keras.layers.Layer):
    """The sum of the cells of a stack whose cell axis is the tensor axis
    ``axis``, without that axis."""

    def __init__(self, axis, keepdims=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.keepdims = bool(keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config

    def call(self, x):
        return keras.ops.sum(x, axis=self.axis, keepdims=self.keepdims)


class CellWeight(Multiply):
    """A stack of cells weighted by their memberships, of time ``(T,)`` or
    none and dim ``(N,)``: the stack's shape. The memberships may hold one
    sample, or none, to weigh every sample of the cells' window alike, a
    broadcast along time that plain arithmetic refuses; :class:`LocalModel`
    checks the times."""

    def output_shape(self, *inputs):
        stack, mu = inputs[0].shape, inputs[1].shape
        if stack.time and stack.time[-1] == 1 and mu.time:
            return stack.replace(time=(*stack.time[:-1], mu.time[-1]))
        return stack if stack.time else stack.replace(time=mu.time)


class CellSum(Layer):
    """Sum of the cells of a stack, without the cell axis."""

    def output_shape(self, *inputs):
        stack = inputs[0].shape
        return stack.replace(dim=_cell_dim(stack.dim))

    def build_layer(self):
        stack = self.preds[0].shape  # type: ignore[union-attr]
        return CellSumImpl(
            axis=stack.dim_axes[-1], keepdims=stack.dim_rank == 1, name=self.name
        )


class LocalModel:
    """
    Local models blended by fuzzy membership degrees.

    ::

        LocalModel(input_function, output_function, vectorized=False)(inputs, activations)

    The activations ``a_k``, of time ``(T,)`` or none and dim ``(n_k,)``, are
    combined into ``N = n_1 * n_2 * ...`` joint memberships ``mu`` (their
    outer product, flattened row-major: cell ``(i_1, i_2)`` is number
    ``i_1 * n_2 + i_2``). Their time axis is the cells' one, or 1 to weigh
    every sample alike. The result is::

        sum_i output_function_i(mu_i * input_function_i(inputs))

    Either function is one of:

    * ``None``: no function. Without an input function ``inputs`` is a single
      stream, weighted as it is; without an output function the weighted
      cells are only summed.
    * A weightless elementwise layer (``Sin()``, ``Tanh()``, ``ReLU()``...):
      applied once - to the inputs, or to all the weighted cells together.
    * Any other layer (``Fir(features=1)``...): applied to every cell, which
      all share its weights. For weights of its own in each cell, pass a
      factory.
    * A callable, which is a factory ``f((i_1, i_2, ...)) -> layer`` called once
      per cell; what it returns is applied to that cell alone.
    * With ``vectorized=True``, a callable or a layer (other than a weightless
      elementwise one) is instead called once on all the cells: the input
      function on ``inputs``, the output function on the weighted cells. It
      returns all the cells at once, on a last dim axis of ``N`` entries:
      ``(*cell, N)``, or ``(N,)`` for scalar cells - e.g. ``Fir(features=N)``.

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
            if isinstance(function, type):
                raise TypeError(
                    f"LocalModel: the {role} function is the class "
                    f"{function.__name__}; pass a layer instance, shared by every "
                    f"cell, or a factory such as lambda index: {function.__name__}(...)"
                    ", one layer per cell."
                )
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
            if shape.time_rank > 1 or shape.dim_rank != 1:
                raise ValueError(
                    f"{self.name}: activation {index} must have shape time (T,) "
                    f"or none, dim (cells,), got {shape}."
                )

        indices = list(np.ndindex(*(activation.dim[0] for activation in activations)))

        # The input functions of every cell, stacked on the last dim axis.
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
                if isinstance(shared, list):
                    raise ValueError(
                        f"{self.name}: the elementwise input function is applied "
                        f"to each of the {len(inputs)} inputs, and a cell is one "
                        "stream: pass one input, or a function that combines them."
                    )
                values = [shared] * len(indices)
            else:
                values = [
                    self._cell("in", function, i, index)(inputs)
                    for i, index in enumerate(indices)
                ]
            stack = CellStack(name=f"{self.name}_inputs")(values)

        # The memberships, of dim (N,), line up with the last dim axis of the
        # stack, its cell axis, and with its inner time.
        times = {
            stream.time[-1] if stream.time else 1 for stream in [*activations, stack]
        }
        if len(times - {1}) > 1:
            raise ValueError(
                f"{self.name}: the activations and the cells have time axes of "
                f"{sorted(times)} samples; they must match, or be 1."
            )
        mu = (
            activations[0]
            if len(activations) == 1
            else FuzzyProduct(name=f"{self.name}_mu")(activations)
        )
        weighted = CellWeight(name=f"{self.name}_weighted")([stack, mu])

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
            and not isinstance(function, _ELEMENTWISE)
        )

    def _check_cells(self, stack, cells, role):
        if not isinstance(stack, Stream) or stack.dim[-1] != cells:
            got = stack.shape if isinstance(stack, Stream) else type(stack).__name__
            raise ValueError(
                f"{self.name}: a vectorized {role} function must return a stream "
                f"whose last dim axis has one entry per cell ({cells}), got {got}."
            )
        return stack

    def _cell(self, role, function, i, index):
        """The callable of cell ``i``, whose multi-index is ``index``."""
        if (role, i) not in self._cells:
            if isinstance(function, Layer):
                # # A single layer repeated over the cells is copied, so each
                # # cell has weights of its own.
                # # (A kernel or bias given as a Parameter is copied as the
                # # same Parameter, and stays shared by every cell.)
                # self._cells[(role, i)] = type(function)(
                #     name=f"{self.name}_{role}{i}", **function._properties
                # )
                self._cells[(role, i)] = function
            else:
                self._cells[(role, i)] = function(index)
        return self._cells[(role, i)]
