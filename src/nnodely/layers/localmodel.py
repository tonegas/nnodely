"""Local model block: local functions blended by fuzzy memberships."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, TypeGuard

import keras
import numpy as np

from nnodely.core.dag import _undeclared_inputs, next_name
from nnodely.core.layer import Identity, Layer, Multiply, _refuse_uncalled
from nnodely.core.modely import Modely
from nnodely.core.shape import Shape, slice_axis
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
from nnodely.layers.input import Input
from nnodely.layers.output import Output
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


def _block_input(stream: Stream, name: str) -> Input:
    """An Input of the shape of ``stream``, which a block reads it as: its dim,
    its sequence, a dynamic axis declared with -1, and its window."""
    seq = tuple(-1 if size is None else size for size in stream.seq)
    return Input(name, dim=stream.dim, seq=seq or None, tw=stream.shape.tw)


@keras.saving.register_keras_serializable(package="nnodely")
class FuzzyProductImpl(keras.layers.Layer):
    """
    Joint membership of several fuzzifications.

    Inputs:  a_k of dim (n_k,)
    Output:  dim (n_1 * n_2 * ...,), row-major: the last input varies
             fastest; the window is kept.
    """

    def call(self, xs):
        xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
        # Each activation (batch, tw, n_k), or (batch, n_k) without a
        # window: the product runs over the last axis, a window of one sample
        # broadcasting along a longer one.
        result = xs[0]
        for activation in xs[1:]:
            cells = result.shape[-1] * activation.shape[-1]  # type: ignore
            result = keras.ops.expand_dims(result, -1) * keras.ops.expand_dims(
                activation, -2
            )
            result = keras.ops.reshape(result, (-1, *result.shape[1:-2], cells))
        return result


class FuzzyProduct(Layer):
    """Outer product of membership vectors, flattened row-major: dim
    ``(N,)``, the window of the longest."""

    def output_shape(self, *inputs):
        cells = prod(activation.dim[0] for activation in inputs)
        windows = [activation.tw for activation in inputs if activation.tw]
        return Shape(dim=(cells,), tw=max(windows) if windows else None)

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
        self._idx = int(idx)
        super().__init__(name=name, idx=self._idx)

    def output_shape(self, *inputs):
        stack = inputs[0].shape
        return stack.replace(dim=_cell_dim(stack.dim))

    def build_layer(self):
        stack = self._preds[0].shape  # type: ignore[union-attr]
        return CellSelectImpl(
            idx=self._idx,
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
    """A stack of cells weighted by their memberships, of dim ``(N,)``, which
    line up with the stack's last dim axis, a broadcast plain arithmetic
    refuses. Either may hold one sample of the window, weighed along the
    other's: :class:`LocalModel` checks the windows."""

    def output_shape(self, *inputs):
        stack, mu = inputs[0].shape, inputs[1].shape
        if stack.tw is None:
            return stack
        return stack.replace(tw=max(stack.tw, mu.tw))


class CellSum(Layer):
    """Sum of the cells of a stack, without the cell axis."""

    def output_shape(self, *inputs):
        stack = inputs[0].shape
        return stack.replace(dim=_cell_dim(stack.dim))

    def build_layer(self):
        stack = self._preds[0].shape  # type: ignore[union-attr]
        return CellSumImpl(
            axis=stack.dim_axes[-1], keepdims=stack.dim_rank == 1, name=self.name
        )


class LocalModel:
    r"""Local models blended by membership degrees: the sum of every cell's model, weighted by its membership.

    Syntax
    ------
    ``LocalModel()(x, mu)``
        ``x`` weighted by each membership of ``mu``, and summed.
    ``LocalModel(input_function=f)(inputs, activations)``
        A model of the inputs in every cell, weighted by the cell's
        membership and summed.
    ``LocalModel(output_function=g)(inputs, activations)``
        ``g`` applied to every weighted cell before the sum.
    ``LocalModel(f, vectorized=True)(inputs, activations)``
        ``f`` computes all the cells at once.

    The activations, typically :class:`Fuzzify` memberships, define the cells.
    One activation of ``n`` memberships gives ``n`` cells. Several activations
    are multiplied together: memberships over 4 gears and 5 speeds give 20
    cells. Each cell has a local model, the input function, whose result is
    weighted by the cell's membership. The output function is applied to each
    weighted cell, and the cells are summed.

    Each call declares the cells in a model of their own, called as a block on
    the given streams, which :meth:`Modely.export_html` shows on a page of its
    own. The functions may read other streams of the graph too. Calling the
    same LocalModel again applies the same cell layers, with the same weights.

    Parameters
    ----------
    input_function : Layer or callable, optional
        The local model of each cell, called with the list of inputs. It is
        one of:

        - ``None``, the default: the one input is weighted as it is;
        - a weightless elementwise layer instance, such as ``Sin()``,
          ``Tanh()``, ``ReLU()`` or ``Square()``: applied once, the same in
          every cell;
        - any other layer instance, such as ``Fir()`` or ``Linear()``: applied
          in every cell, all the cells sharing its weights;
        - a factory, a callable ``f(index) -> layer``: called once per cell
          with the cell's index, a tuple ``(i,)`` for one activation and
          ``(i, j, ...)`` for several. The layer it returns is applied in that
          cell alone, with weights of its own.
    output_function : Layer or callable, optional
        A function of each weighted cell, called with a list of one stream.
        Its forms are those of ``input_function``. ``None``, the default, sums
        the weighted cells as they are. A weightless elementwise layer is
        applied once, to all the weighted cells together.
    vectorized : bool, default False
        ``True`` calls a function once on all the cells, unless it is a
        weightless elementwise layer. The input function is called on the
        inputs, the output function on the weighted cells. It returns the
        cells on a last dim axis of ``N`` entries: ``(*cell, N)``, or ``(N,)``
        for scalar cells, as ``Fir(features=N)`` does.
    name : str, optional
        The name of the block of the first call. Later calls name theirs
        ``<name>_2``, ``<name>_3``, and so on. The default is ``LocalModel``
        followed by a number.

    Call arguments
    --------------
    inputs : Stream or list of Stream
        The streams the input function reads. Without an input function, one
        stream.
    activations : Stream or list of Stream
        The memberships. Each has dim ``(n_k,)`` and no sequence. Either the
        activations and the cells all have a window, or none of them does.
        Their windows are equal, or 1: a membership of one sample weighs every
        sample of the cells' window.

    Returns
    -------
    Stream
        The shape of one output cell. For example, a :class:`Fir` of a
        torque window of 10 samples gives ``Shape(S=(), T=1, D=(1,))``.
        Without functions, the result has the shape of the input.

    Raises
    ------
    TypeError
        If a function is a class ("pass a layer instance, shared by every
        cell, or a factory"), or is not callable ("must be a layer or a
        callable"). If the inputs or the activations are not streams ("must be
        Streams").
    ValueError
        If an activation has a sequence or more than one dim axis ("must have
        no sequence, dim (cells,)"). Without an input function, if there is
        more than one input ("there must be one"). With an elementwise input
        function, if there is more than one input ("a cell is one stream").
        If the windows do not match ("they must match, or be 1"; "all have a
        window, or none of them"). If a vectorized function returns another
        number of cells ("one entry per cell"), or the cells differ in shape
        ("every cell must have the same shape").

    See Also
    --------
    Fuzzify : Memberships to sets placed on centers.
    Fir : A local model of a window.
    Linear : A local model of the features.
    EquationLearner : Another block with learned structure.

    Notes
    -----
    For activations :math:`a_1, \dots, a_K` of :math:`n_1, \dots, n_K`
    memberships, the cell of index :math:`(i_1, \dots, i_K)` has the
    membership

    .. math:: \mu_{i_1 \dots i_K} = \prod_{k=1}^{K} a_{k, i_k}

    The cells are numbered row-major, the last index running fastest. With
    two activations, cell :math:`(i, j)` is number :math:`i \, n_2 + j`. With
    :math:`f_i` and :math:`g_i` the input and output functions of cell
    :math:`i`, the result is

    .. math:: y = \sum_{i=0}^{N-1} g_i\left(\mu_i \, f_i(\text{inputs})\right),
        \quad N = \prod_{k=1}^{K} n_k

    A missing function is the identity. The Parameters of the cells are listed
    by :attr:`Modely.parameters`, and are trained, frozen and saved with the
    model.

    Basic Examples
    --------------
    Two cells with the gains 1 and 10, blended by triangles on 0 and 1. At
    ``g = 0.25`` the memberships are 0.75 and 0.25:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import (
    ...     Fir, Fuzzify, Input, Linear, LocalModel, Modely, Output, Parameter,
    ...     Square, Triangular,
    ... )
    >>> x, g = Input("x", tw=1), Input("g", tw=1)
    >>> mu = Fuzzify([0.0, 1.0], function=Triangular)(g)
    >>> gains = [1.0, 10.0]
    >>> def cell(index):
    ...     value = [[gains[index[0]]]]
    ...     return Linear(kernel=Parameter(f"gain{index[0]}", value=value), bias=False)
    >>> y = LocalModel(input_function=cell)(x, mu)
    >>> model = Modely("blend", inputs=[x, g], outputs=[Output("y", y)]).build()
    >>> samples = {"x": np.array([[[2.0]]]), "g": np.array([[[0.25]]])}
    >>> keras.ops.convert_to_numpy(model(samples)["y"])  # 0.75*1*2 + 0.25*10*2
    array([[[6.5]]], dtype=float32)

    Scope
    -----
    A window of inputs, weighted by a membership of one sample or of the same
    window:

    >>> u = Input("u", tw=3)
    >>> LocalModel()(u, mu).shape
    Shape(S=(), T=3, D=(1,))
    >>> LocalModel()(u, Fuzzify([0.0, 1.0], function=Triangular)(Input("gw", tw=3))).shape
    Shape(S=(), T=3, D=(1,))

    Cells of several features:

    >>> LocalModel(Linear(features=2))(x, mu).shape
    Shape(S=(), T=1, D=(2,))

    Several activations multiply into joint cells, numbered row-major. Here
    2 by 3 cells, each with its number as its gain. At ``g = 1`` and
    ``s = 2`` only cell ``(1, 2)``, number 5, is active:

    >>> s = Input("s", tw=1)
    >>> nu = Fuzzify([0.0, 1.0, 2.0], function=Triangular)(s)
    >>> indices = []
    >>> def numbered(index):
    ...     indices.append(index)
    ...     number = float(index[0] * 3 + index[1])
    ...     return Linear(kernel=Parameter(f"c{index[0]}{index[1]}", value=[[number]]), bias=False)
    >>> y2 = LocalModel(numbered)(x, [mu, nu])
    >>> indices
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    >>> model = Modely("grid", inputs=[x, g, s], outputs=[Output("y2", y2)]).build()
    >>> grid = {"x": np.array([[[2.0]]]), "g": np.array([[[1.0]]]), "s": np.array([[[2.0]]])}
    >>> keras.ops.convert_to_numpy(model(grid)["y2"])  # 5 * 2
    array([[[10.]]], dtype=float32)

    Options
    -------
    ``input_function`` as a layer instance is shared by every cell, while a
    factory gives each cell weights of its own:

    >>> shared = LocalModel(Linear(kernel=Parameter("shared", value=[[3.0]]), bias=False))(x, mu)
    >>> [p.name for p in Modely("one", inputs=[x, g], outputs=[Output("y", shared)]).parameters]
    ['shared']
    >>> [p.name for p in Modely("two", inputs=[x, g], outputs=[Output("y", y)]).parameters]
    ['gain0', 'gain1']

    ``output_function``: a weightless layer applied to every weighted cell,
    :math:`(0.75 \cdot 2)^2 + (0.25 \cdot 2)^2`, or a factory with the gains 1
    and 3, :math:`1.5 + 3 \cdot 0.5`:

    >>> squared = LocalModel(output_function=Square())(x, mu)
    >>> scaled = LocalModel(output_function=lambda index: Linear(
    ...     kernel=Parameter(f"out{index[0]}", value=[[1.0 + 2.0 * index[0]]]), bias=False))(x, mu)
    >>> outputs = [Output("squared", squared), Output("scaled", scaled)]
    >>> result = Modely("outputs", inputs=[x, g], outputs=outputs).build()(samples)
    >>> keras.ops.convert_to_numpy(result["squared"]), keras.ops.convert_to_numpy(result["scaled"])
    (array([[[2.5]]], dtype=float32), array([[[3.]]], dtype=float32))

    ``vectorized=True`` computes both cells with one filter of 2 channels:
    the sum of the window, 6, and 10 times its newest sample, 30:

    >>> kernel = Parameter("both", value=[[1.0, 0.0], [1.0, 0.0], [1.0, 10.0]])
    >>> z = LocalModel(Fir(features=2, kernel=kernel, bias=False), vectorized=True)(u, mu)
    >>> model = Modely("vectorized", inputs=[u, g], outputs=[Output("z", z)]).build()
    >>> window = {"u": np.array([[[1.0], [2.0], [3.0]]]), "g": np.array([[[0.25]]])}
    >>> keras.ops.convert_to_numpy(model(window)["z"])  # 0.75*6 + 0.25*30
    array([[[12.]]], dtype=float32)

    ``name`` names the block of each call:

    >>> engine = LocalModel(name="engine")
    >>> calls = [Output("first", engine(x, mu)), Output("second", engine(x, mu))]
    >>> [block.name for block in Modely("car", inputs=[x, g], outputs=calls).models]
    ['engine', 'engine_2']

    Applications
    ------------
    The engine force of a vehicle, one filter of the torque per gear:

    >>> gear = Input("gear", tw=1)
    >>> torque = Input("torque", tw=10)
    >>> gears = Fuzzify([1.0, 2.0, 3.0, 4.0], function=Triangular, bounds=True)(gear)
    >>> force = LocalModel(lambda index: Fir())(torque, gears)
    >>> force.shape
    Shape(S=(), T=1, D=(1,))

    Properties & Relations
    ----------------------
    Without functions the result is the input times the sum of the
    memberships. Triangles add up to one, so it is the input:

    >>> plain = LocalModel()(x, mu)
    >>> result = Modely("plain", inputs=[x, g], outputs=[Output("plain", plain)]).build()(samples)
    >>> keras.ops.convert_to_numpy(result["plain"])
    array([[[2.]]], dtype=float32)

    Possible Issues
    ---------------
    A class is not a function. Pass an instance, shared by every cell, or a
    factory, one layer per cell:

    >>> LocalModel(Fir)
    Traceback (most recent call last):
        ...
    TypeError: LocalModel: the input function is the class Fir; pass a layer instance, ...
    >>> local = LocalModel(lambda index: Fir())

    Without an input function, there is one input to weigh:

    >>> LocalModel()([x, s], mu)
    Traceback (most recent call last):
        ...
    ValueError: ...without an input function the input is weighted as it is, so there must be one, got 2.

    .. versionadded:: 2.0
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
        self._input_function = input_function
        self._output_function = output_function
        self._vectorized = bool(vectorized)
        self.name = next_name("LocalModel") if name is None else name
        # The layers made for the cells, kept so every call of this LocalModel
        # applies the same ones - and so shares their weights.
        self._cells: dict[tuple[str, int], Callable] = {}
        # The calls made, which name the model of each, and the last model.
        self._calls = 0
        self._model: Modely | None = None

    def __call__(self, inputs, activations):
        inputs = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        activations = (
            list(activations)
            if isinstance(activations, (list, tuple))
            else [activations]
        )
        if not all(isinstance(stream, Stream) for stream in inputs + activations):
            raise TypeError(f"{self.name}: inputs and activations must be Streams.")
        _refuse_uncalled([*inputs, *activations])
        for index, activation in enumerate(activations):
            shape = activation.shape
            if shape.seq or shape.dim_rank != 1:
                raise ValueError(
                    f"{self.name}: activation {index} must have no sequence, dim "
                    f"(cells,), got {shape}."
                )

        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        block_inputs = [
            _block_input(stream, f"{call_name}_input{index}")
            for index, stream in enumerate(inputs)
        ]
        block_activations = [
            _block_input(stream, f"{call_name}_activation{index}")
            for index, stream in enumerate(activations)
        ]
        result = self._declare(block_inputs, block_activations, call_name)
        # The functions may read other streams of the graph: the Inputs those
        # read are inputs of the block too, each given itself.
        read = _undeclared_inputs([*block_inputs, *block_activations], [result])
        self._model = Modely(
            call_name,
            inputs=[*block_inputs, *block_activations, *read],
            outputs=[Output(call_name, result)],
        )
        return self._model([*inputs, *activations, *read])

    def _declare(self, inputs, activations, call_name):
        """The cells on the Inputs of the block, blended and summed; their
        nodes are named after ``call_name``."""
        indices = list(np.ndindex(*(activation.dim[0] for activation in activations)))

        # The input functions of every cell, stacked on the last dim axis.
        function = self._input_function
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
            stack = CellStack(name=f"{call_name}_inputs")(values)

        # The memberships, of dim (N,), line up with the last dim axis of the
        # stack, its cell axis, and with its window.
        times = {stream.tw for stream in [*activations, stack]}
        if None in times and len(times) > 1:
            raise ValueError(
                f"{self.name}: the activations and the cells all have a window, "
                f"or none of them; got {[stream.shape for stream in [*activations, stack]]}."
            )
        if len(times - {1}) > 1:
            raise ValueError(
                f"{self.name}: the activations and the cells have windows of "
                f"{sorted(times)} samples; they must match, or be 1."
            )
        mu = (
            activations[0]
            if len(activations) == 1
            else FuzzyProduct(name=f"{call_name}_mu")(activations)
        )
        weighted = CellWeight(name=f"{call_name}_weighted")([stack, mu])

        function = self._output_function
        if self._is_vectorized(function):
            weighted = self._check_cells(function([weighted]), len(indices), "output")
        elif isinstance(function, _ELEMENTWISE):
            weighted = function([weighted])
        elif function is not None:
            outputs = [
                self._cell("out", function, i, index)(
                    [CellSelect(idx=i, name=f"{call_name}_cell{i}")([weighted])]
                )
                for i, index in enumerate(indices)
            ]
            weighted = CellStack(name=f"{call_name}_outputs")(outputs)
        return CellSum(name=f"{call_name}_sum")([weighted])

    def _is_vectorized(self, function) -> TypeGuard[Callable[..., Any]]:
        return (
            self._vectorized
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
