"""Composable Equation Learner block."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from collections.abc import Callable
from typing import Any, cast

import keras
import numpy as np

from nnodely.core.dag import next_name
from nnodely.core.layer import (
    Add,
    Divide,
    Identity,
    Layer,
    Mod,
    Multiply,
    Power,
    Subtract,
)
from nnodely.core.modely import Modely
from nnodely.core.stream import Stream, object_repr
from nnodely.layers.activations import (
    ELU,
    GELU,
    LeakyReLU,
    PReLU,
    ReLU,
    Sigmoid,
    Softplus,
    Swish,
    Tanh,
)
from nnodely.layers.arithmetic import (
    Abs,
    Ceil,
    Deg2Rad,
    Exp,
    Expm1,
    Floor,
    Log,
    Log1p,
    Log10,
    Log2,
    Maximum,
    Minimum,
    Negative,
    Rad2Deg,
    Round,
    Sign,
    Sqrt,
    Square,
)
from nnodely.layers.fuzzify import Gaussian, Rectangular, Triangular
from nnodely.layers.input import Input
from nnodely.layers.linear import Linear, LinearImpl
from nnodely.layers.output import Output
from nnodely.layers.partitions import Concatenate, Select
from nnodely.layers.trigonometric import (
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,
    Cos,
    Cosh,
    Sin,
    Sinh,
    Tan,
)


_NAMED_FUNCTIONS: dict[str, type[Layer]] = {
    "identity": Identity,
    "sin": Sin,
    "cos": Cos,
    "tan": Tan,
    "asin": Asin,
    "acos": Acos,
    "atan": Atan,
    "atan2": Atan2,
    "sinh": Sinh,
    "cosh": Cosh,
    "asinh": Asinh,
    "acosh": Acosh,
    "atanh": Atanh,
    "exp": Exp,
    "expm1": Expm1,
    "log": Log,
    "log1p": Log1p,
    "log10": Log10,
    "log2": Log2,
    "sqrt": Sqrt,
    "square": Square,
    "abs": Abs,
    "sign": Sign,
    "floor": Floor,
    "ceil": Ceil,
    "round": Round,
    "gaussian": Gaussian,
    "triangular": Triangular,
    "rectangular": Rectangular,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "prelu": PReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "swish": Swish,
    "gelu": GELU,
    "softplus": Softplus,
    "add": Add,
    "subtract": Subtract,
    "multiply": Multiply,
    "divide": Divide,
    "power": Power,
    "mod": Mod,
    "maximum": Maximum,
    "minimum": Minimum,
}
_BINARY_LAYER_TYPES = (
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Mod,
    Maximum,
    Minimum,
    Atan2,
)


@dataclass(frozen=True)
class _FunctionSpec:
    function: Any
    arity: int
    label: str


class EquationLearner:
    r"""A learned equation: a projection of the inputs feeds a list of functions, whose values a second projection can combine.

    Syntax
    ------
    ``EquationLearner(functions)(x)``
        The value of each function, one feature each, on learned arguments.
    ``EquationLearner(functions, linear_out=Linear(features=1))(x)``
        A learned combination of those values.
    ``EquationLearner(functions, threshold=t)(x)``
        The same, with every coefficient smaller than ``t`` dropped.
    ``EquationLearner(functions)([x, u])``
        Several inputs, their features joined.

    The input projection, a :class:`Linear`, gives one scalar argument for
    every argument of every function. Each function is evaluated on its
    arguments, and the results, the basis, are joined along the features. An
    output projection, if given, combines them. After training,
    :meth:`EquationLearner.plot` writes the equation the model computes.

    Each call declares the learner in a model of its own, called as a block on
    the given streams. Calling the same learner again applies the same
    projections, with the same weights.

    Parameters
    ----------
    functions : list
        The functions of the basis, in order. Each item is one of:

        - a name, in any case. Of one argument: ``identity``, ``sin``,
          ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``sinh``, ``cosh``,
          ``asinh``, ``acosh``, ``atanh``, ``exp``, ``expm1``, ``log``,
          ``log1p``, ``log10``, ``log2``, ``sqrt``, ``square``, ``abs``,
          ``sign``, ``floor``, ``ceil``, ``round``; the activations ``relu``,
          ``leaky_relu``, ``elu``, ``prelu``, ``sigmoid``, ``tanh``,
          ``swish``, ``gelu``, ``softplus``; the memberships ``gaussian``,
          ``triangular``, ``rectangular``, 1 at 0, of width 1. Of two
          arguments: ``add``, ``subtract``, ``multiply``, ``divide``,
          ``power``, ``mod``, ``maximum``, ``minimum``, ``atan2``.
        - an nnodely layer class, such as :class:`Sin`, made anew; or an
          instance, such as ``Tanh()``. :class:`Maximum`, :class:`Minimum`
          and :class:`Atan2` take two arguments, the others one.
        - a Python function of streams, such as ``lambda a, b: a * b + a``.
          Its arguments are its positional parameters, and it returns a
          stream.
        - a ``(function, arity)`` pair, for a function whose number of
          arguments cannot be read from its signature, such as
          ``lambda *a: a[0] * a[1]``.
    linear_in : Linear, optional
        The input projection. Its ``features`` must equal the total number of
        arguments of the functions. The default is ``Linear`` with that many
        features and a bias, named ``<name>_linear_in``.
    linear_out : Linear, optional
        The output projection of the basis. The default is none: the result
        is the basis.
    threshold : float, optional
        A number ``>= 0``. Every coefficient of the two projections, weight or
        bias, whose magnitude is below it computes as zero. The default,
        ``None``, drops none. It is also the attribute :attr:`EquationLearner.threshold`,
        settable at any time.
    name : str, optional
        The name of the block of the first call. Later calls name theirs
        ``<name>_2``, and so on. The default is ``EquationLearner`` followed
        by a number.

    Call arguments
    --------------
    inputs : Stream or list of Stream
        Streams of one dim axis, ``D=(d,)``, with any window and sequence.
        Several are joined along the features.

    Returns
    -------
    Stream
        ``Shape(S, T, (n,))`` for ``n`` functions, the basis. With
        ``linear_out``, ``Shape(S, T, (features,))``. For example,
        ``Input("x", tw=1)`` with ``["sin", "multiply"]`` gives
        ``Shape(S=(), T=1, D=(2,))``.

    Raises
    ------
    ValueError
        If ``functions`` is not a list of at least one function ("requires at
        least one function"), or a name is unknown ("Unknown EquationLearner
        function. Available names: ..."). If the number of arguments of a
        function cannot be read ("Cannot infer variadic function arity; use
        (function, arity)") or is 0 ("must accept at least one argument"). If
        ``linear_in`` has another number of features ("linear_in.features
        must equal the total number of function arguments"). If ``threshold``
        is negative ("must be zero or positive"). If an input has more than
        one dim axis ("supports one-dimensional dim shapes").
    TypeError
        If a pair is not ``(function, arity)`` ("Function pairs must have the
        form (callable, arity)"), or a function is not callable ("must be
        callable"). If a function returns something other than a stream
        ("must return a Stream"), or an input is not a stream ("inputs must be
        Stream objects").

    See Also
    --------
    Linear : The projections.
    LocalModel : Another block with learned structure.
    Sin : One of the functions.

    Notes
    -----
    With :math:`x` the joined inputs, the arguments are
    :math:`a = W_{in} x + b_{in}`. Function :math:`\phi_j` of arity
    :math:`m_j` reads the next :math:`m_j` arguments, and the result is

    .. math:: f_j = \phi_j(a_{s_j}, \dots, a_{s_j + m_j - 1}), \quad
        y = W_{out} f + b_{out}

    With a threshold :math:`t`, every coefficient :math:`w` of
    :math:`W_{in}, b_{in}, W_{out}, b_{out}` computes as

    .. math:: \tilde w = \begin{cases} w & |w| \ge t \\ 0 & |w| < t \end{cases}

    The Parameters keep their values. The gradient is straight-through: a
    dropped coefficient still trains with the gradient it would have if kept.
    A coefficient that changes sign is not stuck at zero, and one that grows
    past the threshold counts again. Given projections are thresholded too,
    and left as they are. The threshold is saved with the model.

    A recommended use has two phases. Train without a threshold, then set it
    and fine-tune at a lower learning rate, so that the coefficients left
    adjust to those dropped. ``log`` and ``sqrt`` give NaN where their
    argument is negative.

    Basic Examples
    --------------
    The basis ``x`` and ``x * x``, combined with known coefficients:
    :math:`y = 2x + 0.5x^2`, 6 at :math:`x = 2`:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import EquationLearner, Input, Linear, Modely, Output, Parameter, Sin, Tanh
    >>> x = Input("x", tw=1)
    >>> learner = EquationLearner(
    ...     ["identity", "multiply"],
    ...     linear_in=Linear(features=3, kernel=Parameter("w_in", value=[[1.0, 1.0, 1.0]]), bias=False),
    ...     linear_out=Linear(kernel=Parameter("w_out", value=[[2.0], [0.5]]), bias=False),
    ... )
    >>> y = learner(x)
    >>> model = Modely("law", inputs=[x], outputs=[Output("y", y)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[[2.0]]])})["y"])
    array([[[6.]]], dtype=float32)
    >>> learner.plot()
    Eq(y, 0.5*x**2 + 2*x)

    Scope
    -----
    Without ``linear_out`` the result is the basis, one feature per function:

    >>> EquationLearner(["sin", "multiply"])(x).shape
    Shape(S=(), T=1, D=(2,))

    Several inputs are joined, and the window and the sequence are kept:

    >>> v, w = Input("v", dim=2, tw=3), Input("w", tw=3)
    >>> EquationLearner(["sin"], linear_out=Linear(features=2))([v, w]).shape
    Shape(S=(), T=3, D=(2,))
    >>> EquationLearner(["sin", "exp"])(Input("q", seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(2,))

    Options
    -------
    ``functions`` by name, class, instance, Python function and pair. The
    default ``linear_in`` gives 1 + 1 + 1 + 2 + 3 = 8 arguments:

    >>> mixed = EquationLearner(
    ...     ["sin", Sin, Tanh(), lambda a, b: a * b + a, (lambda *a: a[0] * a[1] * a[2], 3)],
    ...     name="mixed",
    ... )
    >>> basis = mixed(x)
    >>> [(p.name, p.shape.dim) for p in Modely("m", inputs=[x], outputs=[Output("f", basis)]).parameters]
    [('mixed_linear_in_kernel', (1, 8)), ('mixed_linear_in_bias', (8,))]

    ``threshold`` drops the coefficients below it, here the 0.5 of
    :math:`x^2`. It can be set on the built model, which computes with it from
    then on:

    >>> learner.threshold = 1.0
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[[2.0]]])})["y"])
    array([[[4.]]], dtype=float32)
    >>> learner.plot()
    Eq(y, 2*x)
    >>> learner.threshold = None

    ``linear_in`` and ``linear_out`` are given in the Basic Example, and
    ``name`` names the default projection above.

    Applications
    ------------
    The friction force of a sliding mass as an unknown law of its velocity,
    with viscous and Coulomb terms among the candidates:

    >>> velocity = Input("velocity", tw=1)
    >>> friction = EquationLearner(["identity", "sign", "square"], linear_out=Linear(features=1))(velocity)
    >>> friction.shape
    Shape(S=(), T=1, D=(1,))

    Properties & Relations
    ----------------------
    The equation :meth:`EquationLearner.plot` returns computes what the model does:

    >>> equation = learner.plot()
    >>> float(equation.rhs.subs("x", 2.0))
    6.0

    Possible Issues
    ---------------
    ``linear_in`` gives one argument per argument of the functions: ``sin``
    reads 1, ``multiply`` 2:

    >>> EquationLearner(["sin", "multiply"], linear_in=Linear(features=2))
    Traceback (most recent call last):
        ...
    ValueError: linear_in.features must equal the total number of function arguments (3), got 2.

    A function with ``*args`` gives no number of arguments. Pass it in a pair:

    >>> EquationLearner([lambda *a: a[0] * a[1]])
    Traceback (most recent call last):
        ...
    ValueError: Cannot infer variadic function arity; use (function, arity).
    >>> learner2 = EquationLearner([(lambda *a: a[0] * a[1], 2)])

    .. versionadded:: 2.0
    """

    def __init__(
        self,
        functions: list,
        *,
        linear_in: Linear | None = None,
        linear_out: Linear | None = None,
        threshold: float | None = None,
        name: str | None = None,
    ):
        if not isinstance(functions, list) or not functions:
            raise ValueError("EquationLearner requires at least one function.")

        self.name = name or next_name("EquationLearner")
        self._functions = list(functions)
        self._function_specs = [_resolve_function(function) for function in functions]
        self._n_arguments = sum(spec.arity for spec in self._function_specs)
        self._threshold = _checked_threshold(threshold)
        # The input projection, made here when not given, so that every call
        # applies the same one, as it does linear_out.
        self._linear_in_template = linear_in or Linear(
            features=self._n_arguments, name=f"{self.name}_linear_in"
        )
        self._linear_out_template = linear_out

        self._linear_in: Linear | None = None
        self._linear_out: Linear | None = None
        self._model: Modely | None = None
        self._calls = 0
        # The variables of the equation plot() writes, one per input feature
        self._variables: list[str] = []
        # The part of each projection its dropped coefficients compute, made
        # on the projection's weights at the first call and applied by every
        # call, and the nodes those applications made
        self._dropped: dict[str, _DroppedTerms] = {}
        self._dropped_nodes: list[_DroppedTerms] = []

        if linear_in is not None and linear_in._features != self._n_arguments:
            raise ValueError(
                "linear_in.features must equal the total number of function "
                f"arguments ({self._n_arguments}), got {linear_in._features}."
            )

    @property
    def threshold(self) -> float | None:
        """The magnitude below which the coefficients of the projections are dropped.

        A number ``>= 0``, or ``None`` to drop none. A coefficient whose
        magnitude equals the threshold is kept. Set it at any time, after
        training too: a built model computes and trains with the new value
        from then on, without being built again, and saves it.

        Raises
        ------
        ValueError
            If it is set to a negative number ("must be zero or positive").

        See Also
        --------
        EquationLearner.plot : The equation without the dropped coefficients.

        Basic Examples
        --------------
        >>> from nnodely import EquationLearner
        >>> learner = EquationLearner(["sin"], threshold=0.01)
        >>> learner.threshold
        0.01
        >>> learner.threshold = None
        >>> learner.threshold is None
        True

        .. versionadded:: 2.0
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float | None) -> None:
        self._threshold = _checked_threshold(threshold)
        for dropped in [*self._dropped.values(), *self._dropped_nodes]:
            dropped.set_threshold(self._threshold)

    def _kept(self, key: str, projected: Linear, stream: Stream, call_name: str):
        """``projected`` without what its coefficients below the threshold
        compute: always in the graph, so that the threshold can change after
        the model is built; with none, it subtracts zero."""
        dropped = self._dropped.get(key)
        if dropped is None:
            weights = projected._weight_nodes()
            dropped = _DroppedTerms(
                self._threshold,
                features=projected._features,
                kernel=weights[0],
                bias=weights[1] if len(weights) > 1 else False,
                axis=projected._axis,
                shared_kernel=projected._shared_kernel,
                name=f"{self.name}_{key}_dropped",
            )
            self._dropped[key] = dropped
        node = dropped([stream])
        self._dropped_nodes.append(node)
        return Subtract(name=f"{call_name}_{key}_kept")([projected, node])

    def __repr__(self) -> str:
        return object_repr(
            self,
            name=self.name,
            functions=[spec.label for spec in self._function_specs],
            linear_in=self._linear_in_template,
            linear_out=self._linear_out_template,
            threshold=self._threshold,
        )

    def __call__(self, inputs):
        streams = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        if not streams or not all(isinstance(stream, Stream) for stream in streams):
            raise TypeError("EquationLearner inputs must be Stream objects.")
        if any(stream.shape.dim_rank != 1 for stream in streams):
            raise ValueError(
                "EquationLearner currently supports one-dimensional dim shapes."
            )

        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        if self._calls == 1:
            self._variables = [
                stream.name if stream.dim[0] == 1 else f"{stream.name}_{index}"
                for stream in streams
                for index in range(stream.dim[0])
            ]

        internal_inputs = []
        internal_streams = []
        for index, stream in enumerate(streams):
            # The stream's shape: its sequence, a dynamic axis declared with
            # -1, and its window.
            seq = tuple(-1 if size is None else size for size in stream.seq)
            internal_input = Input(
                f"{call_name}_input_{index}",
                dim=stream.dim,
                seq=seq or None,
                tw=stream.shape.tw,
            )
            internal_inputs.append(internal_input)
            internal_streams.append(internal_input)

        combined = (
            internal_streams[0]
            if len(internal_streams) == 1
            else Concatenate(axis=0, name=f"{call_name}_inputs")(internal_streams)
        )

        projected = self._linear_in_template([combined])
        if not isinstance(projected, Linear):
            raise TypeError(
                "EquationLearner input projection did not produce a Linear node."
            )
        self._linear_in = projected
        projected = self._kept("linear_in", projected, combined, call_name)

        terms = []
        argument_index = 0
        for function_index, spec in enumerate(self._function_specs):
            arguments = []
            for local_index in range(spec.arity):
                arguments.append(
                    Select(
                        idx=argument_index,
                        axis=0,
                        name=f"{call_name}_{spec.label}_arg{local_index}",
                    )([projected])
                )
                argument_index += 1
            terms.append(
                _apply_function(
                    spec,
                    arguments,
                    name=f"{call_name}_{spec.label}_{function_index}",
                )
            )

        basis = (
            terms[0]
            if len(terms) == 1
            else Concatenate(axis=0, name=f"{call_name}_basis")(terms)
        )

        result = basis
        if self._linear_out_template is not None:
            projected_output = self._linear_out_template([basis])
            if not isinstance(projected_output, Linear):
                raise TypeError(
                    "EquationLearner output projection did not produce a Linear node."
                )
            self._linear_out = projected_output
            result = self._kept("linear_out", projected_output, basis, call_name)

        internal_output = Output(f"{call_name}_output", result)
        self._model = Modely(
            call_name,
            inputs=internal_inputs,
            outputs=[internal_output],
        )
        return self._model(streams)

    def plot(self, precision: int = 4, ax=None) -> Any:
        """Draw the learned equation with matplotlib, and return it as a sympy ``Eq``.

        It is the equation the model computes now: the current coefficients
        of the projections, less those the threshold drops. The equation is
        expanded, and its numbers are rounded to ``precision`` significant
        digits. Its variables are named after the streams of the first call,
        with ``_i`` for each feature of a vector: ``pos_0``, ``pos_1``.

        The functions are written in mathematical notation where there is
        one, the other layers by name. A Python function is written as the
        expression it computes on symbols, when it can. The drawing uses
        matplotlib's TeX renderer, which needs no LaTeX installation. A long
        equation is drawn on several lines.

        Parameters
        ----------
        precision : int, default 4
            The significant digits of the numbers.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. The default draws on a new figure sized to
            the equation.

        Returns
        -------
        sympy.Eq or list of sympy.Eq
            ``y = ...`` with ``linear_out`` of one feature. A list of
            ``y_k = ...`` with several features, or of ``f_j = ...``, the
            basis, without ``linear_out``. ``sympy.latex(eq)`` gives the
            LaTeX source, and a notebook shows the equation typeset.

        Raises
        ------
        RuntimeError
            If the learner was not called, or its model was not built
            ("plot() reads the coefficients of a built model").

        See Also
        --------
        EquationLearner.threshold : Drops the small coefficients.
        Modely.build : Creates the coefficients.

        Basic Examples
        --------------
        The basis ``sin(x)`` and ``x * x``, without ``linear_out``:

        >>> import matplotlib.pyplot as plt
        >>> import sympy
        >>> from nnodely import EquationLearner, Input, Linear, Modely, Output
        >>> x = Input("x", tw=1)
        >>> learner = EquationLearner(["sin", "multiply"], linear_in=Linear(features=3, kernel="ones", bias=False))
        >>> model = Modely("basis", inputs=[x], outputs=[Output("f", learner(x))]).build()
        >>> learner.plot()
        [Eq(f_0, sin(x)), Eq(f_1, x**2)]
        >>> sympy.latex(learner.plot()[1])
        'f_{1} = x^{2}'

        Options
        -------
        ``ax`` draws on axes of your own:

        >>> figure, axes = plt.subplots()
        >>> equations = learner.plot(ax=axes)
        >>> plt.close("all")

        Possible Issues
        ---------------
        The coefficients exist once the model is built:

        >>> EquationLearner(["sin"]).plot()
        Traceback (most recent call last):
            ...
        RuntimeError: ...plot() reads the coefficients of a built model: call the learner on its inputs and build the model first.

        .. versionadded:: 2.0
        """
        import matplotlib.pyplot as plt

        equations = self._equations(precision)
        lines = "\n".join(
            line for equation in equations for line in _drawn_lines(equation)
        )
        style = {
            "ha": "center",
            "va": "center",
            "multialignment": "left",
            "fontsize": 16,
            "math_fontfamily": "cm",
        }
        # On axes even on a figure of its own, sized to the equation: a
        # notebook shows no figure without axes
        figure = None
        if ax is None:
            figure, ax = plt.subplots(figsize=(1.0, 1.0))
            ax.set_position((0.0, 0.0, 1.0, 1.0))
        ax.axis("off")
        text = ax.text(0.5, 0.5, lines, transform=ax.transAxes, **style)
        if figure is not None:
            box = text.get_window_extent()
            figure.set_size_inches(
                (box.width + 40) / figure.dpi, (box.height + 30) / figure.dpi
            )
        return equations[0] if len(equations) == 1 else equations

    def _equations(self, precision: int) -> list:
        """The equations :meth:`EquationLearner.plot` draws, as sympy ``Eq``."""
        import sympy

        if self._linear_in is None or self._linear_in.kernel is None:
            raise RuntimeError(
                f"{self.name}: plot() reads the coefficients of a built model: "
                "call the learner on its inputs and build the model first."
            )

        def projection(linear: Linear, inputs: list) -> list:
            # The coefficients as the projection computes with them
            shapes = ((len(inputs), linear._features), (linear._features,))
            kernel, bias = (
                np.zeros(shape)
                if weight is None
                else np.asarray(
                    keras.ops.convert_to_numpy(weight), dtype=np.float64
                ).reshape(shape)
                for weight, shape in zip((linear.kernel, linear.bias), shapes)
            )
            if self._threshold is not None:
                kernel, bias = (
                    np.where(np.abs(values) < self._threshold, 0.0, values)
                    for values in (kernel, bias)
                )
            return [
                sum(
                    (_number(kernel[i, j]) * value for i, value in enumerate(inputs)),  # type: ignore
                    _number(bias[j]),  # type: ignore
                )
                for j in range(linear._features)
            ]

        variables = [sympy.Symbol(name) for name in self._variables]
        arguments = iter(projection(self._linear_in, variables))
        basis = [
            _symbolic(spec, index)(*(next(arguments) for _ in range(spec.arity)))
            for index, spec in enumerate(self._function_specs)
        ]
        if self._linear_out is None:
            names = [f"f_{j}" for j in range(len(basis))]
            sides = basis
        else:
            sides = projection(self._linear_out, basis)
            names = ["y"] if len(sides) == 1 else [f"y_{k}" for k in range(len(sides))]
        return [
            sympy.Eq(
                sympy.Symbol(name),
                _rounded(
                    sympy.expand(side, power_exp=False, power_base=False, log=False),
                    precision,
                ),
            )
            for name, side in zip(names, sides)
        ]


def _checked_threshold(threshold: float | None) -> float | None:
    if threshold is not None and threshold < 0:
        raise ValueError(
            f"EquationLearner threshold must be zero or positive, got {threshold}."
        )
    return None if threshold is None else float(threshold)


@keras.saving.register_keras_serializable(package="nnodely")
class _DroppedTermsImpl(LinearImpl):
    """A LinearImpl that keeps only the coefficients whose magnitude is below
    the threshold, a variable of its own, so that it can change after the
    model is built. They pass no gradient: what they compute is subtracted
    from the projection, whose weights train as if all were kept."""

    def __init__(self, threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_threshold = threshold

    def get_config(self):
        return {**super().get_config(), "threshold": self.initial_threshold}

    def build(self, input_shape):
        start = self.initial_threshold
        self.threshold = self.add_weight(
            name="threshold",
            shape=(),
            initializer=keras.initializers.Constant(0.0 if start is None else start),
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, inputs):
        x, *weights = inputs
        dropped = [
            keras.ops.stop_gradient(
                keras.ops.where(
                    keras.ops.abs(weight) < self.threshold,
                    weight,
                    keras.ops.zeros_like(weight),
                )
            )
            for weight in weights
        ]
        return super().call([x, *dropped])


class _DroppedTerms(Linear):
    """What the coefficients of a projection below ``threshold`` compute, on
    the projection's own weights: an EquationLearner subtracts it from the
    projection, which leaves the projection without them. None drops none."""

    def __init__(self, threshold: float | None, **settings):
        super().__init__(**settings)
        self._threshold = threshold
        # A setting like the others: it recreates the layer and is saved
        self._properties["threshold"] = threshold

    def set_threshold(self, threshold: float | None) -> None:
        """The threshold from now on, in the built model too."""
        self._threshold = threshold
        self._properties["threshold"] = threshold
        if self._layer is not None:
            self._layer.threshold.assign(0.0 if threshold is None else threshold)

    def build_layer(self):
        dim = tuple(self._preds[0].dim)  # type: ignore[attr-defined]
        return _DroppedTermsImpl(
            threshold=self._threshold,
            features=self._features,
            dim_rank=len(dim),
            axis=self._resolve_axis(dim),
            shared_kernel=self._shared_kernel,
            name=self.name,
        )


def _resolve_function(function) -> _FunctionSpec:
    explicit_arity = None
    if isinstance(function, tuple):
        if len(function) != 2 or not isinstance(function[1], int):
            raise TypeError("Function pairs must have the form (callable, arity).")
        function, explicit_arity = function

    resolved = (
        _NAMED_FUNCTIONS.get(function.lower())
        if isinstance(function, str)
        else function
    )
    if resolved is None:
        available = ", ".join(sorted(_NAMED_FUNCTIONS))
        raise ValueError(
            f"Unknown EquationLearner function. Available names: {available}."
        )

    if isinstance(resolved, Layer):
        arity = 2 if isinstance(resolved, _BINARY_LAYER_TYPES) else 1
        label = type(resolved).__name__.lower()
    elif inspect.isclass(resolved) and issubclass(resolved, Layer):
        arity = 2 if issubclass(resolved, _BINARY_LAYER_TYPES) else 1
        label = resolved.__name__.lower()
    elif callable(resolved):
        # A given arity spares reading the signature, which may not tell it
        arity = _callable_arity(resolved) if explicit_arity is None else explicit_arity
        label = getattr(resolved, "__name__", type(resolved).__name__).lower()
    else:
        raise TypeError("EquationLearner functions must be callable.")

    if explicit_arity is not None:
        arity = explicit_arity
    if arity <= 0:
        raise ValueError("EquationLearner functions must accept at least one argument.")
    return _FunctionSpec(function=resolved, arity=arity, label=label)


def _callable_arity(function) -> int:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot infer function arity; pass the function as (function, arity)."
        ) from exc

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(
        parameter.kind == parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        raise ValueError("Cannot infer variadic function arity; use (function, arity).")
    return len(positional)


def _apply_function(spec: _FunctionSpec, arguments: list[Stream], name: str) -> Stream:
    function = spec.function
    layer = None
    if isinstance(function, Layer):
        layer = function
    elif inspect.isclass(function) and issubclass(function, Layer):
        layer = function(name=name)

    if layer is not None:
        result = layer(arguments)
    else:
        result = cast(Callable[..., Any], function)(*arguments)
    if not isinstance(result, Stream):
        raise TypeError(
            f"EquationLearner function {spec.label!r} must return a Stream, "
            f"got {type(result).__name__}."
        )
    return result


# =============================================================================
# The equation as sympy writes it, for plot()
# =============================================================================
def _number(value: float):
    """``value`` as a sympy number: an integer when it is one, so that a
    coefficient 1 or 0 disappears from the equation."""
    import sympy

    value = float(value)
    return sympy.Integer(int(value)) if value.is_integer() else sympy.Float(value)


def _rounded(expression, precision: int):
    """``expression`` with its numbers rounded to ``precision`` significant
    digits."""
    import sympy

    return expression.xreplace(
        {
            number: _number(float(f"{float(number):.{precision}g}"))
            for number in expression.atoms(sympy.Float)
        }
    )


def _drawn_lines(equation, width: int = 80) -> list[str]:
    """The lines of ``equation`` plot() draws: ``lhs = terms``, as many terms
    on a line as fit in about ``width`` visible characters, the next lines
    indented."""
    import sympy

    line = f"{sympy.latex(equation.lhs)} = "
    lines = []
    for position, term in enumerate(equation.rhs.as_ordered_terms()):
        negative = term.could_extract_minus_sign()
        sign = (" - " if negative else " + ") if position else ("-" if negative else "")
        piece = sign + _for_mathtext(sympy.latex(-term if negative else term))
        if position and _visible_length(line + piece) > width:
            lines.append(line)
            line = r"\qquad" + piece
        else:
            line += piece
    return [f"${line}$" for line in [*lines, line]]


def _for_mathtext(latex: str) -> str:
    """``latex`` with plain delimiters where nothing tall is inside them:
    matplotlib's renderer draws a ``\\left( ... \\right)`` around a single
    line too small, and puts a gap before the braces sympy writes around the
    arguments of a function. A coefficient is set apart from what follows it
    by a thin space, and a modulo, which matplotlib has no ``\\bmod`` for,
    is written out."""
    pairs = [
        (r"{\left(", r"\right)}", "(", ")"),
        (r"\left(", r"\right)", "(", ")"),
        (r"\left|", r"\right|", "|", "|"),
        (r"\left\lfloor", r"\right\rfloor", r"\lfloor", r"\rfloor"),
        (r"\left\lceil", r"\right\rceil", r"\lceil", r"\rceil"),
    ]
    changed = True
    while changed:
        changed = False
        for left, right, plain_left, plain_right in pairs:
            # An innermost pair, around nothing tall or delimited
            pattern = (
                re.escape(left)
                + r"((?:(?!\\left|\\right|\\frac|\\sqrt).)*?)"
                + re.escape(right)
            )
            latex, count = re.subn(
                pattern, lambda match: plain_left + match[1] + plain_right, latex
            )
            changed = changed or count > 0
    latex = latex.replace(r"\bmod", r"\;\mathrm{mod}\;")
    return re.sub(r"(\d) (?=\\[a-zA-Z]|\(|\|)", r"\1\\,", latex)


def _visible_length(latex: str) -> int:
    """About how many characters ``latex`` shows."""
    shown = re.sub(r"\\(left|right|operatorname)|[{}$]", "", latex)
    return len(re.sub(r"\\[a-zA-Z]+", "x", shown))


def _operator(name: str, latex: str):
    """A sympy function written ``latex(arguments)``."""
    import sympy

    def _latex(self, printer, exp=None):
        arguments = ", ".join(printer._print(argument) for argument in self.args)
        written = rf"{latex}{{\left({arguments} \right)}}"
        return written if exp is None else rf"{written}^{{{exp}}}"

    return type(name, (sympy.Function,), {"_latex": _latex})


def _symbolic(spec: _FunctionSpec, index: int):
    """The function of ``spec`` on sympy expressions."""
    import operator

    import sympy

    function = spec.function
    if isinstance(function, Layer) or inspect.isclass(function):
        layer_type = function if inspect.isclass(function) else type(function)
        notation = {
            Identity: lambda a: a,
            Sin: sympy.sin,
            Cos: sympy.cos,
            Tan: sympy.tan,
            Asin: sympy.asin,
            Acos: sympy.acos,
            Atan: sympy.atan,
            Atan2: _operator("atan2", r"\operatorname{atan2}"),
            Sinh: sympy.sinh,
            Cosh: sympy.cosh,
            Asinh: sympy.asinh,
            Acosh: sympy.acosh,
            Atanh: sympy.atanh,
            Tanh: sympy.tanh,
            Sigmoid: _operator("sigmoid", r"\sigma"),
            Exp: sympy.exp,
            Expm1: lambda a: sympy.exp(a) - 1,  # type: ignore
            Log: sympy.log,
            Log1p: lambda a: sympy.log(1 + a),
            Log10: _operator("log10", r"\log_{10}"),
            Log2: _operator("log2", r"\log_{2}"),
            Sqrt: sympy.sqrt,
            Square: lambda a: a**2,
            Abs: sympy.Abs,
            Sign: sympy.sign,
            Floor: sympy.floor,
            Ceil: sympy.ceiling,
            Round: _operator("round", r"\operatorname{round}"),
            Negative: operator.neg,
            Deg2Rad: lambda a: a * sympy.pi / 180,
            Rad2Deg: lambda a: a * 180 / sympy.pi,
            Add: operator.add,
            Subtract: operator.sub,
            Multiply: operator.mul,
            Divide: operator.truediv,
            Power: operator.pow,
            Mod: sympy.Mod,
            Maximum: sympy.Max,
            Minimum: sympy.Min,
        }
        if layer_type in notation:
            return notation[layer_type]
        # Activations, bumps, layers with settings: by name
        name = layer_type.__name__
        return _operator(name, rf"\operatorname{{{name}}}")

    # A Python function: the expression it computes on symbols, when it can,
    # otherwise its name, or f_index for a lambda
    if spec.label.isidentifier():
        written = _operator(
            spec.label, rf"\operatorname{{{spec.label}}}".replace("_", r"\_")
        )
    else:
        written = _operator(f"f_{index}", f"f_{{{index}}}")

    def on_symbols(*arguments):
        try:
            result = function(*arguments)
        except Exception:
            result = None
        return result if isinstance(result, sympy.Expr) else written(*arguments)

    return on_symbols
