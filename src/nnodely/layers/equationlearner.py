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
    Multiply,
    Power,
    Subtract,
)
from nnodely.core.modely import Modely
from nnodely.core.stream import Stream
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
    Floor,
    Log,
    Log10,
    Negative,
    Sign,
    Sqrt,
)
from nnodely.layers.fuzzify import Gaussian, Rectangular, Triangular
from nnodely.layers.input import Input
from nnodely.layers.linear import Linear, LinearImpl
from nnodely.layers.output import Output
from nnodely.layers.partitions import Concatenate, Select
from nnodely.layers.trigonometric import Acos, Asin, Atan, Atan2, Cos, Sin, Tan


_NAMED_FUNCTIONS: dict[str, type[Layer]] = {
    "identity": Identity,
    "sin": Sin,
    "cos": Cos,
    "tan": Tan,
    "asin": Asin,
    "acos": Acos,
    "atan": Atan,
    "atan2": Atan2,
    "exp": Exp,
    "log": Log,
    "log10": Log10,
    "sqrt": Sqrt,
    "abs": Abs,
    "sign": Sign,
    "floor": Floor,
    "ceil": Ceil,
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
}
_BINARY_LAYER_TYPES = (Add, Subtract, Multiply, Divide, Power, Atan2)


@dataclass(frozen=True)
class _FunctionSpec:
    function: Any
    arity: int
    label: str


class EquationLearner:
    """Build a symbolic function approximator from nnodely blocks.

    The input projection produces one scalar argument for every function
    argument. Each function is evaluated on its assigned arguments, the
    resulting basis terms are concatenated, and an optional output projection
    learns how to combine them.

    Functions may be supplied by name, as nnodely ``Layer`` classes or
    instances, as Python callables, or as ``(function, arity)`` pairs. The
    explicit pair form is useful for callables whose signature cannot be
    inspected. The names are ``identity``, ``sin``, ``cos``, ``tan``,
    ``asin``, ``acos``, ``atan``, ``exp``, ``log``, ``log10``, ``sqrt``,
    ``abs``, ``sign``, ``floor``, ``ceil``, the activations ``relu``,
    ``leaky_relu``, ``elu``, ``prelu``, ``sigmoid``, ``tanh``, ``swish``,
    ``gelu``, ``softplus``, the bumps ``gaussian``, ``triangular``,
    ``rectangular`` (1 at 0, of width 1), and the functions of two arguments
    ``add``, ``subtract``, ``multiply``, ``divide``, ``power``, ``atan2``.
    ``log`` and ``sqrt`` give NaN where their argument is negative.

    After training, :meth:`plot` draws the learned equation and returns it.

    ``threshold`` drops every coefficient of the two projections, weight or
    bias, whose magnitude is below it, so that the learned equation has no
    terms such as ``0.000001 * sin(x)``: they compute as zero, while their
    Parameters keep their values. Training still updates them, with the
    gradient they would have if kept, so a coefficient changing sign is not
    stuck at zero, and one that grows past the threshold counts again. It
    applies to the given projections too, which are left as they are: the
    learner subtracts from each the part its dropped coefficients compute.

    ``learner.threshold`` can be changed at any time, after training too:
    the model computes and trains with the new value from then on, without
    being built again, and saves it. The recommended use is in two phases:
    train without a threshold, then set it and fine-tune at a lower learning
    rate, so that the coefficients left adjust to those dropped::

        model.train(data, epochs=150, lr=0.02)
        learner.threshold = 0.01
        model.train(data, epochs=50, lr=0.002)
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
        self.functions = list(functions)
        self.function_specs = [_resolve_function(function) for function in functions]
        self.n_arguments = sum(spec.arity for spec in self.function_specs)
        self._threshold = _checked_threshold(threshold)
        # The input projection, made here when not given, so that every call
        # applies the same one, as it does linear_out.
        self.linear_in_template = linear_in or Linear(
            features=self.n_arguments, name=f"{self.name}_linear_in"
        )
        self.linear_out_template = linear_out

        self.linear_in: Linear | None = None
        self.linear_out: Linear | None = None
        self.model: Modely | None = None
        self._calls = 0
        # The variables of the equation plot() writes, one per input feature
        self._variables: list[str] = []
        # The part of each projection its dropped coefficients compute, made
        # on the projection's weights at the first call and applied by every
        # call, and the nodes those applications made
        self._dropped: dict[str, _DroppedTerms] = {}
        self._dropped_nodes: list[_DroppedTerms] = []

        if linear_in is not None and linear_in.features != self.n_arguments:
            raise ValueError(
                "linear_in.features must equal the total number of function "
                f"arguments ({self.n_arguments}), got {linear_in.features}."
            )

    @property
    def threshold(self) -> float | None:
        """The magnitude below which the coefficients of the projections are
        dropped; None drops none. Set it at any time: a built model computes
        and trains with the new value from then on."""
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
            weights = projected.preds[1:]
            dropped = _DroppedTerms(
                self._threshold,
                features=projected.features,
                kernel=weights[0],
                bias=weights[1] if len(weights) > 1 else False,
                axis=projected.axis,
                shared_kernel=projected.shared_kernel,
                name=f"{self.name}_{key}_dropped",
            )
            self._dropped[key] = dropped
        node = dropped([stream])
        self._dropped_nodes.append(node)
        return Subtract(name=f"{call_name}_{key}_kept")([projected, node])

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
                sw=stream.shape.time,
            )
            internal_inputs.append(internal_input)
            internal_streams.append(internal_input)

        combined = (
            internal_streams[0]
            if len(internal_streams) == 1
            else Concatenate(axis=0, name=f"{call_name}_inputs")(internal_streams)
        )

        projected = self.linear_in_template([combined])
        if not isinstance(projected, Linear):
            raise TypeError(
                "EquationLearner input projection did not produce a Linear node."
            )
        self.linear_in = projected
        projected = self._kept("linear_in", projected, combined, call_name)

        terms = []
        argument_index = 0
        for function_index, spec in enumerate(self.function_specs):
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
        if self.linear_out_template is not None:
            projected_output = self.linear_out_template([basis])
            if not isinstance(projected_output, Linear):
                raise TypeError(
                    "EquationLearner output projection did not produce a Linear node."
                )
            self.linear_out = projected_output
            result = self._kept("linear_out", projected_output, basis, call_name)

        internal_output = Output(f"{call_name}_output", result)
        self.model = Modely(
            call_name,
            inputs=internal_inputs,
            outputs=[internal_output],
        )
        return self.model(streams)

    def plot(self, precision: int = 4, ax=None):
        """Draw the learned equation with matplotlib, typeset as LaTeX, and
        return it as a sympy ``Eq``: ``y = ...``, a list of ``y_k = ...`` with
        several outputs, or of ``f_j = ...``, the basis, without
        ``linear_out``.

        It is the equation the model computes now, with the current
        coefficients of the projections less those the threshold drops,
        expanded, its numbers rounded to ``precision`` significant digits.
        Its variables are named after the streams of the first call, with
        ``_i`` for each feature of a vector. The functions are written in
        mathematical notation where there is one, the other layers by name,
        and a Python function as the expression it computes on symbols, when
        it can. Draws on ``ax``, or on a new figure sized to the equation, a
        long one on several lines, with matplotlib's TeX renderer, which needs
        no LaTeX installation. ``sympy.latex(eq)`` gives the LaTeX source; a
        notebook shows the returned equation typeset too.
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
        """The equations :meth:`plot` draws, as sympy ``Eq``."""
        import sympy

        if self.linear_in is None or self.linear_in.kernel is None:
            raise RuntimeError(
                f"{self.name}: plot() reads the coefficients of a built model: "
                "call the learner on its inputs and build the model first."
            )

        def projection(linear: Linear, inputs: list) -> list:
            # The coefficients as the projection computes with them
            shapes = ((len(inputs), linear.features), (linear.features,))
            kernel, bias = (
                np.zeros(shape)
                if weight is None
                else np.asarray(
                    keras.ops.convert_to_numpy(weight), dtype=np.float64
                ).reshape(shape)
                for weight, shape in zip((linear.kernel, linear.bias), shapes)
            )
            if self.threshold is not None:
                kernel, bias = (
                    np.where(np.abs(values) < self.threshold, 0.0, values)
                    for values in (kernel, bias)
                )
            return [
                sum(
                    (_number(kernel[i, j]) * value for i, value in enumerate(inputs)),  # type: ignore
                    _number(bias[j]),  # type: ignore
                )
                for j in range(linear.features)
            ]

        variables = [sympy.Symbol(name) for name in self._variables]
        arguments = iter(projection(self.linear_in, variables))
        basis = [
            _symbolic(spec, index)(*(next(arguments) for _ in range(spec.arity)))
            for index, spec in enumerate(self.function_specs)
        ]
        if self.linear_out is None:
            names = [f"f_{j}" for j in range(len(basis))]
            sides = basis
        else:
            sides = projection(self.linear_out, basis)
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
        self.threshold = threshold
        # A setting like the others: it recreates the layer and is saved
        self._properties["threshold"] = threshold

    def set_threshold(self, threshold: float | None) -> None:
        """The threshold from now on, in the built model too."""
        self.threshold = threshold
        self._properties["threshold"] = threshold
        if self._layer is not None:
            self._layer.threshold.assign(0.0 if threshold is None else threshold)

    def build_layer(self):
        dim = tuple(self.preds[0].dim)  # type: ignore[attr-defined]
        return _DroppedTermsImpl(
            threshold=self.threshold,
            features=self.features,
            dim_rank=len(dim),
            axis=self._axis(dim),
            shared_kernel=self.shared_kernel,
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
    by a thin space."""
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
            Tanh: sympy.tanh,
            Sigmoid: _operator("sigmoid", r"\sigma"),
            Exp: sympy.exp,
            Log: sympy.log,
            Log10: _operator("log10", r"\log_{10}"),
            Sqrt: sympy.sqrt,
            Abs: sympy.Abs,
            Sign: sympy.sign,
            Floor: sympy.floor,
            Ceil: sympy.ceiling,
            Negative: operator.neg,
            Deg2Rad: lambda a: a * sympy.pi / 180,
            Add: operator.add,
            Subtract: operator.sub,
            Multiply: operator.mul,
            Divide: operator.truediv,
            Power: operator.pow,
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
