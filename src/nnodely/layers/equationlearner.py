"""Composable Equation Learner block."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from collections.abc import Callable
from typing import Any, cast

import keras

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
from nnodely.layers.input import Input
from nnodely.layers.linear import Linear
from nnodely.layers.output import Output
from nnodely.layers.time_ops import Concatenate, Select
from nnodely.layers.trigonometric import Acos, Asin, Atan, Cos, Sin, Tan


_NAMED_FUNCTIONS: dict[str, type[Layer]] = {
    "identity": Identity,
    "sin": Sin,
    "cos": Cos,
    "tan": Tan,
    "asin": Asin,
    "acos": Acos,
    "atan": Atan,
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
_BINARY_LAYER_TYPES = (Add, Subtract, Multiply, Divide, Power)


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
    inspected.
    """

    def __init__(
        self,
        functions: list,
        *,
        linear_in: Linear | None = None,
        linear_out: Linear | None = None,
        name: str | None = None,
    ):
        if not isinstance(functions, list) or not functions:
            raise ValueError("EquationLearner requires at least one function.")

        self.name = name or next_name("EquationLearner")
        self.functions = list(functions)
        self.function_specs = [_resolve_function(function) for function in functions]
        self.n_arguments = sum(spec.arity for spec in self.function_specs)
        self.n_activations = len(self.function_specs)
        self.linear_in_template = linear_in
        self.linear_out_template = linear_out

        self.linear_in: Linear | None = None
        self.linear_out: Linear | None = None
        self.model: Modely | None = None
        self._calls = 0

        if linear_in is not None and linear_in.out_features != self.n_arguments:
            raise ValueError(
                "linear_in.out_features must equal the total number of function "
                f"arguments ({self.n_arguments}), got {linear_in.out_features}."
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

        internal_inputs = []
        internal_streams = []
        for index, stream in enumerate(streams):
            internal_input = Input(
                f"{call_name}_input_{index}",
                dim=stream.dim,
                seq=stream.seq,
            )
            internal_input.shape.time = stream.time
            internal_input.past = stream.time
            internal_input.input = keras.Input(
                shape=internal_input.shape,
                name=internal_input.name,
            )
            internal_inputs.append(internal_input)
            internal_streams.append(internal_input)

        combined = (
            internal_streams[0]
            if len(internal_streams) == 1
            else Concatenate(axis=0, name=f"{call_name}_inputs")(internal_streams)
        )

        linear_in_template = self.linear_in_template or Linear(
            out_features=self.n_arguments,
            name=f"{call_name}_linear_in",
        )
        projected = linear_in_template([combined])
        if not isinstance(projected, Linear):
            raise TypeError(
                "EquationLearner input projection did not produce a Linear node."
            )
        self.linear_in = projected

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
            result = projected_output

        internal_output = Output(f"{call_name}_output", result)
        self.model = Modely(
            call_name,
            inputs=internal_inputs,
            outputs=[internal_output],
        )
        return self.model(streams)


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
        arity = _callable_arity(resolved)
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
