"""Time integration of a rate signal along its own time axis.

``IntegrateStep`` is shaped like ``DerivativeStep`` and inverse to it: the rate
sample of an interval integrates into the value at the end of that interval,
the window's length is preserved, and ``init`` is the value the signal had
just before the window.
"""

from __future__ import annotations

import keras
import numpy as np

from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank
from nnodely.core.stream import Stream

#: Quadrature rules, by the name the model writes. "euler" is the same
#: rectangular rule under the name it has when the window is one sample and
#: the layer is one step of a recurrence.
_SOLVER_RULE = {
    "euler": "rectangular",
    "rectangular": "rectangular",
    "trapezoidal": "trapezoidal",
}


def _validate_dt(dt, owner: str = "IntegrateStep") -> float:
    """The time step, always given explicitly - as in DerivativeStep."""
    if dt is None:
        raise ValueError(
            f"{owner}: dt is required. Pass the time step of the signal being "
            f"integrated, for example {owner}(solver='euler', dt=0.01)(rate)."
        )
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError(f"{owner}: dt must be a number, got {type(dt).__name__}.")
    if dt <= 0:
        raise ValueError(f"{owner}: dt must be positive, got {dt}.")
    return float(dt)


def _validate_init(init, owner: str = "Integrate"):
    """The constant of integration, as configured - a Stream, a number or None."""
    if init is None or isinstance(init, Stream):
        return init
    if isinstance(init, bool) or not isinstance(init, (int, float)):
        raise TypeError(
            f"{owner}: init must be a Stream, a number or None, got "
            f"{type(init).__name__}."
        )
    return init


def _signal_and_init(owner: str, inputs) -> list:
    """The predecessors of a layer called as ``layer(signal)`` or
    ``layer(init, signal)``: the signal, then the initial condition if there
    is one, a number made a Constant."""
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        inputs = tuple(inputs[0])
    if len(inputs) == 1:
        return list(inputs)
    if len(inputs) != 2:
        raise TypeError(
            f"{owner} is called on the signal, or on the initial condition and "
            f"the signal, {owner}(...)(init, signal); got {len(inputs)} arguments."
        )
    init, signal = inputs
    if not isinstance(init, Stream):
        _validate_init(init, owner)
        init = Stream._coerce_operand(float(init))
    return [signal, init]


def _operator_matrix(rule: str, window_length: int, dt: float) -> np.ndarray:
    """The quadrature as the matrix that applies it to a whole window at once.

    Row ``interval`` of the increments holds the weight each rate sample has in
    the step across that interval; summing the increments up to ``i`` gives the
    value at sample ``i``, so the running integral is a single matmul - the
    same shape of operation as the derivative's banded stencil, and one the
    exporters convert without a scan.

    Sample ``i`` of the rate is the rate *of the interval ending at* ``i``,
    which is the interval a backward difference produced it from. The
    trapezoidal rule averages it with the previous sample; the first interval
    of a window has no previous sample and keeps the rectangle.
    """
    increments = np.zeros((window_length, window_length), dtype=np.float64)
    for interval in range(window_length):
        if rule == "trapezoidal" and interval > 0:
            increments[interval, interval - 1] += dt / 2.0
            increments[interval, interval] += dt / 2.0
        else:
            increments[interval, interval] += dt
    return np.cumsum(increments, axis=0).T.astype(np.float32)


@keras.saving.register_keras_serializable(package="nnodely")
class IntegrateImpl(keras.layers.Layer):
    """Running integral of a window, one integrated sample per rate sample."""

    def __init__(self, dt, rule, axis, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dt = float(dt)
        self.rule = rule
        self.axis = int(axis)

    def build(self, input_shape):
        shapes = (
            input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # type: ignore[list-item]
        )
        window_length = int(shapes[0][self.axis])
        operator = _operator_matrix(self.rule, window_length, self.dt)
        self.operator = self.add_weight(
            name="operator",
            shape=operator.shape,
            initializer=keras.initializers.Constant(operator),  # type: ignore[arg-type]
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, init = inputs[0], inputs[1] if len(inputs) > 1 else None
        else:
            x, init = inputs, None
        time_axis = self.axis

        result = keras.ops.tensordot(x, self.operator, axes=[[time_axis], [0]])  # type: ignore[arg-type]
        # tensordot appends the window axis; put it back where time belongs.
        # The axis is addressed from the front because a negative index
        # survives tracing as a negative permutation, which ONNX rejects.
        last_axis = len(result.shape) - 1
        if last_axis != time_axis:
            result = keras.ops.moveaxis(result, last_axis, time_axis)

        if init is not None:
            # An initial condition may leave out the sequence axes, and a
            # scalar one the dim size: adding it broadcasts along both.
            result = result + expand_to_rank(init, len(result.shape))
        return result

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            return tuple(input_shape[0])
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dt": self.dt,
                "rule": self.rule,
                "axis": self.axis,
            }
        )
        return config


class IntegrateStep(Layer):
    """
    Time integration of a rate signal along its own time axis::

        IntegrateStep(solver="euler"|"rectangular"|"trapezoidal", dt=0.01)(rate)
        IntegrateStep(solver="euler", dt=0.01)(init, rate)

    The window's length is preserved: sample ``i`` of the rate is the rate of
    the interval *ending* at ``i`` - the interval ``DerivativeStep``'s backward
    difference produced it from - and integrates into the value at ``i``::

        y[i] = y[i-1] + dt * rate[i]                   # "euler"/"rectangular"
        y[i] = y[i-1] + dt/2 * (rate[i-1] + rate[i])   # "trapezoidal"

    with ``y[-1] = init``. Two things follow from that convention:

    * **A one-sample window is one integration step.** ``IntegrateStep(dt=dt)(
      velocity, acceleration)`` is the state update
      ``velocity + dt * acceleration``, so a recurrence needs no separate form
      of this layer and no ``+`` written around it.

    * **It inverts DerivativeStep exactly**, with the same initial condition and
      no bookkeeping: ``IntegrateStep(dt=dt)(x0, DerivativeStep(dt=dt)(x0,
      x))`` is ``x``, because the rectangular rule sums back exactly
      the increments the backward difference took apart. (The trapezoidal rule
      trades that exactness for second-order accuracy on a smooth rate.)

    ``init``, given first when the layer is called, is the constant of
    integration - the value the signal had at the sample *before* the window,
    the same instant ``DerivativeStep`` reads its own ``init`` at. A Stream
    (typically the state the window continues), or a number (kept as a
    Constant); called on the rate alone, the layer starts from zero. It is
    what turns a relative increment into the absolute quantity mechanical
    models are written in::

        velocity = IntegrateStep(dt=dt)(v0, acceleration)  # Input(..., sw=n)
        position = IntegrateStep(dt=dt)(x0, velocity)

    Both of those come out of one forward pass over the whole window, so a
    multi-step loss on the trajectory needs no ``rollback`` to unroll it.

    ``dt`` is the time step, in the unit the rate is expressed in, and is
    always explicit: nothing is inferred from the inputs, so the same relation
    cannot silently integrate at two different rates.

    The first interval of a window has no earlier rate sample, so
    ``"trapezoidal"`` keeps the rectangle there; over a one-sample window the
    two rules are therefore the same. There is no ``"rk4"`` and no ``"heun"``:
    both name predictor-corrector schemes that re-evaluate the dynamics at a
    *predicted* state, which no quadrature over an observed rate performs.
    """

    def __init__(
        self,
        solver: str = "euler",
        dt: float | None = None,
        name=None,
    ):
        if isinstance(solver, Stream):
            raise TypeError(
                "IntegrateStep is configured first and called on the rate, like "
                "every other layer: IntegrateStep(solver=..., dt=...)(rate)."
            )
        if solver not in _SOLVER_RULE:
            raise ValueError(
                f"IntegrateStep: solver must be one of {sorted(_SOLVER_RULE)}, got {solver!r}."
            )
        self.solver = solver
        self.dt = _validate_dt(dt)
        super().__init__(name=name, solver=self.solver, dt=self.dt)

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------
    def __call__(self, *inputs):
        # The initial condition is a predecessor like any other, after the
        # rate, so that it is reconnected the same way when reloading.
        return super().__call__(_signal_and_init("IntegrateStep", inputs))

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def output_shape(self, *inputs):
        rate = inputs[0]
        if not isinstance(rate, Stream):
            raise TypeError(
                f"{self.name}: IntegrateStep expects a Stream input, got "
                f"{type(rate).__name__}."
            )
        self._needs_window(rate)
        if len(inputs) > 1:
            self._validate_init_shape(rate, inputs[1])
        # One integrated sample per rate sample: the result stays aligned with
        # the signal it came from.
        return rate.shape

    def _validate_init_shape(self, rate, init):
        init_dim = tuple(init.shape.dim)
        _check_single_sample(self.name, init, "window")
        if init_dim not in ((1,), tuple(rate.shape.dim)):
            raise ValueError(
                f"{self.name}: init has dim {init_dim}, which is neither a scalar "
                f"nor the integrated rate's {tuple(rate.shape.dim)}."
            )
        seq, rate_seq = tuple(init.shape.seq), tuple(rate.shape.seq)
        if init.shape.time is None and seq:
            raise ValueError(
                f"{self.name}: init is a value without a window, so it has no "
                f"sequence either, got {seq}."
            )
        if seq not in ((), rate_seq):
            raise ValueError(
                f"{self.name}: init has the sequence {seq}, which is neither "
                f"empty nor the integrated rate's {rate_seq}."
            )

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def build_layer(self):
        return IntegrateImpl(
            dt=self.dt,
            rule=_SOLVER_RULE[self.solver],
            axis=self.shape.time_axis,
            name=self.name,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self):
        return {"name": self.name, "solver": self.solver, "dt": self.dt}

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The initial condition was serialized as this node's second
        # predecessor, so it is rebuilt with the rest of the graph.
        layer = cls(solver=config["solver"], dt=config.get("dt"), name=config["name"])
        if preds:
            return layer(*preds[::-1])
        return layer


def _check_single_sample(owner: str, init, where: str) -> None:
    """Refuse an initial condition that is not one sample: a window of one,
    or a value - a number, a Parameter, a Constant - without a window."""
    if init.shape.time is None and not init._is_value:
        raise ValueError(
            f"{owner}: init has no window, {init.shape}, and is read from data: "
            "declare the Input with sw=1, one sample."
        )
    if init.shape.time not in (None, 1):
        raise ValueError(
            f"{owner}: init is the value just before the {where}, so it must "
            f"carry a single sample, got {init.shape.time}."
        )


class Integrate:
    """Integrate a rate one step at a time, in a :class:`~nnodely.ModelyLoop`.

    ::

        Integrate(solver="euler"|"rectangular"|"trapezoidal", dt=0.01, init=x0)(rate)

    The high-level counterpart of :class:`IntegrateStep`: the rate is one
    sample of a step, ``T=1`` and no sequence, and the integral is an Input
    closed with its update, which the ModelyLoop feeds back with all its other
    closed Inputs, in its one loop - no Loop of its own. The result is the
    value at the end of the step::

        y[k] = y[k-1] + dt * rate[k]                      # "euler", "rectangular"
        y[k] = y[k-1] + dt/2 * (rate[k-1] + rate[k])      # "trapezoidal"

    These are the rules of ``IntegrateStep``, with the same convention for the
    first step, which has no earlier rate sample and keeps the rectangle: laid
    out along a time window instead, the same samples integrate to the same
    values. :class:`Derivative` with the same ``init`` is its inverse, as
    ``DerivativeStep`` is ``IntegrateStep``'s along a window.

    ``init`` is the value just before the first step, as :meth:`Input.close`
    takes it: a number, a Parameter or a Constant, or an Input of one sample
    (a scalar, or the rate's dim), whose first step the ModelyLoop reads;
    ``None`` for zero. Built as a :class:`~nnodely.Modely` instead, nothing
    feeds the integral back: it is read from data, as every closed Input is.
    """

    def __init__(
        self,
        solver: str = "euler",
        dt: float | None = None,
        init: Stream | float | None = None,
        name: str | None = None,
    ):
        from nnodely.core.dag import next_name

        if isinstance(solver, Stream):
            raise TypeError(
                "Integrate is configured first and called on the rate, like "
                "every other block: Integrate(solver=..., dt=...)(rate)."
            )
        if solver not in _SOLVER_RULE:
            raise ValueError(
                f"Integrate: solver must be one of {sorted(_SOLVER_RULE)}, got {solver!r}."
            )
        self.solver = solver
        self.dt = _validate_dt(dt, "Integrate")
        self.init = _validate_init(init)
        self.name = next_name("Integrate") if name is None else name
        self._calls = 0

    def __call__(self, rate):
        from nnodely.layers.input import Input

        if isinstance(rate, (list, tuple)):
            if len(rate) != 1:
                raise ValueError(
                    f"{self.name}: Integrate integrates one rate, got {len(rate)}."
                )
            rate = rate[0]
        if not isinstance(rate, Stream):
            raise TypeError(
                f"{self.name}: Integrate expects a Stream, got {type(rate).__name__}."
            )
        if rate.seq:
            raise ValueError(
                f"{self.name}: Integrate takes the rate of one step of a "
                f"ModelyLoop, and this rate carries a sequence, {rate.shape}. "
                "Declare it one sample per step, an Input with sw=1 and no seq, "
                "and build the model with ModelyLoop."
            )
        if rate.time != 1:
            raise ValueError(
                f"{self.name}: Integrate takes one rate sample per step, but the "
                f"rate carries {rate.time} time samples. Integrate a time window "
                "with IntegrateStep."
            )
        if isinstance(self.init, Stream):
            self._validate_init_shape(rate, self.init)

        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        dim = tuple(rate.dim)
        state = Input(f"{call_name}_state", dim=dim, sw=1)
        if _SOLVER_RULE[self.solver] == "trapezoidal":
            # The trapezoid averages the rate with the previous one; the first
            # step has none, and keeps the rectangle.
            previous = Input(f"{call_name}_previous_rate", dim=dim, sw=1)
            first = Input(f"{call_name}_first", sw=1)
            previous.close(rate)
            first.close(first * 0.0, init=1.0)
            rate = (previous + (rate - previous) * first + rate) * 0.5
        step = IntegrateStep(solver="euler", dt=self.dt, name=f"{call_name}_step")(
            state, rate
        )
        return state.close(step, init=self.init)

    def _validate_init_shape(self, rate, init):
        dim = tuple(rate.dim)
        _check_single_sample(self.name, init, "first step")
        if tuple(init.dim) not in (dim, (1,)) or (
            tuple(init.dim) == (1,) and len(dim) > 1
        ):
            raise ValueError(
                f"{self.name}: init has dim {tuple(init.dim)}, which is neither "
                f"the rate's {dim} nor, for a one-axis dim, a scalar."
            )
        if init.seq:
            raise ValueError(
                f"{self.name}: init has the sequence {tuple(init.seq)}, but it is "
                "the value of one step, which has none."
            )
