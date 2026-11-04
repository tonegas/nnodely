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
from nnodely.layers.input import Input, _check_history, _declared_seq, _validate_init

#: Quadrature rules, by the name the model writes. "euler" is the same
#: rectangular rule under the name it has when the window is one sample and
#: the layer is one step of a recurrence, as in ``Ode``.
_METHOD_RULE = {
    "euler": "rectangular",
    "rectangular": "rectangular",
    "trapezoidal": "trapezoidal",
}

#: What a layer of each rule is called on before the rate: the value of the
#: integral before the window, and the rate before it, which the trapezoid
#: averages with the first one.
_HISTORY = {
    "rectangular": ("init",),
    "trapezoidal": ("init", "init_rate"),
}


def _validate_dt(dt, owner: str = "IntegrateStep") -> float:
    """The time step, always given explicitly."""
    if dt is None:
        raise ValueError(
            f"{owner}: dt is required. Pass the time step of the signal, for "
            f"example {owner}(dt=0.01)."
        )
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError(f"{owner}: dt must be a number, got {type(dt).__name__}.")
    if dt <= 0:
        raise ValueError(f"{owner}: dt must be positive, got {dt}.")
    return float(dt)


def _validate_method(method, owner: str) -> str:
    if isinstance(method, Stream):
        raise TypeError(
            f"{owner} is configured first and called on the rate, like every "
            f"other layer: {owner}(method=..., dt=...)(rate)."
        )
    if method not in _METHOD_RULE:
        raise ValueError(
            f"{owner}: method must be one of {sorted(_METHOD_RULE)}, got {method!r}."
        )
    return method


def _history_and_signal(owner: str, inputs, history: tuple[str, ...]) -> list:
    """The predecessors of a layer called as ``layer(signal)`` or
    ``layer(*history, signal)``: the signal, then what it starts from - up to
    the ``history`` it reads, in that order - a number made a Constant."""
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        inputs = tuple(inputs[0])
    if not 1 <= len(inputs) <= len(history) + 1:
        given = ", ".join((*history, "signal"))
        raise TypeError(
            f"{owner} is called on the signal, or on {given} - the samples "
            f"before the first, then the signal - {owner}(...)({given}); got "
            f"{len(inputs)} arguments."
        )
    *before, signal = inputs
    values = []
    for value, name in zip(before, history):
        if not isinstance(value, Stream):
            value = Stream._coerce_operand(_validate_init(owner, value, name))
        values.append(value)
    return [signal, *values]


def _operator_matrix(rule: str, window_length: int, dt: float) -> np.ndarray:
    """The quadrature as the matrix that applies it to a whole window at once.

    Row ``interval`` of the increments holds the weight each rate sample has in
    the step across that interval; summing the increments up to ``i`` gives the
    value at sample ``i``, so the running integral is a single matmul - the
    same shape of operation as the derivative's banded stencil, and one the
    exporters convert without a scan.

    Sample ``i`` of the rate is the rate *of the interval ending at* ``i``,
    which is the interval a backward difference produced it from. The
    trapezoidal rule averages it with the previous sample; the previous
    sample of the first interval is the rate before the window, which the
    layer adds with the initial condition, ``dt / 2`` of it to every sample.
    """
    increments = np.zeros((window_length, window_length), dtype=np.float64)
    for interval in range(window_length):
        if rule == "trapezoidal":
            increments[interval, interval] += dt / 2.0
            if interval > 0:
                increments[interval, interval - 1] += dt / 2.0
        else:
            increments[interval, interval] += dt
    return np.cumsum(increments, axis=0).T.astype(np.float32)


@keras.saving.register_keras_serializable(package="nnodely")
class IntegrateImpl(keras.layers.Layer):
    """Running integral of a window, one integrated sample per rate sample.

    Called on the rate, then the initial condition and the rate before the
    window, if given.
    """

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
        x, *history = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        time_axis = self.axis

        result = keras.ops.tensordot(x, self.operator, axes=[[time_axis], [0]])  # type: ignore[arg-type]
        # tensordot appends the window axis; put it back where time belongs.
        # The axis is addressed from the front because a negative index
        # survives tracing as a negative permutation, which ONNX rejects.
        last_axis = len(result.shape) - 1
        if last_axis != time_axis:
            result = keras.ops.moveaxis(result, last_axis, time_axis)

        # An initial condition may leave out the sequence axes, and a scalar
        # one the dim size: adding it broadcasts along both.
        rank = len(result.shape)
        if history:
            result = result + expand_to_rank(history[0], rank)
        if len(history) > 1:
            result = result + (self.dt / 2.0) * expand_to_rank(history[1], rank)
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

        IntegrateStep(method="euler"|"rectangular"|"trapezoidal", dt=0.01)(rate)
        IntegrateStep(method="euler", dt=0.01)(init, rate)
        IntegrateStep(method="trapezoidal", dt=0.01)(init, init_rate, rate)

    The window's length is preserved: sample ``i`` of the rate is the rate of
    the interval *ending* at ``i`` - the interval ``DerivativeStep``'s backward
    difference produced it from - and integrates into the value at ``i``::

        y[i] = y[i-1] + dt * rate[i]                   # "euler"/"rectangular"
        y[i] = y[i-1] + dt/2 * (rate[i-1] + rate[i])   # "trapezoidal"

    with ``y[-1] = init`` and ``rate[-1] = init_rate``. Two things follow
    from that convention:

    * **A one-sample window is one integration step.** ``IntegrateStep(dt=dt)(
      velocity, acceleration)`` is the state update
      ``velocity + dt * acceleration``, so a recurrence needs no separate form
      of this layer and no ``+`` written around it; with the trapezoidal rule,
      ``(velocity, previous_acceleration, acceleration)`` is the trapezoid.

    * **It inverts DerivativeStep exactly**, with the same initial condition and
      no bookkeeping: ``IntegrateStep(dt=dt)(x0, DerivativeStep(dt=dt)(x0,
      x))`` is ``x``, because the rectangular rule sums back exactly
      the increments the backward difference took apart. (The trapezoidal rule
      trades that exactness for second-order accuracy on a smooth rate.)

    The layer is called on the samples before the window, then the rate; a
    sample left out is zero, a number is kept as a Constant:

    * ``init`` is the constant of integration - the value the signal had at
      the sample *before* the window, the same instant ``DerivativeStep``
      reads its own ``init`` at. Typically the state the window continues. It
      is what turns a relative increment into the absolute quantity
      mechanical models are written in::

          velocity = IntegrateStep(dt=dt)(v0, acceleration)  # Input(..., sw=n)
          position = IntegrateStep(dt=dt)(x0, velocity)

    * ``init_rate``, for the trapezoidal rule only, is the rate at the sample
      before the window, which the first trapezoid averages with the first
      rate. Without it the rate before the window is zero, as when the signal
      starts at rest - the history ``DerivativeStep`` assumes too.

    Each is one sample: a window of one, or a value without a window, of a
    scalar dim or the rate's, and no sequence or the rate's. Both integrals
    above come out of one forward pass over the whole window, so a multi-step
    loss on the trajectory needs no rollout to unroll it.

    ``dt`` is the time step, in the unit the rate is expressed in, and is
    always explicit: nothing is inferred from the inputs, so the same relation
    cannot silently integrate at two different rates.

    There is no ``"rk4"`` and no ``"heun"``: both name predictor-corrector
    schemes that re-evaluate the dynamics at a *predicted* state, which no
    quadrature over an observed rate performs; :func:`~nnodely.Ode` does.
    """

    def __init__(
        self,
        method: str = "euler",
        dt: float | None = None,
        name=None,
    ):
        self._method = _validate_method(method, "IntegrateStep")
        self._dt = _validate_dt(dt)
        super().__init__(name=name, method=self._method, dt=self._dt)

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------
    def __call__(self, *inputs):
        # The samples before the window are predecessors like any other,
        # after the rate, so that they are reconnected the same way when
        # reloading.
        history = _HISTORY[_METHOD_RULE[self._method]]
        return super().__call__(_history_and_signal("IntegrateStep", inputs, history))

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def output_shape(self, *inputs):
        rate, *history = inputs
        if not isinstance(rate, Stream):
            raise TypeError(
                f"{self.name}: IntegrateStep expects a Stream input, got "
                f"{type(rate).__name__}."
            )
        self._needs_window(rate)
        for value, name in zip(history, _HISTORY[_METHOD_RULE[self._method]]):
            _check_history(self.name, value, rate, 1, name)
        # One integrated sample per rate sample: the result stays aligned with
        # the signal it came from.
        return rate.shape

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def build_layer(self):
        return IntegrateImpl(
            dt=self._dt,
            rule=_METHOD_RULE[self._method],
            axis=self.shape.time_axis,
            name=self.name,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self):
        return {"name": self.name, "method": self._method, "dt": self._dt}

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The samples before the window were serialized after the rate, so
        # they are rebuilt with the rest of the graph.
        layer = cls(method=config["method"], dt=config["dt"], name=config["name"])
        if preds:
            return layer(*preds[1:], *preds[:1])
        return layer


class Integrate:
    """Integrate a rate one step at a time, in a :class:`~nnodely.ModelyLoop`
    or a :class:`~nnodely.Loop`.

    ::

        Integrate(method="euler"|"rectangular"|"trapezoidal", dt=0.01, init=x0)(rate)
        Integrate(method="trapezoidal", dt=0.01, init=x0, init_rate=r0)(rate)

    The high-level counterpart of :class:`IntegrateStep`: the rate is one
    sample of a step, ``T=1``, and the integral is a hidden Input named after
    the block and the rate, ``<name>_<rate>`` (``Integrate3_u`` for
    ``Integrate(dt=dt)(u)``), fed back with the result, which the loop feeds
    back with all its other fed-back Inputs, in its one rollout - no Loop of
    its own. The trapezoidal rule also keeps the rate of the previous step in
    a hidden Input, ``<name>_<rate>_rate``. The result, named after the
    block, is the value at the end of the step::

        y[k] = y[k-1] + dt * rate[k]                      # "euler", "rectangular"
        y[k] = y[k-1] + dt/2 * (rate[k-1] + rate[k])      # "trapezoidal"

    These are the rules of ``IntegrateStep``, laid out step by step: along a
    time window instead, the same samples integrate to the same values.
    :class:`Derivative` with the same ``init`` is its inverse, as
    ``DerivativeStep`` is ``IntegrateStep``'s along a window.

    ``init`` is the integral just before the first step, ``y[-1]``, and
    ``init_rate``, for the trapezoidal rule only, the rate before it,
    ``rate[-1]``: each the only initial value of its hidden Input, which
    cannot be named, so no loop gives it another. A number, a Parameter or a
    Constant, or an Input of one sample (a scalar, or the rate's dim and
    sequence) - an input of the step, whose first step a ModelyLoop reads and
    to which a Loop binds a stream of its outer graph; ``None`` for zero, a
    rate at rest before the start. Built as a :class:`~nnodely.Modely`
    instead, nothing feeds them back: the step reads ``init`` and
    ``init_rate`` in their place, as the first step of a loop does, and no
    Input of its own.
    """

    def __init__(
        self,
        method: str = "euler",
        dt: float | None = None,
        init: Stream | float | None = None,
        init_rate: Stream | float | None = None,
        name: str | None = None,
    ):
        from nnodely.core.dag import next_name

        self._method = _validate_method(method, "Integrate")
        self._dt = _validate_dt(dt, "Integrate")
        self._init = _validate_init("Integrate", init)
        if init_rate is not None and _METHOD_RULE[method] != "trapezoidal":
            raise ValueError(
                "Integrate: init_rate is the rate before the first step, which "
                f"only the trapezoidal method reads, not {method!r}."
            )
        self._init_rate = _validate_init("Integrate", init_rate, "init_rate")
        self.name = next_name("Integrate") if name is None else name
        self._calls = 0

    def __call__(self, rate):
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
        if rate.time is None:
            raise ValueError(
                f"{self.name}: Integrate acts on the window of its stream, and "
                f"{rate.name!r} has none, {rate.shape}. Declare the Input with a "
                "window, sw=1 for one sample per step."
            )
        if rate.time != 1:
            raise ValueError(
                f"{self.name}: Integrate takes one rate sample per step, but the "
                f"rate carries {rate.time} time samples. Integrate a time window "
                "with IntegrateStep."
            )
        for value, name in ((self._init, "init"), (self._init_rate, "init_rate")):
            if isinstance(value, Stream):
                _check_history(self.name, value, rate, 1, name)

        # Every call integrates a rate of its own, so each has its own state.
        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"

        def hidden(name):
            node = Input(name, dim=rate.dim, seq=_declared_seq(rate), sw=1)
            node._hidden = True
            return node

        # Named after the rate it integrates, as the Input a hand-written
        # feedback would read it from.
        integral = hidden(f"{call_name}_{rate.name}")
        history = [integral]
        if _METHOD_RULE[self._method] == "trapezoidal":
            previous = hidden(f"{call_name}_{rate.name}_rate")
            previous._feed_back(rate, init=self._init_rate)
            history.append(previous)
        value = IntegrateStep(method=self._method, dt=self._dt, name=call_name)(
            *history, rate
        )
        return integral._feed_back(value, init=self._init)
