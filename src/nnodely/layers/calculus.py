"""Calculus with respect to time: the integral and the derivative of a Stream.

They are configured first and then called on the Stream, like every other
layer::

    IntegrateStep(dt=0.1)(rate)                 # integral of rate   over its window
    Integrate(dt=0.1)(rate)                     # integral of rate   step by step
    DerivativeStep(dt=0.1)(fun)                 # d fun / d t        over its window
    Derivative(dt=0.1)(fun)                     # d fun / d t        step by step

Each integral is the inverse of the derivative beside it. ``IntegrateStep``
integrates along the rate's own window and keeps its length: the rate sample
of an interval integrates into the value at the end of that interval, and
``init`` is the value the signal had just before the window.
``DerivativeStep`` is a causal finite difference along ``fun``'s own window
that keeps its length: one derivative per sample, the oldest one reading the
``init`` condition. ``Integrate`` and ``Derivative`` are the same rules across
the steps of a ModelyLoop. The derivative with respect to an Input is
:class:`~nnodely.layers.gradient.Gradient`.
"""

from __future__ import annotations

import keras
import numpy as np

from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank, slice_axis
from nnodely.core.stream import Stream
from nnodely.layers.input import (
    Input,
    _check_history,
    _declared_seq,
    _validate_init,
)


# ======================================================================
# Integration
# ======================================================================
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


def _quadrature_matrix(rule: str, window_length: int, dt: float) -> np.ndarray:
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
        operator = _quadrature_matrix(self.rule, window_length, self.dt)
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
        tw_axis = self.axis

        result = keras.ops.tensordot(x, self.operator, axes=[[tw_axis], [0]])  # type: ignore[arg-type]
        # tensordot appends the window axis; put it back where tw belongs.
        # The axis is addressed from the front because a negative index
        # survives tracing as a negative permutation, which ONNX rejects.
        last_axis = len(result.shape) - 1
        if last_axis != tw_axis:
            result = keras.ops.moveaxis(result, last_axis, tw_axis)

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
    Time integration of a rate signal along its own tw axis::

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

          velocity = IntegrateStep(dt=dt)(v0, acceleration)  # Input(..., tw=n)
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
            axis=self.shape.tw_axis,
            name=self.name,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The samples before the window were serialized after the rate, so
        # they are rebuilt with the rest of the graph.
        layer = super().from_config(config)
        return layer(*preds[1:], *preds[:1]) if preds else layer


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
        dt: float,
        method: str = "euler",
        init: Stream | float | None = None,
        init_rate: Stream | float | None = None,
        name: str | None = None,
    ):
        from nnodely.core.dag import next_name

        self._dt = _validate_dt(dt, "Integrate")
        self._method = _validate_method(method, "Integrate")
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
        if rate.tw is None:
            raise ValueError(
                f"{self.name}: Integrate acts on the window of its stream, and "
                f"{rate.name!r} has none, {rate.shape}. Declare the Input with a "
                "window, tw=1 for one sample per step."
            )
        if rate.tw != 1:
            raise ValueError(
                f"{self.name}: Integrate takes one rate sample per step, but the "
                f"rate carries {rate.tw} time samples. Integrate a time window "
                "with IntegrateStep."
            )
        for value, name in ((self._init, "init"), (self._init_rate, "init_rate")):
            if isinstance(value, Stream):
                _check_history(self.name, value, rate, 1, name)

        # Every call integrates a rate of its own, so each has its own state.
        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"

        def hidden(name):
            node = Input(name, dim=rate.dim, seq=_declared_seq(rate), tw=1)
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


# ======================================================================
# Differentiation
# ======================================================================
# ----------------------------------------------------------------------
# Finite-difference stencils
# ----------------------------------------------------------------------
def _stencil_coefficients(poly_order: int, dt: float) -> np.ndarray:
    """Weights of the backward difference of order ``poly_order``.

    They give the slope, at the newest sample, of the polynomial of degree
    ``poly_order`` through the ``poly_order + 1`` newest samples - the
    backward differentiation formula: ``(-1, 1)/dt`` for 1, ``(1, -4, 3)/(2
    dt)`` for 2, ``(-2, 9, -18, 11)/(6 dt)`` for 3, each exact on a polynomial
    of its degree. The weights are ordered oldest -> newest, as the samples of
    a window are.
    """
    offsets = np.arange(-poly_order, 1, dtype=np.float64)
    design = np.vander(offsets, poly_order + 1, increasing=True)
    # Row 1 of the inverse gives the polynomial's coefficient of t: its slope
    # at the newest sample, t = 0.
    return np.linalg.inv(design)[1] / float(dt)


def _difference_matrix(coefficients, window_length: int) -> np.ndarray:
    """The stencil as the matrix that applies it to a whole window at once.

    Sliding the stencil over a window is a banded matrix product - the
    discrete differentiation operator of the window - so the layer runs as a
    single matmul rather than a scan, and differentiates (and exports) like
    any other linear operation.
    """
    past = len(coefficients) - 1
    matrix = np.zeros((past + window_length, window_length), dtype=np.float32)
    for column in range(window_length):
        for offset, coefficient in enumerate(coefficients):
            matrix[column + offset, column] = coefficient
    return matrix


# ----------------------------------------------------------------------
# Finite difference with respect to time
# ----------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="nnodely")
class TimeDerivativeImpl(keras.layers.Layer):
    """Causal finite difference of a whole window, one derivative per sample.

    The stencil needs ``len(coefficients) - 1`` samples of history to produce
    the oldest derivative of the window; they come from the initial condition
    (a second input) or are taken as zero. Prepending them and applying the
    banded operator matrix makes the layer a single matmul over the padded
    window.
    """

    def __init__(
        self,
        coefficients,
        axis: int,
        init_tw: int = 1,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.axis = int(axis)
        self.init_tw = int(init_tw)

    @property
    def past(self) -> int:
        """Samples of history the stencil reads before the window's first."""
        return len(self.coefficients) - 1

    def build(self, input_shape):
        shapes = (
            input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # type: ignore[list-item]
        )
        window_length = int(shapes[0][self.axis])
        operator = _difference_matrix(self.coefficients, window_length)
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
        tw_axis = self.axis

        # One sample of zeros shaped like the window's first: it both stands
        # in for a missing initial condition and gives a given one the axes it
        # may be broadcasting over - a scalar initial condition carries no dim
        # size, and one may leave out the sequence axes.
        reference = keras.ops.zeros_like(slice_axis(x, tw_axis, slice(0, 1)))
        if init is None:
            history = reference
        else:
            history = expand_to_rank(init, len(reference.shape)) + reference
        if self.init_tw != self.past:
            history = keras.ops.repeat(history, self.past, axis=tw_axis)

        padded = keras.ops.concatenate([history, x], axis=tw_axis)
        result = keras.ops.tensordot(padded, self.operator, axes=[[tw_axis], [0]])  # type: ignore[arg-type]
        # tensordot appends the window axis; put it back where tw belongs.
        # The axis is addressed from the front because a negative index
        # survives tracing as a negative permutation, which ONNX rejects.
        last_axis = len(result.shape) - 1
        if last_axis == tw_axis:  # nothing follows tw, so it already is
            return result
        return keras.ops.moveaxis(result, last_axis, tw_axis)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            return tuple(input_shape[0])
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "coefficients": self.coefficients,
                "axis": self.axis,
                "init_tw": self.init_tw,
            }
        )
        return config


# ----------------------------------------------------------------------
# Symbolic layers
# ----------------------------------------------------------------------
class _FiniteDifference(Layer):
    """What :class:`DerivativeStep` and :class:`Derivative` share: the time
    step ``dt``, the backward difference of order ``poly_order``, and the
    initial condition ``init``, the sample(s) just before the first one
    differentiated.
    """

    def __init__(
        self,
        dt: float | None = None,
        poly_order: int = 1,
        name=None,
        **properties,
    ):
        layer = type(self).__name__
        self._dt = _validate_dt(dt, layer)
        if (
            isinstance(poly_order, bool)
            or not isinstance(poly_order, (int, np.integer))
            or poly_order < 1
        ):
            raise ValueError(
                f"{layer}: poly_order is the degree of the polynomial through "
                "the poly_order + 1 newest samples, a positive integer; got "
                f"{poly_order!r}."
            )
        self._poly_order = int(poly_order)
        super().__init__(
            name=name, dt=self._dt, poly_order=self._poly_order, **properties
        )

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------
    @property
    def past(self) -> int:
        """Samples of history the stencil needs before the first one."""
        return self._poly_order

    def _coefficients(self) -> np.ndarray:
        return _stencil_coefficients(self._poly_order, self._dt)

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def _check_function(self, function):
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: {type(self).__name__} expects a Stream input, got "
                f"{type(function).__name__}."
            )

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def call(self, xs):
        signature = self.input_signature(xs)
        stale = self._layer_signature is not None and self._layer_signature != signature
        if self._layer is None or stale:
            self._layer = self.build_layer()
        self._layer_signature = signature
        return self._layer(xs if len(xs) > 1 else xs[0])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The initial condition was serialized after the signal, so it is
        # rebuilt with the rest of the graph: the layer is called on it, then
        # on the signal.
        layer = super().from_config(config)
        return layer(*preds[1:], *preds[:1]) if preds else layer


class DerivativeStep(_FiniteDifference):
    """Time derivative of a Stream, along its window::

        DerivativeStep(dt=0.1)(x)                          # dx/dt, from 0
        DerivativeStep(dt=0.1)(x0, x)                      # dx/dt, from x0
        DerivativeStep(dt=0.1)(DerivativeStep(dt=0.1)(x))  # d2x/dt2

    The result has the window of ``x``: one derivative per sample, computed
    from that sample and the ones before it.

    ``dt``
        The time between two samples.
    ``poly_order``
        How many samples before the current one each derivative reads, and
        so the degree of the curves it is exact on:

        * ``1`` (default) - ``(x[i] - x[i-1]) / dt``, exact on a line;
        * ``2`` - ``(3 x[i] - 4 x[i-1] + x[i-2]) / (2 dt)``, exact on a parabola;
        * ``3`` - ``(11 x[i] - 18 x[i-1] + 9 x[i-2] - 2 x[i-3]) / (6 dt)``,
          exact on a cubic.

        A higher order is more accurate on smooth signals, but amplifies
        noise more: keep ``1`` for measured signals.
    ``init``, the first argument of the call
        The value of ``x`` before the window, which the first derivatives
        read: a number or a Stream, one sample (repeated as needed) or
        ``poly_order`` samples, oldest first. Zero when left out.

    With ``poly_order=1``, :class:`IntegrateStep` with the same ``init``
    gives ``x`` back. For one step at a time in a ModelyLoop, see
    :class:`Derivative`; with respect to an Input, :class:`Gradient`.
    """

    def __init__(self, dt: float | None = None, poly_order: int = 1, name=None):
        # Its own arguments only: one it does not take is refused, rather than
        # kept as a setting that changes nothing.
        super().__init__(dt=dt, poly_order=poly_order, name=name)

    def __call__(self, *inputs):
        # The initial condition is a predecessor like any other, after the
        # signal, so that it is reconnected the same way when reloading.
        return super().__call__(
            _history_and_signal("DerivativeStep", inputs, ("init",))
        )

    def output_shape(self, *inputs):
        function, *history = inputs
        self._check_function(function)
        self._needs_window(function)
        if history:
            _check_history(self.name, history[0], function, self.past)
        # One derivative per sample: the window is preserved, so the result
        # stays aligned with the signal it came from.
        return function.shape

    def build_layer(self):
        init_tw = 1
        if len(self._preds) > 1:
            init_node = self._preds[1]
            if isinstance(init_node, Stream):
                # A value without a window is one sample.
                init_tw = init_node.shape.tw or 1
        return TimeDerivativeImpl(
            coefficients=self._coefficients(),
            axis=self.shape.tw_axis,  # type: ignore[arg-type]
            init_tw=init_tw,
            name=self.name,
        )


class Derivative(_FiniteDifference):
    """Time derivative of a Stream, one step at a time, in a
    :class:`~nnodely.ModelyLoop`::

        Derivative(dt=0.1)(x)                        # dx/dt, from 0
        Derivative(dt=0.1, init=x0)(x)               # dx/dt, from x0
        Derivative(dt=0.1)(Derivative(dt=0.1)(x))    # d2x/dt2

    ``x`` is one sample per step, ``tw=1``. The layer remembers the last
    ``poly_order`` samples of ``x`` from one step to the next, and gives the
    same values as :class:`DerivativeStep` on a window of the same samples.

    ``dt``, ``poly_order``
        As in :class:`DerivativeStep`.
    ``init``
        The value of ``x`` before the first step: a number, a Parameter, a
        Constant or an Input, one sample (repeated as needed) or
        ``poly_order`` samples, oldest first. Zero when left out.

    The samples it remembers are a hidden Input, ``<name>_<x>``, that the
    loop feeds back. In a plain :class:`~nnodely.Modely`, which has no steps,
    they are always ``init``. With ``poly_order=1``, :class:`Integrate` with
    the same ``init`` gives ``x`` back.
    """

    def __init__(
        self,
        dt: float | None = None,
        poly_order: int = 1,
        init: Stream | float | None = None,
        name=None,
    ):
        self._init = _validate_init("Derivative", init)
        self._calls = 0
        super().__init__(
            dt=dt,
            poly_order=poly_order,
            name=name,
            init=self._init,
        )

    def __call__(self, function):
        if isinstance(function, (list, tuple)) and len(function) == 1:
            function = function[0]
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: Derivative expects a Stream input, got "
                f"{type(function).__name__}."
            )
        self._needs_window(function)
        if function.tw != 1:
            raise ValueError(
                f"{self.name}: Derivative takes one sample per step, but the "
                f"signal carries {function.tw} time samples. Differentiate a "
                "time window with DerivativeStep."
            )
        if isinstance(self._init, Stream):
            _check_history(self.name, self._init, function, self.past)

        # Every call differentiates a signal of its own, so each has its own
        # past.
        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        # Named after the signal whose past it holds.
        past = Input(
            f"{call_name}_{function.name}",
            dim=function.dim,
            seq=_declared_seq(function),
            tw=self.past,
        )
        past._hidden = True
        derivative = DerivativeStep(
            dt=self._dt, poly_order=self._poly_order, name=call_name
        )(past, function)
        past._feed_back(function, init=self._init)
        return derivative

    def build_layer(self):
        raise RuntimeError(
            f"{self.name}: a Derivative is the DerivativeStep it calls on an "
            "Input fed back with its signal, not a layer of its own."
        )
