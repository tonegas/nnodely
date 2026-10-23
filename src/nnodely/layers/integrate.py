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


def _validate_init(init):
    """The constant of integration, as configured - a Stream, a number or None."""
    if init is None or isinstance(init, Stream):
        return init
    if isinstance(init, bool) or not isinstance(init, (int, float)):
        raise TypeError(
            "IntegrateStep: init must be a Stream, a number or None, got "
            f"{type(init).__name__}."
        )
    return init


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

        IntegrateStep(solver="euler"|"rectangular"|"trapezoidal", dt=0.01, init=None)(rate)

    The window's length is preserved: sample ``i`` of the rate is the rate of
    the interval *ending* at ``i`` - the interval ``DerivativeStep``'s backward
    difference produced it from - and integrates into the value at ``i``::

        y[i] = y[i-1] + dt * rate[i]                   # "euler"/"rectangular"
        y[i] = y[i-1] + dt/2 * (rate[i-1] + rate[i])   # "trapezoidal"

    with ``y[-1] = init``. Two things follow from that convention:

    * **A one-sample window is one integration step.** ``IntegrateStep(dt=dt,
      init=velocity)(acceleration)`` is the state update
      ``velocity + dt * acceleration``, so a recurrence needs no separate form
      of this layer and no ``+`` written around it.

    * **It inverts DerivativeStep exactly**, with the same initial condition and
      no bookkeeping: ``IntegrateStep(dt=dt, init=x0)(DerivativeStep(dt=dt,
      init=x0)(x))`` is ``x``, because the rectangular rule sums back exactly
      the increments the backward difference took apart. (The trapezoidal rule
      trades that exactness for second-order accuracy on a smooth rate.)

    ``init`` is the constant of integration - the value the signal had at the
    sample *before* the window, the same instant ``DerivativeStep`` reads its own
    ``init`` at. A Stream (typically the state the window continues), a number
    (kept as a Constant), or ``None`` for zero. It is what turns a relative
    increment into the absolute quantity mechanical models are written in::

        velocity = IntegrateStep(dt=dt, init=v0)(acceleration.sw(n))
        position = IntegrateStep(dt=dt, init=x0)(velocity)

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
        init: Stream | float | None = None,
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
        self.init = _validate_init(init)
        super().__init__(name=name, solver=self.solver, dt=self.dt, init=self.init)

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------
    def _init_stream(self):
        """The initial condition as a node of the graph, if there is one."""
        if self.init is None or isinstance(self.init, Stream):
            return self.init
        return Stream._coerce_operand(float(self.init))

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # The initial condition is a predecessor like any other, but it is
        # configured on the layer rather than passed at the call, so it is
        # appended here and reconnected the same way when reloading.
        if inputs and all(isinstance(value, Stream) for value in inputs):
            init = self._init_stream()
            inputs = [inputs[0]] if init is None else [inputs[0], init]
        return super().__call__(inputs)

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
        if len(inputs) > 1:
            self._validate_init_shape(rate, inputs[1])
        # One integrated sample per rate sample: the result stays aligned with
        # the signal it came from.
        return rate.shape

    def _validate_init_shape(self, rate, init):
        init_dim = tuple(init.shape.dim)
        if init.shape.time != 1:
            raise ValueError(
                f"{self.name}: init is the value just before the window, so it "
                f"must carry a single sample, got {init.shape.time}."
            )
        if init_dim not in ((1,), tuple(rate.shape.dim)):
            raise ValueError(
                f"{self.name}: init has dim {init_dim}, which is neither a scalar "
                f"nor the integrated rate's {tuple(rate.shape.dim)}."
            )
        seq, rate_seq = tuple(init.shape.seq), tuple(rate.shape.seq)
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
        layer = cls(
            solver=config["solver"],
            dt=config.get("dt"),
            init=preds[1] if preds and len(preds) > 1 else None,
            name=config["name"],
        )
        if preds:
            return layer(preds[0])
        return layer


def _declared_seq(seq) -> tuple[int, ...] | None:
    """Sequence axes as an Input declares them: -1 for a dynamic one."""
    axes = tuple(-1 if length is None else int(length) for length in seq)
    return axes or None


class Integrate:
    """Integrate a rate along the horizon of its sequence.

    ::

        Integrate(solver="euler"|"rectangular"|"trapezoidal", dt=0.01, init=x0)(rate)

    The high-level counterpart of :class:`IntegrateStep`: a block built from an
    ``IntegrateStep`` rolled out by a :class:`Loop`, one rate sample per step,
    with the integrated value fed back as the state of the next step. The rate
    carries the horizon as its outermost sequence axis, of fixed length
    (``seq=N``) or dynamic (``seq=-1``, followed at call time), and one sample,
    ``T=1``, per step. The result is shaped like the rate: its element ``k``
    is the value at the end of step ``k``::

        y[k] = y[k-1] + dt * rate[k]                      # "euler", "rectangular"
        y[k] = y[k-1] + dt/2 * (rate[k-1] + rate[k])      # "trapezoidal"

    These are the rules of ``IntegrateStep``, with the same convention for the
    first step, which has no earlier rate sample and keeps the rectangle: laid
    out along a time window instead, the same samples integrate to the same
    values. :class:`Derivative` with the same ``init`` is its inverse along the
    horizon, as ``DerivativeStep`` is ``IntegrateStep``'s along a window.

    ``init`` is the value just before the first step: a Stream of one sample
    (a scalar, or the rate's dim; its sequence is the rate's without the
    horizon), a number, or ``None`` for zero.

    Calling the block returns the ``Loop`` itself, bound to the rate and
    ``init`` it is called with: a stream like any other, to use in further
    relations, ``Output`` or ``minimize``. ``body`` is the one-step
    ``Modely`` the latest call rolls out.
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
        self.body = None
        self._calls = 0

    def __call__(self, rate):
        from nnodely.core.layer import Identity
        from nnodely.core.modely import Modely
        from nnodely.layers.input import Input, read
        from nnodely.layers.loop import Loop
        from nnodely.layers.output import Output

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
        rate = read(rate)
        if not rate.seq:
            raise ValueError(
                f"{self.name}: Integrate runs along the sequence of the rate, its "
                "rollout, and this rate has none. Declare it on an Input with a "
                "sequence, e.g. Input('x', seq=-1), or integrate a time window "
                "with IntegrateStep."
            )
        if rate.time != 1:
            raise ValueError(
                f"{self.name}: Integrate takes one rate sample per step, but the "
                f"rate carries {rate.time} time samples. Integrate a time window "
                "with IntegrateStep."
            )

        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        dim = tuple(rate.dim)
        # A step sees one sample of the rate, within the sequence axes the
        # horizon leaves.
        step_seq = _declared_seq(rate.seq[1:])

        # One step: the state integrates the rate of that step.
        rate_in = Input(f"{call_name}_rate", dim=dim, seq=step_seq)
        state_in = Input(f"{call_name}_state", dim=dim, seq=step_seq)
        rate_step = rate_in
        trapezoidal = _SOLVER_RULE[self.solver] == "trapezoidal"
        step_rate = rate_step
        previous_in = None
        if trapezoidal:
            # The trapezoid of a step averages its rate with the previous one,
            # which the loop feeds back like the state.
            previous_in = Input(f"{call_name}_previous_rate", dim=dim, seq=step_seq)
            step_rate = (previous_in + rate_step) * 0.5
        step = IntegrateStep(
            solver="euler",
            dt=self.dt,
            init=state_in,
            name=f"{call_name}_step",
        )(step_rate)
        next_out = Output(f"{call_name}_next", step)
        body_inputs, body_outputs = [rate_in, state_in], [next_out]
        callback = {state_in: next_out}
        if previous_in is not None:
            rate_out = Output(f"{call_name}_rate_out", Identity()([rate_step]))
            body_inputs.append(previous_in)
            body_outputs.append(rate_out)
            callback[previous_in] = rate_out
        body = Modely(f"{call_name}_body", inputs=body_inputs, outputs=body_outputs)

        # The step rolled out along the rate's horizon, bound to the caller's
        # streams: the loop is returned as the integrated stream itself.
        initial: dict = {}
        if isinstance(self.init, Stream):
            # A scalar init is spread over the rate's dim by the Loop, which
            # broadcasts an initial value to the state it seeds.
            init = read(self.init)
            self._validate_init_shape(rate, init)
            initial[state_in] = init
        elif self.init is not None:
            initial[state_in] = float(self.init)
        if previous_in is not None:
            # The rate has one more sequence axis than the step, the horizon: the
            # first step reads its first sample, so that step is a rectangle.
            initial[previous_in] = rate

        self.body = body
        return Loop(f=body, callback=callback, init=initial, name=call_name)(
            {rate_in: rate}
        )

    def _validate_init_shape(self, rate, init):
        dim, step_seq = tuple(rate.dim), tuple(rate.seq[1:])
        if init.time != 1:
            raise ValueError(
                f"{self.name}: init is the value just before the first step, so "
                f"it must carry a single sample, got {init.time}."
            )
        if tuple(init.dim) not in (dim, (1,)) or (
            tuple(init.dim) == (1,) and len(dim) > 1
        ):
            raise ValueError(
                f"{self.name}: init has dim {tuple(init.dim)}, which is neither "
                f"the rate's {dim} nor, for a one-axis dim, a scalar."
            )
        if tuple(init.seq) != step_seq:
            raise ValueError(
                f"{self.name}: init has the sequence {tuple(init.seq)}, but one "
                "value per sequence of steps needs the rate's sequence without "
                f"the horizon, {step_seq}."
            )
