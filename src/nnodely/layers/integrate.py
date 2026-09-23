"""Time integration of a rate signal along its own time axis.

One form, shaped like ``Derivative`` and exactly inverse to it: the rate
sample of an interval integrates into the value at the end of that interval,
the window's length is preserved, and ``init`` is the value the signal had
just before the window.
"""

from __future__ import annotations

import keras
import numpy as np

from nnodely.core.layer import Layer
from nnodely.core.stream import Stream

#: Quadrature rules, by the name the model writes. "euler" is the same
#: rectangular rule under the name it has when the window is one sample and
#: the layer is one step of a recurrence.
_SOLVER_RULE = {
    "euler": "rectangular",
    "rectangular": "rectangular",
    "trapezoidal": "trapezoidal",
}


def _validate_dt(dt) -> float:
    """The time step, always given explicitly - as in Derivative."""
    if dt is None:
        raise ValueError(
            "Integrate: dt is required. Pass the time step of the signal being "
            "integrated, for example Integrate(solver='euler', dt=0.01)(rate)."
        )
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError(f"Integrate: dt must be a number, got {type(dt).__name__}.")
    if dt <= 0:
        raise ValueError(f"Integrate: dt must be positive, got {dt}.")
    return float(dt)


def _validate_init(init):
    """The constant of integration, as configured - a Stream, a number or None."""
    if init is None or isinstance(init, Stream):
        return init
    if isinstance(init, bool) or not isinstance(init, (int, float)):
        raise TypeError(
            "Integrate: init must be a Stream, a number or None, got "
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

    def __init__(self, dt, rule, dim_rank, init_has_batch=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dt = float(dt)
        self.rule = rule
        self.dim_rank = int(dim_rank)
        self.init_has_batch = bool(init_has_batch)

    def build(self, input_shape):
        shapes = (
            input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # type: ignore[list-item]
        )
        window_length = int(shapes[0][1 + self.dim_rank])
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
        time_axis = 1 + self.dim_rank

        result = keras.ops.tensordot(x, self.operator, axes=[[time_axis], [0]])  # type: ignore[arg-type]
        # tensordot appends the window axis; put it back where time belongs.
        # The axis is addressed from the front because a negative index
        # survives tracing as a negative permutation, which ONNX rejects.
        last_axis = len(result.shape) - 1
        if last_axis != time_axis:
            result = keras.ops.moveaxis(result, last_axis, time_axis)

        if init is not None:
            # A Constant carries no batch axis and a scalar initial condition
            # no dim axis; adding it to the integral broadcasts both up.
            if not self.init_has_batch:
                init = keras.ops.expand_dims(init, axis=0)
            result = result + init
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
                "dim_rank": self.dim_rank,
                "init_has_batch": self.init_has_batch,
            }
        )
        return config


class Integrate(Layer):
    """
    Time integration of a rate signal along its own time axis::

        Integrate(solver="euler"|"rectangular"|"trapezoidal", dt=0.01, init=None)(rate)

    The window's length is preserved: sample ``i`` of the rate is the rate of
    the interval *ending* at ``i`` - the interval ``Derivative``'s backward
    difference produced it from - and integrates into the value at ``i``::

        y[i] = y[i-1] + dt * rate[i]                   # "euler"/"rectangular"
        y[i] = y[i-1] + dt/2 * (rate[i-1] + rate[i])   # "trapezoidal"

    with ``y[-1] = init``. Two things follow from that convention:

    * **A one-sample window is one integration step.** ``Integrate(dt=dt,
      init=velocity.last())(acceleration.last())`` is the state update
      ``velocity + dt * acceleration``, so a recurrence needs no separate form
      of this layer and no ``+`` written around it.

    * **It inverts Derivative exactly**, with the same initial condition and
      no bookkeeping: ``Integrate(dt=dt, init=x0)(Derivative(respect_to=dt,
      init=x0)(x))`` is ``x``, because the rectangular rule sums back exactly
      the increments the backward difference took apart. (The trapezoidal rule
      trades that exactness for second-order accuracy on a smooth rate.)

    ``init`` is the constant of integration - the value the signal had at the
    sample *before* the window, the same instant ``Derivative`` reads its own
    ``init`` at. A Stream (typically the state the window continues), a number
    (kept as a Constant), or ``None`` for zero. It is what turns a relative
    increment into the absolute quantity mechanical models are written in::

        velocity = Integrate(dt=dt, init=v0)(acceleration.sw(n))
        position = Integrate(dt=dt, init=x0)(velocity)

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
                "Integrate is configured first and called on the rate, like "
                "every other layer: Integrate(solver=..., dt=...)(rate)."
            )
        if solver not in _SOLVER_RULE:
            raise ValueError(
                f"Integrate: solver must be one of {sorted(_SOLVER_RULE)}, got {solver!r}."
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
                f"{self.name}: Integrate expects a Stream input, got "
                f"{type(rate).__name__}."
            )
        if len(inputs) > 1:
            self._validate_init_shape(rate, inputs[1])
        # One integrated sample per rate sample: the result stays aligned with
        # the signal it came from.
        return rate.shape.dim, rate.shape.time, rate.shape.seq

    def _validate_init_shape(self, rate, init):
        init_dim, init_seq = tuple(init.shape.dim), tuple(init.shape.seq)
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
        if init_seq not in ((), tuple(rate.shape.seq)):
            raise ValueError(
                f"{self.name}: init has seq {init_seq}, which is neither empty nor "
                f"the integrated rate's {tuple(rate.shape.seq)}."
            )

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def build_layer(self):
        # A Constant or Parameter is a leaf of the graph and carries no batch
        # axis, so the initial condition it holds broadcasts instead.
        init_has_batch = True
        if len(self.preds) > 1:
            init_node = self.preds[1]
            init_has_batch = not (
                isinstance(init_node, Layer) and len(init_node.preds) == 0
            )
        return IntegrateImpl(
            dt=self.dt,
            rule=_SOLVER_RULE[self.solver],
            dim_rank=len(self.dim),
            init_has_batch=init_has_batch,
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
