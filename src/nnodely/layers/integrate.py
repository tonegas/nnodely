"""Time integration of a signal: elementwise cumulative scan over a window,
or a single integration step read directly off a rate Stream's own time
axis - no separate Modely, no state input. The caller adds the step to a
running state with '+' and iterates it with rollback()/Scan/Roll, exactly
like the finite-difference Derivative.
"""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer
from nnodely.core.stream import Stream
from nnodely.utils.utils import find_sample_time

# Quadrature weights (unscaled by dt), ordered oldest -> newest sample. Used
# by step mode (reduce the window to one increment) and, via _CUMULATIVE_RULE
# below, by cumulative mode (keep the window, scan it) - "euler" and
# "rectangular" are the same 1-sample rule, "trapezoidal" and "heun" the
# same 2-sample rule; only the two underlying rules differ, not the names.
_SOLVER_WEIGHTS = {
    "euler": (1.0,),
    "rectangular": (1.0,),
    "trapezoidal": (0.5, 0.5),
    "heun": (0.5, 0.5),
}

_CUMULATIVE_RULE = {
    "euler": "rectangular",
    "rectangular": "rectangular",
    "trapezoidal": "trapezoidal",
    "heun": "trapezoidal",
}


@keras.saving.register_keras_serializable(package="nnodely")
class IntegrateCumulativeImpl(keras.layers.Layer):
    """Elementwise running integral, keeping the full window length.

    The first sample of the window is the integration reference (value 0).
    """

    def __init__(self, dt, dim_rank, rule, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dt = float(dt)
        self.dim_rank = int(dim_rank)
        self.rule = rule

    def _time_slice(self, x, start, stop):
        rank = len(x.shape)
        slices = [slice(None)] * rank
        slices[1 + self.dim_rank] = slice(start, stop)
        return x[tuple(slices)]

    def call(self, x):
        time_axis = 1 + self.dim_rank
        if self.rule == "trapezoidal":
            left = self._time_slice(x, 0, -1)
            right = self._time_slice(x, 1, None)
            segment_area = (left + right) * (self.dt / 2.0)
        else:  # rectangular
            segment_area = self._time_slice(x, 1, None) * self.dt

        cumulative = keras.ops.cumsum(segment_area, axis=time_axis)
        reference = keras.ops.zeros_like(self._time_slice(x, 0, 1))
        return keras.ops.concatenate([reference, cumulative], axis=time_axis)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dt": self.dt,
                "dim_rank": self.dim_rank,
                "rule": self.rule,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="nnodely")
class IntegrateStepImpl(keras.layers.Layer):
    """Reduce f's own time window to one dt-weighted integration increment."""

    def __init__(self, coefficients, dim_rank, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.dim_rank = int(dim_rank)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="coefficients",
            shape=(len(self.coefficients),),
            initializer=keras.initializers.Constant(self.coefficients),  # type: ignore
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, x):
        time_axis = 1 + self.dim_rank
        result = keras.ops.tensordot(x, self.kernel, axes=[[time_axis], [0]])  # type: ignore
        return keras.ops.expand_dims(result, axis=time_axis)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1 + self.dim_rank] = 1
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "coefficients": self.coefficients,
                "dim_rank": self.dim_rank,
            }
        )
        return config


class Integrate(Layer):
    """
    Time integration of a signal, in one of two forms - both configured by
    the same ``solver`` vocabulary (``"euler"``/``"rectangular"``: 1-sample
    rule, ``"trapezoidal"``/``"heun"``: 2-sample rule):

    1. A single integration increment, read directly off ``f``'s own time
       axis - a block of the same model, not a separate Modely::

           Integrate(f, solver="euler"|"trapezoidal", dt=None)

       ``f`` is any Stream (the rate signal); its time window must match
       the chosen solver: ``"euler"``/``"rectangular"`` reads 1 sample
       (the current rate), ``"trapezoidal"``/``"heun"`` reads 2 (the
       previous and current rate). Add the result to a running state with
       ``+`` and iterate with ``rollback()``/``Scan``/``Roll`` yourself,
       e.g.::

           next_velocity = velocity.last() + Integrate(accel.last(), solver="euler")

       A multi-sample window for ``"trapezoidal"``/``"heun"`` can come from
       anywhere - a plain ``x.sw(2)``, or a window grown by unrolling a
       recurrent sub-model with ``Roll``; Integrate only reads whatever
       window ``f`` already has and doesn't care how it was built. There is
       no ``"rk4"``: true (adaptive) Runge-Kutta needs the rate evaluated
       at *predicted* intermediate states, which isn't expressible from a
       passively observed time window.

    2. Elementwise cumulative integral over a window (shape preserved, the
       first sample is the reference value 0), for offline signal
       preprocessing::

           Integrate(solver="trapezoidal"|"rectangular", dt=None)(window_stream)

       Works on any Stream with the right window length, not only a raw
       Input window.

    In both forms, ``dt`` is discovered by walking the ancestor chain for
    an Input's ``sample_time`` unless given explicitly.
    """

    def __init__(
        self,
        f: Stream | None = None,
        solver: str = "euler",
        dt: float | None = None,
        name=None,
    ):
        if f is not None:
            self._build_step(f=f, solver=solver, dt=dt, name=name)
            return

        if solver not in _CUMULATIVE_RULE:
            raise ValueError(
                f"Integrate solver must be one of {sorted(_CUMULATIVE_RULE)}, got {solver!r}."
            )
        self.step_mode = False
        self.solver = solver
        self.dt = None if dt is None else float(dt)
        super().__init__(name=name, solver=self.solver, dt=self.dt)

    def _build_step(self, f, solver, dt, name):
        if not isinstance(f, Stream):
            raise TypeError("Integrate: f must be a Stream.")
        if solver not in _SOLVER_WEIGHTS:
            raise ValueError(
                f"Integrate: solver must be one of {sorted(_SOLVER_WEIGHTS)}, got {solver!r}."
            )
        expected = len(_SOLVER_WEIGHTS[solver])
        if f.shape.time != expected:
            raise ValueError(
                f"Integrate: solver {solver!r} needs a window of {expected} "
                f"sample(s), got {f.shape.time}."
            )
        if dt is None and find_sample_time(f) is None:
            raise ValueError(
                "Integrate: no dt available. Set Input(..., sample_time=...) "
                "somewhere upstream, or pass dt= explicitly to Integrate()."
            )

        self.step_mode = True
        self.solver = solver
        self.dt = None if dt is None else float(dt)

        super().__init__(name=name, preds=[f], dim=f.dim, time=1, seq=f.seq)

    def build_layer(self):
        dt = self._resolve_dt()
        if self.step_mode:
            coefficients = [w * dt for w in _SOLVER_WEIGHTS[self.solver]]
            return IntegrateStepImpl(
                coefficients=coefficients, dim_rank=len(self.dim), name=self.name
            )

        return IntegrateCumulativeImpl(
            dt=dt,
            dim_rank=len(self.dim),
            rule=_CUMULATIVE_RULE[self.solver],
            name=self.name,
        )

    def _resolve_dt(self) -> float:
        if self.dt is not None:
            return self.dt
        sample_time = find_sample_time(self.preds[0])
        if sample_time is None:
            raise ValueError(
                f"{self.name}: no dt available. Set Input(..., sample_time=...) "
                "somewhere upstream, or pass dt= explicitly to Integrate()."
            )
        return float(sample_time)

    def get_config(self):
        mode = "step" if self.step_mode else "cumulative"
        return {"name": self.name, "mode": mode, "solver": self.solver, "dt": self.dt}

    @classmethod
    def from_config(cls, config: dict, preds=None):
        if config.get("mode") == "step":
            if not preds:
                raise ValueError("Integrate step mode requires preds (f) to rebuild.")
            instance = cls.__new__(cls)
            instance._build_step(
                preds[0], config["solver"], config.get("dt"), config["name"]
            )
            return instance
        layer = cls(solver=config["solver"], dt=config.get("dt"), name=config["name"])
        if preds:
            return layer(preds)
        return layer
