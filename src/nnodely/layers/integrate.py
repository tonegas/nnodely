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

# Quadrature weights (unscaled by dt), ordered oldest -> newest sample.
_SOLVER_WEIGHTS = {
    "euler": (1.0,),
    "rectangular": (1.0,),
    "trapezoidal": (0.5, 0.5),
    "heun": (0.5, 0.5),
}


@keras.saving.register_keras_serializable(package="nnodely")
class IntegrateCumulativeImpl(keras.layers.Layer):
    """Elementwise running integral, keeping the full window length.

    The first sample of the window is the integration reference (value 0).
    """

    def __init__(self, dt, dim_rank, method, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dt = float(dt)
        self.dim_rank = int(dim_rank)
        self.method = method

    def _time_slice(self, x, start, stop):
        rank = len(x.shape)
        slices = [slice(None)] * rank
        slices[1 + self.dim_rank] = slice(start, stop)
        return x[tuple(slices)]

    def call(self, x):
        time_axis = 1 + self.dim_rank
        if self.method == "trapezoidal":
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
                "method": self.method,
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
    Time integration of a signal, in one of two forms:

    1. A single integration increment, read directly off ``f``'s own time
       axis - a block of the same model, not a separate Modely::

           Integrate(f, solver="euler"|"trapezoidal"|"heun", dt=None)

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

           Integrate(method="trapezoidal"|"rectangular", dt=None)(window_stream)

       Works on any Stream with the right window length, not only a raw
       Input window.

    In both forms, ``dt`` is discovered by walking the ancestor chain for
    an Input's ``sample_time`` unless given explicitly.
    """

    def __init__(
        self,
        f: Stream | None = None,
        solver: str = "euler",
        method: str = "trapezoidal",
        dt: float | None = None,
        name=None,
    ):
        if f is not None:
            self._build_step(f=f, solver=solver, dt=dt, name=name)
            return

        if method not in ("trapezoidal", "rectangular"):
            raise ValueError(
                f"Integrate method must be 'trapezoidal' or 'rectangular', got {method!r}."
            )
        self.step_mode = False
        self.method = method
        self.dt = None if dt is None else float(dt)
        super().__init__(name=name, method=self.method, dt=self.dt)

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
        if self.step_mode:
            dt = self._resolve_dt()
            coefficients = [w * dt for w in _SOLVER_WEIGHTS[self.solver]]
            return IntegrateStepImpl(
                coefficients=coefficients, dim_rank=len(self.dim), name=self.name
            )

        dt = self._resolve_dt()
        return IntegrateCumulativeImpl(
            dt=dt, dim_rank=len(self.dim), method=self.method, name=self.name
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
        if self.step_mode:
            return {
                "name": self.name,
                "mode": "step",
                "solver": self.solver,
                "dt": self.dt,
            }
        return {
            "name": self.name,
            "mode": "cumulative",
            "method": self.method,
            "dt": self.dt,
        }

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
        layer = cls(method=config["method"], dt=config.get("dt"), name=config["name"])
        if preds:
            return layer(preds)
        return layer
