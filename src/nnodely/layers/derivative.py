"""Causal finite-difference derivative of a signal, always with respect to
time - a block of the same model, reading directly off a Stream's own time
axis. No separate Modely, no keras-model rebuilding.
"""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer
from nnodely.utils.utils import find_sample_time
from nnodely.core.stream import Stream

# Backward finite-difference stencils, ordered oldest -> newest sample to
# match the SampleWindow time-axis convention (index -1 is the current sample).
_BACKWARD_STENCILS = {
    1: (-1.0, 1.0),
    2: (1.0, -2.0, 1.0),
}


@keras.saving.register_keras_serializable(package="nnodely")
class DerivativeImpl(keras.layers.Layer):
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


class Derivative(Layer):
    """
    Causal (backward) finite-difference derivative over a fixed-size window,
    always with respect to time - a block of the same model, not a separate
    Modely::

        Derivative(order=1|2, dt=None)(window_stream)

    Reduces a window of ``order + 1`` samples to one estimated derivative
    sample. Works on *any* Stream whose time length matches the window - a
    raw ``x.sw(order + 1)``, or any relation computed over one (e.g. a
    sub-network's output), not only a direct Input window. ``dt`` is
    discovered by walking the ancestor chain for an Input's ``sample_time``
    unless given explicitly.
    """

    def __init__(self, order: int = 1, dt: float | None = None, name=None):
        if order not in _BACKWARD_STENCILS:
            raise ValueError(
                f"Derivative supports order in {sorted(_BACKWARD_STENCILS)}, got {order}."
            )
        self.order = int(order)
        self.dt = None if dt is None else float(dt)
        super().__init__(name=name, order=self.order, dt=self.dt)

    def build_layer(self):
        pred = self.preds[0]
        if not isinstance(pred, Stream):
            raise TypeError(
                f"{self.name}: Derivative expects a Stream input, got {type(pred).__name__}."
            )
        window = pred.shape.time
        expected = self.order + 1
        if window != expected:
            raise ValueError(
                f"{self.name}: order {self.order} derivative requires a window of "
                f"{expected} samples, got {window}."
            )
        dt = self._resolve_dt()
        stencil = _BACKWARD_STENCILS[self.order]
        coefficients = [c / (dt**self.order) for c in stencil]
        return DerivativeImpl(
            coefficients=coefficients, dim_rank=len(self.dim), name=self.name
        )

    def _resolve_dt(self) -> float:
        if self.dt is not None:
            return self.dt
        sample_time = find_sample_time(self.preds[0])
        if sample_time is None:
            raise ValueError(
                f"{self.name}: no dt available. Set Input(..., sample_time=...) "
                "somewhere upstream, or pass dt= explicitly to Derivative()."
            )
        return float(sample_time)

    def get_config(self):
        return {"name": self.name, "order": self.order, "dt": self.dt}

    @classmethod
    def from_config(cls, config: dict, preds=None):
        layer = cls(order=config["order"], dt=config.get("dt"), name=config["name"])
        if preds:
            return layer(preds)
        return layer
