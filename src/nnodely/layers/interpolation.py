"""Elementwise interpolation layers."""

from __future__ import annotations

import keras
import numpy as np

from nnodely.core.layer import Layer


_AVAILABLE_MODES = ("linear", "polynomial")


def _prepare_points(x_points, y_points, mode):
    x_values = np.asarray(x_points, dtype=np.float64)
    y_values = np.asarray(y_points, dtype=np.float64)
    mode = str(mode).lower()

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("Interpolation points must be one-dimensional.")
    if len(x_values) != len(y_values):
        raise ValueError("x_points and y_points must have the same length.")
    if len(x_values) < 2:
        raise ValueError("Interpolation requires at least two points.")
    if mode not in _AVAILABLE_MODES:
        raise ValueError(
            f"Interpolation mode must be one of {_AVAILABLE_MODES}, got {mode!r}."
        )
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("Interpolation points must contain only finite values.")

    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    if np.any(np.diff(x_values) == 0):
        raise ValueError("x_points must contain unique values.")

    return x_values, y_values, mode


@keras.saving.register_keras_serializable(package="nnodely")
class InterpolationImpl(keras.layers.Layer):
    """Serializable elementwise linear or polynomial interpolation."""

    def __init__(self, x_points, y_points, mode="linear", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        x_values, y_values, mode = _prepare_points(x_points, y_points, mode)
        self.x_points = x_values.tolist()
        self.y_points = y_values.tolist()
        self.mode = mode
        self.coefficients = (
            np.linalg.solve(
                np.vander(x_values, N=len(x_values), increasing=True),
                y_values,
            ).tolist()
            if mode == "polynomial"
            else None
        )

    def call(self, inputs):
        dtype = inputs.dtype
        lower = keras.ops.cast(self.x_points[0], dtype)
        upper = keras.ops.cast(self.x_points[-1], dtype)
        values = keras.ops.clip(inputs, lower, upper)

        if self.mode == "polynomial":
            coefficients = self.coefficients
            if coefficients is None:
                raise RuntimeError("Polynomial coefficients were not initialized.")
            result = keras.ops.zeros_like(values) + keras.ops.cast(
                coefficients[-1], dtype
            )
            for coefficient in reversed(coefficients[:-1]):
                result = result * values + keras.ops.cast(coefficient, dtype)
            return result

        result = keras.ops.zeros_like(values)
        for index in range(len(self.x_points) - 1):
            x_left = keras.ops.cast(self.x_points[index], dtype)
            x_right = keras.ops.cast(self.x_points[index + 1], dtype)
            y_left = keras.ops.cast(self.y_points[index], dtype)
            y_right = keras.ops.cast(self.y_points[index + 1], dtype)
            interpolated = y_left + (y_right - y_left) * (values - x_left) / (
                x_right - x_left
            )
            right_condition = (
                values <= x_right
                if index == len(self.x_points) - 2
                else values < x_right
            )
            interval = keras.ops.logical_and(values >= x_left, right_condition)
            result = keras.ops.where(interval, interpolated, result)
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "x_points": self.x_points,
                "y_points": self.y_points,
                "mode": self.mode,
            }
        )
        return config


class Interpolation(Layer):
    """Interpolate input values using a one-dimensional lookup table.

    Queries outside the point range are clamped to the closest endpoint.
    ``mode="linear"`` performs piecewise-linear interpolation, while
    ``mode="polynomial"`` uses the unique global polynomial passing through
    all supplied points.
    """

    def __init__(self, x_points, y_points, mode="linear", name=None):
        x_values, y_values, mode = _prepare_points(x_points, y_points, mode)
        self.x_points = x_values.tolist()
        self.y_points = y_values.tolist()
        self.mode = mode
        super().__init__(
            name=name,
            x_points=self.x_points,
            y_points=self.y_points,
            mode=self.mode,
        )

    def build_layer(self):
        return InterpolationImpl(
            x_points=self.x_points,
            y_points=self.y_points,
            mode=self.mode,
            name=self.name,
        )

    def get_config(self):
        return {
            "x_points": self.x_points,
            "y_points": self.y_points,
            "mode": self.mode,
        }
