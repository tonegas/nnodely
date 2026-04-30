"""Fuzzify layer."""

from __future__ import annotations

import numpy as np
import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class FuzzifyImpl(keras.layers.Layer):
    def __init__(self, centers, function_name="Triangular", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.centers = tuple(float(value) for value in centers)
        self.function_name = str(function_name)

        if not self.centers:
            raise ValueError("Fuzzify requires at least one center.")

        if any(right <= left for left, right in zip(self.centers, self.centers[1:])):
            raise ValueError("Fuzzify centers must be strictly increasing.")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "centers": list(self.centers),
                "function_name": self.function_name,
            }
        )
        return config

    def _support_bounds(self):
        n = len(self.centers)
        if n == 1:
            center = self.centers[0]
            return [center - 1.0], [center + 1.0]

        left_bounds = []
        right_bounds = []
        for idx, center in enumerate(self.centers):
            if idx == 0:
                step = self.centers[1] - center
                left_bounds.append(center - step)
                right_bounds.append(self.centers[1])
            elif idx == n - 1:
                step = center - self.centers[idx - 1]
                left_bounds.append(self.centers[idx - 1])
                right_bounds.append(center + step)
            else:
                left_bounds.append(self.centers[idx - 1])
                right_bounds.append(self.centers[idx + 1])

        return left_bounds, right_bounds

    def call(self, x):
        left_bounds, right_bounds = self._support_bounds()

        x_dtype = x.dtype if hasattr(x, "dtype") else "float32"
        x = keras.ops.cast(x, x_dtype)

        centers = keras.ops.reshape(
            keras.ops.cast(np.asarray(self.centers), x_dtype), (1, 1, len(self.centers))
        )
        lefts = keras.ops.reshape(
            keras.ops.cast(np.asarray(left_bounds), x_dtype), (1, 1, len(self.centers))
        )
        rights = keras.ops.reshape(
            keras.ops.cast(np.asarray(right_bounds), x_dtype), (1, 1, len(self.centers))
        )

        if self.function_name.lower() == "triangular":
            eps = keras.ops.cast(1e-7, x_dtype)
            left_den = keras.ops.maximum(centers - lefts, eps)
            right_den = keras.ops.maximum(rights - centers, eps)
            rise = (x - lefts) / left_den
            fall = (rights - x) / right_den
            output = keras.ops.maximum(keras.ops.minimum(rise, fall), 0.0)
        elif self.function_name.lower() == "rectangular":
            output = keras.ops.cast(
                keras.ops.logical_and(x > lefts, x < rights), x_dtype
            )
        elif self.function_name.lower() == "gaussian":
            sigma = (rights - lefts) / 6.0
            sigma = keras.ops.maximum(sigma, keras.ops.cast(1e-7, x_dtype))
            exponent = -0.5 * ((x - centers) / sigma) ** 2
            output = keras.ops.exp(exponent)
        else:
            raise ValueError(
                f"Unsupported fuzzify function: {self.function_name!r}. "
                "Supported functions: 'Triangular', 'Rectangular', 'Gaussian'."
            )

        return output


class Fuzzify(Layer):
    """Fuzzify a scalar stream into membership degrees over a set of centers."""

    def __init__(
        self,
        output_dimension: int | None = None,
        range=None,
        centers=None,
        function=None,
        functions="Triangular",
        name=None,
    ):
        if function is not None:
            if functions != "Triangular" and str(functions) != str(function):
                raise ValueError(
                    "Use either 'function' or 'functions' with matching values."
                )
            functions = function

        if centers is not None:
            resolved_centers = [float(value) for value in centers]
        else:
            if output_dimension is None or range is None:
                raise ValueError(
                    "Fuzzify requires either 'centers' or both 'output_dimension' and 'range'."
                )
            if len(range) != 2:
                raise ValueError("Fuzzify range must contain exactly two values.")
            resolved_centers = np.linspace(
                float(range[0]), float(range[1]), int(output_dimension)
            ).tolist()

        if output_dimension is None:
            output_dimension = len(resolved_centers)

        if len(resolved_centers) != int(output_dimension):
            raise ValueError(
                "Fuzzify output_dimension must match the number of centers."
            )

        self.output_dimension = int(output_dimension)
        self.range = (
            tuple(float(value) for value in range) if range is not None else None
        )
        self.centers = tuple(resolved_centers)
        self.functions = str(functions)

        super().__init__(
            name=name,
            output_dimension=self.output_dimension,
            range=self.range,
            centers=self.centers,
            functions=self.functions,
        )

    def output_shape(self, *inputs):
        for inp in inputs:
            if len(inp.dim) != 1:
                raise ValueError(
                    f"Fuzzify currently expects a single input feature axis, got dim={inp.dim}"
                )
        return inputs[0].seq, inputs[0].time, (self.output_dimension,)

    def build_layer(self):
        return FuzzifyImpl(
            centers=self.centers,
            function_name=self.functions,
            name=self.name,
        )
