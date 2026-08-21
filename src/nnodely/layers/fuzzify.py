"""Fuzzify layer."""

from __future__ import annotations

import numpy as np
import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class FuzzifyImpl(keras.layers.Layer):
    def __init__(self, centers, function_name="Triangular", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.centers = sorted(centers)
        self.function_name = function_name

        if not self.centers:
            raise ValueError("Fuzzify requires at least one center.")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "centers": self.centers,
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

    def _reshape_membership_params(self, values, x):
        x_dtype = x.dtype if hasattr(x, "dtype") else "float32"
        rank = len(x.shape)
        shape = [1, len(values)] + [1] * (rank - 2)
        return keras.ops.reshape(
            keras.ops.cast(np.asarray(values), x_dtype),
            tuple(shape),
        )

    def call(self, x):
        left_bounds, right_bounds = self._support_bounds()

        x_dtype = x.dtype if hasattr(x, "dtype") else "float32"
        x = keras.ops.cast(x, x_dtype)

        centers = self._reshape_membership_params(self.centers, x)
        lefts = self._reshape_membership_params(left_bounds, x)
        rights = self._reshape_membership_params(right_bounds, x)

        name = self.function_name.lower()

        if name == "triangular":
            eps = keras.ops.cast(1e-7, x_dtype)
            left_den = keras.ops.maximum(centers - lefts, eps)
            right_den = keras.ops.maximum(rights - centers, eps)
            rise = (x - lefts) / left_den
            fall = (rights - x) / right_den
            output = keras.ops.maximum(
                keras.ops.minimum(rise, fall),
                keras.ops.cast(0.0, x_dtype),
            )

        elif name == "rectangular":
            rectangular_lefts = centers - (rights - centers) / 2.0
            rectangular_rights = centers + (rights - centers) / 2.0
            output = keras.ops.cast(
                keras.ops.logical_and(x > rectangular_lefts, x < rectangular_rights),
                x_dtype,
            )

        elif name == "gaussian":
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
    def __init__(
        self,
        centers: list[float],
        function: str = "Triangular",
        name=None,
    ):
        # if centers is not None:
        #     resolved_centers = [float(value) for value in centers]
        # else:
        #     if output_dimension is None or range is None:
        #         raise ValueError(
        #             "Fuzzify requires either 'centers' or both 'output_dimension' and 'range'."
        #         )

        #     if len(range) != 2:
        #         raise ValueError("Fuzzify range must contain exactly two values.")

        #     resolved_centers = np.linspace(
        #         float(range[0]),
        #         float(range[1]),
        #         int(output_dimension),
        #     ).tolist()

        # if output_dimension is None:
        #     output_dimension = len(resolved_centers)

        # if len(resolved_centers) != int(output_dimension):
        #     raise ValueError(
        #         "Fuzzify output_dimension must match the number of centers."
        #     )

        # self.output_dimension = int(output_dimension)
        # self.range = (
        #     tuple(float(value) for value in range) if range is not None else None
        # )
        # self.centers = tuple(resolved_centers)
        # self.function = str(function)

        self.output_dimension = len(centers)
        self.centers = centers
        self.function = function
        super().__init__(
            name=name,
            # output_dimension=self.output_dimension,
            # range=self.range,
            centers=self.centers,
            function=self.function,
        )

    def output_shape(self, *inputs):
        inp = inputs[0]

        if len(inp.dim) != 1:
            raise ValueError(
                f"Fuzzify currently expects a single input feature axis, got dim={inp.dim}"
            )

        if inp.dim[0] != 1:
            raise ValueError(
                f"Fuzzify expects scalar input dim=(1,), got dim={inp.dim}"
            )

        return (self.output_dimension,), inp.time, inp.seq

    def build_layer(self):
        return FuzzifyImpl(
            centers=self.centers,
            function_name=self.function,
            name=self.name,
        )

    def get_config(self):
        return {
            "centers": self.centers,
            "function": self.function,
        }
