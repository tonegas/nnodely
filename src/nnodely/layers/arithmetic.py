"""Arithmetic layers for nnodely."""

import keras

from nnodely.core.layer import Layer

@keras.saving.register_keras_serializable(package="nnodely")
class ArithmeticImpl(keras.layers.Layer):
    def __init__(self, operation: str, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.operation = operation

    def call(self, inputs):
        return getattr(keras.ops, self.operation)(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"operation": self.operation})
        return config


class Arithmetic(Layer):
    operation = ""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return ArithmeticImpl(operation=self.operation, name=self.name)

    def get_config(self):
        return {"name": self.name}


class Exp(Arithmetic):
    """Wrapper for exponential transform."""

    operation = "exp"
    
class Log(Arithmetic):
    """Wrapper for natural logarithm transform."""

    operation = "log"

class Log10(Arithmetic):
    """Wrapper for base 10 logarithm transform."""

    operation = "log10"

class Sqrt(Arithmetic):
    """Wrapper for square root transform."""

    operation = "sqrt"

class Abs(Arithmetic):
    """Wrapper for absolute value transform."""

    operation = "abs"

class Floor(Arithmetic):
    """Wrapper for floor transform."""

    operation = "floor"

class Ceil(Arithmetic):
    """Wrapper for ceiling transform."""

    operation = "ceil"

## Note: The following class is commented out, as it is not currently in Keras ops.
# .
# class Rad2Deg(Arithmetic):
#     """Wrapper for radians to degrees transform."""

#     operation = "rad2deg"

class Deg2Rad(Arithmetic):
    """Wrapper for degrees to radians transform."""

    operation = "deg2rad"

class Sign(Arithmetic):
    """Wrapper for sign transform."""

    operation = "sign"

class Negative(Arithmetic):
    """Wrapper for negation transform."""

    operation = "negative"

@keras.saving.register_keras_serializable(package="nnodely")
class ClampImpl(keras.layers.Layer):
    def __init__(self, min=None, max=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.min = min
        self.max = max

    def call(self, inputs):
        return keras.ops.clip(
            inputs,
            -float("inf") if self.min is None else self.min,
            float("inf") if self.max is None else self.max,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"min": self.min, "max": self.max})
        return config


class Clamp(Layer):
    """Wrapper for element-wise clipping between min and max.

    A bound left to None is unbounded on that side.
    """

    def __init__(self, min: float | None = None, max: float | None = None, name=None):
        if min is not None and max is not None and min > max:
            raise ValueError(f"{name}: min {min} must not be greater than max {max}.")
        self.min = min
        self.max = max
        super().__init__(name=name, min=self.min, max=self.max)

    def build_layer(self):
        return ClampImpl(min=self.min, max=self.max, name=self.name)

    def get_config(self):
        return {"name": self.name, "min": self.min, "max": self.max}


@keras.saving.register_keras_serializable(package="nnodely")
class SumImpl(keras.layers.Layer):
    """
    Serializable implementation of Sum.

    Runtime tensor shape:
        [batch, dim1, dim2, ..., time, seq1, seq2, ...]
    """

    def __init__(self, axes, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axes = tuple(int(axis) for axis in axes)

    def call(self, x):
        return keras.ops.sum(x, axis=self.axes, keepdims=True)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        for axis in self.axes:
            output_shape[axis] = 1
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axes": self.axes})
        return config


class Sum(Layer):
    """
    Sum over the dim axes, keeping them with length 1.

    Runtime tensor shape:
        [batch, dim1, dim2, ..., time, seq1, seq2, ...]

    With axis=None every dim axis is summed together, otherwise only the
    chosen one.

    Examples
    --------
    dim=(2, 3), axis=None -> dim=(1, 1)
    dim=(2, 3), axis=0 -> dim=(1, 3)
    """

    def __init__(self, axis: int | None = None, name=None):
        self.axis = axis if axis is None else int(axis)
        super().__init__(name=name, axis=self.axis)

    def build_layer(self):
        dim_rank = len(self.dim)
        if self.axis is None:
            axes = range(dim_rank)
        else:
            axis = self.axis + dim_rank if self.axis < 0 else self.axis
            if axis < 0 or axis >= dim_rank:
                raise ValueError(
                    f"{self.name}: axis {self.axis} out of bounds for dim rank {dim_rank}."
                )
            axes = (axis,)

        # Dim axes start immediately after batch.
        return SumImpl(axes=tuple(1 + axis for axis in axes), name=self.name)

    def get_config(self):
        return {"name": self.name, "axis": self.axis}
