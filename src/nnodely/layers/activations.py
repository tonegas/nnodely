"""Activation layers wrappers for nnodely."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


class ReLU(Layer):
    """Wrapper for keras.layers.ReLU."""

    def __init__(
        self,
        max_value: float | None = None,
        negative_slope: float = 0.0,
        threshold: float = 0.0,
        name=None,
    ):
        self.max_value = max_value
        self.negative_slope = float(negative_slope)
        self.threshold = float(threshold)
        super().__init__(
            name=name,
            max_value=self.max_value,
            negative_slope=self.negative_slope,
            threshold=self.threshold,
        )

    def build_layer(self):
        return keras.layers.ReLU(
            max_value=self.max_value,
            negative_slope=self.negative_slope,
            threshold=self.threshold,
            name=self.name,
        )


class LeakyReLU(Layer):
    """Wrapper for keras.layers.LeakyReLU."""

    def __init__(self, negative_slope: float = 0.3, name=None):
        self.negative_slope = float(negative_slope)
        super().__init__(name=name, negative_slope=self.negative_slope)

    def build_layer(self):
        return keras.layers.LeakyReLU(
            negative_slope=self.negative_slope,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class ELUImpl(keras.layers.Layer):
    def __init__(self, alpha: float = 1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)

    def call(self, inputs):
        zero = keras.ops.cast(0.0, inputs.dtype)
        alpha = keras.ops.cast(self.alpha, inputs.dtype)
        return keras.ops.where(
            inputs > zero,
            inputs,
            alpha * (keras.ops.exp(inputs) - 1.0),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class ELU(Layer):
    """Exponential linear unit: ``x`` where it is positive, and
    ``alpha * (exp(x) - 1)`` elsewhere."""

    def __init__(self, alpha: float = 1.0, name=None):
        self.alpha = float(alpha)
        super().__init__(name=name, alpha=self.alpha)

    def build_layer(self):
        return ELUImpl(alpha=self.alpha, name=self.name)


class PReLU(Layer):
    """Wrapper for keras.layers.PReLU."""

    def __init__(self, shared_axes=None, name=None):
        self.shared_axes = shared_axes
        super().__init__(name=name, shared_axes=self.shared_axes)

    def build_layer(self):
        return keras.layers.PReLU(shared_axes=self.shared_axes, name=self.name)


class Softmax(Layer):
    """Wrapper for keras.layers.Softmax.

    By default a sample is normalized as a whole: its values over every dim,
    time and seq axis sum to one, as any other activation reads every value
    of it. ``axis`` normalizes along one axis of the tensor instead, counted
    with the batch axis as 0.
    """

    def __init__(self, axis: int | None = None, name=None):
        self.axis = None if axis is None else int(axis)
        super().__init__(name=name, axis=self.axis)

    def build_layer(self):
        axis = self.axis
        if axis is None:
            axis = list(range(1, 1 + self.preds[0].shape.rank))  # type: ignore
        return keras.layers.Softmax(axis=axis, name=self.name)  # type: ignore


class Sigmoid(Layer):
    """Wrapper for keras.layers.Activation('sigmoid')."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return keras.layers.Activation("sigmoid", name=self.name)


class Tanh(Layer):
    """Wrapper for keras.layers.Activation('tanh')."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return keras.layers.Activation("tanh", name=self.name)


class Swish(Layer):
    """Wrapper for keras.layers.Activation('swish')."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return keras.layers.Activation("swish", name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class GELUImpl(keras.layers.Layer):
    def __init__(self, approximate: bool = True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.approximate = bool(approximate)

    def call(self, inputs):
        return keras.activations.gelu(inputs, approximate=self.approximate)

    def get_config(self):
        config = super().get_config()
        config.update({"approximate": self.approximate})
        return config


class GELU(Layer):
    """Gaussian error linear unit, ``keras.activations.gelu``; ``approximate``
    uses its tanh approximation."""

    def __init__(self, approximate: bool = True, name=None):
        self.approximate = bool(approximate)
        super().__init__(name=name, approximate=self.approximate)

    def build_layer(self):
        return GELUImpl(approximate=self.approximate, name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class SoftplusImpl(keras.layers.Layer):
    def call(self, inputs):
        zero = keras.ops.cast(0.0, inputs.dtype)
        return keras.ops.log(
            1.0 + keras.ops.exp(-keras.ops.abs(inputs))
        ) + keras.ops.maximum(inputs, zero)

    def get_config(self):
        return super().get_config()


class Softplus(Layer):
    """``log(1 + exp(x))``, computed in a form that does not overflow for
    large ``x``."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return SoftplusImpl(name=self.name)
