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
        self._layer = keras.layers.ReLU(
            max_value=self.max_value,
            negative_slope=self.negative_slope,
            threshold=self.threshold,
            name=self.name,
        )
        return self._layer


class LeakyReLU(Layer):
    """Wrapper for keras.layers.LeakyReLU."""

    def __init__(self, negative_slope: float = 0.3, name=None):
        self.negative_slope = float(negative_slope)
        super().__init__(name=name, negative_slope=self.negative_slope)

    def build_layer(self):
        self._layer = keras.layers.LeakyReLU(
            negative_slope=self.negative_slope,
            name=self.name,
        )
        return self._layer


class ELU(Layer):
    """Wrapper for keras.layers.ELU."""

    def __init__(self, alpha: float = 1.0, name=None):
        self.alpha = float(alpha)
        super().__init__(name=name, alpha=self.alpha)

    def build_layer(self):
        self._layer = keras.layers.ELU(alpha=self.alpha, name=self.name)
        return self._layer


class PReLU(Layer):
    """Wrapper for keras.layers.PReLU."""

    def __init__(self, shared_axes=None, name=None):
        self.shared_axes = shared_axes
        super().__init__(name=name, shared_axes=self.shared_axes)

    def build_layer(self):
        self._layer = keras.layers.PReLU(shared_axes=self.shared_axes, name=self.name)
        return self._layer


class Softmax(Layer):
    """Wrapper for keras.layers.Softmax."""

    def __init__(self, axis: int = -1, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def build_layer(self):
        self._layer = keras.layers.Softmax(axis=self.axis, name=self.name)
        return self._layer


class Sigmoid(Layer):
    """Wrapper for keras.layers.Activation('sigmoid')."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Activation("sigmoid", name=self.name)
        return self._layer


class Tanh(Layer):
    """Wrapper for keras.layers.Activation('tanh')."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Activation("tanh", name=self.name)
        return self._layer


class Swish(Layer):
    """Wrapper for keras.layers.Activation('swish')."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Activation("swish", name=self.name)
        return self._layer


class GELU(Layer):
    """Wrapper for keras.layers.Activation('gelu')."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Activation("gelu", name=self.name)
        return self._layer


class Softplus(Layer):
    """Wrapper for keras.layers.Activation('softplus')."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Activation("softplus", name=self.name)
        return self._layer
