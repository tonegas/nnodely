"""Trigonometric layers wrappers for nnodely."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


class Sin(Layer):
    """Wrapper for sine transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Lambda(lambda x: keras.ops.sin(x), name=self.name)
        return self._layer


class Cos(Layer):
    """Wrapper for cosine transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Lambda(lambda x: keras.ops.cos(x), name=self.name)
        return self._layer


class Tan(Layer):
    """Wrapper for tangent transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Lambda(lambda x: keras.ops.tan(x), name=self.name)
        return self._layer


class Asin(Layer):
    """Wrapper for inverse sine transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Lambda(lambda x: keras.ops.arcsin(x), name=self.name)
        return self._layer


class Acos(Layer):
    """Wrapper for inverse cosine transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Lambda(lambda x: keras.ops.arccos(x), name=self.name)
        return self._layer


class Atan(Layer):
    """Wrapper for inverse tangent transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        self._layer = keras.layers.Lambda(lambda x: keras.ops.arctan(x), name=self.name)
        return self._layer
