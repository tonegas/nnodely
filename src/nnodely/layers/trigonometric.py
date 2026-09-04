"""Trigonometric layers wrappers for nnodely."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class TrigImpl(keras.layers.Layer):
    """Elementwise keras.ops trig call. A named op instead of a Lambda closure,
    so the layer survives save/load."""

    def __init__(self, op, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.op = op

    def call(self, x):
        return getattr(keras.ops, self.op)(x)

    def get_config(self):
        config = super().get_config()
        config.update({"op": self.op})
        return config


class Sin(Layer):
    """Wrapper for sine transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return TrigImpl("sin", name=self.name)


class Cos(Layer):
    """Wrapper for cosine transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return TrigImpl("cos", name=self.name)


class Tan(Layer):
    """Wrapper for tangent transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return TrigImpl("tan", name=self.name)


class Asin(Layer):
    """Wrapper for inverse sine transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return TrigImpl("arcsin", name=self.name)


class Acos(Layer):
    """Wrapper for inverse cosine transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return TrigImpl("arccos", name=self.name)


class Atan(Layer):
    """Wrapper for inverse tangent transform."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return TrigImpl("arctan", name=self.name)
