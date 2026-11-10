"""Trigonometric layers wrappers for nnodely."""

from __future__ import annotations

import keras

from nnodely.core.layer import BinaryOp, Layer


@keras.saving.register_keras_serializable(package="nnodely")
class TrigonometricImpl(keras.layers.Layer):
    def __init__(self, operation: str, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.operation = operation

    def call(self, inputs):
        return getattr(keras.ops, self.operation)(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"operation": self.operation})
        return config


class Trigonometric(Layer):
    _operation = ""
    _one_stream = True

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return TrigonometricImpl(operation=self._operation, name=self.name)


class Sin(Trigonometric):
    """Wrapper for sine transform."""

    _operation = "sin"


class Cos(Trigonometric):
    """Wrapper for cosine transform."""

    _operation = "cos"


class Tan(Trigonometric):
    """Wrapper for tangent transform."""

    _operation = "tan"


class Asin(Trigonometric):
    """Wrapper for inverse sine transform."""

    _operation = "arcsin"


class Acos(Trigonometric):
    """Wrapper for inverse cosine transform."""

    _operation = "arccos"


class Atan(Trigonometric):
    """Wrapper for inverse tangent transform."""

    _operation = "arctan"


class Sinh(Trigonometric):
    """Wrapper for hyperbolic sine transform."""

    _operation = "sinh"


class Cosh(Trigonometric):
    """Wrapper for hyperbolic cosine transform."""

    _operation = "cosh"


class Asinh(Trigonometric):
    """Wrapper for inverse hyperbolic sine transform."""

    _operation = "arcsinh"


class Acosh(Trigonometric):
    """Wrapper for inverse hyperbolic cosine transform, defined from 1."""

    _operation = "arccosh"


class Atanh(Trigonometric):
    """Wrapper for inverse hyperbolic tangent transform, defined in (-1, 1)."""

    _operation = "arctanh"


class Atan2(BinaryOp):
    """``atan2(y, x)``, called on ``[y, x]``: the angle of the point ``(x, y)``
    in ``[-pi, pi]``, its quadrant told by the signs of both. With nothing to
    configure, ``Atan2([y, x])`` is ``Atan2()([y, x])``.

    ``y`` and ``x`` broadcast as the arithmetic operators do: a stream
    without a window applies to every sample of the other, a scalar to every
    feature.
    """

    _operation = "arctan2"

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        if len(inputs) != 2:
            raise ValueError(
                f"{self.name}: Atan2 reads two streams, [y, x], got {len(inputs)}."
            )
        return super().output_shape(*inputs)
