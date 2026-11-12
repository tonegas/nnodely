"""Trigonometric layers wrappers for nnodely."""

from __future__ import annotations


from nnodely.layers.arithmetic import BinaryOp, UnaryOp


class Sin(UnaryOp):
    """Wrapper for sine transform."""

    _operation = "sin"


class Cos(UnaryOp):
    """Wrapper for cosine transform."""

    _operation = "cos"


class Tan(UnaryOp):
    """Wrapper for tangent transform."""

    _operation = "tan"


class Asin(UnaryOp):
    """Wrapper for inverse sine transform."""

    _operation = "arcsin"


class Acos(UnaryOp):
    """Wrapper for inverse cosine transform."""

    _operation = "arccos"


class Atan(UnaryOp):
    """Wrapper for inverse tangent transform."""

    _operation = "arctan"


class Sinh(UnaryOp):
    """Wrapper for hyperbolic sine transform."""

    _operation = "sinh"


class Cosh(UnaryOp):
    """Wrapper for hyperbolic cosine transform."""

    _operation = "cosh"


class Asinh(UnaryOp):
    """Wrapper for inverse hyperbolic sine transform."""

    _operation = "arcsinh"


class Acosh(UnaryOp):
    """Wrapper for inverse hyperbolic cosine transform, defined from 1."""

    _operation = "arccosh"


class Atanh(UnaryOp):
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
