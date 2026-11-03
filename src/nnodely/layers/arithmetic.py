"""Arithmetic layers for nnodely."""

import math

import keras

from nnodely.core.layer import BinaryOp, BinaryOpImpl, Layer


def _rad2deg(x):
    return keras.ops.multiply(x, 180.0 / math.pi)


#: The operations keras.ops does not have, by the name ArithmeticImpl is given.
_OPERATIONS = {"rad2deg": _rad2deg}


@keras.saving.register_keras_serializable(package="nnodely")
class ArithmeticImpl(keras.layers.Layer):
    def __init__(self, operation: str, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.operation = operation

    def call(self, inputs):
        operation = _OPERATIONS.get(self.operation) or getattr(
            keras.ops, self.operation
        )
        return operation(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"operation": self.operation})
        return config


class Arithmetic(Layer):
    _operation = ""
    _one_stream = True

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return ArithmeticImpl(operation=self._operation, name=self.name)


class Exp(Arithmetic):
    """Wrapper for exponential transform."""

    _operation = "exp"


class Expm1(Arithmetic):
    """Wrapper for ``exp(x) - 1``, accurate for small x."""

    _operation = "expm1"


class Log(Arithmetic):
    """Wrapper for natural logarithm transform."""

    _operation = "log"


class Log1p(Arithmetic):
    """Wrapper for ``log(1 + x)``, accurate for small x."""

    _operation = "log1p"


class Log10(Arithmetic):
    """Wrapper for base 10 logarithm transform."""

    _operation = "log10"


class Log2(Arithmetic):
    """Wrapper for base 2 logarithm transform."""

    _operation = "log2"


class Sqrt(Arithmetic):
    """Wrapper for square root transform."""

    _operation = "sqrt"


class Square(Arithmetic):
    """Wrapper for square transform."""

    _operation = "square"


class Abs(Arithmetic):
    """Wrapper for absolute value transform."""

    _operation = "abs"


class Floor(Arithmetic):
    """Wrapper for floor transform."""

    _operation = "floor"


class Ceil(Arithmetic):
    """Wrapper for ceiling transform."""

    _operation = "ceil"


class Round(Arithmetic):
    """Wrapper for rounding to the nearest integer, a half to the even one."""

    _operation = "round"


class Deg2Rad(Arithmetic):
    """Wrapper for degrees to radians transform."""

    _operation = "deg2rad"


class Rad2Deg(Arithmetic):
    """Wrapper for radians to degrees transform."""

    _operation = "rad2deg"


class Sign(Arithmetic):
    """Wrapper for sign transform."""

    _operation = "sign"


class Negative(Arithmetic):
    """Wrapper for negation transform."""

    _operation = "negative"


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

    _one_stream = True

    def __init__(self, min: float | None = None, max: float | None = None, name=None):
        if min is not None and max is not None and min > max:
            raise ValueError(
                f"{name or 'Clamp'}: min {min} must not be greater than max {max}."
            )
        self._min = min
        self._max = max
        super().__init__(name=name, min=self._min, max=self._max)

    def build_layer(self):
        return ClampImpl(min=self._min, max=self._max, name=self.name)


class _Extremum(BinaryOp):
    """The elementwise extremum of two or more streams."""

    #: The layer that reduces the dim axes of one stream instead.
    _reduction = ""

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        if len(inputs) < 2:
            kind = type(self).__name__
            raise ValueError(
                f"{self.name}: {kind} compares two or more streams elementwise, "
                f"{kind}([a, b]), got {len(inputs)}. Over the dim axes of one "
                f"stream, use {self._reduction}()(x)."
            )
        return super().output_shape(*inputs)


class Maximum(_Extremum):
    """``max(a, b, ...)`` elementwise, called on ``[a, b, ...]``. With
    nothing to configure, ``Maximum([a, b])`` is ``Maximum()([a, b])``.

    The streams broadcast as the arithmetic operators do, so a bound can be
    a Parameter, learned, where :class:`Clamp` takes a fixed number.
    :class:`Max` reduces the dim axes of one stream instead.
    """

    _operation = "maximum"
    _reduction = "Max"


class Minimum(_Extremum):
    """``min(a, b, ...)`` elementwise, called on ``[a, b, ...]``. With
    nothing to configure, ``Minimum([a, b])`` is ``Minimum()([a, b])``.

    The streams broadcast as the arithmetic operators do, so a bound can be
    a Parameter, learned, where :class:`Clamp` takes a fixed number.
    :class:`Min` reduces the dim axes of one stream instead.
    """

    _operation = "minimum"
    _reduction = "Min"


def _norm(x, axis, keepdims):
    squares = keras.ops.sum(keras.ops.square(x), axis=axis, keepdims=keepdims)
    # The slope of sqrt is infinite at 0: a zero vector is kept out of it, so
    # that its gradient is 0 rather than NaN.
    positive = squares > 0
    safe = keras.ops.where(positive, squares, keras.ops.ones_like(squares))
    return keras.ops.where(
        positive, keras.ops.sqrt(safe), keras.ops.zeros_like(squares)
    )


_REDUCTIONS = {
    "sum": keras.ops.sum,
    "mean": keras.ops.mean,
    "max": keras.ops.max,
    "min": keras.ops.min,
    "norm": _norm,
}


@keras.saving.register_keras_serializable(package="nnodely")
class ReduceImpl(keras.layers.Layer):
    """Serializable implementation of a reduction over the tensor axes
    ``axes``, which keeps them with length 1."""

    def __init__(self, operation: str, axes, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.operation = operation
        self.axes = tuple(int(axis) for axis in axes)

    def call(self, x):
        return _REDUCTIONS[self.operation](x, axis=self.axes, keepdims=True)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        for axis in self.axes:
            output_shape[axis] = 1
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"operation": self.operation, "axes": self.axes})
        return config


@keras.saving.register_keras_serializable(package="nnodely")
class SumImpl(ReduceImpl):
    """The sum, as the Keras files exported by nnodely 2.0.0 name it."""

    def __init__(self, axes, name=None, operation="sum", **kwargs):
        super().__init__(operation, axes, name=name, **kwargs)


class Reduction(Layer):
    """A reduction over the dim axes, which keeps them with length 1.

    With axis=None every dim axis is reduced together, otherwise only the
    chosen one.
    """

    _operation = ""

    def __init__(self, axis: int | None = None, name=None):
        self._axis = axis if axis is None else int(axis)
        super().__init__(name=name, axis=self._axis)

    def build_layer(self):
        shape = self.shape
        axes = (
            shape.dim_axes
            if self._axis is None
            else (shape.dim_axes[shape.resolve_dim(self._axis, self.name)],)
        )
        return ReduceImpl(operation=self._operation, axes=axes, name=self.name)


class Sum(Reduction):
    """
    Sum over the dim axes, keeping them with length 1.

    With axis=None every dim axis is summed together, otherwise only the
    chosen one.

    Examples
    --------
    dim=(2, 3), axis=None -> dim=(1, 1)
    dim=(2, 3), axis=0 -> dim=(1, 3)
    """

    _operation = "sum"


class Mean(Reduction):
    """Mean over the dim axes, keeping them with length 1. ``axis`` as for
    :class:`Sum`."""

    _operation = "mean"


class Max(Reduction):
    """Largest value over the dim axes, keeping them with length 1. ``axis``
    as for :class:`Sum`. :class:`Maximum` compares streams elementwise
    instead."""

    _operation = "max"


class Min(Reduction):
    """Smallest value over the dim axes, keeping them with length 1. ``axis``
    as for :class:`Sum`. :class:`Minimum` compares streams elementwise
    instead."""

    _operation = "min"


class Norm(Reduction):
    """Euclidean norm over the dim axes, keeping them with length 1: the
    length of a vector. ``axis`` as for :class:`Sum`.

    Its gradient at the zero vector, where the norm has none, is 0.
    """

    _operation = "norm"


def _cross(a, b, axis):
    # Written out with gathers rather than keras.ops.cross, whose axis the
    # backends choose differently.
    def turned(x, order):
        return keras.ops.take(x, order, axis=axis)

    ahead = turned(a, [1, 2, 0]) * turned(b, [2, 0, 1])
    return ahead - turned(a, [2, 0, 1]) * turned(b, [1, 2, 0])


@keras.saving.register_keras_serializable(package="nnodely")
class CrossImpl(BinaryOpImpl):
    """The cross product along the tensor axis ``axis``, counted from the
    last, of two operands aligned as BinaryOpImpl aligns them."""

    def __init__(self, axis: int = -1, operation="cross", **kwargs):
        super().__init__(operation=operation, **kwargs)
        self.axis = int(axis)

    def call(self, xs):
        a, b = self.aligned(xs)
        return _cross(a, b, self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class Cross(BinaryOp):
    """The cross product ``a x b`` of two streams of 3-vectors, called on
    ``[a, b]``, along the dim axis ``axis``, the last by default, which has
    length 3 in both.

    The streams broadcast along the window and the sequence as the
    arithmetic operators do: with ``omega`` a Parameter of dim 3,
    ``Cross()([omega, r])`` is ``omega x r`` at every sample of ``r``.
    """

    def __init__(self, axis: int = -1, name=None):
        self._axis = int(axis)
        super().__init__(name=name, axis=self._axis)

    def build_layer(self):
        shape = self._preds[0].shape  # type: ignore[union-attr]
        axis = shape.resolve_dim(self._axis, self.name) - shape.dim_rank
        return CrossImpl(axis=axis, **self._alignment(), name=self.name)

    def output_shape(self, *inputs):
        if len(inputs) != 2:
            raise ValueError(
                f"{self.name}: Cross reads two streams, [a, b], got {len(inputs)}."
            )
        for node in inputs:
            shape = node.shape
            size = shape.dim[shape.resolve_dim(self._axis, self.name)]
            if size != 3:
                raise ValueError(
                    f"{self.name}: Cross multiplies 3-vectors, and dim axis "
                    f"{self._axis} of {node.name!r}, {shape}, has length {size}."
                )
        return super().output_shape(*inputs)
