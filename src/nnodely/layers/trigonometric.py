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
    r"""Sine of every value of a stream, in radians.

    Syntax
    ------
    ``Sin(x)``
        The sine of ``x``; the same as ``Sin()(x)``.
    ``Sin(name="s")(x)``
        The same, with the layer named.
    ``Sin([a, b])``
        The sine of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Sin`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Cos : Cosine.
    Asin : Inverse sine.
    Sinh : Hyperbolic sine.
    Deg2Rad : Degrees to radians.

    Notes
    -----
    .. math:: y = \sin x

    Defined for every real ``x``, with values in :math:`[-1, 1]`. The
    gradient is :math:`\cos x`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Sin
    >>> x = Input("x", tw=2)
    >>> model = Modely("sine", inputs=[x], outputs=[Output("y", Sin(x))]).build()
    >>> result = model({"x": np.array([[[0.0], [np.pi / 2]]])})
    >>> keras.ops.convert_to_numpy(result["y"])
    array([[[0.],
            [1.]]], dtype=float32)

    Scope
    -----
    The sine applies to every value, whatever the shape:

    >>> Sin(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives the sine of each stream:

    >>> a, b = Input("a", tw=1), Input("b", tw=1)
    >>> [stream.shape for stream in Sin([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=1, D=(1,))]

    Options
    -------
    ``name`` names the layer, configured first:

    >>> Sin(name="s")(x).name
    's'

    Applications
    ------------
    The angular acceleration of a pendulum of length :math:`l`,
    :math:`\ddot\theta = -(g / l) \sin\theta`:

    >>> theta = Input("theta", tw=1)
    >>> acceleration = -9.81 / 0.5 * Sin(theta)

    Possible Issues
    ---------------
    A number is not a stream: use a :class:`Constant`, or compute the sine
    with NumPy.

    .. versionadded:: 2.0
    """

    _operation = "sin"


class Cos(Trigonometric):
    r"""Cosine of every value of a stream, in radians.

    Syntax
    ------
    ``Cos(x)``
        The cosine of ``x``; the same as ``Cos()(x)``.
    ``Cos(name="c")(x)``
        The same, with the layer named.
    ``Cos([a, b])``
        The cosine of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Cos`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Sin : Sine.
    Acos : Inverse cosine.
    Cosh : Hyperbolic cosine.
    Deg2Rad : Degrees to radians.

    Notes
    -----
    .. math:: y = \cos x

    Defined for every real ``x``, with values in :math:`[-1, 1]`. The
    gradient is :math:`-\sin x`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Cos, Deg2Rad, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("cosine", inputs=[x], outputs=[Output("y", Cos(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, np.pi]])})["y"])
    array([[ 1., -1.]], dtype=float32)

    Applications
    ------------
    The height of a pendulum bob of length 0.5 m, below its pivot at the
    angle :math:`\theta`:

    >>> theta = Input("theta", tw=1)
    >>> height = -0.5 * Cos(theta)

    Possible Issues
    ---------------
    The angle is in radians. Convert an angle in degrees first:

    >>> y = Cos(Deg2Rad(x))
    >>> model = Modely("degrees", inputs=[x], outputs=[Output("y", y)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 180.0]])})["y"])
    array([[ 1., -1.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "cos"


class Tan(Trigonometric):
    r"""Tangent of every value of a stream, in radians.

    Syntax
    ------
    ``Tan(x)``
        The tangent of ``x``; the same as ``Tan()(x)``.
    ``Tan(name="t")(x)``
        The same, with the layer named.
    ``Tan([a, b])``
        The tangent of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Tan`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Sin : Sine.
    Cos : Cosine.
    Atan : Inverse tangent.

    Notes
    -----
    .. math:: y = \tan x = \frac{\sin x}{\cos x}

    Defined for every real ``x`` but :math:`\pi/2 + k\pi`, where it has
    poles. The gradient is :math:`1 + \tan^2 x`, which grows without bound
    near the poles.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Tan
    >>> x = Input("x", dim=3)
    >>> model = Modely("tangent", inputs=[x], outputs=[Output("y", Tan(x))]).build()
    >>> result = model({"x": np.array([[0.0, np.pi / 4, -np.pi / 4]])})
    >>> keras.ops.convert_to_numpy(result["y"]).round(4)
    array([[ 0.,  1., -1.]], dtype=float32)

    Possible Issues
    ---------------
    In float32, :math:`\pi/2` is not exactly a pole. The result is a large
    finite number, about :math:`-2.3 \cdot 10^7`, not an infinity:

    >>> y = keras.ops.convert_to_numpy(model({"x": np.array([[np.pi / 2, 0.0, 0.0]])})["y"])
    >>> bool(abs(y[0, 0]) > 1e7)
    True

    .. versionadded:: 2.0
    """

    _operation = "tan"


class Asin(Trigonometric):
    r"""Inverse sine of every value of a stream, in radians.

    Syntax
    ------
    ``Asin(x)``
        The inverse sine of ``x``; the same as ``Asin()(x)``.
    ``Asin(name="a")(x)``
        The same, with the layer named.
    ``Asin([a, b])``
        The inverse sine of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Asin`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Sin : Sine.
    Acos : Inverse cosine.
    Atan : Inverse tangent.
    Clamp : Bounds the values of a stream.

    Notes
    -----
    .. math:: y = \arcsin x

    Defined for :math:`-1 \le x \le 1`, with values in
    :math:`[-\pi/2, \pi/2]`. Outside the domain the result is NaN, with no
    error. The gradient is :math:`1 / \sqrt{1 - x^2}`, infinite at
    :math:`x = \pm 1`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Asin, Clamp, Input, Modely, Output
    >>> x = Input("x", dim=3)
    >>> model = Modely("arcsine", inputs=[x], outputs=[Output("y", Asin(x))]).build()
    >>> result = model({"x": np.array([[0.0, 0.5, 1.0]])})
    >>> keras.ops.convert_to_numpy(result["y"]).round(4)  # 0, pi/6, pi/2
    array([[0.    , 0.5236, 1.5708]], dtype=float32)

    Possible Issues
    ---------------
    A value outside :math:`[-1, 1]` gives NaN, with no error:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[2.0, 0.0, 0.0]])})["y"])
    array([[nan,  0.,  0.]], dtype=float32)

    Bound the stream first when it can leave the domain:

    >>> y = Asin(Clamp(min=-1.0, max=1.0)(x))

    .. versionadded:: 2.0
    """

    _operation = "arcsin"


class Acos(Trigonometric):
    r"""Inverse cosine of every value of a stream, in radians.

    Syntax
    ------
    ``Acos(x)``
        The inverse cosine of ``x``; the same as ``Acos()(x)``.
    ``Acos(name="a")(x)``
        The same, with the layer named.
    ``Acos([a, b])``
        The inverse cosine of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Acos`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Cos : Cosine.
    Asin : Inverse sine.
    Atan2 : Angle of a point, in all four quadrants.
    Clamp : Bounds the values of a stream.

    Notes
    -----
    .. math:: y = \arccos x

    Defined for :math:`-1 \le x \le 1`, with values in :math:`[0, \pi]`.
    Outside the domain the result is NaN, with no error. The gradient is
    :math:`-1 / \sqrt{1 - x^2}`, infinite at :math:`x = \pm 1`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Acos, Input, Modely, Output
    >>> x = Input("x", dim=3)
    >>> model = Modely("arccosine", inputs=[x], outputs=[Output("y", Acos(x))]).build()
    >>> result = model({"x": np.array([[1.0, 0.5, 0.0]])})
    >>> keras.ops.convert_to_numpy(result["y"]).round(4)  # 0, pi/3, pi/2
    array([[0.    , 1.0472, 1.5708]], dtype=float32)

    Possible Issues
    ---------------
    A value outside :math:`[-1, 1]` gives NaN, with no error. Bound the
    stream with :class:`Clamp` when it can leave the domain:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[2.0, 1.0, 1.0]])})["y"])
    array([[nan,  0.,  0.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "arccos"


class Atan(Trigonometric):
    r"""Inverse tangent of every value of a stream, in radians.

    Syntax
    ------
    ``Atan(x)``
        The inverse tangent of ``x``; the same as ``Atan()(x)``.
    ``Atan(name="a")(x)``
        The same, with the layer named.
    ``Atan([a, b])``
        The inverse tangent of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Atan`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Tan : Tangent.
    Atan2 : Angle of a point, in all four quadrants.
    Asin : Inverse sine.

    Notes
    -----
    .. math:: y = \arctan x

    Defined for every real ``x``, with values in :math:`(-\pi/2, \pi/2)`.
    The gradient is :math:`1 / (1 + x^2)`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Atan, Atan2, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("arctangent", inputs=[x], outputs=[Output("y", Atan(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 1.0]])})["y"]).round(4)  # 0, pi/4
    array([[0.    , 0.7854]], dtype=float32)

    Possible Issues
    ---------------
    The angle of a point :math:`(p, q)` is not ``Atan(q / p)``: the ratio
    loses the signs, so the quadrant. Use :class:`Atan2`. For the point
    :math:`(-1, 1)`, at :math:`3\pi/4`:

    >>> p, q = Input("p"), Input("q")
    >>> outputs = [Output("ratio", Atan(q / p)), Output("angle", Atan2([q, p]))]
    >>> model = Modely("angle", inputs=[p, q], outputs=outputs).build()
    >>> result = model({"p": np.array([[-1.0]]), "q": np.array([[1.0]])})
    >>> [keras.ops.convert_to_numpy(result[k]).round(4) for k in ("ratio", "angle")]
    [array([[-0.7854]], dtype=float32), array([[2.3562]], dtype=float32)]

    .. versionadded:: 2.0
    """

    _operation = "arctan"


class Sinh(Trigonometric):
    r"""Hyperbolic sine of every value of a stream.

    Syntax
    ------
    ``Sinh(x)``
        The hyperbolic sine of ``x``; the same as ``Sinh()(x)``.
    ``Sinh(name="s")(x)``
        The same, with the layer named.
    ``Sinh([a, b])``
        The hyperbolic sine of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Sinh`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Cosh : Hyperbolic cosine.
    Asinh : Inverse hyperbolic sine.
    Exp : Exponential.

    Notes
    -----
    .. math:: y = \sinh x = \frac{e^x - e^{-x}}{2}

    Defined for every real ``x``. The gradient is :math:`\cosh x`. In
    float32 the result overflows to :math:`\pm\infty` for :math:`|x|` above
    about 89.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Sinh
    >>> x = Input("x", dim=2)
    >>> model = Modely("hsine", inputs=[x], outputs=[Output("y", Sinh(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 1.0]])})["y"]).round(4)
    array([[0.    , 1.1752]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "sinh"


class Cosh(Trigonometric):
    r"""Hyperbolic cosine of every value of a stream.

    Syntax
    ------
    ``Cosh(x)``
        The hyperbolic cosine of ``x``; the same as ``Cosh()(x)``.
    ``Cosh(name="c")(x)``
        The same, with the layer named.
    ``Cosh([a, b])``
        The hyperbolic cosine of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Cosh`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Sinh : Hyperbolic sine.
    Acosh : Inverse hyperbolic cosine.
    Exp : Exponential.

    Notes
    -----
    .. math:: y = \cosh x = \frac{e^x + e^{-x}}{2}

    Defined for every real ``x``, with values in :math:`[1, \infty)`. The
    gradient is :math:`\sinh x`. In float32 the result overflows to
    :math:`\infty` for :math:`|x|` above about 89.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Cosh, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("hcosine", inputs=[x], outputs=[Output("y", Cosh(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 1.0]])})["y"]).round(4)
    array([[1.    , 1.5431]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "cosh"


class Asinh(Trigonometric):
    r"""Inverse hyperbolic sine of every value of a stream.

    Syntax
    ------
    ``Asinh(x)``
        The inverse hyperbolic sine of ``x``; the same as ``Asinh()(x)``.
    ``Asinh(name="a")(x)``
        The same, with the layer named.
    ``Asinh([a, b])``
        The inverse hyperbolic sine of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Asinh`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Sinh : Hyperbolic sine.
    Acosh : Inverse hyperbolic cosine.
    Log : Natural logarithm.

    Notes
    -----
    .. math:: y = \operatorname{arsinh} x = \ln\left(x + \sqrt{x^2 + 1}\right)

    Defined for every real ``x``. The gradient is :math:`1 / \sqrt{x^2 + 1}`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Asinh, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("ahsine", inputs=[x], outputs=[Output("y", Asinh(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 1.0]])})["y"]).round(4)
    array([[0.    , 0.8814]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "arcsinh"


class Acosh(Trigonometric):
    r"""Inverse hyperbolic cosine of every value of a stream.

    Syntax
    ------
    ``Acosh(x)``
        The inverse hyperbolic cosine of ``x``; the same as ``Acosh()(x)``.
    ``Acosh(name="a")(x)``
        The same, with the layer named.
    ``Acosh([a, b])``
        The inverse hyperbolic cosine of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Acosh`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Cosh : Hyperbolic cosine.
    Asinh : Inverse hyperbolic sine.
    Atanh : Inverse hyperbolic tangent.
    Clamp : Bounds the values of a stream.

    Notes
    -----
    .. math:: y = \operatorname{arcosh} x = \ln\left(x + \sqrt{x^2 - 1}\right)

    Defined for :math:`x \ge 1`, with values in :math:`[0, \infty)`. Below 1
    the result is NaN, with no error. The gradient is
    :math:`1 / \sqrt{x^2 - 1}`, infinite at :math:`x = 1`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Acosh, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("ahcosine", inputs=[x], outputs=[Output("y", Acosh(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 2.0]])})["y"]).round(4)
    array([[0.   , 1.317]], dtype=float32)

    Possible Issues
    ---------------
    A value below 1 gives NaN, with no error. Bound the stream with
    :class:`Clamp`, ``Clamp(min=1.0)``, when it can leave the domain:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.5, 1.0]])})["y"])
    array([[nan,  0.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "arccosh"


class Atanh(Trigonometric):
    r"""Inverse hyperbolic tangent of every value of a stream.

    Syntax
    ------
    ``Atanh(x)``
        The inverse hyperbolic tangent of ``x``; the same as ``Atanh()(x)``.
    ``Atanh(name="a")(x)``
        The same, with the layer named.
    ``Atanh([a, b])``
        The inverse hyperbolic tangent of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Atanh`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    See Also
    --------
    Asinh : Inverse hyperbolic sine.
    Acosh : Inverse hyperbolic cosine.
    Clamp : Bounds the values of a stream.

    Notes
    -----
    .. math:: y = \operatorname{artanh} x = \frac{1}{2} \ln\frac{1 + x}{1 - x}

    Defined for :math:`-1 < x < 1`. At :math:`x = \pm 1` the result is
    :math:`\pm\infty`, and outside :math:`[-1, 1]` it is NaN, with no error.
    The gradient is :math:`1 / (1 - x^2)`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Atanh, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("ahtangent", inputs=[x], outputs=[Output("y", Atanh(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 0.5]])})["y"]).round(4)
    array([[0.    , 0.5493]], dtype=float32)

    Possible Issues
    ---------------
    At 1 the result is infinite, and beyond it NaN, with no error:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 2.0]])})["y"])
    array([[inf, nan]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "arctanh"


class Atan2(BinaryOp):
    r"""Angle of the point ``(x, y)``, in radians in :math:`[-\pi, \pi]`, for the streams ``[y, x]``.

    Syntax
    ------
    ``Atan2([y, x])``
        The angle of the point ``(x, y)``; the same as ``Atan2()([y, x])``.
    ``Atan2(name="angle")([y, x])``
        The same, with the layer named.

    The two streams go in one list, ``y`` first, as in ``atan2(y, x)``. The
    signs of both give the quadrant, which ``Atan(y / x)`` loses. ``y`` and
    ``x`` combine as the arithmetic operators combine streams:

    - a stream without a window applies to every sample of the other;
    - a scalar, ``D=(1,)``, applies to every feature of the other.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Atan2`` followed by a number.

    Call arguments
    --------------
    [y, x] : list of two Stream
        The ordinate and the abscissa. Their shapes combine as for ``y + x``.

    Returns
    -------
    Stream
        A stream of the shape of ``y + x``. For example, ``y`` of
        ``Shape(S=(), T=3, D=(1,))`` and ``x`` of ``Shape(S=(), T=None,
        D=(1,))`` give ``Shape(S=(), T=3, D=(1,))``.

    Raises
    ------
    ValueError
        If the list does not hold two streams ("Atan2 reads two streams, [y, x],
        got 1"), or if their shapes do not combine ("cannot be combined").

    See Also
    --------
    Atan : Inverse tangent of one stream.
    Sin : Sine.
    Cos : Cosine.
    Norm : Length of a vector.

    Notes
    -----
    .. math::

        y_{out} = \operatorname{atan2}(y, x), \qquad
        \cos y_{out} = \frac{x}{\sqrt{x^2 + y^2}}, \quad
        \sin y_{out} = \frac{y}{\sqrt{x^2 + y^2}}

    The gradients are :math:`x / (x^2 + y^2)` with respect to ``y`` and
    :math:`-y / (x^2 + y^2)` with respect to ``x``. At the origin the angle is
    0 and the gradient is undefined: NaN on TensorFlow and JAX, 0 on PyTorch.

    Basic Examples
    --------------
    The angles of the points :math:`(1, 1)`, :math:`(-1, 1)` and
    :math:`(-1, -1)`:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Atan2, Constant, Input, Modely, Output
    >>> y, x = Input("y", dim=3), Input("x", dim=3)
    >>> model = Modely("angle", inputs=[y, x], outputs=[Output("a", Atan2([y, x]))]).build()
    >>> result = model({"y": np.array([[1.0, 1.0, -1.0]]), "x": np.array([[1.0, -1.0, -1.0]])})
    >>> keras.ops.convert_to_numpy(result["a"]).round(4)  # pi/4, 3 pi/4, -3 pi/4
    array([[ 0.7854,  2.3562, -2.3562]], dtype=float32)

    Scope
    -----
    A stream without a window applies to every sample of a window:

    >>> Atan2([Input("q", tw=3), Constant("p", value=-1.0)]).shape
    Shape(S=(), T=3, D=(1,))

    Options
    -------
    ``name`` names the layer, configured first:

    >>> Atan2(name="heading")([y, x]).name
    'heading'

    Applications
    ------------
    The heading of a vehicle from the two components of its velocity:

    >>> vx, vy = Input("vx", tw=1), Input("vy", tw=1)
    >>> heading = Atan2([vy, vx])

    Possible Issues
    ---------------
    The two streams go in one list. ``Atan2(y, x)`` reads ``x`` as the name
    of the layer, and fails. A single stream raises:

    >>> Atan2()(y)
    Traceback (most recent call last):
        ...
    ValueError: ...Atan2 reads two streams, [y, x], got 1.
    >>> Atan2([y, x]).shape
    Shape(S=(), T=None, D=(3,))

    .. versionadded:: 2.0
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
