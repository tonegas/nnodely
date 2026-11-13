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
    r"""Exponential of every value of a stream.

    Syntax
    ------
    ``Exp(x)``
        The exponential of ``x``; the same as ``Exp()(x)``.
    ``Exp(name="e")(x)``
        The same, with the layer named.
    ``Exp([a, b])``
        The exponential of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Exp`` followed by a number.

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
    Expm1 : :math:`e^x - 1`, accurate near 0.
    Log : Natural logarithm.
    Sinh : Hyperbolic sine.

    Notes
    -----
    .. math:: y = e^x

    Defined for every real ``x``, with values in :math:`(0, \infty)`. The
    gradient is :math:`e^x`. In float32 the result overflows to
    :math:`\infty` for ``x`` above about 88.7.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Exp, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("exponential", inputs=[x], outputs=[Output("y", Exp(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 1.0]])})["y"]).round(4)
    array([[1.    , 2.7183]], dtype=float32)

    Applications
    ------------
    The discharge of a capacitor, :math:`v = v_0 e^{-t / \tau}`, with
    :math:`\tau = 0.5` s:

    >>> t = Input("t", tw=1)
    >>> voltage = 12.0 * Exp(-t / 0.5)

    Possible Issues
    ---------------
    A large value overflows to infinity, with no error:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[100.0, 0.0]])})["y"])
    array([[inf,  1.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "exp"


class Expm1(Arithmetic):
    r"""Exponential minus one, :math:`e^x - 1`, of every value of a stream, accurate near 0.

    Syntax
    ------
    ``Expm1(x)``
        The value :math:`e^x - 1` of ``x``; the same as ``Expm1()(x)``.
    ``Expm1(name="e")(x)``
        The same, with the layer named.
    ``Expm1([a, b])``
        The value :math:`e^x - 1` of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Expm1`` followed by a number.

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
    Exp : Exponential.
    Log1p : :math:`\ln(1 + x)`, its inverse.

    Notes
    -----
    .. math:: y = e^x - 1

    Defined for every real ``x``, with values in :math:`(-1, \infty)`. The
    gradient is :math:`e^x`. Near 0 it keeps the digits that
    ``Exp(x) - 1`` loses in float32.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Exp, Expm1, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("expm1", inputs=[x], outputs=[Output("y", Expm1(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 1.0]])})["y"]).round(4)
    array([[0.    , 1.7183]], dtype=float32)

    Properties & Relations
    ----------------------
    ``Expm1(x)`` is ``Exp(x) - 1``, more accurate for a small ``x``. For
    :math:`x = 10^{-7}`, scaled by :math:`10^7`:

    >>> s = Input("s")
    >>> outputs = [Output("expm1", Expm1(s)), Output("exp", Exp(s) - 1.0)]
    >>> model = Modely("small", inputs=[s], outputs=outputs).build()
    >>> result = model({"s": np.array([[1e-7]])})
    >>> [(keras.ops.convert_to_numpy(result[k]) * 1e7).round(2) for k in ("expm1", "exp")]
    [array([[1.]], dtype=float32), array([[1.19]], dtype=float32)]

    .. versionadded:: 2.0
    """

    _operation = "expm1"


class Log(Arithmetic):
    r"""Natural logarithm of every value of a stream.

    Syntax
    ------
    ``Log(x)``
        The natural logarithm of ``x``; the same as ``Log()(x)``.
    ``Log(name="l")(x)``
        The same, with the layer named.
    ``Log([a, b])``
        The natural logarithm of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Log`` followed by a number.

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
    Exp : Exponential, its inverse.
    Log1p : :math:`\ln(1 + x)`, accurate near 0.
    Log10 : Base-10 logarithm.
    Log2 : Base-2 logarithm.

    Notes
    -----
    .. math:: y = \ln x

    Defined for :math:`x > 0`. At 0 the result is :math:`-\infty`, and
    below 0 it is NaN, with no error. The gradient is :math:`1 / x`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Log, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("logarithm", inputs=[x], outputs=[Output("y", Log(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 100.0]])})["y"]).round(4)
    array([[0.    , 4.6052]], dtype=float32)

    Possible Issues
    ---------------
    Zero gives minus infinity and a negative value NaN, with no error:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, -1.0]])})["y"])
    array([[-inf,  nan]], dtype=float32)

    Keep the argument positive, for example ``Log(Exp(z) + 1.0)`` or
    ``Log(x * x + 1e-6)``.

    .. versionadded:: 2.0
    """

    _operation = "log"


class Log1p(Arithmetic):
    r"""Logarithm of one plus every value of a stream, :math:`\ln(1 + x)`, accurate near 0.

    Syntax
    ------
    ``Log1p(x)``
        The value :math:`\ln(1 + x)` of ``x``; the same as ``Log1p()(x)``.
    ``Log1p(name="l")(x)``
        The same, with the layer named.
    ``Log1p([a, b])``
        The value :math:`\ln(1 + x)` of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Log1p`` followed by a number.

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
    Log : Natural logarithm.
    Expm1 : :math:`e^x - 1`, its inverse.

    Notes
    -----
    .. math:: y = \ln(1 + x)

    Defined for :math:`x > -1`. At :math:`-1` the result is :math:`-\infty`,
    and below it NaN, with no error. The gradient is :math:`1 / (1 + x)`.
    Near 0 it keeps the digits that ``Log(1 + x)`` loses in float32.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Log, Log1p, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("log1p", inputs=[x], outputs=[Output("y", Log1p(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 1.0]])})["y"]).round(4)
    array([[0.    , 0.6931]], dtype=float32)

    Properties & Relations
    ----------------------
    ``Log1p(x)`` is ``Log(1 + x)``, more accurate for a small ``x``. For
    :math:`x = 10^{-7}`, scaled by :math:`10^7`:

    >>> s = Input("s")
    >>> outputs = [Output("log1p", Log1p(s)), Output("log", Log(s + 1.0))]
    >>> small = Modely("small", inputs=[s], outputs=outputs).build()
    >>> result = small({"s": np.array([[1e-7]])})
    >>> [(keras.ops.convert_to_numpy(result[k]) * 1e7).round(2) for k in ("log1p", "log")]
    [array([[1.]], dtype=float32), array([[1.19]], dtype=float32)]

    Possible Issues
    ---------------
    At :math:`-1` the result is minus infinity, and below it NaN, with no
    error:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-1.0, -2.0]])})["y"])
    array([[-inf,  nan]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "log1p"


class Log10(Arithmetic):
    r"""Base-10 logarithm of every value of a stream.

    Syntax
    ------
    ``Log10(x)``
        The base-10 logarithm of ``x``; the same as ``Log10()(x)``.
    ``Log10(name="l")(x)``
        The same, with the layer named.
    ``Log10([a, b])``
        The base-10 logarithm of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Log10`` followed by a number.

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
    Log : Natural logarithm.
    Log2 : Base-2 logarithm.

    Notes
    -----
    .. math:: y = \log_{10} x = \frac{\ln x}{\ln 10}

    Defined for :math:`x > 0`. At 0 the result is :math:`-\infty`, and
    below 0 it is NaN, with no error. The gradient is
    :math:`1 / (x \ln 10)`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Log10, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("log10", inputs=[x], outputs=[Output("y", Log10(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 1000.0]])})["y"])
    array([[0., 3.]], dtype=float32)

    Possible Issues
    ---------------
    Zero gives minus infinity and a negative value NaN, with no error:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, -1.0]])})["y"])
    array([[-inf,  nan]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "log10"


class Log2(Arithmetic):
    r"""Base-2 logarithm of every value of a stream.

    Syntax
    ------
    ``Log2(x)``
        The base-2 logarithm of ``x``; the same as ``Log2()(x)``.
    ``Log2(name="l")(x)``
        The same, with the layer named.
    ``Log2([a, b])``
        The base-2 logarithm of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Log2`` followed by a number.

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
    Log : Natural logarithm.
    Log10 : Base-10 logarithm.

    Notes
    -----
    .. math:: y = \log_2 x = \frac{\ln x}{\ln 2}

    Defined for :math:`x > 0`. At 0 the result is :math:`-\infty`, and
    below 0 it is NaN, with no error. The gradient is
    :math:`1 / (x \ln 2)`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Log2, Modely, Output
    >>> x = Input("x", dim=4)
    >>> model = Modely("log2", inputs=[x], outputs=[Output("y", Log2(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 2.0, 8.0, 0.5]])})["y"])
    array([[ 0.,  1.,  3., -1.]], dtype=float32)

    Possible Issues
    ---------------
    Zero gives minus infinity and a negative value NaN, with no error:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, -1.0, 1.0, 1.0]])})["y"])
    array([[-inf,  nan,   0.,   0.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "log2"


class Sqrt(Arithmetic):
    r"""Square root of every value of a stream.

    Syntax
    ------
    ``Sqrt(x)``
        The square root of ``x``; the same as ``Sqrt()(x)``.
    ``Sqrt(name="r")(x)``
        The same, with the layer named.
    ``Sqrt([a, b])``
        The square root of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Sqrt`` followed by a number.

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
    Square : Square.
    Norm : Length of a vector, with a gradient at the zero vector.
    Abs : Absolute value.

    Notes
    -----
    .. math:: y = \sqrt{x}

    Defined for :math:`x \ge 0`. Below 0 the result is NaN, with no error.
    The gradient is :math:`1 / (2\sqrt{x})`, infinite at 0.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Gradient, Input, Modely, Output, Sqrt
    >>> x = Input("x", dim=3)
    >>> model = Modely("root", inputs=[x], outputs=[Output("y", Sqrt(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 4.0, 9.0]])})["y"])
    array([[0., 2., 3.]], dtype=float32)

    Possible Issues
    ---------------
    A negative value gives NaN, with no error:

    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-1.0, 4.0, 9.0]])})["y"])
    array([[nan,  2.,  3.]], dtype=float32)

    The gradient is infinite at 0, so a training that reaches 0 gets
    infinities or NaN:

    >>> slope = Gradient(wrt=x)(Sqrt(x))
    >>> model = Modely("slope", inputs=[x], outputs=[Output("d", slope)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.0, 4.0, 1.0]])})["d"])
    array([[ inf, 0.25, 0.5 ]], dtype=float32)

    Keep the argument away from 0, as in ``Sqrt(x + 1e-6)``. For the length
    of a vector use :class:`Norm`, whose gradient at the zero vector is 0.

    .. versionadded:: 2.0
    """

    _operation = "sqrt"


class Square(Arithmetic):
    r"""Square of every value of a stream.

    Syntax
    ------
    ``Square(x)``
        The square of ``x``; the same as ``Square()(x)``.
    ``Square(name="s")(x)``
        The same, with the layer named.
    ``Square([a, b])``
        The square of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Square`` followed by a number.

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
    Sqrt : Square root.
    Norm : Length of a vector.
    Abs : Absolute value.

    Notes
    -----
    .. math:: y = x^2

    Defined for every real ``x``. The gradient is :math:`2x`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Square
    >>> x = Input("x", dim=2)
    >>> model = Modely("square", inputs=[x], outputs=[Output("y", Square(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-3.0, 2.0]])})["y"])
    array([[9., 4.]], dtype=float32)

    Applications
    ------------
    The aerodynamic drag of a vehicle, :math:`F = \frac{1}{2} \rho c_d A v^2`:

    >>> velocity = Input("velocity", tw=1)
    >>> drag = 0.5 * 1.2 * 0.3 * 2.0 * Square(velocity)

    Properties & Relations
    ----------------------
    ``Square(x)`` is ``x * x`` and ``x ** 2``:

    >>> outputs = [Output("a", Square(x)), Output("b", x * x), Output("c", x**2)]
    >>> model = Modely("same", inputs=[x], outputs=outputs).build()
    >>> result = model({"x": np.array([[-3.0, 2.0]])})
    >>> [keras.ops.convert_to_numpy(result[k]) for k in "abc"]
    [array([[9., 4.]], dtype=float32), array([[9., 4.]], dtype=float32), array([[9., 4.]], dtype=float32)]

    .. versionadded:: 2.0
    """

    _operation = "square"


class Abs(Arithmetic):
    r"""Absolute value of every value of a stream.

    Syntax
    ------
    ``Abs(x)``
        The absolute value of ``x``; the same as ``Abs()(x)``.
    ``Abs(name="a")(x)``
        The same, with the layer named.
    ``Abs([a, b])``
        The absolute value of each stream of the list, as a list.
    ``abs(x)``
        The same, written with the Python built-in.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Abs`` followed by a number.

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
    Sign : Sign.
    Square : Square.
    Norm : Length of a vector.

    Notes
    -----
    .. math:: y = |x|

    Defined for every real ``x``. The gradient is :math:`\operatorname{sign}
    x`, :math:`\pm 1`. At 0 it depends on the backend: 0 on TensorFlow and
    PyTorch, 1 on JAX.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Abs, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("absolute", inputs=[x], outputs=[Output("y", Abs(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-2.0, 3.0]])})["y"])
    array([[2., 3.]], dtype=float32)

    Applications
    ------------
    Coulomb friction opposes the motion with a force of constant size; its
    power loss is :math:`\mu N |v|`:

    >>> velocity = Input("velocity", tw=1)
    >>> power_loss = 0.3 * 9.81 * Abs(velocity)

    Properties & Relations
    ----------------------
    ``abs(x)`` is ``Abs(x)``:

    >>> outputs = [Output("a", Abs(x)), Output("b", abs(x))]
    >>> result = Modely("same", inputs=[x], outputs=outputs).build()({"x": np.array([[-2.0, 3.0]])})
    >>> keras.ops.convert_to_numpy(result["a"]), keras.ops.convert_to_numpy(result["b"])
    (array([[2., 3.]], dtype=float32), array([[2., 3.]], dtype=float32))

    .. versionadded:: 2.0
    """

    _operation = "abs"


class Floor(Arithmetic):
    r"""Floor of every value of a stream: the largest integer not greater than it.

    Syntax
    ------
    ``Floor(x)``
        The floor of ``x``; the same as ``Floor()(x)``.
    ``Floor(name="f")(x)``
        The same, with the layer named.
    ``Floor([a, b])``
        The floor of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Floor`` followed by a number.

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
    Ceil : Smallest integer not less than each value.
    Round : Nearest integer.
    Sign : Sign.

    Notes
    -----
    .. math:: y = \lfloor x \rfloor

    The result is the largest integer not greater than ``x``, as a float.

    The gradient is 0 everywhere, at the steps too. A weight that reaches
    the output only through ``Floor`` is not trained by it.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Floor, Gradient, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("step", inputs=[x], outputs=[Output("y", Floor(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-1.5, 1.5]])})["y"])
    array([[-2.,  1.]], dtype=float32)

    Possible Issues
    ---------------
    The gradient is 0, so nothing before ``Floor`` is trained through it:

    >>> slope = Gradient(wrt=x)(Floor(x))
    >>> model = Modely("slope", inputs=[x], outputs=[Output("d", slope)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-1.5, 1.5]])})["d"])
    array([[0., 0.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "floor"


class Ceil(Arithmetic):
    r"""Ceiling of every value of a stream: the smallest integer not less than it.

    Syntax
    ------
    ``Ceil(x)``
        The ceiling of ``x``; the same as ``Ceil()(x)``.
    ``Ceil(name="c")(x)``
        The same, with the layer named.
    ``Ceil([a, b])``
        The ceiling of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Ceil`` followed by a number.

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
    Floor : Largest integer not greater than each value.
    Round : Nearest integer.
    Sign : Sign.

    Notes
    -----
    .. math:: y = \lceil x \rceil

    The result is the smallest integer not less than ``x``, as a float.

    The gradient is 0 everywhere, at the steps too. A weight that reaches
    the output only through ``Ceil`` is not trained by it.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Ceil, Gradient, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("step", inputs=[x], outputs=[Output("y", Ceil(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-1.5, 1.5]])})["y"])
    array([[-1.,  2.]], dtype=float32)

    Possible Issues
    ---------------
    The gradient is 0, so nothing before ``Ceil`` is trained through it:

    >>> slope = Gradient(wrt=x)(Ceil(x))
    >>> model = Modely("slope", inputs=[x], outputs=[Output("d", slope)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-1.5, 1.5]])})["d"])
    array([[0., 0.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "ceil"


class Round(Arithmetic):
    r"""Every value of a stream rounded to the nearest integer, a half to the even one.

    Syntax
    ------
    ``Round(x)``
        The rounded value of ``x``; the same as ``Round()(x)``.
    ``Round(name="r")(x)``
        The same, with the layer named.
    ``Round([a, b])``
        The rounded value of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Round`` followed by a number.

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
    Floor : Largest integer not greater than each value.
    Ceil : Smallest integer not less than each value.

    Notes
    -----
    .. math:: y = \operatorname{round}(x)

    The result is the nearest integer, as a float. A value halfway between
    two integers goes to the even one: 0.5 gives 0 and 2.5 gives 2.

    The gradient is 0 everywhere, at the steps too. A weight that reaches
    the output only through ``Round`` is not trained by it.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Gradient, Input, Modely, Output, Round
    >>> x = Input("x", dim=4)
    >>> model = Modely("step", inputs=[x], outputs=[Output("y", Round(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.5, 1.5, 2.5, 2.6]])})["y"])
    array([[0., 2., 2., 3.]], dtype=float32)

    Possible Issues
    ---------------
    The gradient is 0, so nothing before ``Round`` is trained through it:

    >>> slope = Gradient(wrt=x)(Round(x))
    >>> model = Modely("slope", inputs=[x], outputs=[Output("d", slope)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[0.5, 1.5, 2.5, 2.6]])})["d"])
    array([[0., 0., 0., 0.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "round"


class Deg2Rad(Arithmetic):
    r"""Every value of a stream converted from degrees to radians.

    Syntax
    ------
    ``Deg2Rad(x)``
        The angle in radians of ``x``; the same as ``Deg2Rad()(x)``.
    ``Deg2Rad(name="d")(x)``
        The same, with the layer named.
    ``Deg2Rad([a, b])``
        The angle in radians of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Deg2Rad`` followed by a number.

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
    Rad2Deg : Radians to degrees, its inverse.
    Sin : Sine, of an angle in radians.
    Cos : Cosine, of an angle in radians.

    Notes
    -----
    .. math:: y = \frac{\pi}{180} x

    Defined for every real ``x``. The gradient is :math:`\pi / 180`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Deg2Rad, Input, Modely, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("radians", inputs=[x], outputs=[Output("y", Deg2Rad(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[180.0, 90.0]])})["y"]).round(4)  # pi, pi/2
    array([[3.1416, 1.5708]], dtype=float32)

    Applications
    ------------
    A steering angle logged in degrees enters the trigonometric layers in
    radians:

    >>> from nnodely import Sin
    >>> steering = Input("steering", tw=1)
    >>> lateral = Sin(Deg2Rad(steering))

    .. versionadded:: 2.0
    """

    _operation = "deg2rad"


class Rad2Deg(Arithmetic):
    r"""Every value of a stream converted from radians to degrees.

    Syntax
    ------
    ``Rad2Deg(x)``
        The angle in degrees of ``x``; the same as ``Rad2Deg()(x)``.
    ``Rad2Deg(name="d")(x)``
        The same, with the layer named.
    ``Rad2Deg([a, b])``
        The angle in degrees of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Rad2Deg`` followed by a number.

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
    Deg2Rad : Degrees to radians, its inverse.
    Atan2 : Angle of a point, in radians.

    Notes
    -----
    .. math:: y = \frac{180}{\pi} x

    Defined for every real ``x``. The gradient is :math:`180 / \pi`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Rad2Deg
    >>> x = Input("x", dim=2)
    >>> model = Modely("degrees", inputs=[x], outputs=[Output("y", Rad2Deg(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[np.pi, np.pi / 2]])})["y"])
    array([[180.,  90.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "rad2deg"


class Sign(Arithmetic):
    r"""Sign of every value of a stream: -1, 0 or 1.

    Syntax
    ------
    ``Sign(x)``
        The sign of ``x``; the same as ``Sign()(x)``.
    ``Sign(name="s")(x)``
        The same, with the layer named.
    ``Sign([a, b])``
        The sign of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Sign`` followed by a number.

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
    Abs : Absolute value.
    Clamp : Bounds the values of a stream.

    Notes
    -----
    .. math:: y = \operatorname{sign} x = \begin{cases} -1 & x < 0 \\ 0 & x = 0 \\ 1 & x > 0 \end{cases}

    The result is -1, 0 or 1, as a float.

    The gradient is 0 everywhere, at the steps too. A weight that reaches
    the output only through ``Sign`` is not trained by it.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Gradient, Input, Modely, Output, Sign
    >>> x = Input("x", dim=3)
    >>> model = Modely("step", inputs=[x], outputs=[Output("y", Sign(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-2.0, 0.0, 3.0]])})["y"])
    array([[-1.,  0.,  1.]], dtype=float32)

    Possible Issues
    ---------------
    The gradient is 0, so nothing before ``Sign`` is trained through it:

    >>> slope = Gradient(wrt=x)(Sign(x))
    >>> model = Modely("slope", inputs=[x], outputs=[Output("d", slope)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-2.0, 0.0, 3.0]])})["d"])
    array([[0., 0., 0.]], dtype=float32)

    .. versionadded:: 2.0
    """

    _operation = "sign"


class Negative(Arithmetic):
    """Every value of a stream with its sign changed.

    Syntax
    ------
    ``Negative(x)``
        The opposite of ``x``; the same as ``Negative()(x)``.
    ``Negative(name="n")(x)``
        The same, with the layer named.
    ``Negative([a, b])``
        The opposite of each stream of the list, as a list.
    ``-x``
        The same, written as an operator.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Negative`` followed by a number.

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
    Abs : Absolute value.
    Sign : Sign.

    Notes
    -----
    .. math:: y = -x

    Defined for every real ``x``. The gradient is :math:`-1`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Negative, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("negative", inputs=[x], outputs=[Output("y", Negative(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-2.0, 3.0]])})["y"])
    array([[ 2., -3.]], dtype=float32)

    Properties & Relations
    ----------------------
    ``-x`` is ``Negative(x)``:

    >>> outputs = [Output("a", Negative(x)), Output("b", -x)]
    >>> result = Modely("same", inputs=[x], outputs=outputs).build()({"x": np.array([[-2.0, 3.0]])})
    >>> keras.ops.convert_to_numpy(result["a"]), keras.ops.convert_to_numpy(result["b"])
    (array([[ 2., -3.]], dtype=float32), array([[ 2., -3.]], dtype=float32))

    .. versionadded:: 2.0
    """

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
    r"""Every value of a stream bounded between ``min`` and ``max``.

    Syntax
    ------
    ``Clamp(min=lo, max=hi)(x)``
        Bounds ``x`` between ``lo`` and ``hi``; the same as ``Clamp(lo, hi)(x)``.
    ``Clamp(min=lo)(x)``
        Bounds ``x`` from below only.
    ``Clamp(max=hi)(x)``
        Bounds ``x`` from above only.
    ``Clamp(lo, hi)([a, b])``
        Bounds each stream of the list, as a list.

    The bounds are fixed numbers. For a bound that is a stream, such as a
    learned :class:`Parameter`, use :class:`Maximum` and :class:`Minimum`.

    Parameters
    ----------
    min : float or None, default None
        The lower bound. ``None`` leaves the values unbounded from below.
    max : float or None, default None
        The upper bound. ``None`` leaves the values unbounded from above. It
        must not be smaller than ``min``.
    name : str, optional
        The name of the layer. The default is ``Clamp`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``. A list of streams gives a list, in its order.

    Raises
    ------
    ValueError
        If ``min`` is greater than ``max`` ("min 1 must not be greater than
        max 0").
    TypeError
        If the layer is called before it is configured, ``Clamp(x)``.

    See Also
    --------
    Maximum : Elementwise maximum of streams, for a bound that is a stream.
    Minimum : Elementwise minimum of streams.
    Sigmoid : A smooth bounded function.

    Notes
    -----
    .. math:: y = \min(\max(x, \text{min}), \text{max})

    The gradient is 1 between the bounds and 0 outside them: a value held at
    a bound does not train what comes before it. At a bound itself the
    backends differ: 1 on TensorFlow and PyTorch, 0.5 on JAX.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Clamp, Input, Maximum, Minimum, Modely, Output, Parameter
    >>> x = Input("x", dim=3)
    >>> model = Modely("clamp", inputs=[x], outputs=[Output("y", Clamp(0.0, 1.0)(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-2.0, 0.5, 3.0]])})["y"])
    array([[0. , 0.5, 1. ]], dtype=float32)

    Scope
    -----
    A list bounds each stream:

    >>> a, b = Input("a", tw=2), Input("b", dim=3)
    >>> [stream.shape for stream in Clamp(0.0, 1.0)([a, b])]
    [Shape(S=(), T=2, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``min`` alone bounds from below, and ``max`` alone from above:

    >>> outputs = [Output("low", Clamp(min=0.0)(x)), Output("high", Clamp(max=1.0)(x))]
    >>> result = Modely("one_bound", inputs=[x], outputs=outputs).build()({"x": np.array([[-2.0, 0.5, 3.0]])})
    >>> keras.ops.convert_to_numpy(result["low"]), keras.ops.convert_to_numpy(result["high"])
    (array([[0. , 0.5, 3. ]], dtype=float32), array([[-2. ,  0.5,  1. ]], dtype=float32))

    ``name`` names the layer:

    >>> Clamp(0.0, 1.0, name="duty")(x).name
    'duty'

    Applications
    ------------
    An actuator saturates: the torque it gives stays within :math:`\pm 2`
    N m, whatever the command:

    >>> command = Input("command", tw=1)
    >>> torque = Clamp(min=-2.0, max=2.0)(command)

    Possible Issues
    ---------------
    The lower bound must not exceed the upper one:

    >>> Clamp(1.0, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: ...min 1.0 must not be greater than max 0.0.

    The layer is configured first, then called on its stream:

    >>> Clamp(x)
    Traceback (most recent call last):
        ...
    TypeError: Clamp has settings, so it is configured first and then called on its streams: Clamp(...)(x), not Clamp(x).

    The bounds are numbers, not streams. For a learned bound, use
    :class:`Maximum` and :class:`Minimum`:

    >>> low, high = Parameter("low", value=[0.0]), Parameter("high", value=[1.0])
    >>> bounded = Minimum([Maximum([x, low]), high])

    .. versionadded:: 2.0
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
    r"""Elementwise maximum of two or more streams.

    Syntax
    ------
    ``Maximum([a, b])``
        The maximum of ``a`` and ``b``, value by value; the same as
        ``Maximum()([a, b])``.
    ``Maximum([a, b, c])``
        The maximum of three or more streams.
    ``Maximum(name="m")([a, b])``
        The same, with the layer named.

    The streams go in one list. They combine as the arithmetic operators
    combine streams, so a bound can be a :class:`Parameter`, learned, where
    :class:`Clamp` takes a fixed number:

    - a stream without a window applies to every sample of the others;
    - a scalar, ``D=(1,)``, applies to every feature of the others.

    :class:`Max` gives the maximum over the features of one stream
    instead.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Maximum`` followed by a number.

    Call arguments
    --------------
    [a, b, ...] : list of Stream
        Two or more streams, whose shapes combine as for ``a + b``.

    Returns
    -------
    Stream
        A stream of the shape of ``a + b``. For example, ``a`` of
        ``Shape(S=(), T=3, D=(2,))`` and a Parameter of
        ``Shape(S=(), T=None, D=(1,))`` give ``Shape(S=(), T=3, D=(2,))``.

    Raises
    ------
    ValueError
        If the list holds fewer than two streams ("Maximum compares two or more
        streams elementwise"), or if their shapes do not combine ("cannot be
        combined").

    See Also
    --------
    Minimum : Elementwise minimum of streams.
    Max : Largest value over the features of one stream.
    Clamp : Bounds a stream between fixed numbers.

    Notes
    -----
    .. math:: y = \max(a, b, \dots)

    The gradient flows to the stream that holds the maximum, and is 0 for
    the others. At a tie the backends share it differently.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Max, Maximum, Modely, Output, Parameter
    >>> a, b = Input("a", dim=2), Input("b", dim=2)
    >>> model = Modely("maximum", inputs=[a, b], outputs=[Output("y", Maximum([a, b]))]).build()
    >>> keras.ops.convert_to_numpy(model({"a": np.array([[1.0, 4.0]]), "b": np.array([[3.0, 2.0]])})["y"])
    array([[3., 4.]], dtype=float32)

    Scope
    -----
    A Parameter without a window applies to every sample of a window:

    >>> bound = Parameter("bound", value=[0.0])
    >>> Maximum([Input("w", dim=2, tw=3), bound]).shape
    Shape(S=(), T=3, D=(2,))

    Three streams:

    >>> Maximum([a, b, bound]).shape
    Shape(S=(), T=None, D=(2,))

    Options
    -------
    ``name`` names the layer, configured first:

    >>> Maximum(name="upper")([a, b]).name
    'upper'

    Applications
    ------------
    A learned lower bound: the speed a model predicts never goes below a
    threshold it learns.

    >>> speed = Input("speed", tw=1)
    >>> idle = Parameter("idle", value=[0.5])
    >>> bounded = Maximum([speed, idle])

    Possible Issues
    ---------------
    The streams go in one list: ``Maximum(a, b)`` reads ``b`` as the name of
    the layer, and fails. A single stream raises. For the maximum over the
    features of one stream, use :class:`Max`:

    >>> Maximum()(a)
    Traceback (most recent call last):
        ...
    ValueError: ...Maximum compares two or more streams elementwise, Maximum([a, b]), got 1...
    >>> Max()(a).shape
    Shape(S=(), T=None, D=(1,))

    .. versionadded:: 2.0
    """

    _operation = "maximum"
    _reduction = "Max"


class Minimum(_Extremum):
    r"""Elementwise minimum of two or more streams.

    Syntax
    ------
    ``Minimum([a, b])``
        The minimum of ``a`` and ``b``, value by value; the same as
        ``Minimum()([a, b])``.
    ``Minimum([a, b, c])``
        The minimum of three or more streams.
    ``Minimum(name="m")([a, b])``
        The same, with the layer named.

    The streams go in one list. They combine as the arithmetic operators
    combine streams, so a bound can be a :class:`Parameter`, learned, where
    :class:`Clamp` takes a fixed number:

    - a stream without a window applies to every sample of the others;
    - a scalar, ``D=(1,)``, applies to every feature of the others.

    :class:`Min` gives the minimum over the features of one stream
    instead.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Minimum`` followed by a number.

    Call arguments
    --------------
    [a, b, ...] : list of Stream
        Two or more streams, whose shapes combine as for ``a + b``.

    Returns
    -------
    Stream
        A stream of the shape of ``a + b``. For example, ``a`` of
        ``Shape(S=(), T=3, D=(2,))`` and a Parameter of
        ``Shape(S=(), T=None, D=(1,))`` give ``Shape(S=(), T=3, D=(2,))``.

    Raises
    ------
    ValueError
        If the list holds fewer than two streams ("Minimum compares two or more
        streams elementwise"), or if their shapes do not combine ("cannot be
        combined").

    See Also
    --------
    Maximum : Elementwise maximum of streams.
    Min : Smallest value over the features of one stream.
    Clamp : Bounds a stream between fixed numbers.

    Notes
    -----
    .. math:: y = \min(a, b, \dots)

    The gradient flows to the stream that holds the minimum, and is 0 for
    the others. At a tie the backends share it differently.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Min, Minimum, Modely, Output, Parameter
    >>> a, b = Input("a", dim=2), Input("b", dim=2)
    >>> model = Modely("minimum", inputs=[a, b], outputs=[Output("y", Minimum([a, b]))]).build()
    >>> keras.ops.convert_to_numpy(model({"a": np.array([[1.0, 4.0]]), "b": np.array([[3.0, 2.0]])})["y"])
    array([[1., 2.]], dtype=float32)

    Scope
    -----
    A Parameter without a window applies to every sample of a window:

    >>> bound = Parameter("bound", value=[0.0])
    >>> Minimum([Input("w", dim=2, tw=3), bound]).shape
    Shape(S=(), T=3, D=(2,))

    Three streams:

    >>> Minimum([a, b, bound]).shape
    Shape(S=(), T=None, D=(2,))

    Options
    -------
    ``name`` names the layer, configured first:

    >>> Minimum(name="lower")([a, b]).name
    'lower'

    Applications
    ------------
    A learned upper bound: the force a model predicts never exceeds a
    limit it learns.

    >>> force = Input("force", tw=1)
    >>> limit = Parameter("limit", value=[10.0])
    >>> bounded = Minimum([force, limit])

    Possible Issues
    ---------------
    The streams go in one list: ``Minimum(a, b)`` reads ``b`` as the name of
    the layer, and fails. A single stream raises. For the minimum over the
    features of one stream, use :class:`Min`:

    >>> Minimum()(a)
    Traceback (most recent call last):
        ...
    ValueError: ...Minimum compares two or more streams elementwise, Minimum([a, b]), got 1...
    >>> Min()(a).shape
    Shape(S=(), T=None, D=(1,))

    .. versionadded:: 2.0
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
    r"""Sum of the features of a stream, the dim axes kept with length 1.

    Syntax
    ------
    ``Sum()(x)``
        The sum over every dim axis of ``x``.
    ``Sum(axis=k)(x)``
        The sum over dim axis ``k`` only.
    ``Sum(name="s")(x)``
        The same, with the layer named.

    The sum is taken at every sample of the window and every step of the
    sequence: the window and the sequence are kept. The summed dim axes are
    kept with length 1, so a vector of ``D=(3,)`` gives a scalar ``D=(1,)``.

    Parameters
    ----------
    axis : int or None, default None
        The dim axis to sum over, counted from 0; a negative one counts from
        the last. ``None`` sums over every dim axis together.
    name : str, optional
        The name of the layer. The default is ``Sum`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape. The call takes it alone or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with the summed dim axes of length 1. For example,
        ``Shape(S=(), T=4, D=(2, 3))`` gives ``D=(1, 1)``, and ``D=(1, 3)``
        with ``axis=0``.

    Raises
    ------
    ValueError
        If ``axis`` is not a dim axis of ``x`` ("axis 2 out of bounds for dim
        rank 2"), or if the layer is given several streams ("reads one stream,
        got 2").
    TypeError
        If the layer is called before it is configured, ``Sum(x)``.

    See Also
    --------
    Mean : Mean of the features.
    Norm : Euclidean length of the features.
    Max : Largest feature.
    Fir : Weighted sum over the window.

    Notes
    -----
    For ``axis=None`` and the features :math:`x_1, \dots, x_n` of one sample:

    .. math:: y = \sum_{i=1}^{n} x_i

    The gradient is 1 with respect to every feature.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Sum
    >>> x = Input("x", dim=3)
    >>> model = Modely("sum", inputs=[x], outputs=[Output("y", Sum()(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 2.0, 3.0]])})["y"])
    array([[6.]], dtype=float32)

    Scope
    -----
    The window is kept: each sample is summed on its own.

    >>> w = Input("w", dim=3, tw=4)
    >>> Sum()(w).shape
    Shape(S=(), T=4, D=(1,))

    A matrix of features gives a matrix of one element:

    >>> m = Input("m", dim=(2, 3))
    >>> Sum()(m).shape
    Shape(S=(), T=None, D=(1, 1))

    Options
    -------
    ``axis`` picks one dim axis. On the matrix ``[[1, 2, 3], [4, 5, 6]]``,
    ``axis=0`` sums the rows and ``axis=1`` (or ``-1``) the columns:

    >>> outputs = [Output("rows", Sum(axis=0)(m)), Output("cols", Sum(axis=-1)(m))]
    >>> model = Modely("axes", inputs=[m], outputs=outputs).build()
    >>> result = model({"m": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])})
    >>> keras.ops.convert_to_numpy(result["rows"])
    array([[[5., 7., 9.]]], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["cols"])
    array([[[ 6.],
            [15.]]], dtype=float32)

    ``name`` names the layer:

    >>> Sum(name="total")(x).name
    'total'

    Applications
    ------------
    The kinetic energy of a point mass of 2 kg, from its velocity vector:

    >>> velocity = Input("velocity", dim=3, tw=1)
    >>> energy = 0.5 * 2.0 * Sum()(velocity * velocity)

    Possible Issues
    ---------------
    ``Sum`` adds the features of every sample, not the samples of the window.
    For a sum over the window, use :class:`Fir` with a kernel of ones:

    >>> from nnodely import Fir, Parameter
    >>> ones = Parameter("ones", value=[[1.0], [1.0], [1.0], [1.0]])
    >>> Fir(kernel=ones, bias=False)(Input("s", tw=4)).shape
    Shape(S=(), T=1, D=(1,))

    The layer is configured first, then called on its stream:

    >>> Sum(x)
    Traceback (most recent call last):
        ...
    TypeError: Sum has settings, so it is configured first and then called on its streams: Sum(...)(x), not Sum(x).

    .. versionadded:: 2.0
    """

    _operation = "sum"


class Mean(Reduction):
    r"""Mean of the features of a stream, the dim axes kept with length 1.

    Syntax
    ------
    ``Mean()(x)``
        The mean over every dim axis of ``x``.
    ``Mean(axis=k)(x)``
        The mean over dim axis ``k`` only.
    ``Mean(name="average")(x)``
        The same, with the layer named.

    The mean is taken at every sample of the window and every step of the
    sequence: the window and the sequence are kept. The reduced dim axes are
    kept with length 1, so a vector of ``D=(3,)`` gives a scalar ``D=(1,)``.

    Parameters
    ----------
    axis : int or None, default None
        The dim axis to reduce, counted from 0; a negative one counts from the
        last. ``None`` reduces every dim axis together.
    name : str, optional
        The name of the layer. The default is ``Mean`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape. The call takes it alone or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with the reduced dim axes of length 1. For example,
        ``Shape(S=(), T=4, D=(2, 3))`` gives ``D=(1, 1)``, and ``D=(1, 3)``
        with ``axis=0``.

    Raises
    ------
    ValueError
        If ``axis`` is not a dim axis of ``x`` ("axis 2 out of bounds for dim
        rank 2"), or if the layer is given several streams ("reads one stream,
        got 2").
    TypeError
        If the layer is called before it is configured, ``Mean(x)``.

    See Also
    --------
    Sum : Sum of the features.
    Norm : Euclidean length of the features.
    Max : Largest feature.

    Notes
    -----
    For ``axis=None`` and the features :math:`x_1, \dots, x_n` of one sample:

    .. math:: y = \frac{1}{n} \sum_{i=1}^{n} x_i

    The gradient is :math:`1 / n` with respect to every feature.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Mean, Modely, Output
    >>> x = Input("x", dim=4)
    >>> model = Modely("mean", inputs=[x], outputs=[Output("y", Mean()(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 2.0, 3.0, 6.0]])})["y"])
    array([[3.]], dtype=float32)

    Scope
    -----
    The window is kept, each sample reduced on its own, and a matrix of
    features gives a matrix of one element:

    >>> Mean()(Input("w", dim=3, tw=4)).shape
    Shape(S=(), T=4, D=(1,))
    >>> Mean()(Input("grid", dim=(2, 3))).shape
    Shape(S=(), T=None, D=(1, 1))

    Options
    -------
    ``axis`` picks one dim axis. On the matrix ``[[1.0, 3.0], [5.0, 7.0]]``, ``axis=0``
    reduces the rows and ``axis=-1`` the columns:

    >>> m = Input("m", dim=(2, 2))
    >>> outputs = [Output("rows", Mean(axis=0)(m)), Output("cols", Mean(axis=-1)(m))]
    >>> model = Modely("axes", inputs=[m], outputs=outputs).build()
    >>> result = model({"m": np.array([[[1.0, 3.0], [5.0, 7.0]]])})
    >>> keras.ops.convert_to_numpy(result["rows"])
    array([[[3., 5.]]], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["cols"])
    array([[[2.],
            [6.]]], dtype=float32)

    ``name`` names the layer:

    >>> Mean(name="average")(x).name
    'average'

    .. versionadded:: 2.0
    """

    _operation = "mean"


class Max(Reduction):
    r"""Largest feature of a stream, the dim axes kept with length 1.

    Syntax
    ------
    ``Max()(x)``
        The largest value over every dim axis of ``x``.
    ``Max(axis=k)(x)``
        The largest value over dim axis ``k`` only.
    ``Max(name="peak")(x)``
        The same, with the layer named.

    The largest value is taken at every sample of the window and every step of the
    sequence: the window and the sequence are kept. The reduced dim axes are
    kept with length 1, so a vector of ``D=(3,)`` gives a scalar ``D=(1,)``.

    Parameters
    ----------
    axis : int or None, default None
        The dim axis to reduce, counted from 0; a negative one counts from the
        last. ``None`` reduces every dim axis together.
    name : str, optional
        The name of the layer. The default is ``Max`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape. The call takes it alone or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with the reduced dim axes of length 1. For example,
        ``Shape(S=(), T=4, D=(2, 3))`` gives ``D=(1, 1)``, and ``D=(1, 3)``
        with ``axis=0``.

    Raises
    ------
    ValueError
        If ``axis`` is not a dim axis of ``x`` ("axis 2 out of bounds for dim
        rank 2"), or if the layer is given several streams ("reads one stream,
        got 2").
    TypeError
        If the layer is called before it is configured, ``Max(x)``.

    See Also
    --------
    Min : Smallest feature.
    Maximum : Elementwise maximum of two or more streams.
    Norm : Euclidean length of the features.

    Notes
    -----
    For ``axis=None`` and the features :math:`x_1, \dots, x_n` of one sample:

    .. math:: y = \max_{i} x_i

    The gradient flows to the largest feature, and is 0 for the others.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Max, Maximum, Modely, Output
    >>> x = Input("x", dim=3)
    >>> model = Modely("max", inputs=[x], outputs=[Output("y", Max()(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 4.0, 2.0]])})["y"])
    array([[4.]], dtype=float32)

    Scope
    -----
    The window is kept, each sample reduced on its own, and a matrix of
    features gives a matrix of one element:

    >>> Max()(Input("w", dim=3, tw=4)).shape
    Shape(S=(), T=4, D=(1,))
    >>> Max()(Input("grid", dim=(2, 3))).shape
    Shape(S=(), T=None, D=(1, 1))

    Options
    -------
    ``axis`` picks one dim axis. On the matrix ``[[1.0, 4.0], [3.0, 2.0]]``, ``axis=0``
    reduces the rows and ``axis=-1`` the columns:

    >>> m = Input("m", dim=(2, 2))
    >>> outputs = [Output("rows", Max(axis=0)(m)), Output("cols", Max(axis=-1)(m))]
    >>> model = Modely("axes", inputs=[m], outputs=outputs).build()
    >>> result = model({"m": np.array([[[1.0, 4.0], [3.0, 2.0]]])})
    >>> keras.ops.convert_to_numpy(result["rows"])
    array([[[3., 4.]]], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["cols"])
    array([[[4.],
            [3.]]], dtype=float32)

    ``name`` names the layer:

    >>> Max(name="peak")(x).name
    'peak'

    Possible Issues
    ---------------
    ``Max`` reads one stream. To compare two streams value by value, use
    :class:`Maximum`:

    >>> a, b = Input("a", dim=3), Input("b", dim=3)
    >>> Max()([a, b])
    Traceback (most recent call last):
        ...
    ValueError: ...Max reads one stream, got 2.
    >>> Maximum([a, b]).shape
    Shape(S=(), T=None, D=(3,))

    .. versionadded:: 2.0
    """

    _operation = "max"


class Min(Reduction):
    r"""Smallest feature of a stream, the dim axes kept with length 1.

    Syntax
    ------
    ``Min()(x)``
        The smallest value over every dim axis of ``x``.
    ``Min(axis=k)(x)``
        The smallest value over dim axis ``k`` only.
    ``Min(name="trough")(x)``
        The same, with the layer named.

    The smallest value is taken at every sample of the window and every step of the
    sequence: the window and the sequence are kept. The reduced dim axes are
    kept with length 1, so a vector of ``D=(3,)`` gives a scalar ``D=(1,)``.

    Parameters
    ----------
    axis : int or None, default None
        The dim axis to reduce, counted from 0; a negative one counts from the
        last. ``None`` reduces every dim axis together.
    name : str, optional
        The name of the layer. The default is ``Min`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape. The call takes it alone or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with the reduced dim axes of length 1. For example,
        ``Shape(S=(), T=4, D=(2, 3))`` gives ``D=(1, 1)``, and ``D=(1, 3)``
        with ``axis=0``.

    Raises
    ------
    ValueError
        If ``axis`` is not a dim axis of ``x`` ("axis 2 out of bounds for dim
        rank 2"), or if the layer is given several streams ("reads one stream,
        got 2").
    TypeError
        If the layer is called before it is configured, ``Min(x)``.

    See Also
    --------
    Max : Largest feature.
    Minimum : Elementwise minimum of two or more streams.
    Mean : Mean of the features.

    Notes
    -----
    For ``axis=None`` and the features :math:`x_1, \dots, x_n` of one sample:

    .. math:: y = \min_{i} x_i

    The gradient flows to the smallest feature, and is 0 for the others.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Min, Minimum, Modely, Output
    >>> x = Input("x", dim=3)
    >>> model = Modely("min", inputs=[x], outputs=[Output("y", Min()(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0, 4.0, 2.0]])})["y"])
    array([[1.]], dtype=float32)

    Scope
    -----
    The window is kept, each sample reduced on its own, and a matrix of
    features gives a matrix of one element:

    >>> Min()(Input("w", dim=3, tw=4)).shape
    Shape(S=(), T=4, D=(1,))
    >>> Min()(Input("grid", dim=(2, 3))).shape
    Shape(S=(), T=None, D=(1, 1))

    Options
    -------
    ``axis`` picks one dim axis. On the matrix ``[[1.0, 4.0], [3.0, 2.0]]``, ``axis=0``
    reduces the rows and ``axis=-1`` the columns:

    >>> m = Input("m", dim=(2, 2))
    >>> outputs = [Output("rows", Min(axis=0)(m)), Output("cols", Min(axis=-1)(m))]
    >>> model = Modely("axes", inputs=[m], outputs=outputs).build()
    >>> result = model({"m": np.array([[[1.0, 4.0], [3.0, 2.0]]])})
    >>> keras.ops.convert_to_numpy(result["rows"])
    array([[[1., 2.]]], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["cols"])
    array([[[1.],
            [2.]]], dtype=float32)

    ``name`` names the layer:

    >>> Min(name="trough")(x).name
    'trough'

    Possible Issues
    ---------------
    ``Min`` reads one stream. To compare two streams value by value, use
    :class:`Minimum`:

    >>> a, b = Input("a", dim=3), Input("b", dim=3)
    >>> Min()([a, b])
    Traceback (most recent call last):
        ...
    ValueError: ...Min reads one stream, got 2.
    >>> Minimum([a, b]).shape
    Shape(S=(), T=None, D=(3,))

    .. versionadded:: 2.0
    """

    _operation = "min"


class Norm(Reduction):
    r"""Euclidean length of the features of a stream, the dim axes kept with length 1.

    Syntax
    ------
    ``Norm()(x)``
        The Euclidean norm over every dim axis of ``x``.
    ``Norm(axis=k)(x)``
        The Euclidean norm over dim axis ``k`` only.
    ``Norm(name="length")(x)``
        The same, with the layer named.

    The Euclidean norm is taken at every sample of the window and every step of the
    sequence: the window and the sequence are kept. The reduced dim axes are
    kept with length 1, so a vector of ``D=(3,)`` gives a scalar ``D=(1,)``.

    Parameters
    ----------
    axis : int or None, default None
        The dim axis to reduce, counted from 0; a negative one counts from the
        last. ``None`` reduces every dim axis together.
    name : str, optional
        The name of the layer. The default is ``Norm`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape. The call takes it alone or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with the reduced dim axes of length 1. For example,
        ``Shape(S=(), T=4, D=(2, 3))`` gives ``D=(1, 1)``, and ``D=(1, 3)``
        with ``axis=0``.

    Raises
    ------
    ValueError
        If ``axis`` is not a dim axis of ``x`` ("axis 2 out of bounds for dim
        rank 2"), or if the layer is given several streams ("reads one stream,
        got 2").
    TypeError
        If the layer is called before it is configured, ``Norm(x)``.

    See Also
    --------
    Sum : Sum of the features.
    Sqrt : Square root.
    Cross : Cross product of 3-vectors.

    Notes
    -----
    For ``axis=None`` and the features :math:`x_1, \dots, x_n` of one sample:

    .. math:: y = \sqrt{\sum_{i=1}^{n} x_i^2}

    The norm is always the Euclidean one: there is no argument for another order.
    The gradient is :math:`x_i / y`. At the zero vector, where the norm has
    none, it is 0, so a training through ``Norm`` gives no NaN there.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Norm, Output
    >>> x = Input("x", dim=2)
    >>> model = Modely("norm", inputs=[x], outputs=[Output("y", Norm()(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[3.0, 4.0]])})["y"])
    array([[5.]], dtype=float32)

    Scope
    -----
    The window is kept, each sample reduced on its own, and a matrix of
    features gives a matrix of one element:

    >>> Norm()(Input("w", dim=3, tw=4)).shape
    Shape(S=(), T=4, D=(1,))
    >>> Norm()(Input("grid", dim=(2, 3))).shape
    Shape(S=(), T=None, D=(1, 1))

    Options
    -------
    ``axis`` picks one dim axis. On the matrix ``[[3.0, 4.0], [0.0, 0.0]]``, ``axis=0``
    reduces the rows and ``axis=-1`` the columns:

    >>> m = Input("m", dim=(2, 2))
    >>> outputs = [Output("rows", Norm(axis=0)(m)), Output("cols", Norm(axis=-1)(m))]
    >>> model = Modely("axes", inputs=[m], outputs=outputs).build()
    >>> result = model({"m": np.array([[[3.0, 4.0], [0.0, 0.0]]])})
    >>> keras.ops.convert_to_numpy(result["rows"])
    array([[[3., 4.]]], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["cols"])
    array([[[5.],
            [0.]]], dtype=float32)

    ``name`` names the layer:

    >>> Norm(name="length")(x).name
    'length'

    Applications
    ------------
    The speed of a vehicle from the two components of its velocity:

    >>> velocity = Input("velocity", dim=2, tw=1)
    >>> speed = Norm()(velocity)

    Possible Issues
    ---------------
    The first argument is the axis, not the order of the norm. ``Norm(2)``
    asks for dim axis 2:

    >>> Norm(2)(x)
    Traceback (most recent call last):
        ...
    ValueError: ...axis 2 out of bounds for dim rank 1.

    For the 1-norm, sum the absolute values: ``Sum()(Abs(x))``.

    .. versionadded:: 2.0
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
    r"""Cross product of two streams of 3-vectors, ``a x b``.

    Syntax
    ------
    ``Cross()([a, b])``
        The cross product ``a x b`` along the last dim axis.
    ``Cross(axis=k)([a, b])``
        The cross product along dim axis ``k``.
    ``Cross(name="c")([a, b])``
        The same, with the layer named.

    The two streams go in one list, and the axis has length 3 in both. They
    combine along the window and the sequence as the arithmetic operators
    combine streams: with ``omega`` a :class:`Parameter` of dim 3,
    ``Cross()([omega, r])`` is ``omega x r`` at every sample of ``r``.

    Parameters
    ----------
    axis : int, default -1
        The dim axis that holds the vectors, counted from 0; a negative one
        counts from the last. It has length 3 in both streams, and is the
        same axis of the result for both.
    name : str, optional
        The name of the layer. The default is ``Cross`` followed by a number.

    Call arguments
    --------------
    [a, b] : list of two Stream
        The two factors, in order. Their shapes combine as for ``a + b``.

    Returns
    -------
    Stream
        A stream of the shape of ``a + b``. For example, ``a`` of
        ``Shape(S=(), T=None, D=(3,))`` and ``b`` of
        ``Shape(S=(), T=2, D=(3,))`` give ``Shape(S=(), T=2, D=(3,))``.

    Raises
    ------
    ValueError
        If the list does not hold two streams ("Cross reads two streams, [a,
        b], got 1"). If the axis does not have length 3 ("Cross multiplies
        3-vectors"). If the axis of ``a`` and that of ``b`` are different axes
        of the result ("are different axes of the product"), or if the shapes
        do not combine.
    TypeError
        If the layer is called before it is configured, ``Cross([a, b])``.

    See Also
    --------
    Norm : Euclidean length of a vector.
    Sum : Sum of the features, for a dot product ``Sum()(a * b)``.

    Notes
    -----
    .. math::

        a \times b = (a_2 b_3 - a_3 b_2,\; a_3 b_1 - a_1 b_3,\; a_1 b_2 - a_2 b_1)

    The product is anticommutative, :math:`b \times a = -(a \times b)`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Cross, Input, Modely, Output, Parameter
    >>> a, b = Input("a", dim=3), Input("b", dim=3)
    >>> model = Modely("cross", inputs=[a, b], outputs=[Output("y", Cross()([a, b]))]).build()
    >>> result = model({"a": np.array([[1.0, 0.0, 0.0]]), "b": np.array([[0.0, 1.0, 0.0]])})
    >>> keras.ops.convert_to_numpy(result["y"])  # x times y is z
    array([[0., 0., 1.]], dtype=float32)

    Scope
    -----
    A Parameter without a window applies to every sample of a window:

    >>> omega = Parameter("omega", value=[0.0, 0.0, 2.0])
    >>> r = Input("r", dim=3, tw=2)
    >>> model = Modely("spin", inputs=[r], outputs=[Output("v", Cross()([omega, r]))]).build()
    >>> keras.ops.convert_to_numpy(model({"r": np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])})["v"])
    array([[[ 0.,  2.,  0.],
            [-2.,  0.,  0.]]], dtype=float32)

    Options
    -------
    ``axis`` picks the dim axis of the vectors. With ``D=(3, 2)``, the two
    columns are two 3-vectors:

    >>> c, d = Input("c", dim=(3, 2)), Input("d", dim=(3, 2))
    >>> Cross(axis=0)([c, d]).shape
    Shape(S=(), T=None, D=(3, 2))

    ``name`` names the layer:

    >>> Cross(name="torque")([a, b]).name
    'torque'

    Applications
    ------------
    The velocity of a point of a rigid body, :math:`v = \omega \times r`, as
    in the Scope example. The torque of a force applied at an arm:

    >>> arm = Input("arm", dim=3, tw=1)
    >>> force = Input("force", dim=3, tw=1)
    >>> torque = Cross()([arm, force])

    Possible Issues
    ---------------
    The axis must have length 3 in both streams:

    >>> Cross()([Input("planar", dim=2), b])
    Traceback (most recent call last):
        ...
    ValueError: ...Cross multiplies 3-vectors, and dim axis -1 of 'planar', Shape(S=(), T=None, D=(2,)), has length 2.

    The layer is configured first, then called on the list:

    >>> Cross([a, b])
    Traceback (most recent call last):
        ...
    TypeError: Cross has settings, so it is configured first and then called on its streams: Cross(...)(x), not Cross(x).

    .. versionadded:: 2.0
    """

    def __init__(self, axis: int = -1, name=None):
        self._axis = int(axis)
        super().__init__(name=name, axis=self._axis)

    def build_layer(self):
        result = self.shape
        axis = self._result_axis(self._preds[0], result) - result.dim_rank  # type: ignore[arg-type]
        return CrossImpl(axis=axis, **self._alignment(), name=self.name)

    def _result_axis(self, node, result) -> int:
        """The dim axis of ``result`` that dim axis ``axis`` of ``node`` lines
        up with: dims are lined up from the last axis."""
        shape = node.shape
        axis = shape.resolve_dim(self._axis, self.name)
        return axis + result.dim_rank - shape.dim_rank

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
        result = super().output_shape(*inputs)
        if len({self._result_axis(node, result) for node in inputs}) > 1:
            raise ValueError(
                f"{self.name}: dim axis {self._axis} of {inputs[0].name!r}, "
                f"{inputs[0].shape}, and of {inputs[1].name!r}, "
                f"{inputs[1].shape}, are different axes of the product, {result}."
            )
        return result
