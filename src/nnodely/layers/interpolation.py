"""Elementwise interpolation layers."""

from __future__ import annotations

import keras
import numpy as np

from nnodely.core.layer import Layer


_AVAILABLE_MODES = ("linear", "polynomial")


def _prepare_points(x_points, y_points, mode):
    x_values = np.asarray(x_points, dtype=np.float64)
    y_values = np.asarray(y_points, dtype=np.float64)
    mode = str(mode).lower()

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("Interpolation points must be one-dimensional.")
    if len(x_values) != len(y_values):
        raise ValueError("x_points and y_points must have the same length.")
    if len(x_values) < 2:
        raise ValueError("Interpolation requires at least two points.")
    if mode not in _AVAILABLE_MODES:
        raise ValueError(
            f"Interpolation mode must be one of {_AVAILABLE_MODES}, got {mode!r}."
        )
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("Interpolation points must contain only finite values.")

    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    if np.any(np.diff(x_values) == 0):
        raise ValueError("x_points must contain unique values.")

    return x_values, y_values, mode


@keras.saving.register_keras_serializable(package="nnodely")
class InterpolationImpl(keras.layers.Layer):
    """Serializable elementwise linear or polynomial interpolation."""

    def __init__(self, x_points, y_points, mode="linear", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        x_values, y_values, mode = _prepare_points(x_points, y_points, mode)
        self.x_points = x_values.tolist()
        self.y_points = y_values.tolist()
        self.mode = mode
        self.coefficients = (
            np.linalg.solve(
                np.vander(x_values, N=len(x_values), increasing=True),
                y_values,
            ).tolist()
            if mode == "polynomial"
            else None
        )

    def call(self, inputs):
        dtype = inputs.dtype
        lower = keras.ops.cast(self.x_points[0], dtype)
        upper = keras.ops.cast(self.x_points[-1], dtype)
        values = keras.ops.clip(inputs, lower, upper)

        if self.mode == "polynomial":
            coefficients = self.coefficients
            if coefficients is None:
                raise RuntimeError("Polynomial coefficients were not initialized.")
            result = keras.ops.zeros_like(values) + keras.ops.cast(
                coefficients[-1], dtype
            )
            for coefficient in reversed(coefficients[:-1]):
                result = result * values + keras.ops.cast(coefficient, dtype)
            return result

        result = keras.ops.zeros_like(values)
        for index in range(len(self.x_points) - 1):
            x_left = keras.ops.cast(self.x_points[index], dtype)
            x_right = keras.ops.cast(self.x_points[index + 1], dtype)
            y_left = keras.ops.cast(self.y_points[index], dtype)
            y_right = keras.ops.cast(self.y_points[index + 1], dtype)
            interpolated = y_left + (y_right - y_left) * (values - x_left) / (
                x_right - x_left
            )
            right_condition = (
                values <= x_right
                if index == len(self.x_points) - 2
                else values < x_right
            )
            interval = keras.ops.logical_and(values >= x_left, right_condition)
            result = keras.ops.where(interval, interpolated, result)
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "x_points": self.x_points,
                "y_points": self.y_points,
                "mode": self.mode,
            }
        )
        return config


class Interpolation(Layer):
    r"""Lookup table applied to every value of a stream: the curve through
    the points ``(x_points, y_points)``, read at each value.

    Syntax
    ------
    ``Interpolation(x_points, y_points)(x)``
        Reads the piecewise-linear curve through the points at every value
        of ``x``.
    ``Interpolation(x_points, y_points, mode="polynomial")(x)``
        Reads the polynomial through all the points instead.

    The table is fixed: its points are numbers, not trained. A value
    outside the points reads the curve at the nearest end point, so the
    output is constant beyond either end.

    Parameters
    ----------
    x_points : list of float
        The abscissas of the points. They are finite and all different, at
        least two. Their order does not matter: the points are sorted by
        ``x_points``, each keeping its ``y_points`` value.
    y_points : list of float
        The values at ``x_points``, as many, finite.
    mode : str, default "linear"
        How the curve joins the points, in upper or lower case:

        - ``"linear"``: a straight segment between each two neighboring
          points.
        - ``"polynomial"``: the one polynomial of degree ``n - 1`` through
          the ``n`` points.
    name : str, optional
        The name of the layer. The default is ``Interpolation`` followed by
        a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape, ``Shape(S, T, D)``, with or without a window.
        The call takes it alone, ``Interpolation(...)(x)``, or in a list,
        ``Interpolation(...)([x])``.

    Returns
    -------
    Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=2, D=(3,))`` for a
        window of 2 samples of 3 features.

    Raises
    ------
    ValueError
        When the layer is configured: if ``x_points`` and ``y_points``
        differ in length ("must have the same length"); if there are fewer
        than two points ("requires at least two points"); if ``mode`` is
        not one of the two ("mode must be one of"); if a point is not
        finite ("must contain only finite values"); if two ``x_points``
        are equal ("must contain unique values"); if the points are not
        flat lists ("must be one-dimensional"). When the layer is called on
        several streams ("automatic output shape inference failed").

    See Also
    --------
    Fuzzify : Memberships of a signal to sets placed on centers.
    Clamp : Values held between two bounds.
    LocalModel : Local models blended by operating region.

    Notes
    -----
    With the points sorted, :math:`x_0 < x_1 < \dots < x_{n-1}`, every
    value is first held in the range of the points:

    .. math:: \tilde x = \min(\max(x, x_0), x_{n-1})

    With ``mode="linear"``, for :math:`x_i \le \tilde x \le x_{i+1}`:

    .. math:: y = y_i + (y_{i+1} - y_i) \, \frac{\tilde x - x_i}{x_{i+1} - x_i}

    With ``mode="polynomial"``, :math:`y = \sum_{k=0}^{n-1} c_k \tilde x^k`,
    with the coefficients :math:`c_k` such that the polynomial passes
    through every point.

    The gradient with respect to ``x`` is the slope of the curve inside the
    points, and 0 outside. A polynomial through many points can swing far
    between them: use ``"linear"`` for more than a few points.

    Basic Examples
    --------------
    A table of three points, read at values inside, at and beyond them:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Interpolation, Modely, Output
    >>> x = Input("x", dim=6)
    >>> table = Interpolation(x_points=[0.0, 1.0, 2.0], y_points=[0.0, 2.0, 3.0])
    >>> model = Modely("table", inputs=[x], outputs=[Output("y", table(x))]).build()
    >>> data = {"x": np.array([[-1.0, 0.0, 0.5, 1.0, 1.5, 3.0]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[0. , 0. , 1. , 2. , 2.5, 3. ]], dtype=float32)

    Scope
    -----
    Every value is read in the table, whatever the shape:

    >>> Interpolation([0.0, 1.0], [0.0, 2.0])(Input("v", dim=3, tw=2)).shape
    Shape(S=(), T=2, D=(3,))
    >>> Interpolation([0.0, 1.0], [0.0, 2.0])(Input("s", dim=3, seq=-1, tw=2)).shape
    Shape(S=(dyn,), T=2, D=(3,))

    Options
    -------
    ``x_points`` and ``y_points`` may come in any order of ``x_points``:
    the table above, given in another order, is the same.

    >>> shuffled = Interpolation(x_points=[2.0, 0.0, 1.0], y_points=[3.0, 0.0, 2.0])
    >>> model = Modely("shuffled", inputs=[x], outputs=[Output("y", shuffled(x))]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[0. , 0. , 1. , 2. , 2.5, 3. ]], dtype=float32)

    ``mode="polynomial"`` reads the polynomial through the points, here
    :math:`y = x^2`, still held at the ends:

    >>> square = Interpolation([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], mode="polynomial")
    >>> model = Modely("square", inputs=[x], outputs=[Output("y", square(x))]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[0.  , 0.  , 0.25, 1.  , 2.25, 4.  ]], dtype=float32)

    ``name`` names the layer:

    >>> Interpolation([0.0, 1.0], [0.0, 2.0], name="lookup")(x).name
    'lookup'

    Applications
    ------------
    The rolling resistance coefficient of a tyre, measured at three speeds
    in m/s:

    >>> velocity = Input("velocity", tw=1)
    >>> rolling = Interpolation(x_points=[0.0, 10.0, 30.0], y_points=[0.010, 0.012, 0.020])
    >>> resistance = 1500.0 * 9.81 * rolling(velocity)

    Possible Issues
    ---------------
    The points are given with the layer, before it is called on its
    stream:

    >>> Interpolation(x)
    Traceback (most recent call last):
        ...
    TypeError: ...missing 1 required positional argument: 'y_points'

    Two equal abscissas leave the curve undefined:

    >>> Interpolation([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    Traceback (most recent call last):
        ...
    ValueError: x_points must contain unique values.

    The table reads one stream. Apply it to each stream:

    >>> a, b = Input("a", tw=1), Input("b", tw=1)
    >>> Interpolation([0.0, 1.0], [0.0, 2.0])([a, b])
    Traceback (most recent call last):
        ...
    ValueError: ...automatic output shape inference failed...
    >>> table = Interpolation([0.0, 1.0], [0.0, 2.0])
    >>> ya, yb = table(a), table(b)

    .. versionadded:: 2.0
    """

    def __init__(self, x_points, y_points, mode="linear", name=None):
        x_values, y_values, mode = _prepare_points(x_points, y_points, mode)
        self._x_points = x_values.tolist()
        self._y_points = y_values.tolist()
        self._mode = mode
        super().__init__(
            name=name,
            x_points=self._x_points,
            y_points=self._y_points,
            mode=self._mode,
        )

    def build_layer(self):
        return InterpolationImpl(
            x_points=self._x_points,
            y_points=self._y_points,
            mode=self._mode,
            name=self.name,
        )
