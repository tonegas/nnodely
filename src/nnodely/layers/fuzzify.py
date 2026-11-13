"""Fuzzify, and the membership functions it places on its centers."""

from __future__ import annotations

import inspect
from typing import Any, cast

import keras
import numpy as np

from nnodely.core.dag import next_name
from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank
from nnodely.core.stream import Stream, class_path, registered_class
from nnodely.layers.function import KerasFunction, _is_keras_layer
from nnodely.layers.input import Input
from nnodely.layers.parameter import _Value

_EPS = 1e-7


# The membership formulas, on the distance from the center and on widths that
# are numbers or tensors. A width is held above zero: one trained to zero or
# below collapses its set rather than dividing by zero.
def _floor(width):
    return keras.ops.maximum(width, _EPS)


def _triangle(x, left, right):
    rise = (x + left) / _floor(left)
    fall = (right - x) / _floor(right)
    return keras.ops.maximum(keras.ops.minimum(rise, fall), 0.0)


def _box(x, left, right):
    # Half-open, so that boxes side by side cover their common edge once.
    inside = keras.ops.logical_and(x >= -_floor(left), x < _floor(right))
    return keras.ops.cast(inside, x.dtype)


def _bell(x, sigma, _):
    scaled = x / _floor(sigma)
    return keras.ops.exp(-0.5 * scaled * scaled)


_FORMULAS = {"triangle": _triangle, "box": _box, "bell": _bell}


@keras.saving.register_keras_serializable(package="nnodely")
class BumpImpl(keras.layers.Layer):
    """``formula`` of the input, on ``width``, the left and the right
    width; or, without it, on the second input, a tensor whose first and
    last features are the left and the right width."""

    def __init__(self, formula, width=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.formula = formula
        self.width = None if width is None else [float(value) for value in width]

    def call(self, inputs):
        if self.width is None:
            x, width = inputs
            rank = len(x.shape)
            left = expand_to_rank(width[..., :1], rank)
            right = expand_to_rank(width[..., -1:], rank)
        else:
            x, (left, right) = inputs, self.width
        return _FORMULAS[self.formula](x, left, right)

    def get_config(self):
        config = super().get_config()
        config.update({"formula": self.formula, "width": self.width})
        return config


class _Bump(Layer):
    """An elementwise function of ``x``, 1 at 0 and fading away from it,
    shaped by a width: a number, a ``(left, right)`` pair where the class
    allows one, or a :class:`Parameter` - trained with the model, a
    predecessor of the layer read with ``x``."""

    _formula = "triangle"
    _label = "radius"
    _pair = True

    def __init__(self, width, name=None):
        self._width = _width(width, type(self).__name__, self._label, self._pair)
        # Saved as given: one number for both sides, the pair, or the stream.
        if isinstance(self._width, Stream):
            saved = self._width
        else:
            left, right = self._width
            saved = left if left == right else [left, right]
        super().__init__(name=name, **{self._label: saved})

    def output_shape(self, *inputs):
        # Elementwise: every axis kept, a dynamic sequence axis included.
        return inputs[0].shape

    def __call__(self, inputs) -> Any:
        streams = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        # Called on several streams, it is applied to each of them.
        if len(streams) > 1:
            return [self([stream]) for stream in streams]
        # Read with its stream: alone, it would be taken for the stream.
        if streams and isinstance(self._width, Stream):
            streams.append(self._width)
        return super().__call__(streams)

    def build_layer(self):
        width = None if isinstance(self._width, Stream) else self._width
        return BumpImpl(formula=self._formula, width=width, name=self.name)

    def get_config(self):
        # A width Parameter is saved with the graph, as the predecessor it is.
        config = super().get_config()
        if isinstance(self._width, Stream):
            config[self._label] = None
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        preds = list(preds or [])
        if config.get(cls._label, 1.0) is None:
            config = {**config, cls._label: preds.pop(1)}
        return super().from_config(config, preds=preds)


class Triangular(_Bump):
    r"""Triangular membership of every value of a stream: 1 at 0, falling linearly to 0 at ``radius``.

    Syntax
    ------
    ``Triangular(radius=r)(x)``
        A triangle centered at 0, reaching 0 at ``-r`` and at ``r``.
    ``Triangular(radius=(left, right))(x)``
        A triangle leaning to one side, reaching 0 at ``-left`` and at
        ``right``.
    ``Triangular(radius=r)(x - c)``
        The same triangle centered at ``c``: shift the stream to move it.
    ``Triangular(radius=p)(x)``
        A width held by the :class:`Parameter` ``p``, trained with the model.
    ``Triangular(radius=r)([a, b])``
        The membership of each stream of the list, as a list.

    The function acts on every value alone, so the stream keeps its shape.
    Given to :class:`Fuzzify` as a class, a triangle is placed on each center
    and reaches the centers next to it. Given as an instance, the same
    triangle is placed on every center.

    Parameters
    ----------
    radius : float, (float, float) or Parameter, default 1.0
        The half-widths of the triangle. It is one of:

        - a number ``>= 0``, for both sides;
        - a ``(left, right)`` pair of numbers ``>= 0``;
        - a :class:`Parameter` of dim ``(1,)`` or ``(2,)`` without a window,
          trained, frozen and saved with the model.
    name : str, optional
        The name of the layer. The default is ``Triangular`` followed by a
        number.

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
        If ``radius`` is negative ("radius must not be negative"), is a
        sequence of other than two numbers ("radius is a number or a (left,
        right) pair"), or is a stream other than a Parameter of dim ``(1,)``
        or ``(2,)`` without a window ("a radius stream is a Parameter of dim
        (1,) or (2,), without a window").
    TypeError
        If the layer is called before it is configured, ``Triangular(x)``.

    See Also
    --------
    Rectangular : Rectangular membership.
    Gaussian : Gaussian membership.
    Fuzzify : Memberships to sets placed on centers.

    Notes
    -----
    With the left and right half-widths :math:`l` and :math:`r`:

    .. math:: y = \max\left(0, \min\left(1 + \frac{x}{l}, 1 - \frac{x}{r}\right)\right)

    A half-width below :math:`10^{-7}` is held at :math:`10^{-7}`. A radius
    of 0, or a Parameter trained to 0 or below, collapses the triangle to 0
    everywhere, without NaN. The gradient with respect to ``x`` is
    :math:`1/l` on the rising side, :math:`-1/r` on the falling side and 0
    outside. A width Parameter receives a gradient too.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Fuzzify, Input, Modely, Output, Parameter, Triangular
    >>> x = Input("x", tw=5)
    >>> y = Triangular()(x)
    >>> model = Modely("triangle", inputs=[x], outputs=[Output("y", y)]).build()
    >>> samples = {"x": np.array([[[-1.0], [-0.5], [0.0], [0.5], [1.0]]])}
    >>> keras.ops.convert_to_numpy(model(samples)["y"]).ravel()
    array([0. , 0.5, 1. , 0.5, 0. ], dtype=float32)

    Scope
    -----
    Every value is mapped alone, whatever the shape:

    >>> Triangular()(Input("v", dim=3, seq=2, tw=4)).shape
    Shape(S=(2,), T=4, D=(3,))

    A list gives the membership of each stream:

    >>> a, b = Input("a", tw=1), Input("b", tw=1)
    >>> [stream.shape for stream in Triangular()([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=1, D=(1,))]

    Options
    -------
    ``radius`` as a number, a pair, or a Parameter. A shifted stream moves the
    center:

    >>> outputs = [
    ...     Output("half", Triangular(radius=0.5)(x)),
    ...     Output("leaning", Triangular(radius=(1.0, 0.5))(x)),
    ...     Output("shifted", Triangular(radius=0.5)(x - 0.5)),
    ...     Output("trained", Triangular(radius=Parameter("r", value=[0.5]))(x)),
    ... ]
    >>> model = Modely("radii", inputs=[x], outputs=outputs).build()
    >>> result = model(samples)
    >>> for key in ["half", "leaning", "shifted", "trained"]:
    ...     print(key, keras.ops.convert_to_numpy(result[key]).ravel())
    half [0. 0. 1. 0. 0.]
    leaning [0.  0.5 1.  0.  0. ]
    shifted [0. 0. 0. 1. 0.]
    trained [0. 0. 1. 0. 0.]
    >>> [p.name for p in model.parameters]
    ['r']

    ``name`` names the layer, configured first:

    >>> Triangular(name="tri")(x).name
    'tri'

    Applications
    ------------
    How comfortable a room is, fully at 21 degrees and not at all 3 degrees
    away:

    >>> temperature = Input("temperature", tw=1)
    >>> comfort = Triangular(radius=3.0)(temperature - 21.0)

    Properties & Relations
    ----------------------
    An instance given to :class:`Fuzzify` is the same triangle shifted to each
    center:

    >>> placed = Fuzzify([0.5], function=Triangular(radius=0.5))(x)
    >>> shifted = Triangular(radius=0.5)(x - 0.5)
    >>> model = Modely("same", inputs=[x], outputs=[Output("p", placed), Output("s", shifted)])
    >>> result = model.build()(samples)
    >>> np.array_equal(keras.ops.convert_to_numpy(result["p"]), keras.ops.convert_to_numpy(result["s"]))
    True

    Possible Issues
    ---------------
    The layer is configured first, then called on its stream:

    >>> Triangular(x)
    Traceback (most recent call last):
        ...
    TypeError: Triangular has settings, so it is configured first and then called on its streams: Triangular(...)(x), not Triangular(x).

    A width Parameter has no window. Declare it without ``tw``:

    >>> Triangular(radius=Parameter("w", tw=1, value=[[0.5]]))
    Traceback (most recent call last):
        ...
    ValueError: Triangular: a radius stream is a Parameter of dim (1,) or (2,), without a window; ...
    >>> y = Triangular(radius=Parameter("w", value=[0.5]))(x)

    .. versionadded:: 2.0
    """

    def __init__(self, radius: float | tuple[float, float] | Stream = 1.0, name=None):
        super().__init__(radius, name=name)


class Rectangular(_Bump):
    r"""Rectangular membership of every value of a stream: 1 on ``[-left, right)``, 0 elsewhere.

    Syntax
    ------
    ``Rectangular(radius=r)(x)``
        1 on ``[-r, r)``, 0 elsewhere.
    ``Rectangular(radius=(left, right))(x)``
        1 on ``[-left, right)``, 0 elsewhere.
    ``Rectangular(radius=r)(x - c)``
        The same rectangle centered at ``c``: shift the stream to move it.
    ``Rectangular(radius=p)(x)``
        A width held by the :class:`Parameter` ``p``.
    ``Rectangular(radius=r)([a, b])``
        The membership of each stream of the list, as a list.

    The function acts on every value alone, so the stream keeps its shape.
    The interval is half-open: rectangles side by side hold their common edge
    once. Given to :class:`Fuzzify` as a class, a rectangle is placed on each
    center and reaches halfway to the centers next to it. Between the
    centers, exactly one rectangle then holds every value.

    Parameters
    ----------
    radius : float, (float, float) or Parameter, default 1.0
        The half-widths of the rectangle. It is one of:

        - a number ``>= 0``, for both sides;
        - a ``(left, right)`` pair of numbers ``>= 0``;
        - a :class:`Parameter` of dim ``(1,)`` or ``(2,)`` without a window.
    name : str, optional
        The name of the layer. The default is ``Rectangular`` followed by a
        number.

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
        If ``radius`` is negative ("radius must not be negative"), is a
        sequence of other than two numbers ("radius is a number or a (left,
        right) pair"), or is a stream other than a Parameter of dim ``(1,)``
        or ``(2,)`` without a window ("a radius stream is a Parameter").
    TypeError
        If the layer is called before it is configured, ``Rectangular(x)``.

    See Also
    --------
    Triangular : Triangular membership.
    Gaussian : Gaussian membership.
    Fuzzify : Memberships to sets placed on centers.

    Notes
    -----
    With the left and right half-widths :math:`l` and :math:`r`:

    .. math:: y = \begin{cases} 1 & -l \le x < r \\ 0 & \text{otherwise} \end{cases}

    A half-width below :math:`10^{-7}` is held at :math:`10^{-7}`. A step has
    no gradient, with respect to ``x`` or to the widths. A width Parameter
    read by rectangles alone does not train. One read by another layer too
    trains there, and the rectangles follow it.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Parameter, Rectangular
    >>> x = Input("x", tw=5)
    >>> y = Rectangular(radius=0.5)(x)
    >>> model = Modely("box", inputs=[x], outputs=[Output("y", y)]).build()
    >>> samples = {"x": np.array([[[-1.0], [-0.5], [0.0], [0.5], [1.0]]])}
    >>> keras.ops.convert_to_numpy(model(samples)["y"]).ravel()  # 1 on [-0.5, 0.5)
    array([0., 1., 1., 0., 0.], dtype=float32)

    Scope
    -----
    Every value is mapped alone, whatever the shape:

    >>> Rectangular()(Input("v", dim=3, seq=2, tw=4)).shape
    Shape(S=(2,), T=4, D=(3,))

    Options
    -------
    ``radius`` as a number, a pair, or a Parameter. A shifted stream moves the
    center:

    >>> outputs = [
    ...     Output("unit", Rectangular()(x)),
    ...     Output("leaning", Rectangular(radius=(1.0, 0.5))(x)),
    ...     Output("shifted", Rectangular(radius=0.5)(x - 0.5)),
    ...     Output("param", Rectangular(radius=Parameter("r", value=[0.5]))(x)),
    ... ]
    >>> model = Modely("radii", inputs=[x], outputs=outputs).build()
    >>> result = model(samples)
    >>> for key in ["unit", "leaning", "shifted", "param"]:
    ...     print(key, keras.ops.convert_to_numpy(result[key]).ravel())
    unit [1. 1. 1. 1. 0.]
    leaning [1. 1. 1. 0. 0.]
    shifted [0. 0. 1. 1. 0.]
    param [0. 1. 1. 0. 0.]

    ``name`` names the layer, configured first:

    >>> Rectangular(name="box")(x).name
    'box'

    Applications
    ------------
    A switch that is on while a valve is between 20 % and 80 % open:

    >>> opening = Input("opening", tw=1)
    >>> on = Rectangular(radius=0.3)(opening - 0.5)

    Possible Issues
    ---------------
    A width read by rectangles alone receives no gradient, and keeps its
    value in training. Use :class:`Triangular` or :class:`Gaussian` for a
    width that trains.

    The layer is configured first, then called on its stream:

    >>> Rectangular(x)
    Traceback (most recent call last):
        ...
    TypeError: Rectangular has settings, so it is configured first and then called on its streams: Rectangular(...)(x), not Rectangular(x).

    .. versionadded:: 2.0
    """

    _formula = "box"

    def __init__(self, radius: float | tuple[float, float] | Stream = 1.0, name=None):
        super().__init__(radius, name=name)


class Gaussian(_Bump):
    r"""Gaussian membership of every value of a stream: 1 at 0, of standard deviation ``sigma``.

    Syntax
    ------
    ``Gaussian(sigma=s)(x)``
        A bell centered at 0, of standard deviation ``s``.
    ``Gaussian(sigma=s)(x - c)``
        The same bell centered at ``c``: shift the stream to move it.
    ``Gaussian(sigma=p)(x)``
        A width held by the :class:`Parameter` ``p``, trained with the model.
    ``Gaussian(sigma=s)([a, b])``
        The membership of each stream of the list, as a list.

    The function acts on every value alone, so the stream keeps its shape.
    Given to :class:`Fuzzify` as a class, a bell is placed on each center.
    Its ``sigma`` is then a sixth of the span between the centers next to it,
    so three sigmas reach each of them.

    Parameters
    ----------
    sigma : float or Parameter, default 1.0
        The standard deviation of the bell. It is one of:

        - a number ``> 0``;
        - a :class:`Parameter` of dim ``(1,)`` without a window, trained,
          frozen and saved with the model.
    name : str, optional
        The name of the layer. The default is ``Gaussian`` followed by a
        number.

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
        If ``sigma`` is 0 or negative ("sigma must be positive"), is not a
        number ("sigma is a number or a Parameter"), or is a stream other than
        a Parameter of dim ``(1,)`` without a window ("a sigma stream is a
        Parameter of dim (1,), without a window").
    TypeError
        If the layer is called before it is configured, ``Gaussian(x)``.

    See Also
    --------
    Triangular : Triangular membership.
    Rectangular : Rectangular membership.
    Fuzzify : Memberships to sets placed on centers.
    Exp : Exponential.

    Notes
    -----
    .. math:: y = \exp\left(-\frac{x^2}{2 \sigma^2}\right)

    Values are in :math:`(0, 1]`. A ``sigma`` Parameter trained below
    :math:`10^{-7}` is held at :math:`10^{-7}`: the bell collapses to 1 at 0
    and 0 elsewhere, without NaN. The gradient with respect to ``x`` is
    :math:`-x y / \sigma^2`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Parameter, Gaussian
    >>> x = Input("x", tw=5)
    >>> y = Gaussian(sigma=0.5)(x)
    >>> model = Modely("bell", inputs=[x], outputs=[Output("y", y)]).build()
    >>> samples = {"x": np.array([[[-1.0], [-0.5], [0.0], [0.5], [1.0]]])}
    >>> np.round(keras.ops.convert_to_numpy(model(samples)["y"]).ravel(), 4)
    array([0.1353, 0.6065, 1.    , 0.6065, 0.1353], dtype=float32)

    Scope
    -----
    Every value is mapped alone, whatever the shape:

    >>> Gaussian()(Input("v", dim=3, seq=2, tw=4)).shape
    Shape(S=(2,), T=4, D=(3,))

    Options
    -------
    ``sigma`` as a Parameter, here with the value of the Basic Example, and a
    shifted stream:

    >>> outputs = [
    ...     Output("trained", Gaussian(sigma=Parameter("s", value=[0.5]))(x)),
    ...     Output("shifted", Gaussian(sigma=0.5)(x - 0.5)),
    ... ]
    >>> model = Modely("sigmas", inputs=[x], outputs=outputs).build()
    >>> result = model(samples)
    >>> for key in ["trained", "shifted"]:
    ...     print(key, np.round(keras.ops.convert_to_numpy(result[key]).ravel(), 4))
    trained [0.1353 0.6065 1.     0.6065 0.1353]
    shifted [0.0111 0.1353 0.6065 1.     0.6065]

    ``name`` names the layer, configured first:

    >>> Gaussian(name="bell")(x).name
    'bell'

    Applications
    ------------
    The weight of a speed near the cruise speed of 30 m/s:

    >>> speed = Input("speed", tw=1)
    >>> near_cruise = Gaussian(sigma=5.0)(speed - 30.0)

    Possible Issues
    ---------------
    A Gaussian has one width, not a pair:

    >>> Gaussian(sigma=(1.0, 2.0))
    Traceback (most recent call last):
        ...
    ValueError: Gaussian: sigma is a number or a Parameter, got (1.0, 2.0).

    .. versionadded:: 2.0
    """

    _formula = "bell"
    _label = "sigma"
    _pair = False

    def __init__(self, sigma: float | Stream = 1.0, name=None):
        super().__init__(sigma, name=name)


def _width(width, owner: str, label: str, pair: bool) -> list[float] | Stream:
    """A width checked: the left and right widths of a number, or of a
    ``(left, right)`` pair if ``pair``; or a Parameter of dim ``(1,)``, or
    ``(2,)`` if ``pair``, without a window."""
    dims = ((1,), (2,)) if pair else ((1,),)
    if isinstance(width, Stream):
        if (
            not width._is_value
            or tuple(width.shape.dim) not in dims
            or width.shape.tw is not None
        ):
            raise ValueError(
                f"{owner}: a {label} stream is a Parameter of dim "
                f"{' or '.join(map(str, dims))}, without a window; got "
                f"{width.name!r}, {width.shape}."
            )
        return width
    if isinstance(width, (int, float)):
        values = [float(width)] * 2
    elif pair:
        values = [float(value) for value in width]
        if len(values) != 2:
            raise ValueError(
                f"{owner}: radius is a number or a (left, right) pair, got {width}."
            )
    else:
        raise ValueError(f"{owner}: {label} is a number or a Parameter, got {width}.")
    if pair and min(values) < 0:
        raise ValueError(f"{owner}: radius must not be negative, got {width}.")
    if not pair and values[0] <= 0:
        raise ValueError(f"{owner}: sigma must be positive, got {width}.")
    return values


_MEMBERSHIPS = {
    "triangular": Triangular,
    "rectangular": Rectangular,
    "gaussian": Gaussian,
}


# How Fuzzify sizes a membership class on a center, from the distances to the
# centers on its left and on its right.
_PLACED = {
    "Triangular": lambda left, right: (left, right),
    "Rectangular": lambda left, right: (left / 2.0, right / 2.0),
    "Gaussian": lambda left, right: ((left + right) / 6.0,) * 2,
}


def _neighbor_gaps(centers, count):
    """Each of the ``count`` centers along the last axis of ``centers``, and
    its distances to the centers on its left and on its right; the first
    and the last reuse the one gap they have, and a lone center gets 1 on
    both sides."""
    if count == 1:
        return 1.0, 1.0
    gaps = centers[..., 1:] - centers[..., :-1]
    left = keras.ops.concatenate([gaps[..., :1], gaps], axis=-1)
    right = keras.ops.concatenate([gaps, gaps[..., -1:]], axis=-1)
    return left, right


@keras.saving.register_keras_serializable(package="nnodely")
class FuzzifyImpl(keras.layers.Layer):
    """The memberships of ``x`` to the sets on its centers, read as
    ``[x, *centers, *weights]``.

    ``centers`` lists the ``count`` centers: numbers, and None for each read
    from the next input; or it is None for one input holding them all along
    its last axis. ``functions`` gives each center a membership class placed
    on it, by name, or the index of one of ``layers``, Keras layers reading
    ``x - center`` and as many weight inputs as ``weights`` gives them.
    """

    def __init__(
        self,
        centers,
        count,
        functions,
        layers=(),
        weights=(),
        bounds: bool | list | tuple = False,
        normalize=False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.centers = centers
        self.count = int(count)
        self.functions = list(functions)
        self.function_layers = [
            layer
            if isinstance(layer, keras.layers.Layer)
            else keras.saving.deserialize_keras_object(layer)
            for layer in layers
        ]
        self.weight_counts = [int(count) for count in weights]
        self.bounds = list(bounds) if isinstance(bounds, (list, tuple)) else bounds
        self.normalize = bool(normalize)

    def _center_tensor(self, x, inputs):
        """The centers along a last axis that broadcasts against ``x``, and
        the inputs left after them."""
        rank = len(x.shape)
        if self.centers is None:
            return expand_to_rank(inputs[0], rank), inputs[1:]
        if None not in self.centers:
            return keras.ops.convert_to_tensor(self.centers, dtype=x.dtype), inputs
        read = self.centers.count(None)
        streams = [expand_to_rank(tensor, rank) for tensor in inputs[:read]]
        # The numbers are given the shape every stream broadcasts to.
        zero = sum(keras.ops.zeros_like(stream) for stream in streams)
        given = iter(streams)
        columns = [
            zero + (next(given) if center is None else center)
            for center in self.centers
        ]
        return keras.ops.concatenate(columns, axis=-1), inputs[read:]

    def call(self, inputs):
        x, *rest = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        centers, rest = self._center_tensor(x, rest)
        weights = []
        for count in self.weight_counts:
            weights.append(rest[:count])
            rest = rest[count:]

        if self.bounds is True:
            low = keras.ops.min(centers, axis=-1, keepdims=True)
            high = keras.ops.max(centers, axis=-1, keepdims=True)
            x = keras.ops.minimum(keras.ops.maximum(x, low), high)
        elif self.bounds:
            x = keras.ops.clip(x, *self.bounds)

        # Every center at once: the distances along the last axis, and the
        # widths of the placed memberships from the centers as they are, so
        # that they follow centers that move.
        shifted = x - centers
        left, right = _neighbor_gaps(centers, self.count)
        memberships: list = [None] * self.count
        for name in dict.fromkeys(f for f in self.functions if isinstance(f, str)):
            formula = _FORMULAS[_MEMBERSHIPS[name.lower()]._formula]
            values = formula(shifted, *_PLACED[name](left, right))
            if self.functions.count(name) == self.count:
                memberships = [values]
                break
            for index, function in enumerate(self.functions):
                if function == name:
                    memberships[index] = values[..., index : index + 1]
        for index, function in enumerate(self.functions):
            if isinstance(function, int):
                layer, extra = self.function_layers[function], weights[function]
                distance = shifted[..., index : index + 1]
                memberships[index] = (
                    layer([distance, *extra]) if extra else layer(distance)
                )

        result = keras.ops.concatenate(memberships, axis=-1)
        if self.normalize:
            total = keras.ops.sum(result, axis=-1, keepdims=True)
            result = result / keras.ops.where(total == 0, 1.0, total)
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "centers": self.centers,
                "count": self.count,
                "functions": self.functions,
                "layers": [
                    keras.saving.serialize_keras_object(layer)
                    for layer in self.function_layers
                ],
                "weights": self.weight_counts,
                "bounds": self.bounds,
                "normalize": self.normalize,
            }
        )
        return config


class Fuzzify(Layer):
    r"""Memberships of a scalar stream to sets placed on centers: one feature per center.

    Syntax
    ------
    ``Fuzzify(centers, function=Triangular)(x)``
        The membership of ``x`` to a triangle on each center.
    ``Fuzzify(centers, function=[f_0, f_1, ...])(x)``
        A function of its own for each center.
    ``Fuzzify(Parameter("c", value=[...]), function)(x)``
        Centers trained with the model.
    ``Fuzzify(centers, function, bounds=True)(x)``
        ``x`` held between the lowest and the highest center first.
    ``Fuzzify(centers, function, normalize=True)(x)``
        Memberships divided by their sum, so they add up to one.

    Each center gives one feature: the value of its function at ``x``. The
    membership functions :class:`Triangular`, :class:`Rectangular` and
    :class:`Gaussian` are placed on each center and sized by the centers next
    to it. Any other layer is applied to ``x - center``.

    Every sample of the window is evaluated alone, and the sequence axes are
    kept. The result is typically the activation of a :class:`LocalModel`.

    Parameters
    ----------
    centers : list of float, Stream, or list of float and Stream
        Where the sets are placed. It is one of:

        - a list of numbers: fixed centers, sorted in ascending order, each
          with its function;
        - a :class:`Parameter` of dim ``(n,)``: ``n`` centers trained with the
          model;
        - a stream of dim ``(n,)`` computed from the inputs, such as a
          :class:`Linear` with ``features=n``: every sample has centers of
          its own;
        - a list of numbers and scalar streams, dim ``(1,)``: the numbers stay
          fixed, the streams move.

        A stream of centers holds one sample: no window, or a window of 1. It
        has no sequence, or the sequence of ``x``. Centers that move are not
        sorted: give them in ascending order.
    function : type, str, Layer, keras.layers.Layer, or a list of them
        The function of every center, or a list of one per center, paired
        with ``centers`` as given, before they are sorted. Each is one of:

        - :class:`Triangular`, :class:`Rectangular` or :class:`Gaussian`, as
          a class or by name, in any case (``"triangular"``): placed on the
          center and sized by its neighbors, see Notes;
        - another nnodely layer class, such as :class:`Sin`: made anew for
          each center and applied to ``x - center``;
        - an nnodely layer instance, such as ``Triangular(radius=0.2)``:
          shared by every center and applied to ``x - center``;
        - a ``keras.layers.Layer`` subclass or instance whose ``call`` reads
          one argument, ``x - center``: likewise, wrapped in a
          :class:`KerasFunction`, so the model still saves and exports.
    bounds : bool or (float, float), default False
        Holds ``x`` within limits before the functions read it:

        - ``False``: ``x`` is read as it is;
        - ``True``: ``x`` is held between the lowest and the highest center,
          so a value beyond them belongs to the outer set;
        - ``(left, right)``: ``x`` is held between two numbers,
          ``left <= right``.
    normalize : bool, default False
        ``True`` divides the memberships of each sample by their sum, so they
        add up to one. A sample with all memberships zero stays zero. Use it
        for memberships that are not negative.
    name : str, optional
        The name of the layer. The default is ``Fuzzify`` followed by a
        number.

    Call arguments
    --------------
    x : Stream
        A scalar stream, ``Shape(S, T, (1,))``, with or without a window and a
        sequence. The call takes it alone, ``Fuzzify(...)(x)``, or in a list,
        ``Fuzzify(...)([x])``.

    Returns
    -------
    Stream
        ``Shape(S, T, (n,))`` for ``n`` centers. A function that gives several
        values per center, such as ``Linear(features=2)``, adds that many
        features for its center. For example, ``Input("x", tw=4)`` and 3
        centers give ``Shape(S=(), T=4, D=(3,))``.

    Raises
    ------
    ValueError
        If ``centers`` is empty ("requires at least one center"), or a stream
        of centers is not a vector of one sample ("a stream of centers is a
        vector of shape (n,), one sample"). If an item of a list is not a
        number or a scalar stream ("in a list every center is a number or a
        scalar stream"). If the functions are not one, or one per center
        ("functions for ... centers"), or a name is unknown ("Unsupported
        fuzzify function"). If ``bounds`` is not ``True``, ``False`` or an
        ordered pair ("bounds is True, False or (left, right)"). At the call:
        if ``x`` is not scalar ("Fuzzify reads a scalar signal"), or centers
        have another sequence than ``x`` ("centers with a sequence have the
        signal's").
    TypeError
        If a function is not a layer class, instance or name ("a function is
        an nnodely or Keras layer class or instance"). If a Keras ``call``
        needs more than one argument ("a custom function reads one
        argument"), or a layer class cannot be made without arguments
        ("cannot be made without arguments; pass an instance"). If the call
        is given more than one stream ("Fuzzify reads one stream").

    See Also
    --------
    LocalModel : Local models blended by these memberships.
    Triangular : Triangular membership.
    Rectangular : Rectangular membership.
    Gaussian : Gaussian membership.
    KerasFunction : A Keras layer as an nnodely layer.

    Notes
    -----
    For the center :math:`c_k`, the gaps to the centers next to it are
    :math:`l_k = c_k - c_{k-1}` and :math:`r_k = c_{k+1} - c_k`. The first and
    the last center reuse the one gap they have. A lone center has gaps of 1.
    The membership functions placed on :math:`c_k` are:

    .. math::

        \mu_k^{\triangle} = \max\left(0, \min\left(1 + \frac{x - c_k}{l_k},
        1 - \frac{x - c_k}{r_k}\right)\right)

        \mu_k^{\square} = 1 \text{ if } -\frac{l_k}{2} \le x - c_k <
        \frac{r_k}{2}, \text{ else } 0

        \mu_k^{G} = \exp\left(-\frac{(x - c_k)^2}{2 \sigma_k^2}\right),
        \quad \sigma_k = \frac{l_k + r_k}{6}

    Between two centers the triangles add up to one, and exactly one
    rectangle is 1. A triangle on the outer center reaches 0 one gap beyond
    it, a rectangle half a gap beyond it. A gap of 0 or below, centers that
    meet or cross, is held at :math:`10^{-7}`.

    Any other function :math:`f` gives :math:`\mu_k = f(x - c_k)`. With
    ``bounds``, :math:`x` is first replaced by
    :math:`\min(\max(x, \text{left}), \text{right})`. With ``normalize``,
    :math:`\mu_k` is replaced by :math:`\mu_k / \sum_j \mu_j`.

    Centers that move take their sets with them, widths included. The
    gradient reaches them through both, except through rectangles, which
    are steps. The Parameters of the function layers are listed by
    :attr:`Modely.parameters`, and are trained, frozen and saved with the
    model. Calling the same Fuzzify again applies the same function layers.
    To load a model with a custom Keras function in another session, define
    its class first, decorated with
    ``@keras.saving.register_keras_serializable()``.

    Basic Examples
    --------------
    Triangles on 0, 0.5 and 1. The value 0.25 is halfway between the first
    two centers, and 0.5 is on the second:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import (
    ...     Fir, Fuzzify, Gaussian, Input, Linear, LocalModel, Modely, Output,
    ...     Parameter, Rectangular, Select, Sin, Triangular,
    ... )
    >>> x = Input("x", tw=2)
    >>> mu = Fuzzify([0.0, 0.5, 1.0], function=Triangular)(x)
    >>> model = Modely("fuzzy", inputs=[x], outputs=[Output("mu", mu)]).build()
    >>> samples = {"x": np.array([[[0.25], [0.5]]])}
    >>> keras.ops.convert_to_numpy(model(samples)["mu"])
    array([[[0.5, 0.5, 0. ],
            [0. , 1. , 0. ]]], dtype=float32)

    Scope
    -----
    Every sample of a window is evaluated alone, and a sequence is kept:

    >>> mu.shape
    Shape(S=(), T=2, D=(3,))
    >>> Fuzzify([0.0, 0.5, 1.0], function=Triangular)(Input("s", seq=5, tw=4)).shape
    Shape(S=(5,), T=4, D=(3,))
    >>> Fuzzify([0.0, 0.5, 1.0], function=Triangular)(Input("n")).shape
    Shape(S=(), T=None, D=(3,))

    Centers held by a Parameter vector, a list with a scalar Parameter, and a
    stream computed from an input. Here all three are at 0, 0.5 and 1:

    >>> trained = Parameter("centers", value=[0.0, 0.5, 1.0])
    >>> middle = Parameter("middle", value=[0.5])
    >>> u = Input("u", tw=1)
    >>> computed = Linear(features=3, kernel=Parameter("k", value=[[0.0, 0.5, 1.0]]), bias=False)(u)
    >>> outputs = [
    ...     Output("vector", Fuzzify(trained, function=Triangular)(x)),
    ...     Output("listed", Fuzzify([0.0, middle, 1.0], function=Triangular)(x)),
    ...     Output("computed", Fuzzify(computed, function=Triangular)(x)),
    ... ]
    >>> model = Modely("centers", inputs=[x, u], outputs=outputs).build()
    >>> result = model({"x": np.array([[[0.25], [0.5]]]), "u": np.array([[[1.0]]])})
    >>> np.stack([keras.ops.convert_to_numpy(result[key])[0, 0] for key in ["vector", "listed", "computed"]])
    array([[0.5, 0.5, 0. ],
           [0.5, 0.5, 0. ],
           [0.5, 0.5, 0. ]], dtype=float32)

    Options
    -------
    ``function`` by name, in any case:

    >>> Fuzzify([0.0, 1.0], function="gaussian")(x).shape
    Shape(S=(), T=2, D=(2,))

    A list gives one function per center, paired before the centers are
    sorted. Here the triangle is on 0 and the rectangle on 1:

    >>> paired = Fuzzify([1.0, 0.0], function=[Rectangular, Triangular])(x)

    Another layer class is applied to ``x - center``: ``Sin`` gives
    :math:`\sin(x - c)`. An instance is shared by every center: here a wider
    triangle. A custom Keras layer works the same way:

    >>> class Squared(keras.layers.Layer):
    ...     def call(self, x):
    ...         return x * x
    >>> waves = Fuzzify([0.0, 0.5], function=Sin)(x)
    >>> wide = Fuzzify([0.0, 0.5, 1.0], function=Triangular(radius=1.0))(x)
    >>> squares = Fuzzify([0.0, 0.5, 1.0], function=Squared)(x)
    >>> outputs = [Output(name, stream) for name, stream in
    ...            [("paired", paired), ("waves", waves), ("wide", wide), ("squares", squares)]]
    >>> model = Modely("functions", inputs=[x], outputs=outputs).build()
    >>> result = model({"x": np.array([[[0.25], [0.75]]])})
    >>> keras.ops.convert_to_numpy(result["paired"])
    array([[[0.75, 0.  ],
            [0.25, 1.  ]]], dtype=float32)
    >>> np.round(keras.ops.convert_to_numpy(result["waves"]), 4)
    array([[[ 0.2474, -0.2474],
            [ 0.6816,  0.2474]]], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["wide"])
    array([[[0.75, 0.75, 0.25],
            [0.25, 0.75, 0.75]]], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["squares"])
    array([[[0.0625, 0.0625, 0.5625],
            [0.5625, 0.0625, 0.0625]]], dtype=float32)

    A function giving several values adds them all, here 2 per center:

    >>> Fuzzify([0.0, 0.5, 1.0], function=Linear(features=2))(x).shape
    Shape(S=(), T=2, D=(6,))

    ``bounds`` holds ``x`` before the functions read it. Beyond the outer
    triangles, 1.5 and -1 belong to no set without it:

    >>> outputs = [
    ...     Output("free", Fuzzify([0.0, 0.5, 1.0], function=Triangular)(x)),
    ...     Output("held", Fuzzify([0.0, 0.5, 1.0], function=Triangular, bounds=True)(x)),
    ...     Output("pair", Fuzzify([0.0, 0.5, 1.0], function=Triangular, bounds=(0.0, 0.75))(x)),
    ... ]
    >>> model = Modely("bounds", inputs=[x], outputs=outputs).build()
    >>> result = model({"x": np.array([[[1.5], [-1.0]]])})
    >>> for key in ["free", "held", "pair"]:
    ...     print(key, keras.ops.convert_to_numpy(result[key])[0].tolist())
    free [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    held [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    pair [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]]

    ``normalize=True`` divides by the sum. Wide triangles on 0, 0.5 and 1 give
    ``[0.5, 1, 0.5]`` at 0.5, which add up to 2:

    >>> shares = Fuzzify([0.0, 0.5, 1.0], function=Triangular(radius=1.0), normalize=True)(x)
    >>> model = Modely("normalized", inputs=[x], outputs=[Output("shares", shares)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[[0.5], [0.5]]])})["shares"])[0, 0]
    array([0.25, 0.5 , 0.25], dtype=float32)

    ``name`` names the layer:

    >>> Fuzzify([0.0, 1.0], function=Triangular, name="regimes")(x).name
    'regimes'

    Applications
    ------------
    A gain scheduled by the gear: one filter of the torque per gear, blended
    by the gear's memberships:

    >>> gear = Input("gear", tw=1)
    >>> torque = Input("torque", tw=10)
    >>> gears = Fuzzify([1.0, 2.0, 3.0, 4.0], function=Triangular, bounds=True)(gear)
    >>> force = LocalModel(lambda cell: Fir())(torque, gears)
    >>> force.shape
    Shape(S=(), T=1, D=(1,))

    Properties & Relations
    ----------------------
    Between the outer centers, the triangles add up to one:

    >>> x5 = Input("x5", tw=5)
    >>> mu5 = Fuzzify([0.0, 0.5, 1.0], function=Triangular)(x5)
    >>> model = Modely("partition", inputs=[x5], outputs=[Output("mu5", mu5)]).build()
    >>> values = np.array([[[0.0], [0.125], [0.375], [0.625], [1.0]]])
    >>> keras.ops.convert_to_numpy(model({"x5": values})["mu5"]).sum(axis=-1)
    array([[1., 1., 1., 1., 1.]], dtype=float32)

    Possible Issues
    ---------------
    The stream is scalar. Select one feature of a vector:

    >>> position = Input("position", dim=2, tw=1)
    >>> Fuzzify([0.0, 1.0], function=Triangular)(position)
    Traceback (most recent call last):
        ...
    ValueError: ...Fuzzify reads a scalar signal, got dim (2,).
    >>> Fuzzify([0.0, 1.0], function=Triangular)(Select(idx=0)(position)).shape
    Shape(S=(), T=1, D=(2,))

    A list of functions has one per center:

    >>> Fuzzify([0.0, 0.5, 1.0], function=[Triangular, Gaussian])
    Traceback (most recent call last):
        ...
    ValueError: Fuzzify: 2 functions for 3 centers; give one function, or one per center.

    A layer class with required settings cannot be made for each center. Pass
    an instance, shared by every center:

    >>> Fuzzify([0.0, 1.0], function=Select)(x)
    Traceback (most recent call last):
        ...
    TypeError: Fuzzify: Select cannot be made without arguments; pass an instance, ...

    .. versionadded:: 2.0
    """

    def __init__(
        self,
        centers,
        function,
        bounds: bool | tuple[float, float] = False,
        normalize: bool = False,
        name=None,
    ):
        name = next_name("Fuzzify") if name is None else name
        self._bounds = _bounds(bounds, name)
        self._normalize = bool(normalize)
        self._centers, self._center_streams = _centers(centers, name)
        count = (
            self._center_streams[0].shape.dim[0]
            if self._centers is None
            else len(self._centers)
        )
        functions = (
            list(function)
            if isinstance(function, (list, tuple))
            else [function] * count
        )
        if len(functions) != count:
            raise ValueError(
                f"Fuzzify: {len(functions)} functions for {count} centers; "
                "give one function, or one per center."
            )
        functions = [_membership_function(item) for item in functions]
        # Fixed centers are sorted, each with its function.
        if self._centers is not None and None not in self._centers:
            fixed = self._centers
            order = sorted(range(count), key=lambda index: fixed[index])
            self._centers = [fixed[index] for index in order]
            functions = [functions[index] for index in order]
        self._count = count
        self._functions = functions
        # The layer each center applies to x - center, None for a membership
        # placed on it: made on the first call and shared by every later one.
        self._layers: list[Layer | None] | None = None
        # Of a node: those layers, each once, applied to its signal.
        self._applied: list[Layer] = []
        self._probes: list[Layer] = []
        super().__init__(
            name=name,
            centers=centers,
            function=function,
            bounds=bounds,
            normalize=normalize,
        )

    def _make_layers(self) -> list[Layer | None]:
        """For each center, the layer it applies to ``x - center``: a layer
        instance as it is, a class made anew and named after it, a Keras
        layer wrapped in a :class:`KerasFunction`; None for a membership
        placed on it."""
        wrappers: dict[int, KerasFunction] = {}
        layers: list[Layer | None] = []
        for index, function in enumerate(self._functions):
            if isinstance(function, type) and function in _MEMBERSHIPS.values():
                layers.append(None)
            elif isinstance(function, keras.layers.Layer):
                # A Keras instance is shared by the centers, as a layer is: it
                # is wrapped once, however many centers it is given to.
                if id(function) not in wrappers:
                    wrappers[id(function)] = KerasFunction(
                        function, name=f"{self.name}_{type(function).__name__}"
                    )
                layers.append(wrappers[id(function)])
            elif isinstance(function, Layer):
                layers.append(function)
            else:
                name = f"{self.name}_{function.__name__}_{index}"
                if issubclass(function, keras.layers.Layer):
                    layers.append(KerasFunction(function, name=name))
                    continue
                try:
                    layers.append(function(name=name))
                except TypeError as exc:
                    raise TypeError(
                        f"Fuzzify: {function.__name__} cannot be made without "
                        f"arguments; pass an instance, such as "
                        f"{function.__name__}(...), shared by every center."
                    ) from exc
        return layers

    def __call__(self, inputs):
        streams = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        if len(streams) != 1 or not isinstance(streams[0], Stream):
            raise TypeError(f"{self.name}: Fuzzify reads one stream, its signal.")
        x = streams[0]
        if tuple(x.shape.dim) != (1,):
            raise ValueError(
                f"{self.name}: Fuzzify reads a scalar signal, got dim {x.shape.dim}."
            )
        for center in self._center_streams:
            seq = tuple(center.shape.seq)
            if seq not in ((), tuple(x.shape.seq)):
                raise ValueError(
                    f"{self.name}: centers with a sequence have the signal's, "
                    f"{tuple(x.shape.seq)}; got {seq}."
                )
        if self._layers is None:
            self._layers = self._make_layers()
        # Each layer is applied once to the signal: the shape it gives, and
        # the Parameters it reads, predecessors of the Fuzzify too.
        self._applied = list(
            {id(layer): layer for layer in self._layers if layer is not None}.values()
        )
        self._probes = [layer([x]) for layer in self._applied]
        weights = [weight for probe in self._probes for weight in probe._preds[1:]]
        node = super().__call__([x, *self._center_streams, *weights])
        node._layers, node._applied = self._layers, self._applied
        node._probes = self._probes
        return node

    def output_shape(self, *inputs):
        # The memberships side by side: a placed one has the signal's shape,
        # one computed by a layer the shape the layer gives.
        given = {
            id(layer): probe.shape for layer, probe in zip(self._applied, self._probes)
        }
        shapes = [
            inputs[0].shape if layer is None else given[id(layer)]
            for layer in self._layers or []
        ]
        first = shapes[0]
        if any(
            shape.dim_rank != 1 or (shape.seq, shape.tw) != (first.seq, first.tw)
            for shape in shapes
        ):
            raise ValueError(
                f"{self.name}: the functions give memberships that cannot be put "
                f"side by side: {', '.join(map(str, shapes))}."
            )
        return first.replace(dim=(sum(shape.dim[0] for shape in shapes),))

    def _function_specs(self) -> list[str | int]:
        """Each center's function: a placed membership by name, or the index
        of its layer among the applied ones."""
        applied = [id(layer) for layer in self._applied]
        return [
            function.__name__ if layer is None else applied.index(id(layer))
            for function, layer in zip(self._functions, self._layers or [])
        ]

    def build_layer(self):
        return FuzzifyImpl(
            centers=self._centers,
            count=self._count,
            functions=self._function_specs(),
            layers=[probe.build_layer() for probe in self._probes],
            weights=[len(probe._preds) - 1 for probe in self._probes],
            bounds=self._bounds,
            normalize=self._normalize,
            name=self.name,
        )

    def get_config(self):
        # The arguments as checked: the fixed centers sorted, a stream among
        # them saved as the predecessor it is, and the functions by name or
        # by the layer applied to x - center. Those layers are saved inside,
        # each once, with how many of the predecessors are the weights it reads.
        config = super().get_config()
        config.update(
            {
                "centers": self._centers,
                "function": self._function_specs(),
                "layers": [
                    {
                        "class_name": class_path(type(layer)),
                        "config": layer.get_config(),
                        "weights": len(probe._preds) - 1,
                    }
                    for layer, probe in zip(self._applied, self._probes)
                ],
                "bounds": list(self._bounds)
                if isinstance(self._bounds, tuple)
                else self._bounds,
                "normalize": self._normalize,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        x, *rest = preds or []
        centers = config["centers"]
        if centers is None:
            centers, rest = rest[0], rest[1:]
        else:
            read = iter(rest)
            centers = [next(read) if center is None else center for center in centers]
            rest = list(read)
        layers = []
        for saved in config["layers"]:
            weights, rest = rest[: saved["weights"]], rest[saved["weights"] :]
            layer_class = cast(type[Layer], registered_class(saved["class_name"]))
            if layer_class is None:
                raise ValueError(
                    f"Fuzzify {config['name']!r} was saved with a "
                    f"{saved['class_name']!r} function, which is not defined: "
                    "define or import it before loading."
                )
            node = layer_class.from_config(saved["config"], preds=[x, *weights])
            layers.append(node._source)
        config = {key: value for key, value in config.items() if key != "layers"}
        bounds = config["bounds"]
        config.update(
            {
                "centers": centers,
                "function": [
                    layers[item] if isinstance(item, int) else item
                    for item in config["function"]
                ],
                "bounds": tuple(bounds) if isinstance(bounds, list) else bounds,
            }
        )
        return super().from_config(config, preds=[x])

    def plot(self, x_range: tuple[float, float] | None = None, points=401, ax=None):
        """Draw the memberships with matplotlib, and return the axes.

        The memberships are drawn as the layer computes them now. Centers and
        widths held by Parameters are drawn at their current values: the
        initial ones before training, the trained ones after. Call it on the
        Fuzzify, even before it is applied, or on the stream it returns.

        Parameters
        ----------
        x_range : (float, float), optional
            The span of the signal drawn. The default spans the centers and
            one outer gap beyond each end, where the outer placed sets end.
        points : int, default 401
            The number of values of the signal drawn.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. The default draws on a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on. Each membership is a line in the color of its
            center, labeled ``c = <center>``, and a dotted vertical line marks
            each center. A function giving several values per center labels
            its lines ``μ0``, ``μ1``, and so on. The title is the name of the
            layer, the x label the name of the stream it reads.

        Raises
        ------
        ValueError
            If the centers are computed from the inputs ("is computed, with
            no value of its own"). If a function layer has weights and its
            model is not built yet ("has no value until its model is
            built").

        See Also
        --------
        Modely.build : Creates the weights a function layer reads.

        Basic Examples
        --------------
        >>> import matplotlib.pyplot as plt
        >>> from nnodely import Fuzzify, Input, Triangular
        >>> regimes = Fuzzify([0.0, 0.5, 1.0], function=Triangular, name="regimes")
        >>> ax = regimes.plot()
        >>> ax.get_title(), [line.get_label() for line in ax.get_lines()[:3]]
        ('regimes', ['c = 0', 'c = 0.5', 'c = 1'])

        Options
        -------
        ``x_range`` and ``points`` choose the values drawn. On the stream, the
        x label is the name of the signal:

        >>> ax = regimes(Input("speed", tw=1)).plot(x_range=(0.0, 1.0), points=5)
        >>> ax.get_xlabel(), ax.get_lines()[0].get_ydata()
        ('speed', array([1. , 0.5, 0. , 0. , 0. ], dtype=float32))

        ``ax`` draws on axes of your own:

        >>> figure, axes = plt.subplots()
        >>> regimes.plot(ax=axes) is axes
        True
        >>> plt.close("all")

        Possible Issues
        ---------------
        Centers held by a Parameter declared with a window, ``tw=1``, are not
        drawn: matplotlib raises ``ValueError`` ("x and y must have same first
        dimension"). Declare the Parameter without a window,
        ``Parameter("c", value=[0.0, 0.5, 1.0])``. It computes the same
        memberships.

        .. versionadded:: 2.0
        """
        import matplotlib.pyplot as plt

        if self._layers is None:
            # Not applied yet: applied to a signal of its own, as a model would.
            self([Input(f"{self.name}_plot")])
        streams = [
            *self._center_streams,
            *(weight for probe in self._probes for weight in probe._preds[1:]),
        ]
        values = [_current_value(stream, self.name) for stream in streams]
        read = iter(values)
        if self._centers is None:
            centers = np.ravel(next(read))
        else:
            centers = np.array(
                [np.ravel(next(read))[0] if c is None else c for c in self._centers]
            )
        if x_range is None:
            ordered = np.sort(centers)
            gaps = np.diff(ordered)
            left = gaps[0] if len(gaps) and gaps[0] > 0 else 1.0
            right = gaps[-1] if len(gaps) and gaps[-1] > 0 else 1.0
            x_range = (ordered[0] - left, ordered[-1] + right)
        x = np.linspace(*x_range, points, dtype="float32")
        layer = self.build_layer() if self._layer is None else self._layer
        memberships = np.asarray(
            keras.ops.convert_to_numpy(layer([x[:, None], *values]))
        )

        if ax is None:
            ax = plt.subplots(figsize=(8, 3), layout="constrained")[1]
        # One membership per center is drawn in the color of its center.
        per_center = memberships.shape[-1] == len(centers)
        for index in range(memberships.shape[-1]):
            label = f"c = {centers[index]:.3g}" if per_center else f"μ{index}"
            ax.plot(x, memberships[:, index], color=f"C{index}", label=label)
        for index, center in enumerate(centers):
            color = f"C{index}" if per_center else "gray"
            ax.axvline(center, color=color, linestyle=":", linewidth=0.8)
        signal = self._preds[0].name if self._preds else "signal"
        ax.set(title=self.name, xlabel=signal, ylabel="membership")
        ax.grid(True)
        ax.legend(fontsize=8)
        return ax


def _bounds(bounds, owner: str) -> bool | tuple[float, float]:
    """``bounds`` checked: a bool, or ``(left, right)`` numbers in order."""
    if isinstance(bounds, bool):
        return bounds
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        left, right = bounds
        numbers = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (left, right)
        )
        if numbers and left <= right:
            return float(left), float(right)
    raise ValueError(
        f"{owner}: bounds is True, False or (left, right), two numbers with "
        f"left <= right; got {bounds!r}."
    )


def _centers(centers, owner: str) -> tuple[list[float | None] | None, list[Stream]]:
    """``centers`` checked, as the list of them - numbers, None for each read
    from a stream - and the streams they are read from: one stream of shape
    ``(n,)``, of one sample, with None for the list; or the scalar streams
    of a list of them and numbers."""
    if isinstance(centers, Stream):
        if centers.shape.dim_rank != 1 or not _one_step(centers):
            raise ValueError(
                f"{owner}: a stream of centers is a vector of shape (n,), one "
                f"sample; got dim {centers.shape.dim}, tw {centers.shape.tw}."
            )
        return None, [centers]
    items = list(centers)
    if not items:
        raise ValueError("Fuzzify requires at least one center.")
    listed: list[float | None] = []
    streams: list[Stream] = []
    for index, item in enumerate(items):
        if not isinstance(item, Stream):
            listed.append(float(item))
            continue
        if tuple(item.shape.dim) != (1,) or not _one_step(item):
            raise ValueError(
                f"{owner}: in a list every center is a number or a scalar "
                f"stream, dim (1,) and one sample; center {index} has "
                f"dim {item.shape.dim}, tw {item.shape.tw}."
            )
        listed.append(None)
        streams.append(item)
    return listed, streams


def _one_step(stream: Stream) -> bool:
    """Whether a stream holds one sample - a window of one, or none; its
    sequence, if any, must be the signal's."""
    return stream.shape.tw in (None, 1)


def _current_value(stream: Stream, owner: str):
    """The value a Parameter or a Constant holds now, with a batch axis of
    one: its variable once its model is built, the value given before."""
    if not isinstance(stream, _Value):
        raise ValueError(
            f"{owner}: plot() draws centers and widths held by Parameters or "
            f"Constants; {stream.name!r} is computed, with no value of its own."
        )
    value = stream.value if stream._variable is None else stream.value_numpy
    if value is None:
        raise ValueError(
            f"{owner}: {stream.name!r} has no value until its model is built."
        )
    return np.asarray(value, dtype="float32")[None]


def _membership_function(function):
    """A ``function`` of :class:`Fuzzify` checked: a membership name made its
    class, and anything but a layer class or instance, nnodely or Keras,
    refused."""
    if isinstance(function, str):
        try:
            return _MEMBERSHIPS[function.lower()]
        except KeyError:
            raise ValueError(
                f"Unsupported fuzzify function: {function!r}. Supported names: "
                "'Triangular', 'Rectangular', 'Gaussian'; or any nnodely layer."
            ) from None
    if _is_keras_layer(function):
        _check_one_argument(function)
        return function
    if isinstance(function, Layer) or (
        isinstance(function, type) and issubclass(function, Layer)
    ):
        return function
    raise TypeError(
        "Fuzzify: a function is an nnodely or Keras layer class or instance, "
        f"or a membership name, got {function!r}."
    )


def _check_one_argument(function) -> None:
    """Refuse a Keras layer whose ``call`` needs more than one argument: a
    function of :class:`Fuzzify` reads ``x - center`` alone. Arguments with a
    default, such as ``training=None``, are left to Keras."""
    layer_class = function if isinstance(function, type) else type(function)
    parameters = list(inspect.signature(layer_class.call).parameters.values())[1:]
    required = [
        parameter.name
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if len(required) > 1:
        raise TypeError(
            f"Fuzzify: a custom function reads one argument, x - center; "
            f"{layer_class.__name__}.call needs {len(required)}: "
            f"{', '.join(required)}."
        )
