"""Fuzzify and the membership functions it places on its centers."""

from __future__ import annotations

import inspect

import keras

from nnodely.core.dag import next_name
from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank, slice_axis
from nnodely.core.stream import Stream
from nnodely.layers.function import KerasFunction, _is_keras_layer
from nnodely.layers.parameter import Constant
from nnodely.layers.partitions import Concatenate, Select

_EPS = 1e-7


def _radius(radius, owner: str) -> tuple[float, float]:
    """``radius`` as its distances to the left and to the right of the center:
    a number for both, or a ``(left, right)`` pair."""
    if isinstance(radius, (int, float)):
        left = right = float(radius)
    else:
        values = [float(value) for value in radius]
        if len(values) != 2:
            raise ValueError(
                f"{owner}: radius is a number or a (left, right) pair, got {radius}."
            )
        left, right = values
    if left < 0 or right < 0:
        raise ValueError(f"{owner}: radius must not be negative, got {radius}.")
    return left, right


# The membership formulas, on a center and widths that are numbers for a
# membership of its own and tensors for one on a moving center of a Fuzzify.
def _floor(width):
    """``width`` kept above zero, so that it can divide."""
    if isinstance(width, (int, float)):
        return max(width, _EPS)
    return keras.ops.maximum(width, _EPS)


def _triangle(x, center, left, right):
    rise = (x - (center - left)) / _floor(left)
    fall = ((center + right) - x) / _floor(right)
    return keras.ops.maximum(keras.ops.minimum(rise, fall), 0.0)


def _box(x, center, left, right):
    # Half-open, so that boxes side by side cover their common edge once.
    inside = keras.ops.logical_and(x >= center - left, x < center + right)
    return keras.ops.cast(inside, x.dtype)


def _bell(x, center, sigma):
    scaled = (x - center) / _floor(sigma)
    return keras.ops.exp(-0.5 * scaled * scaled)


@keras.saving.register_keras_serializable(package="nnodely")
class TriangularImpl(keras.layers.Layer):
    def __init__(self, center=0.0, left=1.0, right=1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.center = float(center)
        self.left = float(left)
        self.right = float(right)

    def call(self, inputs):
        return _triangle(inputs, self.center, self.left, self.right)

    def get_config(self):
        config = super().get_config()
        config.update({"center": self.center, "left": self.left, "right": self.right})
        return config


@keras.saving.register_keras_serializable(package="nnodely")
class RectangularImpl(keras.layers.Layer):
    def __init__(self, center=0.0, left=1.0, right=1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.center = float(center)
        self.left = float(left)
        self.right = float(right)

    def call(self, inputs):
        return _box(inputs, self.center, self.left, self.right)

    def get_config(self):
        config = super().get_config()
        config.update({"center": self.center, "left": self.left, "right": self.right})
        return config


@keras.saving.register_keras_serializable(package="nnodely")
class GaussianImpl(keras.layers.Layer):
    def __init__(self, center=0.0, sigma=1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.center = float(center)
        self.sigma = float(sigma)

    def call(self, inputs):
        return _bell(inputs, self.center, self.sigma)

    def get_config(self):
        config = super().get_config()
        config.update({"center": self.center, "sigma": self.sigma})
        return config


class _Membership(Layer):
    """A membership function: applied to every value of a stream, it keeps
    the stream's shape. :class:`Fuzzify` places one on each of its centers,
    sized by the centers next to it, with :meth:`_placed` - or, on centers
    that move, evaluates it with :meth:`_moving`."""

    _one_stream = True

    def output_shape(self, *inputs):
        # Elementwise: every axis kept, a dynamic time axis included.
        return inputs[0].shape

    @classmethod
    def _placed(cls, center, left, right, name):
        """The membership on ``center``, whose neighbors are ``left`` and
        ``right`` away."""
        raise NotImplementedError

    @staticmethod
    def _moving(x, center, left, right):
        """:meth:`_placed` evaluated on ``x``, the center and the distances
        to its neighbors given as tensors."""
        raise NotImplementedError


class Triangular(_Membership):
    """
    Triangular membership: 1 at ``center``, falling linearly to 0 at
    ``radius`` from it, and 0 beyond. ``radius`` is a number, or a
    ``(left, right)`` pair for a triangle leaning to one side::

        Triangular(center=0.5, radius=0.5)([x])   # 0 at 0 and at 1
        Triangular(center=1.0, radius=(1.0, 2.0))         # from 0, up at 1, to 3

    In a :class:`Fuzzify`, each triangle reaches the centers next to it, so
    between two centers the memberships add up to one.
    """

    def __init__(
        self,
        center: float = 0.0,
        radius: float | tuple[float, float] = 1.0,
        name=None,
    ):
        self.center = float(center)
        self.left, self.right = _radius(radius, "Triangular")
        super().__init__(
            name=name,
            center=self.center,
            radius=self.left if self.left == self.right else [self.left, self.right],
        )

    def build_layer(self):
        return TriangularImpl(
            center=self.center, left=self.left, right=self.right, name=self.name
        )

    @classmethod
    def _placed(cls, center, left, right, name):
        return cls(center=center, radius=(left, right), name=name)

    @staticmethod
    def _moving(x, center, left, right):
        return _triangle(x, center, left, right)


class Rectangular(_Membership):
    """
    Rectangular membership: 1 on ``[center - left, center + right)``, 0
    elsewhere. ``radius`` is a number for both sides, or a ``(left, right)``
    pair::

        Rectangular(center=0.5, radius=0.25)([x])   # 1 on [0.25, 0.75)

    In a :class:`Fuzzify`, each rectangle reaches halfway to the centers next
    to it, so between the centers exactly one set holds every value.
    """

    def __init__(
        self,
        center: float = 0.0,
        radius: float | tuple[float, float] = 1.0,
        name=None,
    ):
        self.center = float(center)
        self.left, self.right = _radius(radius, "Rectangular")
        super().__init__(
            name=name,
            center=self.center,
            radius=self.left if self.left == self.right else [self.left, self.right],
        )

    def build_layer(self):
        return RectangularImpl(
            center=self.center, left=self.left, right=self.right, name=self.name
        )

    @classmethod
    def _placed(cls, center, left, right, name):
        return cls(center=center, radius=(left / 2.0, right / 2.0), name=name)

    @staticmethod
    def _moving(x, center, left, right):
        return _box(x, center, left / 2.0, right / 2.0)


class Gaussian(_Membership):
    """
    Gaussian membership: ``exp(-((x - center) / sigma)^2 / 2)``, 1 at
    ``center``::

        Gaussian(center=0.5, sigma=0.1)([x])

    In a :class:`Fuzzify`, ``sigma`` is a sixth of the span between the
    centers next to it, so three sigmas reach each of them.
    """

    def __init__(self, center: float = 0.0, sigma: float = 1.0, name=None):
        self.center = float(center)
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError(f"Gaussian: sigma must be positive, got {sigma}.")
        super().__init__(name=name, center=self.center, sigma=self.sigma)

    def build_layer(self):
        return GaussianImpl(center=self.center, sigma=self.sigma, name=self.name)

    @classmethod
    def _placed(cls, center, left, right, name):
        return cls(center=center, sigma=max((left + right) / 6.0, _EPS), name=name)

    @staticmethod
    def _moving(x, center, left, right):
        return _bell(x, center, (left + right) / 6.0)


_MEMBERSHIPS = {
    "triangular": Triangular,
    "rectangular": Rectangular,
    "gaussian": Gaussian,
}


def _neighbor_distances(centers):
    """For each of the sorted ``centers``, its distances to the centers on
    its left and on its right; the first and the last reuse the one gap they
    have, and a lone center gets 1 on both sides."""
    if len(centers) == 1:
        return [(1.0, 1.0)]
    gaps = [after - before for before, after in zip(centers, centers[1:])]
    return [
        (gaps[max(index - 1, 0)], gaps[min(index, len(gaps) - 1)])
        for index in range(len(centers))
    ]


def _neighbors(centers, index, rank, axis):
    """Center ``index`` of a tensor of centers, which lie along its tensor
    axis ``axis``, and its distances to the centers on its left and on its
    right, read as :func:`_neighbor_distances` reads fixed centers; each
    given ``rank`` axes, to broadcast against the signal."""

    def center(k):
        return expand_to_rank(slice_axis(centers, axis, slice(k, k + 1)), rank)

    def gap(k):
        return center(k + 1) - center(k)

    count = centers.shape[axis]
    if count == 1:
        return center(0), 1.0, 1.0
    return center(index), gap(max(index - 1, 0)), gap(min(index, count - 2))


@keras.saving.register_keras_serializable(package="nnodely")
class MovingMembershipImpl(keras.layers.Layer):
    def __init__(self, axis, membership="Triangular", index=0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.membership = membership
        self.index = int(index)

    def call(self, inputs):
        x, centers = inputs
        center, left, right = _neighbors(centers, self.index, len(x.shape), self.axis)
        function = _MEMBERSHIPS[self.membership.lower()]
        return function._moving(x, center, left, right)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"axis": self.axis, "membership": self.membership, "index": self.index}
        )
        return config


class MovingMembership(Layer):
    """The ``membership`` of :class:`Fuzzify` on center ``index`` of a
    stream of centers, read with the signal as ``[x, centers]``: sized by
    the centers next to it as on fixed centers, but as they are computed, so
    that a gradient reaches them."""

    def __init__(self, membership: str = "Triangular", index: int = 0, name=None):
        self.membership = membership
        self.index = int(index)
        super().__init__(name=name, membership=membership, index=self.index)

    def output_shape(self, *inputs):
        return inputs[0].shape

    def build_layer(self):
        # The centers lie along their one dim axis.
        centers = self.preds[1].shape  # type: ignore[union-attr]
        return MovingMembershipImpl(
            axis=centers.dim_axes[0],
            membership=self.membership,
            index=self.index,
            name=self.name,
        )


class Fuzzify:
    """
    Membership degrees of a scalar signal to fuzzy sets placed on
    ``centers``: one feature per center, in ascending order of the centers.
    On a time window every step is evaluated alone, so a window of ``T``
    samples gives a ``(len(centers), T)`` matrix. It is typically the
    activation of a :class:`LocalModel`.

    ``centers`` are fixed numbers, or move with the model: one stream of
    shape ``(n,)`` - a :class:`Parameter` vector, which trains, or a stream
    computed from the inputs, such as a :class:`Fir` with ``features=n``,
    which gives every sample its own centers - or a list of scalar streams,
    numbers among them for centers kept fixed. Centers that move are used in
    the order given, which should be ascending: the memberships are placed
    on them and sized by the centers next to them as they are computed, so
    the gradient reaches them (a :class:`Rectangular` has none).

    ``function`` is the membership of every center, or a list of one per
    center (paired with ``centers`` as given, before they are sorted). Each is

    - :class:`Triangular`, :class:`Rectangular` or :class:`Gaussian`, the
      class or its name: placed on the center, its width set by the centers
      next to it;
    - any other nnodely layer class, made anew for each center, or a layer
      instance, shared by all of them: applied to ``x - center``;
    - a custom function, a Keras layer whose ``call`` reads one argument,
      ``x - center``: likewise, a ``keras.layers.Layer`` subclass made anew
      for each center or an instance shared by all, each wrapped in a
      :class:`KerasFunction` so the model still saves and exports.

    ::

        Fuzzify([0.0, 0.5, 1.0], function=Triangular)([x])
        Fuzzify([0.0, 0.5, 1.0], function=Sin)([x])           # sin(x - c)
        Fuzzify([0.0, 0.5, 1.0], function=Gaussian(sigma=0.1))       # one instance, of x - c
        Fuzzify([0.0, 0.5, 1.0], function=[Rectangular, Triangular, Gaussian])
        Fuzzify([0.0, 0.5, 1.0], function=MyKerasLayer)([x])
        Fuzzify(Parameter("c", value=[0.0, 0.5, 1.0]))([x])  # trained
        Fuzzify([0.0, Parameter("m", value=[0.5]), 1.0])([x])

    The result is a :class:`Concatenate` of the memberships, named after the
    Fuzzify; calling one Fuzzify again reuses the layers it made.
    """

    def __init__(
        self,
        centers,
        function: str
        | type[Layer]
        | Layer
        | type[keras.layers.Layer]
        | keras.layers.Layer
        | list = Triangular,
        name=None,
    ):
        self.name = next_name("Fuzzify") if name is None else name
        centers = _centers(centers, self.name)
        count = centers.shape.dim[0] if isinstance(centers, Stream) else len(centers)
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
        # A Keras layer instance is shared by the centers, as a layer is: it is
        # wrapped once, however many centers it is given to.
        wrappers: dict[int, KerasFunction] = {}
        for index, item in enumerate(functions):
            if isinstance(item, keras.layers.Layer):
                if id(item) not in wrappers:
                    wrappers[id(item)] = KerasFunction(
                        item, name=f"{self.name}_{type(item).__name__}"
                    )
                functions[index] = wrappers[id(item)]

        self.centers: list[float] | Stream
        if isinstance(centers, Stream):
            # Not sorted: their values are known only once computed.
            self.centers, self.functions = centers, functions
        else:
            order = sorted(range(len(centers)), key=lambda index: centers[index])
            self.centers = [centers[index] for index in order]
            self.functions = [functions[index] for index in order]
        # The layers of each center, made on the first call and shared by
        # every later one.
        self._layers: list[tuple[Layer, bool]] | None = None

    def _make_layers(self) -> list[tuple[Layer, bool]]:
        """For each center its layer, and whether it is placed on the center
        (or reads ``x - center``)."""
        layers = []
        for index, function in enumerate(self.functions):
            if isinstance(function, Layer):
                layers.append((function, False))
                continue
            # Named after its class; a lone layer is the result itself, so it
            # takes the name of the Fuzzify.
            name = (
                self.name
                if len(self.functions) == 1
                else f"{self.name}_{function.__name__}_{index}"
            )
            if issubclass(function, _Membership):
                if isinstance(self.centers, Stream):
                    layer = MovingMembership(function.__name__, index, name=name)
                else:
                    left, right = _neighbor_distances(self.centers)[index]
                    layer = function._placed(self.centers[index], left, right, name)
                layers.append((layer, True))
            elif issubclass(function, keras.layers.Layer):
                layers.append((KerasFunction(function, name=name), False))
            else:
                try:
                    layers.append((function(name=name), False))
                except TypeError as exc:
                    raise TypeError(
                        f"Fuzzify: {function.__name__} cannot be made without "
                        f"arguments; pass an instance, such as "
                        f"{function.__name__}(...), shared by every center."
                    ) from exc
        return layers

    def __call__(self, inputs) -> Stream:
        streams = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        if len(streams) != 1 or not isinstance(streams[0], Stream):
            raise TypeError(f"{self.name}: Fuzzify reads one stream, its signal.")
        x = streams[0]
        if tuple(x.shape.dim) != (1,):
            raise ValueError(
                f"{self.name}: Fuzzify reads a scalar signal, got dim {x.shape.dim}."
            )
        # A membership placed on centers that move reads them with the signal.
        placed_inputs: list[Stream] = [x]
        if isinstance(self.centers, Stream):
            outer = tuple(self.centers.shape.time[:-1])
            if outer not in ((), tuple(x.shape.time[:-1])):
                raise ValueError(
                    f"{self.name}: centers with outer time axes have the signal's, "
                    f"{tuple(x.shape.time[:-1])}; got {outer}."
                )
            placed_inputs = [x, self.centers]
        if self._layers is None:
            self._layers = self._make_layers()

        memberships = [
            layer(placed_inputs if placed else [x - center])
            for (layer, placed), center in zip(self._layers, self._each_center())
        ]
        if len(memberships) == 1:
            return memberships[0]
        return Concatenate(axis=0, name=self.name)(memberships)

    def _each_center(self):
        """Every center as a function reads it: a number, or a scalar
        selected from the stream of centers."""
        if not isinstance(self.centers, Stream):
            return list(self.centers)
        return [
            Select(idx=index, axis=0, name=f"{self.name}_center_{index}")(
                [self.centers]
            )
            for index in range(len(self.functions))
        ]


def _centers(centers, owner: str) -> list[float] | Stream:
    """``centers`` checked: fixed, a list of numbers; or moving, one stream
    of shape ``(n,)``, of one time step or none, given as it is or
    concatenated from a list of scalar streams and numbers."""
    if isinstance(centers, Stream):
        if centers.shape.dim_rank != 1 or not _one_step(centers):
            raise ValueError(
                f"{owner}: a stream of centers is a vector of shape (n,), one "
                f"sample of inner time or none; got dim {centers.shape.dim}, time "
                f"{centers.shape.time}."
            )
        return centers
    items = list(centers)
    if not items:
        raise ValueError("Fuzzify requires at least one center.")
    if not any(isinstance(item, Stream) for item in items):
        return [float(item) for item in items]

    streams = []
    for index, item in enumerate(items):
        if not isinstance(item, Stream):
            item = Constant(f"{owner}_center_{index}", value=[float(item)])
        elif tuple(item.shape.dim) != (1,) or not _one_step(item):
            raise ValueError(
                f"{owner}: in a list every center is a number or a scalar "
                f"stream, dim (1,) and one time step or none; center {index} has "
                f"dim {item.shape.dim}, time {item.shape.time}."
            )
        streams.append(item)
    if len(streams) == 1:
        return streams[0]
    return Concatenate(axis=0, name=f"{owner}_centers")(streams)


def _one_step(stream: Stream) -> bool:
    """Whether a stream holds one step of inner time; its outer time, if any,
    must be the signal's."""
    return stream.shape.inner_time == 1


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
