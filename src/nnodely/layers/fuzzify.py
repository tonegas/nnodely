"""Fuzzify and the membership functions it places on its centers."""

from __future__ import annotations

import keras

from nnodely.core.dag import next_name
from nnodely.core.layer import Layer
from nnodely.core.stream import Stream
from nnodely.layers.function import KerasFunction, _is_keras_layer
from nnodely.layers.time_ops import Concatenate

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


@keras.saving.register_keras_serializable(package="nnodely")
class TriangularImpl(keras.layers.Layer):
    def __init__(self, center=0.0, left=1.0, right=1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.center = float(center)
        self.left = float(left)
        self.right = float(right)

    def call(self, inputs):
        rise = (inputs - (self.center - self.left)) / max(self.left, _EPS)
        fall = ((self.center + self.right) - inputs) / max(self.right, _EPS)
        return keras.ops.maximum(keras.ops.minimum(rise, fall), 0.0)

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
        inside = keras.ops.logical_and(
            inputs > self.center - self.left, inputs < self.center + self.right
        )
        return keras.ops.cast(inside, inputs.dtype)

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
        scaled = (inputs - self.center) / max(self.sigma, _EPS)
        return keras.ops.exp(-0.5 * scaled * scaled)

    def get_config(self):
        config = super().get_config()
        config.update({"center": self.center, "sigma": self.sigma})
        return config


class _Membership(Layer):
    """A membership function: applied to every value of a stream, it keeps
    the stream's shape. :class:`Fuzzify` places one on each of its centers,
    sized by the centers next to it, with :meth:`_placed`."""

    def output_shape(self, *inputs):
        # Elementwise: every axis kept, a dynamic sequence axis included.
        return inputs[0].shape.dimensions

    @classmethod
    def _placed(cls, center, left, right, name):
        """The membership on ``center``, whose neighbors are ``left`` and
        ``right`` away."""
        raise NotImplementedError


class Triangular(_Membership):
    """
    Triangular membership: 1 at ``center``, falling linearly to 0 at
    ``radius`` from it, and 0 beyond. ``radius`` is a number, or a
    ``(left, right)`` pair for a triangle leaning to one side::

        Triangular(center=0.5, radius=0.5)([x.last()])   # 0 at 0 and at 1
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


class Rectangular(_Membership):
    """
    Rectangular membership: 1 strictly within ``radius`` of ``center``, 0
    elsewhere. ``radius`` is a number, or a ``(left, right)`` pair::

        Rectangular(center=0.5, radius=0.25)([x.last()])   # 1 on (0.25, 0.75)

    In a :class:`Fuzzify`, each rectangle reaches halfway to the next center.
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
        return cls(center=center, radius=right / 2.0, name=name)


class Gaussian(_Membership):
    """
    Gaussian membership: ``exp(-((x - center) / sigma)^2 / 2)``, 1 at
    ``center``::

        Gaussian(center=0.5, sigma=0.1)([x.last()])

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


class Fuzzify:
    """
    Membership degrees of a scalar signal to fuzzy sets placed on
    ``centers``: one feature per center, in ascending order of the centers.
    On a time window every step is evaluated alone, so a window of ``T``
    samples gives a ``(len(centers), T)`` matrix. It is typically the
    activation of a :class:`LocalModel`.

    ``function`` is the membership of every center, or a list of one per
    center (paired with ``centers`` as given, before they are sorted). Each is

    - :class:`Triangular`, :class:`Rectangular` or :class:`Gaussian`, the
      class or its name: placed on the center, its width set by the centers
      next to it;
    - any other nnodely layer class, made anew for each center, or a layer
      instance, shared by all of them: applied to ``x - center``;
    - a custom function, a Keras layer: likewise, a ``keras.layers.Layer``
      subclass made anew for each center or an instance shared by all, each
      wrapped in a :class:`KerasFunction` so the model still saves and
      exports.

    Called on further streams after the signal, ``Fuzzify(...)([x, y])``, it
    passes them to the functions after ``x - center``: a Keras layer with
    ``call(self, x, y)`` gives ``f(x - center, y)`` on every center.

    ::

        Fuzzify([0.0, 0.5, 1.0], function=Triangular)([x.last()])
        Fuzzify([0.0, 0.5, 1.0], function=Sin)([x.last()])           # sin(x - c)
        Fuzzify([0.0, 0.5, 1.0], function=Gaussian(sigma=0.1))       # x - c, alike
        Fuzzify([0.0, 0.5, 1.0], function=[Rectangular, Triangular, Gaussian])
        Fuzzify([0.0, 0.5, 1.0], function=MyKerasLayer)([x.last(), y.last()])

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
        centers = [float(center) for center in centers]
        if not centers:
            raise ValueError("Fuzzify requires at least one center.")
        functions = (
            list(function)
            if isinstance(function, (list, tuple))
            else [function] * len(centers)
        )
        if len(functions) != len(centers):
            raise ValueError(
                f"Fuzzify: {len(functions)} functions for {len(centers)} centers; "
                "give one function, or one per center."
            )
        self.name = next_name("Fuzzify") if name is None else name
        functions = [_membership_function(item) for item in functions]
        # A Keras layer instance is shared by the centers, as a layer is.
        functions = [
            KerasFunction(item, name=f"{self.name}_function")
            if isinstance(item, keras.layers.Layer)
            else item
            for item in functions
        ]

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
        distances = _neighbor_distances(self.centers)
        for index, (center, function, (left, right)) in enumerate(
            zip(self.centers, self.functions, distances)
        ):
            # A lone membership is the result itself, so it takes the name.
            name = self.name if len(self.centers) == 1 else f"{self.name}_{index}"
            if isinstance(function, Layer):
                layers.append((function, False))
            elif issubclass(function, _Membership):
                layers.append((function._placed(center, left, right, name), True))
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
        if not streams or not all(isinstance(stream, Stream) for stream in streams):
            raise TypeError(
                f"{self.name}: Fuzzify reads its signal, and then any further "
                "streams for its functions."
            )
        x, extra = streams[0], streams[1:]
        if tuple(x.shape.dim) != (1,):
            raise ValueError(
                f"{self.name}: Fuzzify reads a scalar signal, got dim {x.shape.dim}."
            )
        if self._layers is None:
            self._layers = self._make_layers()
        if extra and any(placed for _, placed in self._layers):
            raise ValueError(
                f"{self.name}: a membership function reads the signal alone; "
                "further streams are for the other functions."
            )

        memberships = [
            layer([x] if placed else [x - center, *extra])
            for (layer, placed), center in zip(self._layers, self.centers)
        ]
        if len(memberships) == 1:
            return memberships[0]
        return Concatenate(axis=0, name=self.name)(memberships)


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
    if (
        isinstance(function, Layer)
        or (isinstance(function, type) and issubclass(function, Layer))
        or _is_keras_layer(function)
    ):
        return function
    raise TypeError(
        "Fuzzify: a function is an nnodely or Keras layer class or instance, "
        f"or a membership name, got {function!r}."
    )
