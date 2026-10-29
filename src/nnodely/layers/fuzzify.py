"""Fuzzify, and the membership functions it places on its centers."""

from __future__ import annotations

import inspect
from typing import Any, cast

import keras
import numpy as np

from nnodely.core.dag import next_name
from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank
from nnodely.core.stream import NODE_REGISTRY, Stream
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
        if streams and all(isinstance(stream, Stream) for stream in streams):
            # Called on several streams, it is applied to each of them.
            if len(streams) > 1:
                return [self([stream]) for stream in streams]
            if isinstance(self._width, Stream):
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
    """
    Triangular membership: 1 at 0, falling linearly to 0 at ``radius`` from
    it, and 0 beyond. ``radius`` is a number, a ``(left, right)`` pair for a
    triangle leaning to one side, or a :class:`Parameter` of dim ``(1,)`` or
    ``(2,)``, which trains the width with the model. To center it, shift
    the signal::

        Triangular(radius=0.5)([x - 0.5])          # 0 at 0 and at 1
        Triangular(radius=(1.0, 2.0))([x - 1.0])   # from 0, up at 1, to 3
        Triangular(radius=Parameter("r", value=[0.5]))([x])

    Given to :class:`Fuzzify` as a class, a triangle on each center reaches
    the centers next to it, so between two centers the memberships add up to
    one.
    """

    def __init__(self, radius: float | tuple[float, float] | Stream = 1.0, name=None):
        super().__init__(radius, name=name)


class Rectangular(_Bump):
    """
    Rectangular membership: 1 on ``[-left, right)``, 0 elsewhere. ``radius``
    is a number for both sides, a ``(left, right)`` pair, or a
    :class:`Parameter` of dim ``(1,)`` or ``(2,)``::

        Rectangular(radius=0.25)([x - 0.5])   # 1 on [0.25, 0.75)

    A step has no gradient: a Parameter read by rectangles alone does not
    train, one used elsewhere too trains there, and the rectangles follow.

    Given to :class:`Fuzzify` as a class, a rectangle on each center reaches
    halfway to the centers next to it, so between the centers exactly one
    set holds every value.
    """

    _formula = "box"

    def __init__(self, radius: float | tuple[float, float] | Stream = 1.0, name=None):
        super().__init__(radius, name=name)


class Gaussian(_Bump):
    """
    Gaussian membership: ``exp(-(x / sigma)^2 / 2)``, 1 at 0. ``sigma`` is a
    positive number, or a :class:`Parameter` of dim ``(1,)``::

        Gaussian(sigma=0.1)([x - 0.5])

    Given to :class:`Fuzzify` as a class, ``sigma`` is a sixth of the span
    between the centers next to it, so three sigmas reach each of them.
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
            or width.shape.time is not None
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
    """
    Membership degrees of a scalar signal to fuzzy sets placed on
    ``centers``: one feature per center, in ascending order of the centers.
    On a time window every step is evaluated alone, so a window of ``T``
    samples gives ``T`` rows of ``len(centers)`` memberships. It is typically
    the activation of a :class:`LocalModel`.

    ``centers`` are fixed numbers, or move with the model: one stream of
    shape ``(n,)`` - a :class:`Parameter` vector, which trains, or a stream
    computed from the inputs, such as a :class:`Fir` with ``features=n``,
    which gives every sample its own centers - or a list of scalar streams,
    numbers among them for centers kept fixed. Centers that move are used in
    the order given, which should be ascending.

    ``function`` is the function of every center, or a list of one per
    center (paired with ``centers`` as given, before they are sorted). Each is

    - :class:`Triangular`, :class:`Rectangular` or :class:`Gaussian` as a
      class, or its name: placed on the center and sized by the centers next
      to it, as they are computed - centers that move take their sets with
      them, widths included, and the gradient reaches them through both;
    - any other nnodely layer class, made anew for each center, or a layer
      instance, shared by all of them, ``Triangular(radius=...)`` among
      them: applied to ``x - center``;
    - a custom function, a Keras layer whose ``call`` reads one argument,
      ``x - center``: likewise, a ``keras.layers.Layer`` subclass made anew
      for each center or an instance shared by all, each wrapped in a
      :class:`KerasFunction` so the model still saves and exports.

    ::

        Fuzzify([0.0, 0.5, 1.0], function=Triangular)([x])
        Fuzzify([0.0, 0.5, 1.0], function=Sin)([x])               # sin(x - c)
        Fuzzify([0.0, 0.5, 1.0], function=Gaussian(sigma=0.1))([x])
        Fuzzify([0.0, 0.5, 1.0], function=[Rectangular, Triangular, Gaussian])
        Fuzzify([0.0, 0.5, 1.0], function=MyKerasLayer)([x])
        Fuzzify(Parameter("c", value=[0.0, 0.5, 1.0]), function=Triangular)([x])
        Fuzzify([0.0, Parameter("m", value=[0.5]), 1.0], function=Triangular)([x])

    ``bounds`` holds the signal within ``(left, right)`` before the functions
    read it: below ``left`` it is read as ``left``, above ``right`` as
    ``right``. ``True`` takes the lowest and the highest center (centers that
    move as they are computed), so a signal beyond the centers belongs to the
    outer set rather than to none; ``False``, the default, leaves it as it is.

    ``normalize=True`` divides the memberships by their sum, so that they add
    up to one; memberships that are all zero stay zero. It is meant for
    memberships that are not negative. ``False``, the default, keeps the
    values of the functions.

    A Fuzzify is one layer, computing every membership at once. The layers
    it applies to ``x - center`` are computed inside it: the Parameters they
    read are predecessors of the Fuzzify. Calling it again applies the same
    layers.
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
        self.bounds = _bounds(bounds, name)
        self.normalize = bool(normalize)
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
        weights = [weight for probe in self._probes for weight in probe.preds[1:]]
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
            shape.dim_rank != 1 or (shape.seq, shape.time) != (first.seq, first.time)
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
            weights=[len(probe.preds) - 1 for probe in self._probes],
            bounds=self.bounds,
            normalize=self.normalize,
            name=self.name,
        )

    def get_config(self):
        # The layers applied to x - center are saved inside, each once, with
        # how many of the predecessors are the weights it reads.
        return {
            "name": self.name,
            "centers": self._centers,
            "function": self._function_specs(),
            "layers": [
                {
                    "class_name": type(layer).__name__,
                    "config": layer.get_config(),
                    "weights": len(probe.preds) - 1,
                }
                for layer, probe in zip(self._applied, self._probes)
            ],
            "bounds": list(self.bounds)
            if isinstance(self.bounds, tuple)
            else self.bounds,
            "normalize": self.normalize,
        }

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
            layer_class = cast(type[Layer], NODE_REGISTRY[saved["class_name"]])
            node = layer_class.from_config(saved["config"], preds=[x, *weights])
            layers.append(node._source)
        bounds = config["bounds"]
        layer = cls(
            centers=centers,
            function=[
                layers[item] if isinstance(item, int) else item
                for item in config["function"]
            ],
            bounds=tuple(bounds) if isinstance(bounds, list) else bounds,
            normalize=config["normalize"],
            name=config["name"],
        )
        return layer(x)

    def plot(self, x_range: tuple[float, float] | None = None, points=401, ax=None):
        """Draw the memberships with matplotlib, as the layer computes them
        now: centers and widths that are Parameters at their current value,
        the trained one after training. ``x_range`` is the span of the signal
        drawn, by default the centers and the outer gap beyond each end, where
        the outer sets placed on them end. Draws on ``ax``, or on a new
        figure, and returns the axes. Centers computed from the inputs have
        no value of their own to draw.
        """
        import matplotlib.pyplot as plt

        if self._layers is None:
            # Not applied yet: applied to a signal of its own, as a model would.
            self([Input(f"{self.name}_plot")])
        streams = [
            *self._center_streams,
            *(weight for probe in self._probes for weight in probe.preds[1:]),
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
        signal = self.preds[0].name if self.preds else "signal"
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
                f"sample; got dim {centers.shape.dim}, time {centers.shape.time}."
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
                f"dim {item.shape.dim}, time {item.shape.time}."
            )
        listed.append(None)
        streams.append(item)
    return listed, streams


def _one_step(stream: Stream) -> bool:
    """Whether a stream holds one sample - a window of one, or none; its
    sequence, if any, must be the signal's."""
    return stream.shape.time in (None, 1)


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
