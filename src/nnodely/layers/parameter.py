import math
from numbers import Integral
from typing import Any, cast

import numpy as np
import keras

from nnodely.core.layer import Layer, _refuse_non_streams, _refuse_uncalled
from nnodely.core.shape import Shape
from nnodely.layers.input import _dimension, _positive_sizes
from nnodely.utils.printers import _table
from nnodely.utils.utils import _per_slice_initializer, _serialized_initializer


@keras.saving.register_keras_serializable(package="nnodely")
class ParameterImpl(keras.layers.Layer):
    """A value of fixed shape, the same for every sample of the batch.

    It is called with a tensor of the graph, of which it reads only the batch
    size: the value, held as ``value_shape``, comes out ``(batch,
    *value_shape)`` like every stream.
    """

    trainable_value = True

    def __init__(
        self,
        value_shape,
        value=None,
        initializer="random_normal",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.value_shape = tuple(int(axis) for axis in value_shape)
        self.value = (
            None
            if value is None
            else np.asarray(value, dtype=np.float32).reshape(self.value_shape)
        )
        self.initializer = initializer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_shape": self.value_shape,
                "value": None if self.value is None else self.value.tolist(),
                "initializer": self.initializer,
            }
        )
        return config

    def _initializer(self):
        """What the variable starts from: the value given, or else a draw of
        the initializer."""
        if self.value is None:
            return keras.initializers.get(self.initializer)
        return keras.initializers.Constant(value=self.value.tolist())

    def build(self, input_shape=None):
        self.variable = self.add_weight(
            name="value",
            shape=self.value_shape,
            initializer=self._initializer(),
            trainable=self.trainable_value,
            dtype="float32",
        )
        super().build(input_shape)

    def reinitialize(self):
        """Give the variable an initial value again, as it was first given
        one."""
        initializer = cast(keras.initializers.Initializer, self._initializer())
        self.variable.assign(initializer(tuple(self.variable.shape), dtype="float32"))

    def call(self, anchor):
        value = keras.ops.expand_dims(self.variable, 0)
        if anchor is None:
            return value
        batch = keras.ops.shape(anchor)[0]
        return keras.ops.broadcast_to(value, (batch, *self.value_shape))


@keras.saving.register_keras_serializable(package="nnodely")
class ConstantImpl(ParameterImpl):
    trainable_value = False


def _declared_shape(owner: str, dim, time, seq) -> Shape:
    """The shape ``dim``, ``time`` and ``seq`` declare, the sizes of the axes
    as for an Input: ``dim`` one feature by default, each axis at least 1;
    ``time`` the window, none by default; ``seq`` the sequence axes,
    outermost first, each of a fixed length - a value holds every step - and
    only with a window."""
    sequence = () if seq is None else _positive_sizes(seq)
    if sequence is None:
        raise ValueError(
            f"{owner}: seq is the length of each sequence axis, an int or a "
            f"tuple of them, outermost first, each at least 1 - a value holds "
            f"every step of its sequence, so none is dynamic; got {seq!r}."
        )
    if sequence and time is None:
        raise ValueError(
            f"{owner}: a sequence needs a window: declare time as well, time=1 "
            "for one sample per step."
        )
    if time is not None:
        window = _positive_sizes(time) if isinstance(time, Integral) else None
        if window is None:
            raise ValueError(
                f"{owner}: time is the length of the window, an int of at least "
                f"1, or None for no window; got {time!r}."
            )
        time = window[0]
    return Shape(dim=_dimension(owner, dim), time=time, seq=sequence)


def _take_axes(shape: tuple, sizes: tuple) -> tuple[int, ...] | None:
    """The axes of ``shape`` that have ``sizes``, in their order: each size
    takes the first axis of that size not taken yet, so axes of one size
    keep their order. None if a size finds none."""
    free, taken = list(range(len(shape))), []
    for size in sizes:
        axis = next((axis for axis in free if shape[axis] == size), None)
        if axis is None:
            return None
        free.remove(axis)
        taken.append(axis)
    return tuple(taken)


def _value_layout(owner: str, value, dim, time, seq) -> tuple[np.ndarray, Shape]:
    """``value`` laid out ``(*seq, time, *dim)``, as a tensor holds a sample,
    and the shape of its stream, declared as for an Input
    (:func:`_declared_shape`).

    Without ``dim``, ``seq`` and ``time`` take the axes of the value of their
    sizes, and the axes they leave are its dim, in their order: all of them
    when neither is given - like a number, the value then has no window and
    applies to every sample of any stream. A dim of one when they leave none.
    A value of one axis - a list of numbers - holds as many values as the
    shape, in its order, row-major. A value of several axes has the sizes the
    shape declares, in any order: its axes are moved into place, those of one
    size keeping their order.
    """
    array = np.atleast_1d(np.asarray(value, dtype=np.float32))
    if array.size == 0:
        raise ValueError(f"{owner}: value must not be empty.")
    shape = _declared_shape(owner, dim, time, seq)
    if dim is None:
        # A value's sequence has fixed lengths: none is None.
        sequence = cast(tuple[int, ...], shape.seq)
        outer = (*sequence, *(() if shape.time is None else (shape.time,)))
        taken = _take_axes(array.shape, outer)
        if taken is not None:
            rest = tuple(axis for axis in range(array.ndim) if axis not in taken)
            dims = tuple(array.shape[axis] for axis in rest) or (1,)
            array = np.transpose(array, (*taken, *rest)).reshape(*outer, *dims)
            return array, shape.replace(dim=dims)
        if array.ndim > 1:
            raise ValueError(
                f"{owner}: shape error, a value of shape {array.shape} has no "
                f"axes of the sizes seq and time declare, (*seq, time) = "
                f"{outer}; the axes they leave are its dim."
            )

    declared = shape.tuple
    if array.ndim == 1:
        if array.size != math.prod(declared):
            raise ValueError(
                f"{owner}: shape error, {array.size} values given where the "
                f"declared shape (*seq, time, *dim) = {declared} holds "
                f"{math.prod(declared)}."
            )
        return array.reshape(declared), shape
    order = _take_axes(array.shape, declared)
    if order is None or len(order) != array.ndim:
        raise ValueError(
            f"{owner}: shape error, a value of shape {array.shape} does not have "
            f"the sizes of the declared shape (*seq, time, *dim) = {declared}, "
            f"in any order. Give a value of those sizes, or the "
            f"{math.prod(declared)} values as a list."
        )
    return np.transpose(array, order), shape


class _Value(Layer):
    """What a Parameter and a Constant share: a value the model holds rather
    than reads from data, laid out and batched like any other stream.

    The two differ only in whether training changes the value.
    """

    _impl: type[ParameterImpl] = ParameterImpl
    _trainable = True
    # Not read from data: without a window it applies to any stream's.
    _is_value = True

    def __init__(self, name, value, initializer, dim, time, seq):
        owner = type(self).__name__ + ("" if name is None else f" {name!r}")
        if value is None:
            shape = _declared_shape(owner, dim, time, seq)
        else:
            value, shape = _value_layout(owner, value, dim, time, seq)

        self.value = value
        self._initializer = initializer

        super().__init__(
            name=name,
            time=shape.time,
            dim=shape.dim,
            seq=shape.seq,
            value=None if value is None else value.tolist(),
            initializer=initializer,
        )
        # The value as it is held: its sequence axes, its window if it has
        # one, then its dim - the shape of one sample of its stream.
        self._value_shape = self.shape.tuple

    def build_layer(self):
        return self._impl(
            value_shape=self._value_shape,
            value=self.value,
            initializer=self._initializer,
            trainable=self._trainable,
            name=self.name,
        )

    def get_config(self):
        # Read from no stream, a value keeps its shape in its config rather
        # than taking it from its inputs as other layers do: the sizes of its
        # axes, and the value laid out (*seq, time, *dim) as they declare it.
        config = super().get_config()
        config.update(self.shape.get_config())
        return config

    @property
    def _variable(self):
        return getattr(self._layer, "variable", None)

    @property
    def value_numpy(self):
        return keras.ops.convert_to_numpy(self._variable)


class Parameter(_Value):
    """
    Trainable symbolic parameter.

    ``seq``, ``time`` and ``dim`` declare its shape as for an
    :class:`~nnodely.Input`: ``dim`` the feature axes, one feature by
    default, each of at least one; ``time`` the window, ``T``; ``seq`` the
    sequence axes, outermost first, each of a fixed length, and only with a
    window. Without ``time`` it has no window, and applies to every sample of
    any stream. Without ``value``, ``initializer``, any Keras initializer,
    draws its initial value.

    ``value`` sets the initial value: a list of numbers holds as many as the
    shape, in its order (row-major); an array of several axes has the sizes
    of the shape, in any order - its axes are moved into place, those of one
    size keeping their order. Anything else raises ``ValueError``. Given a
    value and no ``dim``, ``seq`` and ``time`` take the axes of the value of
    their sizes, and the axes they leave are the dim, in their order - all of
    them when neither is given, one feature when they leave none.
    :attr:`value` and the variable hold it laid out ``(*seq, time, *dim)``,
    without the window when it has none::

        Parameter("p", value=np.ones((2, 3, 2)))                   # S=(), T=None, D=(2, 3, 2)
        Parameter("p", value=np.ones((2, 4, 6)), time=4)           # S=(), T=4, D=(2, 6)
        Parameter("p", value=[1.0, 2.0, 3.0], time=3)              # S=(), T=3, D=(1,)
        Parameter("p", value=np.ones((3, 4, 2)), seq=2, time=3, dim=4)  # S=(2,), T=3, D=(4,)
        Parameter("p", dim=(3, 2), time=4, initializer="ones")     # S=(), T=4, D=(3, 2)

    ``trainable=False`` keeps the value as it is through training, as
    :meth:`freeze` does, before or after the model is built; the flag is
    saved with the model.

    Like every stream it is laid out with a batch axis first, the same value
    for every sample::

        (batch, *seq, [time,] *dim)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        value=None,
        initializer="random_normal",
        time=None,
        dim=None,
        seq=None,
        trainable: bool = True,
    ):
        self._trainable = bool(trainable)
        super().__init__(name, value, initializer, dim, time, seq)

    @property
    def trainable(self) -> bool:
        """Whether training changes the value. Set before the model is built
        or after: the next training follows it. A Parameter shared by several
        models is one value, frozen or trained for all of them."""
        return self._trainable

    @trainable.setter
    def trainable(self, trainable: bool):
        self._trainable = bool(trainable)
        if self._layer is not None:
            self._layer.trainable = self._trainable

    def freeze(self):
        """Keep the value as it is through training; :meth:`unfreeze` undoes
        it. Returns the Parameter."""
        self.trainable = False
        return self

    def unfreeze(self):
        """Let training change the value again. Returns the Parameter."""
        self.trainable = True
        return self

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initializer": _serialized_initializer(self._initializer),
                "trainable": self._trainable,
            }
        )
        return config

    @property
    def param(self):
        return self._variable

    def _reinitialize(self) -> None:
        if self._layer is not None:
            self._layer.reinitialize()


class Parameters(list):
    """The Parameters of a model, as :attr:`Modely.parameters` lists them: a
    list of the Parameter objects, printed as a table of their names, shapes,
    numbers of values and whether training changes them - a section per model
    when the models it calls have Parameters of their own - and a last row
    totalling the values, all and trainable.

    ``groups`` pairs the name of each model with its Parameters, the model
    itself first."""

    def __init__(self, groups: list[tuple[str, list]], title: str = ""):
        super().__init__(parameter for _, group in groups for parameter in group)
        self._groups = groups
        self._title = title

    def __getitem__(self, key):  # type: ignore[override]
        """A Parameter by its name, or by position as in a list."""
        if isinstance(key, str):
            for parameter in self:
                if parameter.name == key:
                    return parameter
            raise KeyError(
                f"{self._title or 'Parameters'}: no Parameter named {key!r}; "
                f"the names are {[parameter.name for parameter in self]}."
            )
        return super().__getitem__(key)

    def __str__(self):
        header = ("parameter", "seq", "time", "dim", "values", "trainable")

        def row(parameter):
            return (
                parameter.name,
                str(tuple(parameter.shape.seq)),
                str(parameter.shape.time),
                str(tuple(parameter.shape.dim)),
                str(math.prod(parameter.shape.tuple)),
                "yes" if parameter.trainable else "no",
            )

        # A section per model only when the models it calls have some.
        sections = [
            (name, [row(parameter) for parameter in group])
            for name, group in self._groups
            if group
        ]
        if not any(group for _, group in self._groups[1:]):
            sections = [("", [row(parameter) for parameter in self])]
        values = [math.prod(p.shape.tuple) for p in self]
        trainable = sum(v for v, p in zip(values, self) if p.trainable)
        total = ("total", "", "", "", str(sum(values)), str(trainable))
        return _table(header, sections, total)

    __repr__ = __str__


class Constant(_Value):
    """
    Non-trainable symbolic constant layer.

    A :class:`Parameter` that training does not change, and that requires a
    ``value``: ``seq``, ``time`` and ``dim`` declare its shape, and the value
    fills it, as for a Parameter - without ``dim``, the axes ``seq`` and
    ``time`` leave are the dim::

        Constant("c", value=[1.0, 2.0, 3.0])                 # S=(), T=None, D=(3,)
        Constant("c", value=[1.0, 2.0, 3.0], time=3)         # S=(), T=3, D=(1,)
        Constant("c", value=[[1.0, 2.0, 3.0]], time=1, dim=3)  # S=(), T=1, D=(3,)

    Numbers used in arithmetic with a stream become constants automatically.

    Like every stream it is laid out with a batch axis first, the same value
    for every sample::

        (batch, *seq, [time,] *dim)
    """

    _impl = ConstantImpl
    _trainable = False

    def __init__(
        self, name: str | None = None, *, value, time=None, dim=None, seq=None
    ):
        if value is None:
            raise ValueError("Constant requires a value.")
        super().__init__(name, value, None, dim, time, seq)

    @property
    def constant(self):
        return self._variable

    def get_config(self):
        # A Constant is its value: it has no initializer.
        config = super().get_config()
        del config["initializer"]
        return config


@keras.saving.register_keras_serializable(package="nnodely")
class WeightImpl(ParameterImpl):
    """A layer's weight: a variable of the weight's own shape.

    The layer reads it whole, not one value per sample, so it comes out with
    a batch axis of one: ``(1, *weight_shape)``. The initializer fills it
    with draws of ``draw_shape``.
    """

    def __init__(self, value_shape, weight_shape, draw_shape, name=None, **kwargs):
        super().__init__(value_shape=value_shape, name=name, **kwargs)
        self.weight_shape = tuple(int(axis) for axis in weight_shape)
        self.draw_shape = tuple(int(axis) for axis in draw_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"weight_shape": self.weight_shape, "draw_shape": self.draw_shape}
        )
        return config

    def _initializer(self):
        return _per_slice_initializer(self.initializer, self.draw_shape)

    def build(self, input_shape=None):
        self.variable = self.add_weight(
            name="value",
            shape=self.weight_shape,
            initializer=self._initializer(),
            trainable=True,
            dtype="float32",
        )
        keras.layers.Layer.build(self, input_shape)

    def call(self, anchor):
        return keras.ops.expand_dims(
            keras.ops.reshape(self.variable, self.value_shape), axis=0
        )


class Weight(Parameter):
    """A layer's kernel or bias held as a Parameter.

    A layer given an initializer rather than a Parameter makes one of these
    when it is first called. Its dim is the shape of the weight, and its
    variable has exactly that shape. ``initializer`` fills it with draws of
    shape ``draw``: by default the whole weight, read as a matrix of its last
    axis by all the others, as a dense layer reads its kernel. The layer
    reads it whole rather than one value per sample, so its stream has a
    batch axis of one.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        shape,
        initializer="glorot_uniform",
        draw=None,
        trainable: bool = True,
    ):
        self._weight_shape = tuple(int(axis) for axis in shape)
        if draw is None:
            draw = (
                self._weight_shape
                if len(self._weight_shape) <= 2
                else (math.prod(self._weight_shape[:-1]), self._weight_shape[-1])
            )
        self._draw_shape = tuple(int(axis) for axis in draw)
        super().__init__(
            name,
            initializer=initializer,
            dim=self._weight_shape,
            trainable=trainable,
        )

    def build_layer(self):
        return WeightImpl(
            value_shape=self.shape.tuple,
            weight_shape=self._weight_shape,
            draw_shape=self._draw_shape,
            initializer=self._initializer,
            trainable=self._trainable,
            name=self.name,
        )

    def get_config(self):
        # Declared by its shape and the shape its initializer draws, which
        # take the place of a Parameter's value and axes.
        config = super().get_config()
        for key in ("value", "dim", "time", "seq"):
            del config[key]
        config.update({"shape": self._weight_shape, "draw": self._draw_shape})
        return config


class _ParameterWeights(Layer):
    """A layer computing with weights held by Parameters, its predecessors
    after its input.

    The weights are named by ``_labels``, in the order its Keras layer reads
    them: ``[x, *weights]``. ``weights`` sets each one: a Parameter, used as
    given, or an initializer, by name or object, from which the layer makes
    a :class:`Weight` of the shape its input asks for, the first time it is
    called, and shares it with every later call. ``True`` takes the layer's
    default initializer for it, ``_defaults``, and ``False`` leaves it out.

    A subclass gives the shapes of its weights for an input with
    :meth:`_weight_shapes`, and the shapes their initializers draw with
    :meth:`_draw_shapes`, one for each label.
    """

    _labels: tuple[str, ...] = ("kernel", "bias")
    _defaults: dict = {"kernel": "glorot_uniform", "bias": "zeros"}

    def __init__(self, name, weights: dict, **properties):
        self._settings = dict(weights)
        # Made, or taken as given, on the first call.
        self._weights: dict[str, Parameter] | None = None
        super().__init__(name=name, **properties)

    def _weight_shapes(self, x) -> tuple[tuple[int, ...], ...]:
        raise NotImplementedError

    def _draw_shapes(self, shapes: tuple) -> tuple:
        """The shapes the initializers of weights of ``shapes`` draw, one for
        each; None for the Weight's own default."""
        return (None,) * len(shapes)

    def _parameters(self, x) -> list[Parameter]:
        """The weights for the input ``x``, made on the first call. Each must
        have the shape the input asks for, axes of size 1 aside: they leave
        the layout of the values as it is."""
        shapes = dict(zip(self._labels, self._weight_shapes(x)))
        if self._weights is None:
            draws = dict(zip(self._labels, self._draw_shapes(tuple(shapes.values()))))
            self._weights = {
                label: self._weight(
                    label, self._settings[label], shapes[label], draws[label]
                )
                for label in self._labels
                if self._settings[label] is not False
            }
        for label, weight in self._weights.items():
            if _without_ones(weight.shape.tuple) != _without_ones(shapes[label]):
                raise ValueError(
                    f"{self.name}: {label} must have the shape "
                    f"{tuple(shapes[label])}, axes of size 1 aside, got "
                    f"{weight.shape.tuple}."
                )
        return list(self._weights.values())

    def _weight(self, label, setting, shape, draw) -> Parameter:
        if isinstance(setting, Parameter):
            return setting
        if setting is True:
            setting = self._defaults[label]
        return Weight(
            f"{self.name}_{label}", shape=shape, initializer=setting, draw=draw
        )

    def __call__(self, inputs) -> Any:
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # Before the weights are sized on it.
        _refuse_non_streams(self, inputs)
        if self._one_stream and len(inputs) > 1:
            # Applied to each stream, with the same weights.
            return [self([x]) for x in inputs]
        # The weights are predecessors like the input, but configured on the
        # layer, so they are appended here and reconnected when reloading.
        # They take the place of any further stream, which would otherwise be
        # dropped without a word.
        if len(inputs) > 1:
            raise ValueError(
                f"{self.name}: {type(self).__name__} reads one stream, got "
                f"{len(inputs)}. Apply one to each, or join them first, "
                "with Concatenate or TimeConcatenate."
            )
        # Before the weights are sized on it.
        _refuse_uncalled(inputs)
        if not inputs:
            raise ValueError(
                f"{self.name}: {type(self).__name__} reads one stream, got "
                "none. Apply it to a stream first."
            )
        return self._apply([inputs[0], *self._parameters(inputs[0])])

    def get_config(self):
        # The weights are saved with the graph, as the predecessors they are:
        # an argument that sets one only says whether it is there.
        config = super().get_config()
        for label in self._labels:
            if label in config:
                config[label] = config[label] is not False
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        preds = preds or []
        layer = super().from_config(config)
        # The weights it has come back as its predecessors after its input,
        # in the order of their labels.
        weights = iter(preds[1:])
        for label in cls._labels:
            if layer._settings[label] is not False:
                layer._settings[label] = next(weights)
        return layer(preds[:1]) if preds else layer

    def _weight_nodes(self) -> list[Parameter]:
        return cast(list[Parameter], self._preds[1:]) if self._preds else []

    def _weight_value(self, label: str):
        """The variable of the Parameter holding the weight ``label``, in
        that Parameter's shape; None without it or before the model is
        built."""
        present = [name for name in self._labels if self._settings[name] is not False]
        weight = dict(zip(present, self._weight_nodes())).get(label)
        return None if weight is None else weight.param

    @property
    def kernel(self):
        """The variable of the Parameter holding the kernel, in that
        Parameter's shape; None before the model is built."""
        return self._weight_value("kernel")

    @property
    def bias(self):
        """The variable of the Parameter holding the bias, in that
        Parameter's shape; None without a bias or before the model is
        built."""
        return self._weight_value("bias")


def _without_ones(shape):
    """``shape`` without its axes of size 1."""
    return tuple(axis for axis in shape if axis != 1)


def _check_weight_size(layer_name, label, weight_shape, expected):
    """Raise unless a weight tensor ``(batch, ...)`` holds the values of a
    weight of shape ``expected``."""
    held = math.prod(int(axis) for axis in weight_shape[1:])
    if held != math.prod(expected):
        raise ValueError(
            f"{layer_name}: its {label} holds {held} values where the input asks "
            f"for {tuple(expected)}. The weights of a layer fit the input shape "
            "they were made for."
        )
