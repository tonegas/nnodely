import math
from typing import Any, cast

import numpy as np
import keras

from nnodely.core.layer import Layer, _refuse_non_streams, _refuse_uncalled
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


def _value_layout(value, dim, time, seq, owner: str):
    """``value`` laid out ``(*seq, time, *dim)``, as a tensor holds a sample,
    with the sizes of its dim axes, of its window - None when it has none of
    its own - and of its sequence axes.

    Without ``dim``, ``time`` and ``seq``, every axis of ``value`` is a dim
    axis and the value has no window: like a number, it applies to every
    sample of any stream.
    Otherwise they name the axes of ``value`` that play each part: ``seq`` an
    axis or a tuple of axes, outermost first, ``time`` one axis, the window.
    The axes they leave are the dim axes, unless ``dim`` names them; a value
    with none keeps a dim of one.
    """
    array = np.atleast_1d(np.asarray(value, dtype=np.float32))
    if array.size == 0:
        raise ValueError(f"{owner}: value must not be empty.")
    if time is not None and not isinstance(time, (int, np.integer)):
        raise TypeError(
            f"{owner}: time is one axis of the value, its window, got {time!r}; "
            "the axes outside the window are seq."
        )

    def axes(spec, label):
        items = (
            ()
            if spec is None
            else (spec,)
            if isinstance(spec, (int, np.integer))
            else tuple(spec)
        )
        for axis in items:
            if not -array.ndim <= axis < array.ndim:
                raise ValueError(
                    f"{owner}: {label} axis {axis} is not an axis of a value of "
                    f"shape {array.shape}."
                )
        return tuple(int(axis) % array.ndim for axis in items)

    seq_axes, time_axes = axes(seq, "seq"), axes(time, "time")
    taken = seq_axes + time_axes
    dim_axes = (
        axes(dim, "dim")
        if dim is not None
        else tuple(axis for axis in range(array.ndim) if axis not in taken)
    )
    order = taken + dim_axes
    if sorted(order) != list(range(array.ndim)):
        raise ValueError(
            f"{owner}: dim, time and seq must take each axis of a value of shape "
            f"{array.shape} once, got dim={dim_axes}, time={time_axes}, "
            f"seq={seq_axes}."
        )

    array = np.transpose(array, order)
    if not dim_axes:
        array = array[..., np.newaxis]
    sizes, seq_rank = array.shape, len(seq_axes)
    window = sizes[seq_rank] if time_axes else None
    return array, sizes[len(taken) :], window, sizes[:seq_rank]


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
        if value is not None:
            value, dim, time, seq = _value_layout(
                value, dim, time, seq, self.__class__.__name__
            )

        self.value = value
        self._initializer = initializer

        super().__init__(
            name=name,
            time=time,
            dim=dim,
            seq=seq,
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
        # than taking it from its inputs as other layers do. A value is saved
        # laid out (*seq, time, *dim), with the axes that play each part.
        config = super().get_config()
        config.update(self.shape.get_config())
        seq_rank = self.shape.seq_rank
        has_time = self.shape.time is not None
        if self.value is not None:
            first_dim = seq_rank + int(has_time)
            config.update(
                {
                    "seq": list(range(seq_rank)),
                    "time": seq_rank if has_time else None,
                    "dim": list(range(first_dim, len(self._value_shape))),
                }
            )
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

    Without ``value``, ``dim``, ``time`` and ``seq`` are the sizes of its
    axes - ``seq`` outermost first - and ``initializer``, any Keras
    initializer, draws its initial value. Without ``time`` it has no window,
    and applies to every sample of any stream.

    ``value`` sets the shape and the initial value. Every axis of the value is
    a dim axis, and the value has no window, unless ``dim``, ``time`` and
    ``seq`` say otherwise: given with a value they are axes of the value -
    ``seq`` an axis or a tuple of axes, outermost first, ``time`` one axis -
    and the axes they leave are the dim axes. :attr:`value` and the variable
    hold the value laid out ``(*seq, time, *dim)``, without the window when
    none is given::

        Parameter("p", value=np.ones((2, 3, 2)))                     # S=(), T=None, D=(2, 3, 2)
        Parameter("p", value=np.ones((2, 3, 2)), dim=0, time=1, seq=2)  # S=(2,), T=3, D=(2,)
        Parameter("p", value=[[1.0, 2.0, 3.0]], time=1)               # S=(), T=3, D=(1,)

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
    numbers of values and whether training changes them."""

    def __init__(self, parameters, title: str = ""):
        super().__init__(parameters)
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
        header = ("name", "seq", "time", "dim", "values", "trainable")
        rows = [
            (
                parameter.name,
                str(tuple(parameter.shape.seq)),
                str(parameter.shape.time),
                str(tuple(parameter.shape.dim)),
                str(math.prod(parameter.shape.tuple)),
                "yes" if parameter.trainable else "no",
            )
            for parameter in self
        ]
        widths = [
            max(len(row[column]) for row in (header, *rows))
            for column in range(len(header))
        ]

        def line(row):
            return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

        values = [int(row[4]) for row in rows]
        trainable = sum(v for v, p in zip(values, self) if p.trainable)
        return "\n".join(
            [
                *([self._title] if self._title else []),
                line(header).rstrip(),
                line(tuple("-" * width for width in widths)).rstrip(),
                *(line(row).rstrip() for row in rows),
                f"{len(self)} parameters, {sum(values)} values, {trainable} trainable",
            ]
        )

    __repr__ = __str__


class Constant(_Value):
    """
    Non-trainable symbolic constant layer.

    ``value`` sets its shape as for a :class:`Parameter`: every axis of the
    value is a dim axis, and the value has no window, unless ``dim``,
    ``time`` and ``seq`` name the axes of the value that play each part::

        Constant("c", value=[1.0, 2.0, 3.0])                 # S=(), T=None, D=(3,)
        Constant("c", value=[[1.0, 2.0, 3.0]], time=1)       # S=(), T=3, D=(1,)

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
