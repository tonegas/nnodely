import math
from typing import cast

import numpy as np
import keras

from nnodely.core.layer import Layer
from nnodely.core.stream import Stream
from nnodely.utils.utils import _per_slice_initializer, _serialized_initializer


@keras.saving.register_keras_serializable(package="nnodely")
class ParameterImpl(keras.layers.Layer):
    """A value of fixed shape, the same for every sample of the batch.

    It is called with a tensor of the graph, of which it reads only the batch
    size: the value comes out ``(batch, *dim, time, *seq)`` like every stream.
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

    def build(self, input_shape=None):
        initializer = (
            keras.initializers.get(self.initializer)
            if self.value is None
            else keras.initializers.Constant(value=self.value.tolist())
        )

        self.variable = self.add_weight(
            name="value",
            shape=self.value_shape,
            initializer=initializer,
            trainable=self.trainable_value,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, anchor):
        value = keras.ops.expand_dims(self.variable, axis=0)
        if anchor is None:
            return value
        batch = keras.ops.shape(anchor)[0]
        return keras.ops.broadcast_to(value, (batch, *self.value_shape))


@keras.saving.register_keras_serializable(package="nnodely")
class ConstantImpl(ParameterImpl):
    trainable_value = False


def _value_layout(value, dim, time, seq, owner: str):
    """``value`` laid out ``(*dim, time, *seq)``, and the sizes of those axes.

    Without ``dim``, ``time`` and ``seq``, every axis of ``value`` is a dim
    axis and the value holds one time step. Otherwise they name the axes of
    ``value`` that play each part: ``dim`` and ``seq`` an axis or a tuple of
    axes, ``time`` one axis. The axes ``time`` and ``seq`` leave are the dim
    axes, unless ``dim`` names them; a value with none keeps a dim of one.
    """
    array = np.atleast_1d(np.asarray(value, dtype=np.float32))
    if array.size == 0:
        raise ValueError(f"{owner}: value must not be empty.")

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

    time_axes = axes(time, "time")
    if len(time_axes) > 1:
        raise ValueError(f"{owner}: time is one axis of value, got {time}.")
    seq_axes = axes(seq, "seq")
    dim_axes = (
        axes(dim, "dim")
        if dim is not None
        else tuple(
            axis for axis in range(array.ndim) if axis not in time_axes + seq_axes
        )
    )
    order = dim_axes + time_axes + seq_axes
    if sorted(order) != list(range(array.ndim)):
        raise ValueError(
            f"{owner}: dim, time and seq must take each axis of a value of shape "
            f"{array.shape} once, got dim={dim_axes}, time={time_axes}, "
            f"seq={seq_axes}."
        )

    array = np.transpose(array, order)
    if not dim_axes:
        array = array[np.newaxis]
    dim_rank = max(1, len(dim_axes))
    if not time_axes:
        array = np.expand_dims(array, axis=dim_rank)
    return (
        array,
        array.shape[:dim_rank],
        array.shape[dim_rank],
        array.shape[dim_rank + 1 :],
    )


class _Value(Layer):
    """What a Parameter and a Constant share: a value the model holds rather
    than reads from data, laid out and batched like any other stream.

    The two differ only in whether training changes the value.
    """

    _impl: type[ParameterImpl] = ParameterImpl

    def __init__(self, name, value, initializer, dim, time, seq):
        if value is not None:
            value, dim, time, seq = _value_layout(
                value, dim, time, seq, self.__class__.__name__
            )

        self.value = value
        self.initializer = initializer

        super().__init__(
            name=name,
            seq=seq,
            time=time,
            dim=dim,
            value=None if value is None else value.tolist(),
            initializer=initializer,
        )

    def build_layer(self):
        return self._impl(
            value_shape=self.shape.tuple,
            value=self.value,
            initializer=self.initializer,
            name=self.name,
        )

    def get_config(self):
        # Read from no stream, a value keeps its shape in its config rather
        # than taking it from its inputs as other layers do. A value is saved
        # laid out (*dim, time, *seq), with the axes that play each part.
        config = Stream.get_config(self)
        if self.value is not None:
            dim_rank, seq_rank = len(self.shape.dim), len(self.shape.seq)
            config.update(
                {
                    "dim": list(range(dim_rank)),
                    "time": dim_rank,
                    "seq": list(range(dim_rank + 1, dim_rank + 1 + seq_rank)),
                    "value": self.value.tolist(),
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
    axes, and ``initializer``, any Keras initializer, draws its initial value.

    ``value`` sets the shape and the initial value. Every axis of the value is
    a dim axis, and the value holds one time step, unless ``dim``, ``time``
    and ``seq`` say otherwise: given with a value they are axes of the value
    - ``dim`` and ``seq`` an axis or a tuple of axes, ``time`` one axis - and
    the axes ``time`` and ``seq`` leave are the dim axes::

        Parameter("p", value=np.ones((2, 3, 2)))                       # D=(2, 3, 2), T=1
        Parameter("p", value=np.ones((2, 3, 2)), dim=0, time=1, seq=2)  # D=(2,), T=3, S=(2,)
        Parameter("p", value=[[1.0, 2.0, 3.0]], time=1)                 # D=(1,), T=3

    Like every stream it is laid out with a batch axis first, the same value
    for every sample::

        (batch, *dim, time, *seq)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        value=None,
        initializer="random_normal",
        seq=None,
        time=None,
        dim=None,
    ):
        super().__init__(name, value, initializer, dim, time, seq)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value": None if self.value is None else self.value.tolist(),
                "initializer": _serialized_initializer(self.initializer),
            }
        )
        return config

    @property
    def param(self):
        return self._variable


class Constant(_Value):
    """
    Non-trainable symbolic constant layer.

    ``value`` sets its shape as for a :class:`Parameter`: every axis of the
    value is a dim axis, and the value holds one time step, unless ``dim``,
    ``time`` and ``seq`` name the axes of the value that play each part::

        Constant("c", value=[1.0, 2.0, 3.0])                 # D=(3,), T=1
        Constant("c", value=[[1.0, 2.0, 3.0]], time=1)       # D=(1,), T=3

    Numbers used in arithmetic with a stream become constants automatically.

    Like every stream it is laid out with a batch axis first, the same value
    for every sample::

        (batch, *dim, time, *seq)
    """

    _impl = ConstantImpl

    def __init__(
        self, name: str | None = None, *, value, seq=None, time=None, dim=None
    ):
        if value is None:
            raise ValueError("Constant requires a value.")
        super().__init__(name, value, None, dim, time, seq)

    @property
    def constant(self):
        return self._variable

    @classmethod
    def from_config(cls, config: dict, preds=None):
        return cls(
            name=config["name"],
            value=config["value"],
            dim=config.get("dim"),
            time=config.get("time"),
            seq=config.get("seq"),
        )


@keras.saving.register_keras_serializable(package="nnodely")
class WeightImpl(ParameterImpl):
    """A layer's weight: a variable of the weight's own shape.

    The layer reads it whole, not one value per sample, so it comes out with
    a batch axis of one: ``(1, *weight_shape, 1)``. The initializer fills it
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

    def build(self, input_shape=None):
        self.variable = self.add_weight(
            name="value",
            shape=self.weight_shape,
            initializer=_per_slice_initializer(self.initializer, self.draw_shape),
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
    ):
        self.weight_shape = tuple(int(axis) for axis in shape)
        if draw is None:
            draw = (
                self.weight_shape
                if len(self.weight_shape) <= 2
                else (math.prod(self.weight_shape[:-1]), self.weight_shape[-1])
            )
        self.draw_shape = tuple(int(axis) for axis in draw)
        super().__init__(name, initializer=initializer, dim=self.weight_shape, time=1)

    def build_layer(self):
        return WeightImpl(
            value_shape=self.shape.tuple,
            weight_shape=self.weight_shape,
            draw_shape=self.draw_shape,
            initializer=self.initializer,
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "shape": self.weight_shape,
            "initializer": _serialized_initializer(self.initializer),
            "draw": self.draw_shape,
        }


class _ParameterWeights(Layer):
    """A layer computing with a kernel and a bias held by Parameters, its
    predecessors after its input.

    ``kernel`` and ``bias`` are each a Parameter, used as given, or an
    initializer, by name or object: the layer then makes a :class:`Weight`
    of the shape its input asks for, the first time it is called, and shares
    it with every later call. ``bias=True`` takes the layer's default bias
    initializer, and ``bias=False`` leaves the bias out.

    A subclass gives the kernel and bias shapes for an input with
    :meth:`_weight_shapes`, and the shapes their initializers draw with
    :meth:`_draw_shapes`. Its Keras layer is called on ``[x, kernel]`` or
    ``[x, kernel, bias]``.
    """

    _bias_initializer = "zeros"

    def __init__(self, name, kernel, bias, **properties):
        self._kernel = kernel
        self._bias = bias
        # Made, or taken as given, on the first call.
        self._weights: list[Parameter] | None = None
        super().__init__(name=name, kernel=kernel, bias=bias, **properties)

    def _weight_shapes(self, x) -> tuple[tuple[int, ...], tuple[int, ...]]:
        raise NotImplementedError

    def _draw_shapes(self, kernel_shape, bias_shape):
        """The shapes the kernel and bias initializers draw; None for the
        Weight's own default."""
        return None, None

    def _parameters(self, x) -> list[Parameter]:
        """The kernel and the bias for the input ``x``, made on the first
        call. Each must have the shape of its weight, axes of size 1 aside:
        they leave the layout of the values as it is."""
        kernel_shape, bias_shape = self._weight_shapes(x)
        if self._weights is None:
            kernel_draw, bias_draw = self._draw_shapes(kernel_shape, bias_shape)
            self._weights = [
                self._weight("kernel", self._kernel, kernel_shape, kernel_draw)
            ]
            if self._bias is not False:
                bias = self._bias_initializer if self._bias is True else self._bias
                self._weights.append(self._weight("bias", bias, bias_shape, bias_draw))
        for label, weight, expected in zip(
            ("kernel", "bias"), self._weights, (kernel_shape, bias_shape)
        ):
            if _without_ones(weight.shape.tuple) != _without_ones(expected):
                raise ValueError(
                    f"{self.name}: {label} must have the shape {tuple(expected)}, "
                    f"axes of size 1 aside, got {weight.shape.tuple}."
                )
        return self._weights

    def _weight(self, label, setting, shape, draw) -> Parameter:
        if isinstance(setting, Parameter):
            return setting
        return Weight(
            f"{self.name}_{label}", shape=shape, initializer=setting, draw=draw
        )

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # The weights are predecessors like the input, but configured on the
        # layer, so they are appended here and reconnected when reloading.
        if inputs and all(isinstance(value, Stream) for value in inputs):
            inputs = [inputs[0], *self._parameters(inputs[0])]
        return super().__call__(inputs)

    def get_config(self):
        # The weights are saved with the graph, as the predecessors they are:
        # the config only says whether one of them is a bias.
        config = super().get_config()
        del config["kernel"]
        config["bias"] = config["bias"] is not False
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        preds = preds or []
        weights = iter(preds[1:])
        layer = cls(
            **{
                **config,
                "kernel": next(weights),
                "bias": next(weights) if config["bias"] else False,
            }
        )
        return layer(preds[0])

    def _weight_nodes(self) -> list[Parameter]:
        return cast(list[Parameter], self.preds[1:]) if self.preds else []

    @property
    def kernel(self):
        """The variable of the Parameter holding the kernel, in that
        Parameter's shape; None before the model is built."""
        weights = self._weight_nodes()
        return weights[0].param if weights else None

    @property
    def bias(self):
        """The variable of the Parameter holding the bias, in that
        Parameter's shape; None without a bias or before the model is
        built."""
        weights = self._weight_nodes()
        return weights[1].param if len(weights) > 1 else None


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
