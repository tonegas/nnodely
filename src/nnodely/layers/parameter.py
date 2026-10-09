import math
import warnings
from typing import cast

import numpy as np
import keras

from nnodely.core.layer import Layer
from nnodely.core.stream import Shape, Stream
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


def _value_array(value, owner: str) -> np.ndarray:
    """``value`` laid out ``(dim, time, *seq)``.

    A number or a vector is one time step of its dim, a matrix gives the dim
    and time axes, and any further axis is a sequence axis.
    """
    array = np.atleast_1d(np.asarray(value, dtype=np.float32))
    if array.size == 0:
        raise ValueError(f"{owner}: value must not be empty.")
    return array[:, np.newaxis] if array.ndim == 1 else array


class _Value(Layer):
    """What a Parameter and a Constant share: a value the model holds rather
    than reads from data, laid out and batched like any other stream.

    The two differ only in whether training changes the value.
    """

    _impl: type[ParameterImpl] = ParameterImpl

    def __init__(self, name, value, initializer, dim, time, seq):
        overridden = []
        if value is not None:
            value = _value_array(value, self.__class__.__name__)
            shape = Shape(dim=value.shape[0], time=value.shape[1], seq=value.shape[2:])
            given = {"dim": dim, "time": time, "seq": seq}
            overridden = [
                axis
                for axis, size in given.items()
                if size is not None
                and getattr(Shape(**{axis: size}), axis) != getattr(shape, axis)
            ]
            dim, time, seq = shape.dimensions

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
        if overridden:
            warnings.warn(
                f"{self.name}: the shape of value is {self.shape.tuple}, which "
                f"overrides the {', '.join(overridden)} given with it.",
                UserWarning,
                stacklevel=3,
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
        # than taking it from its inputs as other layers do.
        return Stream.get_config(self)

    @property
    def _variable(self):
        return getattr(self._layer, "variable", None)

    @property
    def value_numpy(self):
        return keras.ops.convert_to_numpy(self._variable)


class Parameter(_Value):
    """
    Trainable symbolic parameter.

    ``value`` sets its shape and initial value: a number or a vector is one
    time step of its ``dim``, a matrix is ``(dim, time)``, and further axes
    are ``seq`` axes. Without ``value`` the shape is given by ``dim``, ``time``
    and ``seq``, and ``initializer``, any Keras initializer, draws the initial
    value. A ``value`` overrides the ``dim``, ``time`` and ``seq`` given with it.

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

    ``value`` sets its shape: a number or a vector is one time step of its
    ``dim``, a matrix is ``(dim, time)``, and further axes are ``seq`` axes.
    Numbers used in arithmetic with a stream become constants automatically.

    Like every stream it is laid out with a batch axis first, the same value
    for every sample::

        (batch, *dim, time, *seq)
    """

    _impl = ConstantImpl

    def __init__(self, name: str | None = None, *, value):
        if value is None:
            raise ValueError("Constant requires a value.")
        super().__init__(name, value, None, None, None, None)

    @property
    def constant(self):
        return self._variable

    def get_config(self):
        config = super().get_config()
        config.update({"value": self.value.tolist()})  # type: ignore
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        return cls(name=config["name"], value=config["value"])


@keras.saving.register_keras_serializable(package="nnodely")
class WeightImpl(ParameterImpl):
    """A layer's weight: a variable of the weight's own shape.

    The layer reads it whole, not one value per sample, so it comes out with
    a batch axis of one: ``(1, *weight_shape, 1)``. The initializer draws
    every slice of its last ``rank`` axes alone.
    """

    def __init__(self, value_shape, weight_shape, rank, name=None, **kwargs):
        super().__init__(value_shape=value_shape, name=name, **kwargs)
        self.weight_shape = tuple(int(axis) for axis in weight_shape)
        self.rank = int(rank)

    def get_config(self):
        config = super().get_config()
        config.update({"weight_shape": self.weight_shape, "rank": self.rank})
        return config

    def build(self, input_shape=None):
        self.variable = self.add_weight(
            name="value",
            shape=self.weight_shape,
            initializer=_per_slice_initializer(self.initializer, self.rank),
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
    variable has exactly that shape; ``initializer`` draws every slice of its
    last ``rank`` axes alone (all of them by default). The layer reads it
    whole rather than one value per sample, so its stream has a batch axis of
    one.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        shape,
        initializer="glorot_uniform",
        rank: int | None = None,
    ):
        self.weight_shape = tuple(int(axis) for axis in shape)
        self.slice_rank = len(self.weight_shape) if rank is None else int(rank)
        super().__init__(name, initializer=initializer, dim=self.weight_shape, time=1)

    def build_layer(self):
        return WeightImpl(
            value_shape=self.shape.tuple,
            weight_shape=self.weight_shape,
            rank=self.slice_rank,
            initializer=self.initializer,
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "shape": self.weight_shape,
            "initializer": _serialized_initializer(self.initializer),
            "rank": self.slice_rank,
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
    :meth:`_weight_shapes`, and its Keras layer is called on ``[x, kernel]``
    or ``[x, kernel, bias]``.
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

    def _parameters(self, x) -> list[Parameter]:
        """The kernel and the bias for the input ``x``, made on the first
        call. Each must have the shape of its weight, axes of size 1 aside:
        they leave the layout of the values as it is."""
        kernel_shape, bias_shape = self._weight_shapes(x)
        if self._weights is None:
            self._weights = [self._weight("kernel", self._kernel, kernel_shape, 2)]
            if self._bias is not False:
                bias = self._bias_initializer if self._bias is True else self._bias
                self._weights.append(self._weight("bias", bias, bias_shape, 1))
        for label, weight, expected in zip(
            ("kernel", "bias"), self._weights, (kernel_shape, bias_shape)
        ):
            if _without_ones(weight.shape.tuple) != _without_ones(expected):
                raise ValueError(
                    f"{self.name}: {label} must have the shape {tuple(expected)}, "
                    f"axes of size 1 aside, got {weight.shape.tuple}."
                )
        return self._weights

    def _weight(self, label, setting, shape, rank) -> Parameter:
        if isinstance(setting, Parameter):
            return setting
        return Weight(
            f"{self.name}_{label}", shape=shape, initializer=setting, rank=rank
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
