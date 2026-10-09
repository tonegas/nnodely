import warnings
from typing import Any

import numpy as np
import keras

from nnodely.core.layer import Layer
from nnodely.core.stream import Shape, Stream
from nnodely.utils.utils import _serialized_initializer


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


class _ParameterWeights(Layer):
    """A layer whose kernel and bias may be given as Parameters.

    A subclass keeps the two settings as ``self._kernel`` and ``self._bias``
    and builds a Keras layer with ``kernel`` and ``bias`` attributes. A
    Parameter given as either becomes a predecessor of the layer, after its
    input, and stands in for that weight: the layer computes with its value,
    and the graph saves it.
    """

    _kernel: Any
    _bias: Any

    def _parameters(self) -> list[Parameter]:
        """The Parameters given as kernel and bias, in this order."""
        return [p for p in (self._kernel, self._bias) if isinstance(p, Parameter)]

    def _check_parameters(self, kernel_shape, bias_shape):
        """Raise unless each Parameter has the shape of the weight it stands
        for. Axes of size 1 leave the layout of the values as it is, so they
        do not count."""
        for label, value, expected in (
            ("kernel", self._kernel, kernel_shape),
            ("bias", self._bias, bias_shape),
        ):
            if not isinstance(value, Parameter):
                continue
            if _without_ones(value.shape.tuple) != _without_ones(expected):
                raise ValueError(
                    f"{self.name}: {label} must have the shape {tuple(expected)}, "
                    f"axes of size 1 aside, got {value.shape.tuple}."
                )

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # The Parameters are predecessors like the input, but configured on
        # the layer, so they are appended here and reconnected the same way
        # when reloading.
        if inputs and all(isinstance(value, Stream) for value in inputs):
            inputs = [inputs[0], *self._parameters()]
        return super().__call__(inputs)

    def get_config(self):
        return {
            **super().get_config(),
            "kernel": _weight_config(self._kernel),
            "bias": _weight_config(self._bias),
        }

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The Parameters named in the config follow the input among the
        # predecessors, kernel first.
        given = iter((preds or [])[1:])
        config = {
            **config,
            **{
                key: next(given)
                for key in ("kernel", "bias")
                if isinstance(config.get(key), dict) and "parameter" in config[key]
            },
        }
        layer = cls(**config)
        return layer(preds[0]) if preds else layer

    @property
    def kernel(self):
        """The kernel the layer computes with: the variable of the Parameter
        given as kernel, in the Parameter's shape, or the layer's own weight
        once built."""
        if isinstance(self._kernel, Parameter):
            return self._kernel.param
        return None if self._layer is None else self._layer.kernel

    @property
    def bias(self):
        """The bias the layer computes with: the variable of the Parameter
        given as bias, in the Parameter's shape, or the layer's own weight
        once built; None without a bias."""
        if isinstance(self._bias, Parameter):
            return self._bias.param
        return None if self._layer is None else self._layer.bias


def _weight_config(value):
    """A kernel or bias setting as a saved config holds it.

    A Parameter is saved with the graph, as a predecessor of the layer, so
    the config only names it; an initializer is kept by its name or config.
    """
    if isinstance(value, Parameter):
        return {"parameter": value.name}
    if value is None or isinstance(value, bool):
        return value
    return _serialized_initializer(value)


def _without_ones(shape):
    """``shape`` without its axes of size 1."""
    return tuple(axis for axis in shape if axis != 1)
