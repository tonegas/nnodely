import warnings

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
