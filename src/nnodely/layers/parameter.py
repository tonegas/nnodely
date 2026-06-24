import numpy as np
import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class ParameterImpl(keras.layers.Layer):
    def __init__(
        self,
        parameter_shape,
        value=None,
        initializer="random_normal",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.parameter_shape = tuple(parameter_shape)
        self.value = None if value is None else np.asarray(value, dtype=np.float32)
        self.initializer = initializer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "parameter_shape": self.parameter_shape,
                "value": None if self.value is None else self.value.tolist(),
                "initializer": self.initializer,
            }
        )
        return config

    def build(self, input_shape=None):
        if self.value is not None:
            value = np.asarray(self.value, dtype=np.float32)
            if value.shape != self.parameter_shape:
                try:
                    value = np.reshape(value, self.parameter_shape)
                except Exception as e:
                    raise ValueError(
                        f"Parameter value shape {value.shape} is incompatible "
                        f"with expected shape {self.parameter_shape}"
                    ) from e

            initializer = keras.initializers.Constant(value=value.tolist())
        else:
            initializer = keras.initializers.get(self.initializer)

        self.param = self.add_weight(
            name="value",
            shape=self.parameter_shape,
            initializer=initializer,
            trainable=True,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, anchor):
        value = keras.ops.expand_dims(self.param, axis=0)
        zero = (
            keras.ops.sum(
                anchor,
                axis=tuple(range(1, len(anchor.shape))),
            )
            * 0.0
        )
        zero = keras.ops.reshape(
            zero,
            (-1,) + (1,) * len(self.parameter_shape),
        )
        return value + zero


class Parameter(Layer):
    """
    Trainable symbolic parameter layer.

    Shape without batch:
        dim + time + seq
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
        if value is not None:
            arr = np.asarray(value, dtype=np.float32)

            if arr.ndim == 0:
                arr = arr.reshape(1)
                dim = (1,)
                time = ()
                seq = None
            elif arr.ndim == 1:
                arr = arr.reshape(arr.shape[0], 1)
                dim = (arr.shape[0],)
                time = ()
                seq = None
            else:
                dim = tuple(arr.shape[:-1])
                time = tuple(arr.shape[-1:])
                seq = None

            value = arr

        self.value = value
        self.initializer = initializer

        super().__init__(
            name=name,
            seq=seq,
            time=time,
            dim=dim,
            value=None
            if value is None
            else np.asarray(value, dtype=np.float32).tolist(),
            initializer=initializer,
        )

    def output_shape(self, *inputs):
        return self.dim, self.time, self.seq

    def build_layer(self):
        return ParameterImpl(
            parameter_shape=self.shape,
            value=self.value,
            initializer=self.initializer,
            name=self.name,
        )

    @property
    def param(self):
        if self._layer is not None and hasattr(self._layer, "param"):
            return self._layer.param
        return None

    @property
    def value_numpy(self):
        return keras.ops.convert_to_numpy(self.param)