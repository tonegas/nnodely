import numpy as np
import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class ConstantImpl(keras.layers.Layer):
    def __init__(self, value, constant_shape, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.value = np.asarray(value, dtype=np.float32)
        self.constant_shape = constant_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value": self.value.tolist(),
                "constant_shape": self.constant_shape,
            }
        )
        return config

    def build(self, input_shape=None):
        self.constant = self.add_weight(
            name="value",
            shape=self.constant_shape,
            initializer=keras.initializers.Constant(value=self.value.tolist()),
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, anchor):
        return self.constant


class Constant(Layer):
    """
    Non-trainable symbolic constant layer.

    Shape without batch:
        dim + time + seq
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        value,
        dim=None,
    ):
        if value is None:
            raise ValueError("Constant requires a value.")

        arr = np.atleast_1d(np.asarray(value, dtype=np.float32))

        if dim is None:
            dim = arr.shape[0]
            time = arr.shape[1] if arr.ndim > 1 else None
            seq = arr.shape[2:] if arr.ndim > 2 else None
        else:
            time = arr.shape[dim] if arr.ndim > dim else None
            seq = arr.shape[dim + 1 :] if arr.ndim > dim + 1 else None

        self.value = arr
        super().__init__(
            name=name,
            seq=seq,
            time=time,
            dim=dim,
            value=arr.tolist(),
        )

    def build_layer(self):
        return ConstantImpl(
            value=self.value,
            constant_shape=self.shape.tuple,
            name=self.name,
        )

    @property
    def constant(self):
        if self._layer is not None and hasattr(self._layer, "constant"):
            return self._layer.constant
        return None

    @property
    def value_numpy(self):
        return keras.ops.convert_to_numpy(self.constant)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value": self.value.tolist(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        node = cls(
            name=config["name"],
            value=config["value"],
        )
        return node
