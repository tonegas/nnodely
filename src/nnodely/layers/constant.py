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
        value = np.asarray(self.value, dtype=np.float32)

        if value.shape != self.constant_shape:
            try:
                value = np.reshape(value, self.constant_shape)
            except Exception as e:
                raise ValueError(
                    f"Constant value shape {value.shape} is incompatible "
                    f"with expected shape {self.constant_shape}"
                ) from e

        self.constant = self.add_weight(
            name="value",
            shape=self.constant_shape,
            initializer=keras.initializers.Constant(value=value.tolist()),
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, anchor):
        value = keras.ops.expand_dims(self.constant, axis=0)

        zero = (
            keras.ops.sum(
                anchor,
                axis=tuple(range(1, len(anchor.shape))),
            )
            * 0.0
        )

        zero = keras.ops.reshape(
            zero,
            (-1,) + (1,) * len(self.constant_shape),
        )

        return value + zero


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
    ):
        if value is None:
            raise ValueError("Constant requires a value.")

        arr = np.asarray(value, dtype=np.float32)

        if arr.ndim == 0:
            arr = arr.reshape(1)
            dim = (1,)
            time = None
            seq = None
        elif arr.ndim == 1:
            arr = arr.reshape(arr.shape[0], 1)
            dim = (arr.shape[0],)
            time = None
            seq = None
        else:
            dim = tuple(arr.shape[:-1])
            time = arr.shape[-1]
            seq = None

        self.value = arr
        super().__init__(
            name=name,
            seq=seq,
            time=time,
            dim=dim,
            value=arr.tolist(),
        )

    def output_shape(self, *inputs):
        return self.dimensions

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


# @keras.saving.register_keras_serializable(package="nnodely")
# class ConstantImpl(keras.layers.Layer):
#     def __init__(self, value, name=None, **kwargs):
#         super().__init__(name=name, **kwargs)

#         self.value = np.asarray(
#             value,
#             dtype=np.float32,
#         )

#     def build(self, input_shape=None):
#         self.constant = self.add_weight(
#             name="value",
#             shape=self.value.shape,
#             initializer=keras.initializers.Constant(
#                 self.value.tolist(),
#             ),
#             trainable=False,
#             dtype="float32",
#         )

#         super().build(input_shape)

#     def call(self, inputs=None):
#         return self.constant

#     def get_config(self):
#         config = super().get_config()

#         config.update(
#             {
#                 "value": self.value.tolist(),
#             }
#         )

#         return config


# class Constant(Layer):
#     def __init__(
#         self,
#         name: str | None = None,
#         *,
#         value,
#         dim=None,
#         time=None,
#         seq=None,
#     ):
#         self.value = np.asarray(
#             value,
#             dtype=np.float32,
#         )

#         super().__init__(
#             name=name,
#             dim=dim if dim is not None else self.value.shape[0],
#             time=time
#             if time is not None
#             else self.value.shape[1]
#             if self.value.ndim > 1
#             else 1,
#             seq=seq
#             if seq is not None
#             else self.value.shape[2:]
#             if self.value.ndim > 2
#             else (),
#             value=self.value.tolist(),
#         )

#     def output_shape(self, *inputs):
#         return self.dimensions

#     def build_layer(self):
#         return ConstantImpl(
#             value=self.value,
#             name=self.name,
#         )

#     @property
#     def constant(self):
#         if self._layer is None:
#             return None

#         return self._layer.constant

#     @property
#     def value_numpy(self):
#         if self.constant is None:
#             return None

#         return keras.ops.convert_to_numpy(
#             self.constant,
#         )

#     def get_config(self):
#         config = super().get_config()

#         config.update(
#             {
#                 "value": self.value.tolist(),
#                 "dim": self.shape.dim,
#                 "time": self.shape.time,
#                 "seq": self.shape.seq,
#             }
#         )
#         return config

#     @classmethod
#     def from_config(
#         cls,
#         config,
#         preds=None,
#     ):
#         return cls(
#             name=config["name"],
#             value=config["value"],
#             dim=config["dim"],
#             time=config["time"],
#             seq=config["seq"],
#         )
