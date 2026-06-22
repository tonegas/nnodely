import numpy as np
import keras

from nnodely.core.stream import Stream


class _ConstantLayer(keras.layers.Layer):
    def __init__(self, constant, **kwargs):
        super().__init__(dtype=constant.dtype, **kwargs)
        self.constant_node = constant

    def build(self, input_shape=None):
        shape = self.constant_node.shape

        value = np.asarray(self.constant_node.value, dtype=np.float32)
        if value.shape != shape:
            try:
                value = np.reshape(value, shape)
            except Exception as e:
                raise ValueError(
                    f"Constant '{self.constant_node.name}' value shape {value.shape} "
                    f"is incompatible with expected shape {shape}"
                ) from e

        initializer = keras.initializers.Constant(value=value.tolist())

        self.constant_node.constant = self.add_weight(
            name="value",
            shape=shape,
            initializer=initializer,
            trainable=False,
            dtype=self.constant_node.dtype,
        )

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"constant": self.constant_node})
        return config

    def call(self, anchor):
        value = keras.ops.expand_dims(self.constant_node.constant, axis=0)
        zero = keras.ops.sum(anchor, axis=tuple(range(1, len(anchor.shape)))) * 0.0
        zero = keras.ops.reshape(zero, (-1,) + (1,) * len(self.constant_node.shape))
        return value + zero


class Constant(Stream):
    """
    Non-trainable symbolic source node.

    Value is mandatory.

    Shape without batch:
        dim + (time,) + seq
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        value,
        dtype="float32",
    ):
        if value is None:
            raise ValueError("Constant requires a value.")

        arr = np.asarray(value, dtype=np.float32)

        # Convention: dim + (time,) + seq
        if arr.ndim == 0:
            arr = arr.reshape(1)
            dim = (1,)
            time = ()
            seq = None
        elif arr.ndim == 1:
            # [D]
            arr = arr.reshape(arr.shape[0], 1)
            dim = (arr.shape[0],)
            time = ()
            seq = None
        else:
            # assume [dim..., time]
            dim = tuple(arr.shape[:-1])
            time = tuple(arr.shape[-1:])
            seq = None
        
        super().__init__(
            name=name,
            seq=seq,
            time=time,
            dim=dim,
            preds=[],
        )

        self.value = arr
        self.dtype = dtype

        self._state = {
            "constant": None,
            "layer": None,
        }

    def __copy__(self):
        new = self.__class__(
            name=self.name,
            value=self.value,
            dtype=self.dtype,
        )
        new.preds = list(self.preds)

        # Important when flatten() copies the graph.
        new._state = self._state
        return new

    def as_tensor(self, anchor):
        if anchor is None:
            raise ValueError(
                f"Constant '{self.name}' needs an anchor tensor to enter the Keras graph."
            )

        if self._layer is None:
            self._layer = _ConstantLayer(self, name=f"{self.name}_tensor")

        return self._layer(anchor)

    @property
    def constant(self):
        return self._state["constant"]

    @constant.setter
    def constant(self, value):
        self._state["constant"] = value

    @property
    def _layer(self):
        return self._state["layer"]

    @_layer.setter
    def _layer(self, value):
        self._state["layer"] = value

    @property
    def value_numpy(self):
        if self.constant is None:
            return np.asarray(self.value, dtype=np.float32)
        return keras.ops.convert_to_numpy(self.constant)
