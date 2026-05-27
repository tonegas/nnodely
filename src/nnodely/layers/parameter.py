import numpy as np
import keras

from nnodely.core.stream import Stream


class _ParameterLayer(keras.layers.Layer):
    def __init__(self, parameter, **kwargs):
        super().__init__(dtype=parameter.dtype, **kwargs)
        self.parameter = parameter

    def build(self, input_shape=None):
        shape = self.parameter.shape

        if self.parameter.value is not None:
            value = np.asarray(self.parameter.value, dtype=np.float32)
            if value.shape != shape:
                try:
                    value = np.reshape(value, shape)
                except Exception as e:
                    raise ValueError(
                        f"Parameter '{self.parameter.name}' value shape {value.shape} "
                        f"is incompatible with expected shape {shape}"
                    ) from e

            initializer = keras.initializers.Constant(value=value.tolist())
        else:
            initializer = keras.initializers.get(self.parameter.initializer)

        self.parameter.param = self.add_weight(
            name="value",
            shape=shape,
            initializer=initializer,
            trainable=True,
            dtype=self.parameter.dtype,
        )

        super().build(input_shape)

    # def call(self, anchor):
    #     batch = keras.ops.shape(anchor)[0]

    #     out_shape = keras.ops.concatenate(
    #         [
    #             keras.ops.reshape(batch, (1,)),
    #             keras.ops.convert_to_tensor(self.parameter.shape, dtype="int32"),
    #         ],
    #         axis=0,
    #     )

    #     value = keras.ops.expand_dims(self.parameter.param, axis=0)
    #     return keras.ops.broadcast_to(value, out_shape)
    def call(self, anchor):
        value = keras.ops.expand_dims(self.parameter.param, axis=0)

        axes = tuple(range(1, len(anchor.shape)))
        zeros = keras.ops.sum(anchor, axis=axes, keepdims=True) * 0.0

        return zeros + value


class Parameter(Stream):
    """
    Trainable symbolic source node.

    Shape without batch:
        dim + (time,) + seq
    """

    def __init__(
        self,
        name: str,
        *,
        value=None,
        initializer="random_normal",
        seq=None,
        time=None,
        dim=None,
        dtype="float32",
    ):
        if value is not None:
            arr = np.asarray(value, dtype=np.float32)

            # Convention: dim + (time,) + seq
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
                dim = (1,)
                time = 1
                seq = None
            elif arr.ndim == 1:
                # [D] -> [D, time=1]
                arr = arr.reshape(arr.shape[0], 1)
                dim = (arr.shape[0],)
                time = 1
                seq = None
            else:
                # assume [dim..., time]
                dim = tuple(arr.shape[:-1])
                time = arr.shape[-1]
                seq = None

            value = arr

        super().__init__(
            name=name,
            seq=seq,
            time=time,
            dim=dim,
            preds=[],
        )

        self.value = value
        self.initializer = initializer
        self.dtype = dtype

        self._state = {
            "param": None,
            "layer": None,
        }

    def __copy__(self):
        new = self.__class__(
            name=self.name,
            value=self.value,
            initializer=self.initializer,
            seq=self.seq,
            time=self.time,
            dim=self.dim,
            dtype=self.dtype,
        )
        new.preds = list(self.preds)

        # critical: original and flattened copy share param/layer
        new._state = self._state
        return new

    def as_tensor(self, anchor):
        if anchor is None:
            raise ValueError(
                f"Parameter '{self.name}' needs an anchor tensor to enter the Keras graph."
            )

        if self._layer is None:
            self._layer = _ParameterLayer(self, name=f"{self.name}_tensor")

        return self._layer(anchor)

    @property
    def param(self):
        return self._state["param"]

    @param.setter
    def param(self, value):
        self._state["param"] = value

    @property
    def _layer(self):
        return self._state["layer"]

    @_layer.setter
    def _layer(self, value):
        self._state["layer"] = value

    @property
    def value_numpy(self):
        if self.param is None:
            return None
        return keras.ops.convert_to_numpy(self.param)
