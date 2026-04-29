import numpy as np
import keras

from nnodely.core.stream import Stream
# from nnodely.core.dag import to_tuple


class Parameter(Stream):
    """
    Trainable symbolic source node.

    A Parameter is a source Stream with a standalone trainable Keras weight
    stored in `self.param`.

    Shape without batch:
        seq + (time,) + dim
    """

    def __init__(
        self,
        name: str,
        *,
        value: list[int | float | list] | float | int | None = None,
        initializer: str = "random_normal",
        seq: int | tuple[int, ...] | None = None,
        time: int | None = None,
        dim: int | tuple[int, ...] | None = None,
        dtype: str = "float32",
    ):
        if value:
            arr = np.asarray(value, dtype=np.float32)

            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
                seq = None
                time = 1
                dim = (1,)
            elif arr.ndim == 1:
                arr = arr.reshape(arr.shape[0], 1)
                seq = None
                time = arr.shape[0]
                dim = (1,)
            else:
                seq = None
                time = arr.shape[0]
                dim = tuple(arr.shape[1:])

        super().__init__(
            name=name,
            seq=seq,
            time=time,
            dim=dim,
            predecessors=[],
        )

        self.value = value
        self.initializer = initializer
        self.dtype = dtype

        self.param = None

    def build_parameter(self) -> keras.Variable:
        """
        Create the trainable Keras weight if it does not exist yet.
        """
        if self.param is not None:
            return self.param

        shape = self.shape

        if self.value is not None:
            value = np.asarray(self.value, dtype=np.float32)
            if value.shape != shape:
                try:
                    value = np.reshape(value, shape)
                except Exception as e:
                    raise ValueError(
                        f"Parameter '{self.name}' value shape {value.shape} "
                        f"is incompatible with expected shape {shape}"
                    ) from e
            init = value
        else:
            if self.initializer == "zeros":
                init = np.zeros(shape, dtype=np.float32)
            elif self.initializer == "ones":
                init = np.ones(shape, dtype=np.float32)
            elif self.initializer == "random_normal":
                init = np.random.randn(*shape).astype(np.float32)
            elif self.initializer == "random_uniform":
                init = np.random.uniform(-0.05, 0.05, size=shape).astype(np.float32)
            else:
                raise ValueError(f"Unsupported initializer: {self.initializer!r}")

        self.param = keras.Variable(
            initializer=init,
            shape=shape,
            dtype=self.dtype,
            trainable=True,
            name=self.name,
        )
        return self.param

    def as_tensor(self, anchor: "Parameter") -> list[int | float | list] | float:
        v = self.build_parameter()
        return keras.layers.Lambda(
            lambda x: v,
            output_shape=self.shape,
            name=self.name,
        )(anchor)

    @property
    def value_numpy(self) -> np.ndarray | None:
        if self.param is None:
            return None
        return np.array(self.param)
