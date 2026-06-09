import numpy as np
import keras

from nnodely.core.stream import Stream


class Constant(Stream):
    """
    Non-trainable symbolic source node.

    The value is mandatory and defines the symbolic shape automatically.

    Shape convention inferred from value:
    - scalar        -> time=1, dim=(1,)
    - shape (T,)    -> time=T, dim=(1,)
    - shape (T, D...) -> time=T, dim=(D...)

    seq is kept empty by default.
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

        if arr.ndim == 0:
            seq = None
            time = ()
            dim = (1,)
        elif arr.ndim == 1:
            seq = None
            time = ()
            dim = (1,)
        else:
            seq = None
            time = ()
            dim = tuple(arr.shape[1:])

        super().__init__(
            name=name,
            seq=seq,
            time=time,
            dim=dim,
            preds=[],
        )

        self.value = arr
        self.dtype = dtype
        self.constant = None

    def build_constant(self):
        if self.constant is not None:
            return self.constant

        self.constant = keras.Variable(
            initializer=self.value,
            shape=self.shape,
            dtype=self.dtype,
            trainable=False,
            name=self.name,
        )
        return self.constant

    def as_tensor(self, anchor):
        v = self.build_constant()
        return keras.layers.Lambda(
            lambda x: v,
            output_shape=self.shape,
            name=f"{self.name}_tensor",
        )(anchor)

    @property
    def value_numpy(self):
        if self.constant is None:
            return self.value
        return np.array(self.constant)
