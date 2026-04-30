"""Type aliases for nnodely package."""

from typing import TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from nnodely.core.stream import Stream
    import numpy as np
    import tensorflow as tf
    import keras

# Type alias for stream operands (used in Stream arithmetic operations)
StreamOperand: TypeAlias = "Stream | list[int | float | list] | int | float"
TensorLike: TypeAlias = "Stream | np.ndarray | tf.Tensor | keras.KerasTensor"
