"""2D convolution, and flattening and reshaping of the dim axes."""

from __future__ import annotations

import math

import keras

from nnodely.core.layer import Layer
from nnodely.utils.utils import _serialized_initializer


def _pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    first, second = value
    return int(first), int(second)


@keras.saving.register_keras_serializable(package="nnodely")
class Conv2DImpl(keras.layers.Layer):
    """A 2D convolution of every image of a stream.

    Input convention:
        [batch, *time, height, width, channels]

    Output convention:
        [batch, *time, height', width', filters]
    """

    def __init__(
        self,
        filters: int,
        kernel_size,
        strides=1,
        padding: str = "valid",
        use_bias: bool = True,
        name=None,
        initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = int(filters)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides)
        self.padding = padding
        self.use_bias = bool(use_bias)
        # Channels last, as the dim axes end: TensorFlow has no channels-first
        # convolution on CPU either.
        self.conv = keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            data_format="channels_last",
            kernel_initializer=initializer,  # type: ignore[arg-type]
            bias_initializer=bias_initializer,  # type: ignore[arg-type]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "use_bias": self.use_bias,
                "initializer": self.conv.kernel_initializer,
                "bias_initializer": self.conv.bias_initializer,
            }
        )
        return config

    def build(self, input_shape):
        self.conv.build((None, *(int(axis) for axis in input_shape[-3:])))
        super().build(input_shape)

    def call(self, x):
        rank = len(x.shape)
        image = tuple(int(axis) for axis in x.shape[-3:])

        # [batch, *time, H, W, C]: every image of the batch and time axes is
        # one conv sample.
        lead = keras.ops.shape(x)[: rank - 3]
        y = self.conv(keras.ops.reshape(x, (-1, *image)))

        # [samples, H', W', F] -> [batch, *time, H', W', F]
        return keras.ops.reshape(y, (*lead, *y.shape[1:]))


class Conv2D(Layer):
    """
    2D convolution of a stream of images, laid out channels last on the dim
    axes. The same kernels are applied at every step of every time axis::

        input:  (batch, *time, height, width, channels)
        output: (batch, *time, height', width', filters)

    ``padding="valid"`` gives ``height' = (height - kernel) // stride + 1``,
    ``padding="same"`` gives ``height' = ceil(height / stride)``; the same for
    the width.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, int],
        strides: int | tuple[int, int] = 1,
        padding: str = "valid",
        use_bias: bool = True,
        name=None,
        initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "glorot_uniform",
    ):
        if padding not in ("valid", "same"):
            raise ValueError(f'padding must be "valid" or "same", got {padding!r}.')
        self.filters = int(filters)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides)
        self.padding = padding
        self.use_bias = bool(use_bias)
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        super().__init__(
            name=name,
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            initializer=initializer,
            bias_initializer=bias_initializer,
        )

    def output_shape(self, *inputs):
        # Declared rather than probed with a dummy tensor, which a time axis
        # left dynamic cannot be given.
        shape = inputs[0].shape
        if shape.dim_rank != 3:
            raise ValueError(
                f"{self.name}: Conv2D takes images with dim (height, width, channels), "
                f"got dim {tuple(shape.dim)}."
            )
        *image, _ = shape.dim
        size = []
        for length, kernel, stride in zip(image, self.kernel_size, self.strides):
            if self.padding == "same":
                size.append(math.ceil(length / stride))
            elif length < kernel:
                raise ValueError(
                    f"{self.name}: a {self.kernel_size} kernel does not fit "
                    f"{tuple(image)} images without padding."
                )
            else:
                size.append((length - kernel) // stride + 1)
        return shape.replace(dim=(*size, self.filters))

    def build_layer(self):
        return Conv2DImpl(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            name=self.name,
            initializer=self.initializer,
            bias_initializer=self.bias_initializer,
        )

    def get_config(self):
        return {
            "name": self.name,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "initializer": _serialized_initializer(self.initializer),
            "bias_initializer": _serialized_initializer(self.bias_initializer),
        }

    @property
    def kernel(self):
        if self._layer is not None:
            return self._layer.conv.kernel
        return None

    @property
    def bias(self):
        if self._layer is not None and self._layer.use_bias:
            return self._layer.conv.bias
        return None


@keras.saving.register_keras_serializable(package="nnodely")
class FlattenImpl(keras.layers.Layer):
    """The ``dim_rank`` dim axes of a stream merged into one, in C order."""

    def __init__(self, dim_rank: int, **kwargs):
        super().__init__(**kwargs)
        self.dim_rank = int(dim_rank)

    def get_config(self):
        config = super().get_config()
        config.update({"dim_rank": self.dim_rank})
        return config

    def call(self, x):
        rank = len(x.shape)
        features = math.prod(int(axis) for axis in x.shape[rank - self.dim_rank :])

        # [batch, *time, *dim] -> [batch, *time, features]: the dim axes, last
        # and contiguous, merge into one.
        lead = keras.ops.shape(x)[: rank - self.dim_rank]
        return keras.ops.reshape(x, (*lead, features))


class Flatten(Layer):
    """
    Merges the dim axes of a stream into one, in C order, so that a layer
    acting on the first dim axis sees every feature::

        input:  (batch, *time, *dim)
        output: (batch, *time, prod(dim))
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        return shape.replace(dim=(math.prod(shape.dim),))

    def build_layer(self):
        return FlattenImpl(dim_rank=self.preds[0].shape.dim_rank, name=self.name)  # type: ignore


@keras.saving.register_keras_serializable(package="nnodely")
class ReshapeImpl(keras.layers.Layer):
    """The ``dim_rank`` dim axes of a stream laid out as ``shape``, in C order."""

    def __init__(self, dim_rank: int, shape, **kwargs):
        super().__init__(**kwargs)
        self.dim_rank = int(dim_rank)
        self.target_shape = tuple(int(axis) for axis in shape)

    def get_config(self):
        config = super().get_config()
        config.update({"dim_rank": self.dim_rank, "shape": self.target_shape})
        return config

    def call(self, x):
        # [batch, *time, *dim] -> [batch, *time, *shape]: the dim axes, last
        # and contiguous, take the new shape.
        lead = keras.ops.shape(x)[: len(x.shape) - self.dim_rank]
        return keras.ops.reshape(x, (*lead, *self.target_shape))


class Reshape(Layer):
    """
    Lays the dim axes of a stream out as ``shape``, in C order - the inverse
    of ``Flatten``, which turns a flat feature vector back into an image::

        input:  (batch, *time, *dim)
        output: (batch, *time, *shape)

    ``shape`` must hold as many features as ``dim``.
    """

    def __init__(self, shape: int | tuple[int, ...], name=None):
        self.target = (int(shape),) if isinstance(shape, int) else tuple(int(axis) for axis in shape)
        super().__init__(name=name, shape=self.target)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        if math.prod(shape.dim) != math.prod(self.target):
            raise ValueError(
                f"{self.name}: cannot reshape dim {tuple(shape.dim)} "
                f"({math.prod(shape.dim)} features) into {self.target} "
                f"({math.prod(self.target)} features)."
            )
        return shape.replace(dim=self.target)

    def build_layer(self):
        return ReshapeImpl(
            dim_rank=self.preds[0].shape.dim_rank,  # type: ignore
            shape=self.target,
            name=self.name,
        )
