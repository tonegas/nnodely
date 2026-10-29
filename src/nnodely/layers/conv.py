"""2D convolution, and flattening and reshaping of the dim axes."""

from __future__ import annotations

import math

import keras

from nnodely.core.layer import Layer
from nnodely.layers.parameter import Parameter, _check_weight_size, _ParameterWeights


def _pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    first, second = value
    return int(first), int(second)


def _refuse_several(layer, inputs) -> None:
    """Refuse several streams to a layer that reads one, rather than dropping
    the others."""
    if len(inputs) > 1:
        raise ValueError(
            f"{layer.name}: {type(layer).__name__} reads one stream, got "
            f"{len(inputs)}. Apply one to each."
        )


@keras.saving.register_keras_serializable(package="nnodely")
class Conv2DImpl(keras.layers.Layer):
    """A 2D convolution of every image of a stream.

    Input convention:
        [batch, *seq, time, height, width, channels], or without the window

    Output convention:
        [batch, *seq, time, height', width', filters]

    It owns no weight: it is called on ``[x, kernel]`` or ``[x, kernel,
    bias]``, the values of the Parameters holding them, and reads their first
    sample.
    """

    def __init__(
        self,
        filters: int,
        kernel_size,
        strides: int | tuple[int, int] = 1,
        padding: str = "valid",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = int(filters)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides)
        self.padding = padding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
            }
        )
        return config

    def build(self, input_shape):
        channels = int(input_shape[0][-1])
        self.kernel_shape = (*self.kernel_size, channels, self.filters)
        self.bias_shape = (self.filters,)
        for label, shape, expected in zip(
            ("kernel", "bias"), input_shape[1:], (self.kernel_shape, self.bias_shape)
        ):
            _check_weight_size(self.name, label, shape, expected)
        super().build(input_shape)

    def call(self, inputs):
        x, kernel, *bias = inputs
        rank = len(x.shape)
        image = tuple(int(axis) for axis in x.shape[-3:])

        # [batch, *seq, time, H, W, C]: every image of the batch, the sequence
        # and the window is one conv sample. Channels last, as the dim axes
        # end: TensorFlow has no channels-first convolution on CPU either.
        lead = keras.ops.shape(x)[: rank - 3]
        y = keras.ops.conv(
            keras.ops.reshape(x, (-1, *image)),
            keras.ops.reshape(kernel[0], self.kernel_shape),
            strides=self.strides,  # type: ignore[arg-type]
            padding=self.padding,
            data_format="channels_last",
        )
        if bias:
            y = y + keras.ops.reshape(bias[0][0], self.bias_shape)

        # [samples, H', W', F] -> [batch, *seq, time, H', W', F]
        return keras.ops.reshape(y, (*lead, *y.shape[1:]))


class Conv2D(_ParameterWeights):
    """
    2D convolution of a stream of images, laid out channels last on the dim
    axes. The same kernels are applied to every image, at every sample of
    the window and every step of the sequence::

        input:  (batch, *seq, time, height, width, channels)
        output: (batch, *seq, time, height', width', filters)
        kernel: (kernel_height, kernel_width, channels, filters)
        bias:   (filters,)

    A stream without a window is convolved too, ``(batch, *seq, height,
    width, channels)``. ``padding="valid"`` gives ``height' = (height -
    kernel) // stride + 1``, ``padding="same"`` gives ``height' = ceil(height
    / stride)``; the same for the width.

    ``kernel`` and ``bias`` are Parameters, predecessors of the layer, as for
    :class:`Linear`: assigning, training or freezing them changes the
    convolution. Each is given as a :class:`Parameter`, of the shape of the
    weight (axes of size 1 aside), or as a Keras initializer, by name or
    object, from which the layer makes one, named ``"<name>_kernel"`` or
    ``"<name>_bias"``; the kernel is drawn whole, as Keras draws a
    convolution kernel, its fans counted over the receptive field.
    ``bias=True`` draws the bias with ``"glorot_uniform"``, and
    ``bias=False`` leaves it out.
    """

    _bias_initializer = "glorot_uniform"

    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, int],
        strides: int | tuple[int, int] = 1,
        padding: str = "valid",
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        name=None,
    ):
        if padding not in ("valid", "same"):
            raise ValueError(f'padding must be "valid" or "same", got {padding!r}.')
        self.filters = int(filters)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides)
        self.padding = padding
        super().__init__(
            name,
            kernel,
            bias,
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

    def _image(self, shape) -> tuple[int, int, int]:
        """The height, width and channels of the images of ``shape``."""
        if shape.dim_rank != 3:
            raise ValueError(
                f"{self.name}: Conv2D takes images with dim (height, width, channels), "
                f"got dim {tuple(shape.dim)}."
            )
        height, width, channels = shape.dim
        return height, width, channels

    def _weight_shapes(self, x):
        *_, channels = self._image(x.shape)
        return (*self.kernel_size, channels, self.filters), (self.filters,)

    def _draw_shapes(self, kernel_shape, bias_shape):
        # The kernel whole: Keras counts the fans of a convolution kernel over
        # its receptive field, which the Weight's default matrix would not.
        return kernel_shape, None

    def output_shape(self, *inputs):
        # Declared rather than probed with dummy tensors, which would have to
        # stand in for the weights too.
        shape = inputs[0].shape
        *image, _ = self._image(shape)
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
            name=self.name,
        )


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

        # [batch, *seq, time, *dim] -> [batch, *seq, time, features]: the dim
        # axes, last and contiguous, merge into one.
        lead = keras.ops.shape(x)[: rank - self.dim_rank]
        return keras.ops.reshape(x, (*lead, features))


class Flatten(Layer):
    """
    Merges the dim axes of a stream into one, in C order, so that a layer
    acting on the first dim axis sees every feature::

        input:  (batch, *seq, time, *dim)
        output: (batch, *seq, time, prod(dim))

    The sequence and the window, if any, are kept.
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        _refuse_several(self, inputs)
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
        # [batch, *seq, time, *dim] -> [batch, *seq, time, *shape]: the dim
        # axes, last and contiguous, take the new shape.
        lead = keras.ops.shape(x)[: len(x.shape) - self.dim_rank]
        return keras.ops.reshape(x, (*lead, *self.target_shape))


class Reshape(Layer):
    """
    Lays the dim axes of a stream out as ``shape``, in C order - the inverse
    of ``Flatten``, which turns a flat feature vector back into an image::

        input:  (batch, *seq, time, *dim)
        output: (batch, *seq, time, *shape)

    ``shape`` must hold as many features as ``dim``. The sequence and the
    window, if any, are kept.
    """

    def __init__(self, shape: int | tuple[int, ...], name=None):
        self.target = (int(shape),) if isinstance(shape, int) else tuple(int(axis) for axis in shape)
        super().__init__(name=name, shape=self.target)

    def output_shape(self, *inputs):
        _refuse_several(self, inputs)
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
