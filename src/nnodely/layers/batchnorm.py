"""Batch normalization layer for nnodely."""

from __future__ import annotations

import keras

from nnodely.layers.parameter import _ParameterWeights


@keras.saving.register_keras_serializable(package="nnodely")
class BatchNormImpl(keras.layers.Layer):
    """Batch normalization along ``axis``, as keras.layers.BatchNormalization
    computes it, with its scale and offset read from its inputs: it is called
    on ``[x, gamma, beta]``, without gamma unless ``scale`` and without beta
    unless ``center``, each the value of a Parameter, read whole. It holds the
    moving statistics, which training updates as Keras does."""

    def __init__(
        self,
        axis: int,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.center = bool(center)
        self.scale = bool(scale)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
            }
        )
        return config

    def build(self, input_shape):
        shape = input_shape[0] if self.center or self.scale else input_shape
        size = (int(shape[self.axis]),)
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=size,
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=size,
            initializer="ones",
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def reset_statistics(self):
        """The moving statistics as they start: mean 0, variance 1."""
        self.moving_mean.assign(keras.ops.zeros(self.moving_mean.shape))
        self.moving_variance.assign(keras.ops.ones(self.moving_variance.shape))

    def call(self, inputs, training=None):
        x, *weights = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        if training and self.trainable:
            axes = [axis for axis in range(len(x.shape)) if axis != self.axis]
            mean, variance = keras.ops.moments(x, axes=axes)
            self.moving_mean.assign(
                self.moving_mean * self.momentum + mean * (1.0 - self.momentum)
            )
            self.moving_variance.assign(
                self.moving_variance * self.momentum + variance * (1.0 - self.momentum)
            )
        else:
            mean, variance = self.moving_mean, self.moving_variance
        values = iter(weight[0] for weight in weights)
        gamma = next(values) if self.scale else None
        beta = next(values) if self.center else None
        return keras.ops.batch_normalization(
            x,
            mean,
            variance,
            axis=self.axis,
            offset=beta,
            scale=gamma,
            epsilon=self.epsilon,
        )


class BatchNorm(_ParameterWeights):
    """
    Batch normalization, as keras.layers.BatchNormalization computes it.

    Input and output::

        [batch, *seq, tw, *dim]

    Statistics are accumulated per feature: by default along the first dim
    axis. ``axis`` picks another tensor axis, counted with the batch as 0.

    The scale and the offset are :class:`Parameter` objects, predecessors of
    the layer: ``"<name>_gamma"``, starting from ones, unless ``scale`` is
    False, and ``"<name>_beta"``, starting from zeros, unless ``center`` is
    False. :attr:`Modely.parameters` lists them and :meth:`Modely.freeze`
    freezes them. The moving mean and variance are statistics of the data
    rather than weights: the layer holds them, and training updates them.
    """

    _labels = ("gamma", "beta")
    _defaults = {"gamma": "ones", "beta": "zeros"}

    def __init__(
        self,
        axis: int | None = None,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        name=None,
    ):
        self._axis = None if axis is None else int(axis)
        self._momentum = float(momentum)
        self._epsilon = float(epsilon)
        self._center = bool(center)
        self._scale = bool(scale)
        super().__init__(
            name,
            {"gamma": "ones" if scale else False, "beta": "zeros" if center else False},
            axis=self._axis,
            momentum=self._momentum,
            epsilon=self._epsilon,
            center=self._center,
            scale=self._scale,
        )

    def _normalized_axis(self, x) -> int:
        """The tensor axis of ``x`` normalized, counted with the batch as 0."""
        axis = x.shape.dim_axes[0] if self._axis is None else self._axis
        return axis % (x.shape.rank + 1)

    def _weight_shapes(self, x):
        size = (x.shape.tuple[self._normalized_axis(x) - 1],)
        return size, size

    def output_shape(self, *inputs):
        # Normalization preserves every axis, including a dynamic sequence axis
        # that cannot be probed with a dummy tensor.
        return inputs[0].shape

    def input_signature(self, xs) -> tuple:
        # The statistics hold one value for each position of the normalized
        # axis, whatever the length of the others: a dynamic sequence and a
        # fixed one share them.
        return (len(xs[0].shape), xs[0].shape[self._normalized_axis(self._preds[0])])

    def build_layer(self):
        return BatchNormImpl(
            axis=self._normalized_axis(self._preds[0]),
            momentum=self._momentum,
            epsilon=self._epsilon,
            center=self._center,
            scale=self._scale,
            name=self.name,
        )

    def _reinitialize(self) -> None:
        if self._layer is not None:
            self._layer.reset_statistics()
