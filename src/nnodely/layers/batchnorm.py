"""Batch normalization layer wrapper for nnodely."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


class BatchNorm(Layer):
    """
    Wrapper for keras.layers.BatchNormalization.

    Input and output::

        [batch, *seq, time, *dim]

    Statistics are accumulated per feature: by default along the first dim
    axis. ``axis`` picks another tensor axis, counted with the batch as 0.
    """

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
            name=name,
            axis=self._axis,
            momentum=self._momentum,
            epsilon=self._epsilon,
            center=self._center,
            scale=self._scale,
        )

    def output_shape(self, *inputs):
        # Normalization preserves every axis, including a dynamic sequence axis
        # that cannot be probed with a dummy tensor.
        return inputs[0].shape

    def build_layer(self):
        axis = self._axis
        if axis is None:
            axis = self._preds[0].shape.dim_axes[0]  # type: ignore[union-attr]
        return keras.layers.BatchNormalization(
            axis=axis,
            momentum=self._momentum,
            epsilon=self._epsilon,
            center=self._center,
            scale=self._scale,
            name=self.name,
        )
