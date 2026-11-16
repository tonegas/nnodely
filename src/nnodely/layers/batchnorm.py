"""Batch normalization layer for nnodely."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


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


class BatchNorm(Layer):
    r"""Batch normalization of a stream: every feature shifted to zero mean
    and scaled to unit variance, then scaled and offset by trained values.

    Syntax
    ------
    ``BatchNorm()(x)``
        Normalizes every feature of ``x``, with a trained scale and offset.
    ``BatchNorm(axis=k)(x)``
        Normalizes along the tensor axis ``k`` instead of the features.
    ``BatchNorm(center=False, scale=False)(x)``
        Normalizes only, with no trained scale or offset.

    The layer behaves differently in training and in inference:

    - In training, it normalizes with the mean and the variance of the
      batch. It also updates its moving mean and moving variance with them.
    - In inference, it normalizes with the moving mean and the moving
      variance. Calling the model on data and :meth:`Modely.validate` are
      inference.

    Parameters
    ----------
    axis : int or None, default None
        The axis whose positions each get their own statistics, scale and
        offset. ``None`` is the first feature axis. An int is a tensor axis,
        counted with the batch axis as 0: the sequence axes first, then the
        window, then the features. ``-1`` is the last feature axis.
    momentum : float, default 0.99
        How slowly the moving statistics follow the batches, between 0 and
        1. ``0`` replaces them with the statistics of the last batch.
    epsilon : float, default 1e-3
        A small number added to the variance, so that the division never
        is by zero.
    center : bool, default True
        ``True`` adds a trained offset, the Parameter ``<name>_beta``,
        starting from zeros. ``False`` leaves it out.
    scale : bool, default True
        ``True`` multiplies by a trained scale, the Parameter
        ``<name>_gamma``, starting from ones. ``False`` leaves it out.
    name : str, optional
        The name of the layer. The default is ``BatchNorm`` followed by a
        number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape, ``Shape(S, T, D)``, with or without a window.
        The call takes it alone, ``BatchNorm()(x)``, or in a list,
        ``BatchNorm()([x])``.

    Returns
    -------
    Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=2, D=(3,))`` for a
        window of 2 samples of 3 features.

    Raises
    ------
    ValueError
        If the layer is called on several streams ("reads one stream, got
        2").
    TypeError
        If the layer is called before it is configured, ``BatchNorm(x)``.

    See Also
    --------
    Softmax : Positive values that sum to one.
    Linear : Learned affine map of the features.
    Parameter : A trained value of the model.
    Modely.freeze : Keeps Parameters as they are through training.

    Notes
    -----
    For every position of the normalized axis, with :math:`\gamma` the
    scale, :math:`\beta` the offset and :math:`\epsilon` = ``epsilon``:

    .. math:: y = \gamma \, \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

    In training, :math:`\mu` and :math:`\sigma^2` are the mean and the
    variance of the batch, over every other axis: the batch, the sequence,
    the window and the other features. After each training step, with
    :math:`m` = ``momentum``, the moving statistics become

    .. math::

        \hat\mu \leftarrow m\,\hat\mu + (1 - m)\,\mu, \qquad
        \hat\sigma^2 \leftarrow m\,\hat\sigma^2 + (1 - m)\,\sigma^2

    In inference, :math:`\mu = \hat\mu` and :math:`\sigma^2 =
    \hat\sigma^2`. They start from 0 and 1, so an untrained layer divides
    by :math:`\sqrt{1 + \epsilon}`.

    Only the scale and the offset are trained. They are :class:`Parameter`
    objects: :attr:`Modely.parameters` lists them, and :meth:`Modely.freeze`
    freezes them. The moving statistics are not Parameters: the layer holds
    them, training updates them even when the Parameters are frozen, and
    they are saved with the model. ``Modely.build(keep_weights=False)``
    resets them to 0 and 1.

    Basic Examples
    --------------
    An untrained layer, in inference, divides by :math:`\sqrt{1.001}`:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import BatchNorm, DataLoader, Input, Linear, Modely, Output
    >>> x = Input("x", tw=3)
    >>> model = Modely("norm", inputs=[x], outputs=[Output("y", BatchNorm(name="bn")(x))]).build()
    >>> result = model({"x": np.array([[[1.0], [2.0], [3.0]]])})
    >>> np.round(keras.ops.convert_to_numpy(result["y"]), 4)
    array([[[0.9995],
            [1.999 ],
            [2.9985]]], dtype=float32)

    Its scale and offset are Parameters, one per feature:

    >>> [(p.name, p.shape.dim) for p in model.parameters]
    [('bn_gamma', (1,)), ('bn_beta', (1,))]

    Scope
    -----
    A vector stream has statistics for each feature:

    >>> v = Input("v", dim=3, tw=2)
    >>> y = BatchNorm(name="per_feature")(v)
    >>> y.shape
    Shape(S=(), T=2, D=(3,))
    >>> [(p.name, p.shape.dim) for p in Modely("f", inputs=[v], outputs=[Output("y", y)]).parameters]
    [('per_feature_gamma', (3,)), ('per_feature_beta', (3,))]

    A stream without a window, or with a sequence, keeps its shape:

    >>> BatchNorm()(Input("nw", dim=3)).shape
    Shape(S=(), T=None, D=(3,))
    >>> BatchNorm()(Input("s", dim=3, seq=-1, tw=2)).shape
    Shape(S=(dyn,), T=2, D=(3,))

    Options
    -------
    ``axis`` picks the tensor axis. On ``v``, axis 1 is the window: each of
    its 2 samples gets its own statistics:

    >>> y = BatchNorm(axis=1, name="per_sample")(v)
    >>> [(p.name, p.shape.dim) for p in Modely("w", inputs=[v], outputs=[Output("y", y)]).parameters]
    [('per_sample_gamma', (2,)), ('per_sample_beta', (2,))]

    ``center=False`` leaves out the offset, and ``scale=False`` the scale:

    >>> y = BatchNorm(center=False, name="no_offset")(v)
    >>> [p.name for p in Modely("c", inputs=[v], outputs=[Output("y", y)]).parameters]
    ['no_offset_gamma']
    >>> y = BatchNorm(center=False, scale=False)(v)
    >>> len(Modely("n", inputs=[v], outputs=[Output("y", y)]).parameters)
    0

    ``momentum`` and ``epsilon`` set the moving statistics and the division.
    With ``momentum=0``, one training step on a window of 1 and 3, of mean 2
    and variance 1, sets the moving statistics to them. With
    ``epsilon=0``, inference then gives exactly :math:`x - 2`:

    >>> w = Input("w", tw=2)
    >>> target = Input("target", tw=2)
    >>> y = BatchNorm(momentum=0.0, epsilon=0.0)(w)
    >>> model = Modely("stats", inputs=[w, target], outputs=[Output("y", y)])
    >>> model = model.minimize("fit", y, target).build()
    >>> sample = np.array([[[1.0], [3.0]]])
    >>> keras.ops.convert_to_numpy(model({"w": sample})["y"])  # mean 0, variance 1
    array([[[1.],
            [3.]]], dtype=float32)
    >>> data = DataLoader(model, {"w": sample, "target": np.array([[[-1.0], [1.0]]])}, windowed=True)
    >>> history = model.train(data, epochs=1, printer=None)
    >>> keras.ops.convert_to_numpy(model({"w": sample})["y"])  # mean 2, variance 1
    array([[[-1.],
            [ 1.]]], dtype=float32)

    The target here is the normalized batch, so the loss and the gradient
    are zero, and the scale and the offset stay 1 and 0.

    ``name`` names the layer and its Parameters:

    >>> y = BatchNorm(name="normalize")(x)
    >>> [p.name for p in Modely("named", inputs=[x], outputs=[Output("y", y)]).parameters]
    ['normalize_gamma', 'normalize_beta']

    Applications
    ------------
    Signals of very different scales, such as a position in metres and a
    current in milliamperes, normalized before a network reads them:

    >>> signals = Input("signals", dim=2, tw=1)
    >>> hidden = Linear(features=8)(BatchNorm()(signals))

    Possible Issues
    ---------------
    Two streams are not normalized together. Apply one layer to each, or
    join them first:

    >>> a, b = Input("a", tw=1), Input("b", tw=1)
    >>> BatchNorm()([a, b])
    Traceback (most recent call last):
        ...
    ValueError: ...BatchNorm reads one stream, got 2...

    The moving statistics are not Parameters. :meth:`Modely.freeze` keeps
    the scale and the offset, while training still updates the statistics,
    and so changes the inference output.

    .. versionadded:: 2.0
    """

    _weight_labels = ("gamma", "beta")
    _weight_defaults = {"gamma": "ones", "beta": "zeros"}

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
            weights={
                "gamma": "ones" if scale else False,
                "beta": "zeros" if center else False,
            },
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
