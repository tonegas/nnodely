"""Finite impulse response layer."""

from __future__ import annotations

import keras

from nnodely.core.shape import BATCH_AXIS, Shape, transpose
from nnodely.core.layer import Layer
from nnodely.layers.parameter import Parameter, _check_weight_size


@keras.saving.register_keras_serializable(package="nnodely")
class FirImpl(keras.layers.Layer):
    """A filter over the window of every element, at every step of the
    sequence.

    Each element of the dim axes has the window projected onto
    ``features`` channels, with a kernel of its own or, with
    ``shared_kernel``, one kernel shared by all of them; elements are never
    mixed. ``(batch, *seq, window, *dim)`` becomes
    ``(batch, *seq, 1, *dim, features)``, the channel axis dropped when
    there is one channel and a scalar dim ``(1,)`` dropped when there are more.

    It owns no weight: it is called on ``[x, kernel]`` or ``[x, kernel,
    bias]``, the values of the Parameters holding them, and reads their first
    sample.
    """

    def __init__(self, features, shared_kernel=False, seq_rank=0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.features = int(features)
        self.shared_kernel = bool(shared_kernel)
        self.seq_rank = int(seq_rank)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "shared_kernel": self.shared_kernel,
                "seq_rank": self.seq_rank,
            }
        )
        return config

    def build(self, input_shape):
        shape = Shape.from_tensor_shape(input_shape[0][1:], seq_rank=self.seq_rank)
        dim = tuple(int(axis) for axis in shape.dim)
        # A scalar has a single element: its kernel is the shared one.
        elements = () if self.shared_kernel or dim == (1,) else dim
        self.kernel_shape = (*elements, shape.tw, self.features)
        self.bias_shape = (*elements, self.features)
        for label, shape, expected in zip(
            ("kernel", "bias"), input_shape[1:], (self.kernel_shape, self.bias_shape)
        ):
            _check_weight_size(self.name, label, shape, expected)
        super().build(input_shape)

    def call(self, inputs):
        x, kernel, *bias = inputs
        kernel = keras.ops.reshape(kernel[0], self.kernel_shape)

        shape = Shape.from_tensor_shape(x.shape[1:], seq_rank=self.seq_rank)
        dim = tuple(int(axis) for axis in shape.dim)
        seq = shape.seq_axes

        # (batch, *seq, *dim, 1, window): the window of every element is
        # multiplied by its kernel, so neither the batch nor a sequence axis
        # left dynamic has to be reshaped.
        x = transpose(x, [BATCH_AXIS, *seq, *shape.dim_axes, shape.tw_axis])
        y = keras.ops.matmul(keras.ops.expand_dims(x, -2), kernel)
        y = keras.ops.squeeze(y, -2)
        if bias:
            y = y + keras.ops.reshape(bias[0][0], self.bias_shape)

        # (batch, *seq, *dim, out) is laid out as the output already, which
        # drops the channel axis when there is one channel and a scalar dim
        # when there are more; it lacks only its window, one step long.
        if self.features == 1:
            y = keras.ops.squeeze(y, len(y.shape) - 1)
        elif dim == (1,):
            y = keras.ops.squeeze(y, 1 + len(seq))
        return keras.ops.expand_dims(y, 1 + len(seq))


class Fir(Layer):
    r"""Learned weighted sum over the window of a stream: one sample per window.

    Syntax
    ------
    ``Fir(features=1)(x)``
        Filters every feature of ``x`` over its window, into one sample.
    ``Fir(features=n)(x)``
        Gives every feature ``n`` output channels, each with its own filter.
    ``Fir(kernel=k, bias=False)(x)``
        Filters with the coefficients the :class:`Parameter` ``k`` holds.

    The window of ``x``, ``T`` samples, becomes one sample. Each feature is
    filtered over its own window, and features are never mixed: use
    :class:`Linear` to combine them. The sequence axes are kept, and the
    filter is applied at every step.

    Parameters
    ----------
    features : int, default 1
        The output channels of every feature. ``1`` keeps the features of
        ``x``.
    kernel : str, keras.initializers.Initializer or Parameter, default "glorot_uniform"
        The coefficients. An initializer, by name or object, draws them; a
        :class:`Parameter` of shape ``(*D, T, features)`` holds them. Row 0
        multiplies the oldest sample of the window. A scalar ``x``, ``D=(1,)``,
        or a shared kernel leaves out ``*D``.
    bias : bool, str, keras.initializers.Initializer or Parameter, default True
        The offset added to every channel. ``True`` starts it from zeros,
        ``False`` leaves it out. An initializer, or a Parameter of shape
        ``(*D, features)``, sets it.
    shared_kernel : bool, default False
        ``True`` filters every feature with the same kernel and bias.
    name : str, optional
        The name of the layer. Its weights are named ``<name>_kernel`` and
        ``<name>_bias``. The default is ``Fir`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream with a window, ``Shape(S, T, D)``. The call takes it alone,
        ``Fir()(x)``, or in a list, ``Fir()([x])``.

    Returns
    -------
    Stream
        ``Shape(S, 1, D)`` with ``features=1``. With more channels,
        ``Shape(S, 1, (*D, features))``, or ``Shape(S, 1, (features,))``
        for a scalar ``x``. For example, a window of 5 samples of 3 features,
        ``Shape(S=(), T=5, D=(3,))``, gives ``D=(3,)``, and ``D=(3, 2)``
        with ``features=2``.

    Raises
    ------
    ValueError
        If ``x`` has no window ("Fir acts on the window of its stream, and
        'x' has none"), or if a kernel or bias Parameter does not have the
        shape of the weight ("kernel must have the shape ...").
    TypeError
        If the layer is called before it is configured, ``Fir(x)``.

    See Also
    --------
    Linear : Affine map over the features of every sample.
    TimeSelect : One sample of a window.
    TimeRange : A part of a window.
    LocalModel : Filters blended by operating region.

    Notes
    -----
    For one feature and one channel, with the window :math:`x_0, \dots,
    x_{T-1}` from the oldest sample to the newest:

    .. math:: y = \sum_{i=0}^{T-1} k_i \, x_i + b

    The kernel and the bias are :class:`Parameter` objects. They are listed
    by :attr:`Modely.parameters`, and are trained, frozen and saved with the
    model. After :meth:`Modely.build`, the stream the layer returns exposes
    their Keras variables as ``kernel`` and ``bias``.

    Basic Examples
    --------------
    A weighted sum of the last three samples, with known coefficients:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Fir, Input, Linear, Modely, Output, Parameter
    >>> x = Input("x", tw=3)
    >>> k = Parameter("k", value=[[0.25], [0.25], [0.5]])  # oldest sample first
    >>> y = Fir(kernel=k, bias=False)(x)
    >>> model = Modely("fir_demo", inputs=[x], outputs=[Output("y", y)]).build()
    >>> result = model({"x": np.array([[[1.0], [2.0], [3.0]]])})
    >>> keras.ops.convert_to_numpy(result["y"])  # 0.25*1 + 0.25*2 + 0.5*3
    array([[[2.25]]], dtype=float32)

    Scope
    -----
    A window of several features is filtered feature by feature:

    >>> v = Input("v", dim=3, tw=4)
    >>> Fir()(v).shape
    Shape(S=(), T=1, D=(3,))

    A sequence keeps its axes, the filter applied at every step:

    >>> Fir()(Input("s", seq=5, tw=4)).shape
    Shape(S=(5,), T=1, D=(1,))

    Options
    -------
    ``features`` adds an axis of that many channels, which replaces the
    feature of a scalar:

    >>> Fir(features=2)(v).shape
    Shape(S=(), T=1, D=(3, 2))
    >>> Fir(features=2)(x).shape
    Shape(S=(), T=1, D=(2,))

    ``kernel`` and ``bias`` take an initializer by name:

    >>> y = Fir(kernel="zeros", bias="ones")(x)

    ``shared_kernel=True`` filters the 3 features of ``v`` with one kernel of
    4 coefficients, instead of one kernel each:

    >>> shared = Fir(shared_kernel=True, name="shared")(v)
    >>> own = Fir(name="own")(v)
    >>> model = Modely("kernels", inputs=[v], outputs=[Output("a", shared), Output("b", own)])
    >>> [(p.name, p.shape.dim) for p in model.parameters]
    [('shared_kernel', (4, 1)), ('shared_bias', (1,)), ('own_kernel', (3, 4, 1)), ('own_bias', (3, 1))]

    ``name`` names the layer and its weights:

    >>> y = Fir(name="filter")(x)
    >>> model = Modely("named", inputs=[x], outputs=[Output("y", y)])
    >>> [p.name for p in model.parameters]
    ['filter_kernel', 'filter_bias']

    Applications
    ------------
    The discretized mass-spring-damper predicts the next position from the
    last positions and the current force. Its structure is two filters:

    >>> position = Input("position", tw=4)
    >>> force = Input("force", tw=1)
    >>> next_position = Fir()(position) + Fir(bias=False)(force)
    >>> next_position.shape
    Shape(S=(), T=1, D=(1,))

    Properties & Relations
    ----------------------
    On a window of one sample, a filter is the :class:`Linear` map of that
    sample:

    >>> w = Input("w", tw=1)
    >>> fir = Fir(kernel=Parameter("k_fir", value=[[2.0]]), bias=False)(w)
    >>> linear = Linear(kernel=Parameter("k_lin", value=[[2.0]]), bias=False)(w)
    >>> model = Modely("same", inputs=[w], outputs=[Output("fir", fir), Output("lin", linear)])
    >>> result = model.build()({"w": np.array([[[3.0]]])})
    >>> keras.ops.convert_to_numpy(result["fir"]), keras.ops.convert_to_numpy(result["lin"])
    (array([[[6.]]], dtype=float32), array([[[6.]]], dtype=float32))

    Possible Issues
    ---------------
    A stream without a window cannot be filtered. Declare a window on the
    Input:

    >>> Fir()(Input("current"))
    Traceback (most recent call last):
        ...
    ValueError: ...Fir acts on the window of its stream, and 'current' has none...
    >>> Fir()(Input("current", tw=1)).shape
    Shape(S=(), T=1, D=(1,))

    The layer is configured first, then called on its stream:

    >>> Fir(x)
    Traceback (most recent call last):
        ...
    TypeError: Fir has settings, so it is configured first and then called on its streams: Fir(...)(x), not Fir(x).

    .. versionadded:: 2.0
    """

    _weight_labels = ("kernel", "bias")
    _weight_defaults = {"kernel": "glorot_uniform", "bias": "zeros"}

    def __init__(
        self,
        features: int = 1,
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        shared_kernel: bool = False,
        name=None,
    ):
        self._features = int(features)
        self._shared_kernel = bool(shared_kernel)
        super().__init__(
            name,
            weights={"kernel": kernel, "bias": bias},
            kernel=kernel,
            bias=bias,
            features=self._features,
            shared_kernel=self._shared_kernel,
        )

    def _weight_shapes(self, x):
        # The kernel runs along the window: there has to be one.
        self._needs_window(x)
        dim = tuple(x.dim)
        elements = () if self._shared_kernel or dim == (1,) else dim
        return (*elements, x.shape.tw, self._features), (
            *elements,
            self._features,
        )

    def _draw_shapes(self, shapes):
        # The filter and the bias of each element alone, as for a scalar.
        kernel_shape, bias_shape = shapes
        return kernel_shape[-2:], bias_shape[-1:]

    def output_shape(self, *inputs):
        # Declared rather than probed with a dummy tensor, which a tw axis
        # left dynamic cannot be given.
        dim = tuple(inputs[0].dim)
        if self._features == 1:
            out_dim = dim
        elif dim == (1,):
            out_dim = (self._features,)
        else:
            out_dim = (*dim, self._features)
        # The window becomes one sample; the sequence is kept.
        shape = inputs[0].shape
        return shape.replace(dim=out_dim, tw=1)

    @property
    def kernel(self):
        """The Keras variable of the Parameter holding the kernel, in that
        Parameter's shape; None before :meth:`Modely.build`."""
        return self._weight_value("kernel")

    @property
    def bias(self):
        """The Keras variable of the Parameter holding the bias, in that
        Parameter's shape; None without a bias or before
        :meth:`Modely.build`."""
        return self._weight_value("bias")

    def build_layer(self):
        return FirImpl(
            features=self._features,
            shared_kernel=self._shared_kernel,
            seq_rank=self._preds[0].shape.seq_rank,  # type: ignore[union-attr]
            name=self.name,
        )
