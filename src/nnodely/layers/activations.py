"""Activation layers wrappers for nnodely."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer


class ReLU(Layer):
    r"""Rectified linear unit of every value of a stream: ``max(x, 0)``.

    Syntax
    ------
    ``ReLU()(x)``
        The rectified linear unit of ``x``: ``x`` where it is positive, 0
        elsewhere.
    ``ReLU(max_value=m, negative_slope=a, threshold=t)(x)``
        The generalized form: capped at ``m``, with slope ``a`` below ``t``.
    ``ReLU()([a, b])``
        The rectified linear unit of each stream of the list, as a list.

    The layer acts on every value alone, whatever the shape of the stream.
    It has no weights.

    Parameters
    ----------
    max_value : float or None, default None
        The largest output value. ``None`` leaves the output unbounded. A
        number must be 0 or more.
    negative_slope : float, default 0.0
        The slope of the output at and below ``threshold``. It must be 0 or
        more.
    threshold : float, default 0.0
        The value at and below which the output follows ``negative_slope``.
        It must be 0 or more.
    name : str, optional
        The name of the layer. The default is ``ReLU`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(4,), T=2, D=(3,))`` for a
        sequence of 4 windows of 2 samples of 3 features. A list of streams
        gives a list, in its order.

    Raises
    ------
    ValueError
        If ``max_value``, ``negative_slope`` or ``threshold`` is negative
        ("cannot be a negative value"). The layer raises when it is called.
    TypeError
        If the layer is called before it is configured, ``ReLU(x)``.

    See Also
    --------
    LeakyReLU : A fixed slope below zero.
    PReLU : A trained slope below zero, one per feature.
    ELU : An exponential below zero.
    Softplus : A smooth approximation of the ReLU.

    Notes
    -----
    With :math:`m` = ``max_value``, :math:`a` = ``negative_slope`` and
    :math:`t` = ``threshold``:

    .. math::

        y = \begin{cases}
            m & x \ge m \\
            x & t < x < m \\
            a\,(x - t) & x \le t
        \end{cases}

    The defaults give :math:`y = \max(x, 0)`. The gradient is 1 between
    :math:`t` and :math:`m`, :math:`a` below :math:`t` and 0 above
    :math:`m`. With :math:`t > 0` the output jumps from :math:`0` to
    :math:`t` at :math:`x = t`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, LeakyReLU, Linear, Modely, Output, ReLU
    >>> x = Input("x", dim=5)
    >>> model = Modely("relu", inputs=[x], outputs=[Output("y", ReLU()(x))]).build()
    >>> data = {"x": np.array([[-2.0, -1.0, 0.0, 1.0, 3.0]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[0., 0., 0., 1., 3.]], dtype=float32)

    Scope
    -----
    Every value is rectified, whatever the shape:

    >>> ReLU()(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives the rectified linear unit of each stream:

    >>> a, b = Input("a", tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in ReLU()([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    The function ``apply`` runs a layer on one sample of ``x``:

    >>> def apply(layer, values=(-2.0, -1.0, 0.0, 1.0, 3.0)):
    ...     model = Modely("demo", inputs=[x], outputs=[Output("y", layer(x))])
    ...     return keras.ops.convert_to_numpy(model.build()({"x": np.array([values])})["y"])

    ``max_value`` caps the output:

    >>> apply(ReLU(max_value=2.0))
    array([[0., 0., 0., 1., 2.]], dtype=float32)

    ``negative_slope`` multiplies the values below the threshold:

    >>> apply(ReLU(negative_slope=0.5))
    array([[-1. , -0.5,  0. ,  1. ,  3. ]], dtype=float32)

    ``threshold`` moves the corner: here the values up to 1 give 0.

    >>> apply(ReLU(threshold=1.0), (0.0, 0.5, 1.0, 2.0, 3.0))
    array([[0., 0., 0., 2., 3.]], dtype=float32)

    ``name`` names the layer:

    >>> ReLU(name="rectifier")(x).name
    'rectifier'

    Applications
    ------------
    A contact force pushes only while two bodies overlap. With the gap
    between them, negative when they overlap, and a stiffness of 1000 N/m:

    >>> gap = Input("gap", tw=1)
    >>> force = 1000.0 * ReLU()(-gap)

    A hidden layer of a small network:

    >>> velocity = Input("velocity", tw=1)
    >>> drag = Linear(features=1)(ReLU()(Linear(features=8)(velocity)))

    Properties & Relations
    ----------------------
    ``ReLU(negative_slope=a)`` is ``LeakyReLU(negative_slope=a)``:

    >>> apply(LeakyReLU(negative_slope=0.5))
    array([[-1. , -0.5,  0. ,  1. ,  3. ]], dtype=float32)

    Possible Issues
    ---------------
    The layer is configured first, then called on its stream:

    >>> ReLU(x)
    Traceback (most recent call last):
        ...
    TypeError: ReLU has settings, so it is configured first and then called on its streams: ReLU(...)(x), not ReLU(x).
    >>> ReLU()(x).shape
    Shape(S=(), T=None, D=(5,))

    A negative setting raises when the layer is called:

    >>> ReLU(max_value=-1.0)(x)
    Traceback (most recent call last):
        ...
    ValueError: max_value of a ReLU layer cannot be a negative value...

    .. versionadded:: 2.0
    """

    def __init__(
        self,
        max_value: float | None = None,
        negative_slope: float = 0.0,
        threshold: float = 0.0,
        name=None,
    ):
        self._max_value = max_value
        self._negative_slope = float(negative_slope)
        self._threshold = float(threshold)
        super().__init__(
            name=name,
            max_value=self._max_value,
            negative_slope=self._negative_slope,
            threshold=self._threshold,
        )

    def build_layer(self):
        return keras.layers.ReLU(
            max_value=self._max_value,
            negative_slope=self._negative_slope,
            threshold=self._threshold,
            name=self.name,
        )


class LeakyReLU(Layer):
    r"""Leaky rectified linear unit of every value of a stream: ``x`` where it
    is positive, ``negative_slope * x`` elsewhere.

    Syntax
    ------
    ``LeakyReLU()(x)``
        The leaky rectified linear unit of ``x``, with a slope of 0.3 below
        zero.
    ``LeakyReLU(negative_slope=a)(x)``
        The same, with a slope of ``a`` below zero.
    ``LeakyReLU()([a, b])``
        The leaky rectified linear unit of each stream of the list, as a list.

    Unlike :class:`ReLU`, the output below zero is not flat, so the gradient
    never vanishes there. The slope is fixed: :class:`PReLU` trains it.

    Parameters
    ----------
    negative_slope : float, default 0.3
        The slope of the output below zero. It must be 0 or more.
    name : str, optional
        The name of the layer. The default is ``LeakyReLU`` followed by a
        number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=None, D=(5,))`` for
        one sample of 5 features. A list of streams gives a list, in its
        order.

    Raises
    ------
    ValueError
        If ``negative_slope`` is negative ("cannot be None or negative
        value"). The layer raises when it is called.
    TypeError
        If the layer is called before it is configured, ``LeakyReLU(x)``.

    See Also
    --------
    ReLU : Zero below zero, with an optional cap and threshold.
    PReLU : A trained slope below zero, one per feature.
    ELU : An exponential below zero.

    Notes
    -----
    With :math:`a` = ``negative_slope``:

    .. math::

        y = \begin{cases} x & x > 0 \\ a\,x & x \le 0 \end{cases}

    The gradient is 1 above zero and :math:`a` below.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, LeakyReLU, Linear, Modely, Output, ReLU
    >>> x = Input("x", dim=5)
    >>> model = Modely("leaky", inputs=[x], outputs=[Output("y", LeakyReLU()(x))]).build()
    >>> data = {"x": np.array([[-2.0, -1.0, 0.0, 1.0, 3.0]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"])  # -0.3 * 2, -0.3 * 1
    array([[-0.6, -0.3,  0. ,  1. ,  3. ]], dtype=float32)

    Scope
    -----
    Every value is transformed, whatever the shape:

    >>> LeakyReLU()(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives one stream for each:

    >>> a, b = Input("a", tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in LeakyReLU()([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``negative_slope`` sets the slope below zero:

    >>> y = LeakyReLU(negative_slope=0.5)(x)
    >>> model = Modely("half", inputs=[x], outputs=[Output("y", y)]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[-1. , -0.5,  0. ,  1. ,  3. ]], dtype=float32)

    ``name`` names the layer:

    >>> LeakyReLU(name="leaky")(x).name
    'leaky'

    Applications
    ------------
    A hidden layer of a small network, which keeps a gradient for negative
    inputs:

    >>> velocity = Input("velocity", tw=1)
    >>> drag = Linear(features=1)(LeakyReLU()(Linear(features=8)(velocity)))

    Properties & Relations
    ----------------------
    ``LeakyReLU(negative_slope=a)`` is ``ReLU(negative_slope=a)``:

    >>> y = ReLU(negative_slope=0.5)(x)
    >>> model = Modely("same", inputs=[x], outputs=[Output("y", y)]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[-1. , -0.5,  0. ,  1. ,  3. ]], dtype=float32)

    Possible Issues
    ---------------
    The layer is configured first, then called on its stream:

    >>> LeakyReLU(x)
    Traceback (most recent call last):
        ...
    TypeError: LeakyReLU has settings, so it is configured first and then called on its streams: LeakyReLU(...)(x), not LeakyReLU(x).

    .. versionadded:: 2.0
    """

    def __init__(self, negative_slope: float = 0.3, name=None):
        self._negative_slope = float(negative_slope)
        super().__init__(name=name, negative_slope=self._negative_slope)

    def build_layer(self):
        return keras.layers.LeakyReLU(
            negative_slope=self._negative_slope,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class ELUImpl(keras.layers.Layer):
    def __init__(self, alpha: float = 1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)

    def call(self, inputs):
        zero = keras.ops.cast(0.0, inputs.dtype)
        alpha = keras.ops.cast(self.alpha, inputs.dtype)
        # exp of the negative part only: exp(x) of a large x overflows, and
        # its infinite gradient poisons the branch where() does not take.
        return keras.ops.where(
            inputs > zero,
            inputs,
            alpha * (keras.ops.exp(keras.ops.minimum(inputs, zero)) - 1.0),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class ELU(Layer):
    r"""Exponential linear unit of every value of a stream: ``x`` where it is
    positive, ``alpha * (exp(x) - 1)`` elsewhere.

    Syntax
    ------
    ``ELU()(x)``
        The exponential linear unit of ``x``, which tends to -1 below zero.
    ``ELU(alpha=a)(x)``
        The same, tending to ``-a`` below zero.
    ``ELU()([a, b])``
        The exponential linear unit of each stream of the list, as a list.

    The output is smooth at zero for ``alpha=1``, and bounded below by
    ``-alpha``.

    Parameters
    ----------
    alpha : float, default 1.0
        The value the output tends to, with a minus sign, for large negative
        ``x``.
    name : str, optional
        The name of the layer. The default is ``ELU`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=None, D=(5,))`` for
        one sample of 5 features. A list of streams gives a list, in its
        order.

    Raises
    ------
    TypeError
        If the layer is called before it is configured, ``ELU(x)``.

    See Also
    --------
    ReLU : Zero below zero.
    LeakyReLU : A fixed slope below zero.
    GELU : Gaussian error linear unit.
    Softplus : A smooth approximation of the ReLU.

    Notes
    -----
    With :math:`\alpha` = ``alpha``:

    .. math::

        y = \begin{cases} x & x > 0 \\ \alpha\,(e^{x} - 1) & x \le 0 \end{cases}

    The gradient is 1 above zero and :math:`\alpha e^{x}` below. The
    exponential is computed on the negative part alone, so a large
    positive ``x`` neither overflows nor gives an infinite gradient.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import ELU, Input, Linear, Modely, Output
    >>> x = Input("x", dim=5)
    >>> model = Modely("elu", inputs=[x], outputs=[Output("y", ELU()(x))]).build()
    >>> data = {"x": np.array([[-2.0, -1.0, 0.0, 1.0, 3.0]])}
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)  # exp(-1) - 1
    array([[-0.8647, -0.6321,  0.    ,  1.    ,  3.    ]], dtype=float32)

    Scope
    -----
    Every value is transformed, whatever the shape:

    >>> ELU()(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives one stream for each:

    >>> a, b = Input("a", tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in ELU()([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``alpha`` scales the negative part:

    >>> model = Modely("elu2", inputs=[x], outputs=[Output("y", ELU(alpha=2.0)(x))]).build()
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)
    array([[-1.7293, -1.2642,  0.    ,  1.    ,  3.    ]], dtype=float32)

    ``name`` names the layer:

    >>> ELU(name="elu")(x).name
    'elu'

    Applications
    ------------
    A hidden layer of a small network:

    >>> velocity = Input("velocity", tw=1)
    >>> drag = Linear(features=1)(ELU()(Linear(features=8)(velocity)))

    Possible Issues
    ---------------
    The layer is configured first, then called on its stream:

    >>> ELU(x)
    Traceback (most recent call last):
        ...
    TypeError: ELU has settings, so it is configured first and then called on its streams: ELU(...)(x), not ELU(x).

    .. versionadded:: 2.0
    """

    def __init__(self, alpha: float = 1.0, name=None):
        self._alpha = float(alpha)
        super().__init__(name=name, alpha=self._alpha)

    def build_layer(self):
        return ELUImpl(alpha=self._alpha, name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class PReLUImpl(keras.layers.Layer):
    """``x`` where it is positive and ``alpha * x`` elsewhere, as
    keras.layers.PReLU computes it. It is called on ``[x, alpha]``: the
    slopes are the value of a Parameter, read whole."""

    def call(self, inputs):
        x, alpha = inputs
        return keras.ops.relu(x) - alpha[0] * keras.ops.relu(-x)


class PReLU(Layer):
    r"""Parametric rectified linear unit of every value of a stream: ``x``
    where it is positive, ``alpha * x`` elsewhere, with ``alpha`` trained.

    Syntax
    ------
    ``PReLU()(x)``
        Trains one slope below zero for every feature of ``x``.
    ``PReLU(shared_axes=k)(x)``
        Shares the slopes along the tensor axes ``k`` instead.
    ``PReLU()([a, b])``
        Applies the same slopes to each stream of the list, as a list.

    The slopes ``alpha`` are a :class:`Parameter`: it can be read, frozen and
    saved like any other. They start from zero, so an untrained PReLU is a
    :class:`ReLU`.

    Parameters
    ----------
    shared_axes : int, list of int or None, default None
        The tensor axes along which one slope is shared, counted with the
        batch axis as 0: the sequence axes first, then the window, then the
        features. ``None`` shares the slopes along the sequence and the
        window, so there is one slope per feature.
    name : str, optional
        The name of the layer. Its slopes are named ``<name>_alpha``. The
        default is ``PReLU`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams of one shape.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=2, D=(3,))`` for a
        window of 2 samples of 3 features. A list of streams gives a list, in
        its order.

    Raises
    ------
    ValueError
        If ``x`` has a dynamic sequence and ``shared_axes`` does not share
        the slopes along it ("PReLU learns a slope for every step of the
        dynamic sequence").
    TypeError
        If the layer is called before it is configured, ``PReLU(x)``.

    See Also
    --------
    ReLU : Zero below zero.
    LeakyReLU : A fixed slope below zero.
    Parameter : A trained value of the model.
    Modely.parameters : The Parameters of a model.

    Notes
    -----
    .. math::

        y = \begin{cases} x & x > 0 \\ \alpha\,x & x \le 0 \end{cases}

    The gradient is 1 above zero and :math:`\alpha` below. The gradient
    with respect to :math:`\alpha` is :math:`\min(x, 0)`.

    The Parameter ``<name>_alpha`` is laid out as a sample of ``x``, with
    the shared axes of size 1. A window of 2 samples of 3 features,
    ``(2, 3)``, has slopes of shape ``(1, 3)`` by default. A stream without
    a window of 3 features has slopes of shape ``(3,)``.
    :attr:`Modely.parameters` lists it, and :meth:`Modely.freeze` freezes
    it. After :meth:`Modely.build`, its ``param`` attribute is the Keras
    variable holding the current slopes.

    Basic Examples
    --------------
    The slopes start from zero:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Linear, Modely, Output, PReLU
    >>> x = Input("x", dim=3, tw=2)
    >>> model = Modely("prelu", inputs=[x], outputs=[Output("y", PReLU(name="act")(x))]).build()
    >>> data = {"x": np.array([[[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[[0., 2., 0.],
            [4., 0., 6.]]], dtype=float32)

    :attr:`Modely.parameters` lists the slopes, one per feature:

    >>> print(model.parameters)
    ┏━━━━━┳━━━━━━━━━━━┳━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
    ┃ idx ┃ parameter ┃ seq ┃ tw ┃ dim    ┃ values ┃ trainable ┃
    ┡━━━━━╇━━━━━━━━━━━╇━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
    │ 0   │ act_alpha │ ()  │ -  │ (1, 3) │ 3      │ yes       │
    ├─────┼───────────┼─────┼────┼────────┼────────┼───────────┤
    │     │ total     │     │    │        │ 3      │ 3         │
    └─────┴───────────┴─────┴────┴────────┴────────┴───────────┘
    >>> alpha = model.parameters["act_alpha"]
    >>> alpha.value_numpy
    array([[0., 0., 0.]], dtype=float32)

    Training changes them. Here they are set by hand, one per feature:

    >>> _ = alpha.param.assign(np.array([[0.5, 0.25, 0.0]], dtype="float32"))
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[[-0.5 ,  2.  ,  0.  ],
            [ 4.  , -1.25,  6.  ]]], dtype=float32)

    Scope
    -----
    A sequence, a dynamic one included, shares the slopes of every step:

    >>> s = Input("s", dim=3, seq=-1, tw=2)
    >>> PReLU()(s).shape
    Shape(S=(dyn,), T=2, D=(3,))

    A list applies the same slopes to each stream:

    >>> a, b = Input("a", dim=3, tw=1), Input("b", dim=3, tw=1)
    >>> ya, yb = PReLU(name="shared")([a, b])
    >>> model = Modely("both", inputs=[a, b], outputs=[Output("ya", ya), Output("yb", yb)])
    >>> [p.name for p in model.parameters]
    ['shared_alpha']

    Options
    -------
    ``shared_axes`` picks the axes that share a slope. On ``x``, axis 1 is
    the window and axis 2 the features. ``shared_axes=2`` gives one slope
    per sample of the window, and ``[1, 2]`` one slope for all:

    >>> per_sample = PReLU(shared_axes=2, name="per_sample")(x)
    >>> one = PReLU(shared_axes=[1, 2], name="one")(x)
    >>> model = Modely("axes", inputs=[x], outputs=[Output("a", per_sample), Output("b", one)])
    >>> [(p.name, p.shape.dim) for p in model.parameters]
    [('per_sample_alpha', (2, 1)), ('one_alpha', (1, 1))]

    ``name`` names the layer and its slopes:

    >>> y = PReLU(name="slopes")(x)
    >>> [p.name for p in Modely("named", inputs=[x], outputs=[Output("y", y)]).parameters]
    ['slopes_alpha']

    Applications
    ------------
    A hidden layer of a small network, its slopes trained with the rest:

    >>> velocity = Input("velocity", tw=1)
    >>> drag = Linear(features=1)(PReLU()(Linear(features=8)(velocity)))

    Possible Issues
    ---------------
    One slope per step of a dynamic sequence cannot be made: share the
    slopes along the sequence axis, axis 1:

    >>> PReLU(shared_axes=3)(s)
    Traceback (most recent call last):
        ...
    ValueError: ...PReLU learns a slope for every step of the dynamic sequence of 's'...
    >>> PReLU(shared_axes=[1, 3])(s).shape
    Shape(S=(dyn,), T=2, D=(3,))

    The layer is configured first, then called on its stream:

    >>> PReLU(x)
    Traceback (most recent call last):
        ...
    TypeError: PReLU has settings, so it is configured first and then called on its streams: PReLU(...)(x), not PReLU(x).

    .. versionadded:: 2.0
    """

    _weight_labels = ("alpha",)
    _weight_defaults = {"alpha": "zeros"}

    def __init__(self, shared_axes=None, name=None):
        self._shared_axes = shared_axes
        super().__init__(
            name, weights={"alpha": "zeros"}, shared_axes=self._shared_axes
        )

    def _weight_shapes(self, x):
        shared_axes = self._shared_axes
        if shared_axes is None:
            window = [] if x.shape.tw_axis is None else [x.shape.tw_axis]
            shared_axes = [*x.shape.seq_axes, *window]
        elif isinstance(shared_axes, int):
            shared_axes = [shared_axes]
        shape = list(x.shape.tuple)
        for axis in shared_axes:
            shape[axis - 1] = 1
        if None in shape:
            raise ValueError(
                f"{self.name}: PReLU learns a slope for every step of the "
                f"dynamic sequence of {x.name!r}, {x.shape}, whose length is not "
                "known: share the slopes along it, with shared_axes."
            )
        return (tuple(shape),)

    def output_shape(self, *inputs):
        return inputs[0].shape

    def build_layer(self):
        return PReLUImpl(name=self.name)


class Softmax(Layer):
    r"""Softmax of a stream: positive values that sum to one, by default over
    each whole sample.

    Syntax
    ------
    ``Softmax()(x)``
        Normalizes every sample of ``x`` as a whole: its values over the
        sequence, the window and the features sum to one.
    ``Softmax(axis=-1)(x)``
        Normalizes along the last feature axis: the features of every
        sample of the window sum to one.
    ``Softmax()([a, b])``
        The softmax of each stream of the list, as a list.

    The default is not the softmax over the features. A stream with a
    window of 2 samples of 2 features has 4 values that sum to one. Use
    ``axis=-1`` for the softmax of each feature vector.

    Parameters
    ----------
    axis : int or None, default None
        The axis normalized. ``None`` normalizes all the values of a sample
        together. An int is a tensor axis, counted with the batch axis as 0:
        the sequence axes first, then the window, then the features. ``-1``
        is the last feature axis.
    name : str, optional
        The name of the layer. The default is ``Softmax`` followed by a
        number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=2, D=(2,))`` for a
        window of 2 samples of 2 features. A list of streams gives a list,
        in its order.

    Raises
    ------
    ValueError
        If ``axis`` is not an axis of ``x`` ("automatic output shape
        inference failed").
    TypeError
        If the layer is called before it is configured, ``Softmax(x)``.

    See Also
    --------
    Sigmoid : Logistic function of every value alone.
    Fuzzify : Memberships that can be normalized to sum to one.
    LocalModel : Local models blended by weights such as these.

    Notes
    -----
    Over the values :math:`x_j` normalized together:

    .. math:: y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

    Every output is in :math:`(0, 1)`. Adding the same number to every
    :math:`x_j` does not change the result.

    Basic Examples
    --------------
    The 4 values of a window of 2 samples of 2 features sum to one:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Linear, Modely, Output, Softmax
    >>> x = Input("x", dim=2, tw=2)
    >>> model = Modely("softmax", inputs=[x], outputs=[Output("y", Softmax()(x))]).build()
    >>> data = {"x": np.array([[[1.0, 1.0], [2.0, 2.0]]])}
    >>> y = keras.ops.convert_to_numpy(model(data)["y"])
    >>> np.round(y, 4)
    array([[[0.1345, 0.1345],
            [0.3655, 0.3655]]], dtype=float32)
    >>> round(float(y.sum()), 4)
    1.0

    Scope
    -----
    The shape of the stream is kept, a sequence included:

    >>> Softmax()(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives the softmax of each stream:

    >>> a, b = Input("a", dim=3, tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in Softmax()([a, b])]
    [Shape(S=(), T=1, D=(3,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``axis=-1`` normalizes the features of every sample of the window:

    >>> model = Modely("features", inputs=[x], outputs=[Output("y", Softmax(axis=-1)(x))]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[[0.5, 0.5],
            [0.5, 0.5]]], dtype=float32)

    ``axis=1`` is the window here: every feature sums to one over the 2
    samples.

    >>> model = Modely("window", inputs=[x], outputs=[Output("y", Softmax(axis=1)(x))]).build()
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)
    array([[[0.2689, 0.2689],
            [0.7311, 0.7311]]], dtype=float32)

    ``name`` names the layer:

    >>> Softmax(name="shares")(x).name
    'shares'

    Applications
    ------------
    The shares of 3 operating regimes, from the speed of a vehicle, which
    sum to one:

    >>> speed = Input("speed", tw=1)
    >>> shares = Softmax(axis=-1)(Linear(features=3)(speed))
    >>> shares.shape
    Shape(S=(), T=1, D=(3,))

    Possible Issues
    ---------------
    With a window, the default spreads the sum over the samples too. Here
    the 2 features of each sample sum to 0.5, not 1:

    >>> model = Modely("whole", inputs=[x], outputs=[Output("y", Softmax()(x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.zeros((1, 2, 2))})["y"])
    array([[[0.25, 0.25],
            [0.25, 0.25]]], dtype=float32)

    Give ``axis=-1`` for one sum per sample, as in Options.

    The layer is configured first, then called on its stream:

    >>> Softmax(x)
    Traceback (most recent call last):
        ...
    TypeError: Softmax has settings, so it is configured first and then called on its streams: Softmax(...)(x), not Softmax(x).

    .. versionadded:: 2.0
    """

    def __init__(self, axis: int | None = None, name=None):
        self._axis = None if axis is None else int(axis)
        super().__init__(name=name, axis=self._axis)

    def build_layer(self):
        axis = self._axis
        if axis is None:
            axis = list(range(1, 1 + self._preds[0].shape.rank))  # type: ignore
        return keras.layers.Softmax(axis=axis, name=self.name)  # type: ignore


class Sigmoid(Layer):
    r"""Logistic sigmoid of every value of a stream: ``1 / (1 + exp(-x))``.

    Syntax
    ------
    ``Sigmoid(x)``
        The sigmoid of ``x``; the same as ``Sigmoid()(x)``.
    ``Sigmoid(name="s")(x)``
        The same, with the layer named.
    ``Sigmoid([a, b])``
        The sigmoid of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Sigmoid`` followed by a
        number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=None, D=(5,))`` for
        one sample of 5 features. A list of streams gives a list, in its
        order.

    See Also
    --------
    Tanh : Hyperbolic tangent, with values in (-1, 1).
    Softmax : Positive values that sum to one.
    Swish : The input times its sigmoid.
    Softplus : Its integral, a smooth ReLU.

    Notes
    -----
    .. math:: y = \sigma(x) = \frac{1}{1 + e^{-x}}

    Defined for every real ``x``, with values in :math:`(0, 1)` and
    :math:`\sigma(0) = 0.5`. The gradient is :math:`\sigma(x)\,(1 -
    \sigma(x))`, which vanishes for large :math:`|x|`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Sigmoid, Tanh
    >>> x = Input("x", dim=5)
    >>> model = Modely("sigmoid", inputs=[x], outputs=[Output("y", Sigmoid(x))]).build()
    >>> data = {"x": np.array([[-2.0, -1.0, 0.0, 1.0, 3.0]])}
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)
    array([[0.1192, 0.2689, 0.5   , 0.7311, 0.9526]], dtype=float32)

    Scope
    -----
    The sigmoid applies to every value, whatever the shape:

    >>> Sigmoid(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives the sigmoid of each stream:

    >>> a, b = Input("a", tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in Sigmoid([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``name`` names the layer, configured first:

    >>> Sigmoid(name="s")(x).name
    's'

    Applications
    ------------
    A smooth switch between two friction coefficients, 0.8 on dry road and
    0.3 on wet road, from a wetness signal:

    >>> wetness = Input("wetness", tw=1)
    >>> mu = 0.8 - 0.5 * Sigmoid(10.0 * (wetness - 0.5))

    Properties & Relations
    ----------------------
    The hyperbolic tangent is a scaled sigmoid, :math:`\tanh x = 2\sigma(2x)
    - 1`:

    >>> model = Modely("relation", inputs=[x], outputs=[Output("a", Tanh(x)), Output("b", 2.0 * Sigmoid(2.0 * x) - 1.0)]).build()
    >>> result = model(data)
    >>> np.allclose(keras.ops.convert_to_numpy(result["a"]), keras.ops.convert_to_numpy(result["b"]), atol=1e-6)
    True

    Possible Issues
    ---------------
    A number is not a stream: use a :class:`Constant`, or compute the
    sigmoid with NumPy.

    .. versionadded:: 2.0
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return keras.layers.Activation("sigmoid", name=self.name)


class Tanh(Layer):
    r"""Hyperbolic tangent of every value of a stream.

    Syntax
    ------
    ``Tanh(x)``
        The hyperbolic tangent of ``x``; the same as ``Tanh()(x)``.
    ``Tanh(name="t")(x)``
        The same, with the layer named.
    ``Tanh([a, b])``
        The hyperbolic tangent of each stream of the list, as a list.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Tanh`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=None, D=(5,))`` for
        one sample of 5 features. A list of streams gives a list, in its
        order.

    See Also
    --------
    Sigmoid : Logistic sigmoid, with values in (0, 1).
    Atanh : Inverse hyperbolic tangent.
    Sinh : Hyperbolic sine.

    Notes
    -----
    .. math:: y = \tanh x = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Defined for every real ``x``, with values in :math:`(-1, 1)`. It is odd,
    :math:`\tanh(-x) = -\tanh x`. The gradient is :math:`1 - \tanh^2 x`,
    which vanishes for large :math:`|x|`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Linear, Modely, Output, Sigmoid, Tanh
    >>> x = Input("x", dim=5)
    >>> model = Modely("tanh", inputs=[x], outputs=[Output("y", Tanh(x))]).build()
    >>> data = {"x": np.array([[-2.0, -1.0, 0.0, 1.0, 3.0]])}
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)
    array([[-0.964 , -0.7616,  0.    ,  0.7616,  0.9951]], dtype=float32)

    Scope
    -----
    The hyperbolic tangent applies to every value, whatever the shape:

    >>> Tanh(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives the hyperbolic tangent of each stream:

    >>> a, b = Input("a", tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in Tanh([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``name`` names the layer, configured first:

    >>> Tanh(name="t")(x).name
    't'

    Applications
    ------------
    A motor torque that saturates at 2 N m:

    >>> command = Input("command", tw=1)
    >>> torque = 2.0 * Tanh(command)

    A hidden layer of a small network, as in physics-informed models:

    >>> t = Input("t", tw=1)
    >>> u = Linear(features=1)(Tanh(Linear(features=16)(t)))

    Properties & Relations
    ----------------------
    The hyperbolic tangent is a scaled :class:`Sigmoid`,
    :math:`\tanh x = 2\sigma(2x) - 1`:

    >>> model = Modely("relation", inputs=[x], outputs=[Output("a", Tanh(x)), Output("b", 2.0 * Sigmoid(2.0 * x) - 1.0)]).build()
    >>> result = model(data)
    >>> np.allclose(keras.ops.convert_to_numpy(result["a"]), keras.ops.convert_to_numpy(result["b"]), atol=1e-6)
    True

    Possible Issues
    ---------------
    A number is not a stream: use a :class:`Constant`, or compute the
    hyperbolic tangent with NumPy.

    .. versionadded:: 2.0
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return keras.layers.Activation("tanh", name=self.name)


class Swish(Layer):
    r"""Swish of every value of a stream: ``x * sigmoid(x)``.

    Syntax
    ------
    ``Swish(x)``
        The swish of ``x``; the same as ``Swish()(x)``.
    ``Swish(name="s")(x)``
        The same, with the layer named.
    ``Swish([a, b])``
        The swish of each stream of the list, as a list.

    Swish, also called SiLU, is a smooth function close to :class:`ReLU`.
    It dips slightly below zero for negative ``x``.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Swish`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=None, D=(5,))`` for
        one sample of 5 features. A list of streams gives a list, in its
        order.

    See Also
    --------
    Sigmoid : Logistic sigmoid.
    GELU : Gaussian error linear unit, a close relative.
    ReLU : Rectified linear unit.

    Notes
    -----
    .. math:: y = x\,\sigma(x) = \frac{x}{1 + e^{-x}}

    Defined for every real ``x``. Its minimum is about -0.278, at
    :math:`x \approx -1.278`. The gradient is :math:`\sigma(x)\,(1 + x\,(1 -
    \sigma(x)))`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Linear, Modely, Output, Sigmoid, Swish
    >>> x = Input("x", dim=5)
    >>> model = Modely("swish", inputs=[x], outputs=[Output("y", Swish(x))]).build()
    >>> data = {"x": np.array([[-2.0, -1.0, 0.0, 1.0, 3.0]])}
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)
    array([[-0.2384, -0.2689,  0.    ,  0.7311,  2.8577]], dtype=float32)

    Scope
    -----
    The swish applies to every value, whatever the shape:

    >>> Swish(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives the swish of each stream:

    >>> a, b = Input("a", tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in Swish([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``name`` names the layer, configured first:

    >>> Swish(name="s")(x).name
    's'

    Applications
    ------------
    A hidden layer of a small network:

    >>> velocity = Input("velocity", tw=1)
    >>> drag = Linear(features=1)(Swish(Linear(features=8)(velocity)))

    Properties & Relations
    ----------------------
    ``Swish(x)`` is ``x * Sigmoid(x)``:

    >>> model = Modely("relation", inputs=[x], outputs=[Output("a", Swish(x)), Output("b", x * Sigmoid(x))]).build()
    >>> result = model(data)
    >>> np.allclose(keras.ops.convert_to_numpy(result["a"]), keras.ops.convert_to_numpy(result["b"]), atol=1e-6)
    True

    Possible Issues
    ---------------
    A number is not a stream: use a :class:`Constant`, or compute the swish
    with NumPy.

    .. versionadded:: 2.0
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return keras.layers.Activation("swish", name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class GELUImpl(keras.layers.Layer):
    def __init__(self, approximate: bool = True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.approximate = bool(approximate)

    def call(self, inputs):
        return keras.activations.gelu(inputs, approximate=self.approximate)

    def get_config(self):
        config = super().get_config()
        config.update({"approximate": self.approximate})
        return config


class GELU(Layer):
    r"""Gaussian error linear unit of every value of a stream:
    ``x * Phi(x)``, with ``Phi`` the standard normal distribution function.

    Syntax
    ------
    ``GELU()(x)``
        The GELU of ``x``, computed with its tanh approximation.
    ``GELU(approximate=False)(x)``
        The exact GELU of ``x``, computed with the error function.
    ``GELU()([a, b])``
        The GELU of each stream of the list, as a list.

    The GELU is a smooth function close to :class:`ReLU`. The default is
    the tanh approximation, unlike ``keras.activations.gelu``.

    Parameters
    ----------
    approximate : bool, default True
        ``True`` computes the tanh approximation. ``False`` computes the
        exact formula with the error function. The two differ by less than
        about 0.001.
    name : str, optional
        The name of the layer. The default is ``GELU`` followed by a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=None, D=(5,))`` for
        one sample of 5 features. A list of streams gives a list, in its
        order.

    Raises
    ------
    TypeError
        If the layer is called before it is configured, ``GELU(x)``.

    See Also
    --------
    Swish : The input times its sigmoid, a close relative.
    ReLU : Rectified linear unit.
    ELU : Exponential linear unit.

    Notes
    -----
    With ``approximate=False``:

    .. math:: y = x\,\Phi(x) = \frac{x}{2}\left(1 + \operatorname{erf}\frac{x}{\sqrt{2}}\right)

    With ``approximate=True``:

    .. math:: y = \frac{x}{2}\left(1 + \tanh\left(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right)\right)

    Defined for every real ``x``. Its minimum is about -0.17, at
    :math:`x \approx -0.75`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import GELU, Input, Linear, Modely, Output
    >>> x = Input("x", dim=5)
    >>> model = Modely("gelu", inputs=[x], outputs=[Output("y", GELU()(x))]).build()
    >>> data = {"x": np.array([[-2.0, -1.0, 0.0, 1.0, 3.0]])}
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)
    array([[-0.0454, -0.1588,  0.    ,  0.8412,  2.9964]], dtype=float32)

    Scope
    -----
    The GELU applies to every value, whatever the shape:

    >>> GELU()(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives the GELU of each stream:

    >>> a, b = Input("a", tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in GELU()([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``approximate=False`` computes the exact formula:

    >>> y = GELU(approximate=False)(x)
    >>> model = Modely("exact", inputs=[x], outputs=[Output("y", y)]).build()
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)
    array([[-0.0455, -0.1587,  0.    ,  0.8413,  2.996 ]], dtype=float32)

    ``name`` names the layer:

    >>> GELU(name="gelu")(x).name
    'gelu'

    Applications
    ------------
    A hidden layer of a small network:

    >>> velocity = Input("velocity", tw=1)
    >>> drag = Linear(features=1)(GELU()(Linear(features=8)(velocity)))

    Possible Issues
    ---------------
    The layer is configured first, then called on its stream:

    >>> GELU(x)
    Traceback (most recent call last):
        ...
    TypeError: GELU has settings, so it is configured first and then called on its streams: GELU(...)(x), not GELU(x).

    .. versionadded:: 2.0
    """

    def __init__(self, approximate: bool = True, name=None):
        self._approximate = bool(approximate)
        super().__init__(name=name, approximate=self._approximate)

    def build_layer(self):
        return GELUImpl(approximate=self._approximate, name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class SoftplusImpl(keras.layers.Layer):
    def call(self, inputs):
        zero = keras.ops.cast(0.0, inputs.dtype)
        return keras.ops.log(
            1.0 + keras.ops.exp(-keras.ops.abs(inputs))
        ) + keras.ops.maximum(inputs, zero)

    def get_config(self):
        return super().get_config()


class Softplus(Layer):
    r"""Softplus of every value of a stream: ``log(1 + exp(x))``.

    Syntax
    ------
    ``Softplus(x)``
        The softplus of ``x``; the same as ``Softplus()(x)``.
    ``Softplus(name="s")(x)``
        The same, with the layer named.
    ``Softplus([a, b])``
        The softplus of each stream of the list, as a list.

    The softplus is a smooth :class:`ReLU`, always positive. It is computed
    in a form that does not overflow for large ``x``.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``Softplus`` followed by a
        number.

    Call arguments
    --------------
    x : Stream or list of Stream
        A stream of any shape, or a list of streams.

    Returns
    -------
    Stream or list of Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``, for example ``Shape(S=(), T=None, D=(3,))`` for
        one sample of 3 features. A list of streams gives a list, in its
        order.

    See Also
    --------
    ReLU : Rectified linear unit.
    Sigmoid : Its gradient.
    Exp : Exponential, another positive function.

    Notes
    -----
    .. math:: y = \log(1 + e^{x}) = \log(1 + e^{-|x|}) + \max(x, 0)

    The right-hand form is the one computed: it is exact for every real
    ``x``, with no overflow. The values are in :math:`(0, \infty)`, with
    :math:`y(0) = \log 2`. The gradient is the sigmoid
    :math:`1 / (1 + e^{-x})`.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Parameter, Softplus
    >>> x = Input("x", dim=3)
    >>> model = Modely("softplus", inputs=[x], outputs=[Output("y", Softplus(x))]).build()
    >>> data = {"x": np.array([[-100.0, 0.0, 100.0]])}
    >>> np.round(keras.ops.convert_to_numpy(model(data)["y"]), 4)
    array([[  0.    ,   0.6931, 100.    ]], dtype=float32)

    Scope
    -----
    The softplus applies to every value, whatever the shape:

    >>> Softplus(Input("v", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(3,))

    A list gives the softplus of each stream:

    >>> a, b = Input("a", tw=1), Input("b", dim=3)
    >>> [stream.shape for stream in Softplus([a, b])]
    [Shape(S=(), T=1, D=(1,)), Shape(S=(), T=None, D=(3,))]

    Options
    -------
    ``name`` names the layer, configured first:

    >>> Softplus(name="s")(x).name
    's'

    Applications
    ------------
    A damping coefficient that has to stay positive, trained from a free
    Parameter:

    >>> velocity = Input("velocity", tw=1)
    >>> damping = Softplus(Parameter("raw_damping", value=0.0))
    >>> force = -1.0 * damping * velocity

    Properties & Relations
    ----------------------
    ``Softplus(x) - Softplus(-x)`` is ``x``:

    >>> model = Modely("relation", inputs=[x], outputs=[Output("y", Softplus(x) - Softplus(-x))]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[-2.0, 0.0, 3.0]])})["y"])
    array([[-2.,  0.,  3.]], dtype=float32)

    Possible Issues
    ---------------
    A number is not a stream: use a :class:`Constant`, or compute the
    softplus with NumPy.

    .. versionadded:: 2.0
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return SoftplusImpl(name=self.name)


# An activation reads one stream: called on several, it is applied to each.
for _activation in (
    ReLU,
    LeakyReLU,
    ELU,
    PReLU,
    Softmax,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
    Softplus,
):
    _activation._one_stream = True
