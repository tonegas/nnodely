from __future__ import annotations

import keras

from nnodely.layers.parameter import Parameter, _check_weight_size, _ParameterWeights


@keras.saving.register_keras_serializable(package="nnodely")
class LinearImpl(keras.layers.Layer):
    """
    Linear projection of the dim axes, ``x @ kernel + bias``, at every
    sample of the window and every step of the sequence.

    Input convention:
        [batch, *seq, tw, *dim]

    Output convention, with ``axis=None``:
        [batch, *seq, tw, features]

    With ``axis=k`` only dim axis ``k`` is projected, and ``features``
    takes its place among the dim axes: with a matrix of its own at every
    position of the other dim axes, or with ``shared_kernel`` one for all.

    It owns no weight: it is called on ``[x, kernel]`` or ``[x, kernel,
    bias]``, the values of the Parameters holding them, and reads their first
    sample.
    """

    def __init__(
        self,
        features: int = 1,
        dim_rank: int = 1,
        axis=None,
        shared_kernel: bool = False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.features = int(features)
        self.dim_rank = int(dim_rank)
        self.axis = None if axis is None else int(axis)
        self.shared_kernel = bool(shared_kernel)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "dim_rank": self.dim_rank,
                "axis": self.axis,
                "shared_kernel": self.shared_kernel,
            }
        )
        return config

    def build(self, input_shape):
        # The dim axes are the last ones.
        dim = tuple(int(axis) for axis in input_shape[0][-self.dim_rank :])
        self.kernel_shape, self.bias_shape = _linear_weight_shapes(
            dim, self.features, self.axis, self.shared_kernel
        )
        for label, shape, expected in zip(
            ("kernel", "bias"), input_shape[1:], (self.kernel_shape, self.bias_shape)
        ):
            _check_weight_size(self.name, label, shape, expected)
        super().build(input_shape)

    def call(self, inputs):
        x, kernel, *bias = inputs
        kernel = keras.ops.reshape(kernel[0], self.kernel_shape)

        if self.axis is None:
            # The dim, the last axes of (batch, *seq, tw, *dim), contracted
            # with the first axes of the kernel: (batch, *seq, tw, out).
            y = keras.ops.tensordot(x, kernel, axes=self.dim_rank)
            if bias:
                y = y + keras.ops.reshape(bias[0][0], self.bias_shape)
            return y

        # The projected axis moved last, (batch, *tw, *kept, 1, projected):
        # every position of the kept axes times its matrix, or the shared one.
        # The axes are addressed from the front, which ONNX needs.
        last = len(x.shape) - 1
        projected = last + 1 - self.dim_rank + self.axis
        if projected != last:
            x = keras.ops.moveaxis(x, projected, last)
        y = keras.ops.matmul(keras.ops.expand_dims(x, -2), kernel)
        y = keras.ops.squeeze(y, -2)
        if bias:
            y = y + keras.ops.reshape(bias[0][0], self.bias_shape)
        # out back in place of the projected axis.
        return y if projected == last else keras.ops.moveaxis(y, last, projected)


def _linear_weight_shapes(dim, features, axis, shared_kernel):
    """The kernel and bias shapes of a Linear on ``dim``."""
    if axis is None:
        return (*dim, features), (features,)
    kept = () if shared_kernel else (*dim[:axis], *dim[axis + 1 :])
    return (*kept, dim[axis], features), (*kept, features)


class Linear(_ParameterWeights):
    r"""Learned affine map of the features of every sample of a stream:
    ``x @ kernel + bias``.

    Syntax
    ------
    ``Linear(features=n)(x)``
        Maps all the features of every sample of ``x`` to ``n`` outputs.
    ``Linear(features=n, axis=k)(x)``
        Maps only the feature axis ``k``, with a matrix for every position
        of the other feature axes.
    ``Linear(features=n, axis=k, shared_kernel=True)(x)``
        The same, with one matrix shared by every position.
    ``Linear(kernel=W, bias=False)(x)``
        Maps with the coefficients the :class:`Parameter` ``W`` holds.

    The map acts on the features, ``D``, of each sample alone. Every sample
    of the window and every step of the sequence is mapped with the same
    weights, and samples are never mixed: use :class:`Fir` to combine the
    samples of a window.

    Parameters
    ----------
    features : int, default 1
        The number of outputs. They take the place of all the features, or
        of the feature axis ``axis``.
    kernel : str, keras.initializers.Initializer or Parameter, default "glorot_uniform"
        The coefficients. An initializer, by name or object, draws them. A
        :class:`Parameter` holds them, of the shape listed in Notes. Axes of
        size 1 aside, any other shape raises.
    bias : bool, str, keras.initializers.Initializer or Parameter, default True
        The offset added to every output. ``True`` draws it with
        ``"glorot_uniform"``, not zeros. ``False`` leaves it out. An
        initializer, or a Parameter of the shape listed in Notes, sets it.
    axis : int or None, default None
        The feature axis mapped, counted among the feature axes only: ``0``
        is the first, ``-1`` the last. ``None`` maps all the features at
        once.
    shared_kernel : bool, default False
        ``True`` maps every position of the other feature axes with the same
        kernel and bias. It needs an ``axis``.
    name : str, optional
        The name of the layer. Its weights are named ``<name>_kernel`` and
        ``<name>_bias``. The default is ``Linear`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape, ``Shape(S, T, D)``, with or without a window.
        The call takes it alone, ``Linear()(x)``, or in a list,
        ``Linear()([x])``.

    Returns
    -------
    Stream
        The sequence and the window of ``x`` are kept, a missing window
        staying missing. With ``axis=None``, ``Shape(S, T, D)`` gives
        ``Shape(S, T, (features,))``. With ``axis=k``, the feature axis
        ``k`` is replaced by ``features``, even when ``features=1``. For
        example, ``Shape(S=(), T=2, D=(2, 3))`` with ``features=4`` gives
        ``D=(4,)``, ``D=(2, 4)`` with ``axis=1`` and ``D=(4, 3)`` with
        ``axis=0``.

    Raises
    ------
    ValueError
        If ``shared_kernel=True`` is given without an ``axis`` ("needs an
        axis"). If ``axis`` is not a feature axis of ``x`` ("is not an axis
        of the dim"). If a kernel or bias Parameter does not have the shape
        of the weight ("kernel must have the shape ..."). If the layer is
        called on several streams ("reads one stream, got 2").
    TypeError
        If the layer is called before it is configured, ``Linear(x)``.

    See Also
    --------
    Fir : Learned weighted sum over the window.
    Concatenate : Joins streams along the features, to map them together.
    EquationLearner : Learned combination of elementary functions.
    Parameter : A trained value of the model.

    Notes
    -----
    With ``axis=None``, every sample :math:`x`, its features flattened to
    a vector, is mapped as

    .. math:: y_j = \sum_i x_i \, W_{ij} + b_j

    With ``axis=k``, the same map runs along axis ``k`` at every position
    of the other feature axes, with a kernel and a bias of its own unless
    ``shared_kernel=True``.

    The weights have these shapes, with :math:`D_k` the size of axis ``k``
    and ``D_other`` the other feature axes:

    - ``axis=None``: kernel ``(*D, features)``, bias ``(features,)``.
    - ``axis=k``: kernel ``(*D_other, D_k, features)``, bias
      ``(*D_other, features)``.
    - ``axis=k, shared_kernel=True``: kernel ``(D_k, features)``, bias
      ``(features,)``.

    On ``D=(2, 3)`` with ``features=4``, ``axis=1`` gives a kernel
    ``(2, 3, 4)`` and a bias ``(2, 4)``, and ``axis=0`` a kernel
    ``(3, 2, 4)`` and a bias ``(3, 4)``.

    The kernel and the bias are :class:`Parameter` objects: they can be
    passed, shared and frozen. They are listed by :attr:`Modely.parameters`,
    and are trained, frozen and saved with the model. With an ``axis``, the
    initializer draws the matrix of each position as its own
    :math:`D_k \times` ``features`` matrix. After :meth:`Modely.build`, the
    stream the layer returns exposes their Keras variables as ``kernel``
    and ``bias``.

    One Linear object called on several streams applies the same weights to
    each.

    Basic Examples
    --------------
    Three outputs from the 2 features of every sample of a window, with
    known coefficients:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Fir, Input, Linear, Modely, Output, Parameter, Tanh
    >>> x = Input("x", dim=2, tw=2)
    >>> W = Parameter("W", value=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    >>> y = Linear(features=3, kernel=W, bias=False)(x)
    >>> model = Modely("linear_demo", inputs=[x], outputs=[Output("y", y)]).build()
    >>> result = model({"x": np.array([[[1.0, 2.0], [3.0, 4.0]]])})
    >>> keras.ops.convert_to_numpy(result["y"])  # x1, x2, x1 + x2 at each sample
    array([[[1., 2., 3.],
            [3., 4., 7.]]], dtype=float32)

    Scope
    -----
    A stream without a window, and a sequence, keep their axes:

    >>> Linear(features=2)(Input("v", dim=3)).shape
    Shape(S=(), T=None, D=(2,))
    >>> Linear(features=2)(Input("s", dim=3, seq=5, tw=2)).shape
    Shape(S=(5,), T=2, D=(2,))

    A scalar stream gets ``features`` outputs:

    >>> Linear(features=4)(Input("u", tw=3)).shape
    Shape(S=(), T=3, D=(4,))

    Several feature axes are mapped together:

    >>> e = Input("e", dim=(2, 3), tw=2)
    >>> Linear(features=4)(e).shape
    Shape(S=(), T=2, D=(4,))

    Options
    -------
    ``features`` sets the number of outputs:

    >>> Linear(features=5)(x).shape
    Shape(S=(), T=2, D=(5,))

    ``kernel`` and ``bias`` take an initializer by name, or a Parameter.
    Here the kernel is all ones and the bias a Parameter:

    >>> v = Input("v", dim=2)
    >>> z = Linear(kernel="ones", bias=Parameter("b", value=[10.0]))(v)
    >>> model = Modely("offset", inputs=[v], outputs=[Output("z", z)]).build()
    >>> keras.ops.convert_to_numpy(model({"v": np.array([[1.0, 2.0]])})["z"])  # 1 + 2 + 10
    array([[13.]], dtype=float32)

    ``bias=False`` leaves the bias out:

    >>> y = Linear(features=2, bias=False, name="nobias")(v)
    >>> [p.name for p in Modely("nb", inputs=[v], outputs=[Output("y", y)]).parameters]
    ['nobias_kernel']

    ``axis`` maps one feature axis. On ``D=(2, 3)``, ``axis=1`` maps each
    row of 3 features with a matrix of its own, ``axis=0`` each column:

    >>> Linear(features=4, axis=1)(e).shape
    Shape(S=(), T=2, D=(2, 4))
    >>> Linear(features=4, axis=0)(e).shape
    Shape(S=(), T=2, D=(4, 3))

    ``shared_kernel=True`` maps every row with one matrix, instead of one
    each:

    >>> own = Linear(features=4, axis=1, name="own")(e)
    >>> shared = Linear(features=4, axis=1, shared_kernel=True, name="shared")(e)
    >>> model = Modely("kernels", inputs=[e], outputs=[Output("a", own), Output("b", shared)])
    >>> [(p.name, p.shape.dim) for p in model.parameters]
    [('own_kernel', (2, 3, 4)), ('own_bias', (2, 4)), ('shared_kernel', (3, 4)), ('shared_bias', (4,))]

    With known coefficients, a shared kernel of ones sums each row of a
    ``2 x 2`` sample:

    >>> m = Input("m", dim=(2, 2))
    >>> ones = Parameter("ones", value=[[1.0], [1.0]])
    >>> rows = Linear(axis=1, shared_kernel=True, kernel=ones, bias=False)(m)
    >>> model = Modely("rows", inputs=[m], outputs=[Output("rows", rows)]).build()
    >>> keras.ops.convert_to_numpy(model({"m": np.array([[[1.0, 2.0], [3.0, 4.0]]])})["rows"])
    array([[[3.],
            [7.]]], dtype=float32)

    ``name`` names the layer and its weights:

    >>> y = Linear(name="gain")(x)
    >>> [p.name for p in Modely("named", inputs=[x], outputs=[Output("y", y)]).parameters]
    ['gain_kernel', 'gain_bias']

    Applications
    ------------
    The acceleration of a mass-spring-damper is a linear map of its
    position, velocity and force. Their 3 features give one output:

    >>> state = Input("state", dim=3, tw=1)  # position, velocity, force
    >>> acceleration = Linear(features=1, bias=False)(state)
    >>> acceleration.shape
    Shape(S=(), T=1, D=(1,))

    A small network of the time, as in physics-informed models:

    >>> t = Input("t", tw=1)
    >>> u = Linear(features=1)(Tanh(Linear(features=16)(t)))

    Properties & Relations
    ----------------------
    On a window of one sample, a scalar map is the :class:`Fir` of that
    sample:

    >>> w = Input("w", tw=1)
    >>> linear = Linear(kernel=Parameter("k_lin", value=[[2.0]]), bias=False)(w)
    >>> fir = Fir(kernel=Parameter("k_fir", value=[[2.0]]), bias=False)(w)
    >>> model = Modely("same", inputs=[w], outputs=[Output("lin", linear), Output("fir", fir)])
    >>> result = model.build()({"w": np.array([[[3.0]]])})
    >>> keras.ops.convert_to_numpy(result["lin"]), keras.ops.convert_to_numpy(result["fir"])
    (array([[[6.]]], dtype=float32), array([[[6.]]], dtype=float32))

    Possible Issues
    ---------------
    Two streams are not mapped together. Join them first with
    :class:`Concatenate`, or apply the layer to each:

    >>> a, b = Input("a", tw=1), Input("b", tw=1)
    >>> Linear()([a, b])
    Traceback (most recent call last):
        ...
    ValueError: ...Linear reads one stream, got 2...

    ``shared_kernel`` needs an ``axis``:

    >>> Linear(shared_kernel=True)
    Traceback (most recent call last):
        ...
    ValueError: Linear shared_kernel shares the kernel along the dim axes it does not project, so it needs an axis...

    A kernel Parameter must have the shape of the weight, here ``(2, 3)``
    for 2 features and 3 outputs:

    >>> Linear(features=3, kernel=Parameter("K", value=[[1.0, 2.0]]))(x)
    Traceback (most recent call last):
        ...
    ValueError: ...kernel must have the shape (2, 3), axes of size 1 aside, got (1, 2).

    The layer is configured first, then called on its stream:

    >>> Linear(x)
    Traceback (most recent call last):
        ...
    TypeError: Linear has settings, so it is configured first and then called on its streams: Linear(...)(x), not Linear(x).

    .. versionadded:: 2.0
    """

    _defaults = {"kernel": "glorot_uniform", "bias": "glorot_uniform"}

    def __init__(
        self,
        features: int = 1,
        kernel: Parameter | str | keras.initializers.Initializer = "glorot_uniform",
        bias: Parameter | str | keras.initializers.Initializer | bool = True,
        axis: int | None = None,
        shared_kernel: bool = False,
        name=None,
    ):
        if shared_kernel and axis is None:
            raise ValueError(
                "Linear shared_kernel shares the kernel along the dim axes it "
                "does not project, so it needs an axis: without one, the whole "
                "dim is projected at once."
            )
        self._features = int(features)
        self._axis = None if axis is None else int(axis)
        self._shared_kernel = bool(shared_kernel)
        super().__init__(
            name,
            {"kernel": kernel, "bias": bias},
            kernel=kernel,
            bias=bias,
            features=self._features,
            axis=self._axis,
            shared_kernel=self._shared_kernel,
        )

    def _resolve_axis(self, dim) -> int | None:
        """``axis`` counted from the first dim axis, checked against ``dim``."""
        if self._axis is None:
            return None
        if not -len(dim) <= self._axis < len(dim):
            raise ValueError(
                f"{self.name}: axis {self._axis} is not an axis of the dim {dim}."
            )
        return self._axis % len(dim)

    def _weight_shapes(self, x):
        dim = tuple(x.dim)
        return _linear_weight_shapes(
            dim, self._features, self._resolve_axis(dim), self._shared_kernel
        )

    def _draw_shapes(self, shapes):
        # Along an axis, the matrix and the bias of each position alone; the
        # whole-dim kernel as the Weight's default, one matrix.
        if self._axis is None:
            return None, None
        kernel_shape, bias_shape = shapes
        return kernel_shape[-2:], bias_shape[-1:]

    def output_shape(self, *inputs):
        # Declared rather than probed with dummy tensors, which would have to
        # stand in for the weights too.
        dim = tuple(inputs[0].dim)
        axis = self._resolve_axis(dim)
        out_dim = (
            (self._features,)
            if axis is None
            else (*dim[:axis], self._features, *dim[axis + 1 :])
        )
        return inputs[0].shape.replace(dim=out_dim)

    def build_layer(self):
        dim = tuple(self._preds[0].dim)  # type: ignore[attr-defined]
        return LinearImpl(
            features=self._features,
            dim_rank=len(dim),
            axis=self._resolve_axis(dim),
            shared_kernel=self._shared_kernel,
            name=self.name,
        )
