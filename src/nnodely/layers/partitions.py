"""Partitions of a stream: the selections and slices of its dim axes, its
window and its sequence axes, and the concatenations that join streams along
them.

The dim operations count the dim axes from 0. The time operations act on
the window, ``T``, of which a stream has one or none: they take no axis,
always keep it, and refuse a stream without one. The sequence operations act on a sequence axis, counted outermost
first: ``axis=0``, their default, is the outermost one - the rollout of the
outermost Loop; a sequence axis may be dynamic (``None``), its length known
only at run time. Each layer resolves the axis it acts on with the
:class:`Shape` of its input, and gives its Keras layer that tensor axis.
"""

import keras

from nnodely.core.layer import Layer
from nnodely.core.shape import _is_scalar, expand_to_rank, resolve_axis, slice_axis
from nnodely.core.stream import Stream


@keras.saving.register_keras_serializable(package="nnodely")
class SelectImpl(keras.layers.Layer):
    """Serializable selection of index ``idx`` along the tensor axis ``axis``,
    which is dropped, or kept with length 1 when ``keep_dim``."""

    def __init__(
        self,
        idx: int,
        axis: int,
        keep_dim: bool = False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.idx = int(idx)
        self.axis = int(axis)
        self.keep_dim = bool(keep_dim)

    def call(self, x):
        # An integer index drops the axis. A negative one counts from the end
        # of the data, so -1 is the last step of a sequence of any length.
        if not self.keep_dim:
            return slice_axis(x, self.axis, self.idx)
        end = None if self.idx == -1 else self.idx + 1
        return slice_axis(x, self.axis, slice(self.idx, end))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.keep_dim:
            output_shape[self.axis] = 1
        else:
            del output_shape[self.axis]
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "idx": self.idx,
                "axis": self.axis,
                "keep_dim": self.keep_dim,
            }
        )
        return config


class Select(Layer):
    r"""One index of a dim axis of a stream, the axis dropped.

    Syntax
    ------
    ``Select(idx)(x)``
        The feature ``idx`` of ``x``, along its first dim axis.
    ``Select(idx, axis=k)(x)``
        Index ``idx`` along dim axis ``k``.
    ``Select(idx, keep_dim=True)(x)``
        The same, the axis kept with length 1.

    The selection is made at every sample of the window and every step of the
    sequence: the window and the sequence are kept. The selected dim axis is
    dropped. A stream always has a dim, so selecting from a vector gives a
    scalar, ``D=(1,)``, with ``keep_dim`` or without.

    Parameters
    ----------
    idx : int
        The index along the axis, counted from 0. A negative one counts from
        the end: ``-1`` is the last. It must lie in ``[-n, n)`` for an axis of
        length ``n``.
    axis : int, default 0
        The dim axis, counted from 0. A negative one counts from the last.
    keep_dim : bool, default False
        ``True`` keeps the selected axis with length 1, so the result keeps
        the number of dim axes of ``x``.
    name : str, optional
        The name of the layer. The default is ``Select`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape, with dim axis ``axis``. The call takes it alone
        or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with dim axis ``axis`` dropped, or of length 1 with
        ``keep_dim=True``. For example, ``D=(4, 3)`` gives ``D=(3,)``, and
        ``D=(1, 3)`` with ``keep_dim=True``. ``D=(4,)`` gives ``D=(1,)``.

    Raises
    ------
    ValueError
        If ``idx`` is outside the axis ("idx 4 out of bounds for dim axis 0, of
        size 4"), or if ``axis`` is not a dim axis of ``x`` ("axis 1 out of
        bounds for dim rank 1").

    See Also
    --------
    Range : A part of a dim axis.
    Concatenate : Streams joined along a dim axis.
    TimeSelect : One sample of the window.
    SeqSelect : One step of a sequence axis.

    Notes
    -----
    For a vector :math:`x_0, \dots, x_{n-1}`:

    .. math:: y = x_{\text{idx}}, \qquad x_{-k} = x_{n-k}

    The gradient is 1 for the selected feature and 0 for the others.

    Basic Examples
    --------------
    The second and the last feature of a vector:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Range, Select
    >>> v = Input("v", dim=4)
    >>> outputs = [Output("second", Select(1)(v)), Output("last", Select(-1)(v))]
    >>> model = Modely("select", inputs=[v], outputs=outputs).build()
    >>> result = model({"v": np.array([[1.0, 2.0, 3.0, 4.0]])})
    >>> keras.ops.convert_to_numpy(result["second"]), keras.ops.convert_to_numpy(result["last"])
    (array([[2.]], dtype=float32), array([[4.]], dtype=float32))

    Scope
    -----
    The window and the sequence are kept:

    >>> Select(0)(Input("w", dim=3, seq=2, tw=5)).shape
    Shape(S=(2,), T=5, D=(1,))

    From a matrix, ``Select`` takes a row, along dim axis 0:

    >>> m = Input("m", dim=(2, 3))
    >>> Select(1)(m).shape
    Shape(S=(), T=None, D=(3,))

    Options
    -------
    ``axis`` picks the dim axis. On the matrix ``[[1, 2, 3], [4, 5, 6]]``,
    index 2 of axis 1 is the last column:

    >>> model = Modely("column", inputs=[m], outputs=[Output("y", Select(2, axis=1)(m))]).build()
    >>> keras.ops.convert_to_numpy(model({"m": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])})["y"])
    array([[3., 6.]], dtype=float32)

    ``keep_dim=True`` keeps the axis, with length 1:

    >>> Select(2, axis=1, keep_dim=True)(m).shape
    Shape(S=(), T=None, D=(2, 1))

    ``name`` names the layer:

    >>> Select(0, name="x_position")(v).name
    'x_position'

    Applications
    ------------
    The altitude of a vehicle, the third feature of its position:

    >>> position = Input("position", dim=3, tw=1)
    >>> altitude = Select(2)(position)

    Properties & Relations
    ----------------------
    ``Select(i, keep_dim=True)`` has the shape of ``Range(i, i + 1)``:

    >>> Select(1, keep_dim=True)(m).shape, Range(1, 2)(m).shape
    (Shape(S=(), T=None, D=(1, 3)), Shape(S=(), T=None, D=(1, 3)))

    Possible Issues
    ---------------
    The index must lie within the axis:

    >>> Select(4)(v)
    Traceback (most recent call last):
        ...
    ValueError: ...idx 4 out of bounds for dim axis 0, of size 4.

    The layer is configured with the index first, then called on its stream:
    ``Select(1)(v)``, not ``Select(v)``.

    .. versionadded:: 2.0
    """

    def __init__(self, idx: int, axis: int = 0, keep_dim: bool = False, name=None):
        self._idx = int(idx)
        self._axis = int(axis)
        self._keep_dim = bool(keep_dim)
        super().__init__(
            name=name, idx=self._idx, axis=self._axis, keep_dim=self._keep_dim
        )

    def _selected(self, shape):
        """The dim axis, the index on it, and whether the axis is kept."""
        axis = shape.resolve_dim(self._axis, self.name)
        idx = _index(self.name, self._idx, shape.dim[axis], f"dim axis {axis}")
        return axis, idx, self._keep_dim or shape.dim_rank == 1

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        axis, _, keep = self._selected(shape)
        kept = (1,) if keep else ()
        return shape.replace(dim=shape.dim[:axis] + kept + shape.dim[axis + 1 :])

    def build_layer(self):
        # The input's shape, not this node's: once applied, a Select carries
        # its own output shape, without the selected axis.
        shape = self._preds[0].shape  # type: ignore[union-attr]
        axis, idx, keep = self._selected(shape)
        return SelectImpl(
            idx=idx,
            axis=shape.dim_axes[axis],
            keep_dim=keep,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class RangeImpl(keras.layers.Layer):
    """Serializable range ``[start, end)`` along the tensor axis ``axis``;
    ``None`` leaves a bound open."""

    def __init__(self, start, end, axis: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.start = None if start is None else int(start)
        self.end = None if end is None else int(end)
        self.axis = int(axis)

    def call(self, x):
        return slice_axis(x, self.axis, slice(self.start, self.end))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = _range_length(
            self.name, self.start, self.end, input_shape[self.axis]
        )
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "start": self.start,
                "end": self.end,
                "axis": self.axis,
            }
        )
        return config


class Range(Layer):
    r"""A contiguous part ``[start, end)`` of a dim axis of a stream.

    Syntax
    ------
    ``Range(start, end)(x)``
        The features ``start`` to ``end - 1`` of ``x``, along its first dim
        axis.
    ``Range(start, end, axis=k)(x)``
        The same part of dim axis ``k``.

    The end is excluded. The axis is kept, with length ``end - start``, also
    when that is 1. The part is taken at every sample of the window and every
    step of the sequence: the window and the sequence are kept.

    Parameters
    ----------
    start : int
        The first index, counted from 0. A negative one counts from the end.
    end : int
        The index after the last one. A negative one counts from the end. To
        reach the last feature, give the length of the axis: there is no open
        bound, unlike :class:`TimeRange`.
    axis : int, default 0
        The dim axis, counted from 0. A negative one counts from the last.
    name : str, optional
        The name of the layer. The default is ``Range`` followed by a number.

    Call arguments
    --------------
    x : Stream
        A stream of any shape, with dim axis ``axis``. The call takes it alone
        or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with dim axis ``axis`` of length ``end - start``.
        For example, ``D=(9, 3)`` gives ``D=(4, 3)`` with ``Range(0, 4)``, and
        ``D=(9, 2)`` with ``Range(1, 3, axis=1)``.

    Raises
    ------
    ValueError
        If the range is empty, reversed or beyond the axis ("invalid range [2,
        0) for dim size 4"), or if ``axis`` is not a dim axis of ``x``.
    TypeError
        If a bound is ``None`` ("int() argument must be ...").

    See Also
    --------
    Select : One index of a dim axis.
    Concatenate : Streams joined along a dim axis.
    TimeRange : A part of the window.
    SeqRange : A part of a sequence axis.

    Notes
    -----
    For a vector :math:`x_0, \dots, x_{n-1}`:

    .. math:: y = (x_{\text{start}}, \dots, x_{\text{end} - 1})

    Basic Examples
    --------------
    The second and third features of a vector:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, Range, Select
    >>> v = Input("v", dim=4)
    >>> model = Modely("range", inputs=[v], outputs=[Output("y", Range(1, 3)(v))]).build()
    >>> keras.ops.convert_to_numpy(model({"v": np.array([[1.0, 2.0, 3.0, 4.0]])})["y"])
    array([[2., 3.]], dtype=float32)

    Scope
    -----
    The window and the sequence are kept:

    >>> Range(0, 2)(Input("w", dim=3, seq=2, tw=5)).shape
    Shape(S=(2,), T=5, D=(2,))

    A range one feature long keeps the axis:

    >>> Range(2, 3)(v).shape
    Shape(S=(), T=None, D=(1,))

    Options
    -------
    Negative bounds count from the end: ``Range(-3, -1)`` of 4 features is
    ``Range(1, 3)``.

    >>> model = Modely("negative", inputs=[v], outputs=[Output("y", Range(-3, -1)(v))]).build()
    >>> keras.ops.convert_to_numpy(model({"v": np.array([[1.0, 2.0, 3.0, 4.0]])})["y"])
    array([[2., 3.]], dtype=float32)

    ``axis`` picks the dim axis. On the matrix ``[[1, 2, 3], [4, 5, 6]]``,
    the first two columns:

    >>> m = Input("m", dim=(2, 3))
    >>> model = Modely("columns", inputs=[m], outputs=[Output("y", Range(0, 2, axis=1)(m))]).build()
    >>> keras.ops.convert_to_numpy(model({"m": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])})["y"])
    array([[[1., 2.],
            [4., 5.]]], dtype=float32)

    ``name`` names the layer:

    >>> Range(0, 2, name="planar")(v).name
    'planar'

    Applications
    ------------
    The horizontal part of a position vector, its first two features:

    >>> position = Input("position", dim=3, tw=1)
    >>> planar = Range(0, 2)(position)

    Properties & Relations
    ----------------------
    ``Range(i, i + 1)`` has the shape of ``Select(i, keep_dim=True)``:

    >>> Range(1, 2)(m).shape, Select(1, keep_dim=True)(m).shape
    (Shape(S=(), T=None, D=(1, 3)), Shape(S=(), T=None, D=(1, 3)))

    Possible Issues
    ---------------
    An end of 0 is the start of the axis, not its end. For the last features,
    give the length of the axis as the end:

    >>> Range(-2, 0)(v)
    Traceback (most recent call last):
        ...
    ValueError: ...invalid range [-2, 0) for dim size 4.
    >>> Range(-2, 4)(v).shape
    Shape(S=(), T=None, D=(2,))

    .. versionadded:: 2.0
    """

    def __init__(self, start: int, end: int, axis: int = 0, name=None):
        self._start = int(start)
        self._end = int(end)
        self._axis = int(axis)
        super().__init__(name=name, start=self._start, end=self._end, axis=self._axis)

    def build_layer(self):
        if self._preds is None or len(self._preds) == 0:
            raise ValueError(f"{self.name}: no input layer to select range from.")
        shape = self._preds[0].shape  # type: ignore[union-attr]
        axis = shape.resolve_dim(self._axis, self.name)

        dim_size = shape.dim[axis]
        start = self._start
        end = self._end
        if start < 0:
            start += dim_size
        if end < 0:
            end += dim_size

        if start < 0 or end > dim_size or start >= end:
            raise ValueError(
                f"{self.name}: invalid range [{self._start}, {self._end}) for "
                f"dim size {dim_size}."
            )

        return RangeImpl(
            start=start,
            end=end,
            axis=shape.dim_axes[axis],
            name=self.name,
        )


class TimeSelect(Layer):
    r"""One sample of the window of a stream.

    Syntax
    ------
    ``TimeSelect(idx)(x)``
        Sample ``idx`` of the window of ``x``, as a window one sample long,
        ``T=1``.
    ``TimeSelect(idx, keep_tw=False)(x)``
        The same sample, without a window.

    The window is ordered oldest first: ``0`` is the oldest sample and ``-1``
    the newest. The selection is made at every step of the sequence, and the
    features are kept.

    By default the window is kept, one sample long. A target, a sample fed
    back to a state, and a layer acting on the window need it so. A stream of
    ``T=1`` combines only with streams of one sample or without a window. With
    ``keep_tw=False`` the window is dropped, so the sample applies to every
    sample of a window it is combined with: ``w - TimeSelect(-1,
    keep_tw=False)(w)`` subtracts the newest sample from each.

    Parameters
    ----------
    idx : int
        The sample, counted from the oldest, 0. A negative one counts from the
        newest, ``-1``. It must lie in ``[-T, T)``.
    keep_tw : bool, default True
        ``True`` keeps the window, one sample long. ``False`` drops it.
    name : str, optional
        The name of the layer. The default is ``TimeSelect`` followed by a
        number.

    Call arguments
    --------------
    x : Stream
        A stream with a window, ``Shape(S, T, D)``. The call takes it alone or
        in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, 1, D)``, or ``Shape(S, None, D)`` with ``keep_tw=False``.
        For example, ``Shape(S=(3,), T=5, D=(2,))`` gives
        ``Shape(S=(3,), T=1, D=(2,))``.

    Raises
    ------
    ValueError
        If ``x`` has no window ("TimeSelect acts on the window of its stream,
        and 'x' has none"), or if ``idx`` is outside it ("idx 4 out of bounds
        for the window, of size 4").

    See Also
    --------
    TimeRange : A part of the window.
    TimeConcatenate : Streams joined along the window.
    Select : One index of a dim axis.
    SeqSelect : One step of a sequence axis.

    Notes
    -----
    For the window :math:`x_0, \dots, x_{T-1}`, from the oldest sample to the
    newest:

    .. math:: y = x_{\text{idx}}, \qquad x_{-k} = x_{T-k}

    Basic Examples
    --------------
    The oldest and the newest sample of the window ``[1, 2, 3, 4]``:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, TimeRange, TimeSelect
    >>> x = Input("x", tw=4)
    >>> outputs = [Output("oldest", TimeSelect(0)(x)), Output("newest", TimeSelect(-1)(x))]
    >>> model = Modely("samples", inputs=[x], outputs=outputs).build()
    >>> data = {"x": np.array([[[1.0], [2.0], [3.0], [4.0]]])}
    >>> result = model(data)
    >>> keras.ops.convert_to_numpy(result["oldest"]), keras.ops.convert_to_numpy(result["newest"])
    (array([[[1.]]], dtype=float32), array([[[4.]]], dtype=float32))

    Scope
    -----
    The features and the sequence are kept:

    >>> TimeSelect(-1)(Input("v", dim=2, seq=3, tw=5)).shape
    Shape(S=(3,), T=1, D=(2,))

    Options
    -------
    ``keep_tw=False`` drops the window. The sample then applies to every
    sample of the window:

    >>> newest = TimeSelect(-1, keep_tw=False)(x)
    >>> newest.shape
    Shape(S=(), T=None, D=(1,))
    >>> model = Modely("relative", inputs=[x], outputs=[Output("y", x - newest)]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[[-3.],
            [-2.],
            [-1.],
            [ 0.]]], dtype=float32)

    ``name`` names the layer:

    >>> TimeSelect(-1, name="now")(x).name
    'now'

    Applications
    ------------
    A one-step-ahead model reads the past of a signal and is trained against
    its next sample. Declare the union window, 4 past samples and 1 future
    one, and take the parts:

    >>> from nnodely import Fir
    >>> y = Input("y", tw=(4, 1))
    >>> prediction = Fir()(TimeRange(0, 4)(y))
    >>> target = TimeSelect(-1)(y)
    >>> prediction.shape, target.shape
    (Shape(S=(), T=1, D=(1,)), Shape(S=(), T=1, D=(1,)))

    Properties & Relations
    ----------------------
    ``TimeSelect(i)`` has the shape of ``TimeRange(i, i + 1)``:

    >>> TimeSelect(1)(x).shape, TimeRange(1, 2)(x).shape
    (Shape(S=(), T=1, D=(1,)), Shape(S=(), T=1, D=(1,)))

    Possible Issues
    ---------------
    A sample with its window, ``T=1``, does not combine with a longer window.
    Drop the window with ``keep_tw=False``:

    >>> x - TimeSelect(-1)(x)
    Traceback (most recent call last):
        ...
    ValueError: ...streams with windows of 1 and 4 cannot be combined...
    >>> (x - TimeSelect(-1, keep_tw=False)(x)).shape
    Shape(S=(), T=4, D=(1,))

    A stream without a window has no sample to select. Declare the Input with
    a window:

    >>> TimeSelect(0)(Input("plain"))
    Traceback (most recent call last):
        ...
    ValueError: ...TimeSelect acts on the window of its stream, and 'plain' has none...

    .. versionadded:: 2.0
    """

    def __init__(self, idx: int, keep_tw: bool = True, name=None):
        self._idx = int(idx)
        self._keep_tw = bool(keep_tw)
        super().__init__(name=name, idx=self._idx, keep_tw=self._keep_tw)

    def output_shape(self, *inputs):
        self._needs_window(inputs[0])
        shape = inputs[0].shape
        _index(self.name, self._idx, shape.tw, "the window")
        return shape.replace(tw=1 if self._keep_tw else None)

    def build_layer(self):
        shape = self._preds[0].shape  # type: ignore[union-attr]
        return SelectImpl(
            idx=_index(self.name, self._idx, shape.tw, "the window"),
            axis=shape.tw_axis,
            keep_dim=self._keep_tw,
            name=self.name,
        )


class TimeRange(Layer):
    r"""A contiguous part ``[start, end)`` of the window of a stream.

    Syntax
    ------
    ``TimeRange(start, end)(x)``
        The samples ``start`` to ``end - 1`` of the window of ``x``.
    ``TimeRange(-n, None)(x)``
        The newest ``n`` samples.
    ``TimeRange(None, n)(x)``
        The oldest ``n`` samples.

    The window is ordered oldest first, and the end is excluded. The window is
    kept, with ``end - start`` samples, also when that is 1. ``None`` leaves a
    bound open, and a negative bound counts from the newest sample. The part
    is taken at every step of the sequence, and the features are kept.

    Parameters
    ----------
    start : int or None
        The first sample, counted from the oldest, 0. A negative one counts
        from the end. ``None`` starts from the oldest.
    end : int or None
        The sample after the last one. A negative one counts from the end.
        ``None`` reaches the newest.
    name : str, optional
        The name of the layer. The default is ``TimeRange`` followed by a
        number.

    Call arguments
    --------------
    x : Stream
        A stream with a window, ``Shape(S, T, D)``. The call takes it alone or
        in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, end - start, D)``. For example, ``Shape(S=(), T=10,
        D=(1,))`` gives ``T=3`` with ``TimeRange(2, 5)``.

    Raises
    ------
    ValueError
        If ``x`` has no window ("TimeRange acts on the window of its stream"),
        or if the range is empty, reversed or beyond the window ("invalid range
        [2, 2) for an axis of size 4").
    TypeError
        If ``end`` is missing ("missing 1 required positional argument:
        'end'"): give ``None`` for an open end.

    See Also
    --------
    TimeSelect : One sample of the window.
    TimeConcatenate : Streams joined along the window.
    Range : A part of a dim axis.
    SeqRange : A part of a sequence axis.
    Fir : Weighted sum over the window.

    Notes
    -----
    For the window :math:`x_0, \dots, x_{T-1}`, from the oldest sample to the
    newest:

    .. math:: y = (x_{\text{start}}, \dots, x_{\text{end} - 1})

    Basic Examples
    --------------
    The two middle samples of the window ``[1, 2, 3, 4]``:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Fir, Input, Modely, Output, TimeRange, TimeSelect
    >>> x = Input("x", tw=4)
    >>> model = Modely("part", inputs=[x], outputs=[Output("y", TimeRange(1, 3)(x))]).build()
    >>> data = {"x": np.array([[[1.0], [2.0], [3.0], [4.0]]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[[2.],
            [3.]]], dtype=float32)

    Scope
    -----
    The features and the sequence are kept:

    >>> TimeRange(0, 2)(Input("v", dim=3, seq=4, tw=6)).shape
    Shape(S=(4,), T=2, D=(3,))

    Options
    -------
    ``None`` leaves a bound open, and negative bounds count from the end:

    >>> outputs = [
    ...     Output("newest", TimeRange(-2, None)(x)),
    ...     Output("oldest", TimeRange(None, 2)(x)),
    ...     Output("but_last", TimeRange(0, -1)(x)),
    ... ]
    >>> result = Modely("bounds", inputs=[x], outputs=outputs).build()(data)
    >>> [keras.ops.convert_to_numpy(result[k]).ravel() for k in ("newest", "oldest", "but_last")]
    [array([3., 4.], dtype=float32), array([1., 2.], dtype=float32), array([1., 2., 3.], dtype=float32)]

    ``name`` names the layer:

    >>> TimeRange(0, 2, name="past")(x).name
    'past'

    Applications
    ------------
    A model of the next sample reads the past 4 samples of a signal, and is
    trained against the sample after them:

    >>> y = Input("y", tw=(4, 1))
    >>> prediction = Fir()(TimeRange(0, 4)(y))
    >>> target = TimeSelect(-1)(y)

    Properties & Relations
    ----------------------
    ``TimeRange(i, i + 1)`` has the shape of ``TimeSelect(i)``:

    >>> TimeRange(1, 2)(x).shape, TimeSelect(1)(x).shape
    (Shape(S=(), T=1, D=(1,)), Shape(S=(), T=1, D=(1,)))

    Possible Issues
    ---------------
    The end is excluded, so a range from a sample to itself is empty:

    >>> TimeRange(2, 2)(x)
    Traceback (most recent call last):
        ...
    ValueError: ...invalid range [2, 2) for an axis of size 4.
    >>> TimeRange(2, 3)(x).shape
    Shape(S=(), T=1, D=(1,))

    .. versionadded:: 2.0
    """

    def __init__(self, start: int | None, end: int | None, name=None):
        self._start = None if start is None else int(start)
        self._end = None if end is None else int(end)
        super().__init__(name=name, start=self._start, end=self._end)

    def output_shape(self, *inputs):
        self._needs_window(inputs[0])
        shape = inputs[0].shape
        return shape.replace(
            tw=_range_length(self.name, self._start, self._end, shape.tw)
        )

    def build_layer(self):
        shape = self._preds[0].shape  # type: ignore[union-attr]
        return RangeImpl(
            start=self._start, end=self._end, axis=shape.tw_axis, name=self.name
        )


@keras.saving.register_keras_serializable(package="nnodely")
class TimeConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along the tensor axis ``axis``: the window
    or a sequence axis."""

    def __init__(self, axis: int, input_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.input_rank = int(input_rank)

    def call(self, xs):
        values = _concatenation_values(xs, self.input_rank)
        return keras.ops.concatenate(values, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return _concatenated_output_shape(input_shape, self.axis)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "input_rank": self.input_rank,
            }
        )
        return config


class TimeConcatenate(Layer):
    r"""Two or more streams joined along the window.

    Syntax
    ------
    ``TimeConcatenate([a, b])``
        The window of ``a`` followed by that of ``b``; the same as
        ``TimeConcatenate()([a, b])``.
    ``TimeConcatenate(name="w")([a, b, c])``
        Three or more streams, with the layer named.

    The samples of ``a`` come first, so they are the oldest of the result.
    Every stream has a window, and the same sequence and features.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``TimeConcatenate`` followed by
        a number.

    Call arguments
    --------------
    [a, b, ...] : list of Stream
        Two or more streams with a window, in the order of the result.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with ``T`` the sum of the windows. For example,
        ``T=2`` and ``T=3`` give ``T=5``.

    Raises
    ------
    ValueError
        If the list holds fewer than two streams ("concatenate requires at
        least two inputs"). If a stream has no window ("TimeConcatenate acts on
        the window of its stream"). If the sequences or the features differ
        ("all inputs must have the same sequence and dim").

    See Also
    --------
    TimeRange : A part of the window.
    TimeSelect : One sample of the window.
    Concatenate : Streams joined along a dim axis.
    SeqConcatenate : Streams joined along a sequence axis.

    Notes
    -----
    For the windows :math:`a_0, \dots, a_{n-1}` and :math:`b_0, \dots,
    b_{m-1}`:

    .. math:: y = (a_0, \dots, a_{n-1}, b_0, \dots, b_{m-1})

    Basic Examples
    --------------
    The two oldest samples of the window ``[1, 2, 3, 4]``, then the newest:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, TimeConcatenate, TimeRange, TimeSelect
    >>> x = Input("x", tw=4)
    >>> y = TimeConcatenate([TimeRange(0, 2)(x), TimeSelect(-1)(x)])
    >>> model = Modely("join", inputs=[x], outputs=[Output("y", y)]).build()
    >>> data = {"x": np.array([[[1.0], [2.0], [3.0], [4.0]]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"]).ravel()
    array([1., 2., 4.], dtype=float32)

    Scope
    -----
    Samples of one step join into a window, and a sequence is kept:

    >>> TimeConcatenate([Input("p", tw=1), Input("q", tw=1), Input("r", tw=1)]).shape
    Shape(S=(), T=3, D=(1,))
    >>> s = Input("s", dim=2, seq=3, tw=2)
    >>> TimeConcatenate([s, s]).shape
    Shape(S=(3,), T=4, D=(2,))

    Options
    -------
    ``name`` names the layer:

    >>> TimeConcatenate(name="window")([x, x]).name
    'window'

    Applications
    ------------
    A window shifted by one sample: the oldest one dropped, a new one, such
    as a prediction, appended:

    >>> prediction = TimeSelect(-1)(x) * 2.0
    >>> shifted = TimeConcatenate([TimeRange(1, 4)(x), prediction])
    >>> shifted.shape
    Shape(S=(), T=4, D=(1,))

    Properties & Relations
    ----------------------
    Joining the two halves of a window gives the window back:

    >>> halves = TimeConcatenate([TimeRange(0, 2)(x), TimeRange(2, 4)(x)])
    >>> model = Modely("halves", inputs=[x], outputs=[Output("y", halves)]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"]).ravel()
    array([1., 2., 3., 4.], dtype=float32)

    Possible Issues
    ---------------
    The streams have the same features. To join features, use
    :class:`Concatenate`:

    >>> v = Input("v", dim=2, tw=1)
    >>> TimeConcatenate([x, v])
    Traceback (most recent call last):
        ...
    ValueError: ...all inputs must have the same sequence and dim...

    The streams go in one list: ``TimeConcatenate(a, b)`` reads ``b`` as the
    name of the layer, and fails.

    .. versionadded:: 2.0
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        self._inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        for input_node in inputs:
            self._needs_window(input_node)
        shape = inputs[0].shape
        for input_node in inputs[1:]:
            other = input_node.shape
            if other.dim != shape.dim or other.seq != shape.seq:
                raise ValueError(
                    f"{self.name}: all inputs must have the same sequence and dim, "
                    f"got {shape} and {other}."
                )
        return shape.replace(tw=sum(node.shape.tw for node in inputs))  # type: ignore[misc]

    def build_layer(self):
        shape = _concatenation_inputs(self)[0].shape
        return TimeConcatenateImpl(
            axis=shape.tw_axis,  # type: ignore[arg-type]
            input_rank=shape.rank,
            name=self.name,
        )


class TimeReverse(Layer):
    r"""The window of a stream in reverse order, newest sample first.

    Syntax
    ------
    ``TimeReverse(x)``
        The window of ``x``, newest sample first; the same as
        ``TimeReverse()(x)``.
    ``TimeReverse(name="r")(x)``
        The same, with the layer named.

    The shape is unchanged. The order is reversed at every step of the
    sequence, and the features are kept.

    Parameters
    ----------
    name : str, optional
        The name of the layer. The default is ``TimeReverse`` followed by a
        number.

    Call arguments
    --------------
    x : Stream
        A stream with a window, ``Shape(S, T, D)``.

    Returns
    -------
    Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``.

    Raises
    ------
    ValueError
        If ``x`` has no window ("TimeReverse acts on the window of its
        stream").

    See Also
    --------
    TimeSelect : One sample of the window.
    TimeRange : A part of the window.
    SeqReverse : A sequence axis in reverse order.

    Notes
    -----
    For the window :math:`x_0, \dots, x_{T-1}`, from the oldest sample to the
    newest:

    .. math:: y_i = x_{T-1-i}

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, TimeReverse, TimeSelect
    >>> x = Input("x", tw=4)
    >>> model = Modely("reverse", inputs=[x], outputs=[Output("y", TimeReverse(x))]).build()
    >>> data = {"x": np.array([[[1.0], [2.0], [3.0], [4.0]]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"]).ravel()
    array([4., 3., 2., 1.], dtype=float32)

    Scope
    -----
    The features and the sequence are kept:

    >>> TimeReverse(Input("v", dim=3, seq=2, tw=5)).shape
    Shape(S=(2,), T=5, D=(3,))

    Properties & Relations
    ----------------------
    The first sample of the reversed window is the newest one:

    >>> outputs = [Output("a", TimeSelect(0)(TimeReverse(x))), Output("b", TimeSelect(-1)(x))]
    >>> result = Modely("first", inputs=[x], outputs=outputs).build()(data)
    >>> keras.ops.convert_to_numpy(result["a"]), keras.ops.convert_to_numpy(result["b"])
    (array([[[4.]]], dtype=float32), array([[[4.]]], dtype=float32))

    Possible Issues
    ---------------
    A stream without a window has nothing to reverse:

    >>> TimeReverse(Input("plain"))
    Traceback (most recent call last):
        ...
    ValueError: ...TimeReverse acts on the window of its stream, and 'plain' has none...

    .. versionadded:: 2.0
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        self._needs_window(inputs[0])
        return inputs[0].shape

    def build_layer(self):
        shape = self._preds[0].shape  # type: ignore[union-attr]
        return TimeReverseImpl(axis=shape.tw_axis, name=self.name)


class SeqSelect(Layer):
    r"""One step of a sequence axis of a stream, the axis removed.

    Syntax
    ------
    ``SeqSelect(idx)(x)``
        Step ``idx`` of the outermost sequence axis of ``x``.
    ``SeqSelect(idx, axis=k)(x)``
        Step ``idx`` of sequence axis ``k``.

    The sequence axes are counted outermost first: ``axis=0``, the default,
    is the outermost one, the rollout of the outermost :class:`Loop`. The
    selected axis is removed, while :class:`TimeSelect` keeps the window. The
    window and the features are kept.

    A negative ``idx`` counts from the end, also on a dynamic axis, whose
    length is known only from the data: ``SeqSelect(-1)`` is the last step of
    a rollout of any length.

    Parameters
    ----------
    idx : int
        The step, counted from 0. A negative one counts from the end: ``-1``
        is the last. On an axis of known length ``n`` it must lie in
        ``[-n, n)``.
    axis : int, default 0
        The sequence axis, counted from the outermost, 0. A negative one
        counts from the innermost, ``-1``.
    name : str, optional
        The name of the layer. The default is ``SeqSelect`` followed by a
        number.

    Call arguments
    --------------
    x : Stream
        A stream with a sequence, ``Shape(S, T, D)``. The call takes it alone
        or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` without sequence axis ``axis``. For example,
        ``S=(5,)`` gives ``S=()``, and ``S=(2, 3)`` gives ``S=(2,)`` with
        ``axis=1``.

    Raises
    ------
    ValueError
        If ``idx`` is outside the axis ("idx 3 out of bounds for seq axis 0, of
        size 3"), or if ``x`` has no sequence axis ``axis`` ("axis 0 out of
        bounds for seq rank 0").

    See Also
    --------
    SeqRange : A part of a sequence axis.
    SeqStack : Streams stacked into a new sequence axis.
    TimeSelect : One sample of the window.
    Loop : A rollout along a sequence axis.

    Notes
    -----
    For the steps :math:`x_0, \dots, x_{n-1}` of the axis:

    .. math:: y = x_{\text{idx}}, \qquad x_{-k} = x_{n-k}

    Basic Examples
    --------------
    The first and the last step of a sequence ``[1, 2, 3]``:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, SeqRange, SeqSelect
    >>> s = Input("s", seq=3, tw=1)
    >>> outputs = [Output("first", SeqSelect(0)(s)), Output("last", SeqSelect(-1)(s))]
    >>> model = Modely("steps", inputs=[s], outputs=outputs).build()
    >>> data = {"s": np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)}
    >>> result = model(data)
    >>> keras.ops.convert_to_numpy(result["first"]), keras.ops.convert_to_numpy(result["last"])
    (array([[[1.]]], dtype=float32), array([[[3.]]], dtype=float32))
    >>> SeqSelect(-1)(s).shape
    Shape(S=(), T=1, D=(1,))

    Scope
    -----
    On a dynamic axis, ``-1`` is the last step, whatever the length of the
    data:

    >>> d = Input("d", seq=-1, tw=1)
    >>> model = Modely("dynamic", inputs=[d], outputs=[Output("y", SeqSelect(-1)(d))]).build()
    >>> keras.ops.convert_to_numpy(model({"d": np.arange(1.0, 6.0).reshape(1, 5, 1, 1)})["y"])
    array([[[5.]]], dtype=float32)

    Options
    -------
    ``axis`` picks the sequence axis, outermost first. On two sequences of 3
    steps, ``[1, 2, 3]`` and ``[4, 5, 6]``, the first step of each:

    >>> w = Input("w", seq=(2, 3), tw=1)
    >>> model = Modely("inner", inputs=[w], outputs=[Output("y", SeqSelect(0, axis=1)(w))]).build()
    >>> result = model({"w": np.arange(1.0, 7.0).reshape(1, 2, 3, 1, 1)})
    >>> keras.ops.convert_to_numpy(result["y"]).ravel()
    array([1., 4.], dtype=float32)
    >>> SeqSelect(0, axis=1)(w).shape
    Shape(S=(2,), T=1, D=(1,))

    ``name`` names the layer:

    >>> SeqSelect(-1, name="final")(s).name
    'final'

    Applications
    ------------
    A terminal objective reads the last step of a rollout, of any length:

    >>> position = Input("position", seq=-1, tw=1)
    >>> final_position = SeqSelect(-1)(position)

    Properties & Relations
    ----------------------
    ``SeqSelect(i)`` has the shape of ``SeqRange(i, i + 1)``, which removes the
    axis too:

    >>> SeqSelect(1)(s).shape, SeqRange(1, 2)(s).shape
    (Shape(S=(), T=1, D=(1,)), Shape(S=(), T=1, D=(1,)))

    Possible Issues
    ---------------
    A stream without a sequence has no step to select:

    >>> SeqSelect(0)(Input("plain", tw=1))
    Traceback (most recent call last):
        ...
    ValueError: ...axis 0 out of bounds for seq rank 0.

    .. versionadded:: 2.0
    """

    def __init__(self, idx: int, axis: int = 0, name=None):
        self._idx = int(idx)
        self._axis = int(axis)
        super().__init__(name=name, idx=self._idx, axis=self._axis)

    def _selected(self, shape):
        """The sequence axis and the index on it."""
        axis = shape.resolve_seq(self._axis, self.name)
        return axis, _index(self.name, self._idx, shape.seq[axis], f"seq axis {axis}")

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        axis, _ = self._selected(shape)
        return shape.replace(seq=shape.seq[:axis] + shape.seq[axis + 1 :])

    def build_layer(self):
        shape = self._preds[0].shape  # type: ignore[union-attr]
        axis, idx = self._selected(shape)
        return SelectImpl(
            idx=idx, axis=shape.seq_axes[axis], keep_dim=False, name=self.name
        )


class SeqRange(Layer):
    r"""A contiguous part ``[start, end)`` of a sequence axis of a stream.

    Syntax
    ------
    ``SeqRange(start, end)(x)``
        The steps ``start`` to ``end - 1`` of the outermost sequence axis.
    ``SeqRange(start, end, axis=k)(x)``
        The same part of sequence axis ``k``.

    The sequence axes are counted outermost first, as in :class:`SeqSelect`.
    The end is excluded. ``None`` leaves a bound open, and a negative bound
    counts from the end. The window and the features are kept.

    The axis is kept, with ``end - start`` steps. A range of one step removes
    it, as :class:`SeqSelect` does. On a dynamic axis, bounds of 0 or more
    give an axis of known length, while a negative bound or an open end give
    a dynamic axis.

    Parameters
    ----------
    start : int or None
        The first step, counted from 0. A negative one counts from the end.
        ``None`` starts from the first step.
    end : int or None
        The step after the last one. A negative one counts from the end.
        ``None`` reaches the last step.
    axis : int, default 0
        The sequence axis, counted from the outermost, 0. A negative one
        counts from the innermost, ``-1``.
    name : str, optional
        The name of the layer. The default is ``SeqRange`` followed by a
        number.

    Call arguments
    --------------
    x : Stream
        A stream with a sequence, ``Shape(S, T, D)``. The call takes it alone
        or in a list of one.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with sequence axis ``axis`` of ``end - start``
        steps, or without it for one step. For example, ``S=(10,)`` gives
        ``S=(3,)`` with ``SeqRange(2, 5)``, and ``S=(6, 4)`` gives ``S=(6,)``
        with ``SeqRange(1, 2, axis=1)``.

    Raises
    ------
    ValueError
        If the range is empty, reversed or beyond the axis ("invalid range [2,
        2) for an axis of size 3"), or if ``x`` has no sequence axis ``axis``
        ("out of bounds for seq rank").

    See Also
    --------
    SeqSelect : One step of a sequence axis.
    SeqConcatenate : Streams joined along a sequence axis.
    TimeRange : A part of the window.

    Notes
    -----
    For the steps :math:`x_0, \dots, x_{n-1}` of the axis:

    .. math:: y = (x_{\text{start}}, \dots, x_{\text{end} - 1})

    Basic Examples
    --------------
    The first two steps of a sequence ``[1, 2, 3]``:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, SeqRange, SeqSelect
    >>> s = Input("s", seq=3, tw=1)
    >>> model = Modely("part", inputs=[s], outputs=[Output("y", SeqRange(0, 2)(s))]).build()
    >>> data = {"s": np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)}
    >>> keras.ops.convert_to_numpy(model(data)["y"]).ravel()
    array([1., 2.], dtype=float32)
    >>> SeqRange(0, 2)(s).shape
    Shape(S=(2,), T=1, D=(1,))

    Scope
    -----
    On a dynamic axis, bounds of 0 or more give a known length, and a
    negative bound a dynamic one:

    >>> d = Input("d", seq=-1, tw=1)
    >>> SeqRange(0, 2)(d).shape, SeqRange(-2, None)(d).shape
    (Shape(S=(2,), T=1, D=(1,)), Shape(S=(dyn,), T=1, D=(1,)))

    Options
    -------
    ``None`` leaves a bound open, and negative bounds count from the end:

    >>> model = Modely("last_two", inputs=[s], outputs=[Output("y", SeqRange(-2, None)(s))]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"]).ravel()
    array([2., 3.], dtype=float32)

    ``axis`` picks the sequence axis, outermost first:

    >>> SeqRange(0, 2, axis=1)(Input("w", seq=(2, 3), tw=1)).shape
    Shape(S=(2, 2), T=1, D=(1,))

    ``name`` names the layer:

    >>> SeqRange(0, 2, name="start")(s).name
    'start'

    Properties & Relations
    ----------------------
    A range of one step removes the axis: ``SeqRange(i, i + 1)`` has the shape
    of ``SeqSelect(i)``:

    >>> SeqRange(1, 2)(s).shape, SeqSelect(1)(s).shape
    (Shape(S=(), T=1, D=(1,)), Shape(S=(), T=1, D=(1,)))

    Possible Issues
    ---------------
    The end is excluded, and cannot go beyond the axis:

    >>> SeqRange(0, 4)(s)
    Traceback (most recent call last):
        ...
    ValueError: ...invalid range [0, 4) for an axis of size 3.
    >>> SeqRange(0, None)(s).shape
    Shape(S=(3,), T=1, D=(1,))

    .. versionadded:: 2.0
    """

    def __init__(self, start: int | None, end: int | None, axis: int = 0, name=None):
        self._start = None if start is None else int(start)
        self._end = None if end is None else int(end)
        self._axis = int(axis)
        super().__init__(name=name, start=self._start, end=self._end, axis=self._axis)

    def _range(self, shape):
        """The sequence axis and the length of the range on it."""
        axis = shape.resolve_seq(self._axis, self.name)
        return axis, _range_length(self.name, self._start, self._end, shape.seq[axis])

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        axis, length = self._range(shape)
        kept = () if length == 1 else (length,)
        return shape.replace(seq=shape.seq[:axis] + kept + shape.seq[axis + 1 :])

    def build_layer(self):
        shape = self._preds[0].shape  # type: ignore[union-attr]
        axis, length = self._range(shape)
        if length == 1:
            # One step: selected, and its axis removed.
            return SelectImpl(
                idx=0 if self._start is None else self._start,
                axis=shape.seq_axes[axis],
                keep_dim=False,
                name=self.name,
            )
        return RangeImpl(
            start=self._start, end=self._end, axis=shape.seq_axes[axis], name=self.name
        )


class SeqConcatenate(Layer):
    r"""Two or more streams joined along a sequence axis.

    Syntax
    ------
    ``SeqConcatenate()([a, b])``
        The steps of ``a`` followed by those of ``b``, along the outermost
        sequence axis.
    ``SeqConcatenate(axis=k)([a, b, c])``
        Three or more streams, joined along sequence axis ``k``.

    The sequence axes are counted outermost first, as in :class:`SeqSelect`.
    The streams have the same features, the same window and the same other
    sequence axes. When the joined axis is dynamic in one stream, it is
    dynamic in the result.

    Parameters
    ----------
    axis : int, default 0
        The sequence axis to join along, counted from the outermost, 0. A
        negative one counts from the innermost, ``-1``.
    name : str, optional
        The name of the layer. The default is ``SeqConcatenate`` followed by a
        number.

    Call arguments
    --------------
    [a, b, ...] : list of Stream
        Two or more streams with a sequence, in the order of the result.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with sequence axis ``axis`` as long as the sum of
        the streams' lengths. For example, ``S=(2,)`` and ``S=(3,)`` give
        ``S=(5,)``; ``S=(3,)`` and ``S=(dyn,)`` give ``S=(dyn,)``.

    Raises
    ------
    ValueError
        If the list holds fewer than two streams ("concatenate requires at
        least two inputs"). If the streams differ elsewhere ("all inputs must
        have the same dim, window and sequence axes but seq axis 0"). If a
        stream has no sequence axis ``axis`` ("out of bounds for seq rank").
    TypeError
        If the layer is called before it is configured,
        ``SeqConcatenate([a, b])``.

    See Also
    --------
    SeqStack : Streams stacked into a new sequence axis.
    SeqRange : A part of a sequence axis.
    TimeConcatenate : Streams joined along the window.
    Concatenate : Streams joined along a dim axis.

    Notes
    -----
    For the steps :math:`a_0, \dots, a_{n-1}` and :math:`b_0, \dots, b_{m-1}`:

    .. math:: y = (a_0, \dots, a_{n-1}, b_0, \dots, b_{m-1})

    Basic Examples
    --------------
    A sequence ``[1, 2]`` followed by ``[3, 4, 5]``:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, SeqConcatenate
    >>> a, b = Input("a", seq=2, tw=1), Input("b", seq=3, tw=1)
    >>> y = SeqConcatenate()([a, b])
    >>> model = Modely("join", inputs=[a, b], outputs=[Output("y", y)]).build()
    >>> data = {"a": np.array([1.0, 2.0]).reshape(1, 2, 1, 1), "b": np.array([3.0, 4.0, 5.0]).reshape(1, 3, 1, 1)}
    >>> keras.ops.convert_to_numpy(model(data)["y"]).ravel()
    array([1., 2., 3., 4., 5.], dtype=float32)
    >>> y.shape
    Shape(S=(5,), T=1, D=(1,))

    Scope
    -----
    A dynamic axis makes the joined axis dynamic:

    >>> SeqConcatenate()([a, Input("d", seq=-1, tw=1)]).shape
    Shape(S=(dyn,), T=1, D=(1,))

    Options
    -------
    ``axis`` picks the sequence axis, outermost first:

    >>> w = Input("w", seq=(2, 3), tw=1)
    >>> SeqConcatenate(axis=1)([w, w]).shape
    Shape(S=(2, 6), T=1, D=(1,))

    ``name`` names the layer:

    >>> SeqConcatenate(name="trajectory")([a, b]).name
    'trajectory'

    Applications
    ------------
    A trajectory made of a measured part and a predicted one, as one
    sequence for an objective:

    >>> measured, predicted = Input("measured", seq=4, tw=1), Input("predicted", seq=6, tw=1)
    >>> SeqConcatenate()([measured, predicted]).shape
    Shape(S=(10,), T=1, D=(1,))

    Possible Issues
    ---------------
    The streams have the same window. Take the same part of each with
    :class:`TimeRange` or :class:`TimeSelect` first:

    >>> SeqConcatenate()([a, Input("c", seq=2, tw=2)])
    Traceback (most recent call last):
        ...
    ValueError: ...all inputs must have the same dim, window and sequence axes but seq axis 0...

    Streams without a sequence have no axis to join along. Stack them into a
    new one with :class:`SeqStack`:

    >>> p, q = Input("p", tw=1), Input("q", tw=1)
    >>> SeqConcatenate()([p, q])
    Traceback (most recent call last):
        ...
    ValueError: ...axis 0 out of bounds for seq rank 0.

    .. versionadded:: 2.0
    """

    def __init__(self, axis: int = 0, name=None):
        self._axis = int(axis)
        super().__init__(name=name, axis=self._axis)

    def output_shape(self, *inputs):
        self._inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        shape = inputs[0].shape
        axis = shape.resolve_seq(self._axis, self.name)

        def others(seq):
            return seq[:axis] + seq[axis + 1 :]

        for input_node in inputs[1:]:
            other = input_node.shape
            if (
                other.dim != shape.dim
                or other.tw != shape.tw
                or other.seq_rank != shape.seq_rank
                or others(other.seq) != others(shape.seq)
            ):
                raise ValueError(
                    f"{self.name}: all inputs must have the same dim, window and "
                    f"sequence axes but seq axis {self._axis}, got {shape} and {other}."
                )
        sizes = [input_node.shape.seq[axis] for input_node in inputs]
        joined = None if None in sizes else sum(int(size) for size in sizes)  # type: ignore[arg-type]
        return shape.replace(seq=shape.seq[:axis] + (joined,) + shape.seq[axis + 1 :])

    def build_layer(self):
        shape = _concatenation_inputs(self)[0].shape
        axis = shape.resolve_seq(self._axis, self.name)
        return TimeConcatenateImpl(
            axis=shape.seq_axes[axis], input_rank=shape.rank, name=self.name
        )


class SeqStack(Layer):
    r"""Two or more streams of one shape stacked into a new sequence axis.

    Syntax
    ------
    ``SeqStack()([a, b])``
        ``a`` and ``b`` as the two steps of a new outermost sequence axis.
    ``SeqStack(axis=-1)([a, b, c])``
        Three or more streams, stacked into a new innermost sequence axis.

    The new axis has one step per stream, in the order of the list. ``axis``
    is its position among the sequence axes of the result, counted outermost
    first. The default, 0, makes it the new outermost axis, as a Loop's
    rollout is; ``-1`` makes it the new innermost one. The streams have the
    same shape, and the window and the features are kept.

    Parameters
    ----------
    axis : int, default 0
        The position of the new axis among the sequence axes of the result,
        counted from the outermost, 0. A negative one counts from the
        innermost, ``-1``.
    name : str, optional
        The name of the layer. The default is ``SeqStack`` followed by a
        number.

    Call arguments
    --------------
    [a, b, ...] : list of Stream
        Two or more streams of the same shape, in the order of the steps.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with a new sequence axis of as many steps as
        streams. For example, 4 streams of ``S=()`` give ``S=(4,)``, and 2 of
        ``S=(5,)`` give ``S=(2, 5)``, or ``S=(5, 2)`` with ``axis=-1``.

    Raises
    ------
    ValueError
        If the list holds fewer than two streams ("concatenate requires at
        least two inputs"). If the shapes differ ("all inputs must have the
        same shape when stacking"). If ``axis`` is beyond the sequence axes of
        the result ("out of bounds for seq rank").
    TypeError
        If the layer is called before it is configured, ``SeqStack([a, b])``.

    See Also
    --------
    SeqConcatenate : Streams joined along an existing sequence axis.
    SeqSelect : One step of a sequence axis.
    TimeConcatenate : Streams joined along the window.

    Notes
    -----
    For the streams :math:`x^{(0)}, \dots, x^{(k-1)}`, step :math:`i` of the
    new axis is :math:`x^{(i)}`.

    Basic Examples
    --------------
    Three scalar samples as a sequence of 3 steps:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, SeqSelect, SeqStack, TimeConcatenate
    >>> a, b, c = Input("a", tw=1), Input("b", tw=1), Input("c", tw=1)
    >>> y = SeqStack()([a, b, c])
    >>> y.shape
    Shape(S=(3,), T=1, D=(1,))
    >>> model = Modely("stack", inputs=[a, b, c], outputs=[Output("y", y)]).build()
    >>> data = {"a": np.array([[[1.0]]]), "b": np.array([[[2.0]]]), "c": np.array([[[3.0]]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"]).ravel()
    array([1., 2., 3.], dtype=float32)

    Scope
    -----
    Streams with a sequence get a new outermost axis:

    >>> s = Input("s", seq=5, tw=3)
    >>> SeqStack()([s, s]).shape
    Shape(S=(2, 5), T=3, D=(1,))

    Options
    -------
    ``axis=-1`` makes the new axis the innermost:

    >>> SeqStack(axis=-1)([s, s]).shape
    Shape(S=(5, 2), T=3, D=(1,))

    ``name`` names the layer:

    >>> SeqStack(name="steps")([a, b]).name
    'steps'

    Properties & Relations
    ----------------------
    Step ``i`` of the stack is stream ``i``:

    >>> model = Modely("second", inputs=[a, b, c], outputs=[Output("y", SeqSelect(1)(y))]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[[2.]]], dtype=float32)

    Possible Issues
    ---------------
    ``SeqStack`` makes a sequence of steps, not a window. To join samples
    into a window, use :class:`TimeConcatenate`:

    >>> TimeConcatenate([a, b, c]).shape
    Shape(S=(), T=3, D=(1,))

    The streams have the same shape:

    >>> SeqStack()([a, Input("w", tw=2)])
    Traceback (most recent call last):
        ...
    ValueError: ...all inputs must have the same shape when stacking.

    .. versionadded:: 2.0
    """

    def __init__(self, axis: int = 0, name=None):
        self._axis = int(axis)
        super().__init__(name=name, axis=self._axis)

    def _stacked(self, inputs):
        """The shape of the stack, and the position of its new axis."""
        shape = inputs[0].shape
        axis = resolve_axis(self._axis, shape.seq_rank + 1, self.name, "seq")
        seq = shape.seq[:axis] + (len(inputs),) + shape.seq[axis:]
        return shape.replace(seq=seq), axis

    def output_shape(self, *inputs):
        self._inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        if any(
            input_node.shape.dimensions != inputs[0].shape.dimensions
            for input_node in inputs
        ):
            raise ValueError(
                f"{self.name}: all inputs must have the same shape when stacking."
            )
        return self._stacked(inputs)[0]

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        stacked, axis = self._stacked(inputs)
        return StackImpl(
            axis=stacked.seq_axes[axis],
            input_rank=inputs[0].shape.rank,
            name=self.name,
        )


class SeqReverse(Layer):
    r"""A sequence axis of a stream in reverse order, last step first.

    Syntax
    ------
    ``SeqReverse()(x)``
        The outermost sequence axis of ``x``, last step first.
    ``SeqReverse(axis=k)(x)``
        Sequence axis ``k``, last step first.

    The sequence axes are counted outermost first, as in :class:`SeqSelect`.
    The shape is unchanged, a dynamic axis included.

    Parameters
    ----------
    axis : int, default 0
        The sequence axis, counted from the outermost, 0. A negative one
        counts from the innermost, ``-1``.
    name : str, optional
        The name of the layer. The default is ``SeqReverse`` followed by a
        number.

    Call arguments
    --------------
    x : Stream
        A stream with a sequence, ``Shape(S, T, D)``. The call takes it alone
        or in a list of one.

    Returns
    -------
    Stream
        A stream of the shape of ``x``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``.

    Raises
    ------
    ValueError
        If ``x`` has no sequence axis ``axis`` ("axis 0 out of bounds for seq
        rank 0").
    TypeError
        If the layer is called before it is configured, ``SeqReverse(x)``.

    See Also
    --------
    TimeReverse : The window in reverse order.
    SeqSelect : One step of a sequence axis.
    SeqRange : A part of a sequence axis.

    Notes
    -----
    For the steps :math:`x_0, \dots, x_{n-1}` of the axis:

    .. math:: y_i = x_{n-1-i}

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output, SeqReverse, SeqSelect
    >>> s = Input("s", seq=3, tw=1)
    >>> model = Modely("reverse", inputs=[s], outputs=[Output("y", SeqReverse()(s))]).build()
    >>> data = {"s": np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)}
    >>> keras.ops.convert_to_numpy(model(data)["y"]).ravel()
    array([3., 2., 1.], dtype=float32)

    Scope
    -----
    A dynamic axis is reversed whatever the length of the data:

    >>> d = Input("d", seq=-1, tw=1)
    >>> model = Modely("dynamic", inputs=[d], outputs=[Output("y", SeqReverse()(d))]).build()
    >>> keras.ops.convert_to_numpy(model({"d": np.arange(1.0, 6.0).reshape(1, 5, 1, 1)})["y"]).ravel()
    array([5., 4., 3., 2., 1.], dtype=float32)

    Options
    -------
    ``axis`` picks the sequence axis, outermost first. On two sequences of 3
    steps, ``[1, 2, 3]`` and ``[4, 5, 6]``, ``axis=1`` reverses each:

    >>> w = Input("w", seq=(2, 3), tw=1)
    >>> model = Modely("inner", inputs=[w], outputs=[Output("y", SeqReverse(axis=1)(w))]).build()
    >>> keras.ops.convert_to_numpy(model({"w": np.arange(1.0, 7.0).reshape(1, 2, 3, 1, 1)})["y"]).ravel()
    array([3., 2., 1., 6., 5., 4.], dtype=float32)

    ``name`` names the layer:

    >>> SeqReverse(name="backward")(s).name
    'backward'

    Properties & Relations
    ----------------------
    The first step of the reversed sequence is the last one:

    >>> outputs = [Output("a", SeqSelect(0)(SeqReverse()(s))), Output("b", SeqSelect(-1)(s))]
    >>> result = Modely("first", inputs=[s], outputs=outputs).build()(data)
    >>> keras.ops.convert_to_numpy(result["a"]), keras.ops.convert_to_numpy(result["b"])
    (array([[[3.]]], dtype=float32), array([[[3.]]], dtype=float32))

    Possible Issues
    ---------------
    ``SeqReverse`` has a setting, so it is configured first:

    >>> SeqReverse(s)
    Traceback (most recent call last):
        ...
    TypeError: SeqReverse has settings, so it is configured first and then called on its streams: SeqReverse(...)(x), not SeqReverse(x).

    .. versionadded:: 2.0
    """

    def __init__(self, axis: int = 0, name=None):
        self._axis = int(axis)
        super().__init__(name=name, axis=self._axis)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        shape.resolve_seq(self._axis, self.name)
        return shape

    def build_layer(self):
        shape = self._preds[0].shape  # type: ignore[union-attr]
        axis = shape.resolve_seq(self._axis, self.name)
        return TimeReverseImpl(axis=shape.seq_axes[axis], name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class ConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along a dim axis, the tensor axis ``axis``
    of the inputs of ``input_rank`` axes; the inputs ``repeated`` flags, of a
    single sample, are repeated along the sequence and the window of the
    input ``reference``."""

    def __init__(
        self, axis: int, input_rank: int, repeated=(), reference=0, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.input_rank = int(input_rank)
        self.repeated = tuple(bool(flag) for flag in repeated)
        self.reference = int(reference)

    def call(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) < 2:
            raise ValueError(
                "Concatenate implementations require at least two tensors."
            )
        rank = self.input_rank + 1
        # Zeros of one feature along the axis, shaped like the reference:
        # added to a single sample, they repeat it along its window.
        zeros = keras.ops.zeros_like(
            slice_axis(xs[self.reference], self.axis, slice(0, 1))
        )
        values = [
            expand_to_rank(value, rank) + zeros if repeated else value
            for value, repeated in zip(xs, self.repeated or (False,) * len(xs))
        ]
        return keras.ops.concatenate(values, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return _concatenated_output_shape(input_shape, self.axis)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "input_rank": self.input_rank,
                "repeated": self.repeated,
                "reference": self.reference,
            }
        )
        return config


class Concatenate(Layer):
    r"""Two or more streams joined along a dim axis.

    Syntax
    ------
    ``Concatenate()([a, b])``
        The features of ``a`` followed by those of ``b``, along the first dim
        axis.
    ``Concatenate(axis=k)([a, b, c])``
        Three or more streams, joined along dim axis ``k``.

    The streams have the same sequence and the same window. The exceptions:

    - a single sample, ``T=1``, is repeated along the window of the others;
    - a value without a window, a :class:`Parameter` or a
      :class:`Constant`, is repeated along the window of the others.

    An Input without a window does not join a stream with one. The dim axes
    other than ``axis`` have the same length in every stream.

    Parameters
    ----------
    axis : int, default 0
        The dim axis to join along, counted from 0. A negative one counts from
        the last.
    name : str, optional
        The name of the layer. The default is ``Concatenate`` followed by a
        number.

    Call arguments
    --------------
    [a, b, ...] : list of Stream
        Two or more streams, in the order of the result.

    Returns
    -------
    Stream
        ``Shape(S, T, D)`` with dim axis ``axis`` as long as the sum of the
        lengths in the streams. For example, ``D=(1,)`` and ``D=(2,)`` give
        ``D=(3,)``; ``D=(2, 3)`` twice gives ``D=(4, 3)``, or ``D=(2, 6)``
        with ``axis=1``.

    Raises
    ------
    ValueError
        If the list holds fewer than two streams ("concatenate requires at
        least two inputs"). If the streams do not match ("all inputs must have
        matching dimensions except dim axis 0, and the same sequence and window
        or a single sample"). If ``axis`` is not a dim axis.
    TypeError
        If the layer is called before it is configured, ``Concatenate([a,
        b])``, or on a number ("is called on streams, got float").

    See Also
    --------
    Select : One index of a dim axis.
    Range : A part of a dim axis.
    TimeConcatenate : Streams joined along the window.
    SeqConcatenate : Streams joined along a sequence axis.
    SeqStack : Streams stacked into a new sequence axis.

    Notes
    -----
    For two vectors :math:`a = (a_1, \dots, a_n)` and
    :math:`b = (b_1, \dots, b_m)`:

    .. math:: y = (a_1, \dots, a_n, b_1, \dots, b_m)

    Basic Examples
    --------------
    A scalar and a vector of 2 features, over a window of 2 samples:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Concatenate, Input, Modely, Output, Parameter, TimeSelect
    >>> a, b = Input("a", tw=2), Input("b", dim=2, tw=2)
    >>> model = Modely("join", inputs=[a, b], outputs=[Output("y", Concatenate()([a, b]))]).build()
    >>> data = {"a": np.array([[[1.0], [2.0]]]), "b": np.array([[[3.0, 4.0], [5.0, 6.0]]])}
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[[1., 3., 4.],
            [2., 5., 6.]]], dtype=float32)

    Scope
    -----
    A Parameter, without a window, is repeated at every sample:

    >>> p = Parameter("p", value=[5.0])
    >>> model = Modely("value", inputs=[a], outputs=[Output("y", Concatenate()([a, p]))]).build()
    >>> keras.ops.convert_to_numpy(model({"a": data["a"]})["y"])
    array([[[1., 5.],
            [2., 5.]]], dtype=float32)

    So is a single sample, ``T=1``:

    >>> newest = TimeSelect(-1)(b)
    >>> model = Modely("sample", inputs=[a, b], outputs=[Output("y", Concatenate()([a, newest]))]).build()
    >>> keras.ops.convert_to_numpy(model(data)["y"])
    array([[[1., 5., 6.],
            [2., 5., 6.]]], dtype=float32)

    Options
    -------
    ``axis`` picks the dim axis of matrices:

    >>> m = Input("m", dim=(2, 3))
    >>> Concatenate()([m, m]).shape, Concatenate(axis=1)([m, m]).shape
    (Shape(S=(), T=None, D=(4, 3)), Shape(S=(), T=None, D=(2, 6)))

    ``name`` names the layer:

    >>> Concatenate(name="state")([a, b]).name
    'state'

    Applications
    ------------
    The state of a mass-spring-damper, its position and its velocity, as one
    vector for a :class:`Linear` map:

    >>> from nnodely import Linear
    >>> position, velocity = Input("position", tw=1), Input("velocity", tw=1)
    >>> acceleration = Linear()(Concatenate()([position, velocity]))

    Possible Issues
    ---------------
    An Input without a window does not join one with a window. Declare it
    with a window of one sample, which is repeated:

    >>> Concatenate()([a, Input("u")])
    Traceback (most recent call last):
        ...
    ValueError: ...all inputs must have matching dimensions except dim axis 0, and the same sequence and window or a single sample.
    >>> Concatenate()([a, Input("u", tw=1)]).shape
    Shape(S=(), T=2, D=(2,))

    The layer is configured first, then called on the list:
    ``Concatenate()([a, b])``, not ``Concatenate([a, b])``.

    .. versionadded:: 2.0
    """

    def __init__(self, axis: int = 0, name=None):
        self._axis = int(axis)
        super().__init__(name=name, axis=self._axis)

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        index = next(
            (i for i, node in enumerate(inputs) if not _is_scalar(node.shape._steps)),
            0,
        )
        reference = inputs[index].shape
        axis = reference.resolve_dim(self._axis, self.name)

        for input_node in inputs:
            shape = input_node.shape
            compatible_dim = shape.dim_rank == reference.dim_rank and all(
                size == reference.dim[index]
                for index, size in enumerate(shape.dim)
                if index != axis
            )
            # A single sample broadcasts as arithmetic broadcasts it: with a
            # window of one, or as a value; data without a window does not
            # take one.
            single = _is_scalar(shape._steps) and (
                shape.tw is not None or input_node._is_value or reference.tw is None
            )
            if not compatible_dim or not (shape._steps == reference._steps or single):
                raise ValueError(
                    f"{self.name}: all inputs must have matching dimensions "
                    f"except dim axis {self._axis}, and the same sequence and "
                    "window or a single sample."
                )

        return ConcatenateImpl(
            axis=reference.dim_axes[axis],
            input_rank=reference.rank,
            repeated=[node.shape._steps != reference._steps for node in inputs],
            reference=index,
            name=self.name,
        )


def _concatenation_inputs(layer: Layer) -> list[Stream]:
    inputs = list(layer._inputs or layer._preds)
    if len(inputs) < 2:
        raise ValueError(f"{layer.name}: concatenate requires at least two inputs.")
    if not all(isinstance(input_node, Stream) for input_node in inputs):
        raise TypeError(f"{layer.name}: concatenate inputs must be Stream objects.")
    return [input_node for input_node in inputs if isinstance(input_node, Stream)]


def _concatenation_values(xs, input_rank):
    """The tensors to concatenate, each of ``input_rank`` axes and a batch."""
    if not isinstance(xs, (list, tuple)) or len(xs) < 2:
        raise ValueError("Concatenate implementations require at least two tensors.")
    if any(len(value.shape) != input_rank + 1 for value in xs):
        raise ValueError(
            "All tensors must be laid out (batch, *seq, [tw,] *dim), of rank "
            f"{input_rank + 1}."
        )
    return list(xs)


def _concatenated_output_shape(input_shapes, axis):
    shapes: list[list[int | None]] = [list(shape) for shape in input_shapes]
    if len(shapes) < 2:
        raise ValueError("Concatenate implementations require at least two tensors.")

    # A single sample of fewer sequence axes lacks leading ones, after the
    # batch, and is repeated along the others' sequence and window.
    rank = max(len(shape) for shape in shapes)
    shapes = [shape[:1] + [1] * (rank - len(shape)) + shape[1:] for shape in shapes]
    output_shape: list[int | None] = [
        next((size for size in sizes if size != 1), 1) for sizes in zip(*shapes)
    ]
    axis_sizes = [shape[axis] for shape in shapes]
    output_shape[axis] = (
        None
        if any(size is None for size in axis_sizes)
        else sum(int(size) for size in axis_sizes)
    )
    return tuple(output_shape)


def _index(owner: str, idx: int, size, where: str) -> int:
    """``idx`` on an axis ``size`` long, ``where`` naming it, checked and
    counted from the start; on a dynamic axis a negative one stays counted
    from the end."""
    if size is None:
        return idx
    if not -size <= idx < size:
        raise ValueError(
            f"{owner}: idx {idx} out of bounds for {where}, of size {size}."
        )
    return idx % size


def _range_length(name: str, start, end, size):
    """Length of ``[start, end)`` on an axis of ``size``, None when dynamic.

    An open start is 0 and an open end the length of the axis. On a dynamic
    axis a negative or open end depends on the data, so the result is dynamic.
    """
    first = 0 if start is None else start
    if size is None:
        if end is None or first < 0 or end < 0:
            return None
        last = end
    else:
        first = first + size if first < 0 else first
        last = size if end is None else end + size if end < 0 else end
    if first < 0 or (size is not None and last > size) or first >= last:
        raise ValueError(
            f"{name}: invalid range [{start}, {end}) for an axis of size "
            f"{'dyn' if size is None else size}."
        )
    return last - first


@keras.saving.register_keras_serializable(package="nnodely")
class StackImpl(keras.layers.Layer):
    """Serializable stack of tensors into a new tensor axis."""

    def __init__(self, axis: int, input_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.input_rank = int(input_rank)

    def call(self, xs):
        values = _concatenation_values(xs, self.input_rank)
        return keras.ops.stack(values, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape.insert(self.axis, len(input_shape))
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "input_rank": self.input_rank})
        return config


@keras.saving.register_keras_serializable(package="nnodely")
class TimeReverseImpl(keras.layers.Layer):
    """Serializable reversal of the elements along a tensor axis."""

    def __init__(self, axis: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)

    def call(self, x):
        return keras.ops.flip(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
