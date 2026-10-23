"""Partitions of a stream: the windows, selections and slices of its dim
axes, its window and its sequence axes, and the concatenations that join
streams along them.

The dim operations count the dim axes from 0. The time operations act on
the window, ``T``, of which a stream has one: they take no axis, and always
keep it. The sequence operations act on a sequence axis, counted outermost
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
class SampleWindowImpl(keras.layers.Layer):
    def __init__(
        self,
        start: int,
        window_size: int,
        axis: int,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.start = int(start)
        self.window_size = int(window_size)
        self.axis = int(axis)

    def call(self, x):
        # A window that covers the whole time axis is the tensor itself. Taking
        # it as a slice anyway costs an op whose own derivative (a scatter into
        # a zero tensor) has no ONNX equivalent, so a differentiated graph would
        # stop exporting for no reason.
        if self.start == 0 and self.window_size == x.shape[self.axis]:
            return x

        return slice_axis(
            x, self.axis, slice(self.start, self.start + self.window_size)
        )

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.window_size
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "start": self.start,
                "window_size": self.window_size,
                "axis": self.axis,
            }
        )
        return config


class SampleWindow(Layer):
    """
    A window of an Input: its ``past`` samples up to the current one and its
    ``future`` samples after it, cut from the union of the Input's windows.
    """

    # The window is cut from the data of the Input, not its current sample.
    _reads_data = True

    def __init__(self, past: int, future: int, name=None):
        self.past = int(past)
        self.future = int(future)
        self.window_size = self.past + self.future
        super().__init__(
            name=name,
            time=self.window_size,
            past=self.past,
            future=self.future,
        )

    def output_shape(self, *inputs):
        # The window replaces T; the sequence is kept.
        return inputs[0].shape.replace(time=self.window_size)

    def build_layer(self):
        from nnodely.layers.input import Input

        if self.window_size <= 0:
            raise ValueError(
                f"{self.name}: past + future must be positive, got {self.window_size}."
            )

        predecessor = self.preds[0]
        pred_past = (
            predecessor.past
            if isinstance(predecessor, (Input, SampleWindow))
            else predecessor.shape.time  # type: ignore[union-attr]
        )
        start = pred_past - self.past

        return SampleWindowImpl(
            start=start,
            window_size=self.window_size,
            axis=predecessor.shape.time_axis,  # type: ignore[union-attr]
            name=self.name,
        )


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
    """
    Select one index along a chosen dim axis.

    The selected dim axis is dropped, or with ``keep_dim=True`` kept with
    length 1. A stream always has a dim, so selecting from a vector gives a
    scalar ``(1,)`` either way.

    Examples
    --------
    dim=(4, 3), axis=0 -> dim=(3,)
    dim=(4, 3), axis=1, keep_dim=True -> dim=(4, 1)
    dim=(4,) -> dim=(1,)
    """

    def __init__(self, idx: int, axis: int = 0, keep_dim: bool = False, name=None):
        self.idx = int(idx)
        self.axis = int(axis)
        self.keep_dim = bool(keep_dim)
        super().__init__(
            name=name, idx=self.idx, axis=self.axis, keep_dim=self.keep_dim
        )

    def _selected(self, shape):
        """The dim axis, the index on it, and whether the axis is kept."""
        axis = shape.resolve_dim(self.axis, self.name)
        idx = _index(self.name, self.idx, shape.dim[axis], f"dim axis {axis}")
        return axis, idx, self.keep_dim or shape.dim_rank == 1

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        axis, _, keep = self._selected(shape)
        kept = (1,) if keep else ()
        return shape.replace(dim=shape.dim[:axis] + kept + shape.dim[axis + 1 :])

    def build_layer(self):
        # The input's shape, not this node's: once applied, a Select carries
        # its own output shape, without the selected axis.
        shape = self.preds[0].shape  # type: ignore[union-attr]
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
    """
    Select a contiguous range [start, end) along a chosen dim axis.

    The selected dim axis is resized to end - start.

    Examples
    --------
    dim=(9, 3), axis=0, start=0, end=4 -> dim=(4, 3)
    dim=(9, 3), axis=1, start=1, end=3 -> dim=(9, 2)
    """

    def __init__(self, start: int, end: int, axis: int = 0, name=None):
        self.start = int(start)
        self.end = int(end)
        self.axis = int(axis)
        super().__init__(name=name, start=self.start, end=self.end, axis=self.axis)

    def build_layer(self):
        if self.preds is None or len(self.preds) == 0:
            raise ValueError(f"{self.name}: no input layer to select range from.")
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = shape.resolve_dim(self.axis, self.name)

        dim_size = shape.dim[axis]
        start = self.start
        end = self.end
        if start < 0:
            start += dim_size
        if end < 0:
            end += dim_size

        if start < 0 or end > dim_size or start >= end:
            raise ValueError(
                f"{self.name}: invalid range [{self.start}, {self.end}) for "
                f"dim size {dim_size}."
            )

        return RangeImpl(
            start=start,
            end=end,
            axis=shape.dim_axes[axis],
            name=self.name,
        )


class TimeSelect(Layer):
    """
    Select one sample of the window, ``T``, as :class:`Select` does along a
    dim axis; the window is kept, one sample long. A negative ``idx`` counts
    from the end, so -1 is the newest sample.

    Examples
    --------
    T=5, idx=-1 -> T=1
    S=(3,), T=5, idx=0 -> S=(3,), T=1
    """

    def __init__(self, idx: int, name=None):
        self.idx = int(idx)
        super().__init__(name=name, idx=self.idx)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        _index(self.name, self.idx, shape.time, "the window")
        return shape.replace(time=1)

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        return SelectImpl(
            idx=_index(self.name, self.idx, shape.time, "the window"),
            axis=shape.time_axis,
            keep_dim=True,
            name=self.name,
        )


class TimeRange(Layer):
    """
    Select a contiguous range [start, end) of the window, ``T``, as
    :class:`Range` does along a dim axis: the window is kept, resized.
    ``None`` leaves a bound open, and negative bounds count from the end.

    Examples
    --------
    T=10, start=2, end=5 -> T=3
    S=(4,), T=6, start=1, end=3 -> S=(4,), T=2
    T=6, start=-2, end=None -> T=2
    """

    def __init__(self, start: int | None, end: int | None, name=None):
        self.start = None if start is None else int(start)
        self.end = None if end is None else int(end)
        super().__init__(name=name, start=self.start, end=self.end)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        return shape.replace(
            time=_range_length(self.name, self.start, self.end, shape.time)
        )

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        return RangeImpl(
            start=self.start, end=self.end, axis=shape.time_axis, name=self.name
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
    """
    Concatenate two or more streams along the window, ``T``, as
    :class:`Concatenate` does along a dim axis. All inputs must share the
    sequence and the dim.

    Examples
    --------
    T=2 + T=3 -> T=5
    S=(4,), T=2 + S=(4,), T=3 -> S=(4,), T=5
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        self.inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        shape = inputs[0].shape
        for input_node in inputs[1:]:
            other = input_node.shape
            if other.dim != shape.dim or other.seq != shape.seq:
                raise ValueError(
                    f"{self.name}: all inputs must have the same sequence and dim, "
                    f"got {shape} and {other}."
                )
        return shape.replace(time=sum(node.shape.time for node in inputs))

    def build_layer(self):
        shape = _concatenation_inputs(self)[0].shape
        return TimeConcatenateImpl(
            axis=shape.time_axis, input_rank=shape.rank, name=self.name
        )


class TimeReverse(Layer):
    """
    Reverse the order of the samples of the window, ``T``: newest first.

    Examples
    --------
    T=5 -> T=5, its samples newest first
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        return inputs[0].shape

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        return TimeReverseImpl(axis=shape.time_axis, name=self.name)


class SeqSelect(Layer):
    """
    Select one step of a sequence axis, which is removed. ``axis`` counts
    the sequence axes outermost first: the default, 0, is the outermost one,
    the rollout of the outermost Loop. A negative ``idx`` counts from the
    end, also on a dynamic axis, so -1 is the last step of a rollout of any
    length.

    Examples
    --------
    S=(5,), T=1, idx=-1 -> S=(), T=1
    S=(dyn, 3), T=1, idx=-1 -> S=(3,), T=1
    S=(3, 5), T=1, idx=0, axis=1 -> S=(3,), T=1
    """

    def __init__(self, idx: int, axis: int = 0, name=None):
        self.idx = int(idx)
        self.axis = int(axis)
        super().__init__(name=name, idx=self.idx, axis=self.axis)

    def _selected(self, shape):
        """The sequence axis and the index on it."""
        axis = shape.resolve_seq(self.axis, self.name)
        return axis, _index(self.name, self.idx, shape.seq[axis], f"seq axis {axis}")

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        axis, _ = self._selected(shape)
        return shape.replace(seq=shape.seq[:axis] + shape.seq[axis + 1 :])

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis, idx = self._selected(shape)
        return SelectImpl(
            idx=idx, axis=shape.seq_axes[axis], keep_dim=False, name=self.name
        )


class SeqRange(Layer):
    """
    Select a contiguous range [start, end) of a sequence axis, counted
    outermost first as in :class:`SeqSelect`. The axis is resized, and
    removed when the range is one step long. ``None`` leaves a bound open,
    and negative bounds count from the end. On a dynamic axis a negative
    bound or an open end gives a dynamic axis.

    Examples
    --------
    S=(10,), T=1, start=2, end=5 -> S=(3,), T=1
    S=(dyn,), T=1, start=0, end=4 -> S=(4,), T=1
    S=(dyn,), T=1, start=-3, end=None -> S=(dyn,), T=1
    S=(6, 4), T=1, start=1, end=2, axis=1 -> S=(6,), T=1
    """

    def __init__(self, start: int | None, end: int | None, axis: int = 0, name=None):
        self.start = None if start is None else int(start)
        self.end = None if end is None else int(end)
        self.axis = int(axis)
        super().__init__(name=name, start=self.start, end=self.end, axis=self.axis)

    def _range(self, shape):
        """The sequence axis and the length of the range on it."""
        axis = shape.resolve_seq(self.axis, self.name)
        return axis, _range_length(self.name, self.start, self.end, shape.seq[axis])

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        axis, length = self._range(shape)
        kept = () if length == 1 else (length,)
        return shape.replace(seq=shape.seq[:axis] + kept + shape.seq[axis + 1 :])

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis, length = self._range(shape)
        if length == 1:
            # One step: selected, and its axis removed.
            return SelectImpl(
                idx=0 if self.start is None else self.start,
                axis=shape.seq_axes[axis],
                keep_dim=False,
                name=self.name,
            )
        return RangeImpl(
            start=self.start, end=self.end, axis=shape.seq_axes[axis], name=self.name
        )


class SeqConcatenate(Layer):
    """
    Concatenate two or more streams along a sequence axis, counted
    outermost first as in :class:`SeqSelect`. All inputs must share the dim,
    the window and every other sequence axis. A dynamic input makes the
    joined axis dynamic.

    Examples
    --------
    S=(2,), T=1 + S=(3,), T=1 -> S=(5,), T=1
    S=(3, 4), T=2 + S=(3, 1), T=2, axis=1 -> S=(3, 5), T=2
    S=(3,), T=1 + S=(dyn,), T=1 -> S=(dyn,), T=1
    """

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def output_shape(self, *inputs):
        self.inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        shape = inputs[0].shape
        axis = shape.resolve_seq(self.axis, self.name)

        def others(seq):
            return seq[:axis] + seq[axis + 1 :]

        for input_node in inputs[1:]:
            other = input_node.shape
            if (
                other.dim != shape.dim
                or other.time != shape.time
                or other.seq_rank != shape.seq_rank
                or others(other.seq) != others(shape.seq)
            ):
                raise ValueError(
                    f"{self.name}: all inputs must have the same dim, window and "
                    f"sequence axes but seq axis {self.axis}, got {shape} and {other}."
                )
        sizes = [input_node.shape.seq[axis] for input_node in inputs]
        joined = None if None in sizes else sum(int(size) for size in sizes)  # type: ignore[arg-type]
        return shape.replace(seq=shape.seq[:axis] + (joined,) + shape.seq[axis + 1 :])

    def build_layer(self):
        shape = _concatenation_inputs(self)[0].shape
        axis = shape.resolve_seq(self.axis, self.name)
        return TimeConcatenateImpl(
            axis=shape.seq_axes[axis], input_rank=shape.rank, name=self.name
        )


class SeqStack(Layer):
    """
    Stack two or more streams of one shape into a new sequence axis.

    ``axis`` is the position of the new axis among the sequence axes of the
    result, counted outermost first: the default, 0, makes it the new
    outermost one, as a Loop's rollout, and -1 the new innermost one.

    Examples
    --------
    4 x S=(), T=1 -> S=(4,), T=1
    2 x S=(5,), T=3 -> S=(2, 5), T=3
    2 x S=(5,), T=3, axis=-1 -> S=(5, 2), T=3
    """

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def _stacked(self, inputs):
        """The shape of the stack, and the position of its new axis."""
        shape = inputs[0].shape
        axis = resolve_axis(self.axis, shape.seq_rank + 1, self.name, "seq")
        seq = shape.seq[:axis] + (len(inputs),) + shape.seq[axis:]
        return shape.replace(seq=seq), axis

    def output_shape(self, *inputs):
        self.inputs = list(inputs)
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
    """
    Reverse the order of the steps of a sequence axis, counted outermost
    first as in :class:`SeqSelect`.

    Examples
    --------
    S=(dyn,), T=1 -> S=(dyn,), T=1, its steps last first
    """

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        shape.resolve_seq(self.axis, self.name)
        return shape

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = shape.resolve_seq(self.axis, self.name)
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
        # added to a single sample, they repeat it along its time.
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
    """Concatenate two or more streams along a selected dim axis.

    The streams must have the same sequence and window; a single sample - a
    Parameter, a number, an Input read directly - is repeated along the
    others', as arithmetic broadcasts it.
    """

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        index = next(
            (i for i, node in enumerate(inputs) if not _is_scalar(node.shape._steps)),
            0,
        )
        reference = inputs[index].shape
        axis = reference.resolve_dim(self.axis, self.name)

        for input_node in inputs:
            shape = input_node.shape
            compatible_dim = shape.dim_rank == reference.dim_rank and all(
                size == reference.dim[index]
                for index, size in enumerate(shape.dim)
                if index != axis
            )
            if not compatible_dim or not (
                shape._steps == reference._steps or _is_scalar(shape._steps)
            ):
                raise ValueError(
                    f"{self.name}: all inputs must have matching dimensions "
                    f"except dim axis {self.axis}, and the same sequence and "
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
    inputs = list(layer.inputs or layer.preds)
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
            "All tensors must be laid out (batch, *seq, time, *dim), of rank "
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
