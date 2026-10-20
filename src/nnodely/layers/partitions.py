"""Partitions of a stream: the windows, selections and slices of its dim
and time axes, and the concatenations that join streams along them.

The dim operations count the dim axes from 0. The time ones act on the inner
time, the window. The sequence ones act on the sequence: the time axes
outside the window, counted outermost first - ``axis=0``, their default, is
the outer time, the rollout of the outermost Loop - one of which may be
dynamic (``None``), its length known only at run time. Each layer resolves
the axis it acts on with the :class:`Shape` of its input, and gives its Keras
layer that tensor axis.
"""

import keras

from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank, resolve_axis, slice_axis
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
        # The window replaces the inner time; the outer time axes are kept.
        shape = inputs[0].shape
        return shape.replace(time=(*shape.time[:-1], self.window_size))

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
            else predecessor.shape.inner_time  # type: ignore[union-attr]
        )
        start = pred_past - self.past

        return SampleWindowImpl(
            start=start,
            window_size=self.window_size,
            axis=predecessor.shape.inner_time_axis,  # type: ignore[union-attr]
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class SelectImpl(keras.layers.Layer):
    """Serializable implementation of Select, on tensor axis ``axis``."""

    def __init__(
        self,
        idx: int,
        axis: int,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.idx = int(idx)
        self.axis = int(axis)

    def call(self, x):
        return slice_axis(x, self.axis, slice(self.idx, self.idx + 1))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = 1
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "idx": self.idx,
                "axis": self.axis,
            }
        )
        return config


class Select(Layer):
    """
    Select one index along a chosen dim axis.

    The selected dim axis is kept with length 1.

    Examples
    --------
    dim=(4, 3), axis=0 -> dim=(1, 3)
    dim=(4, 3), axis=1 -> dim=(4, 1)
    """

    def __init__(self, idx: int, axis: int = 0, name=None):
        self.idx = int(idx)
        self.axis = int(axis)
        super().__init__(name=name, idx=self.idx, axis=self.axis)

    def build_layer(self):
        # The input's shape, not this node's: once applied, a Select carries
        # its own output shape, in which the selected axis is 1 long.
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = shape.resolve_dim(self.axis, self.name)

        dim_size = shape.dim[axis]
        idx = self.idx
        if idx < 0:
            idx += dim_size
        if idx < 0 or idx >= dim_size:
            raise ValueError(
                f"{self.name}: idx {self.idx} out of bounds for dim axis {axis} "
                f"of size {dim_size}."
            )

        return SelectImpl(
            idx=idx,
            axis=shape.dim_axes[axis],
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class RangeImpl(keras.layers.Layer):
    """Serializable implementation of Range, on tensor axis ``axis``."""

    def __init__(
        self,
        start: int,
        end: int,
        axis: int,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.start = int(start)
        self.end = int(end)
        self.axis = int(axis)

    def call(self, x):
        return slice_axis(x, self.axis, slice(self.start, self.end))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.end - self.start
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


@keras.saving.register_keras_serializable(package="nnodely")
class TimeSelectImpl(keras.layers.Layer):
    """Serializable implementation of selection along the time axis, the
    tensor axis ``axis``."""

    def __init__(self, idx: int, axis: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.idx = int(idx)
        self.axis = int(axis)

    def call(self, x):
        return slice_axis(x, self.axis, slice(self.idx, self.idx + 1))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = 1
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"idx": self.idx, "axis": self.axis})
        return config


class TimeSelect(Layer):
    """Select one value along the inner time, keeping the axis with length 1."""

    def __init__(self, idx: int, name=None):
        self.idx = int(idx)
        super().__init__(name=name, idx=self.idx)

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        input_time = int(shape.inner_time)
        idx = self.idx
        if idx < 0:
            idx += input_time
        if idx < 0 or idx >= input_time:
            raise ValueError(
                f"{self.name}: idx {self.idx} out of bounds for time length "
                f"{input_time}."
            )
        return TimeSelectImpl(idx=idx, axis=shape.inner_time_axis, name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class TimeRangeImpl(keras.layers.Layer):
    """Serializable implementation of a contiguous range along the time axis,
    the tensor axis ``axis``."""

    def __init__(self, start: int, end: int, axis: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.start = int(start)
        self.end = int(end)
        self.axis = int(axis)

    def call(self, x):
        return slice_axis(x, self.axis, slice(self.start, self.end))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.end - self.start
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end, "axis": self.axis})
        return config


class TimeRange(Layer):
    """Select a contiguous range [start, end) along the inner time."""

    def __init__(self, start: int, end: int, name=None):
        self.start = int(start)
        self.end = int(end)
        super().__init__(name=name, start=self.start, end=self.end)

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        input_time = int(shape.inner_time)
        start = self.start
        end = self.end
        if start < 0:
            start += input_time
        if end < 0:
            end += input_time
        if start < 0 or end > input_time or start >= end:
            raise ValueError(
                f"{self.name}: invalid range [{self.start}, {self.end}) for "
                f"time length {input_time}."
            )
        return TimeRangeImpl(
            start=start, end=end, axis=shape.inner_time_axis, name=self.name
        )


@keras.saving.register_keras_serializable(package="nnodely")
class TimeConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along the single time axis, the tensor axis
    ``axis``."""

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
    """Concatenate two or more streams along the inner time."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        reference = inputs[0].shape
        for input_node in inputs[1:]:
            if (
                input_node.dim != reference.dim
                or input_node.time[:-1] != reference.time[:-1]
            ):
                raise ValueError(
                    f"{self.name}: all inputs must have matching dim and outer time "
                    "axes when concatenating time."
                )

        return TimeConcatenateImpl(
            axis=reference.inner_time_axis,
            input_rank=reference.rank,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class ConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along a dim axis, the tensor axis ``axis``
    of the inputs of ``input_rank`` axes; an input without time is repeated
    along the time of the others."""

    def __init__(self, axis: int, input_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.input_rank = int(input_rank)

    def call(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) < 2:
            raise ValueError(
                "Concatenate implementations require at least two tensors."
            )
        rank = self.input_rank + 1
        reference = next(value for value in xs if len(value.shape) == rank)
        # Zeros of one feature along the axis, shaped like the timed inputs:
        # added to an input without time, they repeat it along that time.
        zeros = keras.ops.zeros_like(slice_axis(reference, self.axis, slice(0, 1)))
        values = [
            value if len(value.shape) == rank else expand_to_rank(value, rank) + zeros
            for value in xs
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
            }
        )
        return config


class Concatenate(Layer):
    """Concatenate two or more streams along a selected dim axis.

    The streams with time must have the same time axes; a stream without
    time - a Parameter, a number - is repeated along them, as arithmetic
    broadcasts it.
    """

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        reference = next(
            (node.shape for node in inputs if node.shape.time), inputs[0].shape
        )
        axis = reference.resolve_dim(self.axis, self.name)

        for input_node in inputs:
            shape = input_node.shape
            compatible_dim = shape.dim_rank == reference.dim_rank and all(
                size == reference.dim[index]
                for index, size in enumerate(shape.dim)
                if index != axis
            )
            if not compatible_dim or (shape.time and shape.time != reference.time):
                raise ValueError(
                    f"{self.name}: all inputs must have matching dimensions "
                    f"except dim axis {self.axis}, and matching time axes or none."
                )

        return ConcatenateImpl(
            axis=reference.dim_axes[axis],
            input_rank=reference.rank,
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
            "All tensors must be laid out (batch, *time, *dim), of rank "
            f"{input_rank + 1}."
        )
    return list(xs)


def _concatenated_output_shape(input_shapes, axis):
    shapes: list[list[int | None]] = [list(shape) for shape in input_shapes]
    if len(shapes) < 2:
        raise ValueError("Concatenate implementations require at least two tensors.")

    # The axis counts the axes of the longest shape; a shorter one, without
    # time, lacks the leading time axes.
    output_shape: list[int | None] = list(max(shapes, key=len))
    axis_sizes = [shape[axis - (len(output_shape) - len(shape))] for shape in shapes]
    output_shape[axis] = (
        None
        if any(size is None for size in axis_sizes)
        else sum(int(size) for size in axis_sizes)
    )
    return tuple(output_shape)


# ---------------------------------------------------------------------------
# Sequence axes: the time axes outside the window, outermost first
# ---------------------------------------------------------------------------
def _sequence_axis(shape, axis: int, owner: str, extra: int = 0) -> int:
    """Sequence axis ``axis`` of ``shape``, counted outermost first, a
    negative one from the window, checked; ``extra`` more positions for an
    axis to add. It is also the index of that axis in ``shape.time``."""
    return resolve_axis(axis, len(shape.time[:-1]) + extra, owner, "seq")


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
            f"{name}: invalid range [{start}, {end}) for seq size "
            f"{'dyn' if size is None else size}."
        )
    return last - first


@keras.saving.register_keras_serializable(package="nnodely")
class SeqSelectImpl(keras.layers.Layer):
    """Serializable selection of one element along a tensor axis, dropped."""

    def __init__(self, idx: int, axis: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.idx = int(idx)
        self.axis = int(axis)

    def call(self, x):
        # An integer index drops the axis, and a negative one counts from the
        # end of the data, so -1 is the last step of a rollout of any length.
        return slice_axis(x, self.axis, self.idx)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        del output_shape[self.axis]
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"idx": self.idx, "axis": self.axis})
        return config


class SeqSelect(Layer):
    """
    Select one element along a sequence axis, removing the axis.

    Unlike :class:`Select` and :class:`TimeSelect`, which keep their axis with
    length 1, the sequence axis is dropped, so the result can close a Loop
    callback on an input with one sequence axis less. ``axis`` counts the
    sequence axes outermost first: the default, 0, is the outer time. A
    negative ``idx`` counts from the end, also on a dynamic axis.

    Examples
    --------
    time=(dyn, 1), idx=-1 -> time=(1,)
    time=(3, 5, 1), idx=0, axis=1 -> time=(3, 1)
    """

    def __init__(self, idx: int, axis: int = 0, name=None):
        self.idx = int(idx)
        self.axis = int(axis)
        super().__init__(name=name, idx=self.idx, axis=self.axis)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        axis = _sequence_axis(shape, self.axis, self.name)
        size = shape.time[axis]
        if size is not None and not -size <= self.idx < size:
            raise ValueError(
                f"{self.name}: idx {self.idx} out of bounds for seq axis {axis} "
                f"of size {size}."
            )
        return shape.replace(time=shape.time[:axis] + shape.time[axis + 1 :])

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = _sequence_axis(shape, self.axis, self.name)
        return SeqSelectImpl(idx=self.idx, axis=shape.time_axes[axis], name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class SeqRangeImpl(keras.layers.Layer):
    """Serializable range ``[start, end)`` along a tensor axis."""

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
        config.update({"start": self.start, "end": self.end, "axis": self.axis})
        return config


class SeqRange(Layer):
    """
    Select a contiguous range [start, end) along a sequence axis.

    ``None`` leaves a bound open, and negative bounds count from the end. On a
    dynamic axis a negative bound or an open end gives a dynamic axis.
    ``axis`` counts the sequence axes outermost first.

    Examples
    --------
    time=(10, 1), start=2, end=5 -> time=(3, 1)
    time=(dyn, 1), start=0, end=4 -> time=(4, 1)
    time=(dyn, 1), start=-3, end=None -> time=(dyn, 1)
    """

    def __init__(self, start: int | None, end: int | None, axis: int = 0, name=None):
        self.start = None if start is None else int(start)
        self.end = None if end is None else int(end)
        self.axis = int(axis)
        super().__init__(name=name, start=self.start, end=self.end, axis=self.axis)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        axis = _sequence_axis(shape, self.axis, self.name)
        length = _range_length(self.name, self.start, self.end, shape.time[axis])
        return shape.replace(
            time=shape.time[:axis] + (length,) + shape.time[axis + 1 :]
        )

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = _sequence_axis(shape, self.axis, self.name)
        return SeqRangeImpl(
            start=self.start,
            end=self.end,
            axis=shape.time_axes[axis],
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class SeqConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along an existing tensor axis."""

    def __init__(self, axis: int, input_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.input_rank = int(input_rank)

    def call(self, xs):
        values = _concatenation_values(xs, self.input_rank)
        return keras.ops.concatenate(values, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        sizes = [shape[self.axis] for shape in input_shape]
        output_shape[self.axis] = None if None in sizes else sum(sizes)
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "input_rank": self.input_rank})
        return config


class SeqConcatenate(Layer):
    """
    Concatenate two or more streams along an existing sequence axis.

    All inputs must share dim and every other time axis. A dynamic input
    makes the joined axis dynamic. ``axis`` counts the sequence axes
    outermost first.

    Examples
    --------
    time=(3, 1) + time=(5, 1) -> time=(8, 1)
    time=(3, 1) + time=(dyn, 1) -> time=(dyn, 1)
    """

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def output_shape(self, *inputs):
        self.inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        shape = inputs[0].shape
        axis = _sequence_axis(shape, self.axis, self.name)

        def others(time):
            return time[:axis] + time[axis + 1 :]

        for input_node in inputs[1:]:
            other = input_node.shape
            if (
                other.dim != shape.dim
                or other.time_rank != shape.time_rank
                or others(other.time) != others(shape.time)
            ):
                raise ValueError(
                    f"{self.name}: all inputs must have matching dim and time "
                    f"axes except seq axis {self.axis}."
                )
        sizes = [input_node.shape.time[axis] for input_node in inputs]
        joined = None if None in sizes else sum(int(size) for size in sizes)  # type: ignore[arg-type]
        return shape.replace(
            time=shape.time[:axis] + (joined,) + shape.time[axis + 1 :]
        )

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        shape = inputs[0].shape
        axis = _sequence_axis(shape, self.axis, self.name)
        return SeqConcatenateImpl(
            axis=shape.time_axes[axis],
            input_rank=shape.rank,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class SeqStackImpl(keras.layers.Layer):
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


class SeqStack(Layer):
    """
    Stack two or more streams into a new sequence axis.

    All inputs must have the same shape. ``axis`` is the position of the new
    axis among the sequence axes of the result, counted outermost first: the
    default, 0, makes it the new outer time.

    Examples
    --------
    4 x time=(1,) -> time=(4, 1)
    2 x time=(5, 1), axis=1 -> time=(5, 2, 1)
    """

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def _stacked(self, inputs):
        """The shape of the stack, and the position of its new time axis."""
        shape = inputs[0].shape
        axis = _sequence_axis(shape, self.axis, self.name, extra=1)
        time = shape.time[:axis] + (len(inputs),) + shape.time[axis:]
        return shape.replace(time=time), axis

    def output_shape(self, *inputs):
        self.inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        if any(
            input_node.shape.dimensions != inputs[0].shape.dimensions
            for input_node in inputs
        ):
            raise ValueError(
                f"{self.name}: all inputs must have matching dim and time axes "
                "when stacking."
            )
        return self._stacked(inputs)[0]

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        stacked, axis = self._stacked(inputs)
        return SeqStackImpl(
            axis=stacked.time_axes[axis],
            input_rank=inputs[0].shape.rank,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class SeqReverseImpl(keras.layers.Layer):
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


class SeqReverse(Layer):
    """
    Reverse the order of the elements along a sequence axis, counted
    outermost first.

    Examples
    --------
    time=(dyn, 1) -> time=(dyn, 1)
    """

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        _sequence_axis(shape, self.axis, self.name)
        return shape

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = _sequence_axis(shape, self.axis, self.name)
        return SeqReverseImpl(axis=shape.time_axes[axis], name=self.name)
