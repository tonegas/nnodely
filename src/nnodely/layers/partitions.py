"""Partitions of a stream: the windows, selections and slices of its dim,
time and sequence axes, and the concatenations that join streams along them.

The stream layout is ``(batch, *dim, time, *seq)``. The dim operations count
the dim axes from 0, the time ones act on the one time axis, and the sequence
ones count the ``seq`` axes only - ``axis=-1`` is the last, the rollout axis
appended by the innermost Loop - one of which may be dynamic (``None``), its
length known only at run time.
"""

import keras

from nnodely.core.layer import Layer
from nnodely.core.stream import Stream


@keras.saving.register_keras_serializable(package="nnodely")
class SampleWindowImpl(keras.layers.Layer):
    def __init__(
        self,
        start: int,
        window_size: int,
        dim_rank: int,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.start = int(start)
        self.window_size = int(window_size)
        self.dim_rank = int(dim_rank)

    def call(self, x):
        # Convention:
        # [batch, dim1, dim2, ..., time, seq1, seq2, ...]
        # This is valid because time axis is after batch + dim axes.
        time_axis = 1 + self.dim_rank

        # A window that covers the whole time axis is the tensor itself. Taking
        # it as a slice anyway costs an op whose own derivative (a scatter into
        # a zero tensor) has no ONNX equivalent, so a differentiated graph would
        # stop exporting for no reason.
        if self.start == 0 and self.window_size == x.shape[time_axis]:
            return x

        slices = (
            [slice(None)] * time_axis
            + [slice(self.start, self.start + self.window_size)]
            + [slice(None)] * (len(x.shape) - time_axis - 1)
        )
        return x[tuple(slices)]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1 + self.dim_rank] = self.window_size
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "start": self.start,
                "window_size": self.window_size,
                "dim_rank": self.dim_rank,
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
        shape = inputs[0].shape
        return shape.dim, self.window_size, shape.seq

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
            dim_rank=len(self.dim),
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class SelectImpl(keras.layers.Layer):
    """
    Serializable implementation of Select.

    Runtime tensor shape:
        [batch, dim1, dim2, ..., time, seq1, seq2, ...]
    """

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
        # Dim axes start immediately after batch.
        keras_axis = 1 + self.axis

        slices = (
            [slice(None)] * keras_axis
            + [slice(self.idx, self.idx + 1)]
            + [slice(None)] * (len(x.shape) - keras_axis - 1)
        )

        return x[tuple(slices)]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1 + self.axis] = 1
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

    Runtime tensor shape:
        [batch, dim1, dim2, ..., time, seq1, seq2, ...]

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

    def _resolve_dim_axis(self, dim_rank: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += dim_rank

        if axis < 0 or axis >= dim_rank:
            raise ValueError(
                f"{self.name}: axis {self.axis} out of bounds for dim rank {dim_rank}."
            )

        return axis

    def build_layer(self):
        # The input's dim, not this node's: once applied, a Select carries its
        # own output shape, in which the selected axis is 1 long.
        input_dim = self.preds[0].dim  # type: ignore
        axis = self._resolve_dim_axis(len(input_dim))

        dim_size = input_dim[axis]
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
            axis=axis,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class RangeImpl(keras.layers.Layer):
    """
    Serializable implementation of Range.

    Runtime tensor shape:
        [batch, dim1, dim2, ..., time, seq1, seq2, ...]
    """

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
        # Dim axes start immediately after batch.
        keras_axis = 1 + self.axis

        slices = (
            [slice(None)] * keras_axis
            + [slice(self.start, self.end)]
            + [slice(None)] * (len(x.shape) - keras_axis - 1)
        )

        return x[tuple(slices)]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1 + self.axis] = self.end - self.start
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

    Runtime tensor shape:
        [batch, dim1, dim2, ..., time, seq1, seq2, ...]

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

    def _resolve_dim_axis(self, dim_rank: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += dim_rank

        if axis < 0 or axis >= dim_rank:
            raise ValueError(
                f"{self.name}: axis {self.axis} out of bounds for dim rank {dim_rank}."
            )

        return axis

    def build_layer(self):
        if self.preds is None or len(self.preds) == 0:
            raise ValueError(f"{self.name}: no input layer to select range from.")
        input_dim = self.preds[0].dim  # type: ignore
        axis = self._resolve_dim_axis(len(input_dim))

        dim_size = input_dim[axis]
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
            axis=axis,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class TimeSelectImpl(keras.layers.Layer):
    """Serializable implementation of selection along the time axis."""

    def __init__(self, idx: int, dim_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.idx = int(idx)
        self.dim_rank = int(dim_rank)

    def call(self, x):
        # Runtime shape: [batch, dim1, ..., time, seq1, ...]
        time_axis = 1 + self.dim_rank
        slices = (
            [slice(None)] * time_axis
            + [slice(self.idx, self.idx + 1)]
            + [slice(None)] * (len(x.shape) - time_axis - 1)
        )
        return x[tuple(slices)]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1 + self.dim_rank] = 1
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"idx": self.idx, "dim_rank": self.dim_rank})
        return config


class TimeSelect(Layer):
    """Select one value along the time axis, keeping the axis with length 1."""

    def __init__(self, idx: int, name=None):
        self.idx = int(idx)
        super().__init__(name=name, idx=self.idx)

    def build_layer(self):
        input_time = getattr(self.preds[0], "time", None)
        if input_time is None:
            raise ValueError(
                f"{self.name}: Input layer does not have a 'time' attribute."
            )
        idx = self.idx
        if idx < 0:
            idx += input_time
        if idx < 0 or idx >= input_time:
            raise ValueError(
                f"{self.name}: idx {self.idx} out of bounds for time length "
                f"{input_time}."
            )
        return TimeSelectImpl(
            idx=idx,
            dim_rank=len(self.dim),
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class TimeRangeImpl(keras.layers.Layer):
    """Serializable implementation of a contiguous range along the time axis."""

    def __init__(self, start: int, end: int, dim_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.start = int(start)
        self.end = int(end)
        self.dim_rank = int(dim_rank)

    def call(self, x):
        # Runtime shape: [batch, dim1, ..., time, seq1, ...]
        time_axis = 1 + self.dim_rank
        slices = (
            [slice(None)] * time_axis
            + [slice(self.start, self.end)]
            + [slice(None)] * (len(x.shape) - time_axis - 1)
        )
        return x[tuple(slices)]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1 + self.dim_rank] = self.end - self.start
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end, "dim_rank": self.dim_rank})
        return config


class TimeRange(Layer):
    """Select a contiguous range [start, end) along the time axis."""

    def __init__(self, start: int, end: int, name=None):
        self.start = int(start)
        self.end = int(end)
        super().__init__(name=name, start=self.start, end=self.end)

    def build_layer(self):
        input_time = getattr(self.preds[0], "time", None)
        if input_time is None:
            raise ValueError(
                f"{self.name}: Input layer does not have a 'time' attribute."
            )
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
            start=start,
            end=end,
            dim_rank=len(self.dim),
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class TimeConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along the single time axis."""

    def __init__(self, dim_rank: int, input_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim_rank = int(dim_rank)
        self.input_rank = int(input_rank)

    def call(self, xs):
        values = _concatenation_values(xs, self.input_rank)
        return keras.ops.concatenate(values, axis=1 + self.dim_rank)

    def compute_output_shape(self, input_shape):
        return _concatenated_output_shape(input_shape, semantic_axis=self.dim_rank)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim_rank": self.dim_rank,
                "input_rank": self.input_rank,
            }
        )
        return config


class TimeConcatenate(Layer):
    """Concatenate two or more streams along the time axis."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        reference = inputs[0].shape
        for input_node in inputs[1:]:
            if input_node.dim != reference.dim or input_node.seq != reference.seq:
                raise ValueError(
                    f"{self.name}: all inputs must have matching dim and seq "
                    "shapes when concatenating time."
                )

        return TimeConcatenateImpl(
            dim_rank=reference.dim_rank,
            input_rank=reference.rank,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class ConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along one semantic dim axis."""

    def __init__(self, axis: int, input_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.input_rank = int(input_rank)

    def call(self, xs):
        values = _concatenation_values(xs, self.input_rank)
        return keras.ops.concatenate(values, axis=1 + self.axis)

    def compute_output_shape(self, input_shape):
        return _concatenated_output_shape(input_shape, semantic_axis=self.axis)

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
    """Concatenate two or more streams along a selected dim axis."""

    def __init__(self, axis: int = 0, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def _resolve_dim_axis(self, dim_rank: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += dim_rank
        if axis < 0 or axis >= dim_rank:
            raise ValueError(
                f"{self.name}: axis {self.axis} out of bounds for dim rank {dim_rank}."
            )
        return axis

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        reference = inputs[0].shape
        axis = self._resolve_dim_axis(reference.dim_rank)

        for input_node in inputs[1:]:
            shape = input_node.shape
            compatible_dim = shape.dim_rank == reference.dim_rank and all(
                size == reference.dim[index]
                for index, size in enumerate(shape.dim)
                if index != axis
            )
            if (
                not compatible_dim
                or shape.time != reference.time
                or shape.seq != reference.seq
            ):
                raise ValueError(
                    f"{self.name}: all inputs must have matching dimensions "
                    f"except dim axis {self.axis}, and matching time and seq shapes."
                )

        return ConcatenateImpl(
            axis=axis,
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
    """The tensors to concatenate, each laid out ``(batch, *dim, time, *seq)``."""
    if not isinstance(xs, (list, tuple)) or len(xs) < 2:
        raise ValueError("Concatenate implementations require at least two tensors.")
    if any(len(value.shape) != input_rank + 1 for value in xs):
        raise ValueError(
            "All tensors must be laid out (batch, *dim, time, *seq), of rank "
            f"{input_rank + 1}."
        )
    return list(xs)


def _concatenated_output_shape(input_shapes, semantic_axis):
    shapes: list[list[int | None]] = [list(shape) for shape in input_shapes]
    if len(shapes) < 2:
        raise ValueError("Concatenate implementations require at least two tensors.")

    axis = 1 + semantic_axis
    output_shape: list[int | None] = list(shapes[0])
    axis_sizes = [shape[axis] for shape in shapes]
    output_shape[axis] = (
        None
        if any(size is None for size in axis_sizes)
        else sum(int(size) for size in axis_sizes)
    )
    return tuple(output_shape)


# ---------------------------------------------------------------------------
# Sequence axes
# ---------------------------------------------------------------------------
def _resolve_seq_axis(name: str, axis: int, seq_rank: int) -> int:
    resolved = axis + seq_rank if axis < 0 else axis
    if resolved < 0 or resolved >= seq_rank:
        raise ValueError(f"{name}: axis {axis} out of bounds for seq rank {seq_rank}.")
    return resolved


def _tensor_axis(dim_rank: int, seq_axis: int) -> int:
    # Runtime shape: [batch, dim1, ..., time, seq1, ...]
    return 2 + dim_rank + seq_axis


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
        slices = [slice(None)] * self.axis + [self.idx]
        return x[tuple(slices)]

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
    callback on an input with one sequence axis less. A negative ``idx`` counts
    from the end, also on a dynamic axis.

    Examples
    --------
    seq=(dyn,), idx=-1 -> seq=()
    seq=(5, 3), idx=0, axis=0 -> seq=(3,)
    """

    def __init__(self, idx: int, axis: int = -1, name=None):
        self.idx = int(idx)
        self.axis = int(axis)
        super().__init__(name=name, idx=self.idx, axis=self.axis)

    def output_shape(self, *inputs):
        dim, time, seq = inputs[0].shape.dimensions
        axis = _resolve_seq_axis(self.name, self.axis, len(seq))
        size = seq[axis]
        if size is not None and not -size <= self.idx < size:
            raise ValueError(
                f"{self.name}: idx {self.idx} out of bounds for seq axis {axis} "
                f"of size {size}."
            )
        return dim, time, seq[:axis] + seq[axis + 1 :]

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = _resolve_seq_axis(self.name, self.axis, shape.seq_rank)
        return SeqSelectImpl(
            idx=self.idx,
            axis=_tensor_axis(shape.dim_rank, axis),
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class SeqRangeImpl(keras.layers.Layer):
    """Serializable range ``[start, end)`` along a tensor axis."""

    def __init__(self, start, end, axis: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.start = None if start is None else int(start)
        self.end = None if end is None else int(end)
        self.axis = int(axis)

    def call(self, x):
        slices = [slice(None)] * self.axis + [slice(self.start, self.end)]
        return x[tuple(slices)]

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

    Examples
    --------
    seq=(10,), start=2, end=5 -> seq=(3,)
    seq=(dyn,), start=0, end=4 -> seq=(4,)
    seq=(dyn,), start=-3, end=None -> seq=(dyn,)
    """

    def __init__(self, start: int | None, end: int | None, axis: int = -1, name=None):
        self.start = None if start is None else int(start)
        self.end = None if end is None else int(end)
        self.axis = int(axis)
        super().__init__(name=name, start=self.start, end=self.end, axis=self.axis)

    def output_shape(self, *inputs):
        dim, time, seq = inputs[0].shape.dimensions
        axis = _resolve_seq_axis(self.name, self.axis, len(seq))
        length = _range_length(self.name, self.start, self.end, seq[axis])
        return dim, time, seq[:axis] + (length,) + seq[axis + 1 :]

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = _resolve_seq_axis(self.name, self.axis, shape.seq_rank)
        return SeqRangeImpl(
            start=self.start,
            end=self.end,
            axis=_tensor_axis(shape.dim_rank, axis),
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

    All inputs must share dim, time and the other sequence axes. A dynamic
    input makes the joined axis dynamic.

    Examples
    --------
    seq=(3,) + seq=(5,) -> seq=(8,)
    seq=(3,) + seq=(dyn,) -> seq=(dyn,)
    """

    def __init__(self, axis: int = -1, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def output_shape(self, *inputs):
        self.inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        dim, time, seq = inputs[0].shape.dimensions
        axis = _resolve_seq_axis(self.name, self.axis, len(seq))
        for input_node in inputs[1:]:
            other = input_node.shape
            if (
                other.dim != dim
                or other.time != time
                or other.seq_rank != len(seq)
                or other.seq[:axis] + other.seq[axis + 1 :]
                != seq[:axis] + seq[axis + 1 :]
            ):
                raise ValueError(
                    f"{self.name}: all inputs must have matching dim, time and seq "
                    f"shapes except seq axis {self.axis}."
                )
        sizes = [input_node.seq[axis] for input_node in inputs]
        joined = None if None in sizes else sum(sizes)
        return dim, time, seq[:axis] + (joined,) + seq[axis + 1 :]

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        shape = inputs[0].shape
        axis = _resolve_seq_axis(self.name, self.axis, shape.seq_rank)
        return SeqConcatenateImpl(
            axis=_tensor_axis(shape.dim_rank, axis),
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
    axis among the sequence axes of the result; ``-1`` appends it last.

    Examples
    --------
    4 x seq=() -> seq=(4,)
    2 x seq=(5,), axis=0 -> seq=(2, 5)
    """

    def __init__(self, axis: int = -1, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def output_shape(self, *inputs):
        self.inputs = list(inputs)
        inputs = _concatenation_inputs(self)
        dim, time, seq = inputs[0].shape.dimensions
        axis = _resolve_seq_axis(self.name, self.axis, len(seq) + 1)
        if any(
            input_node.shape.dimensions != (dim, time, seq) for input_node in inputs
        ):
            raise ValueError(
                f"{self.name}: all inputs must have matching dim, time and seq "
                "shapes when stacking."
            )
        return dim, time, seq[:axis] + (len(inputs),) + seq[axis:]

    def build_layer(self):
        inputs = _concatenation_inputs(self)
        shape = inputs[0].shape
        axis = _resolve_seq_axis(self.name, self.axis, shape.seq_rank + 1)
        return SeqStackImpl(
            axis=_tensor_axis(shape.dim_rank, axis),
            input_rank=shape.rank,
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
    Reverse the order of the elements along a sequence axis.

    Examples
    --------
    seq=(dyn,) -> seq=(dyn,)
    """

    def __init__(self, axis: int = -1, name=None):
        self.axis = int(axis)
        super().__init__(name=name, axis=self.axis)

    def output_shape(self, *inputs):
        shape = inputs[0].shape
        _resolve_seq_axis(self.name, self.axis, shape.seq_rank)
        return shape.dimensions

    def build_layer(self):
        shape = self.preds[0].shape  # type: ignore[union-attr]
        axis = _resolve_seq_axis(self.name, self.axis, shape.seq_rank)
        return SeqReverseImpl(axis=_tensor_axis(shape.dim_rank, axis), name=self.name)
