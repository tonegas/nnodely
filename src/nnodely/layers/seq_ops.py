"""Operations on the sequence axes: the counterpart of the dim and time ones.

Every operation acts on one sequence axis, counted among the ``seq`` axes only;
``axis=-1`` is the last one, the rollout axis appended by the innermost Loop.
A sequence axis may be dynamic (``None``), its length known only at run time.
"""

from nnodely.core.layer import Layer
from nnodely.layers.time_ops import _concatenation_inputs, _concatenation_values

import keras


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
