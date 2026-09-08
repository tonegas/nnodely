from nnodely.core.layer import Layer
from nnodely.core.stream import Stream

import keras


@keras.saving.register_keras_serializable(package="nnodely")
class SampleWindowImpl(keras.layers.Layer):
    def __init__(
        self,
        start: int,
        window_size: int,
        dim_rank: int | None = None,
        output_shape_no_batch=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.start = int(start)
        self.window_size = int(window_size)

        if dim_rank is None:
            if output_shape_no_batch is None:
                raise ValueError("SampleWindowImpl requires dim_rank.")
            dim_rank = len(tuple(output_shape_no_batch)) - 1
        self.dim_rank = int(dim_rank)

    def call(self, x):
        # Convention:
        # [batch, dim1, dim2, ..., time, seq1, seq2, ...]
        # This is valid because time axis is after batch + dim axes.
        time_axis = 1 + self.dim_rank

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
    Layer che estrae finestra temporale. Se window_size < input.time, applica slice.
    Simmetrico agli altri layer: usa build_layer e call.
    """

    def __init__(self, past: int, future: int, name=None):
        self.past = int(past)
        self.future = int(future)
        self.window_size = self.past + self.future
        super().__init__(
            name=name, time=self.window_size, past=self.past, future=self.future
        )

    def build_layer(self):
        from nnodely.layers.input import Input

        if self.window_size <= 0:
            raise ValueError(
                f"{self.name}: past + future must be positive, got {self.window_size}."
            )

        pred_past = (
            self.preds[0].past
            if isinstance(self.preds[0], (Input, SampleWindow))
            else 0
        )
        start = pred_past - self.past

        return SampleWindowImpl(
            start=start,
            window_size=self.window_size,
            dim_rank=len(self.dim),
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "past": self.past,
            "future": self.future,
        }


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
        axis = self._resolve_dim_axis(len(self.dim))

        idx = self.idx
        if idx < 0:
            idx += self.dim[axis]

        return SelectImpl(
            idx=idx,
            axis=axis,
            name=self.name,
        )

    def get_config(self):
        return {
            "idx": self.idx,
            "axis": self.axis,
        }


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

    def get_config(self):
        return {
            "name": self.name,
            "idx": self.idx,
        }


@keras.saving.register_keras_serializable(package="nnodely")
class TimeConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along the single time axis."""

    def __init__(self, dim_rank: int, input_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim_rank = int(dim_rank)
        self.input_rank = int(input_rank)

    def call(self, xs):
        values, rank_offset = _align_concatenate_inputs(xs, self.input_rank)
        return keras.ops.concatenate(
            values,
            axis=rank_offset + self.dim_rank,
        )

    def compute_output_shape(self, input_shape):
        return _concatenated_output_shape(
            input_shape,
            semantic_axis=self.dim_rank,
            input_rank=self.input_rank,
        )

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

    def get_config(self):
        return {}


@keras.saving.register_keras_serializable(package="nnodely")
class ConcatenateImpl(keras.layers.Layer):
    """Serializable concatenation along one semantic dim axis."""

    def __init__(self, axis: int, input_rank: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.input_rank = int(input_rank)

    def call(self, xs):
        values, rank_offset = _align_concatenate_inputs(xs, self.input_rank)
        return keras.ops.concatenate(values, axis=rank_offset + self.axis)

    def compute_output_shape(self, input_shape):
        return _concatenated_output_shape(
            input_shape,
            semantic_axis=self.axis,
            input_rank=self.input_rank,
        )

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

    def get_config(self):
        return {"axis": self.axis}


def _concatenation_inputs(layer: Layer) -> list[Stream]:
    inputs = list(layer.inputs or layer.preds)
    if len(inputs) < 2:
        raise ValueError(f"{layer.name}: concatenate requires at least two inputs.")
    if not all(isinstance(input_node, Stream) for input_node in inputs):
        raise TypeError(f"{layer.name}: concatenate inputs must be Stream objects.")
    return [input_node for input_node in inputs if isinstance(input_node, Stream)]


def _align_concatenate_inputs(xs, input_rank):
    if not isinstance(xs, (list, tuple)) or len(xs) < 2:
        raise ValueError("Concatenate implementations require at least two tensors.")

    target_rank = max(len(value.shape) for value in xs)
    if target_rank not in (input_rank, input_rank + 1):
        raise ValueError(
            f"Unexpected tensor rank {target_rank}; expected {input_rank} "
            f"or {input_rank + 1}."
        )

    values = []
    for value in xs:
        if len(value.shape) not in (input_rank, target_rank):
            raise ValueError("All tensors must share the same semantic rank.")
        while len(value.shape) < target_rank:
            value = keras.ops.expand_dims(value, axis=0)
        values.append(value)
    return values, target_rank - input_rank


def _concatenated_output_shape(input_shapes, semantic_axis, input_rank):
    shapes: list[list[int | None]] = [list(shape) for shape in input_shapes]
    if len(shapes) < 2:
        raise ValueError("Concatenate implementations require at least two tensors.")

    target_rank = max(len(shape) for shape in shapes)
    shapes = [[1] * (target_rank - len(shape)) + shape for shape in shapes]
    axis = target_rank - input_rank + semantic_axis
    output_shape: list[int | None] = list(shapes[0])
    axis_sizes = [shape[axis] for shape in shapes]
    output_shape[axis] = (
        None
        if any(size is None for size in axis_sizes)
        else sum(int(size) for size in axis_sizes)
    )
    return tuple(output_shape)
