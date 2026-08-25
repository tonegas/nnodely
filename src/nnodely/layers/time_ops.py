from nnodely.core.layer import Layer

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
        output_shape_no_batch=None,
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
