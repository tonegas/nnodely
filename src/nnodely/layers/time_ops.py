from nnodely.core.layer import Layer

import keras


# class SampleWindow(Layer):
#     """
#     Layer che estrae finestra temporale. Se window_size < input.time, applica slice.
#     Simmetrico agli altri layer: usa build_layer e call.
#     """

#     def __init__(self, past: int, future: int, name=None):
#         self.past = int(past)
#         self.future = int(future)
#         self.window_size = self.past + self.future
#         super().__init__(name=name, past=self.past, future=self.future)

#     def output_shape(self, *inputs):
#         inp = inputs[0]
#         return inp.dim, self.past + self.future, inp.seq

#     def build_layer(self):
#         from nnodely.layers.input import Input

#         if self.window_size <= 0:
#             raise ValueError(
#                 f"{self.name}: past + future must be positive, got {self.window_size}."
#             )

#         pred_past = self.preds[0].past if isinstance(self.preds[0], Input) else 0
#         slices = (
#             [slice(None)] * (1 + len(self.dim))
#             + [slice(pred_past - self.past, (pred_past - self.past) + self.window_size)]
#             + [slice(None)] * len(self.seq)
#         )
#         return keras.layers.Lambda(
#             lambda x: x[tuple(slices)],
#             output_shape=self.shape,
#             name=self.name,
#         )


@keras.saving.register_keras_serializable(package="nnodely")
class SampleWindowImpl(keras.layers.Layer):
    def __init__(
        self, start: int, window_size: int, output_shape_no_batch, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.start = int(start)
        self.window_size = int(window_size)
        self.output_shape_no_batch = tuple(output_shape_no_batch)

    def call(self, x):
        # Convention:
        # [batch, dim1, dim2, ..., time, seq1, seq2, ...]
        dim_rank = len(self.output_shape_no_batch) - 1

        # This is valid because time axis is after batch + dim axes.
        time_axis = 1 + dim_rank

        slices = (
            [slice(None)] * time_axis
            + [slice(self.start, self.start + self.window_size)]
            + [slice(None)] * (len(x.shape) - time_axis - 1)
        )

        return x[tuple(slices)]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.output_shape_no_batch

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "start": self.start,
                "window_size": self.window_size,
                "output_shape_no_batch": self.output_shape_no_batch,
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

    def output_shape(self, *inputs):
        inp = inputs[0]
        return inp.dim, self.past + self.future, inp.seq

    def build_layer(self):
        from nnodely.layers.input import Input

        if self.window_size <= 0:
            raise ValueError(
                f"{self.name}: past + future must be positive, got {self.window_size}."
            )

        pred_past = self.preds[0].past if isinstance(self.preds[0], Input) else 0
        start = pred_past - self.past

        return SampleWindowImpl(
            start=start,
            window_size=self.window_size,
            output_shape_no_batch=self.shape,
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
        output_shape_no_batch,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.idx = int(idx)
        self.axis = int(axis)
        self.output_shape_no_batch = tuple(output_shape_no_batch)

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
        return (input_shape[0],) + self.output_shape_no_batch

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "idx": self.idx,
                "axis": self.axis,
                "output_shape_no_batch": self.output_shape_no_batch,
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

    def output_shape(self, *inputs):
        inp = inputs[0]

        if len(inp.dim) < 1:
            raise ValueError(f"{self.name}: Select requires at least one dim axis.")

        dim = list(inp.dim)
        axis = self._resolve_dim_axis(len(dim))

        idx = self.idx
        if idx < 0:
            idx += dim[axis]

        if idx < 0 or idx >= dim[axis]:
            raise ValueError(
                f"{self.name}: idx {self.idx} out of bounds for dim axis {axis} "
                f"of size {dim[axis]}."
            )

        dim[axis] = 1

        return tuple(dim), inp.time, inp.seq

    def build_layer(self):
        axis = self._resolve_dim_axis(len(self.dim))

        idx = self.idx
        if idx < 0:
            idx += self.dim[axis]

        return SelectImpl(
            idx=idx,
            axis=axis,
            output_shape_no_batch=tuple(self.shape),
            name=self.name,
        )
