from nnodely.core.layer import Layer

import keras


class SampleWindow(Layer):
    """
    Layer che estrae finestra temporale. Se window_size < input.time, applica slice.
    Simmetrico agli altri layer: usa build_layer e call.
    """

    def __init__(self, past: int, future: int, name=None):
        self.past = int(past)
        self.future = int(future)
        self.window_size = self.past + self.future
        super().__init__(name=name, past=self.past, future=self.future)

    def output_shape(self, *inputs):
        """Output ha time=window_size."""
        return inputs[0].seq, self.window_size, inputs[0].dim

    def build_layer(self):
        """Crea Lambda per slice se window_size < input.time, altrimenti Identity."""
        seq_t = self.seq
        dim_t = self.dim
        slices = (
            [slice(None)] * (1 + len(seq_t))
            + [slice(self.past, self.past + self.window_size)]
            + [slice(None)] * len(dim_t)
        )
        return keras.layers.Lambda(lambda x: x[tuple(slices)], name=self.name)


class Select(Layer):
    """
    Select one index along a chosen dim axis.

    Runtime tensor shape:
        [batch, *seq, time, dim1, dim2, ...]

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
            raise ValueError(
                f"{self.name}: SampleSelect requires at least one dim axis."
            )

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
        return inp.seq, inp.time, tuple(dim)

    def build_layer(self):
        axis = self._resolve_dim_axis(len(self.dim))

        idx = self.idx
        if idx < 0:
            idx += self.dim[axis]

        # Runtime tensor:
        # [batch, *seq, time, dim1, dim2, ...]
        #
        # First dim axis starts after:
        # batch + seq axes + time axis
        keras_axis = 1 + len(self.seq) + 1 + axis

        slices = (
            [slice(None)] * keras_axis
            + [slice(idx, idx + 1)]
            + [slice(None)] * (len(self.dim) - axis - 1)
        )

        return keras.layers.Lambda(
            lambda x: x[tuple(slices)],
            name=self.name,
        )


# ## TODO: SamplePart, SampleSelect and make it time dependend with past and future
# class SampleWindow(Layer):
#     """
#     Keep the last `window_size` samples along `axis`.
#     Default axis is the time axis.
#     """
#     node_type = "SampleWindow"

#     def __init__(self, window_size: int, axis="time", name=None):
#         self.window_size = int(window_size)
#         self.axis = axis
#         super().__init__(name=name, window_size=self.window_size, axis=self.axis)

#     def output_shape(self, seqs, times, dims):
#         seq, time, dim = self._input_shape_parts(seqs, times, dims)
#         axis_len = self._axis_len(seqs, times, dims)
#         out_len = min(self.window_size, axis_len)
#         return _updated_shape_after_slice(seq, time, dim, self.axis, out_len)

#     def build_layer(self):
#         pred = self.predecessors[0]
#         seq, time, dim = pred.seq, pred.time, pred.dim
#         axis_index = _axis_to_shape_index(self.axis, seq, time, dim)
#         axis_len = (tuple(seq) + (time,) + tuple(dim))[axis_index]
#         n = min(self.window_size, axis_len)

#         if n >= axis_len:
#             self._layer = keras.layers.Identity(name=self.name)
#             return self._layer

#         batch_axis = axis_index + 1

#         def fn(x):
#             slices = [slice(None)] * len(x.shape)
#             slices[batch_axis] = slice(-n, None)
#             return x[tuple(slices)]

#         self._layer = keras.layers.Lambda(fn, name=self.name)
#         return self._layer


# class SamplePart(_SliceBase):
#     """
#     Extract x[..., i:j, ...] along `axis`.
#     Default axis is the time axis.
#     """
#     node_type = "SamplePart"

#     def __init__(self, i: int | None, j: int | None, axis="time", name=None):
#         self.i = i
#         self.j = j
#         super().__init__(axis=axis, name=name, i=i, j=j)

#     def output_shape(self, seqs, times, dims):
#         seq, time, dim = self._input_shape_parts(seqs, times, dims)
#         axis_len = self._axis_len(seqs, times, dims)
#         _, _, out_len = _normalize_slice_bounds(self.i, self.j, axis_len)
#         return _updated_shape_after_slice(seq, time, dim, self.axis, out_len)

#     def build_layer(self):
#         pred = self.predecessors[0]
#         seq, time, dim = pred.seq, pred.time, pred.dim
#         axis_index = _axis_to_shape_index(self.axis, seq, time, dim)
#         axis_len = (tuple(seq) + (time,) + tuple(dim))[axis_index]
#         start, stop, _ = _normalize_slice_bounds(self.i, self.j, axis_len)

#         batch_axis = axis_index + 1

#         def fn(x):
#             slices = [slice(None)] * len(x.shape)
#             slices[batch_axis] = slice(start, stop)
#             return x[tuple(slices)]

#         self._layer = keras.layers.Lambda(fn, name=self.name)
#         return self._layer


# class SampleSelect(_SliceBase):
#     """
#     Select a single index along `axis`, but keep the axis with length 1.
#     Default axis is the time axis.

#     Example:
#         x.select(3)  -> shape becomes length-1 on selected axis
#     """
#     node_type = "SampleSelect"

#     def __init__(self, i: int, axis="time", name=None):
#         self.i = int(i)
#         super().__init__(axis=axis, name=name, i=self.i)

#     def output_shape(self, seqs, times, dims):
#         seq, time, dim = self._input_shape_parts(seqs, times, dims)
#         return _updated_shape_after_slice(seq, time, dim, self.axis, 1)

#     def build_layer(self):
#         pred = self.predecessors[0]
#         seq, time, dim = pred.seq, pred.time, pred.dim
#         axis_index = _axis_to_shape_index(self.axis, seq, time, dim)
#         axis_len = (tuple(seq) + (time,) + tuple(dim))[axis_index]

#         idx = self.i
#         if idx < 0:
#             idx += axis_len
#         if idx < 0 or idx >= axis_len:
#             raise ValueError(
#                 f"{self.name}: index {self.i} out of range for axis length {axis_len}"
#             )

#         batch_axis = axis_index + 1

#         def fn(x):
#             slices = [slice(None)] * len(x.shape)
#             slices[batch_axis] = slice(idx, idx + 1)
#             return x[tuple(slices)]

#         self._layer = keras.layers.Lambda(fn, name=self.name)
#         return self._layer
