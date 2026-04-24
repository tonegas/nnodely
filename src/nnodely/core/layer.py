from __future__ import annotations

import keras


from nnodely.core.dag import next_name
from nnodely.core.stream import Stream


def _is_stream(x):
    return isinstance(x, Stream)


class Layer(Stream):
    """
    Symbolic DAG node that can lazily build its concrete Keras layer.

    - If called with Stream(s): returns a new symbolic node
    - If called with tensor(s): applies the built Keras layer
    """

    def __init__(
        self, name=None, predecessors=None, seq=None, time=None, dim=None, **kwargs
    ):
        self._properties = dict(kwargs)

        super().__init__(
            name=next_name(self.__class__.__name__) if name is None else name,
            seq=seq,
            time=time,
            dim=dim,
            predecessors=predecessors,
        )

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------

    def output_shape(self, seqs, times, dims):
        """
        Default: propagate shape of the first predecessor.
        seqs, times, dims are tuples/lists, one entry per predecessor.
        """
        return seqs, times, dims

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------

    def build_layer(self):
        """
        Override in subclasses.
        Must set self._layer and return it.
        """
        return keras.layers.Identity(name=self.name)

    def call(self, *xs):
        layer = self.build_layer()
        if len(xs) == 1:
            return layer(xs[0])
        return layer(list(xs))

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------

    def __call__(self, *inputs):
        # Symbolic mode: Layer(Stream, ...) -> new Stream node
        if all(_is_stream(x) for x in inputs):
            out_seq, out_time, out_dim = self.output_shape(
                inputs[0].seq, inputs[0].time, inputs[0].dim
            )
            node = self.__class__(**self._properties)
            node.seq = out_seq
            node.time = out_time
            node.dim = out_dim
            node.predecessors = list(inputs)
            return node

        # Tensor mode: Layer(tensor, ...) -> KerasTensor / Tensor
        return self.call(*inputs)


class BinaryOp(Layer):
    keras_op = None

    def output_shape(self, seqs, times, dims):
        ref = self.predecessors[0] if self.predecessors else None
        if ref is not None:
            for p in self.predecessors[1:]:
                if ref.shape != p.shape:
                    raise ValueError(
                        f"{self.__class__.__name__} requires identical shapes, got {ref.shape} and {p.shape}"
                    )
        return seqs, times, dims

    def build_layer(self):
        if self.keras_op is not None:
            return self.keras_op(name=self.name)
        else:
            return super().build_layer()


class Add(BinaryOp):
    keras_op = keras.layers.Add


class Subtract(BinaryOp):
    keras_op = keras.layers.Subtract


class Multiply(BinaryOp):
    keras_op = keras.layers.Multiply


class Divide(BinaryOp):
    keras_op = keras.layers.Lambda

    def build_layer(self):
        return keras.layers.Lambda(lambda xs: xs[0] / xs[1], name=self.name)
