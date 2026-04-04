"""Stream-aware layers."""

from __future__ import annotations

import keras

import copy

from nnodely.core.dag import get_seq_time_dim, next_name, same_shape
from nnodely.core.stream import Stream


def _is_stream(x):
    return isinstance(x, Stream)
    

class Layer(Stream):
    """
    Symbolic DAG node that can lazily build its concrete Keras layer.

    - If called with Stream(s): returns a new symbolic node
    - If called with tensor(s): applies the built Keras layer
    """
    node_type = "Layer"

    def __init__(self, name=None, predecessors=None, seq=(), time=1, dim=(1,), **kwargs):
        self._layer = None
        self._properties = dict(kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        super().__init__(
            name=name or next_name(self.node_type),
            node_type=self.node_type,
            seq=seq,
            time=time,
            dim=dim,
            predecessors=predecessors or [],
        )

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------

    def output_shape(self, seqs, times, dims):
        """
        Default: propagate shape of the first predecessor.
        seqs, times, dims are tuples/lists, one entry per predecessor.
        """
        return seqs[0], times[0], dims[0]

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------

    def build_layer(self):
        """
        Override in subclasses.
        Must set self._layer and return it.
        """
        self._layer = keras.layers.Identity(name=self.name)
        return self._layer

    def ensure_layer(self):
        if self._layer is None:
            self._layer = self.build_layer()
        return self._layer

    def call(self, *xs):
        layer = self.ensure_layer()
        if len(xs) == 1:
            return layer(xs[0])
        return layer(list(xs))

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------

    def _clone_for_graph(self):
        """
        Create a fresh symbolic node with the same configuration,
        but without reusing the concrete built Keras layer.
        """
        cfg = copy.copy(self._properties)
        new_node = self.__class__(**cfg)
        return new_node

    def __call__(self, *inputs):
        if not inputs:
            raise TypeError(f"{self.__class__.__name__} expects at least one input")

        # Symbolic mode: Layer(Stream, ...) -> new Stream node
        if all(_is_stream(x) for x in inputs):
            seqs, times, dims = zip(*(get_seq_time_dim(x) for x in inputs))
            out_seq, out_time, out_dim = self.output_shape(seqs, times, dims)

            node = self._clone_for_graph()
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
                if not same_shape(ref, p):
                    raise ValueError(
                        f"{self.node_type} requires identical shapes, got {ref.shape} and {p.shape}"
                    )
        return seqs[0], times[0], dims[0]

    def build_layer(self):
        return self.keras_op(name=self.name)


class Add(BinaryOp):
    node_type = "Add"
    keras_op = keras.layers.Add


class Subtract(BinaryOp):
    node_type = "Subtract"
    keras_op = keras.layers.Subtract


class Multiply(BinaryOp):
    node_type = "Multiply"
    keras_op = keras.layers.Multiply


class Divide(BinaryOp):
    node_type = "Divide"
    keras_op = keras.layers.Lambda

    def build_layer(self):
        return keras.layers.Lambda(lambda xs: xs[0] / xs[1], name=self.name)
