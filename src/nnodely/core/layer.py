"""
Layer - Symbolic DAG node that can lazily build its concrete Keras layer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence

import keras

from nnodely.core.dag import next_name
from nnodely.core.stream import Stream
from nnodely.core.types import Shape


class Layer(Stream, ABC):
    """Symbolic DAG node that can lazily build its concrete Keras layer.

    :param node_type: Semantic node type.

    .. note::
        - If called with Stream(s): returns a new symbolic node
        - If called with tensor(s): applies the built Keras layer
    """

    node_type: str = "Layer"

    _layer: keras.Layer
    _properties: dict

    def __init__(
        self,
        name: str | None = None,
        predecessors: list[Stream] | None = None,
        seq: Shape = (),
        time: int = 1,
        dim: Shape = (1,),
        **kwargs,
    ) -> None:
        super().__init__(
            name=name or next_name(self.node_type),
            node_type=self.node_type,
            seq=seq,
            time=time,
            dim=dim,
            predecessors=predecessors or [],
        )

        self._layer = keras.layers.Identity(name=self.name)
        self._properties = dict(kwargs)

    def output_shape(
        self, seqs: Sequence[Shape], times: Sequence[int], dims: Sequence[Shape]
    ) -> tuple[Shape, int, Shape]:
        """Propagate the shape of the first predecessor

        :param seqs: Sequences of predecessors.
        :param times: Times of predecessors.
        :param dims: Dimensions of predecessors.
        :return: The shape of the first predecessor
        """

        return seqs[0], times[0], dims[0]

    @abstractmethod
    def build_layer(self) -> keras.Layer:
        """Build the layer."""
        ...

    def call(self, *args):
        self._layer = self.build_layer()
        if len(args) == 1:
            return self._layer(args[0])
        return self._layer(list(args))

    def __call__(self, *inputs):
        if not inputs:
            raise TypeError(f"{self.__class__.__name__} expects at least one input")

        # Symbolic mode: Layer(Stream, ...) -> new Stream node
        if all(isinstance(x, Stream) for x in inputs):
            seqs, times, dims = zip(*(x.dimensions for x in inputs))
            out_seq, out_time, out_dim = self.output_shape(seqs, times, dims)

            node = self.__class__(**self._properties)  # HACK: needs further inspection
            node.seq = out_seq
            node.time = out_time
            node.dim = out_dim
            node.predecessors = list(inputs)
            return node

        # Tensor mode: Layer(tensor, ...) -> KerasTensor / Tensor
        return self.call(*inputs)


class BinaryOp(Layer):
    def output_shape(
        self, seqs: Sequence[Shape], times: Sequence[int], dims: Sequence[Shape]
    ) -> tuple[Shape, int, Shape]:
        ref = self.predecessors[0] if self.predecessors else None
        if ref is not None:
            for p in self.predecessors[1:]:
                if ref.shape != p.shape:
                    raise ValueError(
                        f"{self.node_type} requires identical shapes, got {ref.shape} and {p.shape}"
                    )
        return seqs[0], times[0], dims[0]


class Add(BinaryOp):
    node_type = "Add"

    def build_layer(self) -> keras.Layer:
        return keras.layers.Add(name=self.name)


class Subtract(BinaryOp):
    node_type = "Subtract"

    def build_layer(self) -> keras.Layer:
        return keras.layers.Subtract(name=self.name)


class Multiply(BinaryOp):
    node_type = "Multiply"

    def build_layer(self) -> keras.Layer:
        return keras.layers.Multiply(name=self.name)


class Divide(BinaryOp):
    node_type = "Divide"

    def build_layer(self) -> keras.Layer:
        return keras.layers.Lambda(lambda xs: xs[0] / xs[1], name=self.name)
