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

    def __init__(
        self, name: str | None = None, predecessors: list[Stream] | None = None, seq: int | tuple[int, ...] | None = None, time: int | None = None, dim: int | tuple[int, ...] | None = None, **kwargs
    ):
        self._properties = dict(kwargs)

        super().__init__(
            name=next_name(self.__class__.__name__) if name is None else name,
            seq=seq,
            time=time,
            dim=dim,
            predecessors=predecessors,
        )

        self._layer = keras.layers.Identity(name=self.name)
        self._properties = dict(kwargs)

    def output_shape(self, *inputs: Stream) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
        """
        Default: propagate shape of the first predecessor.
        By default all inputs must have the same shape, but this can be overridden in subclasses.
        """
        return inputs[0].dimensions

    @abstractmethod
    def build_layer(self) -> keras.Layer:
        """Build the layer."""
        ...

    def call(self, *args):
        self._layer = self.build_layer()
        if len(args) == 1:
            return self._layer(args[0])
        return self._layer(list(args))

    def __call__(self, *inputs: Stream | dict[str, Stream]) -> Stream:
        from typing import cast as type_cast
        # Symbolic mode: Layer(Stream, ...) -> new Stream node
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            out_seq, out_time, out_dim = self.output_shape(*list(inputs[0].values()))
        else:
            out_seq, out_time, out_dim = self.output_shape(*[type_cast(Stream, x) for x in inputs])
        node = self.__class__(**self._properties)
        node.seq = out_seq
        node.time = out_time
        node.dim = out_dim
        node.predecessors = list(inputs)
        return node

        # Tensor mode: Layer(tensor, ...) -> KerasTensor / Tensor
        # Never called
        #return self.call(*inputs)


class BinaryOp(Layer):
    keras_op = None

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

    def build_layer(self) -> keras.Layer:
        return keras.layers.Lambda(lambda xs: xs[0] / xs[1], name=self.name)
