from __future__ import annotations

import keras

from nnodely.core.stream import Stream
from nnodely.core.dag import next_name


class Layer(Stream):
    """
    Symbolic DAG node that can lazily build its concrete Keras layer.

    - If called with Stream(s): returns a new symbolic node
    - If called with tensor(s): applies the built Keras layer
    """

    def __init__(self, name=None, preds=None, seq=None, time=None, dim=None, **kwargs):
        self._properties = dict(kwargs)
        self.inputs = None

        super().__init__(
            name=next_name(self.__class__.__name__) if name is None else name,
            dim=dim,
            time=time,
            seq=seq,
            preds=preds,
        )

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------

    def output_shape(self, *inputs):
        """
        Default: propagate shape of the first predecessor.
        By default all inputs must have the same shape, but this can be overridden in subclasses.
        """
        return inputs[0].dimensions

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------

    def build_layer(self):
        """
        Override in subclasses.
        """
        return keras.layers.Identity(name=self.name)

    def call(self, xs):
        layer = self.build_layer()
        if len(xs) == 1:
            return layer(xs[0])
        return layer(xs)

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # Symbolic mode: Layer(Stream, ...) -> new Stream node
        if all(isinstance(x, Stream) for x in inputs):
            out_dim, out_time, out_seq = self.output_shape(*inputs)
            node = self.__class__(name=self.name, **self._properties)
            node.inputs = inputs
            node.seq = out_seq
            node.time = out_time
            node.dim = out_dim
            node.preds = list(inputs)
            return node

        # Tensor mode: Layer(tensor, ...) -> KerasTensor / Tensor
        return self.call(*inputs)


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

    def build_layer(self):
        return keras.layers.Lambda(lambda xs: xs[0] / xs[1], name=self.name)
