from __future__ import annotations

import keras
from typing import Any

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
        self._layer = None

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
        if self._layer is None:
            self._layer = self.build_layer()
        if len(xs) == 1:
            return self._layer(xs[0])
        return self._layer(xs)

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------

    def __call__(self, inputs) -> Any:
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # Symbolic mode: Layer(Stream, ...) -> new Stream node

        ret = []
        if all(isinstance(x, Stream) for x in inputs):
            out_shapes = self.output_shape(*inputs)
            if not isinstance(out_shapes, list):
                out_shapes = [out_shapes]
            for idx, (out_dim, out_time, out_seq) in enumerate(out_shapes):
                node = self.__class__(name=f"{self.name}_{idx}", **self._properties)
                node.inputs = inputs
                node.seq = out_seq
                node.time = out_time
                node.dim = out_dim
                node.preds = inputs  # type: ignore
                ret.append(node)

            return ret if len(ret) > 1 else ret[0]

        # Tensor mode: Layer(tensor, ...) -> KerasTensor / Tensor
        ret = self.call(*inputs)
        return ret if len(ret) > 1 else ret[0]


class BinaryOp(Layer):
    keras_op = None

    def output_shape(self, *inputs):
        ref = inputs[0].dimensions
        return ref

    def build_layer(self):
        if self.keras_op is None:
            return super().build_layer()
        return self.keras_op(name=self.name)


class Add(BinaryOp):
    keras_op = keras.layers.Add


class Subtract(BinaryOp):
    keras_op = keras.layers.Subtract


class Multiply(BinaryOp):
    keras_op = keras.layers.Multiply


@keras.saving.register_keras_serializable(package="nnodely")
class DivideImpl(keras.layers.Layer):
    def call(self, xs):
        return xs[0] / xs[1]

    def get_config(self):
        return super().get_config()


class Divide(BinaryOp):
    def build_layer(self):
        return DivideImpl(name=self.name)


@keras.saving.register_keras_serializable(package="nnodely")
class PowerImpl(keras.layers.Layer):
    def call(self, xs):
        return keras.ops.power(xs[0], xs[1])

    def get_config(self):
        return super().get_config()


class Power(BinaryOp):
    def build_layer(self):
        return PowerImpl(name=self.name)


class Identity(Layer):
    def build_layer(self):
        return keras.layers.Identity(name=self.name)
