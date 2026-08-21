from __future__ import annotations
from abc import abstractmethod

import keras
from typing import Any

from nnodely.core.stream import Stream, Shape
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
    @abstractmethod
    def output_shape(self, *inputs):
        """
        Return the output shape or output shapes produced by this layer.

        Expected single-output format:
            (dim, time, seq)

        Expected multi-output format:
            [
                (dim_1, time_1, seq_1),
                (dim_2, time_2, seq_2),
                ...
            ]
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    @abstractmethod
    def build_layer(self):
        """
        Return the concrete Keras layer used during model construction.
        """
        raise NotImplementedError

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
                node = self.__class__(name=self.name, **self._properties)
                node.inputs = inputs
                node.shape = Shape(dim=out_dim, time=out_time, seq=out_seq)
                node.preds = inputs  # type: ignore
                ret.append(node)
            return ret if len(ret) > 1 else ret[0]

        # Tensor mode: Layer(tensor, ...) -> KerasTensor / Tensor
        ret = self.call(*inputs)
        return ret if len(ret) > 1 else ret[0]

    @classmethod
    def from_config(
        cls,
        config: dict,
        preds=None,
    ):
        layer = cls(**config)

        if preds is None or len(preds) == 0:
            return layer

        return layer(preds)


class BinaryOp(Layer):
    keras_op = None

    def output_shape(self, *inputs):
        ref = inputs[0].dimensions
        return ref

    def build_layer(self) -> keras.layers.Layer:
        if self.keras_op is None:
            raise NotImplementedError(
                "Subclasses must define a keras_op class attribute."
            )
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

    def output_shape(self, *inputs):
        return inputs[0].dimensions
