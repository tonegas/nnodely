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
    def output_shape(self, *inputs):
        """Infer the output shape by executing the concrete layer on dummy inputs.

        Layers with value-dependent output shapes or layers that change the semantic
        sequence rank can override this method.
        """
        if not inputs:
            raise ValueError(
                f"{self.name}: automatic shape inference requires at least one input. "
                "Layers without inputs must implement output_shape()."
            )

        zero_input_layers = [
            isinstance(input_node, Layer) and len(input_node.preds) == 0
            for input_node in inputs
        ]
        dummy_inputs = [
            keras.ops.zeros(
                input_node.shape.tuple
                if is_zero_input_layer
                else (1, *input_node.shape.tuple)
            )
            for input_node, is_zero_input_layer in zip(inputs, zero_input_layers)
        ]

        # Some build_layer() implementations need the symbolic input context to
        # choose axes or configure the concrete Keras layer.
        self.inputs = list(inputs)
        self.preds = list(inputs)
        self.shape = inputs[0].shape

        # Shape probing must not become the executable graph layer. Graph context
        # (for example an Input's final window) may still change while the DAG is
        # being declared, and trainable weights should only be created at build().
        inference_layer = self.build_layer()

        try:
            outputs = inference_layer(
                dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs
            )
        except Exception as exc:
            raise ValueError(
                f"{self.name}: automatic output shape inference failed. "
                "Implement output_shape() for layers whose shape cannot be inferred "
                "from dummy tensors."
            ) from exc

        multiple_outputs = isinstance(outputs, (list, tuple))
        output_values = list(outputs) if multiple_outputs else [outputs]
        inferred = [
            self._shape_from_tensor(
                output,
                max(input_node.shape.seq_rank for input_node in inputs),
                has_batch=not all(zero_input_layers),
            )
            for output in output_values
        ]
        return inferred if multiple_outputs else inferred[0]

    def _shape_from_tensor(self, tensor, seq_rank: int, *, has_batch: bool):
        shape = tuple(tensor.shape)
        if has_batch:
            shape = shape[1:]
        # A layer may consume sequence axes (for example a recurrent body
        # relation reduced to one step), so preserve only those still present.
        seq_rank = min(seq_rank, max(0, len(shape) - 2))
        minimum_rank = seq_rank + 2  # At least one dim axis and one time axis.
        if len(shape) < minimum_rank:
            raise ValueError(
                f"{self.name}: inferred tensor shape {shape} cannot be represented "
                f"with {seq_rank} sequence dimensions. Implement output_shape() "
                "to describe this layer's semantic axes."
            )
        if any(axis is None for axis in shape):
            raise ValueError(
                f"{self.name}: inferred tensor shape {shape} contains dynamic axes. "
                "Implement output_shape() explicitly for dynamic output shapes."
            )

        time_index = len(shape) - seq_rank - 1
        dim = tuple(int(axis) for axis in shape[:time_index])
        time = int(shape[time_index])
        seq = tuple(int(axis) for axis in shape[time_index + 1 :])
        return dim, time, seq

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
    operation = None

    def build_layer(self) -> keras.layers.Layer:
        if self.operation is None:
            raise NotImplementedError(
                "Subclasses must define an operation class attribute."
            )
        input_has_batch = tuple(
            not (isinstance(node, Layer) and len(node.preds) == 0)
            for node in (self.inputs or self.preds)
        )
        return BinaryOpImpl(
            operation=self.operation,
            input_has_batch=input_has_batch,
            name=self.name,
        )


@keras.saving.register_keras_serializable(package="nnodely")
class BinaryOpImpl(keras.layers.Layer):
    def __init__(
        self,
        operation: str,
        input_has_batch: tuple[bool, ...] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.operation = operation
        self.input_has_batch = input_has_batch

    def call(self, xs):
        has_batch = self.input_has_batch or tuple(True for _ in xs)
        graph_has_batch = any(has_batch)
        target_rank = max(
            len(value.shape) + int(graph_has_batch and not value_has_batch)
            for value, value_has_batch in zip(xs, has_batch)
        )
        values = list(xs)
        for index, (value, value_has_batch) in enumerate(zip(values, has_batch)):
            if graph_has_batch and not value_has_batch:
                value = keras.ops.expand_dims(value, axis=0)
            while len(value.shape) < target_rank:
                value = keras.ops.expand_dims(value, axis=-1)
            values[index] = value

        operations = {
            "add": keras.ops.add,
            "subtract": keras.ops.subtract,
            "multiply": keras.ops.multiply,
        }
        operation = operations[self.operation]
        result = values[0]
        for value in values[1:]:
            result = operation(result, value)
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "operation": self.operation,
                "input_has_batch": self.input_has_batch,
            }
        )
        return config


class Add(BinaryOp):
    operation = "add"


class Subtract(BinaryOp):
    operation = "subtract"


class Multiply(BinaryOp):
    operation = "multiply"


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
