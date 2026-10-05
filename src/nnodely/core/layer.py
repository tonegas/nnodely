from __future__ import annotations
from abc import abstractmethod

import keras
from typing import Any

from nnodely.core.stream import Stream, Shape
from nnodely.core.dag import next_name


def _claim_keras_name(layer, taken: dict[str, Any]) -> None:
    """Give ``layer`` a name no other layer of the Keras model being built has.

    Two nnodely layers may share a name - a model and its reloaded copy, or
    layers named alike by hand - while the layers of one Keras model may not:
    a layer arriving at a name already taken takes the first free suffix. A
    layer applied several times keeps the one name it has.
    """
    if layer.name in taken and taken[layer.name] is not layer:
        index = 1
        while f"{layer.name}_{index}" in taken:
            index += 1
        layer.name = f"{layer.name}_{index}"
    taken[layer.name] = layer


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
        self._layer_signature = None
        # The layer this node applies. Every node made by calling one layer
        # object is an application of it, and they all share its weights;
        # layers created apart never do, whatever their names.
        self._source = self

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

        dummy_inputs = [
            keras.ops.zeros((1, *input_node.shape.tuple)) for input_node in inputs
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
                output, max(input_node.shape.seq_rank for input_node in inputs)
            )
            for output in output_values
        ]
        return inferred if multiple_outputs else inferred[0]

    def _shape_from_tensor(self, tensor, seq_rank: int):
        shape = tuple(tensor.shape)[1:]
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

    def input_signature(self, xs):
        """The input shapes a concrete layer was built for.

        Weights belong to the shape they were made for, so a layer carried
        into another graph - a model used as a block brings its own along -
        can only be reused where its inputs still look the same. Layers that
        read no input at all (Parameters, Constants) are shape-independent and
        keep a constant signature.
        """
        if not self.preds:
            return ()
        try:
            return tuple(tuple(x.shape[1:]) for x in xs)
        except (AttributeError, TypeError):
            # A layer whose inputs are not plain tensors - a recurrent body
            # takes structures - cannot be checked this way, and is reused as
            # it always was.
            return None

    def call(self, xs):
        signature = self.input_signature(xs)
        stale = self._layer_signature is not None and self._layer_signature != signature
        if self._layer is None or stale:
            self._layer = self.build_layer()
        self._layer_signature = signature
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
                node._source = self._source
                node.inputs = inputs
                node.shape = Shape(dim=out_dim, time=out_time, seq=out_seq)
                node.preds = inputs  # type: ignore
                ret.append(node)
            return ret if len(ret) > 1 else ret[0]

        # Tensor mode: Layer(tensor, ...) -> KerasTensor / Tensor
        ret = self.call(*inputs)
        return ret if len(ret) > 1 else ret[0]

    def get_config(self):
        # The arguments the layer was created with, which rebuild it; its shape
        # follows from the streams it is called on again.
        return {"name": self.name, **self._properties}

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
        return BinaryOpImpl(operation=self.operation, name=self.name)

    def output_shape(self, *inputs):
            # A dynamic sequence axis cannot be probed with a dummy tensor, but a
            # product of streams of one shape keeps that shape. A dynamic axis
            # matches the build length of the other side (a Loop rollout declared
            # with `length`), since both follow the data, and stays dynamic.
            shapes = [input_node.shape.tuple for input_node in inputs]
            ranks = {input_node.shape.seq_rank for input_node in inputs}
            if len({len(shape) for shape in shapes}) == 1 and len(ranks) == 1:
                axes = [{axis for axis in column if axis is not None} for column in zip(*shapes)]
                if all(len(sizes) <= 1 for sizes in axes):
                    merged = [None if None in column else sizes.pop() for column, sizes in zip(zip(*shapes), axes)]
                    seq_rank = ranks.pop()
                    time_index = len(merged) - seq_rank - 1
                    return tuple(merged[:time_index]), merged[time_index], tuple(merged[time_index + 1 :])
            return super().output_shape(*inputs)


_BINARY_OPERATIONS = {
    "add": keras.ops.add,
    "subtract": keras.ops.subtract,
    "multiply": keras.ops.multiply,
    "divide": keras.ops.divide,
    "power": keras.ops.power,
}


@keras.saving.register_keras_serializable(package="nnodely")
class BinaryOpImpl(keras.layers.Layer):
    """An elementwise operation between streams laid out
    ``(batch, *dim, time, *seq)``.

    An operand with fewer axes - a number carries no sequence axis - gets its
    missing axes at the end, so it broadcasts along those. Aligned from the
    end instead, as broadcasting does by default, its batch axis would line
    up with the other operand's first dim axis.
    """

    def __init__(self, operation: str, **kwargs):
        super().__init__(**kwargs)
        self.operation = operation

    def call(self, xs):
        rank = max(len(value.shape) for value in xs)
        values = []
        for value in xs:
            while len(value.shape) < rank:
                value = keras.ops.expand_dims(value, axis=-1)
            values.append(value)

        operation = _BINARY_OPERATIONS[self.operation]
        result = values[0]
        for value in values[1:]:
            result = operation(result, value)
        return result

    def get_config(self):
        config = super().get_config()
        config.update({"operation": self.operation})
        return config


class Add(BinaryOp):
    operation = "add"


class Subtract(BinaryOp):
    operation = "subtract"


class Multiply(BinaryOp):
    operation = "multiply"


class Divide(BinaryOp):
    operation = "divide"


class Power(BinaryOp):
    operation = "power"


class Identity(Layer):
    def output_shape(self, *inputs):
        # Every axis is preserved, including a dynamic sequence axis that cannot
        # be probed with a dummy tensor.
        return inputs[0].shape.dimensions

    def build_layer(self):
        return keras.layers.Identity(name=self.name)
