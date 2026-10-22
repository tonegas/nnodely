from __future__ import annotations
import inspect
from abc import abstractmethod

import keras
from typing import Any

from nnodely.core.shape import Shape, align
from nnodely.core.stream import Stream
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


def _are_streams(value) -> bool:
    """A Stream, or a non-empty list or tuple of Streams."""
    if isinstance(value, Stream):
        return True
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(item, Stream) for item in value)
    )


class _LayerType(type):
    """Lets a layer that takes no configuration be applied as it is created.

    A layer whose constructor takes nothing but ``name`` - ``Sin``, ``Exp``,
    ``Sigmoid`` - has nothing to configure, so ``Sin(x)`` is ``Sin()(x)`` and
    ``Sin(x, name="s")`` is ``Sin(name="s")(x)``. Every other layer is
    configured first and then called on its streams.
    """

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parameters = [
            parameter.name
            for parameter in inspect.signature(cls.__init__).parameters.values()
            if parameter.name != "self"
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls._applied_on_creation = parameters == ["name"]
        # Whether the first setting has a default: then a stream given in its
        # place is a layer called before it was configured, ReLU(x).
        first = next(
            (
                parameter
                for parameter in inspect.signature(cls.__init__).parameters.values()
                if parameter.name != "self"
            ),
            None,
        )
        cls._configured_first = (
            first is not None
            and first.default is not inspect.Parameter.empty
            and first.name != "name"
        )

    def __call__(cls, *args, **kwargs):
        if cls._applied_on_creation and args and _are_streams(args[0]):
            return super().__call__(*args[1:], **kwargs)(args[0])
        if cls._configured_first and args and _are_streams(args[0]):
            raise TypeError(
                f"{cls.__name__} has settings, so it is configured first and then "
                f"called on its streams: {cls.__name__}(...)(x), not "
                f"{cls.__name__}(x)."
            )
        return super().__call__(*args, **kwargs)


class Layer(Stream, metaclass=_LayerType):
    """
    Symbolic DAG node that can lazily build its concrete Keras layer.

    - If called with Stream(s): returns a new symbolic node
    - If called with tensor(s): applies the built Keras layer
    """

    # A layer that reads one stream, such as an activation, called on several
    # is applied to each of them, and returns one result per stream.
    _one_stream = False

    def __init__(self, name=None, preds=None, time=None, dim=None, **kwargs):
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
            preds=preds,
        )

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def output_shape(self, *inputs):
        """Infer the output Shape by executing the concrete layer on dummy inputs.

        Layers with value-dependent output shapes or layers that change the
        number of time axes override this method, and return a Shape, or a
        list of them for several outputs.
        """
        if not inputs:
            raise ValueError(
                f"{self.name}: automatic shape inference requires at least one input. "
                "Layers without inputs must implement output_shape()."
            )

        # A dynamic time axis (None) is probed with one step, then declared
        # dynamic again in the shape inferred from it.
        dummy_inputs = [
            keras.ops.zeros(
                (1, *(1 if axis is None else axis for axis in input_node.shape.tuple))
            )
            for input_node in inputs
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
        # The dummies hold one sample: more come out of a layer that reads one
        # stream and was given several, stacked on the batch axis.
        if any(output.shape[0] != 1 for output in output_values):
            raise ValueError(
                f"{self.name}: {type(self).__name__} reads one stream, got "
                f"{len(inputs)}."
            )
        inferred = [
            self._restore_dynamic_axes(
                self._shape_from_tensor(
                    output, max(input_node.shape.time_rank for input_node in inputs)
                ),
                inputs,
            )
            for output in output_values
        ]
        return inferred if multiple_outputs else inferred[0]

    @staticmethod
    def _restore_dynamic_axes(shape, inputs):
        """Declare dynamic again the time axes probed with one step.

        A time axis of the result is dynamic where an input with as many time
        axes has a dynamic one - the same rule a product of streams follows. A
        layer that adds or removes time axes is left as probed.
        """
        dynamic = {
            index
            for input_node in inputs
            if input_node.shape.time_rank == shape.time_rank
            for index, axis in enumerate(input_node.shape.time)
            if axis is None
        }
        if not dynamic:
            return shape
        return shape.replace(
            time=tuple(
                None if index in dynamic else axis
                for index, axis in enumerate(shape.time)
            )
        )

    def _shape_from_tensor(self, tensor, time_rank: int):
        shape = tuple(tensor.shape)[1:]
        # A layer may consume time axes (for example a recurrent body relation
        # reduced to one step), so preserve only those still present, and at
        # least one dim axis.
        time_rank = min(time_rank, max(0, len(shape) - 1))
        if not shape:
            raise ValueError(
                f"{self.name}: inferred tensor shape {shape} has no dim axis. "
                "Implement output_shape() to describe this layer's axes."
            )
        if any(axis is None for axis in shape):
            raise ValueError(
                f"{self.name}: inferred tensor shape {shape} contains dynamic axes. "
                "Implement output_shape() explicitly for dynamic output shapes."
            )

        return Shape.from_tensor_shape(
            tuple(int(axis) for axis in shape), time_rank=time_rank
        )

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
            if self._one_stream and len(inputs) > 1:
                # A layer that reads one stream, such as an activation, is
                # applied to each of several: one result per stream, in order.
                return [self([x]) for x in inputs if isinstance(x, Stream)]
            out_shapes = self.output_shape(*inputs)
            if not isinstance(out_shapes, list):
                out_shapes = [out_shapes]
            for out_shape in out_shapes:
                if not isinstance(out_shape, Shape):
                    raise TypeError(
                        f"{self.name}: {type(self).__name__}.output_shape() must "
                        f"return a Shape, got {out_shape!r}."
                    )
                node = self.__class__(name=self.name, **self._properties)
                node._source = self._source
                node.inputs = inputs
                # A copy: the shape of an Input changes as windows are taken.
                node.shape = out_shape.replace()
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


_BINARY_OPERATIONS = {
    "add": keras.ops.add,
    "subtract": keras.ops.subtract,
    "multiply": keras.ops.multiply,
    "divide": keras.ops.divide,
    "power": keras.ops.power,
    "arctan2": keras.ops.arctan2,
}


@keras.saving.register_keras_serializable(package="nnodely")
class BinaryOpImpl(keras.layers.Layer):
    """An elementwise operation between streams of ``time_ranks`` time axes
    and ``dim_ranks`` dim axes.

    An operand without time, or with fewer dim axes, gets the axes it misses
    where :func:`align` puts them, so it broadcasts along those, as
    :meth:`Shape.broadcast` declares. Aligned from the end instead, as
    broadcasting does by default, a dim axis could line up with a time axis.
    """

    def __init__(self, operation: str, time_ranks=(), dim_ranks=(), **kwargs):
        super().__init__(**kwargs)
        self.operation = operation
        self.time_ranks = tuple(int(rank) for rank in time_ranks)
        self.dim_ranks = tuple(int(rank) for rank in dim_ranks)

    def call(self, xs):
        time_rank, dim_rank = max(self.time_ranks), max(self.dim_ranks)
        values = [
            align(value, value_time_rank, value_dim_rank, time_rank, dim_rank)
            for value, value_time_rank, value_dim_rank in zip(
                xs, self.time_ranks, self.dim_ranks
            )
        ]

        operation = _BINARY_OPERATIONS[self.operation]
        result = values[0]
        for value in values[1:]:
            result = operation(result, value)
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "operation": self.operation,
                "time_ranks": self.time_ranks,
                "dim_ranks": self.dim_ranks,
            }
        )
        return config


class BinaryOp(Layer):
    operation = None

    def build_layer(self) -> keras.layers.Layer:
        if self.operation is None:
            raise NotImplementedError(
                "Subclasses must define an operation class attribute."
            )
        shapes = [input_node.shape for input_node in self.preds]  # type: ignore[union-attr]
        return BinaryOpImpl(
            operation=self.operation,
            time_ranks=tuple(shape.time_rank for shape in shapes),
            dim_ranks=tuple(shape.dim_rank for shape in shapes),
            name=self.name,
        )

    def output_shape(self, *inputs):  # type: ignore
        # Declared rather than probed: a dynamic time axis cannot be given to
        # a dummy tensor, and which operands may broadcast is a rule of the
        # streams, not of their tensors.
        return Shape.broadcast([input_node.shape for input_node in inputs], self.name)


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
        # Every axis is preserved, including a dynamic time axis that cannot
        # be probed with a dummy tensor.
        return inputs[0].shape

    def build_layer(self):
        return keras.layers.Identity(name=self.name)
