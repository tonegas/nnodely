from __future__ import annotations
import inspect
import math
from abc import abstractmethod

import keras
from typing import Any

from nnodely.core.shape import Shape
from nnodely.core.stream import Stream, _object_repr
from nnodely.core.dag import _free_name, _next_name


def _claim_keras_name(layer, taken: dict[str, Any]) -> None:
    """Give ``layer`` a name no other layer of the Keras model being built has.

    Two nnodely layers may share a name, as a model and its reloaded copy
    do, or layers named alike by hand. The layers of one Keras model may not:
    a layer arriving at a name already taken takes the first free suffix. A
    layer applied several times keeps the one name it has.
    """
    if layer.name in taken and taken[layer.name] is not layer:
        layer.name = _free_name(layer.name, taken)
    taken[layer.name] = layer


def _refuse_uncalled(streams) -> None:
    """Refuse a layer among ``streams`` that was never called on a stream:
    it reads nothing, and a model ran it on an arbitrary input of its own.
    Parameters and Constants read no stream by design."""
    from nnodely.layers.parameter import _Value

    for node in streams:
        if isinstance(node, Layer) and not node._preds and not isinstance(node, _Value):
            # A layer bound some other way, as a Loop, says how itself.
            _ = node.shape
            kind = type(node).__name__
            raise ValueError(
                f"The {kind} {node.name!r} was never called on a stream, so it "
                f"has nothing to compute on: call it on the streams it reads, "
                f"as in {kind}()(x), and use what that returns rather than the "
                "layer itself."
            )


def _refuse_non_streams(layer, inputs) -> None:
    """Refuse anything but streams among ``inputs``: a layer is applied to
    the streams of a graph, never to numbers or arrays."""
    given = [type(value).__name__ for value in inputs if not isinstance(value, Stream)]
    if given:
        raise TypeError(
            f"{layer.name}: {type(layer).__name__} is called on streams, got "
            f"{', '.join(given)}. A number or an array enters the graph as a "
            "Constant."
        )


def _without_ones(shape):
    """``shape`` without its axes of size 1."""
    return tuple(axis for axis in shape if axis != 1)


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

    A layer whose constructor takes nothing but ``name``, such as ``Sin``,
    ``Exp`` or ``Sigmoid``, has nothing to configure, so ``Sin(x)`` is ``Sin()(x)`` and
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

    @property
    def __signature__(cls) -> inspect.Signature:
        # The constructor's, so that help() and the documentation show the
        # settings of the layer rather than the (*args, **kwargs) of __call__.
        signature = inspect.signature(cls.__init__)
        return signature.replace(parameters=list(signature.parameters.values())[1:])

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

    Called on a stream, or a list of streams, it returns a new node of the
    graph; ``_call`` applies the concrete Keras layer to tensors when the
    model is built.
    """

    # A layer that reads one stream, such as an activation, called on several
    # is applied to each of them, and returns one result per stream.
    _one_stream = False

    # The weights the layer computes with, as Keras layers declare theirs with
    # add_weight: Parameters, which follow its one input among its
    # predecessors, in the order of these labels, as its Keras layer reads
    # them, [x, *weights]. The constructor sets each with ``weights``: a
    # Parameter, used as it is; an initializer, by name or object, from which
    # the layer makes a Parameter of the shape its input asks for, the first
    # time it is called, shared by every later call; True for the default
    # initializer of _weight_defaults; False for none. A layer with weights
    # gives their shapes for an input with _weight_shapes, and may give the
    # shapes their initializers draw with _draw_shapes.
    _weight_labels: tuple[str, ...] = ()
    _weight_defaults: dict[str, Any] = {}

    def __init__(self, name=None, tw=None, dim=None, seq=None, weights=None, **kwargs):
        self._properties = dict(kwargs)
        self._weight_settings = dict(weights or {})
        # The Parameters holding the weights, made or taken on the first call.
        self._weight_params: dict | None = None
        self._inputs = None
        self._layer: Any = None
        self._layer_signature = None
        # The layer this node applies. Every node made by calling one layer
        # object is an application of it, and they all share its weights;
        # layers created apart never do, whatever their names.
        self._source = self
        # The shape of the stream its last call returned, None before one.
        self._called_shape: Shape | None = None

        super().__init__(
            name=_next_name(self.__class__.__name__) if name is None else name,
            dim=dim,
            tw=tw,
            seq=seq,
        )

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def output_shape(self, *inputs):
        """Infer the output Shape by executing the concrete layer on dummy inputs.

        Layers with value-dependent output shapes or layers that change the
        number of sequence axes override this method, and return a Shape, or
        a list of them for several outputs.
        """
        if not inputs:
            raise ValueError(
                f"{self.name}: automatic shape inference requires at least one input. "
                "Layers without inputs must implement output_shape()."
            )

        # A dynamic sequence axis (None) is probed with one step, then
        # declared dynamic again in the shape inferred from it.
        dummy_inputs = [
            keras.ops.zeros(
                (1, *(1 if axis is None else axis for axis in input_node.shape.tuple))
            )
            for input_node in inputs
        ]

        # Some build_layer() implementations need the symbolic input context to
        # choose axes or configure the concrete Keras layer.
        self._inputs = list(inputs)
        self._preds = list(inputs)
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
        # The result has a window if a stream it is computed from has one.
        window = any(input_node.shape.tw is not None for input_node in inputs)
        inferred = [
            self._restore_dynamic_axes(
                self._shape_from_tensor(
                    output,
                    max(input_node.shape.seq_rank for input_node in inputs),
                    window,
                ),
                inputs,
            )
            for output in output_values
        ]
        return inferred if multiple_outputs else inferred[0]

    @staticmethod
    def _restore_dynamic_axes(shape, inputs):
        """Declare dynamic again the sequence axes probed with one step.

        A sequence axis of the result is dynamic where an input with as many
        sequence axes has a dynamic one, the same rule a product of streams
        follows. A layer that adds or removes sequence axes is left as probed.
        """
        dynamic = {
            index
            for input_node in inputs
            if input_node.shape.seq_rank == shape.seq_rank
            for index, axis in enumerate(input_node.shape.seq)
            if axis is None
        }
        if not dynamic:
            return shape
        return shape.replace(
            seq=tuple(
                None if index in dynamic else axis
                for index, axis in enumerate(shape.seq)
            )
        )

    def _shape_from_tensor(self, tensor, seq_rank: int, window: bool = True):
        shape = tuple(tensor.shape)[1:]
        # A layer may consume sequence axes (for example a recurrent body
        # relation reduced to one step), so preserve only those still
        # present, the window if any and at least one dim axis.
        seq_rank = min(seq_rank, max(0, len(shape) - 1 - int(window)))
        if len(shape) < 1 + int(window):
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
            tuple(int(axis) for axis in shape), seq_rank=seq_rank, tw=window
        )

    def _needs_window(self, stream) -> None:
        """Refuse ``stream`` if it has no window, the axis this layer acts on."""
        if stream.shape.tw is None:
            raise ValueError(
                f"{self.name}: {type(self).__name__} acts on the window of its "
                f"stream, and {stream.name!r} has none, {stream.shape}. Declare "
                "the Input with a window, tw=n for its last n samples."
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

    def _input_signature(self, xs) -> tuple | None:
        """The input shapes a concrete layer was built for.

        Weights belong to the shape they were made for, so a layer carried
        into another graph (a model used as a block brings its own along)
        can only be reused where its inputs still look the same. Layers that
        read no input at all (Parameters, Constants) are shape-independent and
        keep a constant signature.
        """
        if not self._preds:
            return ()
        try:
            return tuple(tuple(x.shape[1:]) for x in xs)
        except (AttributeError, TypeError):
            # A layer whose inputs are not plain tensors, such as a recurrent
            # body, which takes structures, cannot be checked this way. It is
            # reused as it always was.
            return None

    def _copied(self, memo: dict) -> None:
        """Called on a copy of this layer made for one call of the model
        holding it, ``memo`` the copies that call made: a layer holding a
        node of the graph besides its streams takes the copy it means."""

    def _reinitialize(self) -> None:
        """Give the weights this layer holds their initial values again, as
        :meth:`Modely.build` does with ``keep_weights=False``. A layer
        computing with Parameters holds none: they are initialized as nodes
        of their own."""

    def _call(self, xs):
        signature = self._input_signature(xs)
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

    def __call__(self, inputs: Any) -> Any:
        inputs = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]

        _refuse_non_streams(self, inputs)
        if self._one_stream and len(inputs) > 1:
            # A layer that reads one stream, such as an activation, is
            # applied to each of several, with the same weights: one result
            # per stream, in order.
            return [self([x]) for x in inputs]
        _refuse_uncalled(inputs)
        if self._weight_labels:
            # The weights are predecessors like the input, but configured on
            # the layer, so they are appended here and reconnected when
            # reloading. They take the place of any further stream, which
            # would otherwise be dropped without a word.
            if not inputs:
                raise ValueError(
                    f"{self.name}: {type(self).__name__} reads one stream, got "
                    "none. Apply it to a stream first."
                )
            if len(inputs) > 1:
                raise ValueError(
                    f"{self.name}: {type(self).__name__} reads one stream, got "
                    f"{len(inputs)}. Apply one to each, or join them first, "
                    "with Concatenate or TimeConcatenate."
                )
            inputs = [inputs[0], *self._obtain_weights(inputs[0])]
        return self._apply(inputs)

    def _apply(self, inputs) -> Any:
        """The nodes of the layer applied to ``inputs``, checked streams: one
        per output shape."""
        out_shapes = self.output_shape(*inputs)
        if not isinstance(out_shapes, list):
            out_shapes = [out_shapes]
        ret = []
        for out_shape in out_shapes:
            if not isinstance(out_shape, Shape):
                raise TypeError(
                    f"{self.name}: {type(self).__name__}.output_shape() must "
                    f"return a Shape, got {out_shape!r}."
                )
            node = self.__class__(name=self.name, **self._properties)
            node._source = self._source
            node._inputs = inputs
            node.shape = out_shape
            node._preds = inputs  # type: ignore
            # Computed from values alone (numbers, Parameters, Constants), a
            # result is a value too.
            node._is_value = all(x._is_value for x in inputs)
            ret.append(node)
        self._source._called_shape = ret[-1].shape
        return ret if len(ret) > 1 else ret[0]

    def __repr__(self) -> str:
        """A stream the layer returned prints as a Stream. The layer itself
        prints its name, the Parameter of each weight (or, before the first
        call makes it, its initializer) and the shape its last call
        returned."""
        if self._source is not self:
            return super().__repr__()
        weights = {label: self._weight_text(label) for label in self._weight_labels}
        return _object_repr(self, name=self.name, **weights, shape=self._called_shape)

    def _weight_text(self, label: str) -> str | None:
        """What the weight ``label`` prints as: the name of its Parameter, or
        its initializer before the first call; None without the weight."""
        if self._weight_params and label in self._weight_params:
            return self._weight_params[label].name
        setting = self._weight_settings.get(label, True)
        if setting is False:
            return None
        if setting is True:
            setting = self._weight_defaults[label]
        if isinstance(setting, Stream):
            return setting.name
        return setting if isinstance(setting, str) else type(setting).__name__

    def get_config(self):
        # The arguments the layer was created with, which rebuild it; its shape
        # follows from the streams it is called on again. Its weights are saved
        # with the graph, as the predecessors they are: an argument that sets
        # one only says whether it is there.
        config = {**super().get_config(), **self._properties}
        for label in self._weight_labels:
            if label in config:
                config[label] = config[label] is not False
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        layer = super().from_config(config)
        if not preds:
            return layer
        if cls._weight_labels:
            # The weights it has come back as its predecessors after its
            # input, in the order of their labels.
            weights = iter(preds[1:])
            for label in cls._weight_labels:
                if layer._weight_settings[label] is not False:
                    layer._weight_settings[label] = next(weights)
            return layer(preds[:1])
        return layer(preds)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _weight_shapes(self, x) -> tuple[tuple[int, ...], ...]:
        """The shapes of the weights for the input ``x``, one for each label."""
        raise NotImplementedError

    def _draw_shapes(self, shapes: tuple) -> tuple:
        """The shapes the initializers of weights of ``shapes`` draw, one for
        each; None for the default, the whole weight read as a matrix of its
        last axis by all the others, as a dense layer reads its kernel."""
        return (None,) * len(shapes)

    def _obtain_weights(self, x) -> list:
        """Create the weights of the layer on its first call, and return them;
        on every later call, return the ones already created.

        The shape of a weight depends on the input: ``Fir(features=2)``
        cannot make its kernel until it is called on ``x``, whose window
        sets its length. So they are made on the first call, from what the
        constructor was given for each label: a Parameter, used as it is;
        an initializer, or ``True`` for the default one, for a new Parameter
        ``<layer>_<label>``; ``False`` for none."""
        from nnodely.layers.parameter import Parameter

        shapes = dict(zip(self._weight_labels, self._weight_shapes(x)))
        if self._weight_params is None:
            draws = dict(
                zip(self._weight_labels, self._draw_shapes(tuple(shapes.values())))
            )
            self._weight_params = {}
            for label in self._weight_labels:
                setting = self._weight_settings[label]
                if setting is False:
                    continue
                if not isinstance(setting, Parameter):
                    if setting is True:
                        setting = self._weight_defaults[label]
                    shape, draw = shapes[label], draws[label]
                    if draw is None:
                        draw = (
                            shape
                            if len(shape) <= 2
                            else (math.prod(shape[:-1]), shape[-1])
                        )
                    setting = Parameter(
                        f"{self.name}_{label}",
                        dim=shape,
                        initializer=setting,
                        draw=draw,
                    )
                self._weight_params[label] = setting
        for label, weight in self._weight_params.items():
            if _without_ones(weight.shape.tuple) != _without_ones(shapes[label]):
                raise ValueError(
                    f"{self.name}: {label} must have the shape "
                    f"{tuple(shapes[label])}, axes of size 1 aside, got "
                    f"{weight.shape.tuple}."
                )
        return list(self._weight_params.values())

    @property
    def weights(self) -> dict:
        """The Parameters holding its weights, by label, such as
        ``{"kernel": ..., "bias": ...}``: empty before its first call and for
        a layer without weights. The same on the layer and on every stream a
        call of it returns."""
        return dict(self._source._weight_params or {})


class Identity(Layer):
    def output_shape(self, *inputs):
        # Every axis is preserved, including a dynamic sequence axis that cannot
        # be probed with a dummy tensor.
        return inputs[0].shape

    def build_layer(self):
        return keras.layers.Identity(name=self.name)
