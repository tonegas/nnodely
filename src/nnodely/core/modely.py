import os
import warnings
from pathlib import Path

from typing import Any, Callable

from nnodely.core.dag import toposort, flatten, _flatten_graph
from nnodely.core.validation import ValidationResult, score_signal
from nnodely.utils.utils import (
    MaskedLoss,
    _resolve_loss,
    _resolve_optimizer,
    _weighted_sequence_loss,
)
from nnodely.utils.printers import _resolve_printer
from nnodely.utils import validation_plot
from nnodely.core.registry import ModelSerializer
from nnodely.core.stream import Stream, Node
from nnodely.layers.constant import Constant
from nnodely.core.dataloader import DataLoader

import numpy as np
from typing import cast

from nnodely.layers.output import Output
from nnodely.layers.input import Input
from nnodely.core.layer import Layer

import keras

#: Samples evaluated per forward pass during validation. Validation is defined
#: one sample at a time - nothing about the result depends on this number - so
#: it is a memory bound rather than a parameter worth exposing.
_INFERENCE_CHUNK = 256


def _traces_backward_pass(model) -> bool:
    """True if any layer of the graph evaluates a backward pass of its own.

    Layers declare this themselves, so the export path does not have to know
    which ones they are. The walk is recursive because such a layer can sit
    inside a nested model - a recurrent body, or the sub-graph a Derivative
    differentiates.
    """
    seen: set[int] = set()
    stack = [model]
    while stack:
        layer = stack.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        if getattr(layer, "_traces_backward_pass", False):
            return True
        stack.extend(getattr(layer, "_layers", None) or [])
    return False


def _pair_onnx_names(values, expected):
    """Map each exported tensor to the Modely name it stands for.

    An exporter that renames the graph also reorders it, so position alone is
    not evidence of identity. A tensor whose shape matches exactly one expected
    name is that one whatever its position; only tensors left ambiguous - two
    outputs of the same shape - fall back to pairing in order.
    """

    def graph_shape(value):
        dims = value.type.tensor_type.shape.dim
        return tuple(int(axis.dim_value) for axis in dims[1:] if axis.dim_value)

    shapes = [graph_shape(value) for value in values]
    pending_graph = list(range(len(values)))
    pending_expected = list(range(len(expected)))

    pairs: dict[int, int] = {}
    for index in list(pending_graph):
        matches = [
            other for other in pending_expected if expected[other][1] == shapes[index]
        ]
        same_shape = [
            other for other in pending_graph if shapes[other] == shapes[index]
        ]
        if len(matches) == 1 and len(same_shape) == 1:
            pairs[index] = matches[0]
            pending_graph.remove(index)
            pending_expected.remove(matches[0])
    for index, other in zip(pending_graph, pending_expected):
        pairs[index] = other

    return {
        values[index].name: expected[other][0]
        for index, other in pairs.items()
        if values[index].name != expected[other][0]
    }


def _static_input_signature(model, batch_size: int):
    """The model's own input structure, with the batch axis fixed."""

    def spec(tensor):
        shape = (batch_size,) + tuple(int(axis) for axis in tensor.shape[1:])
        return keras.InputSpec(shape=shape, dtype=tensor.dtype, name=tensor.name)

    return [keras.tree.map_structure(spec, model._inputs_struct)]


def _depends_on_dynamic_input(node) -> bool:
    """True if a stream reads an Input whose sequence length is left dynamic.

    Those are the inputs a DataLoader pads when simulations differ in length,
    so they are the streams whose padded rollout steps the mask removes.
    """
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Input) and any(n is None for n in current.seq):
            return True
        stack.extend(current.preds)
    return False


def _reaches_trainable_weight(node) -> bool:
    """False only for a stream known to be computed without trainable weights.

    The walk enters the body of a called model as well. A layer not built yet
    cannot tell, and is taken to train: the check must never flag a minimizer
    that does.
    """
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Layer):
            layer = current._layer
            if layer is None or layer.trainable_weights:
                return True
        inner = getattr(current, "model", None)
        if isinstance(inner, Modely):
            stack.extend(inner.outputs)
        stack.extend(current.preds)
    return False


def _validate_gain(gain, minimizer: str) -> float:
    """A minimizer's weight in the total loss: a finite number, not negative.

    A negative gain would maximize the error it weighs.
    """
    if isinstance(gain, (bool, np.bool_)) or not isinstance(
        gain, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"The gain of minimizer {minimizer!r} must be a number, got "
            f"{type(gain).__name__}."
        )
    gain = float(gain)
    if not np.isfinite(gain) or gain < 0:
        raise ValueError(
            f"The gain of minimizer {minimizer!r} must be finite and not "
            f"negative, got {gain}."
        )
    return gain


def _seq_weights_for(seq_weights, steps: int | None, minimizer: str):
    """One weight per step of a last axis ``steps`` long, normalized to mean one.

    A callable is given the number of steps. Weights are not negative, and not
    all zero: a negative weight would maximize the error of its step, and all
    zeros would train nothing. ``steps=None`` leaves the length unchecked.
    """
    weights = seq_weights(steps) if callable(seq_weights) else seq_weights
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 1 or steps is not None and weights.shape != (steps,):
        raise ValueError(
            f"The seq_weights of minimizer {minimizer!r} must be one weight per "
            f"step of the last axis, {steps} of them, got shape {weights.shape}."
        )
    if not np.isfinite(weights).all() or (weights < 0).any() or not weights.any():
        raise ValueError(
            f"The seq_weights of minimizer {minimizer!r} must be finite, not "
            f"negative and not all zero, got {weights.tolist()}."
        )
    return weights / weights.mean()


def _validate_seq_weights(seq_weights, source: Stream, minimizer: str):
    """The seq_weights of a minimizer, checked as far as its source allows.

    A dynamic sequence follows the data, whatever length its stream declares,
    so its weights are checked once the data sets the length.
    """
    if seq_weights is None:
        return None
    steps = None if _depends_on_dynamic_input(source) else source.shape.tuple[-1]
    if callable(seq_weights):
        if steps is not None:
            _seq_weights_for(seq_weights, steps, minimizer)
        return seq_weights
    return _seq_weights_for(seq_weights, steps, minimizer)


class _MinimizerTerm:
    """One minimizer, resolved against the outputs of the built graph."""

    def __init__(
        self,
        name,
        source,
        target,
        target_rank,
        loss,
        masked,
        log_name,
        gain=1.0,
        seq_weights=None,
    ):
        self.name = name
        self.gain = gain  # weight of this term in the total loss
        # weight of each step of the last axis, or a callable of its length
        self.seq_weights = seq_weights
        self.source = source  # name of the graph output holding the prediction
        self.target = target  # name of the graph output holding the reference
        self.target_rank = target_rank  # rank of the target without batch axis
        self.masked = masked
        self.loss = MaskedLoss(loss) if masked else loss
        # Logged under the key compile() gave a per-output loss, which is the
        # one the printers and existing training scripts read.
        self.tracker = keras.metrics.Mean(name=f"{log_name}_loss")


class _MinimizerTerms:
    """Plain holder, so Keras does not track the terms as model state.

    The minimizers can change between two calls to train() without a new
    build(), and state tracked by the model could only ever grow.
    """

    def __init__(self, terms):
        self.terms = list(terms)


@keras.saving.register_keras_serializable(package="nnodely")
class MinimizerModel(keras.Model):
    """The built graph, trained on the minimizers of its Modely.

    Keras compiles one loss per output against a label fed from the dataset,
    but a minimizer compares two streams of the graph: its target can be an
    Output, a computed stream or a window narrower than the dataset column.
    Both sides are therefore read from the same forward pass, and the loss is
    computed here. The labels carry only the padding mask, when simulations of
    different lengths had to be padded to one rollout width.
    """

    def set_minimizers(self, terms):
        self._nnodely_minimizers = _MinimizerTerms(terms)

    def _minimizer_terms(self):
        holder = getattr(self, "_nnodely_minimizers", None)
        return holder.terms if holder is not None else []

    @property
    def metrics(self):
        return super().metrics + [term.tracker for term in self._minimizer_terms()]

    @staticmethod
    def _align_target(term, target, source):
        """Give a target built from constants the batch and axes of its source.

        Only such a target is broadcast. Two streams of the graph that differ
        in shape are an error: a loss would broadcast one over the other and
        silently compare, say, one sample with a whole window.
        """
        batch_free = len(target.shape) == term.target_rank
        if batch_free:
            target = keras.ops.expand_dims(target, axis=0)
            while len(target.shape) < len(source.shape):
                target = keras.ops.expand_dims(target, axis=-1)

        source_axes, target_axes = tuple(source.shape[1:]), tuple(target.shape[1:])
        compatible = len(source_axes) == len(target_axes) and all(
            s is None or t is None or s == t or (batch_free and t == 1)
            for s, t in zip(source_axes, target_axes)
        )
        if not compatible:
            raise ValueError(
                f"Minimizer {term.name!r} compares {term.source!r} of shape "
                f"{source_axes} with {term.target!r} of shape {target_axes}. "
                "They must have the same shape."
            )
        if batch_free:
            target = keras.ops.broadcast_to(target, keras.ops.shape(source))
        return target

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=True
    ):
        terms = self._minimizer_terms()
        if not terms:
            return super().compute_loss(x, y, y_pred, sample_weight, training=training)

        total = None
        for term in terms:
            if y_pred is None:
                raise ValueError(
                    f"Minimizer {term.name!r} compares {term.source!r} with "
                    f"{term.target!r}, but the forward pass did not return "
                    "the graph outputs it needs. Call build() after adding a "
                    "minimizer, before train()."
                )
            source = y_pred[term.source]
            target = self._align_target(term, y_pred[term.target], source)
            # The mask is one row per sample, one column per rollout step, and
            # the rollout is the last axis of every padded stream.
            rollout = (
                target.shape[-1] in (None, y.shape[-1]) if y is not None else False
            )
            if term.masked and rollout:
                valid = keras.ops.cast(y, "bool")
                for _ in range(len(target.shape) - 2):
                    valid = keras.ops.expand_dims(valid, axis=1)
                target = keras.ops.where(
                    valid, target, keras.ops.full_like(target, np.nan)
                )
            if term.seq_weights is not None:
                # The rollout is unrolled, so its length is static here.
                weights = _seq_weights_for(
                    term.seq_weights, source.shape[-1], term.name
                )
                value = _weighted_sequence_loss(term.loss, target, source, weights)
            else:
                # A loss function returns one value per sample, a Loss instance
                # the reduced value: the mean is how compile() reduces either.
                value = keras.ops.mean(term.loss(target, source))
            # Logged unweighted by the gain, as Keras logs a per-output loss
            # under loss_weights; only the total the optimizer sees is weighted.
            # The seq_weights are part of the loss itself, so they are logged.
            term.tracker.update_state(value)
            weighted = value if term.gain == 1.0 else value * term.gain
            total = weighted if total is None else total + weighted
        if self.losses:
            total = total + keras.ops.sum(self.losses)
        return total


class Modely:
    """A model-structured neural network.

    ``inputs`` and ``outputs`` delimit the graph of streams that makes up the
    model. Objectives (:meth:`minimize`) and feedback (:meth:`rollback`) are
    declared on it, then :meth:`build` creates the Keras model and its weights.

    A built model is called with ``{input name: array}`` and returns
    ``{output name: tensor}``. Called with a list of streams instead, it
    becomes a block of a larger graph and returns its outputs as streams.
    """

    name: str
    inputs: list[Input]
    outputs: list[Output]
    order: list[Node]

    def __init__(self, name: str, inputs: list[Input], outputs: list[Output]) -> None:
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.order = toposort(self)
        self.model = None  # Keras model build from this DAG
        self.train_inputs = []  # List of Input nodes that are required for training (derived from minimizers)
        self.train_outputs = []  # List of Output nodes that are required for training (derived from minimizers)
        self.minimizers = []  # List of dicts with keys: 'source', 'target', 'loss', 'name'
        self._minimizer_outputs: dict[Node, str] = {}  # minimizer side -> graph output
        self._roll_callbacks: dict[Input, Stream] = {}
        self._roll_steps: int | None = None
        self._roll_name: str | None = None

    def __repr__(self) -> str:
        items = " \n- ".join(map(str, self.order))
        return f"Model {self.name}:\n - {items}"

    def __call__(self, inputs) -> Any:
        if all(isinstance(v, Node) for v in inputs):
            mc = ModelCall(f"{self.name}_call", self)
            mc.preds = inputs
            mc.inputs_map = {old: new for old, new in zip(self.inputs, inputs)}
            outputs = [IntermediateOutput(output, mc) for output in self.outputs]
            mc.outputs_map = {new: old for old, new in zip(self.outputs, outputs)}
            return outputs[0] if len(outputs) == 1 else outputs

        # tensor execution mode
        if self.model is None:
            raise ValueError("Model build failed, model is still None.")
        if isinstance(inputs, dict):
            expected = {inp.name for inp in self.train_inputs}
            extra = set(inputs) - expected
            if extra:
                inputs = {k: v for k, v in inputs.items() if k in expected}
            for inp in self.train_inputs:
                value = inputs[inp.name]
                if not hasattr(value, "shape"):
                    value = np.asarray(value)
                if len(value.shape) < inp.shape.rank and value.size == int(
                    np.prod(inp.shape.tuple)
                ):
                    value = np.reshape(value, inp.shape.tuple)
                inputs[inp.name] = value
        for idx, inp in enumerate(self.train_inputs):
            if isinstance(inputs, dict):
                if len(inputs[inp.name].shape) == inp.shape.rank:
                    if type(inputs[inp.name]) is np.ndarray:
                        inputs[inp.name] = np.expand_dims(inputs[inp.name], axis=0)
                    else:
                        inputs[inp.name] = keras.ops.expand_dims(
                            inputs[inp.name], axis=0
                        )
            else:
                if len(inputs[idx].shape) == inp.shape.rank:
                    if type(inputs[idx]) is np.ndarray:
                        inputs[idx] = np.expand_dims(inputs[idx], axis=0)
                    else:
                        inputs[idx] = keras.ops.expand_dims(inputs[idx], axis=0)
        return self.model(inputs)

    @property
    def built(self):
        """True once :meth:`build` has been called."""
        return self.model is not None

    def build(self):
        """Create the Keras model of the graph, with its weights.

        Objectives and feedback have to be declared before, because they add
        to the graph that is built. Returns the model itself.
        """
        from nnodely.core.layer import Identity

        # A minimizer reads both of its sides from the forward pass, so every
        # source and target has to be an output of the graph. An Input is not
        # the value of any layer, so it is exposed through an Identity.
        extra_outputs = []
        self._minimizer_outputs = {}
        for minimizer in self.minimizers:
            for node in (minimizer["source"], minimizer["target"]):
                if node in self._minimizer_outputs:
                    continue
                exposed = Identity()([node]) if isinstance(node, Input) else node
                if exposed not in self.outputs:
                    extra_outputs.append(exposed)
                self._minimizer_outputs[node] = exposed.name

        self.train_outputs = self.outputs + extra_outputs
        feedback_streams = list(dict.fromkeys(self._roll_callbacks.values()))
        graph_outputs = self.train_outputs + [
            stream for stream in feedback_streams if stream not in self.train_outputs
        ]
        flat, flatten_memo = _flatten_graph(
            self.name,
            self.inputs,
            graph_outputs,
            return_memo=True,
        )
        self.train_inputs = [node for node in flat.order if isinstance(node, Input)]

        flat_graph_outputs = [flatten_memo[node] for node in graph_outputs]
        keras_inputs, keras_outputs = self.resolve_graph(
            flat.order, output_nodes=flat_graph_outputs
        )
        # Graph flattening builds shallow copies. Keep the public symbolic nodes
        # connected to the concrete Keras layers created from those copies, and
        # to the input shapes those were built for - the handle and what it
        # fits travel together, so composing this model later can tell whether
        # its layers can be reused where they land.
        for source_node, flat_node in flatten_memo.items():
            if isinstance(source_node, Layer) and isinstance(flat_node, Layer):
                source_node._layer = flat_node._layer
                source_node._layer_signature = flat_node._layer_signature

        body_class = keras.Model if self._roll_callbacks else MinimizerModel
        body_model = body_class(
            name=self.name + "_train",
            inputs=keras_inputs,
            outputs=keras_outputs,
        )
        if self._roll_callbacks:
            from nnodely.layers.roll import ModelRollImpl

            callbacks = {
                input_node.name: feedback.name
                for input_node, feedback in self._roll_callbacks.items()
            }
            roll_layer = ModelRollImpl(
                model=body_model,
                callbacks=callbacks,
                output_names=tuple(node.name for node in self.train_outputs),
                input_time_axes={
                    node.name: node.shape.dim_rank + 1 for node in self._roll_callbacks
                },
                steps=cast(int, self._roll_steps),
                name=self._roll_name or f"{self.name}_roll",
            )
            final_outputs = roll_layer(keras_inputs)
            self.model = MinimizerModel(
                name=self.name + "_train",
                inputs=keras_inputs,
                outputs=final_outputs,
            )
        else:
            self.model = body_model
        return self

    def resolve_graph(self, order, output_nodes=None):
        # Applying one layer several times yields several nodes that carry its
        # name, so tensors are keyed by node identity: keying them by name made
        # each application overwrite the previous one, and every consumer then
        # read the same tensor. The concrete Keras layer is still keyed by name,
        # so those applications share one layer and therefore its weights, the
        # way repeated calls do in Keras.
        input_tensors = {
            node.name: node.input for node in order if isinstance(node, Input)
        }
        tensor_map: dict[Node, Any] = {
            node: input_tensors[node.name] for node in order if isinstance(node, Input)
        }
        layers_by_name: dict[str, Any] = {}

        for node in [n for n in order if not isinstance(n, Input)]:
            if isinstance(node, Layer):
                shared = layers_by_name.get(node.name)
                if shared is not None:
                    node._layer = shared
                if len(node.preds) == 0:  ## Parameters and Constants
                    anchor = next(iter(tensor_map.values()), None)
                    tensor_map[node] = node.call([anchor])
                else:
                    tensor_map[node] = node.call(
                        [tensor_map[pred] for pred in node.preds]
                    )
                layers_by_name[node.name] = node._layer
            else:  ## Output or other non-Layer node
                tensor_map[node] = tensor_map[node.preds[0]]

        keras_inputs = {node.name: tensor_map[node] for node in self.train_inputs}
        output_nodes = self.train_outputs if output_nodes is None else output_nodes

        # Outputs are addressed by name, so two different nodes claiming one name
        # would silently drop one of them. Calling a model several times produces
        # exactly that: its outputs all carry the body's output names.
        keras_outputs = {}
        claimed: dict[str, Node] = {}
        for node in output_nodes:
            owner = claimed.get(node.name)
            if owner is not None and owner is not node:
                raise ValueError(
                    f"Model {self.name!r} has two different outputs named "
                    f"{node.name!r}. Calling one model several times returns outputs "
                    "that share its output names, so wrap them in Output nodes with "
                    "distinct names before exposing them."
                )
            claimed[node.name] = node
            keras_outputs[node.name] = tensor_map[node]
        return keras_inputs, keras_outputs

    # -------------------------------------------------------------------------
    # Train API
    # -------------------------------------------------------------------------

    def minimize(
        self,
        name: str,
        source: Output | Stream | str,
        target: Output | Stream | str | float | None = None,
        loss: str | dict[str, Any] | keras.losses.Loss | Callable = "mse",
        gain: float = 1.0,
        seq_weights: Any = None,
    ):
        """Register a Keras loss to minimize during training.

        name: identifier for this loss (used as an output name in the training model)
        source: node/stream producing predictions (e.g., an Output node), or the
            name of a stream of the model
        target: node/stream providing target values (usually derived from an
            Input), the name of a stream of the model, a number, or None for
            zero - a number becomes a Constant
        loss: Keras loss name, serialized config, Loss instance, or callable
        gain: weight of this loss in the total training loss, 1 by default -
            0.5 gives it half the importance, 2 twice. Its own logged and
            validated loss stays unweighted, the way Keras logs a loss_weight
        seq_weights: one weight per step of the last axis of the source - the
            rollout of a Loop, or the window of a stream - such as
            ``np.exp(0.1 * np.arange(N))`` to weigh the last predictions more,
            or a callable of the number of steps returning them, such as
            ``lambda n: np.exp(0.1 * np.arange(n))``, for a sequence whose
            length is dynamic. Normalized to mean one, so only their profile
            matters; the logged and validated loss is the weighted one. None
            weighs every step the same. It suits losses computed element by
            element, as the regression losses are

        A name identifies one minimizer: registering a name again replaces the
        minimizer it names, in its place, and warns that it did.
        """
        source = self._minimizer_stream(source, "source", name)
        if target is None:  ## Transform it into a Constant with value zero
            target = 0.0
        # Any real number, numpy scalars included; a bool is not a target value.
        if isinstance(target, (int, float, np.integer, np.floating)) and not isinstance(
            target, (bool, np.bool_)
        ):
            target = Constant(name=None, value=[float(target)])
        target = self._minimizer_stream(target, "target", name)
        resolved_loss = _resolve_loss(loss)
        gain = _validate_gain(gain, name)
        seq_weights = _validate_seq_weights(seq_weights, source, name)
        minimizer = {
            "name": name,
            "source": source,
            "target": target,
            "loss": resolved_loss,
            "gain": gain,
            "seq_weights": seq_weights,
        }
        for index, existing in enumerate(self.minimizers):
            if existing["name"] == name:
                warnings.warn(
                    f"Minimizer {name!r} already exists in model {self.name!r}: "
                    "it is replaced by the new one.",
                    UserWarning,
                    stacklevel=2,
                )
                self.minimizers[index] = minimizer
                return self
        self.minimizers.append(minimizer)
        return self

    def _named_streams(self) -> list[Stream]:
        """Every stream a minimizer can name: the model's graph, its declared
        inputs, and the streams the registered minimizers already compare."""
        roots: list[Node] = [*self.outputs, *self.inputs]
        for minimizer in self.minimizers:
            roots.extend((minimizer["source"], minimizer["target"]))
        streams: dict[int, Stream] = {}
        stack = list(roots)
        while stack:
            node = stack.pop()
            if id(node) in streams:
                continue
            if isinstance(node, Stream):
                streams[id(node)] = node
            stack.extend(node.preds)
        return list(streams.values())

    def _minimizer_stream(self, value, side: str, minimizer: str) -> Stream:
        """A minimizer side as a Stream, looking a name up in the model."""
        if isinstance(value, Stream):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"The {side} of minimizer {minimizer!r} must be a Stream or the "
                f"name of one, got {type(value).__name__}."
            )
        matches = [stream for stream in self._named_streams() if stream.name == value]
        if not matches:
            outputs = sorted(output.name for output in self.outputs)
            raise ValueError(
                f"The {side} {value!r} of minimizer {minimizer!r} is not the name "
                f"of a stream of model {self.name!r}. Its outputs are {outputs}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"The {side} {value!r} of minimizer {minimizer!r} is ambiguous: "
                f"{len(matches)} streams of model {self.name!r} carry that name. "
                "Pass the stream itself."
            )
        return matches[0]

    def remove_minimizer(self, name: str):
        """Remove a registered minimizer by name; an unknown name only warns."""
        if all(minimizer["name"] != name for minimizer in self.minimizers):
            warnings.warn(
                f"Model {self.name!r} has no minimizer named {name!r}: nothing "
                "was removed.",
                UserWarning,
                stacklevel=2,
            )
            return
        self.minimizers = [m for m in self.minimizers if m["name"] != name]

    # -------------------------------------------------------------------------
    # Validation API
    # -------------------------------------------------------------------------

    def _predict(self, x_data: dict, n_samples: int) -> dict[str, np.ndarray]:
        """Run the graph over the whole dataset with the training path disabled.

        ``training=False`` is the backend-independent switch: it is what tells
        every Keras layer to take its inference branch, so no per-backend module
        mode has to be toggled here.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call build() before validate().")
        input_names = [node.name for node in self.train_inputs]
        missing = [name for name in input_names if name not in x_data]
        if missing:
            raise ValueError(f"Validation data is missing model inputs: {missing}.")

        # An output built only from constants and parameters has no batch axis:
        # every chunk returns the same value, which is kept once, not stacked.
        ranks = {node.name: node.shape.rank for node in self.train_outputs}
        chunks: dict[str, list[np.ndarray]] = {}
        batch_free: dict[str, np.ndarray] = {}
        for start in range(0, n_samples, _INFERENCE_CHUNK):
            stop = min(start + _INFERENCE_CHUNK, n_samples)
            batch = {name: x_data[name][start:stop, ...] for name in input_names}
            for name, value in self.model(batch, training=False).items():
                value = keras.ops.convert_to_numpy(value)
                if value.ndim == ranks.get(name):  # type: ignore
                    batch_free[name] = value  # type: ignore
                else:
                    chunks.setdefault(name, []).append(value)  # type: ignore

        predictions = {
            name: np.concatenate(values, axis=0) for name, values in chunks.items()
        }
        predictions.update(batch_free)
        return predictions

    def _minimizer_key(self, node) -> str | None:
        """Name under which the forward pass returns a minimizer's side."""
        return self._minimizer_outputs.get(node, node.name)

    @staticmethod
    def _broadcast_batch_free(value, source_values: np.ndarray) -> np.ndarray:
        """A target with no batch axis, laid out like its source's predictions.

        The numpy twin of what training does: prepend the batch axis, append
        the axes the target omits, then broadcast - so a dim-2 Constant lines
        up with the dim axis, not the time axis.
        """
        value = np.asarray(value, dtype=np.float32)[np.newaxis, ...]
        while value.ndim < source_values.ndim:
            value = value[..., np.newaxis]
        return np.broadcast_to(value, source_values.shape)

    def _dataset_name(self, node, x_data: dict) -> str | None:
        """Name of the dataset column a node ultimately reads, if any."""
        if node.name in x_data:
            return node.name
        preds = getattr(node, "preds", [])
        if len(preds) == 1:
            return self._dataset_name(preds[0], x_data)
        return None

    def _validation_target(
        self, minimizer: dict, x_data: dict, predictions: dict, n_samples: int
    ) -> tuple[np.ndarray, str]:
        """Resolve the reference signal a minimizer is scored against.

        The evaluated target stream wins over the dataset column it reads: a
        target declared as ``y.sw(2)`` on an input that carries a wider window
        elsewhere in the graph is two steps long, not as wide as the column.
        The column still names the signal, which is what a reader recognizes.
        """
        source_values = predictions[self._minimizer_key(minimizer["source"])]
        target = minimizer["target"]
        label = self._dataset_name(target, x_data) or target.name

        key = self._minimizer_key(target)
        if key in predictions:
            values = predictions[key]
            if values.ndim == target.shape.rank:  # built from constants only
                return self._broadcast_batch_free(values, source_values), label
            return np.asarray(values[:n_samples]), label

        if label in x_data:
            return np.asarray(x_data[label][:n_samples]), label

        value = getattr(target, "value_numpy", None)
        if value is None:
            raise ValueError(
                f"Validation target {target.name!r} of minimizer "
                f"{minimizer['name']!r} is neither in the dataset nor a constant."
            )
        return self._broadcast_batch_free(value, source_values), target.name

    def validate(
        self,
        val_data,
        out_dir: str | os.PathLike | None = None,
        show: bool = False,
        history: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Score the built model on ``val_data`` and draw what it did.

        Every minimizer is evaluated on the validation set and reported with
        the loss it was trained on plus the indicators a mechanical
        system-identification report is read for: RMSE, MAE, peak error, bias,
        NRMSE, the FIT percentage, R2 and correlation.

        Parameters
        ----------
        val_data:
            A :class:`DataLoader`, or anything exposing ``as_dict()`` and
            ``__len__``.
        out_dir:
            Folder the figures are written to as PNG. ``None`` saves nothing.
        show:
            Open the figures in an interactive window - zoom, pan and edit the
            curves with the usual Matplotlib toolbar.
        history:
            The dictionary returned by :meth:`train`, drawn as loss curves.

        The summary is printed and the whole result returned, so the numbers
        can be asserted on or logged as well as read.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call build() before validate().")
        if not self.minimizers:
            raise ValueError("No minimizers defined. Cannot infer validation targets.")

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)

        x_data = {
            name: np.asarray(values) for name, values in val_data.as_dict().items()
        }

        n_samples = len(val_data)
        if n_samples == 0:
            raise ValueError("Validation dataset is empty.")

        x_data = {
            name: np.asarray(values) for name, values in val_data.as_dict().items()
        }
        predictions = self._predict(x_data, n_samples)

        signals = {}
        for minimizer in self.minimizers:
            name = minimizer["name"]
            source = minimizer["source"]
            source_key = self._minimizer_key(source)
            if source_key not in predictions:
                raise ValueError(
                    f"Minimizer {name!r} sources {source.name!r}, which the built "
                    "model does not expose as an output."
                )
            y_true, target_name = self._validation_target(
                minimizer, x_data, predictions, n_samples
            )
            signals[name] = score_signal(
                name=name,
                source=source.name,
                target=target_name,
                loss_fn=minimizer["loss"],
                seq_weights=(
                    None
                    if minimizer.get("seq_weights") is None
                    else _seq_weights_for(
                        minimizer["seq_weights"], y_true.shape[-1], name
                    )
                ),
                y_true=y_true,
                y_pred=np.asarray(predictions[source_key][:n_samples]),
            )

        result = ValidationResult(
            model=self.name,
            samples=n_samples,
            signals=signals,
            figures=[],
            history=history,
        )
        if out_dir is not None or show:
            result.figures = validation_plot.render(result, out_dir=out_dir, show=show)
        print(result.summary())
        return result

    def _training_arrays(self, data: DataLoader) -> tuple[dict, np.ndarray | None]:
        """The model's inputs and, when simulations were padded, their mask.

        Both sides of every minimizer come from the forward pass, so the only
        label left is the mask that tells real rollout steps from padded ones.
        """
        x_data = {name: np.asarray(values) for name, values in data.as_dict().items()}
        mask = getattr(data, "mask", None)
        return x_data, None if mask is None else np.asarray(mask, dtype=np.float32)

    def _minimizer_terms(self) -> list[_MinimizerTerm]:
        """Resolve every minimizer to the graph outputs of the built model."""
        assert self.model is not None
        # The forward pass returns a dict keyed by the names of these nodes.
        output_names = {node.name for node in self.train_outputs}

        def output_name(minimizer, side):
            node = minimizer[side]
            name = self._minimizer_outputs.get(node)
            if name is None and not isinstance(node, Input):
                name = node.name
            if name not in output_names:
                raise ValueError(
                    f"The {side} {node.name!r} of minimizer {minimizer['name']!r} "
                    "is not an output of the built model. Call build() after "
                    "adding a minimizer, before train()."
                )
            return name

        return [
            _MinimizerTerm(
                name=minimizer["name"],
                source=output_name(minimizer, "source"),
                target=output_name(minimizer, "target"),
                target_rank=minimizer["target"].shape.rank,
                loss=_resolve_loss(minimizer["loss"]),
                masked=_depends_on_dynamic_input(minimizer["source"])
                or _depends_on_dynamic_input(minimizer["target"]),
                log_name=minimizer["source"].name,
                gain=minimizer.get("gain", 1.0),
                seq_weights=minimizer.get("seq_weights"),
            )
            for minimizer in self.minimizers
        ]

    def _check_minimizers_train_weights(self, km: keras.Model) -> None:
        """Every minimizer should compare something the network computes.

        A minimizer whose source and target reach no trainable weight adds a
        constant to the loss and trains nothing. Alongside minimizers that do
        train, it is only worth a warning; when none of them trains the
        network's weights, the training could not change anything.
        """
        if not km.trainable_weights:
            return  # Keras itself warns that the model has nothing to train
        weightless = [
            minimizer["name"]
            for minimizer in self.minimizers
            if not _reaches_trainable_weight(minimizer["source"])
            and not _reaches_trainable_weight(minimizer["target"])
        ]
        if len(weightless) == len(self.minimizers):
            raise ValueError(
                f"No minimizer of model {self.name!r} depends on a trainable "
                f"weight: {weightless} compare streams computed without any, so "
                "training could not change the network. Compare an output of "
                "the network with its reference."
            )
        for name in weightless:
            warnings.warn(
                f"Minimizer {name!r} of model {self.name!r} depends on no "
                "trainable weight: it adds a constant to the loss and trains "
                "nothing.",
                UserWarning,
                stacklevel=3,
            )

    def train(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
        epochs: int = 10,
        batch_size: int = 1,
        optimizer: str | dict[str, Any] | keras.optimizers.Optimizer | None = None,
        lr: float = 1e-3,
        shuffle: bool = True,
        optimizer_kwargs: dict[str, Any] | None = None,
        printer: str | keras.callbacks.Callback | None = "legacy",
    ):
        """Train the model with any Keras optimizer.

        ``optimizer`` may be a Keras optimizer name, a serialized Keras
        optimizer configuration, or an optimizer instance. When a name (or
        ``None``) is provided, ``lr`` and ``optimizer_kwargs`` are used to
        construct it. Optimizer instances and serialized configurations retain
        their own learning-rate configuration.

        ``val_data`` is evaluated at the end of every epoch and its losses are
        returned alongside the training ones under ``val_`` keys. Only the two
        curves together say whether a falling training loss is the model
        learning the system or memorizing the training set, so pass it whenever
        a held-out set exists - :meth:`validate` draws them on one axis.

        ``printer`` selects how progress is rendered: ``"tiny"`` prints a compact
        summary of the training progress, ``"legacy"`` prints the scrolling
        per-minimizer loss table of the original nnodely trainer, ``"nnodely"``
        drives the animated machine-room console, ``None`` prints nothing, and
        any Keras callback is used as given.
        """
        if not self.minimizers:
            raise ValueError("No minimizers defined. Call minimize() before train().")

        n_samples = len(train_data)
        if n_samples == 0:
            raise ValueError("train_data is empty.")

        # Ensure model is built
        if not self.model:
            raise ValueError("Model is not built. Call build() before training.")

        resolved_optimizer = _resolve_optimizer(optimizer, lr, optimizer_kwargs)
        self.model.set_minimizers(self._minimizer_terms())
        self._check_minimizers_train_weights(self.model)

        x_data, mask = self._training_arrays(train_data)

        validation_data = None
        val_mask = None
        if val_data is not None:
            if len(val_data) == 0:
                raise ValueError("val_data is empty.")
            val_x, val_mask = self._training_arrays(val_data)
            validation_data = val_x if val_mask is None else (val_x, val_mask)

        self.model.compile(optimizer=resolved_optimizer, jit_compile="auto")
        history = self.model.fit(
            x=x_data,
            y=mask,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=0,  # type: ignore
            callbacks=[_resolve_printer(printer, epochs, self.minimizers, self.name)],
            validation_data=validation_data,
        )

        return history.history

    # -------------------------------------------------------------------------
    # Flatten API
    # -------------------------------------------------------------------------

    def flatten(self) -> "Modely":
        """Return an equivalent model with every sub-model inlined."""
        return flatten(model=self)

    # -------------------------------------------------------------------------
    # Visualization API
    # -------------------------------------------------------------------------

    def export_html(
        self,
        out_dir: str | os.PathLike,
        filename: str | None = None,
        *,
        open_subgraph_in_new_tab: bool = False,
        physics: bool = True,
    ) -> str:
        """
        Export this Modely DAG to interactive HTML using vis-network.

        ``out_dir`` is the folder the pages are written to; a path ending in
        ``.html`` names the root page instead and its parent becomes the folder.

        Every block that wraps a model (``ModelCall``, ``Loop``, ``Roll``) is
        exported recursively as its own page, linked from the block node and
        annotated with the ports that bind the body to the graph above it.

        Returns
        -------
        str
            Path to the root exported HTML file.
        """
        from nnodely.utils.plot import export_html

        return export_html(
            model=self,
            out_dir=out_dir,
            filename=filename,
            open_subgraph_in_new_tab=open_subgraph_in_new_tab,
            physics=physics,
        )

    def summary(self):
        """Print the Keras summary of the built model."""
        if self.model is not None:
            self.model.summary()

    def plot(
        self, to_file: str, include_minimizers: bool = True, flatten: bool = False
    ):
        """
        Render a left-to-right graph of the model DAG to `to_file` using graphviz.

        Shapes:
        - Inputs / Outputs: rounded
        - Intermediate relations: square
        - Sub-models: folder
        - Minimizers: hexagon, unique color per loss type

        If include_minimizers=True, each minimizer is shown as a dedicated node
        labeled with its loss type and connected from source/target.
        """
        from nnodely.utils.plot import plot_graphviz

        return plot_graphviz(
            model=self,
            to_file=to_file,
            include_minimizers=include_minimizers,
            flatten=flatten,
        )

    # -------------------------------------------------------------------------
    # roll API
    # -------------------------------------------------------------------------
    def rollback(
        self,
        rollback: dict[str | Input, str | Stream],
        steps: int,
        name: str | None = None,
    ) -> "Modely":
        """Configure temporal feedback directly on this model.

        Each mapping is ``input: stream``. After every model evaluation, the
        stream's one-step result is appended to the input's temporal window.
        The model is unrolled for ``steps`` evaluations and exposes only the
        outputs produced by the final evaluation, preserving their original
        symbolic shapes.
        """
        if self.built:
            raise ValueError("roll() must be called before build().")
        if not isinstance(rollback, dict) or not rollback:
            raise ValueError("roll must be a non-empty input: stream mapping.")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("roll steps must be a positive integer.")

        graph_streams = [node for node in self.order if isinstance(node, Stream)]

        def resolve(nodes, value, kind):
            if isinstance(value, str):
                matches = [node for node in nodes if node.name == value]
                if len(matches) != 1:
                    raise ValueError(f"roll {kind} {value!r} was not found uniquely.")
                return matches[0]
            if value not in nodes:
                raise ValueError(
                    f"roll {kind} {getattr(value, 'name', value)!r} "
                    "does not belong to this Modely."
                )
            return value

        callbacks = {}
        for input_ref, stream_ref in rollback.items():
            input_node = resolve(self.inputs, input_ref, "input")
            stream = resolve(graph_streams, stream_ref, "stream")
            expected = tuple(input_node.dim) + (1,) + tuple(input_node.seq)
            if stream.shape.tuple != expected:
                raise ValueError(
                    f"roll stream {stream.name!r} must produce one temporal "
                    f"sample with shape {expected}, got {stream.shape.tuple}."
                )
            callbacks[input_node] = stream

        self._roll_callbacks = callbacks
        self._roll_steps = steps
        self._roll_name = name
        return self

    # -------------------------------------------------------------------------
    # Save and load
    # -------------------------------------------------------------------------

    def save(self, path):
        """Save the model - graph, objectives and weights - to ``path``."""
        ModelSerializer.serialize(self, path)

    @classmethod
    def load(cls, path):
        """Load a model saved with :meth:`save`."""
        return ModelSerializer.load(path)

    def export_keras(self, filename: str):
        """Save the built Keras model to a ``.keras`` file."""
        if self.model is None:
            raise ValueError("Model is not built. Call build() before export_keras().")

        if not isinstance(self.model, keras.Model):
            raise TypeError(f"Expected keras.Model, got {type(self.model)}.")

        self.model.save(filename)

    @staticmethod
    def import_keras(filename: str, safe_mode: bool = True):
        """Load a ``.keras`` file written by :meth:`export_keras` as a ``keras.Model``."""
        path = Path(filename)
        if path.suffix.lower() != ".keras":
            path = path.with_suffix(".keras")
        return keras.models.load_model(
            path,
            safe_mode=safe_mode,
        )

    def export_onnx(
        self,
        filename: str | os.PathLike,
        *,
        input_signature=None,
        batch_size: int | None = None,
        opset_version: int | None = None,
        verbose: bool = False,
    ) -> Path:
        """Export the built inference graph to ONNX.

        ONNX export traces the built Keras graph; it does not deserialize
        nnodely layer configurations. An explicit ``input_signature`` is
        recommended when dynamic dimensions must remain fixed at export time.
        ``batch_size`` is the shorthand for the common case: it exports with
        the batch axis fixed to that many samples instead of left dynamic.

        A layer that differentiates a sub-graph - ``Derivative`` with respect
        to an Input - records the backward pass in the traced graph, and the
        shape arithmetic it introduces has no ONNX equivalent while the batch
        axis is dynamic. Such a model is therefore exported with a batch of
        one unless a signature or ``batch_size`` says otherwise.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call build() before export_onnx().")

        if keras.backend.backend() == "jax":
            # Keras reaches ONNX from jax through jax2tf, and jax 0.4.36 removed
            # graph serialization, so jax2tf now emits the whole model as one
            # opaque XlaCallModule node that tf2onnx has no converter for.
            raise NotImplementedError(
                "ONNX export is not available on the jax backend: jax>=0.4.36 "
                "makes jax2tf emit an XlaCallModule node that tf2onnx cannot "
                "convert. Export from the tensorflow or torch backend instead "
                "(set KERAS_BACKEND before importing keras)."
            )

        path = Path(filename)
        if path.suffix.lower() != ".onnx":
            path = path.with_suffix(".onnx")
        path.parent.mkdir(parents=True, exist_ok=True)

        export_model = self.model
        if keras.backend.backend() == "torch" and isinstance(self.model.input, dict):
            # Keras' Torch ONNX exporter does not currently accept dictionary
            # signatures. Trace an equivalent positional wrapper instead.
            export_inputs = [
                keras.Input(
                    shape=tuple(tensor.shape[1:]),
                    dtype=tensor.dtype,
                    name=tensor.name,
                )
                for tensor in self.model.inputs
            ]
            input_map = {
                tensor.name: export_input
                for tensor, export_input in zip(self.model.inputs, export_inputs)
            }
            export_model = keras.Model(
                export_inputs,
                self.model(input_map, training=False),
                name=f"{self.model.name}_onnx",
            )

        # Keras requires a model to have been called before export. Modely.build
        # creates a Functional model, but does not necessarily execute it.
        if not getattr(export_model, "_called", False):
            warmup_inputs = {}
            for tensor in export_model.inputs:
                shape = tuple(1 if dim is None else int(dim) for dim in tensor.shape)
                warmup_inputs[tensor.name] = np.zeros(shape, dtype=np.float32)
            export_model(warmup_inputs, training=False)

        if _traces_backward_pass(export_model) and keras.backend.backend() == "torch":
            raise NotImplementedError(
                "ONNX export of a model that differentiates a sub-graph - a "
                "Derivative with respect to an Input - is not supported on the "
                "'torch' backend: its exporter traces a forward pass only and "
                "cannot record the backward pass such a layer evaluates. The "
                "'tensorflow' and 'jax' backends export it, because there the "
                "backward pass becomes ordinary graph operations."
            )

        if input_signature is None:
            if batch_size is None and _traces_backward_pass(export_model):
                batch_size = 1
            if batch_size is not None:
                input_signature = _static_input_signature(export_model, batch_size)

        export_kwargs: dict[str, Any] = {"verbose": verbose}
        if input_signature is not None:
            export_kwargs["input_signature"] = input_signature
        if opset_version is not None:
            export_kwargs["opset_version"] = opset_version

        export_model.export(path, format="onnx", **export_kwargs)
        self._set_onnx_io_names(path)
        return path

    def _set_onnx_io_names(self, path: Path) -> None:
        """Restore Modely input/output names if an exporter replaced them."""
        try:
            import onnx
        except ImportError:
            return

        if self.model is None or not self.built:
            raise ValueError("Model is not built. Call build() before export_onnx().")
        onnx_model = onnx.load(str(path))
        graph = onnx_model.graph
        expected_inputs = [
            (tensor.name, tuple(int(axis) for axis in tensor.shape[1:] if axis))
            for tensor in self.model.inputs
        ]
        expected_outputs = [
            (node.name, tuple(node.shape.tuple)) for node in self.train_outputs
        ]
        rename = {}

        # Only an exporter that dropped the names has to be corrected. One that
        # kept them may still have reordered them - both tf2onnx and the Torch
        # exporter sort their outputs - and pairing those off by position would
        # rename each tensor after a different one, silently swapping two
        # outputs' values.
        for values, expected in (
            (graph.input, expected_inputs),
            (graph.output, expected_outputs),
        ):
            names = [value.name for value in values]
            if len(names) != len(expected) or set(names) == {
                name for name, _ in expected
            }:
                continue
            rename.update(_pair_onnx_names(values, expected))
        if not rename:
            return

        for collection in (
            graph.input,
            graph.output,
            graph.value_info,
            graph.initializer,
        ):
            for value in collection:
                value.name = rename.get(value.name, value.name)
        for node in graph.node:
            for idx, name in enumerate(node.input):
                node.input[idx] = rename.get(name, name)
            for idx, name in enumerate(node.output):
                node.output[idx] = rename.get(name, name)

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(path))

    @staticmethod
    def validate_onnx(
        filename: str | os.PathLike,
        inputs: dict,
        *,
        return_dict: bool = False,
        providers: list[str] | None = None,
    ):
        """Run an exported ONNX model and return its outputs."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "validate_onnx() requires the optional 'onnxruntime' package."
            ) from exc

        path = Path(filename)
        if path.suffix.lower() != ".onnx":
            path = path.with_suffix(".onnx")
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        session_kwargs = {} if providers is None else {"providers": providers}
        session = ort.InferenceSession(str(path), **session_kwargs)  # type: ignore
        input_names = [item.name for item in session.get_inputs()]
        missing = [name for name in input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing ONNX inputs: {missing}")

        feed = {
            name: np.asarray(inputs[name], dtype=np.float32) for name in input_names
        }
        outputs = session.run(None, feed)
        if return_dict:
            return {
                item.name: value for item, value in zip(session.get_outputs(), outputs)
            }
        return outputs


class ModelCall(Node):
    inputs_map: dict[Node, Node]
    outputs_map: dict[Node, Node]

    def __init__(self, name: str, model: Modely):
        super().__init__(name=name)
        self.model = model
        self.inputs_map = {}
        self.outputs_map = {}


class IntermediateOutput(Output):
    pred: ModelCall

    def __init__(self, out: Stream, model_call: ModelCall) -> None:
        super().__init__(name=out.name, stream=out)
        self.pred = model_call
        self.preds = [model_call]
