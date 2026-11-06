import os
import warnings

from typing import Any, Callable

from nnodely.core import export, validation
from nnodely.core.dag import (
    toposort,
    flatten,
    _flatten_graph,
    _parameters,
    _parameters_and_models,
    _nodes_and_models,
    _read_inputs,
    _read_nodes,
    _source_input_tensors,
    _source_names,
)
from nnodely.core.minimizer import (
    MinimizerModel,
    _check_minimizers_train_weights,
    _minimizer_stream,
    _resolve_minimizer_terms,
    _validate_gain,
    _validate_seq_weights,
)
from nnodely.core.validation import ValidationResult
from nnodely.utils.utils import (
    _resolve_early_stopping,
    _resolve_loss,
    _resolve_optimizer,
)
from nnodely.utils.printers import _resolve_printer
from nnodely.core.registry import ModelSerializer
from nnodely.core.stream import Stream, Node
from nnodely.layers.parameter import Constant, Parameter, Parameters
from nnodely.core.dataloader import DataLoader

import numpy as np

from nnodely.layers.output import Output, Outputs
from nnodely.layers.input import Input, Inputs
from nnodely.core.layer import Layer, _claim_keras_name, _refuse_uncalled

import keras


class Modely:
    """A model-structured neural network.

    ``inputs`` and ``outputs`` delimit the graph of streams that makes up the
    model. Objectives (:meth:`minimize`) are declared on it, then
    :meth:`build` creates the Keras model and its weights.

    A model is called with a dict keyed by Input or input name, or with a
    list holding one value per input in the order of ``inputs``. Called on
    arrays, every one with its batch axis, a built model runs and returns
    ``{output name: tensor}``; called on streams, it becomes a block of a
    larger graph and returns its outputs as streams.
    """

    name: str
    _inputs: list[Input]
    _outputs: list[Output]
    _order: list[Node]

    def __init__(self, name: str, inputs: list[Input], outputs: list[Output]) -> None:
        self.name = name
        self._inputs = inputs
        self._outputs = outputs
        self._order = toposort(self)
        # Read as an output, or bound to a Loop, it passed no layer call.
        _refuse_uncalled(self._order)
        self.model = None  # Keras model build from this DAG
        self.train_inputs = []  # List of Input nodes that are required for training (derived from minimizers)
        self._train_outputs = []  # List of Output nodes that are required for training (derived from minimizers)
        self.inference_model = (
            None  # Keras model of the declared outputs only, without minimizers
        )
        self.inference_inputs = []  # List of Input nodes the declared outputs read (no minimizer-only input)
        # Inputs the data may leave out: they read zeros.
        self.optional_inputs: set[str] = set()
        self._minimizers = []  # List of dicts with keys: 'source', 'target', 'loss', 'name'
        self._minimizer_outputs: dict[Node, str] = {}  # minimizer side -> graph output

    def __repr__(self) -> str:
        return f"Model {self.name}"

    def __call__(self, inputs) -> Any:
        """Run the model on arrays, or use it as a block on streams.

        Either way ``inputs`` is a dict keyed by Input or input name, a list
        with one value per input, in the order of ``inputs``, or the value
        alone of a model with one input. A key that names no input the model
        reads is ignored.
        """
        if isinstance(inputs, Node) or hasattr(inputs, "shape"):
            # A stream or an array alone: the value of the one input.
            if len(self._inputs) != 1:
                names = [node.name for node in self._inputs]
                raise ValueError(
                    f"Model {self.name!r} has {len(self._inputs)} inputs {names}: "
                    "called with a value alone it has one; give a list, one "
                    "value per input in that order, or a dict."
                )
            inputs = [inputs]
        if isinstance(inputs, dict):
            values = list(inputs.values())
            given: dict[Any, Any] = {}
            for key, value in inputs.items():
                name = key.name if isinstance(key, Input) else key
                if name in given:
                    raise ValueError(
                        f"Model {self.name!r} is given two values for its input "
                        f"{name!r}: key it by the Input or by its name, once."
                    )
                given[name] = value
        elif isinstance(inputs, (list, tuple)):
            values = list(inputs)
            # One value per input, no more and no less: paired off with zip, an
            # input left without one would be read from the data of the graph
            # above, and an extra value dropped.
            if len(values) != len(self._inputs):
                names = [node.name for node in self._inputs]
                raise ValueError(
                    f"Model {self.name!r} has {len(self._inputs)} inputs {names}: "
                    "called with a list it takes one value for each, in that "
                    f"order, got {len(values)}."
                )
            given = {node.name: value for node, value in zip(self._inputs, values)}
        else:
            raise TypeError(
                f"Model {self.name!r} is called with a dict keyed by Input or "
                "input name, with a list of streams or arrays, one per input "
                "in their order, or with the stream or array of its one input; "
                f"got {type(inputs).__name__}."
            )
        streams = [isinstance(value, Node) for value in values]
        if all(streams):
            return self._call_stream(given)
        if any(streams):
            raise TypeError(
                f"Model {self.name!r} is called with streams and arrays together: "
                "streams make it a block of a larger graph, arrays run it. Give "
                "it one or the other."
            )
        return self._call_tensor(given)

    def _call_stream(self, streams: dict[str, Any]) -> Any:
        """The model as a block of a larger graph: every declared input reads
        the stream given for it, and the outputs are returned as streams."""
        names = [node.name for node in self._inputs]
        # A declared input left out would be read from the data of the graph
        # above, as an input of its own.
        missing = [name for name in names if name not in streams]
        if missing:
            raise ValueError(
                f"Model {self.name!r} has {len(names)} inputs {names}: used as a "
                f"block it takes a stream for each, and {missing} have none."
            )
        arguments = [streams[name] for name in names]
        for node, stream in zip(self._inputs, arguments):
            self._check_block_input(node, stream)
        mc = ModelCall(f"{self.name}_call", self)
        mc._preds = arguments
        mc._inputs_map = {old: new for old, new in zip(self._inputs, arguments)}
        outputs = [IntermediateOutput(output, mc) for output in self._outputs]
        mc._outputs_map = {new: old for old, new in zip(self._outputs, outputs)}
        return outputs[0] if len(outputs) == 1 else outputs

    def _call_tensor(self, inputs: dict[str, Any]) -> Any:
        """The model run on arrays, ``{input name: array}``: its declared
        outputs, without minimizers."""
        # The arrays come batched, as a DataLoader lays them out; an input only
        # a minimizer reads, such as a target, is neither needed nor passed on.
        if self.inference_model is None:
            raise ValueError("Model is not built. Call build() before calling it.")
        given = [
            inputs[inp.name] for inp in self.inference_inputs if inp.name in inputs
        ]
        left_out = [inp for inp in self.inference_inputs if inp.name not in inputs]
        missing = [
            inp.name
            for inp in left_out
            if inp.name not in self.optional_inputs or not given
        ]
        if missing:
            raise ValueError(
                f"Model {self.name!r} reads the inputs {missing}, which are "
                "missing from the data it was called with."
            )
        # An optional input left out reads zeros, one step of a dynamic
        # sequence: a ModelyLoop starts from the first step of it alone.
        batch = np.shape(given[0])[0]
        zeros = {
            inp.name: np.zeros(
                (batch, *(1 if size is None else size for size in inp.shape.tuple)),
                np.float32,
            )
            for inp in left_out
        }
        inputs = {
            inp.name: inputs[inp.name] if inp.name in inputs else zeros[inp.name]
            for inp in self.inference_inputs
        }
        return self.inference_model(inputs)

    def _check_block_input(self, node: Input, stream) -> None:
        """Refuse a stream that this model, used as a block, cannot read as
        its input ``node``: a block keeps the shapes it was declared with, so
        the stream must have the shape of ``node``. A dynamic sequence axis
        matches any length.
        """
        if node not in self._order:
            return  # an input this model does not read
        if (
            stream.time != node.time
            or not _same_sizes(stream.seq, node.seq)
            or tuple(stream.dim) != tuple(node.dim)
        ):
            raise ValueError(
                f"Model {self.name!r} reads its input {node.name!r} as "
                f"{node.shape}, so used as a block it takes a stream of that "
                "shape for it - the same dim, sequence and window; "
                f"{stream.name!r} is {stream.shape}. A block keeps "
                "the shapes it was declared with: give it a stream of that "
                "shape, or declare the model on the shape it is used with."
            )

    @property
    def built(self):
        """True once :meth:`build` has been called."""
        return self.model is not None

    def build(self, keep_weights: bool = True):
        """Create the Keras model of the graph, with its weights.

        Objectives and feedback have to be declared before, because they add
        to the graph that is built. The weights the graph already holds - of
        a block built on its own, or of an earlier build of this model - are
        kept; with ``keep_weights=False`` every weight the model computes
        with, in the models it calls and the bodies of its Loops too, is
        given its initial value again, frozen ones included. A build that
        fails leaves the model as it was. Returns the model itself.
        """
        from nnodely.core.layer import Identity

        # A minimizer reads both of its sides from the forward pass, so every
        # source and target has to be an output of the graph. An Input is not
        # the value of any layer, so it is exposed through an Identity.
        # Everything is computed aside and checked, and taken by the model
        # only once its Keras models are made.
        extra_outputs = []
        minimizer_outputs: dict[Node, str] = {}
        for minimizer in self._minimizers:
            for node in (minimizer["source"], minimizer["target"]):
                if node in minimizer_outputs:
                    continue
                exposed = Identity()([node]) if isinstance(node, Input) else node
                if exposed not in self._outputs:
                    extra_outputs.append(exposed)
                minimizer_outputs[node] = exposed.name

        train_outputs = self._outputs + extra_outputs
        flat, flatten_memo = _flatten_graph(
            self.name,
            self._inputs,
            train_outputs,
            return_memo=True,
            one_step=True,
        )
        train_inputs = [node for node in flat._order if isinstance(node, Input)]
        if not train_inputs:
            # Keras would only say that its outputs are not KerasTensors.
            raise ValueError(
                f"Model {self.name!r} reads no Input. A model is evaluated on "
                "data, and its Inputs are what the data feeds - the batch too: "
                "a graph of Parameters, Constants and numbers alone, such as a "
                "Loop seeded and driven by numbers only, has nothing to run on. "
                "Read at least one Input."
            )
        # The data is keyed by input name: two Inputs of one name would read
        # the same array, whatever their shapes.
        named: dict[str, Input] = {}
        for node in train_inputs:
            if named.setdefault(node.name, node) is not node:
                raise ValueError(
                    f"Model {self.name!r} reads two different Inputs named "
                    f"{node.name!r}. The data is fed by input name, so every "
                    "Input of a model needs a name of its own."
                )
        # A fed-back Input the model declares among its inputs is fed from
        # data on purpose; any other is a feedback a loop was meant to apply.
        declared = {node.name for node in self._inputs}
        open_feedback = sorted(
            {
                node.name
                for node in train_inputs
                if node.update is not None and node.name not in declared
            }
        )
        if open_feedback:
            warnings.warn(
                f"Model {self.name!r} reads the fed-back Inputs {open_feedback}, "
                "which only a Loop or a ModelyLoop feeds back: built as a Modely, "
                "they are read from data, one step at a time. Roll the model out "
                "with Loop or ModelyLoop, or declare them among the inputs.",
                UserWarning,
                stacklevel=2,
            )

        flat_graph_outputs = [flatten_memo[node] for node in train_outputs]
        keras_inputs, keras_outputs = self._resolve_graph(
            flat._order, train_inputs, flat_graph_outputs
        )
        model = MinimizerModel(
            name=self.name + "_train",
            inputs=keras_inputs,
            outputs=keras_outputs,
        )

        # Inference sees the model as it was declared: its outputs, and only
        # the inputs those read. What the minimizers add - their sides, and the
        # inputs only they read, such as targets - is there for the loss and
        # stays in the training model. The inputs are read off the Keras graph
        # rather than the DAG, because a Parameter or Constant is wired to an
        # arbitrary input for its batch.
        inference_outputs = {
            node.name: keras_outputs[node.name] for node in self._outputs
        }
        read = {
            name
            for tensor in inference_outputs.values()
            for name in _source_names(_source_input_tensors(tensor))
        }
        inference_inputs = [node for node in train_inputs if node.name in read]
        inference_model = keras.Model(
            name=self.name,
            inputs={node.name: keras_inputs[node.name] for node in inference_inputs},
            outputs=inference_outputs,
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
        self._minimizer_outputs = minimizer_outputs
        self._train_outputs = train_outputs
        self.train_inputs = train_inputs
        self.model = model
        self.inference_inputs = inference_inputs
        self.inference_model = inference_model

        if not keep_weights:
            # In place, so every model computing with them - a block built on
            # its own, a body - starts from them too. Each Keras layer once,
            # however many nodes apply it.
            layers: dict[int, Layer] = {}
            for node in _nodes_and_models(self)[0]:
                if isinstance(node, Layer) and node._layer is not None:
                    layers.setdefault(id(node._layer), node)
            for node in layers.values():
                node._reinitialize()
        return self

    def _resolve_graph(self, order, inputs, output_nodes):
        # Applying one layer several times yields one node per application, so
        # tensors are keyed by node identity: keying them by name made each
        # application overwrite the previous one. The concrete Keras layer is
        # keyed by the layer object the node applies, its source: applications
        # of one layer share its weights, the way repeated calls do in Keras,
        # and layers created apart never do - even under one name, as a model
        # and its reloaded copy have.
        input_tensors = {
            node.name: node._input for node in order if isinstance(node, Input)
        }
        tensor_map: dict[Node, Any] = {
            node: input_tensors[node.name] for node in order if isinstance(node, Input)
        }
        layers_by_source: dict[int, Any] = {}
        keras_names: dict[str, Any] = {name: Input for name in input_tensors}

        for node in [n for n in order if not isinstance(n, Input)]:
            if isinstance(node, Layer):
                shared = layers_by_source.get(id(node._source))
                if shared is not None:
                    node._layer, signature = shared
                    # A layer without weights may be built for its input's
                    # shape, an axis included: applied to another, it is built
                    # again. One with weights stays shared, as its weights are.
                    if not node._layer.weights:
                        node._layer_signature = signature
                if len(node._preds) == 0:  ## Parameters and Constants
                    anchor = next(iter(tensor_map.values()), None)
                    tensor_map[node] = node.call([anchor])
                else:
                    tensor_map[node] = node.call(
                        [tensor_map[pred] for pred in node._preds]
                    )
                layers_by_source[id(node._source)] = (
                    node._layer,
                    node._layer_signature,
                )
                _claim_keras_name(node._layer, keras_names)
            else:  ## Output or other non-Layer node
                tensor = tensor_map[node._preds[0]]
                if isinstance(node._preds[0], Input):
                    # A Keras model exposes what its layers compute, never one
                    # of its own inputs: an Input read out is passed through.
                    identity = keras.layers.Identity(name=f"{node.name}_identity")
                    _claim_keras_name(identity, keras_names)
                    tensor = identity(tensor)
                tensor_map[node] = tensor

        keras_inputs = {node.name: tensor_map[node] for node in inputs}

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

        Parameters
        ----------
        name:
            Identifier for this loss, used as an output name in the training
            model.
        source:
            Node or stream producing the predictions (e.g. an Output node), or
            the name of a stream of the model.
        target:
            Node or stream providing the target values (usually derived from
            an Input), the name of a stream of the model, a number, or None
            for zero. A number becomes a Constant.
        loss:
            Keras loss name, serialized config, Loss instance, or callable.
        gain:
            Weight of this loss in the total training loss, 1 by default:
            0.5 gives it half the importance, 2 twice. Its own logged and
            validated loss stays unweighted, the way Keras logs a
            ``loss_weight``.
        seq_weights:
            One weight per step of the outer axis of the source - the
            rollout of its outermost Loop, its outermost sequence axis, or its
            window without one - such as
            ``np.exp(0.1 * np.arange(N))`` to weigh the last predictions more,
            or a callable of the number of steps returning them, such as
            ``lambda n: np.exp(0.1 * np.arange(n))``, for a sequence whose
            length is dynamic. Normalized to mean one, so only their profile
            matters; the logged and validated loss is the weighted one. None
            weighs every step the same. It suits losses computed element by
            element, as the regression losses are.

        A name identifies one minimizer: registering a name again replaces the
        minimizer it names, in its place, and warns that it did.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"A minimizer name is a non-empty string, got {name!r}.")
        # The validation loss of minimizer x is logged as val_x_loss, which is
        # also the training loss of a minimizer named val_x.
        clash = next(
            (
                other["name"]
                for other in self._minimizers
                if other["name"] in (f"val_{name}", name.removeprefix("val_"))
                and other["name"] != name
            ),
            None,
        )
        if clash is not None:
            raise ValueError(
                f"Minimizers {name!r} and {clash!r} cannot both be declared: the "
                "validation loss of one is logged under the training loss name "
                "of the other."
            )
        source = _minimizer_stream(self, source, "source", name)
        if target is None:  ## Transform it into a Constant with value zero
            target = 0.0
        # Any real number, numpy scalars included; a bool is not a target value.
        if isinstance(target, (int, float, np.integer, np.floating)) and not isinstance(
            target, (bool, np.bool_)
        ):
            target = Constant(name=None, value=[float(target)])
        target = _minimizer_stream(self, target, "target", name)
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
        for index, existing in enumerate(self._minimizers):
            if existing["name"] == name:
                warnings.warn(
                    f"Minimizer {name!r} already exists in model {self.name!r}: "
                    "it is replaced by the new one.",
                    UserWarning,
                    stacklevel=2,
                )
                self._minimizers[index] = minimizer
                return self
        self._minimizers.append(minimizer)
        return self

    def remove_minimizer(self, name: str):
        """Remove a registered minimizer by name; an unknown name only warns."""
        if all(minimizer["name"] != name for minimizer in self._minimizers):
            warnings.warn(
                f"Model {self.name!r} has no minimizer named {name!r}: nothing "
                "was removed.",
                UserWarning,
                stacklevel=2,
            )
            return
        self._minimizers = [m for m in self._minimizers if m["name"] != name]

    # -------------------------------------------------------------------------
    # Parameters and freezing
    # -------------------------------------------------------------------------

    @property
    def parameters(self) -> Parameters:
        """Every Parameter the model computes with: those of its relations and
        the weights of its layers, of the models it calls and of the bodies of
        its Loops, in the order the graph reaches them. A
        list of the Parameter objects, printed as a table of their names,
        shapes, numbers of values and whether training changes them."""
        return Parameters(_parameters(self), title=f"Parameters of {self.name}")

    @property
    def inputs(self) -> Inputs:
        """Every Input the model reads - declared or not, inside the models it
        calls, or only by a minimizer - and every Input fed back inside the
        step of a Loop it rolls out, as a ModelyLoop does. A list of the Input
        objects, printed as a table of their shapes and of how each is fed:
        whether it is declared, the stream it is fed back with, what the data
        must hold of it and, fed back by a Loop, what it starts from."""
        return Inputs(self._input_rows(), title=f"Inputs of {self.name}")

    def _input_rows(self) -> list[dict]:
        """A row of :attr:`inputs` per Input the graph reads, then one per
        state of its Loops that does not start from the data of one."""
        from nnodely.layers.loop import Loop

        declared = {node.name for node in self._inputs}
        sides = [
            side
            for minimizer in self._minimizers
            for side in (minimizer["source"], minimizer["target"])
        ]
        forward = _read_inputs(self._outputs, self._inputs)
        read = _read_nodes([*self._outputs, *sides], self._inputs)
        rows = []
        by_name: dict[str, dict] = {}
        for name, node in {
            node.name: node for node in read if isinstance(node, Input)
        }.items():
            if name not in forward:
                data = "target"
            else:
                data = "optional" if name in self.optional_inputs else "required"
            by_name[name] = {
                "input": node,
                "seq": tuple(node.seq),
                "declared": name in declared,
                "fed back with": None if node.update is None else node.update.name,
                "data": data,
                "starts from": None,
            }
            rows.append(by_name[name])

        # The states, inside the step: one that starts from an Input of its
        # own name is that Input, read from data.
        seen: set[int] = set()
        for loop in (node for node in read if isinstance(node, Loop)):
            for state, update, start, rolled in loop._state_starts():
                if id(state) in seen:
                    continue
                seen.add(id(state))
                first = ", first step" if rolled else ""
                row = (
                    by_name.get(state.name)
                    if isinstance(start, Input) and start.name == state.name
                    else None
                )
                if row is not None:
                    starts = f"its data{first}"
                    if state.name in self.optional_inputs:
                        starts += ", else 0"
                else:
                    row = {
                        "input": state,
                        "seq": tuple(state.seq),
                        "declared": False,
                        "data": None,
                    }
                    rows.append(row)
                    if start is None:
                        starts = "0"
                    elif isinstance(start, Stream):
                        starts = f"{start.name}{first}"
                    else:
                        starts = str(start)
                row["fed back with"] = update.name
                row["starts from"] = starts
        return rows

    @property
    def outputs(self) -> Outputs:
        """The outputs the model was declared with, in that order. A list of
        the Output objects, printed as a table of their shapes, the stream
        each names, the Inputs a loop feeds that stream back into and the
        minimizers that read it."""
        return Outputs(self._output_rows(), title=f"Outputs of {self.name}")

    def _output_rows(self) -> list[dict]:
        """A row of :attr:`outputs` per declared output."""
        # Matched by the name :attr:`inputs` prints: the Inputs fed back
        # inside a Loop, and their streams, are copies.
        fed_back: dict[str, list[str]] = {}
        for row in self._input_rows():
            if row["fed back with"] is not None:
                fed_back.setdefault(row["fed back with"], []).append(row["input"].name)
        minimized: dict[int, list[str]] = {}
        for minimizer in self._minimizers:
            for side in ("source", "target"):
                minimized.setdefault(id(minimizer[side]), []).append(
                    f"{minimizer['name']} ({side})"
                )
        rows = []
        for output in self._outputs:
            stream = self._output_stream(output)
            rows.append(
                {
                    "output": output,
                    "stream": stream.name,
                    "feeds back": fed_back.get(stream.name, []),
                    "minimized by": [
                        *minimized.get(id(output), []),
                        *minimized.get(id(output._preds[0]), []),
                    ],
                }
            )
        return rows

    def _output_stream(self, output: Output) -> Node:
        """The stream ``output`` names."""
        return output._preds[0]

    def freeze(self, names: str | list[str] | None = None):
        """Keep Parameters of the model as they are through training.

        Without ``names``, every one: those of its relations and of the
        weights of its layers, of the models it calls and of the bodies of
        its Loops. ``names``, a name or a list of them, picks some: the name
        of a Parameter, as :attr:`parameters` lists it, freezes that one; the
        name of a model the graph reaches - a block it calls, the body of a
        Loop, the model of a LocalModel - every Parameter of that model.
        :meth:`unfreeze` undoes it; one :meth:`Parameter.freeze` freezes a
        single one. A Parameter shared with another model is frozen for that
        model too. Returns the model."""
        for parameter in self._parameters_named(names):
            parameter.freeze()
        return self

    def unfreeze(self, names: str | list[str] | None = None):
        """Let training change Parameters of the model again: every one, or
        those ``names`` picks, as :meth:`freeze` finds them. Returns the
        model."""
        for parameter in self._parameters_named(names):
            parameter.unfreeze()
        return self

    def _parameters_named(self, names) -> list[Parameter]:
        """The Parameters ``names`` picks, every one for None. Each name is
        checked before any of them is used, so an unknown one changes
        nothing."""
        parameters, models = _parameters_and_models(self)
        if names is None:
            return parameters
        names = [names] if isinstance(names, str) else list(names)
        given = [name for name in names if not isinstance(name, str)]
        if given:
            raise TypeError(
                f"Model {self.name!r} picks Parameters by name, a string or a "
                f"list of strings; got {type(given[0]).__name__}."
            )
        picked: dict[int, Parameter] = {}
        for name in names:
            named = [parameter for parameter in parameters if parameter.name == name]
            called = [model for model in models if model.name == name]
            if named and called:
                raise ValueError(
                    f"Model {self.name!r} reaches a Parameter and a model both "
                    f"named {name!r}: rename one of them to pick it."
                )
            if not (named or called):
                raise ValueError(
                    f"Model {self.name!r} reaches no Parameter and no model "
                    f"named {name!r}. Its Parameters are "
                    f"{[parameter.name for parameter in parameters]}, its "
                    f"models {list(dict.fromkeys(model.name for model in models))}."
                )
            for model in called:
                named += _parameters(model)
            picked.update((id(parameter), parameter) for parameter in named)
        return list(picked.values())

    # -------------------------------------------------------------------------
    # Validation API
    # -------------------------------------------------------------------------

    def validate(
        self,
        val_data,
        out_dir: str | os.PathLike | None = None,
        show: bool = False,
        history: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> ValidationResult:
        """Score the built model on ``val_data`` and draw what it did.

        Every minimizer is evaluated as training evaluates it: the same
        forward pass, the same target and padding mask, and its loss as
        :meth:`train` logs it for ``val_data``. It is reported with that loss
        plus the indicators a mechanical system-identification report is
        read for: RMSE, MAE, peak error, bias, error std, NRMSE, FIT, R2, VAF
        and correlation. They are computed for each channel of the signal on
        its own scale, and over the real points only: the rollout steps padded
        onto shorter simulations are left out.

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
        verbose:
            Print the summary; False keeps the validation silent.

        The whole result is returned, so the numbers can be asserted on or
        logged as well as read.
        """
        return validation.validate(
            self,
            val_data,
            out_dir=out_dir,
            show=show,
            history=history,
            verbose=verbose,
        )

    def _training_arrays(self, data: DataLoader) -> tuple[dict, np.ndarray | None]:
        """The model's inputs and, when simulations were padded, their mask.

        Both sides of every minimizer come from the forward pass, so the only
        label left is the mask that tells real rollout steps from padded ones.
        """
        x_data = {name: np.asarray(values) for name, values in data.as_dict().items()}
        mask = getattr(data, "mask", None)
        return x_data, None if mask is None else np.asarray(mask, dtype=np.float32)

    def train(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
        epochs: int = 10,
        batch_size: int = 1,
        optimizer: str | dict[str, Any] | keras.optimizers.Optimizer | None = None,
        early_stopping: str | dict[str, Any] | keras.callbacks.Callback | None = None,
        lr: float = 1e-3,
        shuffle: bool = True,
        optimizer_kwargs: dict[str, Any] | None = None,
        early_stopping_kwargs: dict[str, Any] | None = None,
        printer: str | keras.callbacks.Callback | None = "legacy",
        jit_compile: str = "auto",
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

        ``early_stopping`` ends training before ``epochs`` once it stops
        improving. Like ``optimizer``, it may be the name of the quantity to
        monitor, completed by ``early_stopping_kwargs`` into a
        ``keras.callbacks.EarlyStopping``; that callback's configuration, as a
        dict of its arguments; or a callback instance - a configured
        ``keras.callbacks.EarlyStopping``, or a callback of your own that sets
        ``self.model.stop_training = True``. The quantities are ``"loss"``, the
        total, and ``"<minimizer name>_loss"``, each with a ``val_`` version
        when ``val_data`` is given; monitoring one that is not logged raises.
        For example ``early_stopping="val_loss",
        early_stopping_kwargs={"patience": 10, "restore_best_weights": True}``.

        ``printer`` selects how progress is rendered: ``"tiny"`` prints a compact
        summary of the training progress, ``"legacy"`` prints the scrolling
        per-minimizer loss table of the original nnodely trainer, ``"nnodely"``
        drives the animated machine-room console, ``None`` prints nothing, and
        any Keras callback is used as given.
        """
        if not self._minimizers:
            raise ValueError("No minimizers defined. Call minimize() before train().")
        for argument, value in (("epochs", epochs), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{argument} is a positive integer, got {value!r}.")
        if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr < 0:
            raise ValueError(f"lr is a non-negative number, got {lr!r}.")
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle is True or False, got {shuffle!r}.")
        if not hasattr(train_data, "as_dict"):
            raise TypeError(
                f"train_data is a DataLoader, got {type(train_data).__name__}: "
                "DataLoader(model, source=...) lays the data out for training."
            )

        n_samples = len(train_data)
        if n_samples == 0:
            raise ValueError("train_data is empty.")

        # Ensure model is built
        if not self.model:
            raise ValueError("Model is not built. Call build() before training.")

        resolved_optimizer = _resolve_optimizer(optimizer, lr, optimizer_kwargs)
        logged = [
            "loss",
            *(f"{minimizer['name']}_loss" for minimizer in self._minimizers),
        ]
        if val_data is not None:
            logged += [f"val_{name}" for name in logged]
        stopper = _resolve_early_stopping(early_stopping, early_stopping_kwargs, logged)
        self.model.set_minimizers(_resolve_minimizer_terms(self))
        _check_minimizers_train_weights(self, self.model)

        x_data, mask = self._training_arrays(train_data)

        validation_data = None
        val_mask = None
        if val_data is not None:
            if len(val_data) == 0:
                raise ValueError("val_data is empty.")
            val_x, val_mask = self._training_arrays(val_data)
            validation_data = val_x if val_mask is None else (val_x, val_mask)

        self.model.compile(optimizer=resolved_optimizer, jit_compile=jit_compile)
        history = self.model.fit(
            x=x_data,
            y=mask,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=0,  # type: ignore
            callbacks=[
                _resolve_printer(printer, epochs, self._minimizers, self.name),
                *([stopper] if stopper is not None else []),
            ],
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
    ) -> None:
        """
        Export this Modely DAG to interactive HTML using vis-network.

        ``out_dir`` is the folder the pages are written to; a path ending in
        ``.html`` names the root page instead and its parent becomes the folder.
        The root page is ``filename``, the model name by default, with the
        ``.html`` suffix added when missing.

        Every block that wraps a model (``ModelCall``, ``Loop``) is
        exported recursively as its own page, linked from the block node and
        annotated with the ports that bind the body to the graph above it.
        """
        from nnodely.utils.html_export import export_html

        export_html(
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
        self,
        out_dir: str | os.PathLike,
        filename: str | None = None,
        *,
        flatten: bool = False,
        include_minimizers: bool = True,
        format: str = "png",
    ) -> None:
        """Draw the model with Graphviz, as images in ``out_dir``.

        ``out_dir`` is the folder the images are written to; a path ending in
        an image suffix (``.png``, ``.svg``, ``.pdf``, ...) names the root image
        instead and its parent becomes the folder. The root image is
        ``filename``, the model name by default, in ``format``.

        A node shows its name, coloured by its role; an arrow, the shape it
        carries. Without ``flatten`` every block that wraps a model
        (``ModelCall``, ``Loop``) is one sub-model node, and its body
        is drawn in an image of its own, ``<filename>_<body name>``,
        recursively. With ``flatten`` the whole model is one image, each called
        sub-model inside a box. The recurrence of a Loop is a ``loop`` arrow,
        minimizers - unless ``include_minimizers`` is False - are loss nodes,
        and every image carries a legend. Needs the Graphviz ``dot`` program;
        without it the DOT source of each image is written instead, with a
        warning.
        """
        from nnodely.utils.graphviz_plot import plot_graphviz

        plot_graphviz(
            model=self,
            out_dir=out_dir,
            filename=filename,
            flatten=flatten,
            include_minimizers=include_minimizers,
            format=format,
        )

    # -------------------------------------------------------------------------
    # Save and load
    # -------------------------------------------------------------------------

    def save(self, path, weights: bool = True):
        """Save the model - its graph, and its weights unless ``weights`` is
        False - to ``path``.

        Without weights only the architecture is saved, to be trained later
        or elsewhere: loading it initializes the weights as ``build()`` does.
        """
        ModelSerializer.serialize(self, path, weights=weights)

    @classmethod
    def load(cls, path, weights: bool = True):
        """Load a model saved with :meth:`save`, with its weights unless
        ``weights`` is False or it was saved without them.

        A model loaded without weights is initialized as ``build()`` does. The
        loaded layers are layers of their own: one added to the model, or
        another model loaded from the same file, never shares their weights.
        """
        return ModelSerializer.load(path, weights=weights)

    def export_keras(
        self, path: str | os.PathLike, filename: str | None = None
    ) -> None:
        """Export the model as declared, without its minimizers, like save().

        The file is written to the folder ``path`` as ``filename``, the model
        name by default, with the ``.keras`` suffix added when missing.
        """
        export.export_keras(self, path, filename)

    @staticmethod
    def import_keras(filename: str, safe_mode: bool = True):
        """Load a ``.keras`` file written by :meth:`export_keras` as a ``keras.Model``."""
        return export.import_keras(filename, safe_mode=safe_mode)

    def export_onnx(
        self,
        path: str | os.PathLike,
        filename: str | None = None,
        *,
        input_signature=None,
        batch_size: int | None = None,
        opset_version: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Export the built inference graph to ONNX.

        The model is exported as declared, without its minimizers: only its
        outputs, and only the inputs they read. The file is written to the
        folder ``path`` as ``filename``, the model name by default, with the
        ``.onnx`` suffix added when missing.

        ONNX export traces the built Keras graph; it does not deserialize
        nnodely layer configurations. An explicit ``input_signature`` is
        recommended when dynamic dimensions must remain fixed at export time.
        ``batch_size`` is the shorthand for the common case: it exports with
        the batch axis fixed to that many samples instead of left dynamic.

        A layer that differentiates a sub-graph - a ``Gradient`` with respect
        to an Input - records the backward pass in the traced graph, and the
        shape arithmetic it introduces has no ONNX equivalent while the batch
        axis is dynamic. Such a model is therefore exported with a batch of
        one unless a signature or ``batch_size`` says otherwise.
        """
        export.export_onnx(
            self,
            path,
            filename,
            input_signature=input_signature,
            batch_size=batch_size,
            opset_version=opset_version,
            verbose=verbose,
        )

    @staticmethod
    def validate_onnx(
        filename: str | os.PathLike,
        inputs: dict,
        *,
        return_dict: bool = False,
        providers: list[str] | None = None,
    ):
        """Run an exported ONNX model and return its outputs."""
        return export.validate_onnx(
            filename, inputs, return_dict=return_dict, providers=providers
        )


class ModelCall(Node):
    _inputs_map: dict[Node, Node]
    _outputs_map: dict[Node, Node]

    def __init__(self, name: str, model: Modely):
        super().__init__(name=name)
        self._model = model
        self._inputs_map = {}
        self._outputs_map = {}


class IntermediateOutput(Output):
    _pred: ModelCall

    def __init__(self, out: Stream, model_call: ModelCall) -> None:
        super().__init__(name=out.name, stream=out)
        self._pred = model_call
        self._preds = [model_call]


def _same_sizes(time: tuple, other: tuple) -> bool:
    """Whether two tuples of axis sizes match: as many axes, of the same
    lengths, a dynamic one (None) matching any."""
    return len(time) == len(other) and all(
        None in (size, other_size) or size == other_size
        for size, other_size in zip(time, other)
    )
