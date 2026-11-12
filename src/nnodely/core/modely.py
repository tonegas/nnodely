import os
import warnings

from collections.abc import Mapping
from typing import Any, Callable, Literal

from nnodely.core import export, validation
from nnodely.core.dag import (
    free_name,
    toposort,
    flatten,
    _flatten_graph,
    _undeclared_inputs,
    _parameters,
    _nodes_and_models,
    _nodes_by_model,
    _read_nodes,
    _source_input_tensors,
    NameNotFound,
    Table,
    resolve,
)
from nnodely.core.minimizer import (
    MinimizerModel,
    _check_minimizers_train_weights,
    _minimizer_stream,
    _resolve_minimizer_terms,
    _validate_gain,
)
from nnodely.core.validation import ValidationResult
from nnodely.utils.utils import (
    _loss_key,
    _resolve_early_stopping,
    _resolve_loss,
    _resolve_optimizer,
)
from nnodely.utils.printers import _resolve_printer, _table
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
        self._order = toposort(outputs)
        # Read as an output, or bound to a Loop, it passed no layer call.
        _refuse_uncalled(self._order)
        # One entry per Input: two of one name are listed twice.
        undeclared = sorted(node.name for node in _undeclared_inputs(inputs, outputs))
        if undeclared:
            raise ValueError(
                f"Model {name!r} reads the Inputs {undeclared}, which are not "
                "among its inputs: declare them there. A model declares every "
                "Input it reads, the init of an Integrate, a Derivative or a "
                "Buffer too; only a fed-back Input, or one only a minimizer "
                "reads, such as a target, may be left out."
            )
        self.model = None  # Keras model build from this DAG
        self.train_inputs = []  # List of Input nodes that are required for training (derived from minimizers)
        self._train_outputs = []  # List of Output nodes that are required for training (derived from minimizers)
        self.inference_model = (
            None  # Keras model of the declared outputs only, without minimizers
        )
        self.inference_inputs = []  # List of Input nodes the declared outputs read (no minimizer-only input)
        # Inputs the data may leave out: they read zeros.
        self._optional_inputs: set[Input] = set()
        # Set by build(): the key every Input and output has in the Keras
        # models, its name then, by the id of the object; the Input of the
        # graph each copy the build made stands for, by the id of the copy.
        self._keys: dict[int, str] = {}
        self._originals: dict[int, Input] = {}
        self._minimizers = []  # List of dicts with keys: 'source', 'target', 'loss', 'name'
        self._minimizer_outputs: dict[Node, str] = {}  # minimizer side -> graph output

    def __repr__(self) -> str:
        """The model as a table: its name, then the names of every Input it
        reads, as :attr:`inputs` lists them, of its outputs and of its
        minimizers."""
        rows = [
            ("inputs", ", ".join(node.name for node in self.inputs) or "-"),
            ("outputs", ", ".join(node.name for node in self._outputs) or "-"),
            ("minimizers", ", ".join(m["name"] for m in self._minimizers) or "-"),
        ]
        return _table(("model", self.name), [("", rows)])

    def __call__(self, inputs) -> Any:
        """Run the model on arrays, or use it as a block on streams.

        Either way ``inputs`` is a dict keyed by Input or input name, a list
        with one value per input, in the order of ``inputs``, or the value
        alone of a model with one input. Every key is resolved to its Input
        at once, as ``model.inputs[name]`` resolves it: a name several Inputs
        carry raises, and so does an Input the model does not read. Keyed by
        the Inputs themselves, two of one name are told apart. A key that
        names no input the model reads - a target, or another column - is
        ignored.
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
            given = self._given_by_input(inputs)
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
            if all(isinstance(value, Node) for value in values):
                # Paired by position, not by name: two inputs may share one.
                return self._call_stream(values)
            given = dict(zip(self._inputs, values))
        else:
            raise TypeError(
                f"Model {self.name!r} is called with a dict keyed by Input or "
                "input name, with a list of streams or arrays, one per input "
                "in their order, or with the stream or array of its one input; "
                f"got {type(inputs).__name__}."
            )
        streams = [isinstance(value, Node) for value in values]
        if all(streams):
            # A declared input left out would be read from the data of the
            # graph above, as an input of its own.
            missing = [node.name for node in self._inputs if node not in given]
            if missing:
                names = [node.name for node in self._inputs]
                raise ValueError(
                    f"Model {self.name!r} has {len(names)} inputs {names}: used "
                    f"as a block it takes a stream for each, and {missing} have "
                    "none."
                )
            return self._call_stream([given[node] for node in self._inputs])
        if any(streams):
            raise TypeError(
                f"Model {self.name!r} is called with streams and arrays together: "
                "streams make it a block of a larger graph, arrays run it. Give "
                "it one or the other."
            )
        return self._call_tensor(given)

    def _input_of(self, key, known: list | None = None) -> Input:
        """The Input a key the user gives stands for - the Input itself, or
        its name, resolved as ``model.inputs[name]`` resolves it: a name
        several Inputs carry raises, listing them, and a name none carries, or
        an Input the model does not read, raises :class:`NameNotFound`.
        ``known`` is ``self.inputs._candidates()``, when the caller holds it."""
        return resolve(
            key,
            self.inputs._candidates() if known is None else known,
            kind="Input",
            owner=f"Model {self.name!r}",
            hint="Key it by the Input itself, model.inputs[i], or give it a "
            "name of its own, model.inputs[i].name = 'new_name'.",
            objects=Input,
            missing=lambda key: (
                f"Model {self.name!r} has no Input named {key!r}; its Inputs are "
                f"{[node.name for node in self.inputs]}."
                if isinstance(key, str)
                else f"Model {self.name!r} does not read the Input {key.name!r} "
                "it is keyed by: key it by one of its own, model.inputs[i]."
            ),
        )

    def _given_by_input(self, inputs: Mapping) -> dict[Input, Any]:
        """``{Input: value}`` from a dict keyed by Input or input name, every
        key resolved by :meth:`_input_of`: a name no Input carries is ignored,
        as another column of the data."""
        known = self.inputs._candidates()
        given: dict[Input, Any] = {}
        for key, value in inputs.items():
            if not isinstance(key, (str, Input)):
                continue  # no name and no Input: ignored, as a name none carries
            try:
                node = self._input_of(key, known)
            except NameNotFound:
                if isinstance(key, str):
                    continue  # a name no Input of the model carries: ignored
                raise
            if node in given:
                raise ValueError(
                    f"Model {self.name!r} is given two values for its input "
                    f"{node.name!r}: key it by the Input or by its name, once."
                )
            given[node] = value
        return given

    def _call_stream(self, arguments: list) -> Any:
        """The model as a block of a larger graph: every declared input reads
        the stream given for it, ``arguments`` in their order, and the outputs
        are returned as streams."""
        for node, stream in zip(self._inputs, arguments):
            self._check_block_input(node, stream)
        mc = ModelCall(self.name, self)
        mc._preds = arguments
        mc._inputs_map = {old: new for old, new in zip(self._inputs, arguments)}
        outputs = [IntermediateOutput(output, mc) for output in self._outputs]
        mc._outputs_map = {new: old for old, new in zip(self._outputs, outputs)}
        return outputs[0] if len(outputs) == 1 else outputs

    def _call_tensor(self, inputs: dict[Input, Any]) -> Any:
        """The model run on arrays, ``{Input: array}``: its declared
        outputs, without minimizers, ``{output name: tensor}``.

        Every Input is handed to Keras under the key it was built with, and
        every output returned under its name now: an Input or an output
        renamed since the build is still found, and two Inputs renamed alike
        are still two."""
        # The arrays come batched, as a DataLoader lays them out; an input only
        # a minimizer reads, such as a target, is neither needed nor passed on.
        if self.inference_model is None:
            raise ValueError("Model is not built. Call build() before calling it.")
        read = [(self._original(node), node.name) for node in self.inference_inputs]
        given = [(node, key) for node, key in read if node in inputs]
        left_out = [(node, key) for node, key in read if node not in inputs]
        missing = [
            node.name
            for node, _ in left_out
            if node not in self._optional_inputs or not given
        ]
        if missing:
            raise ValueError(
                f"Model {self.name!r} reads the inputs {missing}, which are "
                "missing from the data it was called with."
            )
        # An optional input left out reads zeros, one step of a dynamic
        # sequence: a ModelyLoop starts from the first step of it alone.
        batch = np.shape(inputs[given[0][0]])[0]
        arrays = {key: inputs[node] for node, key in given}
        for node, key in left_out:
            arrays[key] = np.zeros(
                (batch, *(1 if size is None else size for size in node.shape.tuple)),
                np.float32,
            )
        results = self.inference_model(arrays)
        names = [output.name for output in self._outputs]
        twins = sorted({name for name in names if names.count(name) > 1})
        if twins:
            raise ValueError(
                f"Model {self.name!r} has several outputs named {twins}, renamed "
                "since it was built: the results are keyed by output name, so "
                "give each output a name of its own."
            )
        return {
            output.name: results[self._keys[id(output)]] for output in self._outputs
        }

    def _original(self, node: Input) -> Input:
        """The Input of the graph ``node``, a copy :meth:`build` made, stands
        for: ``node`` itself if it is none."""
        return self._originals.get(id(node), node)

    def _key(self, node: Node) -> str:
        """The key ``node`` - an Input it reads, or one of its outputs - has
        in the Keras models of the last :meth:`build`: its name then."""
        return self._keys.get(id(node)) or node.name

    def _check_inputs_apart(self, nodes) -> None:
        """Refuse to read data for the Inputs ``nodes``, those a build read,
        while several carry one name: the data is keyed by input name, so
        they would read one array. Renamed alike after the build, or loaded
        so, they are still two Inputs, which a rename tells apart again."""
        names = [self._original(node).name for node in nodes]
        twins = sorted({name for name in names if names.count(name) > 1})
        if twins:
            raise ValueError(
                f"Model {self.name!r} reads several Inputs named {twins}: the "
                "data is keyed by input name, so give one of them a name of "
                "its own, model.inputs[i].name = 'new_name' - then build "
                "again to train with it."
            )

    def _check_names_built(self) -> None:
        """Refuse to train, validate or export while an Input or an output
        has another name than the one it was built with: the data and the
        Keras models are keyed by those names."""
        self._check_inputs_apart(self.train_inputs)
        renamed = [
            f"{key!r} is now {node.name!r}"
            for node, key in [
                *((self._original(node), node.name) for node in self.train_inputs),
                *((node, self._key(node)) for node in self._outputs),
            ]
            if node.name != key
        ]
        if renamed:
            raise ValueError(
                f"Model {self.name!r} was built before some of its names "
                f"changed - {', '.join(renamed)}: build it again to use them."
            )

    def _check_block_input(self, node: Input, stream) -> None:
        """Refuse a stream that this model, used as a block, cannot read as
        its input ``node``: a block keeps the shapes it was declared with, so
        the stream must have the shape of ``node``. A dynamic sequence axis
        matches any length.
        """
        if node not in self._order:
            return  # an input this model does not read
        if (
            stream.tw != node.tw
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

        _rename_hidden_states(self)
        _refresh_keras_inputs(self)
        # A minimizer reads both of its sides from the forward pass, so every
        # source and target has to be an output of the graph. An Input is not
        # the value of any layer, so it is exposed through an Identity.
        # Everything is computed aside and checked, and taken by the model
        # only once its Keras models are made.
        # The training model returns them by a key of their own: the name of
        # a declared output for one, else its name when no other output has
        # it - two calls of one block return streams of one name.
        extra_outputs, extra_keys = [], []
        minimizer_outputs: dict[Node, str] = {}
        keys = {node.name for node in self._outputs}
        for minimizer in self._minimizers:
            for node in (minimizer["source"], minimizer["target"]):
                if node in minimizer_outputs:
                    continue
                exposed = Identity()([node]) if isinstance(node, Input) else node
                if exposed in self._outputs:
                    minimizer_outputs[node] = exposed.name
                    continue
                key = free_name(exposed.name, keys)
                keys.add(key)
                extra_outputs.append(exposed)
                extra_keys.append(key)
                minimizer_outputs[node] = key

        train_outputs = self._outputs + extra_outputs
        output_keys = [node.name for node in self._outputs] + extra_keys
        flat_outputs, flatten_memo = _flatten_graph(
            self._inputs, train_outputs, one_step=True
        )
        flat_order = toposort(flat_outputs)
        # A minimizer side, too, is computed by a layer called on streams.
        _refuse_uncalled(flat_order)
        train_inputs = [node for node in flat_order if isinstance(node, Input)]
        if not train_inputs:
            # Keras would only say that its outputs are not KerasTensors.
            raise ValueError(
                f"Model {self.name!r} reads no Input. A model is evaluated on "
                "data, and its Inputs are what the data feeds - the batch too: "
                "a graph of Parameters, Constants and numbers alone, such as a "
                "Loop seeded and driven by numbers only, has nothing to run on. "
                "Read at least one Input."
            )
        # The Input of the graph each copy stands for: the copy shares its
        # Keras input, which a hidden state read as its init does not.
        originals = {
            id(copied): node
            for node, copied in flatten_memo.items()
            if isinstance(node, Input)
            and isinstance(copied, Input)
            and copied._input is node._input
        }
        # The data is keyed by input name: two Inputs of one name would read
        # the same array, whatever their shapes.
        names = [node.name for node in train_inputs]
        twins = [name for name in names if names.count(name) > 1]
        if twins:
            raise ValueError(
                f"Model {self.name!r} reads two different Inputs named "
                f"{twins[0]!r}. The data is fed by input name, so every "
                "Input of a model needs a name of its own."
            )
        # A fed-back Input the model declares among its inputs is fed from
        # data on purpose; any other is a feedback a loop was meant to apply.
        declared = {id(node) for node in self._inputs}
        open_feedback = sorted(
            {
                node.name
                for node in train_inputs
                if node.update is not None
                and id(originals.get(id(node), node)) not in declared
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

        keras_inputs, keras_outputs = self._resolve_graph(
            flat_order, train_inputs, list(zip(output_keys, flat_outputs))
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
            id(source)
            for tensor in inference_outputs.values()
            for source in _source_input_tensors(tensor)
        }
        inference_inputs = [node for node in train_inputs if id(node._input) in read]
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
        self._originals = originals
        self._keys = {
            **{id(originals.get(id(node), node)): node.name for node in train_inputs},
            **{id(node): node.name for node in self._outputs},
        }
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

    def _resolve_graph(self, order, inputs, keyed_outputs):
        # Applying one layer several times yields one node per application, so
        # tensors are keyed by node identity: keying them by name made each
        # application overwrite the previous one. The concrete Keras layer is
        # keyed by the layer object the node applies, its source: applications
        # of one layer share its weights, the way repeated calls do in Keras,
        # and layers created apart never do - even under one name, as a model
        # and its reloaded copy have.
        tensor_map: dict[Node, Any] = {
            node: node._input for node in order if isinstance(node, Input)
        }
        layers_by_source: dict[int, Any] = {}
        keras_names: dict[str, Any] = {
            node._input.name: Input for node in order if isinstance(node, Input)
        }

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

        # The declared outputs are addressed by name, so two different nodes
        # claiming one name would silently drop one of them. Calling a model
        # several times produces exactly that: its outputs all carry the
        # body's output names. The keys of the others are their own.
        keras_outputs = {}
        claimed: dict[str, Node] = {}
        for key, node in keyed_outputs:
            owner = claimed.get(key)
            if owner is not None and owner is not node:
                raise ValueError(
                    f"Model {self.name!r} has two different outputs named "
                    f"{key!r}. Calling one model several times returns outputs "
                    "that share its output names, so wrap them in Output nodes with "
                    "distinct names before exposing them."
                )
            claimed[key] = node
            keras_outputs[key] = tensor_map[node]
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

        A name identifies one minimizer: registering a name again replaces the
        minimizer it names, in its place, and warns that it did.
        """
        self._check_minimizer_name(
            name, [m for m in self._minimizers if m["name"] != name]
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
        minimizer = {
            "name": name,
            "source": source,
            "target": target,
            "loss": resolved_loss,
            "gain": gain,
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

    def remove_minimizer(self, name: str | int):
        """Remove a registered minimizer by name, or by its index in
        :attr:`minimizers`; an unknown name only warns."""
        try:
            picked = self.minimizers[name]
        except NameNotFound:
            warnings.warn(
                f"Model {self.name!r} has no minimizer named {name!r}: nothing "
                "was removed.",
                UserWarning,
                stacklevel=2,
            )
            return
        rows = picked if isinstance(picked, list) else [picked]
        removed = {id(row._minimizer) for row in rows}
        self._minimizers = [m for m in self._minimizers if id(m) not in removed]

    def _check_minimizer_name(self, name, others: list[dict]) -> None:
        """Refuse ``name`` for a minimizer beside the minimizers ``others``."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"A minimizer name is a non-empty string, got {name!r}.")
        # Its loss is logged under a key of its name, loss_<name>, which no
        # other name and no total of Keras can have: only the name is checked.
        if any(other["name"] == name for other in others):
            raise ValueError(
                f"Model {self.name!r} has a minimizer named {name!r} already: "
                "each minimizer has a name of its own, the key of its loss."
            )

    def _set_minimizer(self, minimizer: dict, key: str, value):
        """Set the name, loss or gain - ``key`` - of ``minimizer``, one of
        the model's, checked as :meth:`minimize` checks it."""
        if key == "name":
            self._check_minimizer_name(
                value, [m for m in self._minimizers if m is not minimizer]
            )
        elif key == "gain":
            value = _validate_gain(value, minimizer["name"])
        else:
            value = _resolve_loss(value)
        minimizer[key] = value

    # -------------------------------------------------------------------------
    # Parameters and freezing
    # -------------------------------------------------------------------------

    @property
    def parameters(self) -> Parameters:
        """Every Parameter the model computes with: those of its relations and
        the weights of its layers, of the models it calls and of the bodies of
        its Loops, in the order the graph reaches them. A
        list of the Parameter objects, printed as a table of their names,
        shapes, numbers of values and whether training changes them: a
        section per model, under its name, when the models it computes with
        have Parameters, each Parameter under the first model whose graph
        reaches it. The last row totals the values of all the Parameters and,
        under trainable, those training changes."""
        return Parameters(
            [
                (model.name, [node for node in nodes if isinstance(node, Parameter)])
                for model, nodes in _nodes_by_model(self)
            ],
            title=f"Parameters of {self.name}",
        )

    @property
    def models(self) -> "Models":
        """Every model this one computes with, at every depth: the models it
        calls as blocks, the bodies of its Loops, and theirs, in the order
        the graph reaches them, each once however many times it is called. A
        list of the Modely objects themselves, not copies: they share their
        weights with this model, so training, freezing or building one again
        changes this model too. Picked by name, or by position as in a list,
        and printed as a table of their inputs, outputs and Parameters.

        Each is saved in a folder of its own, so a loaded model lists the same
        ones. A name several of them carry is refused: pick one by its
        index, the index column of the table, ``models[1]``."""
        holders: dict[int, Modely] = {}
        models = [model for model, _ in _nodes_by_model(self, holders)][1:]
        return Models(
            models,
            title=f"Models of {self.name}",
            where=[f"in {holders[id(model)].name!r}" for model in models],
        )

    @property
    def inputs(self) -> Inputs:
        """Every Input the model reads - those it declares, those only a
        minimizer reads, and the fed-back ones, inside the models it calls
        too - and every Input fed back inside the step of a Loop it rolls out,
        as a ModelyLoop does. A list of the Input objects, printed as a table
        of their shapes and of how each is fed:

        - declared: whether the model is declared with it;
        - fed back with: the stream a loop feeds back into the Input at the
          next step; a Modely reads the Input from data instead;
        - data: what the data must hold of it - ``required``, ``optional``
          (zeros where left out), ``target`` (to train and validate) or
          ``-`` (fed back only);
        - starts from: fed back by a Loop, what its first step reads."""
        return Inputs(self._input_rows())

    def _input_rows(self) -> list[dict]:
        """A row of :attr:`inputs` per Input the graph reads, then one per
        state of its Loops that does not start from the data of one."""
        from nnodely.layers.loop import Loop

        # A row per Input object: two Inputs of one name - renamed alike -
        # are two rows, and the model refuses them when built.
        declared = {id(node) for node in self._inputs}
        sides = [
            side
            for minimizer in self._minimizers
            for side in (minimizer["source"], minimizer["target"])
        ]
        # The rows give the Inputs of the graph, not the copies reading it
        # makes: an Input a called model reads is the one to pass on.
        forward_originals: dict[int, Input] = {}
        forward = {
            id(forward_originals.get(id(node), node))
            for node in _read_nodes(self._outputs, self._inputs, forward_originals)
            if isinstance(node, Input)
        }
        originals: dict[int, Input] = {}
        copies: dict[int, Node] = {}
        read = _read_nodes([*self._outputs, *sides], self._inputs, originals, copies)
        rows = []
        by_input: dict[int, dict] = {}
        for node in read:
            if not isinstance(node, Input):
                continue
            original = originals.get(id(node), node)
            if id(original) in by_input:
                continue
            if id(original) not in forward:
                data = "target"
            else:
                data = "optional" if original in self._optional_inputs else "required"
            by_input[id(original)] = {
                "input": original,
                "seq": tuple(node.seq),
                "declared": id(original) in declared,
                "fed back with": None if node.update is None else node.update.name,
                # The stream of the graph it is fed back with.
                "update": None
                if node.update is None
                else copies.get(id(node.update), node.update),
                "data": data,
                "starts from": None,
            }
            rows.append(by_input[id(original)])

        # The states, inside the step: one that starts from an optional Input
        # - the Input of a ModelyLoop it starts from, by object, renamed or
        # not - is that Input, read from data.
        seen: set[int] = set()
        for loop in (node for node in read if isinstance(node, Loop)):
            for state, update, start, rolled in loop._state_starts():
                if id(state) in seen:
                    continue
                seen.add(id(state))
                first = ", first step" if rolled else ""
                # Read here as a copy, too.
                if isinstance(start, Input):
                    start = originals.get(id(start), start)
                row = (
                    by_input.get(id(start))
                    if isinstance(start, Input) and start in self._optional_inputs
                    else None
                )
                if row is not None:
                    starts = f"its data{first}"
                    if start in self._optional_inputs:
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
                row["update"] = loop._step.copied_from(update)
                row["starts from"] = starts
        return rows

    @property
    def outputs(self) -> Outputs:
        """The outputs the model was declared with, in that order. A list of
        the Output objects, printed as a table of their shapes and of:

        - stream: the stream the output names; in a ModelyLoop, the stream of
          the step, collected over the steps;
        - feeds back: the Inputs a loop feeds that stream back into at the
          next step;
        - minimized by: the minimizers that read it, as ``name (source)`` or
          ``name (target)``."""
        return Outputs(self._output_rows())

    def _output_rows(self) -> list[dict]:
        """A row of :attr:`outputs` per declared output."""
        # By the stream of the graph each Input is fed back with: the rows
        # of :attr:`inputs` read copies of it, and give it.
        fed_back: dict[int, list[str]] = {}
        for row in self._input_rows():
            if row.get("update") is not None:
                fed_back.setdefault(id(row["update"]), []).append(row["input"].name)
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
                    "feeds back": fed_back.get(
                        id(stream), fed_back.get(id(output), [])
                    ),
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

    @property
    def minimizers(self) -> "Minimizers":
        """The minimizers of the model, those its :meth:`train` minimizes. A
        list of :class:`Minimizer`, one per minimizer - its name, source,
        target, loss and gain, and the model - printed as a table. Setting
        the loss or gain of one changes it in the model:
        ``model.minimizers[0]["gain"] = 0.1``.

        The minimizers of the models it calls, blocks or Loop bodies, are not
        listed and not trained on: each is minimized by the :meth:`train` of
        its own model."""
        return Minimizers(
            [Minimizer(minimizer, self) for minimizer in self._minimizers]
        )

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
        """The Parameters ``names`` picks, every one for None. A name picks
        among the Parameters and the models the graph reaches; a model stands
        for all its Parameters. A name several carry raises, listing them.
        Each name is checked before any of them is used, so a wrong one
        changes nothing."""
        holders: dict[int, Modely] = {}
        groups = _nodes_by_model(self, holders)
        parameters = [
            node for _, nodes in groups for node in nodes if isinstance(node, Parameter)
        ]
        if names is None:
            return parameters
        keys = [names] if isinstance(names, str) else list(names)
        given = [key for key in keys if not isinstance(key, str)]
        if given:
            raise TypeError(
                f"Model {self.name!r} picks Parameters and models by name, a "
                "string or a list of strings - one of several of a name is "
                "frozen by itself, model.parameters[i].freeze(); got "
                f"{type(given[0]).__name__}."
            )
        models = [model for model, _ in groups[1:]]
        # Each with the list it is in, model.parameters or model.models, and
        # its index there.
        candidates = [
            *(
                (parameter, None, f"(parameters[{index}]) in {model.name!r}")
                for index, (model, parameter) in enumerate(
                    (model, node)
                    for model, nodes in groups
                    for node in nodes
                    if isinstance(node, Parameter)
                )
            ),
            *(
                (model, None, f"(models[{index}]) in {holders[id(model)].name!r}")
                for index, model in enumerate(models)
            ),
        ]
        picked: dict[int, Parameter] = {}
        for key in keys:
            item = resolve(
                key,
                candidates,
                kind="Parameter or model",
                kinds="Parameters and models",
                owner=f"Model {self.name!r}",
                hint="Freeze or unfreeze one by itself, model.parameters[i]"
                ".freeze() or model.models[i].freeze(), or give it a name of "
                "its own, obj.name = 'new_name'.",
                missing=lambda key: (
                    f"Model {self.name!r} reaches no Parameter "
                    f"and no model named {key!r}. Its Parameters are "
                    f"{[parameter.name for parameter in parameters]}, its models "
                    f"{list(dict.fromkeys(model.name for model in models))}."
                ),
            )
            for parameter in (
                [item] if isinstance(item, Parameter) else _parameters(item)
            ):
                picked.setdefault(id(parameter), parameter)
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
        if self.model is not None:
            self._check_names_built()
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
        # The data is keyed by the names of the build it was made for: a
        # DataLoader made before the model was built again holds others.
        missing = [node.name for node in self.train_inputs if node.name not in x_data]
        if missing:
            raise ValueError(
                f"Model {self.name!r} reads the inputs {missing}, which the data "
                f"does not hold; it holds {list(x_data)}. A DataLoader is keyed "
                "by the input names of the build it was made for: make it again "
                "once the model is built again."
            )
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
        total, and ``"loss_<minimizer name>"``, each with a ``val_`` version
        when ``val_data`` is given; monitoring one that is not logged raises.
        For example ``early_stopping="val_loss",
        early_stopping_kwargs={"patience": 10, "restore_best_weights": True}``.

        ``printer`` selects how progress is rendered: ``"tiny"`` prints a compact
        summary of the training progress, ``"legacy"`` prints the scrolling
        per-minimizer loss table, drawn as the tables of a model are,
        ``"nnodely"`` drives the animated machine-room console with a chart of
        the total loss, ``None`` prints nothing, and any Keras callback is used
        as given.
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
        self._check_names_built()

        resolved_optimizer = _resolve_optimizer(optimizer, lr, optimizer_kwargs)
        logged = [
            "loss",
            *(_loss_key(minimizer["name"]) for minimizer in self._minimizers),
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

    # -------------------------------------------------------------------------
    # Save and load
    # -------------------------------------------------------------------------

    def save(self, path, weights: bool | Literal["all"] = True):
        """Save the model - its graph, and its weights unless ``weights`` is
        False - to the folder ``path``.

        Its Inputs need names of their own, as they do to build it, which
        saving its weights does if it was not built.

        Every model it calls as a block, or rolls out in a Loop, is saved in
        a folder of its own inside ``path``, named after it, and theirs inside
        theirs: the loaded model calls them again. The weights they compute
        with here are in this model's file; a block built on its own - a
        Loop body always is - saves its own too. With ``weights="all"``
        every block does, built if it was not, keeping the weights it was
        trained with here: each folder is then a trained model. A block
        saving weights of its own needs Inputs of names of their own too.
        Without weights only the architecture is saved, to be trained later
        or elsewhere: loading it initializes the weights as ``build()`` does.
        """
        if weights not in (True, False, "all"):
            raise ValueError(f"weights is True, False or 'all', got {weights!r}.")
        # Loaded, a model with weights of its own is built: its Inputs need
        # names of their own, and its weights a Keras model to write them
        # from - built to write them, not to run: what that would warn about
        # is said once it is built to run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for model in _nodes_and_models(self)[1]:
                # The model, and every block saving weights of its own.
                if model is self or weights == "all" or (weights and model.built):
                    model._check_inputs_apart(model._inputs)
                    if weights and not model.built:
                        model.build()
        ModelSerializer.serialize(self, path, weights=weights)

    @classmethod
    def load(cls, path, weights: bool = True):
        """Load a model saved with :meth:`save`, with its weights unless
        ``weights`` is False or it was saved without them.

        A model loaded without weights is initialized as ``build()`` does. The
        models it calls are loaded from their folders, each once however many
        times it is called, and so is every Parameter, Constant or Input they
        share. The loaded layers are layers of their own: one added to the
        model, or another model loaded from the same file, never shares their
        weights.
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


class Models(Table):
    """The models a model computes with, as :attr:`Modely.models` lists them:
    a list of the Modely objects, printed as a table of their inputs, their
    outputs, the number of Parameters each computes with and whether it is
    built."""

    _kind, _attr = "model", "models"

    def __init__(self, models, title: str = "", where: list[str] | None = None):
        super().__init__(models)
        self._title = title
        # Where each one is, for the listing of a name several carry.
        self._places = where or ["" for _ in models]

    def _where(self, index, item) -> str:
        return self._places[index]

    def __str__(self):
        header = ("model", "inputs", "outputs", "parameters", "built")
        rows = [
            (
                model.name,
                ", ".join(node.name for node in model._inputs) or "-",
                ", ".join(node.name for node in model._outputs) or "-",
                str(len(_parameters(model))),
                "yes" if model.built else "no",
            )
            for model in self
        ]
        return _table(header, [("", rows)], indexed=True)

    __repr__ = __str__


class Minimizer(Mapping):
    """One minimizer, as :attr:`Modely.minimizers` lists it: its name, source,
    target, loss and gain, and ``model``, the model it is of.

    Each is read as a key or an attribute, ``m["name"]`` or ``m.name``.
    Setting its name, loss or gain - ``m.name = "fit_y"``, as any object is
    renamed, or ``m["name"] = "fit_y"`` - changes the minimizer in its model,
    checked as :meth:`Modely.minimize` checks them, from the next
    :meth:`Modely.train`. What it compares is set by :meth:`Modely.minimize`;
    anything else set raises rather than being lost."""

    _SETTABLE = ("name", "loss", "gain")
    # No attribute of its own to set: one set by mistake would be lost.
    __slots__ = ("_minimizer", "_model")

    def __init__(self, minimizer: dict, model: "Modely"):
        # The model's own entry, not a copy: what is set is set there.
        object.__setattr__(self, "_minimizer", minimizer)
        object.__setattr__(self, "_model", model)

    def __getattr__(self, key):
        if key in ("name", "source", "target", "loss", "gain"):
            return self._minimizer[key]
        if key == "model":
            return self._model
        raise AttributeError(f"Minimizer has no attribute {key!r}.")

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, key):
        return self._model if key == "model" else self._minimizer[key]

    def __iter__(self):
        return iter([*self._minimizer, "model"])

    def __len__(self):
        return len(self._minimizer) + 1

    def __repr__(self):
        return repr({**self._minimizer, "model": self._model})

    def __setitem__(self, key, value):
        if key not in self._SETTABLE:
            raise TypeError(
                f"Minimizer {self._minimizer['name']!r}: its name, loss and gain "
                f"can be set, not its {key!r}; declare it again with minimize() "
                "to change what it compares."
            )
        self._model._set_minimizer(self._minimizer, key, value)


class Minimizers(Table):
    """The minimizers of a model, as :attr:`Modely.minimizers` lists them: a
    list of :class:`Minimizer`, printed as a table of the streams each
    compares, its loss and its gain."""

    _HEADER = ("minimizer", "source", "target", "loss", "gain")

    _kind, _attr = "minimizer", "minimizers"

    def __init__(self, minimizers: list[Minimizer]):
        super().__init__(minimizers)

    def __str__(self):
        def cells(row):
            target = row["target"]
            return (
                row["name"],
                row["source"].name,
                # A Constant of one value - a number given as the target -
                # prints as that value.
                str(float(target.value.ravel()[0]))
                if isinstance(target, Constant)
                and target.value is not None
                and target.value.size == 1
                else target.name,
                validation.loss_name(row["loss"]),
                str(row["gain"]),
            )

        return _table(self._HEADER, [("", [cells(row) for row in self])], indexed=True)

    __repr__ = __str__


def _refresh_keras_inputs(model: Modely) -> None:
    """Give every Input the model reads a Keras input of its name now: an
    Input renamed since it was declared - ``x.name = "speed"`` - keeps the
    Keras input of its first name, which the data would be keyed by."""
    for node in _nodes_and_models(model)[0]:
        if isinstance(node, Input) and node._input.name != node.name:
            node._input = keras.Input(shape=node.shape.tuple, name=node.name)


def _rename_hidden_states(model: Modely) -> None:
    """Give a free name to every state a layer keeps in an Input of its own
    - Integrate, Derivative, Buffer - that has the name of another Input the
    model reads: named after the layer and its signal, ``tank_rate``, it can
    be named like an Input of the user's, ``tank_rate`` too. The user never
    names or reads it, so it yields. The data is fed by input name, so the
    model could not be built otherwise."""
    inputs = [node for node in _nodes_and_models(model)[0] if isinstance(node, Input)]
    taken = {node.name: node for node in inputs if not node._hidden}
    for node in inputs:
        if node._hidden and taken.setdefault(node.name, node) is not node:
            node.name = free_name(node.name, taken)
            node._input = keras.Input(shape=node.shape.tuple, name=node.name)
            taken[node.name] = node


def _same_sizes(sizes: tuple, other: tuple) -> bool:
    """Whether two tuples of axis sizes match: as many axes, of the same
    lengths, a dynamic one (None) matching any."""
    return len(sizes) == len(other) and all(
        None in (size, other_size) or size == other_size
        for size, other_size in zip(sizes, other)
    )
