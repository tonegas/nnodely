import os
import warnings

from typing import Any, Callable

from nnodely.core import export, validation
from nnodely.core.dag import (
    toposort,
    flatten,
    _flatten_graph,
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
from nnodely.utils.utils import _resolve_loss, _resolve_optimizer
from nnodely.utils.printers import _resolve_printer
from nnodely.core.registry import ModelSerializer
from nnodely.core.stream import Stream, Node
from nnodely.layers.constant import Constant
from nnodely.core.dataloader import DataLoader

import numpy as np
from typing import cast

from nnodely.layers.output import Output
from nnodely.layers.input import Input
from nnodely.core.layer import Layer, _claim_keras_name

import keras


class Modely:
    """A model-structured neural network.

    ``inputs`` and ``outputs`` delimit the graph of streams that makes up the
    model. Objectives (:meth:`minimize`) and feedback (:meth:`rollback`) are
    declared on it, then :meth:`build` creates the Keras model and its weights.

    A built model is called with ``{input name: array}``, every array with its
    batch axis, and returns ``{output name: tensor}``. Called with a list of
    streams instead, it
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
        self.inference_model = (
            None  # Keras model of the declared outputs only, without minimizers
        )
        self.inference_inputs = []  # List of Input nodes the declared outputs read (no minimizer-only input)
        self.minimizers = []  # List of dicts with keys: 'source', 'target', 'loss', 'name'
        self._minimizer_outputs: dict[Node, str] = {}  # minimizer side -> graph output
        self._roll_callbacks: dict[Input, Stream] = {}
        self._roll_steps: int | None = None
        self._roll_name: str | None = None

    def __repr__(self) -> str:
        items = " \n- ".join(map(str, self.order))
        return f"Model {self.name}:\n - {items}"

    def __call__(self, inputs) -> Any:
        if not isinstance(inputs, dict):
            # block mode: one stream for every declared input, in their order.
            # Pairing them off with zip made a missing stream an input of the
            # graph above, and dropped an extra one.
            if not isinstance(inputs, (list, tuple)) or not all(
                isinstance(v, Node) for v in inputs
            ):
                raise TypeError(
                    f"Model {self.name!r} is called with a dict of arrays to run "
                    "it, or with a list of streams, one per input, to use it as "
                    f"a block; got {type(inputs).__name__}."
                )
            if len(inputs) != len(self.inputs):
                names = [node.name for node in self.inputs]
                raise ValueError(
                    f"Model {self.name!r} has {len(self.inputs)} inputs {names}: "
                    "used as a block it takes one stream for each, in that "
                    f"order, got {len(inputs)}."
                )
            mc = ModelCall(f"{self.name}_call", self)
            mc.preds = list(inputs)
            mc.inputs_map = {old: new for old, new in zip(self.inputs, inputs)}
            outputs = [IntermediateOutput(output, mc) for output in self.outputs]
            mc.outputs_map = {new: old for old, new in zip(self.outputs, outputs)}
            return outputs[0] if len(outputs) == 1 else outputs

        # tensor execution mode: the declared outputs, without minimizers. The
        # arrays come batched, as a DataLoader lays them out; an input only a
        # minimizer reads, such as a target, is neither needed nor passed on.
        if self.inference_model is None:
            raise ValueError("Model is not built. Call build() before calling it.")
        missing = [inp.name for inp in self.inference_inputs if inp.name not in inputs]
        if missing:
            raise ValueError(
                f"Model {self.name!r} reads the inputs {missing}, which are "
                "missing from the data it was called with."
            )
        inputs = {inp.name: inputs[inp.name] for inp in self.inference_inputs}
        return self.inference_model(inputs)

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
        keras_inputs, keras_outputs = self._resolve_graph(
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
            self.model = MinimizerModel(
                name=self.name + "_train",
                inputs=keras_inputs,
                outputs=self._rolled(body_model, self.train_outputs, keras_inputs),
            )
        else:
            self.model = body_model

        # Inference sees the model as it was declared: its outputs, the streams
        # its rollback feeds back, and only the inputs those read. What the
        # minimizers add - their sides, and the inputs only they read, such as
        # targets - is there for the loss and stays in the training model. The
        # inputs are read off the Keras graph rather than the DAG, because a
        # Parameter or Constant is wired to an arbitrary input for its batch.
        inference_outputs = {
            node.name: keras_outputs[node.name]
            for node in self.outputs
            + [stream for stream in feedback_streams if stream not in self.outputs]
        }
        read = {
            name
            for tensor in inference_outputs.values()
            for name in _source_names(_source_input_tensors(tensor))
        }
        self.inference_inputs = [
            node for node in self.train_inputs if node.name in read
        ]
        inference_inputs = {
            node.name: keras_inputs[node.name] for node in self.inference_inputs
        }
        if self._roll_callbacks:
            inference_body = keras.Model(
                name=self.name + "_body",
                inputs=inference_inputs,
                outputs=inference_outputs,
            )
            inference_outputs = self._rolled(
                inference_body, self.outputs, inference_inputs
            )
        self.inference_model = keras.Model(
            name=self.name, inputs=inference_inputs, outputs=inference_outputs
        )
        return self

    def _rolled(self, body_model, output_nodes, keras_inputs):
        """The outputs of ``body_model`` unrolled over the configured rollback."""
        from nnodely.layers.roll import ModelRollImpl

        roll_layer = ModelRollImpl(
            model=body_model,
            callbacks={
                input_node.name: feedback.name
                for input_node, feedback in self._roll_callbacks.items()
            },
            output_names=tuple(node.name for node in output_nodes),
            input_time_axes={
                node.name: node.shape.dim_rank + 1 for node in self._roll_callbacks
            },
            steps=cast(int, self._roll_steps),
            name=self._roll_name or f"{self.name}_roll",
        )
        return roll_layer(keras_inputs)

    def _resolve_graph(self, order, output_nodes=None):
        # Applying one layer several times yields one node per application, so
        # tensors are keyed by node identity: keying them by name made each
        # application overwrite the previous one. The concrete Keras layer is
        # keyed by the layer object the node applies, its source: applications
        # of one layer share its weights, the way repeated calls do in Keras,
        # and layers created apart never do - even under one name, as a model
        # and its reloaded copy have.
        input_tensors = {
            node.name: node.input for node in order if isinstance(node, Input)
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
                    node._layer = shared
                if len(node.preds) == 0:  ## Parameters and Constants
                    anchor = next(iter(tensor_map.values()), None)
                    tensor_map[node] = node.call([anchor])
                else:
                    tensor_map[node] = node.call(
                        [tensor_map[pred] for pred in node.preds]
                    )
                layers_by_source[id(node._source)] = node._layer
                _claim_keras_name(node._layer, keras_names)
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
            One weight per step of the last axis of the source - the rollout
            of a Loop, or the window of a stream - such as
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

    def validate(
        self,
        val_data,
        out_dir: str | os.PathLike | None = None,
        show: bool = False,
        history: dict[str, Any] | None = None,
        verbose: bool = True,
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
    ) -> None:
        """
        Export this Modely DAG to interactive HTML using vis-network.

        ``out_dir`` is the folder the pages are written to; a path ending in
        ``.html`` names the root page instead and its parent becomes the folder.
        The root page is ``filename``, the model name by default, with the
        ``.html`` suffix added when missing.

        Every block that wraps a model (``ModelCall``, ``Loop``, ``Roll``) is
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
        from nnodely.utils.graphviz_plot import plot_graphviz

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

        A layer that differentiates a sub-graph - ``Derivative`` with respect
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
