import os
from pathlib import Path

from typing import Any, Callable

from nnodely.core.dag import toposort, flatten, _flatten_graph
from nnodely.core.validation import ValidationResult, score_signal
from nnodely.utils.utils import _resolve_loss, _resolve_optimizer
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


class Modely:
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
        """True se build() è stato chiamato, False altrimenti."""
        return self.model is not None

    def build(self):
        from nnodely.core.layer import Identity

        extra_outputs = []
        for minimizer in self.minimizers:
            if not isinstance(minimizer["source"], Output):
                if isinstance(minimizer["source"], Input):
                    extra_outputs.append(Identity()(minimizer["source"]))
                else:
                    extra_outputs.append(minimizer["source"])
            if not isinstance(minimizer["target"], Output):
                if isinstance(minimizer["target"], Input):
                    extra_outputs.append(Identity()(minimizer["target"]))
                else:
                    extra_outputs.append(minimizer["target"])

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

        body_model = keras.Model(
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
            self.model = keras.Model(
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
        source: Output | Stream,
        target: Output | Stream | float | None = None,
        loss: str | dict[str, Any] | keras.losses.Loss | Callable = "mse",
    ):
        """Register a Keras loss to minimize during training.

        name: identifier for this loss (used as an output name in the training model)
        source: node/stream producing predictions (e.g., an Output node)
        target: node/stream providing target values (usually derived from an Input)
        loss: Keras loss name, serialized config, Loss instance, or callable
        """
        resolved_loss = _resolve_loss(loss)
        if target is None:  ## Transform it into a Constant with value zero
            target = Constant(name=None, value=0.0)
        if isinstance(target, float):
            target = Constant(name=None, value=[target])
        self.minimizers.append(
            {
                "name": name,
                "source": source,
                "target": target,
                "loss": resolved_loss,
            }
        )
        return self

    def remove_minimizer(self, name: str):
        """Remove a registered minimizer by name."""
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

        chunks: dict[str, list[np.ndarray]] = {}
        for start in range(0, n_samples, _INFERENCE_CHUNK):
            stop = min(start + _INFERENCE_CHUNK, n_samples)
            batch = {name: x_data[name][start:stop, ...] for name in input_names}
            for name, value in self.model(batch, training=False).items():
                chunks.setdefault(name, []).append(keras.ops.convert_to_numpy(value))  # type: ignore

        return {name: np.concatenate(values, axis=0) for name, values in chunks.items()}

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
        source = minimizer["source"]
        target = minimizer["target"]
        label = self._dataset_name(target, x_data) or target.name

        if target.name in predictions:
            return np.asarray(predictions[target.name][:n_samples]), label

        if label in x_data:
            return np.asarray(x_data[label][:n_samples]), label

        value = getattr(target, "value_numpy", None)
        if value is None:
            raise ValueError(
                f"Validation target {target.name!r} of minimizer "
                f"{minimizer['name']!r} is neither in the dataset nor a constant."
            )
        shape = (n_samples,) + tuple(source.shape)
        return np.broadcast_to(np.asarray(value, dtype=np.float32), shape), target.name

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
            if source.name not in predictions:
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
                y_true=y_true,
                y_pred=np.asarray(predictions[source.name][:n_samples]),
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

    def _supervised_arrays(self, data: DataLoader) -> tuple[dict, dict]:
        """Split a dataset into the model's inputs and the minimizers' labels."""
        n_samples = len(data)
        x_data = {name: np.asarray(values) for name, values in data.as_dict().items()}

        y_data = {}
        for minimizer in self.minimizers:
            source_name = minimizer["source"].name
            target = minimizer["target"]

            label_name = self._dataset_name(target, x_data)
            if label_name is not None:
                y_data[source_name] = x_data[label_name]
                continue

            target_value = getattr(target, "value_numpy", None)
            if target_value is None:
                raise ValueError(
                    f"Training target '{target.name}' must be present in the dataset or be a constant value."
                )

            target_value = np.asarray(target_value, dtype=np.float32)
            y_shape = (n_samples,) + tuple(minimizer["source"].shape)
            y_data[source_name] = np.broadcast_to(target_value, y_shape).astype(
                np.float32
            )

        return x_data, y_data

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
        per-minimizer loss table of the original nnodely trainer, ``"factory"``
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
        km = self.model

        resolved_optimizer = _resolve_optimizer(optimizer, lr, optimizer_kwargs)
        resolved_losses = {
            minimizer["name"]: _resolve_loss(minimizer["loss"])
            for minimizer in self.minimizers
        }

        x_data, y_data = self._supervised_arrays(train_data)

        validation_data = None
        if val_data is not None:
            if len(val_data) == 0:
                raise ValueError("val_data is empty.")
            validation_data = self._supervised_arrays(val_data)

        compile_losses: dict[str, Any] = {
            name: None for name in getattr(km, "output_names", [])
        }
        for minimizer in self.minimizers:
            compile_losses[minimizer["source"].name] = resolved_losses[
                minimizer["name"]
            ]

        km.compile(
            optimizer=resolved_optimizer, loss=compile_losses, jit_compile="auto"
        )
        history = km.fit(
            x=x_data,
            y=y_data,
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
        ModelSerializer.serialize(self, path)

    @classmethod
    def load(cls, path):
        return ModelSerializer.load(path)

    def export_keras(self, filename: str):
        if self.model is None:
            raise ValueError("Model is not built. Call build() before export_keras().")

        if not isinstance(self.model, keras.Model):
            raise TypeError(f"Expected keras.Model, got {type(self.model)}.")

        self.model.save(filename)

    @staticmethod
    def import_keras(filename: str, safe_mode: bool = True):
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
