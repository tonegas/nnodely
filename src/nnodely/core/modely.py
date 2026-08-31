import os
from pathlib import Path

from typing import Any, Callable

from nnodely.core.dag import toposort, flatten, _flatten_graph
from nnodely.utils.utils import _resolve_loss, _resolve_optimizer
from nnodely.core.registry import ModelSerializer
from nnodely.core.stream import Stream, Node
from nnodely.layers.constant import Constant
from nnodely.core.dataloader import DataLoader
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import cast

from nnodely.layers.output import Output
from nnodely.layers.input import Input
from nnodely.core.layer import Layer

import keras


class Modely:
    name: str
    inputs: list[Input]
    outputs: list[Output]
    order: list[Node]
    calls: int

    def __init__(self, name: str, inputs: list[Input], outputs: list[Output]) -> None:
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.order = toposort(self)
        self.model = None  # Keras model build from this DAG
        self.train_inputs = []  # List of Input nodes that are required for training (derived from minimizers)
        self.train_outputs = []  # List of Output nodes that are required for training (derived from minimizers)
        self.minimizers = []  # List of dicts with keys: 'source', 'target', 'loss', 'name'
        self._closed_loop_callbacks: dict[Input, Stream] = {}
        self._closed_loop_steps: int | None = None
        self._closed_loop_name: str | None = None

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
                        inputs[inp.name] = tf.expand_dims(inputs[inp.name], axis=0)
            else:
                if len(inputs[idx].shape) == inp.shape.rank:
                    if type(inputs[idx]) is np.ndarray:
                        inputs[idx] = np.expand_dims(inputs[idx], axis=0)
                    else:
                        inputs[idx] = tf.expand_dims(inputs[idx], axis=0)
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
        feedback_streams = list(dict.fromkeys(self._closed_loop_callbacks.values()))
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
        # connected to the concrete Keras layers created from those copies.
        for source_node, flat_node in flatten_memo.items():
            if isinstance(source_node, Layer) and isinstance(flat_node, Layer):
                source_node._layer = flat_node._layer

        body_model = keras.Model(
            name=self.name + "_train",
            inputs=keras_inputs,
            outputs=keras_outputs,
        )
        if self._closed_loop_callbacks:
            from nnodely.layers.loop import ModelClosedLoopImpl

            callbacks = {
                input_node.name: feedback.name
                for input_node, feedback in self._closed_loop_callbacks.items()
            }
            closed_loop_layer = ModelClosedLoopImpl(
                model=body_model,
                callbacks=callbacks,
                output_names=tuple(node.name for node in self.train_outputs),
                input_time_axes={
                    node.name: node.shape.dim_rank + 1
                    for node in self._closed_loop_callbacks
                },
                output_time_axes={
                    node.name: node.shape.dim_rank + 1 for node in self.train_outputs
                },
                steps=cast(int, self._closed_loop_steps),
                name=self._closed_loop_name or f"{self.name}_closed_loop",
            )
            final_outputs = closed_loop_layer(keras_inputs)
            self.model = keras.Model(
                name=self.name + "_train",
                inputs=keras_inputs,
                outputs=final_outputs,
            )
        else:
            self.model = body_model
        return self

    def resolve_graph(self, order, output_nodes=None):
        tensor_map = {}
        for node in [n for n in order if isinstance(n, Input)]:
            tensor_map[node.name] = node.input

        for node in [n for n in order if not isinstance(n, Input)]:
            if isinstance(node, Layer):
                if len(node.preds) == 0:  ## Parameters and Constants
                    anchor = next(iter(tensor_map.values()), None)
                    tensor_map[node.name] = node.call([anchor])
                else:
                    tensor_map[node.name] = node.call(
                        [tensor_map[pred.name] for pred in node.preds]
                    )
            else:  ## Output or other non-Layer node
                tensor_map[node.name] = tensor_map[node.preds[0].name]

        keras_inputs = {node.name: tensor_map[node.name] for node in self.train_inputs}
        output_nodes = self.train_outputs if output_nodes is None else output_nodes
        keras_outputs = {node.name: tensor_map[node.name] for node in output_nodes}
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

    def validate(
        self,
        val_data,
        batch_size: int = 256,
        out_dir: str | None = "validation",
        show: bool = False,
    ):
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
        input_names = [node.name for node in self.train_inputs]

        # ------------------------------------------------------------------
        # Run prediction
        # ------------------------------------------------------------------
        pred_chunks = {}
        if keras.backend.backend() == "torch":
            self.model.eval()  # Set the model to evaluation mode (important for layers like dropout or batchnorm)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            batch_inputs = {
                name: x_data[name][start:end, ...]
                for name in input_names
                if name in x_data
            }

            preds = self.model(batch_inputs, training=False)
            for name, value in preds.items():
                if keras.backend.backend() == "torch":
                    pred_chunks.setdefault(name, []).append(value.detach().cpu())
                else:
                    pred_chunks.setdefault(name, []).append(value.numpy())

        predictions = {
            name: np.concatenate(chunks, axis=0) for name, chunks in pred_chunks.items()
        }

        metrics = {}

        for minimizer in self.minimizers:
            source = minimizer["source"]
            target = minimizer["target"]

            source_name = source.name
            target_name = target.name

            y_pred = predictions[source_name]
            y_true = (
                x_data[target_name]
                if target_name in x_data
                else predictions[target_name]
            )

            n = min(len(y_pred), len(y_true))
            y_pred = y_pred[:n]
            y_true = y_true[:n]

            error = y_pred - y_true

            mse = float(np.mean(error**2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(error)))

            denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = (
                float(1.0 - np.sum(error**2) / denom) if denom > 1e-12 else float("nan")
            )

            metrics[minimizer["name"]] = {
                "source": source_name,
                "target": target_name,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }

            print(f"\nValidation - {minimizer['name']}")
            print(f"  source : {source_name}")
            print(f"  target : {target_name}")
            print(f"  MSE    : {mse:.6e}")
            print(f"  RMSE   : {rmse:.6e}")
            print(f"  MAE    : {mae:.6e}")
            print(f"  R2     : {r2:.6f}")

            # Flatten for simple plotting
            y_true_plot = y_true.reshape(n, -1)[:, 0]
            y_pred_plot = y_pred.reshape(n, -1)[:, 0]

            plt.figure(figsize=(10, 4))
            plt.plot(y_true_plot, label="target")
            plt.plot(y_pred_plot, label="prediction", alpha=0.8)
            plt.title(f"Validation: {minimizer['name']}")
            plt.xlabel("sample")
            plt.ylabel(source_name)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if out_dir is not None:
                filename = os.path.join(out_dir, f"{minimizer['name']}_prediction.png")
                plt.savefig(filename, dpi=160)

            if show:
                plt.show()

            plt.close()

        return {
            "metrics": metrics,
            "predictions": predictions,
        }

    def train(
        self,
        train_data: DataLoader,
        epochs: int = 1,
        batch_size: int = 1,
        optimizer: str | dict[str, Any] | keras.optimizers.Optimizer | None = None,
        lr: float = 1e-3,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        """Train the model with any Keras optimizer.

        ``optimizer`` may be a Keras optimizer name, a serialized Keras
        optimizer configuration, or an optimizer instance. When a name (or
        ``None``) is provided, ``lr`` and ``optimizer_kwargs`` are used to
        construct it. Optimizer instances and serialized configurations retain
        their own learning-rate configuration.
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

        x_data = {
            name: np.asarray(values) for name, values in train_data.as_dict().items()
        }

        def _resolve_label_name(node):
            if node.name in x_data:
                return node.name

            preds = getattr(node, "preds", [])
            if len(preds) == 1:
                return _resolve_label_name(preds[0])

            return None

        y_data = {}
        for minimizer in self.minimizers:
            source_name = minimizer["source"].name
            target = minimizer["target"]

            label_name = _resolve_label_name(target)
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

        backend = keras.backend.backend()

        # Not used for now, but could be useful for future extensions
        if backend == "tensorfloww":

            @tf.function
            def train_step(
                model,
                batch_inputs,
                batch_targets,
                optimizer,
                losses,
                unique_vars,  # type: ignore
            ):
                with tf.GradientTape() as tape:
                    preds = model(batch_inputs, training=True)
                    total = keras.ops.zeros(())
                    for m in self.minimizers:
                        name = m["name"]
                        source_name = m["source"].name

                        y_pred = preds[source_name]
                        y_true = batch_targets[source_name]
                        loss_obj = losses[name]
                        total = total + keras.ops.mean(loss_obj(y_true, y_pred))

                unique_vars: list[tf.Variable] = []
                seen = set()

                for v in list(km.trainable_weights):  # + self._training_params:
                    vid = id(v)
                    if vid not in seen:
                        seen.add(vid)
                        unique_vars.append(cast(tf.Variable, v))

                gradients = tape.gradient(total, unique_vars)  # type:ignore
                grads_and_vars = [
                    (g, v)
                    for g, v in zip(gradients or [], unique_vars)
                    if g is not None
                ]
                if grads_and_vars:
                    optimizer.apply_gradients(grads_and_vars)
                else:
                    print("Warning: No gradients to apply in this step.")
                return total

            unique_vars: list[tf.Variable] = []
            seen = set()

            losses = resolved_losses

            train_history = {"loss": []}
            idxs = np.arange(n_samples)
            epoch_bar = tqdm(
                range(epochs), desc=f"Training {self.name} in tensorflow", unit="epoch"
            )
            for _ in epoch_bar:
                np.random.shuffle(idxs)
                epoch_losses = []
                for start in range(0, n_samples, batch_size):
                    batch_idx = idxs[start : start + batch_size]
                    batch_samples = [train_data[i] for i in batch_idx]
                    batch_inputs = {}
                    for k in train_data.inputs:
                        batch_inputs[k] = tf.convert_to_tensor(
                            np.stack([sample[k] for sample in batch_samples], axis=0)
                        )
                    batch_targets = {}
                    for minimizer in self.minimizers:
                        source_name = minimizer["source"].name
                        batch_targets[source_name] = tf.convert_to_tensor(
                            np.stack(
                                [y_data[source_name][i] for i in batch_idx], axis=0
                            )
                        )

                    total = train_step(
                        km,
                        batch_inputs,
                        batch_targets,
                        resolved_optimizer,
                        losses,
                        unique_vars,
                    )
                    if isinstance(total, tf.Tensor):
                        epoch_losses.append(float(total.numpy()))

                mean_epoch_loss = float(np.mean(epoch_losses))
                train_history["loss"].append(mean_epoch_loss)
                epoch_bar.set_postfix(epoch_loss=f"{mean_epoch_loss:.3e}")

            return train_history

        if backend == "torchh":
            import torch

            unique_params = []
            seen = set()
            for v in list(km.parameters()):  # + self._training_params:
                vid = id(v)
                if vid not in seen:
                    seen.add(vid)
                    unique_params.append(v)

            native_optimizer: Any = resolved_optimizer
            if not (
                hasattr(native_optimizer, "zero_grad")
                and hasattr(native_optimizer, "step")
            ):
                native_optimizer = torch.optim.Adam(unique_params, lr=lr)

            criterion = torch.nn.MSELoss()
            device = unique_params[0].device if unique_params else torch.device("cpu")
            torch_x_data = {
                name: torch.as_tensor(values, dtype=torch.float32, device=device)
                for name, values in x_data.items()
            }
            torch_y_data = {
                name: torch.as_tensor(values, dtype=torch.float32, device=device)
                for name, values in y_data.items()
            }
            train_history = {"loss": []}
            idxs = np.arange(n_samples)
            epoch_bar = tqdm(
                range(epochs), desc=f"Training {self.name} in torch", unit="epoch"
            )

            km.train()
            for _ in epoch_bar:
                np.random.shuffle(idxs)
                epoch_losses = []

                for start in range(0, n_samples, batch_size):
                    batch_idx = idxs[start : start + batch_size]

                    batch_inputs = {
                        name: tensor[batch_idx] for name, tensor in torch_x_data.items()
                    }
                    batch_targets = {
                        minimizer["source"].name: torch_y_data[
                            minimizer["source"].name
                        ][batch_idx]
                        for minimizer in self.minimizers
                    }

                    native_optimizer.zero_grad()
                    preds = km(batch_inputs, training=True)

                    total = torch.zeros((), dtype=torch.float32, device=device)
                    for minimizer in self.minimizers:
                        source_name = minimizer["source"].name
                        total = total + criterion(
                            preds[source_name], batch_targets[source_name].unsqueeze(-1)
                        )
                    total.backward()
                    native_optimizer.step()

                    epoch_losses.append(float(total.detach().cpu().item()))

                mean_epoch_loss = float(np.mean(epoch_losses))
                train_history["loss"].append(mean_epoch_loss)
                epoch_bar.set_postfix(epoch_loss=f"{mean_epoch_loss:.3e}")

            km.eval()
            return train_history

        compile_losses: dict[str, Any] = {
            name: None for name in getattr(km, "output_names", [])
        }
        for minimizer in self.minimizers:
            compile_losses[minimizer["source"].name] = resolved_losses[
                minimizer["name"]
            ]

        import time
        import sys

        class FancyLossPrinter(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}

                # Clear terminal
                sys.stdout.write("\033[H\033[J")
                sys.stdout.flush()

                elapsed = time.time() - self.start_time

                sep = "─" * 70

                print(sep)
                print(
                    f"Epoch {epoch + 1}/{self.params['epochs']}   Elapsed: {elapsed:.1f}s"
                )
                print(sep)

                for k, v in sorted(logs.items()):
                    print(f"{k:<30} {v:>12.3e}")

                print(sep)

        km.compile(optimizer=resolved_optimizer, loss=compile_losses)
        history = km.fit(
            x=x_data,
            y=y_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose="0",
            callbacks=[FancyLossPrinter()],
        )  # type: ignore[arg-type]

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

        Supported node types in the new framework:
        - input
        - output
        - ordinary Stream / Layer nodes
        - model_call nodes for composed submodels

        Nested submodels are exported recursively as separate HTML pages.

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
    # Closed-loop
    # -------------------------------------------------------------------------
    def closed_loop(
        self,
        closed_loop: dict[str | Input, str | Stream],
        steps: int,
        name: str | None = None,
    ) -> "Modely":
        """Configure temporal feedback directly on this model.

        Each mapping is ``input: stream``. After every model evaluation, the
        stream's one-step result is appended to the input's temporal window.
        Public outputs contain the values produced at every prediction step.
        """
        if self.built:
            raise ValueError("closed_loop() must be called before build().")
        if not isinstance(closed_loop, dict) or not closed_loop:
            raise ValueError("closed_loop must be a non-empty input: stream mapping.")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("closed_loop steps must be a positive integer.")

        graph_streams = [node for node in self.order if isinstance(node, Stream)]

        def resolve(nodes, value, kind):
            if isinstance(value, str):
                matches = [node for node in nodes if node.name == value]
                if len(matches) != 1:
                    raise ValueError(
                        f"closed_loop {kind} {value!r} was not found uniquely."
                    )
                return matches[0]
            if value not in nodes:
                raise ValueError(
                    f"closed_loop {kind} {getattr(value, 'name', value)!r} "
                    "does not belong to this Modely."
                )
            return value

        callbacks = {}
        for input_ref, stream_ref in closed_loop.items():
            input_node = resolve(self.inputs, input_ref, "input")
            stream = resolve(graph_streams, stream_ref, "stream")
            expected = tuple(input_node.dim) + (1,) + tuple(input_node.seq)
            if stream.shape.tuple != expected:
                raise ValueError(
                    f"closed_loop stream {stream.name!r} must produce one temporal "
                    f"sample with shape {expected}, got {stream.shape.tuple}."
                )
            callbacks[input_node] = stream

        self._closed_loop_callbacks = callbacks
        self._closed_loop_steps = steps
        self._closed_loop_name = name
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
        return keras.models.load_model(
            filename + ".keras",
            safe_mode=safe_mode,
        )

    def export_onnx(
        self,
        filename: str | os.PathLike,
        *,
        input_signature=None,
        opset_version: int | None = None,
        verbose: bool = False,
    ) -> Path:
        """Export the built inference graph to ONNX.

        ONNX export traces the built Keras graph; it does not deserialize
        nnodely layer configurations. An explicit ``input_signature`` is
        recommended when dynamic dimensions must remain fixed at export time.
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

        export_kwargs = {"verbose": verbose}
        if input_signature is not None:
            export_kwargs["input_signature"] = input_signature
        if opset_version is not None:
            export_kwargs["opset_version"] = opset_version  # type: ignore

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
        expected_inputs = [tensor.name for tensor in self.model.inputs]
        expected_outputs = [node.name for node in self.train_outputs]
        rename = {}

        if len(graph.input) == len(expected_inputs):
            rename.update(
                {
                    value.name: expected
                    for value, expected in zip(graph.input, expected_inputs)
                    if value.name != expected
                }
            )
        if len(graph.output) == len(expected_outputs):
            rename.update(
                {
                    value.name: expected
                    for value, expected in zip(graph.output, expected_outputs)
                    if value.name != expected
                }
            )
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
