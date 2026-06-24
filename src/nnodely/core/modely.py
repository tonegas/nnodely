import os
import cloudpickle
import copy

from typing import Any

from nnodely.core.dag import toposort, flatten, _flatten_graph
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

    def __repr__(self) -> str:
        items = " \n- ".join(map(str, self.order))
        return f"Model {self.name}:\n - {items}"

    def __call__(self, inputs: list[Node] | Any) -> Any:
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
        for idx, inp in enumerate(self.train_inputs):
            if isinstance(inputs, dict):
                if len(inputs[inp.name].shape) == inp.rank:
                    if type(inputs[inp.name]) is np.ndarray:
                        inputs[inp.name] = np.expand_dims(inputs[inp.name], axis=0)
                    else:
                        inputs[inp.name] = tf.expand_dims(inputs[inp.name], axis=0)
            else:
                if len(inputs[idx].shape) == inp.rank:
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
        flat = _flatten_graph(self.name, self.inputs, self.train_outputs)
        self.train_inputs = [node for node in flat.order if isinstance(node, Input)]

        keras_inputs, keras_outputs = self.resolve_graph(flat.order)
        self.model = keras.Model(
            name=self.name + "_train",
            inputs=keras_inputs,
            outputs=keras_outputs,
        )
        return self

    def resolve_graph(self, order):
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
                    if isinstance(tensor_map[node.name], tuple) and len(tensor_map[node.name]) > 1:
                        idx = node.name.rfind("_")
                        if idx != -1 and node.name[idx + 1 :].isdigit():
                            tensor_map[node.name] = tensor_map[node.name][int(node.name[idx + 1 :])]
            else:  ## Output or other non-Layer node
                tensor_map[node.name] = tensor_map[node.preds[0].name]

        keras_inputs = {node.name: tensor_map[node.name] for node in self.train_inputs}
        keras_outputs = {
            node.name: tensor_map[node.name] for node in self.train_outputs
        }
        return keras_inputs, keras_outputs

    # -------------------------------------------------------------------------
    # Train API
    # -------------------------------------------------------------------------

    def minimize(
        self,
        name: str,
        source: Output | Stream,
        target: Output | Stream | float | None = None,
        loss="mse",
    ):
        """Register a loss to minimize during training.
        name: identifier for this loss (used as an output name in the training model)
        source: node/stream producing predictions (e.g., an Output node)
        target: node/stream providing target values (usually derived from an Input)
        loss: loss identifier accepted by `keras.losses.get`
        """
        if target is None:  ## Transform it into a Constant with value zero
            target = Constant(name=None, value=0.0)
        if isinstance(target, float):
            target = Constant(name=None, value=[target])
        self.minimizers.append(
            {"name": name, "source": source, "target": target, "loss": loss}
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
        val_data: DataLoader | None = None,
        epochs: int = 1,
        batch_size: int = 1,
        optimizer=None,
        lr: float = 1e-3,
        out_dir: str | None = "validation",
    ):
        if not self.minimizers:
            raise ValueError("No minimizers defined. Call minimize() before train().")

        n_samples = len(train_data)
        if n_samples == 0:
            raise ValueError("train_data is empty.")

        # Ensure model is built
        if not self.model:
            raise ValueError("Model is not built. Call build() before training.")
        km = self.model

        # Default optimizer ## TODO: change with a user defined optimizer
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

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
            def train_step(model, batch_inputs, batch_targets, optimizer, losses, unique_vars):
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

            losses = {m["name"]: keras.losses.get(m["loss"]) for m in self.minimizers}

            train_history = {"loss": []}
            idxs = np.arange(n_samples)
            epoch_bar = tqdm(range(epochs), desc=f"Training {self.name} in tensorflow", unit="epoch")
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
                        km, batch_inputs, batch_targets, optimizer, losses, unique_vars
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

            if optimizer is None or not (
                hasattr(optimizer, "zero_grad") and hasattr(optimizer, "step")
            ):
                optimizer = torch.optim.Adam(unique_params, lr=lr)

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
            epoch_bar = tqdm(range(epochs), desc=f"Training {self.name} in torch", unit="epoch")

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

                    optimizer.zero_grad()
                    preds = km(batch_inputs, training=True)

                    total = torch.zeros((), dtype=torch.float32, device=device)
                    for minimizer in self.minimizers:
                        source_name = minimizer["source"].name
                        total = total + criterion(
                            preds[source_name], batch_targets[source_name].unsqueeze(-1)
                        )
                    total.backward()
                    optimizer.step()

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
            compile_losses[minimizer["source"].name] = keras.losses.get(
                minimizer["loss"]
            )

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
                print(f"Epoch {epoch+1}/{self.params['epochs']}   Elapsed: {elapsed:.1f}s")
                print(sep)

                for k, v in sorted(logs.items()):
                    print(f"{k:<30} {v:>12.3e}")

                print(sep)
        
        km.compile(optimizer=optimizer, loss=compile_losses)
        history = km.fit(
            x=x_data,
            y=y_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose="0",
            callbacks=[FancyLossPrinter()]
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
        closed_loop: dict[str | Input, str | Node],
        initial_values: dict[str | Input, str | Node],
        inputs: list[Input] | None = None,
        name: str | None = None,
    ) -> Layer:
        """
        Create a new Modely that rolls out over the rightmost sequence axis.

        Semantics:
        - The layer unrolls over the rightmost sequence axis of its inputs (axis=-1).
        - Inputs without a sequence axis are broadcast across the horizon.
        - If inputs have multiple seq dimensions (nested loops), only the rightmost is
          iterated by this Loop. Remaining seq dims are passed through to the inner model.
        - The closed-loop mapped input is updated each step with the submodel output.
        """
        from nnodely.layers.loop import Loop

        # Validate closed_loop keys and values
        if len(closed_loop) == 0:
            raise ValueError("closed_loop cannot be empty.")

        if inputs is None:
            inputs = self.inputs
            for idx, inp in enumerate(inputs):
                if inp.name not in closed_loop:
                    inputs[idx] = Input(
                        name=inp.name,
                        dim=inp.dim,
                        # time=inp.time,
                        seq=(None,),
                    )
        else:
            for inp, out in closed_loop.items():
                inp_name = inp.name if isinstance(inp, Input) else str(inp)
                out_name = out.name if isinstance(out, Output) else str(out)
                if inp_name not in [node.name for node in inputs]:
                    raise ValueError(
                        f"Closed-loop input '{inp_name}' not found among model inputs."
                    )
                if out_name not in [node.name for node in self.outputs]:
                    raise ValueError(
                        f"Closed-loop output '{out_name}' not found among model outputs."
                    )
        print(
            f"Creating closed-loop model '{name}' with loop mapping: {closed_loop}, anad inputs: {inputs}"
        )
        if not self.built:
            self.build()

        loop_fn = Loop(
            f=self, closed_loop=closed_loop, initial_values=initial_values, name=name
        )
        return loop_fn

    # -------------------------------------------------------------------------
    # Save and load
    # -------------------------------------------------------------------------
    def save(self, filename: str):
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Save weights from current built model before clearing runtime state
        if self.model is None:
            raise ValueError(
                "Model is not built. Call build() before saving the model and weights."
            )
        self.model.save_weights(filename + ".weights.h5")

        model_copy = copy.deepcopy(self)

        # Clear built Keras models
        model_copy.model = None

        # Clear runtime Keras state from symbolic nodes
        for node in getattr(model_copy, "order", []):
            if hasattr(node, "_layer"):
                node._layer = None

            if hasattr(node, "_state"):
                for key in ("layer", "param", "constant"):
                    if key in node._state:
                        node._state[key] = None

        with open(filename + ".pkl", "wb") as out:
            cloudpickle.dump(model_copy, out)

    @staticmethod
    def load(filename: str):
        with open(filename + ".pkl", "rb") as inp:
            model = cloudpickle.load(inp)

        model.build()

        weights_path = filename + ".weights.h5"
        if os.path.exists(weights_path):
            keras_model = model.model
            keras_model.load_weights(weights_path)

        return model

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

    def export_onnx(self, filename: str):
        if self.model is None:
            raise ValueError("Model is not built. Call build() before export_onnx().")

        self.model.export(filename + ".onnx", format="onnx")

    @staticmethod
    def validate_onnx(filename: str, inputs: dict):
        import onnxruntime as ort

        session = ort.InferenceSession(filename + ".onnx")

        outputs = session.run(
            None,
            {k: np.asarray(v, dtype=np.float32) for k, v in inputs.items()},
        )
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
        self.preds = [self.pred]
