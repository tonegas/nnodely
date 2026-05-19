import os

from typing import Any

from nnodely.core.dag import toposort, flatten, toposort_outputs
from nnodely.core.stream import Stream, Node
from nnodely.layers.constant import Constant
from nnodely.layers.parameter import Parameter
from nnodely.core.dataloader import DataLoader
from nnodely.layers.time_ops import SampleWindow
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from typing import cast

from nnodely.layers.output import Output
from nnodely.layers.input import Input

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
        self.calls = 0
        self.model = None  # Keras model build from this DAG
        self._train_model = None  # Keras model used for training (includes extra outputs for minimizers)
        self.train_inputs = []  # List of Input nodes that are required for training (derived from minimizers)
        self._training_params = []  # List of trainable parameters (Keras variables) collected from the DAG
        self.minimizers = []  # List of dicts with keys: 'source', 'target', 'loss', 'name'
        # self.default_sampling = 1.0  # default sampling rate for inputs that require it and don't have it specified at init time

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
            # return outputs

        # tensor execution mode
        if self.model is None:
            raise ValueError("Model build failed, model is still None.")
        return self.model(inputs)

    @property
    def built(self):
        """True se build() è stato chiamato, False altrimenti."""
        return self.model is not None

    def build(self):
        """
        Build the Keras Functional model from the symbolic DAG.

        Traverses from model outputs backwards through:
        - Input nodes
        - Layer/Stream nodes
        - Output nodes
        - IntermediateOutput nodes from nested Modely calls
        """
        ## adjust sample windows with the maximum window size for each input
        for node in self.order:
            if isinstance(node, SampleWindow):
                node_pred = node.preds[0]
                if isinstance(node_pred, Input):
                    node.past = node_pred.past - node.past

        tensor_map = {node: node.input for node in self.inputs}

        keras_outputs = {}
        for out in self.outputs:
            keras_outputs[out.name] = self._resolve_tensor(out, tensor_map)

        keras_inputs = {node.name: node.input for node in self.inputs}
        self.model = keras.Model(
            name=self.name,
            inputs=keras_inputs,
            outputs=keras_outputs,
        )
        extra_outputs, extra_inputs = [], []
        for minimizer in self.minimizers:
            if not isinstance(minimizer["source"], Output):
                extra_outputs.append(minimizer["source"])
            if not isinstance(minimizer["target"], Output):
                extra_outputs.append(minimizer["target"])

        if extra_outputs:
            order = toposort_outputs(extra_outputs)
            extra_inputs = [
                node
                for node in order
                if isinstance(node, Input) and node not in self.inputs
            ]

            for node in extra_inputs:
                tensor_map[node] = node.input
            for node in extra_outputs:
                keras_outputs[node.name] = self._resolve_tensor(node, tensor_map)

            self.train_inputs = self.inputs + extra_inputs
            keras_inputs = {node.name: node.input for node in self.inputs} | {
                node.name: node.input for node in extra_inputs
            }
            self._train_model = keras.Model(
                name=self.name + "_train",
                inputs=keras_inputs,
                outputs=keras_outputs,
            )
        else:
            self.train_inputs = self.inputs
            self._train_model = self.model
        return self

    def _resolve_tensor(self, node, tensor_map):
        if node in tensor_map:
            return tensor_map[node]

        if isinstance(node, Constant):
            anchor = next(iter(tensor_map.values()), None)
            if anchor is None:
                raise ValueError(
                    f"Cannot resolve Constant '{node.name}' without at least one model Input."
                )
            y = node.as_tensor(anchor)
            tensor_map[node] = y
            return y

        if isinstance(node, Parameter):
            anchor = next(iter(tensor_map.values()), None)
            if anchor is None:
                raise ValueError(
                    f"Cannot resolve Parameter '{node.name}' without at least one model Input."
                )
            y = node.as_tensor(anchor)
            tensor_map[node] = y

            if node.param is not None:  # and node.param not in self._training_params:
                self._training_params.append(node.param)

            return y

        if isinstance(node, IntermediateOutput):
            y = self._resolve_intermediate_output(node, tensor_map)
            tensor_map[node] = y
            return y

        pred_tensors = [self._resolve_tensor(pred, tensor_map) for pred in node.preds]
        if isinstance(node, Output):
            if len(pred_tensors) != 1:
                raise ValueError(
                    f"Output '{node.name}' expects exactly one predecessor."
                )
            y = pred_tensors[0]
        elif isinstance(node, Input):
            y = node.input
        else:
            y = node.call(*pred_tensors)

        tensor_map[node] = y
        return y

    def _resolve_intermediate_output(self, node, tensor_map):
        """
        Resolve one output of a nested Modely call.

        `node` is an IntermediateOutput.
        `node.pred` is the ModelCall object.
        """
        call = node.pred
        submodel = call.model

        if submodel.model is None:
            submodel.build()

        sub_inputs = {
            old_input.name: self._resolve_tensor(new_input, tensor_map)
            for old_input, new_input in call.inputs_map.items()
        }

        sub_out = submodel.model(sub_inputs)

        if isinstance(sub_out, dict):
            return sub_out[node.name]

        if len(submodel.outputs) == 1:
            return sub_out

        raise ValueError(
            f"Submodel '{submodel.name}' has multiple outputs but returned a non-dict result."
        )

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
            target = Constant(name=None, value=0.0, dtype="float32")
        if isinstance(target, float):
            target = Constant(name=None, value=[target], dtype="float32")
        self.minimizers.append(
            {"name": name, "source": source, "target": target, "loss": loss}
        )
        return self

    def remove_minimizer(self, name: str):
        """Remove a registered minimizer by name."""
        self.minimizers = [m for m in self.minimizers if m["name"] != name]

    def train(
        self,
        train_data: DataLoader,
        epochs: int = 1,
        batch_size: int = 1,
        optimizer=None,
        lr: float = 1e-3,
    ):
        if not self.minimizers:
            raise ValueError("No minimizers defined. Call minimize() before train().")

        n_samples = len(train_data)
        if n_samples == 0:
            raise ValueError("train_data is empty.")

        # Ensure model is built
        if not self._train_model:
            raise ValueError("Model is not built. Call build() before training.")
        km = self._train_model

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
            target_name = target.name

            label_name = _resolve_label_name(target)
            if label_name is not None:
                y_data[source_name] = x_data[label_name]
                continue

            target_value = getattr(target, "value_numpy", None)
            if target_value is None:
                raise ValueError(
                    f"Training target '{target_name}' must be present in the dataset or be a constant value."
                )

            target_value = np.asarray(target_value, dtype=np.float32)
            y_shape = (n_samples,) + tuple(minimizer["source"].shape)
            y_data[source_name] = np.broadcast_to(target_value, y_shape).astype(
                np.float32
            )

        backend = keras.backend.backend()

        if backend == "tensorflow":

            @tf.function
            def train_step(model, batch_inputs, batch_targets, optimizer, losses):
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

                for v in list(km.trainable_weights) + self._training_params:
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

            losses = {m["name"]: keras.losses.get(m["loss"]) for m in self.minimizers}

            history = {"loss": []}
            idxs = np.arange(n_samples)
            epoch_bar = tqdm(range(epochs), desc=f"Training {self.name}", unit="epoch")
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
                        km, batch_inputs, batch_targets, optimizer, losses
                    )
                    if isinstance(total, tf.Tensor):
                        epoch_losses.append(float(total.numpy()))

                mean_epoch_loss = float(np.mean(epoch_losses))
                history["loss"].append(mean_epoch_loss)
                epoch_bar.set_postfix(epoch_loss=f"{mean_epoch_loss:.3e}")

            return history

        if backend == "torch":
            import torch

            unique_params = []
            seen = set()
            for v in list(km.parameters()) + self._training_params:
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
            history = {"loss": []}
            idxs = np.arange(n_samples)
            epoch_bar = tqdm(range(epochs), desc=f"Training {self.name}", unit="epoch")

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
                            preds[source_name], batch_targets[source_name]
                        )

                    total.backward()
                    optimizer.step()

                    epoch_losses.append(float(total.detach().cpu().item()))

                mean_epoch_loss = float(np.mean(epoch_losses))
                history["loss"].append(mean_epoch_loss)
                epoch_bar.set_postfix(epoch_loss=f"{mean_epoch_loss:.3e}")

            km.eval()
            return history

        compile_losses: dict[str, Any] = {
            name: None for name in getattr(km, "output_names", [])
        }
        for minimizer in self.minimizers:
            compile_losses[minimizer["source"].name] = keras.losses.get(
                minimizer["loss"]
            )

        km.compile(optimizer=optimizer, loss=compile_losses)

        history = km.fit(
            x=x_data,
            y=y_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose="1",
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
        inputs: list[Input],
        closed_loop: dict[str | Node, str | Node],
        name: str | None = None,
    ) -> "Modely":
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

        if not self.built:
            self.build()

        loop_fn = Loop(f=self, closed_loop=closed_loop, name=name)
        loop_outputs = loop_fn(inputs)
        if isinstance(loop_outputs, dict):
            outputs = [
                Output(out_name, loop_outputs[out_name]) for out_name in loop_outputs
            ]
        else:
            outputs = [Output(self.outputs[0].name, loop_outputs)]

        return Modely(
            name=name or self.name + "_closed_loop", inputs=inputs, outputs=outputs
        )

    # -------------------------------------------------------------------------
    # Save and load (pickle)
    # -------------------------------------------------------------------------
    def save(self, filename: str):
        """Save the nnodely Model to a file."""
        import cloudpickle

        with open(filename + ".pkl", "wb") as out:
            cloudpickle.dump(self, out)

    @staticmethod
    def load(filename: str) -> "Modely":
        """Load a nnodely Model from a file."""
        import cloudpickle

        with open(filename + ".pkl", "rb") as inp:
            return cloudpickle.load(inp)

    def export_keras(self, filename: str):
        """Export the built Keras model to a file."""
        if self.built and isinstance(self.model, keras.Model):
            self.model.save(filename + ".keras")

    def import_keras(self, filename: str):
        """Import the built Keras model from a file."""
        self.model = keras.models.load_model(
            filename + ".keras", safe_mode=False
        )  # WARNING: safe_mode=False per custom layers con torch.nn.Module.


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
