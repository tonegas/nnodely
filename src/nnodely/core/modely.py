import os

from typing import cast

from nnodely.core.dag import toposort, flatten
from nnodely.core.stream import Stream

from nnodely.layers.output import Output
from nnodely.layers.constant import Constant
from nnodely.layers.parameter import Parameter
from nnodely.core.dataloader import DataLoader
from nnodely.layers.input import Input

import keras
import tensorflow as tf

import numpy as np
from tqdm import tqdm


class ModelCall(Stream):
    """
    Symbolic node representing the application of a Modely to upstream Stream inputs.
    """

    def __init__(self, model, input_map, output_name, dim, time, seq, predecessors):
        super().__init__(
            name=f"{model.name}",
            # node_type="Model",
            seq=seq,
            time=time,
            dim=dim,
            predecessors=predecessors,
        )
        self.model = model
        self.input_map = input_map
        self.output_map = {node.name: node for node in self.model.outputs}
        self.output_name = output_name


class Modely:
    """
    Modello definito dal DAG. Chiamabile con stream o tensori.
    Dopo build(): solo tensori di forma fissa.
    """

    # -------------------------------------------------------------------------
    # API pubblica
    # -------------------------------------------------------------------------

    def __init__(self, name: str, outputs: Output | list[Output]):
        """Crea modello da DAG. inputs/outputs: Input/Output o dict {name: node}."""
        self.name = name
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        self._order = toposort(self.outputs)
        self.inputs = [node for node in self._order if isinstance(node, Input)]
        self.train_inputs = self.inputs
        self._model = None
        self._train_model = None
        self._input_shapes = {node.name: node.shape for node in self.inputs}
        self._minimizers = []

        self._parameter_nodes = []
        self._parameter_vars = []

    @property
    def built(self):
        """True se build() è stato chiamato, False altrimenti."""
        return self._model is not None

    def build(self):
        self._parameter_nodes = []
        self._parameter_vars = []

        tensor_map = {node.name: node.input for node in self.inputs}
        default_anchor = next(iter(tensor_map.values()), None)

        keras_outputs = {}
        for node in self.outputs:
            keras_outputs[node.name] = self._resolve_tensor(
                node, tensor_map, anchor=default_anchor
            )

        keras_inputs = {node.name: node.input for node in self.inputs}
        self._model = keras.Model(
            name=self.name,
            inputs=keras_inputs,
            outputs=keras_outputs,
        )

        extra_outputs, extra_inputs = [], []
        for minimizer in self._minimizers:
            if not isinstance(minimizer["source"], Output):
                extra_outputs.append(minimizer["source"])
            if not isinstance(minimizer["target"], Output):
                extra_outputs.append(minimizer["target"])

        if extra_outputs:
            order = toposort(extra_outputs)
            extra_inputs = [
                node
                for node in order
                if isinstance(node, Input) and node not in self.inputs
            ]

            for node in extra_inputs:
                tensor_map[node.name] = node.input
            default_anchor = next(iter(tensor_map.values()), None)

            for node in extra_outputs:
                keras_outputs[node.name] = self._resolve_tensor(
                    node, tensor_map, anchor=default_anchor
                )

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
            self._train_model = self._model
        return self

    def __call__(self, inputs):
        # symbolic composition mode
        if isinstance(inputs, Stream):
            return self._call_with_streams({inputs.name: inputs})
        if isinstance(inputs, dict) and all(
            isinstance(v, Stream) for v in inputs.values()
        ):
            return self._call_with_streams(inputs)

        # tensor execution mode
        if self._model is None:
            self.build()
        if self._model is None:
            raise ValueError("Model build failed, _model is still None.")
        out = self._model(inputs)
        if isinstance(out, dict) and len(out) == 1:
            return next(iter(out.values()))
        return out

    def _call_with_streams(self, inputs_dict):
        """
        Compose this model symbolically with upstream Stream inputs.
        Returns a Stream if the model has one output, otherwise a dict.
        """
        predecessors = [inputs_dict[node.name] for node in self.inputs]
        outputs = {}
        for node in self.outputs:
            outputs[node.name] = ModelCall(
                model=self,
                input_map=dict(inputs_dict),
                output_name=node.name,
                seq=node.seq,
                time=node.time,
                dim=node.dim,
                predecessors=predecessors,
            )
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs

    def _resolve_tensor(self, node, tensor_map, anchor=None):
        if node.name in tensor_map:
            return tensor_map[node.name]

        if isinstance(node, Parameter) or isinstance(node, Constant):
            if anchor is None:
                anchor = next(iter(tensor_map.values()), None)
            if anchor is None:
                raise ValueError(
                    f"Cannot convert source node '{node.name}' into a KerasTensor without an anchor."
                )

            y = node.as_tensor(anchor)
            tensor_map[node.name] = y

            if isinstance(node, Parameter):
                if node not in self._parameter_nodes:
                    self._parameter_nodes.append(node)
                if node.param is not None:
                    if not any(node.param is v for v in self._parameter_vars):
                        self._parameter_vars.append(node.param)
            return y

        if isinstance(node, ModelCall):
            y = self._resolve_model_call(node, tensor_map)
            tensor_map[node.name] = y
            return y

        pred_tensors = []
        local_anchor = anchor

        for pred in node.predecessors:
            pt = self._resolve_tensor(pred, tensor_map, anchor=local_anchor)
            pred_tensors.append(pt)
            if local_anchor is None:
                local_anchor = pt

        if isinstance(node, Output):
            y = pred_tensors[0]
        else:
            # layer = node.build_layer()# if node._layer is None else node._layer
            # y = (
            #     layer(pred_tensors[0])
            #     if len(pred_tensors) == 1
            #     else layer(pred_tensors)
            # )
            y = node.call(*pred_tensors)

        tensor_map[node.name] = y
        return y

    def _resolve_model_call(self, node, tensor_map):
        """
        Resolve a symbolic submodel application into a KerasTensor.
        """
        submodel = node.model

        # resolve upstream tensors for each submodel input
        sub_inputs = {}
        for input_name, upstream_stream in node.input_map.items():
            sub_inputs[input_name] = self._resolve_tensor(upstream_stream, tensor_map)

        # call the inner keras model
        sub_out = submodel._model(sub_inputs)

        # if model has one output, keras may return a single tensor instead of dict
        if isinstance(sub_out, dict):
            return sub_out[node.output_name]

        # single-output model
        if len(submodel.outputs) == 1:
            return sub_out

        raise ValueError(
            f"Submodel '{submodel.name}' returned non-dict output for multi-output case."
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
        self._minimizers.append(
            {"name": name, "source": source, "target": target, "loss": loss}
        )
        return self

    def remove_minimizer(self, name: str):
        """Remove a registered minimizer by name."""
        self._minimizers = [m for m in self._minimizers if m["name"] != name]

    def train(
        self,
        train_data: DataLoader,
        epochs: int = 1,
        batch_size: int = 1,
        optimizer=None,
        lr: float = 1e-2,
    ):
        """Custom Keras training loop using a DataLoader iterable.

        Expected train_data:
            DataLoader object yielding samples like:
                {
                    "x": np.array([...]),
                    "x_target": np.array([...]),
                    ...
                }
        """

        @tf.function
        def train_step(batch_inputs, optimizer, losses):
            with tf.GradientTape() as tape:
                preds = km(batch_inputs, training=True)

                total = keras.ops.zeros(())
                for m in self._minimizers:
                    name = m["name"]
                    source_name = m["source"].name
                    target_name = m["target"].name

                    y_pred, y_true = preds[source_name], preds[target_name]

                    loss_obj = losses[name]
                    # total = total + tf.reduce_mean(loss_obj(y_true, y_pred))
                    total = total + keras.ops.mean(
                        loss_obj(y_true, y_pred)
                    )  # ensure scalar loss

            unique_vars: list[tf.Variable] = []
            seen = set()

            for v in list(km.trainable_weights) + list(self._parameter_vars):
                vid = id(v)
                if vid not in seen:
                    seen.add(vid)
                    unique_vars.append(cast(tf.Variable, v))

            total = cast(tf.Tensor, total)
            gradients = tape.gradient(total, unique_vars)
            grads_and_vars = [
                (g, v) for g, v in zip(gradients or [], unique_vars) if g is not None
            ]
            if grads_and_vars:
                optimizer.apply_gradients(grads_and_vars)
            else:
                print("Warning: No gradients to apply in this step.")

            return total

        if not self._minimizers:
            print("No minimizers defined. Call minimize() before train().")
            return

        if train_data is None:
            raise ValueError("No training data selected.")
        n_samples = len(train_data)
        if n_samples == 0:
            raise ValueError("train_data is empty.")

        # Ensure model is built
        if not self._train_model:
            raise ValueError("Model is not built. Call build() before training.")
        km = self._train_model

        # Model input keys that must be loaded by the DataLoader
        for k in [node.name for node in self.train_inputs]:
            if k not in train_data.inputs:
                raise ValueError(
                    f"train_data must provide input '{k}' required by the model."
                )

        # Default optimizer ## TODO: change with a user defined optimizer
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        ## Default losses for each minimizer ## TODO: allow user to specify loss per minimizer
        losses = {m["name"]: keras.losses.MeanSquaredError() for m in self._minimizers}

        history = {"loss": []}
        idxs = np.arange(n_samples)
        epoch_bar = tqdm(range(epochs), desc=f"Training {self.name}", unit="epoch")
        for epoch in epoch_bar:
            np.random.shuffle(idxs)
            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                batch_idx = idxs[start : start + batch_size]

                # Collect samples from DataLoader
                batch_samples = [train_data[i] for i in batch_idx]
                # Build batch inputs:
                batch_inputs = {}
                for k in [node.name for node in self.train_inputs]:
                    batch_inputs[k] = tf.convert_to_tensor(
                        np.stack([sample[k] for sample in batch_samples], axis=0)
                    )

                total = train_step(batch_inputs, optimizer, losses)
                if isinstance(total, tf.Tensor):
                    epoch_losses.append(float(total.numpy()))
                # gradients = tape.gradient(total, km.trainable_weights)
                # # Filter out None gradients (non-differentiable vars) before applying
                # grads_and_vars = [(g, v) for g, v in zip(gradients or [], km.trainable_weights) if g is not None]
                # trainable_vars = list(km.trainable_weights) + list(self._parameter_vars)
                # seen = set()
                # unique_vars = []
                # for v in trainable_vars:
                #     vid = id(v)
                #     if vid not in seen:
                #         seen.add(vid)
                #         unique_vars.append(v)

                # gradients = tape.gradient(total, unique_vars)

            mean_epoch_loss = float(np.mean(epoch_losses))
            history["loss"].append(mean_epoch_loss)
            epoch_bar.set_postfix(epoch_loss=f"{mean_epoch_loss:.3e}")

        return history

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
        if self.built and isinstance(self._model, keras.Model):
            self._model.save(filename + ".keras")

    def import_keras(self, filename: str):
        """Import the built Keras model from a file."""
        self._model = keras.models.load_model(
            filename + ".keras", safe_mode=False
        )  # WARNING: safe_mode=False per custom layers con torch.nn.Module.
