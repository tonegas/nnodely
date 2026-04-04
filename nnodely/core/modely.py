import os 
import json
import re
from pathlib import Path

from nnodely.core.dag import collect_and_order, get_seq_time_dim
from nnodely.core.layer import _is_stream
from nnodely.core.stream import Stream

from nnodely.layers.output import Output
from nnodely.layers.input import Input
from nnodely.layers.constant import Constant
from nnodely.core.dataloader import DataLoader

import keras
import tensorflow as tf

import numpy as np
from tqdm import tqdm

class ModelCall(Stream):
    """
    Symbolic node representing the application of a Modely to upstream Stream inputs.
    """

    def __init__(self, model, input_map, output_name, seq, time, dim, predecessors):
        super().__init__(
            name=f"{model.name}",
            node_type="model",
            seq=seq,
            time=time,
            dim=dim,
            predecessors=predecessors,
        )
        self.model = model
        self.input_map = input_map          # {submodel_input_name: upstream_stream}
        self.output_name = output_name      # which output of the submodel this node represents

class Modely:
    """
    Modello definito dal DAG. Chiamabile con stream o tensori.
    Dopo build(): solo tensori di forma fissa.
    """

    # -------------------------------------------------------------------------
    # API pubblica
    # -------------------------------------------------------------------------

    def __init__(self, name: str, outputs : Output | list[Output]):
        """Crea modello da DAG. inputs/outputs: Input/Output o dict {name: node}."""
        self.name = name
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        self._order = collect_and_order(self.outputs)
        self.inputs = [node for node in self._order if node.node_type == 'input']
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

    # def build(self):
    #     """
    #     Build the Keras model by recursively traversing the DAG
    #     starting from the outputs.
    #     """
    #     self._parameter_nodes = []
    #     self._parameter_vars = []
    #     input_map = {node.name: node.input for node in self.inputs}  # memoization map: node name → KerasTensor

    #     keras_outputs = {}
    #     for node in self.outputs:
    #         keras_outputs[node.name] = self._resolve_tensor(node, input_map)

    #     keras_inputs = {node.name: node.input for node in self.inputs}
    #     self._model = keras.Model(
    #         name=self.name,
    #         inputs=keras_inputs,
    #         outputs=keras_outputs,
    #     )

    #     extra_outputs, extra_inputs = [], []
    #     for minimizer in self._minimizers:
    #         if not isinstance(minimizer["source"], Output):
    #             extra_outputs.append(minimizer["source"])
    #         if not isinstance(minimizer["target"], Output):
    #             extra_outputs.append(minimizer["target"])
    #     if extra_outputs:
    #         order = collect_and_order(extra_outputs)
    #         extra_inputs = [node for node in order if node.node_type == 'input']
    #         input_map = {node.name: node.input for node in self.inputs} | {node.name: node.input for node in extra_inputs} # refresh input map for training
    #         for node in extra_outputs:
    #             keras_outputs[node.name] = self._resolve_tensor(node, input_map)

    #         self.train_inputs = self.inputs + extra_inputs
    #         keras_inputs = {node.name: node.input for node in self.inputs} | {node.name: node.input for node in extra_inputs}
    #         self._train_model = keras.Model(
    #             name=self.name+"_train",
    #             inputs=keras_inputs,
    #             outputs=keras_outputs,
    #         )
    #     else:
    #         self._train_model = self._model
    #     return self
    
    def build(self):
        self._parameter_nodes = []
        self._parameter_vars = []

        input_map = {node.name: node.input for node in self.inputs}
        default_anchor = next(iter(input_map.values()), None)

        keras_outputs = {}
        for node in self.outputs:
            keras_outputs[node.name] = self._resolve_tensor(node, input_map, anchor=default_anchor)

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
            order = collect_and_order(extra_outputs)
            extra_inputs = [node for node in order if node.node_type == "input"]

            known = {node.name for node in self.inputs}
            extra_inputs = [node for node in extra_inputs if node.name not in known]

            input_map = (
                {node.name: node.input for node in self.inputs}
                | {node.name: node.input for node in extra_inputs}
            )
            default_anchor = next(iter(input_map.values()), None)

            for node in extra_outputs:
                keras_outputs[node.name] = self._resolve_tensor(node, input_map, anchor=default_anchor)

            self.train_inputs = self.inputs + extra_inputs
            keras_inputs = (
                {node.name: node.input for node in self.inputs}
                | {node.name: node.input for node in extra_inputs}
            )

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
        if isinstance(inputs, dict) and all(isinstance(v, Stream) for v in inputs.values()):
            return self._call_with_streams(inputs)

        # tensor execution mode
        if self._model is None:
            self.build()
        return self._model(inputs)
    
    def _call_with_streams(self, inputs_dict):
        """
        Compose this model symbolically with upstream Stream inputs.
        Returns a Stream if the model has one output, otherwise a dict.
        """
        # missing = [name for name in self._input_names if name not in inputs_dict]
        # if missing:
        #     raise ValueError(f"Missing model inputs for composition: {missing}")

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

    # def _resolve_tensor(self, node, tensor_map):
    #     """
    #     Recursively resolve a DAG node into a KerasTensor.
    #     Uses memoization via tensor_map.
    #     """
    #     if node.name in tensor_map:
    #         return tensor_map[node.name]

    #     preds = getattr(node, "predecessors", [])

    #     if node.node_type in ("Parameter", "Constant"):
    #         y = node.as_tensor()

    #         if node.node_type == "Parameter":
    #             if node not in self._parameter_nodes:
    #                 self._parameter_nodes.append(node)
    #             if node.param is not None and node.param not in self._parameter_vars:
    #                 self._parameter_vars.append(node.param)
    #         tensor_map[node.name] = y
    #         return y
        
    #     # special case: submodel composition
    #     if node.node_type == "model":
    #         y = self._resolve_model_call(node, tensor_map)
    #         tensor_map[node.name] = y
    #         return y

    #     # Resolve predecessor tensors first
    #     pred_tensors = [self._resolve_tensor(pred, tensor_map) for pred in preds]

    #     # Output node: pure pass-through
    #     if getattr(node, "node_type", None) == "output":
    #         y = pred_tensors[0]
    #     else:
    #         layer = self._get_or_build_layer(node, preds)

    #         if len(pred_tensors) == 1:
    #             y = layer(pred_tensors[0])
    #         else:
    #             y = layer(pred_tensors)

    #     tensor_map[node.name] = y
    #     return y
    
    def _resolve_tensor(self, node, tensor_map, anchor=None):
        if node.name in tensor_map:
            return tensor_map[node.name]

        preds = getattr(node, "predecessors", [])

        if node.node_type in ("Parameter", "Constant"):
            if anchor is None:
                anchor = next(iter(tensor_map.values()), None)
            if anchor is None:
                raise ValueError(
                    f"Cannot convert source node '{node.name}' into a KerasTensor without an anchor."
                )

            y = node.as_tensor(anchor)
            tensor_map[node.name] = y

            if node.node_type == "Parameter":
                if node not in self._parameter_nodes:
                    self._parameter_nodes.append(node)
                if node.param is not None and node.param not in self._parameter_vars:
                    self._parameter_vars.append(node.param)
            return y

        if node.node_type == "model":
            y = self._resolve_model_call(node, tensor_map)
            tensor_map[node.name] = y
            return y

        pred_tensors = []
        local_anchor = anchor

        for pred in preds:
            pt = self._resolve_tensor(pred, tensor_map, anchor=local_anchor)
            pred_tensors.append(pt)
            if local_anchor is None:
                local_anchor = pt

        if node.node_type == "output":
            y = pred_tensors[0]
        else:
            layer = self._get_or_build_layer(node, preds)
            y = layer(pred_tensors[0]) if len(pred_tensors) == 1 else layer(pred_tensors)

        tensor_map[node.name] = y
        return y

    def _get_or_build_layer(self, node, preds):
        """
        Return the concrete Keras layer for a graph node.
        Assumes operational nodes inherit from Stream and expose:
          - node._layer
          - node.build_layer()
        """
        layer = getattr(node, "_layer", None)

        if layer is None:
            # propagate predecessor shape info if needed before build_layer()
            seqs, times, dims = zip(*(get_seq_time_dim(p) for p in preds)) if preds else ([], [], [])

            # optional metadata assignment for layers that need it
            # node.seq = list(seqs)
            # node.time = list(times)
            # node.dim = list(dims)

            if not hasattr(node, "build_layer"):
                raise ValueError(
                    f"Node '{node.name}' ({node.node_type}) has no _layer "
                    f"and no build_layer() method."
                )

            layer = node.build_layer()

        return layer
    
    def _resolve_model_call(self, node, tensor_map):
        """
        Resolve a symbolic submodel application into a KerasTensor.
        """
        submodel = node.model

        if submodel._model is None:
            submodel.build()

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
    
    def minimize(self, name:str, source:Output|Stream, target:Output|Stream|float|None = None, loss='mse'):
        """Register a loss to minimize during training.
        name: identifier for this loss (used as an output name in the training model)
        source: node/stream producing predictions (e.g., an Output node)
        target: node/stream providing target values (usually derived from an Input)
        loss: loss identifier accepted by `keras.losses.get`
        """
        if target is None: ## Transform it into a Constant with value zero
            target = Constant(name=None, value=0.0, dtype='float32')
        if isinstance(target, float):
            target = Constant(name=None, value=[target], dtype='float32')
        self._minimizers.append({'name': name, 'source': source, 'target': target, 'loss': loss})
        return self
    
    def remove_minimizer(self, name:str):
        """Remove a registered minimizer by name."""
        self._minimizers = [m for m in self._minimizers if m['name'] != name]
    
    def train(
        self,
        train_data : DataLoader,
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
                raise ValueError(f"train_data must provide input '{k}' required by the model.")
    
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
                batch_idx = idxs[start:start + batch_size]

                # Collect samples from DataLoader
                batch_samples = [train_data[i] for i in batch_idx]
                # Build batch inputs:
                batch_inputs = {}
                for k in [node.name for node in self.train_inputs]:
                    batch_inputs[k] = tf.convert_to_tensor(np.stack([sample[k] for sample in batch_samples], axis=0))

                with tf.GradientTape() as tape:
                    preds = km(batch_inputs, training=True)

                    total = 0.0
                    for m in self._minimizers:
                        name = m["name"]
                        source_name = m["source"].name
                        target_name = m["target"].name

                        y_pred, y_true = preds[source_name], preds[target_name]

                        loss_obj = losses[name]
                        l = loss_obj(y_true, y_pred)
                        total = total + tf.reduce_mean(l)

                # gradients = tape.gradient(total, km.trainable_weights)
                # # Filter out None gradients (non-differentiable vars) before applying
                # grads_and_vars = [(g, v) for g, v in zip(gradients or [], km.trainable_weights) if g is not None]
                trainable_vars = list(km.trainable_weights) + list(self._parameter_vars)
                seen = set()
                unique_vars = []
                for v in trainable_vars:
                    vid = id(v)
                    if vid not in seen:
                        seen.add(vid)
                        unique_vars.append(v)

                gradients = tape.gradient(total, unique_vars)
                grads_and_vars = [(g, v) for g, v in zip(gradients or [], unique_vars) if g is not None]
                if grads_and_vars:
                    optimizer.apply_gradients(grads_and_vars)
                else:
                    print("Warning: No gradients to apply in this step.")

                epoch_losses.append(float(total.numpy()))

            mean_epoch_loss = float(np.mean(epoch_losses))
            history["loss"].append(mean_epoch_loss)
            epoch_bar.set_postfix(epoch_loss=f"{mean_epoch_loss:.4f}")

        return history

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
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        visited_pages: set[tuple[int, str]] = set()

        def _slug(s: str) -> str:
            s = str(s).strip().lower()
            s = re.sub(r"[^a-z0-9._-]+", "-", s)
            s = re.sub(r"-{2,}", "-", s).strip("-")
            return s or "graph"

        def _node_kind(node) -> str:
            node_type = getattr(node, "node_type", "")
            if node_type == "input":
                return "Input"
            if node_type == "output":
                return "Output"
            if node_type == "model":
                return "Model"
            return "Other"

        def _node_color(kind: str) -> str:
            if kind == "Input":
                return "#2ecc71"
            if kind == "Output":
                return "#e74c3c"
            if kind == "Model":
                return "#3498db"
            return "#95a5a6"

        def _safe_attrs(node) -> dict:
            attrs = {
                "class": type(node).__name__,
                "name": getattr(node, "name", None),
                "node_type": getattr(node, "node_type", None),
            }

            for attr in ("seq", "time", "dim", "shape"):
                if hasattr(node, attr):
                    try:
                        attrs[attr] = getattr(node, attr)
                    except Exception:
                        pass

            preds = getattr(node, "predecessors", None)
            if preds is not None:
                try:
                    attrs["predecessors"] = [getattr(p, "name", str(p)) for p in preds]
                except Exception:
                    attrs["predecessors"] = str(preds)

            layer = getattr(node, "_layer", None)
            if layer is not None:
                try:
                    attrs["keras_layer"] = type(layer).__name__
                except Exception:
                    attrs["keras_layer"] = str(layer)

            if hasattr(node, "_properties"):
                try:
                    attrs["properties"] = dict(getattr(node, "_properties"))
                except Exception:
                    pass

            if getattr(node, "node_type", None) == "model":
                submodel = getattr(node, "model", None)
                if submodel is not None:
                    attrs["submodel"] = getattr(submodel, "name", type(submodel).__name__)

                output_name = getattr(node, "output_name", None)
                if output_name is not None:
                    attrs["output_name"] = output_name

                input_map = getattr(node, "input_map", None)
                if input_map:
                    try:
                        attrs["input_map"] = {
                            k: getattr(v, "name", str(v)) for k, v in input_map.items()
                        }
                    except Exception:
                        attrs["input_map"] = str(input_map)

            return attrs

        def _collect_graph(model_obj):
            """
            Collect nodes and edges from model_obj._order.
            Assumes _order is already topologically sorted.
            """
            nodes = []
            edges = []
            seen_nodes = set()
            seen_edges = set()

            for node in model_obj._order:
                nid = getattr(node, "name", str(node))
                if nid in seen_nodes:
                    continue
                kind = _node_kind(node)
                nodes.append((node, nid, kind, _safe_attrs(node)))
                seen_nodes.add(nid)

            for node in model_obj._order:
                dst = getattr(node, "name", str(node))
                for pred in getattr(node, "predecessors", []) or []:
                    src = getattr(pred, "name", str(pred))
                    edge = (src, dst)
                    if edge in seen_edges:
                        continue
                    edges.append((pred, node))
                    seen_edges.add(edge)

            return nodes, edges

        def _export_one(model_obj, html_name: str, page_title: str, parent_rel: str | None = None) -> str:
            page_key = (id(model_obj), html_name)
            file_path = out_path / html_name

            if page_key in visited_pages and file_path.exists():
                return str(file_path)
            visited_pages.add(page_key)

            graph_nodes, graph_edges = _collect_graph(model_obj)

            # nested pages for model_call nodes
            url_map: dict[str, str] = {}

            for node, nid, kind, attrs in graph_nodes:
                if kind != "Model":
                    continue

                submodel = getattr(node, "model", None)
                if submodel is None:
                    continue

                sub_name = f"{_slug(page_title)}__{_slug(nid)}.html"
                sub_title = f"{page_title} :: {nid}"
                parent_for_child = os.path.relpath(file_path, start=out_path)

                sub_path = _export_one(
                    submodel,
                    sub_name,
                    sub_title,
                    parent_rel=parent_for_child,
                )
                url_map[nid] = os.path.relpath(sub_path, start=out_path)

            vis_nodes = []
            node_details = {}

            for node, nid, kind, attrs in graph_nodes:
                node_details[nid] = attrs

                rec = {
                    "id": nid,
                    "label": nid,
                    "shape": "dot",
                    "size": 14,
                    "color": {
                        "background": _node_color(kind),
                        "border": "#2c3e50",
                        "highlight": {
                            "background": "#f1c40f",
                            "border": "#2c3e50",
                        },
                    },
                    "font": {"color": "#111111"},
                }

                if kind == "Model":
                    rec["shape"] = "diamond"
                    rec["size"] = 18

                if nid in url_map:
                    rec["url"] = url_map[nid]

                vis_nodes.append(rec)

            vis_edges = []
            for pred, node in graph_edges:
                src = getattr(pred, "name", str(pred))
                dst = getattr(node, "name", str(node))
                edge_attrs = {
                    "from_class": type(pred).__name__,
                    "to_class": type(node).__name__,
                    "from_type": getattr(pred, "node_type", None),
                    "to_type": getattr(node, "node_type", None),
                }

                vis_edges.append({
                    "from": src,
                    "to": dst,
                    "arrows": "to",
                    "title": json.dumps(edge_attrs, ensure_ascii=False, indent=2, default=str),
                    "color": {"color": "#7f8c8d"},
                })

            target = "_blank" if open_subgraph_in_new_tab else "_self"

            back_html = ""
            if parent_rel is not None:
                back_html = f"""
                <button id="backBtn" class="btn" title="Go back to parent graph">← Back</button>
                <a class="crumb" href="{parent_rel}">Parent</a>
                """

            html = f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{page_title}</title>
    <style>
        body {{
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
            margin: 0;
            padding: 0;
            background: #ffffff;
            color: #111;
        }}
        header {{
            padding: 12px 16px;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        header h1 {{
            font-size: 16px;
            margin: 0;
            font-weight: 600;
        }}
        .legend {{
            margin-left: auto;
            display: flex;
            gap: 10px;
            align-items: center;
            font-size: 12px;
            color: #444;
            flex-wrap: wrap;
        }}
        .swatch {{
            width: 10px;
            height: 10px;
            border-radius: 3px;
            display: inline-block;
            border: 1px solid #2c3e50;
            margin-right: 6px;
        }}
        .main {{
            display: flex;
            width: 100%;
            height: 92vh;
        }}
        #network {{
            flex: 1 1 auto;
            min-width: 0;
            height: 100%;
        }}
        #inspector {{
            width: 340px;
            min-width: 340px;
            max-width: 340px;
            border-left: 1px solid #eee;
            background: #fafafa;
            overflow: auto;
            padding: 14px;
            box-sizing: border-box;
        }}
        #inspector h2 {{
            font-size: 14px;
            margin: 0 0 10px 0;
            font-weight: 600;
        }}
        #inspector .empty {{
            font-size: 12px;
            color: #666;
            line-height: 1.5;
        }}
        .attr-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            table-layout: fixed;
        }}
        .attr-table th,
        .attr-table td {{
            border: 1px solid #e5e7eb;
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
            word-break: break-word;
        }}
        .attr-table th {{
            width: 34%;
            background: #f3f4f6;
            font-weight: 600;
        }}
        .hint {{
            padding: 8px 16px;
            font-size: 12px;
            color: #555;
            border-top: 1px solid #eee;
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .btn {{
            border: 1px solid #ddd;
            background: #fafafa;
            padding: 6px 10px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 12px;
        }}
        .btn:hover {{
            background: #f2f2f2;
        }}
        .crumb {{
            font-size: 12px;
            color: #444;
            text-decoration: none;
            border-bottom: 1px dashed #bbb;
        }}
        .crumb:hover {{
            color: #111;
            border-bottom-color: #111;
        }}
        .open-link {{
            display: inline-block;
            margin-top: 10px;
            font-size: 12px;
            color: #2563eb;
            text-decoration: none;
        }}
        .open-link:hover {{
            text-decoration: underline;
        }}
    </style>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    </head>
    <body>
    <header>
        {back_html}
        <h1>{page_title}</h1>
        <div class="legend">
            <span><span class="swatch" style="background:#2ecc71"></span>Input</span>
            <span><span class="swatch" style="background:#e74c3c"></span>Output</span>
            <span><span class="swatch" style="background:#3498db"></span>Submodel</span>
            <span><span class="swatch" style="background:#95a5a6"></span>Other</span>
        </div>
    </header>

    <div class="main">
        <div id="network"></div>
        <aside id="inspector">
            <h2>Node details</h2>
            <div id="inspectorContent" class="empty">
                Click a node to inspect its attributes.
            </div>
        </aside>
    </div>

    <div class="hint">
        <span>Hover edges for details.</span>
        <span>Click a node to inspect its attributes.</span>
        <span>Double-click blue diamond nodes to open nested models.</span>
        {"<span>Use ← Back to return to the parent graph.</span>" if parent_rel is not None else ""}
    </div>

    <script>
        const nodes = new vis.DataSet({json.dumps(vis_nodes, ensure_ascii=False)});
        const edges = new vis.DataSet({json.dumps(vis_edges, ensure_ascii=False)});
        const nodeDetails = {json.dumps(node_details, ensure_ascii=False, default=str)};

        const container = document.getElementById('network');
        const inspectorContent = document.getElementById('inspectorContent');
        const data = {{ nodes, edges }};

        const options = {{
            autoResize: true,
            interaction: {{
                hover: true,
                tooltipDelay: 80,
                multiselect: true,
                navigationButtons: true,
                keyboard: true
            }},
            physics: {str(physics).lower()},
            layout: {{
                improvedLayout: true
            }},
            nodes: {{
                borderWidth: 1,
                shadow: true
            }},
            edges: {{
                smooth: {str(physics).lower()},
                shadow: {str(physics).lower()}
            }}
        }};

        const network = new vis.Network(container, data, options);

        function escapeHtml(text) {{
            return String(text)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }}

        function formatValue(value) {{
            if (value === null || value === undefined) return "";
            if (typeof value === "object") {{
                return `<pre style="margin:0; white-space:pre-wrap;">${{escapeHtml(JSON.stringify(value, null, 2))}}</pre>`;
            }}
            return escapeHtml(String(value));
        }}

        function renderInspector(nodeId) {{
            const details = nodeDetails[nodeId];
            const n = nodes.get(nodeId);

            if (!details) {{
                inspectorContent.innerHTML = '<div class="empty">No details available for this node.</div>';
                return;
            }}

            const rows = Object.entries(details).map(([key, value]) => {{
                return `
                <tr>
                    <th>${{escapeHtml(key)}}</th>
                    <td>${{formatValue(value)}}</td>
                </tr>
                `;
            }}).join("");

            let extra = "";
            if (n && n.url) {{
                extra = `<a class="open-link" href="${{escapeHtml(n.url)}}" target="{target}">Open nested model</a>`;
            }}

            inspectorContent.innerHTML = `
                <table class="attr-table">
                    <tbody>${{rows}}</tbody>
                </table>
                ${{extra}}
            `;
        }}

        network.on("click", function(params) {{
            if (!params.nodes || params.nodes.length === 0) return;
            const nodeId = params.nodes[0];
            renderInspector(nodeId);
        }});

        network.on("doubleClick", function(params) {{
            if (!params.nodes || params.nodes.length === 0) return;
            const nodeId = params.nodes[0];
            const n = nodes.get(nodeId);
            if (n && n.url) {{
                const target = "{target}";
                if (target === "_blank") {{
                    window.open(n.url, "_blank");
                }} else {{
                    window.location.href = n.url;
                }}
            }}
        }});

        const backBtn = document.getElementById("backBtn");
        if (backBtn) {{
            backBtn.addEventListener("click", () => {{
                const parentUrl = {json.dumps(parent_rel)};
                if (window.history.length > 1) {{
                    window.history.back();
                }} else {{
                    window.location.href = parentUrl;
                }}
            }});
        }}
    </script>
    </body>
    </html>
    """
            file_path.write_text(html, encoding="utf-8")
            return str(file_path)

        root_name = f"{_slug(filename if filename else self.name)}.html"
        return _export_one(self, root_name, self.name, parent_rel=None)

    def plot(self, to_file: str, include_minimizers: bool = True):
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
        try:
            import graphviz
        except Exception as e:
            raise ImportError(
                "graphviz python package is required for Model.plot(). "
                "Install with `pip install graphviz`."
            ) from e

        fmt = None
        if "." in to_file:
            fmt = to_file.split(".")[-1]

        dot = graphviz.Digraph(name=self.name, format=fmt or "png")
        dot.attr(rankdir="LR")
        dot.attr("graph", bgcolor="white")
        dot.attr("node", fontname="Helvetica", fontsize="10")
        dot.attr("edge", fontname="Helvetica", fontsize="9")

        added_nodes = set()
        added_edges = set()

        def _node_style(node_type):
            if node_type == "input":
                return {
                    "shape": "box",
                    "style": "rounded,filled",
                    "fillcolor": "#00ff15",
                }
            
            if node_type == "output":
                return {
                    "shape": "box",
                    "style": "rounded,filled",
                    "fillcolor": "#ff0000",
                }

            if node_type == "model":
                return {
                    "shape": "folder",
                    "style": "filled",
                    "fillcolor": "#1d73ff",
                }

            return {
                "shape": "box",
                "style": "filled",
                "fillcolor": "#cdcdcd",
            }

        def _node_label(node):
            if node.node_type == "input":
                return f"{node.name}\nInput"
            if node.node_type == "output":
                return f"{node.name}\nOutput"
            if node.node_type == "model":
                sub_name = getattr(getattr(node, "model", None), "name", "Model")
                return f"{node.name}\n{sub_name}"
            return node.name

        def _add_node(node):
            nid = node.name
            if nid in added_nodes:
                return

            style = _node_style(node.node_type)
            dot.node(
                nid,
                label=_node_label(node),
                shape=style["shape"],
                style=style["style"],
                fillcolor=style["fillcolor"],
            )
            added_nodes.add(nid)

        def _add_edge(src: str, dst: str, **attrs):
            edge = (src, dst, tuple(sorted(attrs.items())))
            if edge in added_edges:
                return
            dot.edge(src, dst, **attrs)
            added_edges.add(edge)

        # Add all model nodes from the DAG order
        for node in self._order:
            _add_node(node)

        # Ensure top-level inputs/outputs are always present
        for node in self.inputs:
            _add_node(node)
        for node in self.outputs:
            _add_node(node)

        # Add normal DAG edges
        for node in self._order:
            for pred in getattr(node, "predecessors", []) or []:
                _add_node(pred)
                _add_edge(pred.name, node.name)

        # Add minimizers
        if include_minimizers:
            for i, m in enumerate(self._minimizers):
                loss_name = m.get("loss", "loss")
                min_name = m.get("name", f"loss_{i}")

                source = m.get("source")
                target = m.get("target")

                source_name = source if isinstance(source, str) else getattr(source, "name", str(source))
                target_name = target if isinstance(target, str) else getattr(target, "name", str(target)) if target is not None else None

                loss_node_id = f"__min_{min_name}"
                loss_fill = "#8e44ad"

                dot.node(
                    loss_node_id,
                    label=f"{min_name}\n{loss_name}",
                    shape="hexagon",
                    style="filled",
                    fillcolor=loss_fill,
                    color=loss_fill,
                    fontcolor="white",
                )

                if source_name:
                    _add_edge(source_name, loss_node_id, color=loss_fill, penwidth="1.6")
                if target_name:
                    _add_edge(target_name, loss_node_id, color=loss_fill, style="dashed", penwidth="1.6")

        # Render
        try:
            outpath = to_file
            if "." in to_file:
                outpath = ".".join(to_file.split(".")[:-1])
            dot.render(filename=outpath, cleanup=True)
        except Exception:
            with open(to_file, "w", encoding="utf-8") as f:
                f.write(dot.source)

        return dot

    # -------------------------------------------------------------------------
    # Save and load (pickle)
    # -------------------------------------------------------------------------

    def save(self, filename:str):
        """Save the nnodely Model to a file."""
        import pickle
        with open(filename+".pkl", 'wb') as out:
            pickle.dump(self, out, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename:str) -> 'Modely':
        """Load a nnodely Model from a file."""
        import pickle
        with open(filename+".pkl", 'rb') as inp:
            return pickle.load(inp)
        
    def export_keras(self, filename:str):
        """Export the built Keras model to a file."""
        if not self.built:
            self.build()
        self._model.save(filename+".keras")

    def import_keras(self, filename:str):
        """Import the built Keras model from a file."""
        self._model = keras.models.load_model(filename+".keras", safe_mode=False) # WARNING: safe_mode=False per custom layers con torch.nn.Module.