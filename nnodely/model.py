"""
Model - traversa DAG, build Keras. Nessun grafo globale.

Dimensioni tensori: seq, time, dim.
- seq: tuple, dimensioni sequenziali (es. batch di sequenze)
- time: int, dimensione temporale (finestra)
- dim: tuple, dimensioni features
Shape = seq + (time,) + dim
Default: seq=(), time=1, dim=(1,)
"""
import os 
import json
import re
from pathlib import Path

from nnodely.dag import collect_and_order, to_tuple, get_seq_time_dim, seq_time_dim_to_shape
from nnodely.layer import LayerBase, _is_stream
from nnodely.stream import Stream

from nnodely.output import Output
from nnodely.input import Input
from nnodely.dataloader import DataLoader

import keras

import tensorflow as tf

import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Helper per normalizzazione inputs/outputs
# -----------------------------------------------------------------------------

def _to_dict(items, keys=None):
    """Normalizza singolo/lista/dict → dict.

    keys=None  → chiavi da obj.name  (usato in __init__ per Input/Output)
    keys=[…]   → chiavi predefinite  (usato in __call__ con _input_names)
    """
    if isinstance(items, dict):
        return dict(items)
    if isinstance(items, (list, tuple)):
        if keys is None:
            return {getattr(o, 'name', str(o)): o for o in items}
        if len(items) != len(keys):
            raise TypeError(f"Attesi {len(keys)} valori, ricevuti {len(items)}.")
        return dict(zip(keys, items))
    if keys is None:
        return {getattr(items, 'name', str(items)): items}
    if len(keys) == 1:
        return {keys[0]: items}
    raise TypeError(f"Attesi {len(keys)} valori, ricevuto un singolo elemento.")


def _unwrap_single(obj, keys=None):
    """Se dict con 1 elemento (o keys con 1 elemento), ritorna il valore. Altrimenti obj.
    keys=None → unwrap se len(obj)==1; keys=[k] → ritorna obj[k]."""
    if keys is not None and len(keys) == 1:
        return obj[keys[0]]
    if isinstance(obj, dict) and len(obj) == 1:
        return next(iter(obj.values()))
    return obj


def _stream_reaches_node(stream, target, visited=None):
    """True se stream ha target nella catena dei predecessors (es. SampleWindow → Input)."""
    if stream is target:
        return True
    if not _is_stream(stream) or not getattr(stream, 'predecessors', []):
        return False
    visited = visited or set()
    if id(stream) in visited:
        return False
    visited.add(id(stream))
    return any(_stream_reaches_node(p, target, visited) for p in stream.predecessors)


class Model:
    """
    Modello definito dal DAG. Chiamabile con stream o tensori.
    Dopo build(): solo tensori di forma fissa.
    """

    # -------------------------------------------------------------------------
    # API pubblica
    # -------------------------------------------------------------------------

    def __init__(self, name: str, inputs, outputs):
        """Crea modello da DAG. inputs/outputs: Input/Output o dict {name: node}."""
        self.name = name
        self.inputs = _to_dict(inputs)
        self.outputs = _to_dict(outputs)
        self._order = collect_and_order(list(self.outputs.values()))
        self._input_names = list(self.inputs.keys())
        self._output_names = list(self.outputs.keys())
        self._model = None
        self._input_shapes = {name: get_seq_time_dim(node) for name, node in self.inputs.items()}
        self._validate_inputs_reachable()
        # minimizers: list of dicts {name, source, target, loss}
        self._minimizers = []

    def minimize(self, name:str, source:Output|Input|str, target:Output|Input|str|None = None, loss='mse'):
        """Register a loss to minimize during training.

        name: identifier for this loss (used as an output name in the training model)
        source: node/stream producing predictions (e.g., an Output node)
        target: node/stream providing target values (usually derived from an Input)
        loss: loss identifier accepted by `keras.losses.get`
        """
        self._minimizers.append({'name': name, 'source': source, 'target': target, 'loss': loss})
        return self
    
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
            raise ValueError("No minimizers defined. Call minimize() before train().")

        if train_data is None:
            raise ValueError("No training data selected.")
        n_samples = len(train_data)
        if n_samples == 0:
            raise ValueError("train_data is empty.")

        # Ensure model is built
        if not self._model:
            self.build()
        km = self._model

        # Model input keys that must be loaded by the DataLoader
        X_keys = [k for k in self._input_names if k in train_data.inputs]
        if not X_keys:
            raise ValueError("train_data must contain at least one model input.")

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
                for k in X_keys:
                    batch_inputs[k] = tf.convert_to_tensor(np.stack([sample[k] for sample in batch_samples], axis=0))

                with tf.GradientTape() as tape:
                    preds = km(batch_inputs, training=True)

                    total = 0.0
                    for m in self._minimizers:
                        name = m["name"]
                        source_name = m["source"] if isinstance(m["source"], str) else m["source"].name
                        target_name = m["target"] if isinstance(m["target"], str) else m["target"].name

                        y_pred, y_true = preds[source_name], preds[target_name]

                        loss_obj = losses[name]
                        l = loss_obj(y_true, y_pred)
                        total = total + tf.reduce_mean(l)

                gradients = tape.gradient(total, km.trainable_variables)
                optimizer.apply_gradients(zip(gradients, km.trainable_variables))

                epoch_losses.append(float(total.numpy()))

            mean_epoch_loss = float(np.mean(epoch_losses))
            history["loss"].append(mean_epoch_loss)
            epoch_bar.set_postfix(epoch_loss=f"{mean_epoch_loss:.4f}")

        return history
    
    def inference(self, inputs):
        """Inference mode: call the model with tensor inputs. Requires build()."""
        if not self._model:
            raise ValueError("Model non buildato: chiamare build() prima di inference.")
        return self._call_built_tensor(inputs)

    def export_html(
        self,
        out_dir: str | os.PathLike,
        filename: str|None = None,
        *,
        open_subgraph_in_new_tab: bool = False,
        physics: bool = True,
    ) -> str:
        """
        Export this Model DAG to interactive HTML (vis-network), with recursive
        HTML pages for nested submodels.

        Colors:
        - Input   -> green
        - Output  -> red
        - Model   -> blue (clickable if nested)
        - others  -> default gray

        A right-side panel shows node attributes when a node is clicked.

        Returns the root HTML file path.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        visited_pages: set[tuple[int, str]] = set()

        def _slug(s: str) -> str:
            s = str(s).strip().lower()
            s = re.sub(r"[^a-z0-9._-]+", "-", s)
            s = re.sub(r"-{2,}", "-", s).strip("-")
            return s or "graph"

        def _node_kind(model_obj, node) -> str:
            if node in model_obj.inputs.values():
                return "Input"
            if node in model_obj.outputs.values():
                return "Output"
            if model_obj._is_submodel_node(node):
                return "Model"
            return type(node).__name__

        def _node_color(kind: str) -> str:
            if kind == "Input":
                return "#2ecc71"
            if kind == "Output":
                return "#e74c3c"
            if kind == "Model":
                return "#3498db"
            return "#95a5a6"

        def _safe_attrs(model_obj, node, kind: str) -> dict:
            attrs = {
                # "name": getattr(node, "name", str(node)),
                # "kind": kind,
                "class": type(node).__name__,
            }

            # for attr in ("node_type", "seq", "time", "dim"):
            for attr in ("seq", "time", "dim"):
                if hasattr(node, attr):
                    try:
                        attribute = getattr(node, attr)
                        attrs[attr] = str(attribute) if attribute else '0'
                    except Exception:
                        pass

            # preds = getattr(node, "predecessors", None)
            # if preds is not None:
            #     try:
            #         attrs["predecessors"] = [getattr(p, "name", str(p)) for p in preds]
            #     except Exception:
            #         pass

            layer = getattr(node, "layer", None)
            if layer is not None:
                attrs["layer"] = type(layer).__name__

            if model_obj._is_submodel_node(node):
                submodel = getattr(node, "model", None)
                if submodel is not None:
                    attrs["submodel"] = getattr(submodel, "name", type(submodel).__name__)
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
            nodes = []
            edges = []
            seen_nodes = set()
            seen_edges = set()

            for node in model_obj._order:
                nid = getattr(node, "name", str(node))
                if nid not in seen_nodes:
                    kind = _node_kind(model_obj, node)
                    nodes.append((node, nid, kind, _safe_attrs(model_obj, node, kind)))
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
                        "highlight": {"background": "#f1c40f", "border": "#2c3e50"},
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
                }

                vis_edges.append({
                    "from": src,
                    "to": dst,
                    "arrows": "to",
                    "title": json.dumps(edge_attrs, ensure_ascii=False, indent=2, default=str),
                    "color": {"color": "#7f8c8d"},
                })

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
                <span><span class="swatch" style="background:#3498db"></span>Model</span>
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
                <span>Click blue diamond nodes to open nested models.</span>
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

                function formatValue(value) {{
                if (value === null || value === undefined) return "";
                if (typeof value === "object") {{
                    return `<pre style="margin:0; white-space:pre-wrap;">${{escapeHtml(JSON.stringify(value, null, 2))}}</pre>`;
                }}
                return escapeHtml(String(value));
                }}

                function escapeHtml(text) {{
                return text
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
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
                    extra = `<a class="open-link" href="${{escapeHtml(n.url)}}" target={('"_" + "blank"' if open_subgraph_in_new_tab else '"_self"')}>Open nested model</a>`;
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
                    const target = {('"_" + "blank"' if open_subgraph_in_new_tab else '"_self"')};
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

        import os

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

        def _node_style(node):
            if node in self.inputs.values():
                return {
                    "shape": "box",
                    "style": "rounded,filled",
                    "fillcolor": "#00ff15",
                }
            
            if node in self.outputs.values():
                return {
                    "shape": "box",
                    "style": "rounded,filled",
                    "fillcolor": "#ff0000",
                }

            if self._is_submodel_node(node):
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
            if node in self.inputs.values():
                return f"{node.name}\nInput"
            if node in self.outputs.values():
                return f"{node.name}\nOutput"
            if self._is_submodel_node(node):
                sub_name = getattr(getattr(node, "model", None), "name", "Model")
                return f"{node.name}\n{sub_name}"
            return node.name

        def _add_node(node):
            nid = node.name
            if nid in added_nodes:
                return

            style = _node_style(node)
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
        for node in self.inputs.values():
            _add_node(node)
        for node in self.outputs.values():
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

    @property
    def built(self):
        """True se build() è stato chiamato, False altrimenti."""
        return self._model is not None

    def build(self):
        """Builda il modello Keras. Dopo build(), __call__ con tensori usa il modello persistito."""
        """Traversa DAG con tensori: build + inference. persist=True → salva su self._model."""
        input_shapes = self._input_shapes
        initial_map = {}
        for name, node in self.inputs.items():
            seq, time, dim = input_shapes[name]
            initial_map[node.name] = keras.Input(shape=seq_time_dim_to_shape(seq, time, dim), name=name)

        value_map = self._traverse_dag(initial_map, lambda s, n, p, m: s._resolve_tensor_node(n, p, m))
        model_inputs = {
            name: initial_map[node.name]
            for name, node in self.inputs.items()
        }
        model_outputs = {
            name: keras.layers.Identity(name=name)(value_map[node.name])
            for name, node in self.outputs.items()
        }
        self._model = keras.Model(inputs=model_inputs, outputs=model_outputs, name=self.name)
        self._input_shapes = dict(input_shapes)
        return self

    def __call__(self, inputs : Stream):
        """Esegue il modello. inputs: stream (composizione) o tensori (inference).
        Se non buildato: stream → traversa DAG; tensori → build temporaneo.
        Se buildato: stream → check compatibilità + traversa; tensori → keras.Model."""
        inputs_dict = _to_dict(inputs, keys=self._input_names)
        self._check_no_self_input(inputs_dict)
        are_streams = all(_is_stream(s) for s in inputs_dict.values())
        if self._model:
            return self._call_built_stream(inputs_dict) if are_streams else self._call_built_tensor(inputs_dict)
        elif are_streams:
            return self._call_unbuilt_stream(inputs_dict)
        else:
            self.build()
            out = self._call_built_tensor(inputs_dict)
            self.unbuild()
            return out
            #raise ValueError("Model not built, cannot call with tensors. Call build() first.")

    def unbuild(self):
        """Rimuove il modello Keras buildato. built diventa False."""
        self._model = None
        self._input_shapes = {name: get_seq_time_dim(node) for name, node in self.inputs.items()}
        return self

    # -------------------------------------------------------------------------
    # Validazione DAG Inputs/Outputs
    # -------------------------------------------------------------------------

    def _validate_inputs_reachable(self):
        """Verifica che tutti gli input siano raggiungibili dagli output."""
        order_set = set(self._order)
        for inp in self.inputs.values():
            if inp not in order_set:
                raise ValueError(
                    f"Input '{inp.name}' non raggiungibile dagli output. "
                    f"Output nodes: {[n.name for n in self.outputs.values()]}"
                )

    def _check_no_self_input(self, inputs_dict: dict):
        """Verifica che gli stream passati non derivino dallo stesso input (es. x.sw(5) da x)."""
        for name, stream in inputs_dict.items():
            inp = self.inputs.get(name)
            if inp is not None and _stream_reaches_node(stream, inp):
                raise ValueError(
                    f"Modello '{self.name}': input '{name}' non può ricevere uno stream "
                    f"derivato dallo stesso input (circolare)."
                )

    # -------------------------------------------------------------------------
    # Helper: submodel, unwrap, compatibilità
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_submodel_node(node):
        """True se nodo è Stream con model e input_map (submodel in composizione).
        input_map: {nome_input_submodel: stream} mappa input del submodel agli stream passati."""
        return getattr(node, 'model', None) and getattr(node, 'input_map', None)

    @staticmethod
    def _get_submodel_inputs(node, value_map):
        """Estrae {input_name: val} per il submodel. input_map = {nome_input: stream};
        per ogni stream in input_map, prende val da value_map."""
        return {
            input_name: value_map.get(stream.name)
            for input_name, stream in node.input_map.items()
        }

    def _check_shape_compatibility(self, stream_inputs: dict, expected_shapes: dict, err_prefix: str):
        """Verifica che stream abbiano (seq, time, dim) compatibili con expected_shapes."""
        if not expected_shapes:
            return
        for inp_name, stream in stream_inputs.items():
            if inp_name not in expected_shapes:
                continue
            seq_b, time_b, dim_b = expected_shapes[inp_name]
            seq_s, time_s, dim_s = get_seq_time_dim(stream)
            if (seq_s, time_s, dim_s) != (seq_b, time_b, dim_b):
                raise ValueError(
                    f"{err_prefix} '{inp_name}': "
                    f"atteso (seq={seq_b}, time={time_b}, dim={dim_b}), "
                    f"ricevuto (seq={seq_s}, time={time_s}, dim={dim_s})."
                )

    # -------------------------------------------------------------------------
    # Traversata DAG: stream, tensori, build
    # -------------------------------------------------------------------------

    def _traverse_dag(self, initial_map: dict, resolve_node):
        """Percorre DAG in ordine topologico. resolve_node(self, node, preds, value_map) → valore o None."""
        value_map = dict(initial_map)
        for node in self._order:
            if node.name in value_map:
                continue
            preds = getattr(node, 'predecessors', [])
            if not preds:
                continue
            value = resolve_node(self, node, preds, value_map)
            if value is not None:
                value_map[node.name] = value
        return value_map

    def _run_with_streams(self, inputs_dict: dict):
        """Traversa DAG con stream → {name: stream}. inputs_dict: {nome_input: stream}."""
        initial_map = {
            node.name: inputs_dict[name]
            for name, node in self.inputs.items()
            if name in inputs_dict
        }
        value_map = self._traverse_dag(initial_map, lambda s, n, p, m: s._resolve_stream_node(n, p, m))
        return {name: value_map.get(node.name) for name, node in self.outputs.items()}

    def _resolve_stream_node(self, node, preds, value_map):
        """Risolve nodo → stream."""
        if self._is_submodel_node(node):
            # stream: value_map ha stream, non dict → no unwrap_input
            return self._resolve_submodel(node, value_map, node.model._run_with_streams)
        pred_vals = [value_map.get(p.name) for p in preds]
        if any(v is None for v in pred_vals):
            return None
        layer = getattr(node, 'layer', None)
        pred_val = pred_vals[0]
        if not isinstance(layer, LayerBase):
            return pred_val  # pass-through: Output
        if len(preds) > 1:
            return layer(*pred_vals)
        return layer(pred_val)

    def _resolve_tensor_node(self, node, preds, value_map):
        """Risolve nodo → tensore."""
        if self._is_submodel_node(node):
            # input_map: {nome_input: stream} → verifica che stream abbiano shape attesa
            self._check_shape_compatibility(
                dict(node.input_map), node.model._input_shapes or {},
                f"Modello '{node.model.name}': input incompatibile"
            )
            # tensori: value_map può avere dict da submodel → unwrap per keras single-input
            return self._resolve_submodel(node, value_map, node.model._model)
        layer = getattr(node, 'layer', None)
        pred_val = value_map.get(
            preds[0].name,
            value_map.get(self.inputs[self._input_names[0]].name),
        )
        if not isinstance(layer, LayerBase):
            return pred_val  # pass-through: Output
        self._ensure_layer_built(layer, node, preds[0])
        if len(preds) > 1:
            return layer.call([value_map[p.name] for p in preds])
        return layer.call(pred_val)

    def _resolve_submodel(self, node, value_map, run_fn):
        """Risolve nodo submodel. run_fn(inputs_dict) → output (stream o tensore).
        Input e output: _unwrap_single (stream invariante, tensori da dict)."""
        inputs = self._get_submodel_inputs(node, value_map)
        if any(v is None for v in inputs.values()):
            return None
        inputs = {k: _unwrap_single(v) for k, v in inputs.items()}
        out = run_fn(inputs)
        return _unwrap_single(out)

    def _ensure_layer_built(self, layer, node, node_input):
        """Builda layer se necessario (seq, time, dim da node_input)."""
        seq, time, dim = get_seq_time_dim(node_input)
        built_seq, built_time, built_dim = layer.seq, layer.time, layer.dim
        if getattr(layer, '_layer', None) and (built_seq, built_time, built_dim) != ([seq], [time], [dim]):
            raise ValueError(f"Layer {getattr(layer, 'name', node.name)} already built.")
        if not getattr(layer, '_layer', None):
            layer.seq, layer.time, layer.dim = [seq], [time], [dim]
            layer.build_layer()

    # -------------------------------------------------------------------------
    # Le 4 modalità di __call__: (built, stream) → handler
    # -------------------------------------------------------------------------

    def _call_unbuilt_stream(self, inputs_dict: dict):
        """Modello non buildato + input stream → traversa DAG, restituisce stream."""
        out = self._run_with_streams(inputs_dict)
        return _unwrap_single(out, self._output_names)

    # def _call_unbuilt_tensor(self, inputs_dict: dict):
    #     """Modello non buildato + input tensori → build temporaneo, inference."""
    #     out = self._run_with_tensors(inputs_dict)
    #     return _unwrap_single(out, self._output_names)

    def _call_built_stream(self, inputs_dict: dict):
        """Modello buildato + input stream → check compatibilità, traversa, wrapper Stream.
        input_map = {nome_input: stream} per risolvere il submodel quando usato nel DAG."""
        self._check_shape_compatibility(
            inputs_dict, self._input_shapes or {},
            "Stream incompatibile con modello buildato per"
        )
        out = self._run_with_streams(inputs_dict)
        out_stream = next(iter(out.values()))
        seq, time, dim = get_seq_time_dim(out_stream)
        return Stream(
            self.name, 'Stream',
            seq=seq, time=time, dim=dim,
            predecessors=list(inputs_dict.values()),
            model=self, input_map=dict(inputs_dict)
        )

    def _call_built_tensor(self, inputs_dict: dict):
        """Modello buildato + input tensori → keras.Model inference."""
        if not self._model:
            raise ValueError("Model not built, cannot run with tensors.")
        out = self._model(inputs_dict)
        return _unwrap_single(out, self._output_names)

    # -------------------------------------------------------------------------
    # Save and load (pickle)
    # -------------------------------------------------------------------------

    def save(self, filename:str):
        """Save the nnodely Model to a file."""
        import pickle
        with open(filename+".pkl", 'wb') as out:
            pickle.dump(self, out, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename:str) -> 'Model':
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