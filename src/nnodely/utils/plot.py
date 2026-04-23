import os
import re
import json
from pathlib import Path
from nnodely.core.modely import Modely


def plot_graphviz(
    model: Modely,
    to_file: str,
    include_minimizers: bool = True,
    flatten: bool = False,
):
    try:
        import graphviz
    except Exception as e:
        raise ImportError(
            "graphviz python package is required for Model.plot(). "
            "Install with `pip install graphviz`."
        ) from e

    if flatten:
        model = model.flatten()

    fmt = None
    if "." in to_file:
        fmt = to_file.split(".")[-1]

    dot = graphviz.Digraph(name=model.name, format=fmt or "png")
    dot.attr(rankdir="LR")
    dot.attr("graph", bgcolor="white")
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    added_nodes = set()
    added_edges = set()

    def _node_style(node_type):
        if node_type == "Input":
            return {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "#00ff15",
            }
        if node_type == "Output":
            return {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "#ff0000",
            }
        if node_type == "Model":
            return {
                "shape": "folder",
                "style": "filled",
                "fillcolor": "#1d73ff",
            }
        if node_type == "Parameter":
            return {
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "#ff9900",
            }
        if node_type == "Constant":
            return {
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "#00e5ff",
            }
        return {
            "shape": "box",
            "style": "filled",
            "fillcolor": "#cdcdcd",
        }

    def _add_node(node):
        nid = node.name
        if nid in added_nodes:
            return

        style = _node_style(node.node_type)
        dot.node(
            nid,
            label=f"{node.name}\n{node.node_type}",
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
    for node in model._order:
        _add_node(node)

    # Ensure top-level inputs/outputs are always present
    for node in model.inputs:
        _add_node(node)
    for node in model.outputs:
        _add_node(node)

    # Add normal DAG edges
    for node in model._order:
        for pred in getattr(node, "predecessors", []) or []:
            _add_node(pred)
            shape_label = str(pred.shape) if hasattr(pred, "shape") else ""
            _add_edge(pred.name, node.name, label=shape_label)

    # Add minimizers
    if include_minimizers:
        for i, m in enumerate(model._minimizers):
            loss_name = m.get("loss", "loss")
            min_name = m.get("name", f"loss_{i}")

            source = m.get("source")
            target = m.get("target")

            source_name = (
                source
                if isinstance(source, str)
                else getattr(source, "name", str(source))
            )
            target_name = (
                target
                if isinstance(target, str)
                else getattr(target, "name", str(target))
                if target is not None
                else None
            )

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
                _add_edge(
                    target_name,
                    loss_node_id,
                    color=loss_fill,
                    style="dashed",
                    penwidth="1.6",
                )

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


def export_html(
    model: Modely,
    out_dir: str | os.PathLike,
    filename: str | None = None,
    *,
    open_subgraph_in_new_tab: bool = False,
    physics: bool = True,
) -> str:
    """
    Export this Modely DAG to interactive HTML using vis-network.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    visited_pages: set[tuple[int, str, bool]] = set()

    def _slug(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[^a-z0-9._-]+", "-", s)
        s = re.sub(r"-{2,}", "-", s).strip("-")
        return s or "graph"

    def _node_color(kind: str) -> str:
        if kind == "Input":
            return "#2ecc71"
        if kind == "Output":
            return "#e74c3c"
        if kind == "Model":
            return "#3498db"
        if kind == "Parameter":
            return "#ff9900"
        if kind == "Constant":
            return "#00e5ff"
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
                    attrs[attr] = str(getattr(node, attr))
                except Exception:
                    pass

        # layer = getattr(node, "_layer", None)
        # if layer is not None:
        #     try:
        #         attrs["keras_layer"] = type(layer).__name__
        #     except Exception:
        #         attrs["keras_layer"] = str(layer)

        # if getattr(node, "node_type", None) == "Parameter":
        #     try:
        #         attrs["trainable"] = True
        #     except Exception:
        #         pass

        # if getattr(node, "node_type", None) == "Constant":
        #     try:
        #         attrs["trainable"] = False
        #     except Exception:
        #         pass

        if hasattr(node, "_properties"):
            try:
                attrs["properties"] = dict(getattr(node, "_properties"))
            except Exception:
                pass

        if getattr(node, "node_type", None) == "Loop":
            try:
                attrs["properties"]["f"] = attrs["properties"]["f"].name
                attrs["properties"]["closed_loop"] = {
                    out.name: inp.name
                    for out, inp in attrs["properties"]["closed_loop"].items()
                }
            except Exception:
                pass
        # if getattr(node, "node_type", None) == "Model":
        #     submodel = getattr(node, "model", None)
        #     if submodel is not None:
        #         attrs["submodel"] = getattr(submodel, "name", type(submodel).__name__)

        #     output_name = getattr(node, "output_name", None)
        #     if output_name is not None:
        #         attrs["output_name"] = output_name

        #     input_map = getattr(node, "input_map", None)
        #     if input_map:
        #         try:
        #             attrs["input_map"] = {
        #                 k: getattr(v, "name", str(v)) for k, v in input_map.items()
        #             }
        #         except Exception:
        #             attrs["input_map"] = str(input_map)

        return attrs

    def _collect_graph(model_obj):
        nodes = []
        edges = []
        seen_nodes = set()
        seen_edges = set()

        for node in model_obj._order:
            nid = getattr(node, "name", str(node))
            if nid in seen_nodes:
                continue
            kind = node.node_type  # if hasattr(node, "node_type") else "Other"
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

    def _export_one(
        model_obj,
        html_name: str,
        page_title: str,
        *,
        parent_rel: str | None = None,
        flattened: bool = False,
        standard_rel: str | None = None,
        flattened_rel: str | None = None,
    ) -> str:
        page_key = (id(model_obj), html_name, flattened)
        file_path = out_path / html_name

        if page_key in visited_pages and file_path.exists():
            return str(file_path)
        visited_pages.add(page_key)

        render_model = model_obj.flatten() if flattened else model_obj
        graph_nodes, graph_edges = _collect_graph(render_model)

        # nested pages only from non-flattened graphs
        url_map: dict[str, str] = {}
        if not flattened:
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
                    flattened=False,
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
            # elif kind in ("Parameter", "Constant", "Input", "Output"):
            #     rec["shape"] = "ellipse"
            #     rec["size"] = 16

            if nid in url_map:
                rec["url"] = url_map[nid]

            vis_nodes.append(rec)

        vis_edges = []
        for pred, node in graph_edges:
            src = getattr(pred, "name", str(pred))
            dst = getattr(node, "name", str(node))
            edge_attrs = {
                "from class": type(pred).__name__,
                "to class": type(node).__name__,
                # "from_type": getattr(pred, "node_type", None),
                # "to_type": getattr(node, "node_type", None),
                "shape": str(getattr(pred, "shape", "")),
            }

            vis_edges.append(
                {
                    "from": src,
                    "to": dst,
                    "arrows": "to",
                    "label": str(getattr(pred, "shape", "")),
                    "font": {"align": "middle", "size": 10},
                    "title": json.dumps(
                        edge_attrs, ensure_ascii=False, indent=2, default=str
                    ),
                    "color": {"color": "#7f8c8d"},
                }
            )

        target = "_blank" if open_subgraph_in_new_tab else "_self"

        back_html = ""
        if parent_rel is not None:
            back_html = f"""
            <button id="backBtn" class="btn" title="Go back to parent graph">← Back</button>
            <a class="crumb" href="{parent_rel}">Parent</a>
            """

        flat_button_html = ""
        if not flattened and flattened_rel is not None:
            flat_button_html = f"""
            <a class="toggle-btn" href="{flattened_rel}">Flatten</a>
            """
        elif flattened and standard_rel is not None:
            flat_button_html = f"""
            <a class="toggle-btn" href="{standard_rel}">Standard</a>
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
    .header-left {{
        justify-self: start;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }}
    .header-right {{
        justify-self: end;
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
        height: 90vh;
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
    <!--
    .hint {{
        padding: 8px 16px;
        font-size: 16px;
        color: #555;
        border-top: 1px solid #eee;
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
    }}
    -->
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
    .bottom-bar {{
        height: 4vh;
        display: flex;
        justify-content: center;
        align-items: center;
        background: #fff;
    }}
    .toggle-btn {{
        border: 1px solid #ddd;
        background: #3498db;
        padding: 8px 14px 8px 14px;
        border-radius: 12px;
        cursor: pointer;
        font-size: 16px;
        color: #fafafa;
        text-decoration: none;
    }}
    .toggle-btn:hover {{
        background: #2563eb;
    }}
</style>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
<header>
    <div class="header-left">
        <img src="../imgs/logo_info.png" alt="nnodely logo" width="120" height="40"/>
        <h1>{page_title}{" [flattened]" if flattened else ""}</h1>
        {back_html}
    </div>
    <div class="legend">
        <span><span class="swatch" style="background:#2ecc71"></span>Input</span>
        <span><span class="swatch" style="background:#e74c3c"></span>Output</span>
        <span><span class="swatch" style="background:#3498db"></span>Submodel</span>
        <span><span class="swatch" style="background:#ff9900"></span>Parameter</span>
        <span><span class="swatch" style="background:#00e5ff"></span>Constant</span>
        <span><span class="swatch" style="background:#95a5a6"></span>Relation</span>
    </div>
    <div class="header-right">
        <div class="bottom-bar">
            {flat_button_html}
        </div>
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

<!--
<div class="hint">
    <span>Powered by </span>
    <img src="../imgs/logo_info.png" alt="nnodely logo" width="120" height="40"/>
</div>
-->

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
            shadow: {str(physics).lower()},
            font: {{
                size: 10,
                align: "middle"
            }}
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

    root_base = _slug(filename if filename else model.name)
    root_name = f"{root_base}.html"
    flat_name = f"{root_base}__flattened.html"

    # always export flattened partner page too
    flat_path = _export_one(
        model,
        flat_name,
        model.name,
        parent_rel=root_name,
        flattened=True,
        standard_rel=root_name,
        flattened_rel=None,
    )

    return _export_one(
        model,
        root_name,
        model.name,
        parent_rel=None,
        flattened=False,
        standard_rel=None,
        flattened_rel=os.path.relpath(flat_path, start=out_path),
    )
