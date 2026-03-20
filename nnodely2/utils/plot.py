
import os
from pathlib import Path
import json
import re
from typing import Optional

from nnodely.model import Model

def export_html(
    self,
    out_dir: str | os.PathLike,
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
    import json
    import os
    import re
    from pathlib import Path

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
            "name": getattr(node, "name", str(node)),
            "kind": kind,
            "class": type(node).__name__,
        }

        for attr in ("node_type", "seq", "time", "dim"):
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
                pass

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

    root_name = f"{_slug(self.name)}.html"
    return _export_one(self, root_name, self.name, parent_rel=None)

def make_html(
    model: Model,
    out_dir: str | os.PathLike,
    filename: Optional[str] = None,
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
            "name": getattr(node, "name", str(node)),
            "kind": kind,
            "class": type(node).__name__,
        }

        for attr in ("node_type", "seq", "time", "dim"):
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
                pass

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
        for node, nid, kind, attrs in graph_nodes:
            try:
                tooltip = json.dumps(attrs, ensure_ascii=False, indent=2, default=str)
            except Exception:
                tooltip = str(attrs)

            rec = {
                "id": nid,
                "label": nid,
                "title": tooltip,
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
    #network {{
      width: 100%;
      height: 92vh;
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

  <div id="network"></div>

  <div class="hint">
    <span>Hover nodes and edges for details.</span>
    <span>Click blue diamond nodes to open nested models.</span>
    {"<span>Use ← Back to return to the parent graph.</span>" if parent_rel is not None else ""}
  </div>

  <script>
    const nodes = new vis.DataSet({json.dumps(vis_nodes, ensure_ascii=False)});
    const edges = new vis.DataSet({json.dumps(vis_edges, ensure_ascii=False)});

    const container = document.getElementById('network');
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

    network.on("click", function(params) {{
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

    root_name = f"{_slug(filename if filename else model.name)}.html"
    return _export_one(model, root_name, model.name, parent_rel=None)