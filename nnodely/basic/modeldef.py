import networkx as nx
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Hashable, Optional, Set

from nnodely.basic.graphutils import plot_graphviz_structure

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)
from functools import wraps


def check_neuralized(func):
    """Decorator for ModelGraph methods that modify the graph.

    After the wrapped method runs successfully, sets ``self.neuralized`` to False.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        try:
            self.neuralized = False
        except Exception:
            pass
        return result
    return wrapper

class ModelGraph:
    def __init__(self, name: str = "main"):
        self.model_graph = nx.DiGraph()
        self.name = name
        self.tags = []
        self.counter = 0
        self.neuralized = False
    
    @check_neuralized
    def _clear(self, tags:str|list|None = None):
        if tags is None:
            self.model_graph.clear()
            self.tags = []
            self.counter = 0
        else:
            tags = [tags] if isinstance(tags, str) else tags
            for tag in tags:
                self.removeNode(tag)

    @check_neuralized
    def set_node(self, name, **attrs):
        while name in self.tags:
            self.counter += 1
            name = f"{name}_{self.counter}"
        self.tags.append(name)
        self.model_graph.add_node(name, **attrs)
        return name

    @check_neuralized
    def set_edge(self, from_node, to_node, **attrs):
        if self.get_node(from_node) is None:
            return 
            #raise NameError(f"The node '{from_node}' is not defined.")
        if self.get_node(to_node) is None:
            return
            #raise NameError(f"The node '{to_node}' is not defined.")
        self.model_graph.add_edge(from_node, to_node, **attrs)

    def get_node(self, name):
        return self.model_graph.nodes.get(name, None)

    def get_edge(self, from_node, to_node):
        return self.model_graph.edges.get((from_node, to_node), None)

    @check_neuralized
    def set_node_attr(self, node_name, **attrs):
        for k, v in attrs.items():
            self.model_graph.nodes[node_name][k] = v
    
    @check_neuralized
    def set_edge_attr(self, from_node, to_node, **attrs):
        for k, v in attrs.items():
            self.model_graph.edges[from_node, to_node][k] = v
    
    def get_node_attr(self, node_name, attr_name):
        return self.model_graph.nodes[node_name].get(attr_name, None)
    
    def get_edge_attr(self, from_node, to_node, attr_name):
        return self.model_graph.edges[from_node, to_node].get(attr_name, None)
    
    @check_neuralized
    def removeEdge(self, from_node, to_node):
        if self.get_node(from_node) is None:
            print(f"The node '{from_node}' is not defined.")
            return
        if self.get_node(to_node) is None:
            print(f"The node '{to_node}' is not defined.")
            return
        if self.get_edge(from_node, to_node) is None:
            print(f"The edge '{from_node}->{to_node}' is not defined.")
            return
        self.model_graph.remove_edge(from_node, to_node)

    def is_neuralized(self):
        return self.neuralized

    @check_neuralized
    def removeNode(self, name):
        if self.get_node(name) is None:
            print(f"The node '{name}' is not defined.")
            return
        # Collect incident edges first (copy lists because we'll modify the graph)
        incident_in = list(self.model_graph.in_edges(name))
        incident_out = list(self.model_graph.out_edges(name))
        # Remove incident edges using removeEdge (keeps consistent checks/behavior)
        for a, b in incident_in + incident_out:
            self.removeEdge(a, b)
        # Finally remove the node itself and its tag entry
        self.model_graph.remove_node(name)
        if name in self.tags:
            self.tags.remove(name)
    
    def subgraph(self, node_name) -> nx.DiGraph:
        if self.get_node(node_name) is None:
            raise NameError(f"The node '{node_name}' is not defined.")
        # networkx.ancestors returns all nodes that have a path to node_name
        ancestors = nx.ancestors(self.model_graph, node_name)
        # include target node itself
        nodes = set(ancestors) | {node_name}
        # return a copy of the induced subgraph as a DiGraph
        return self.model_graph.subgraph(nodes).copy()
    
    def plot_graph(self, filename:str, view: bool = False, left_to_right: bool = False, prog: str = 'dot'):
        import matplotlib.pyplot as plt
        # Determine node positions. Prefer Graphviz (dot) for a left-to-right layout
        pos = None
        if left_to_right:
            try:
                from networkx.drawing.nx_agraph import graphviz_layout
                pos = graphviz_layout(self.model_graph, prog=prog, args='-Grankdir=LR')
            except Exception:
                try:
                    from networkx.drawing.nx_pydot import graphviz_layout as pydot_graphviz_layout
                    pos = pydot_graphviz_layout(self.model_graph, prog=prog, args='-Grankdir=LR')
                except Exception:
                    pos = None

        if pos is None:
            # fallback: use a simple layered layout based on distance from input nodes
            try:
                inputs = [n for n, a in self.model_graph.nodes(data=True) if a.get('type') == 'Input']
                if not inputs:
                    raise ValueError
                # compute min distance from any input for each node
                layer = {n: float('inf') for n in self.model_graph.nodes()}
                for src in inputs:
                    lengths = nx.single_source_shortest_path_length(self.model_graph, src)
                    for n, d in lengths.items():
                        if d < layer.get(n, float('inf')):
                            layer[n] = d
                # assign positions: x = layer, y spaced by index within layer
                layers = {}
                for n, d in layer.items():
                    layers.setdefault(d, []).append(n)
                pos = {}
                max_nodes_in_layer = max(len(v) for v in layers.values()) if layers else 1
                for x, (d, nodes) in enumerate(sorted(layers.items())):
                    for i, n in enumerate(sorted(nodes)):
                        # center vertically
                        y = (i - (len(nodes) - 1) / 2.0) / max_nodes_in_layer
                        pos[n] = (x, -y)
            except Exception:
                pos = nx.spring_layout(self.model_graph)
                
        node_colors = []
        for n, attrs in self.model_graph.nodes(data=True):
            ntype = attrs.get("type")
            if ntype == "Input":
                node_colors.append('lightgreen')
            elif ntype == "Constant":
                node_colors.append('lightblue')
            elif ntype == "Parameter":
                node_colors.append('orange')
            elif ntype == "Function":
                node_colors.append('yellow')
            elif ntype == "Output":
                node_colors.append('red')
            elif ntype == "Minimize":
                node_colors.append('purple')
            else:
                node_colors.append('lightgrey')
        # Build labels that include node name and its attributes
        labels = {}
        for n, attrs in self.model_graph.nodes(data=True):
            # attr_lines = []
            # for k, v in attrs.items():
            #     attr_lines.append(f"{k}: {v}")
            labels[n] = n + "\n" #+ "\n".join(attr_lines) if attr_lines else n

        # Draw graph using the composed labels
        nx.draw(self.model_graph, pos, labels=labels, node_color=node_colors, arrows=True)
        plt.title(f"Neural Network Diagram", fontsize=12, fontweight='bold')
        ## Save the figure
        plt.savefig(filename, format="png", bbox_inches='tight')
        if view:
            plt.show()

    def to_graph(model_json):
        from graphutils import to_graph
        return to_graph(model_json)

    def to_json(G):
        from graphutils import to_json
        return to_json(G)

    
# _CURRENT_MODEL_GRAPH = None

# def set_current_model_graph(mg : ModelGraph) -> None: 
#     """Register the active ModelGraph instance used by Input/Relation helpers."""
#     global _CURRENT_MODEL_GRAPH
#     _CURRENT_MODEL_GRAPH = mg

# def get_current_model_graph() -> ModelGraph | None:
#     """Return the currently registered ModelGraph or None."""
#     return _CURRENT_MODEL_GRAPH
    
class ModelManager:
    def __init__(self, name: str):
        self.models = {name: ModelGraph(name=name)}
        self.names = [name]
        self.current = name
        #set_current_model_graph(self.models[name])

    def set_current_model(self, name:str) -> None:
        if name not in self.models:
            raise KeyError(f"model '{name}' not found in models")
        self.current = name

    def get_current_model(self) -> ModelGraph:
        return self.models[self.current]

    def add_model(self, name:str, model:ModelGraph) -> None:
        while name in self.names:
            self.counter += 1
            name = f"{name}_{self.counter}"
        self.names.append(name)
        self.models[name] = model
        return name

    def flatten(self, start: str|None = None) -> ModelGraph:
        # from nnodely.basic.modeldef_merge_helper import merge_models_from_manager
        # return merge_models_from_manager(self, start=start)
    
        if start == None or start not in self.models:
            log.warning(f"Model '{start}' not found in manager. Using current model '{self.current}' instead.")
            start = self.current

        #merged = nx.DiGraph()
        merged = ModelGraph()
        visited = set()

        def entry_nodes(mg: ModelGraph, prefix: str):
            return [prefix + n for n, d in mg.model_graph.nodes(data=True)
                    if mg.model_graph.in_degree(n) == 0 or d.get('type') == 'Input']

        def exit_nodes(mg: ModelGraph, prefix: str):
            return [prefix + n for n, d in mg.model_graph.nodes(data=True)
                    if mg.model_graph.out_degree(n) == 0 or d.get('type') == 'Output']

        def _add_graph(mg: ModelGraph, prefix: str = ''):
            if id(mg) in visited:
                return
            visited.add(id(mg))

            G = mg.model_graph

            # add nodes (recurse into containers)
            for n, attrs in G.nodes(data=True):
                if 'graph' in attrs and isinstance(attrs['graph'], ModelGraph):
                    child_mg = attrs['graph']
                    child_prefix = prefix + n + '_'
                    _add_graph(child_mg, child_prefix)
                else:
                    merged.set_node(prefix + n, **attrs)

            # add edges, rewiring container endpoints
            for u, v, ed in G.edges(data=True):
                u_attrs = G.nodes[u]
                v_attrs = G.nodes[v]
                u_is_container = 'graph' in u_attrs and isinstance(u_attrs['graph'], ModelGraph)
                v_is_container = 'graph' in v_attrs and isinstance(v_attrs['graph'], ModelGraph)

                if not u_is_container and not v_is_container:
                    merged.set_edge(prefix + u, prefix + v, **ed)
                else:
                    sources = [prefix + u] if not u_is_container else exit_nodes(u_attrs['graph'], prefix + u + '_')
                    targets = [prefix + v] if not v_is_container else entry_nodes(v_attrs['graph'], prefix + v + '_')
                    for s in sources:
                        for t in targets:
                            merged.set_edge(s, t, **ed)

        _add_graph(self.models[start], prefix='')
        return merged

    def export_html(self,
        model: str,
        out_dir: str | os.PathLike,
        *,
        open_subgraph_in_new_tab: bool = False,
        physics: bool = True,
    ) -> str:
        """
        Export a NetworkX graph to interactive HTML (vis-network) with recursive subgraph pages.
        Adds a "Back" button on subgraph pages to return to the parent page.

        Returns the root HTML file path as a string.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        visited: Set[int] = set()

        def _slug(s: str) -> str:
            s = s.strip().lower()
            s = re.sub(r"[^a-z0-9._-]+", "-", s)
            s = re.sub(r"-{2,}", "-", s).strip("-")
            return s or "graph"


        def _node_color(attrs: Dict[str, Any], attr_name: str) -> str:
            t = attrs.get("type", None)
            if t == "Input":
                return "#2ecc71"  # green
            if t == "Output":
                return "#e74c3c"  # red
            if attrs.get(attr_name) is not None:
                return "#3498db"  # blue
            if t == 'Parameter':
                return "#f1c40f"  # yellow
            if t == 'Constant':
                return "#000000"  # black
            if t == 'Function':
                return "#9b59b6"  # purple
            if t == 'Minimize':
                return "#e67e22"  # carrot
            return "#95a5a6"      # gray

        def _export_one(graph: nx.Graph, html_name: str, page_title: str, parent_rel: Optional[str] = None,) -> str:
            gid = id(graph)
            file_path = out_path / html_name

            # If already exported, still ensure we wrote the file (maybe first time parent differs),
            # but generally avoid infinite recursion.
            if gid in visited and file_path.exists():
                return str(file_path)
            visited.add(gid)

            # Create stable per-page IDs
            id_map: Dict[Hashable, str] = {}
            used_ids: Set[str] = set()
            for n in graph.nodes():
                sid = str(n)
                if sid in used_ids:
                    k = 1
                    base = sid
                    while f"{base}#{k}" in used_ids:
                        k += 1
                    sid = f"{base}#{k}"
                used_ids.add(sid)
                id_map[n] = sid

            # Export subgraphs first
            url_map: Dict[str, str] = {}
            for n, attrs in graph.nodes(data=True):
                payload = attrs.get("graph", None)
                if payload is None:
                    continue
                subG = payload.model_graph

                sub_name = f"{_slug(page_title)}_{_slug(str(n))}.html"
                sub_title = f"{page_title} :: {n}"
                parent_for_child = os.path.relpath(file_path, start=out_path)
                sub_path = _export_one(
                    subG,
                    sub_name,
                    sub_title,
                    parent_rel=parent_for_child,
                )
                url_map[id_map[n]] = os.path.relpath(sub_path, start=out_path)

            # Nodes for vis-network
            nodes = []
            for n, attrs in graph.nodes(data=True):
                sid = id_map[n]
                attrs_dict = dict(attrs)
                try:
                    tooltip = json.dumps(attrs_dict, ensure_ascii=False, default=str, indent=2)
                except Exception:
                    tooltip = str(attrs_dict)

                node_rec = {
                    "id": sid,
                    "label": str(n),
                    "title": f"{tooltip}",
                    "shape": "dot",
                    "size": 14,
                    "color": {
                        "background": _node_color(attrs_dict, "graph"),
                        "border": "#2c3e50",
                        "highlight": {"background": "#f1c40f", "border": "#2c3e50"},
                    },
                    "font": {"color": "#111111"},
                }
                if sid in url_map:
                    node_rec["shape"] = "diamond"
                    node_rec["size"] = 16
                    node_rec["url"] = url_map[sid]
                nodes.append(node_rec)

            # Edges
            edges = []
            for u, v, eattrs in graph.edges(data=True):
                rec = {
                    "from": id_map[u],
                    "to": id_map[v],
                    "title": f"{json.dumps(dict(eattrs), ensure_ascii=False, default=str, indent=2)}",
                }
                # default arrow
                rec["arrows"] = "to"

                # style by edge 'type' attribute
                etype = eattrs.get("type") if isinstance(eattrs, dict) else None
                if etype == 'connect':
                    # dashed blue line with label 'connect'
                    rec["dashes"] = True
                    rec["color"] = {"color": "#3498db"}
                    rec["label"] = "connect"
                    rec["font"] = {"color": "#111111", "align": "middle"}
                elif etype == 'loop':
                    # dashed red line with label 'loop'
                    rec["dashes"] = True
                    rec["color"] = {"color": "#e74c3c"}
                    rec["label"] = "loop"
                    rec["font"] = {"color": "#111111", "align": "middle"}

                edges.append(rec)

            # Back button (only when parent exists)
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
                    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
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
                    width: 10px; height: 10px; border-radius: 3px; display: inline-block; border: 1px solid #2c3e50;
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
                    <span><span class="swatch" style="background:#3498db"></span>Sub-Model</span>
                    <span><span class="swatch" style="background:#95a5a6"></span>Relation</span>
                    <span><span class="swatch" style="background:#f1c40f"></span>Parameter</span>
                    <span><span class="swatch" style="background:#000000"></span>Constant</span>
                    <span><span class="swatch" style="background:#9b59b6"></span>Function</span>
                    <span><span class="swatch" style="background:#e67e22"></span>Minimize</span>
                    </div>
                </header>

                <div id="network"></div>

                <div class="hint">
                    <span>Hover nodes/edges for attributes.</span>
                    <span>Click blue/diamond nodes to open subgraphs.</span>
                    {"<span>Use ← Back to return to the parent graph.</span>" if parent_rel is not None else ""}
                </div>

                <script>
                    const nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=False)});
                    const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});

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

                    // Back button support
                    const backBtn = document.getElementById("backBtn");
                    if (backBtn) {{
                    backBtn.addEventListener("click", () => {{
                        // Prefer browser history if it actually goes somewhere,
                        // but always provide a safe fallback to the parent page.
                        const parentUrl = {json.dumps(parent_rel)};
                        if (window.history.length > 1) {{
                        window.history.back();
                        // fallback if history back doesn't move (rare):
                        setTimeout(() => {{
                            // If still on same page URL after a moment, go to parentUrl.
                            // (Simple heuristic: just navigate anyway if needed.)
                        }}, 150);
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

        root_path = _export_one(self.models[model].model_graph, model+'.html', model, parent_rel=None)
        return root_path
    
    def plot_structure(self, model:str, filename='nnodely_graph', library='matplotlib', flatten=False, view=False):
        if library not in ['matplotlib', 'graphviz']:
            raise ValueError("Invalid library specified. Use 'matplotlib' or 'graphviz'.")
        if model not in self.models:
            raise KeyError(f"model '{model}' not found in models")
        if library == 'matplotlib':
            graph = self.flatten(start=model) if flatten else self.models[model]
            graph.plot_graph(filename=filename, view=view, left_to_right=True)
        elif library == 'graphviz':
            graph = self.flatten(start=model).model_graph if flatten else self.models[model].model_graph
            plot_graphviz_structure(graph, filename, view=view)

# layout: {{
#     hierarchical: {{
#         enabled: true,
#         direction: 'LR',
#         sortMethod: 'directed',
#         levelSeparation: 150,
#         nodeSpacing: 120
#     }}