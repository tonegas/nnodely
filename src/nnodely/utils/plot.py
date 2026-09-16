import os
import re
import json
import shutil
from pathlib import Path
from nnodely.core.modely import Modely, ModelCall

from nnodely.layers.input import Input
from nnodely.layers.loop import Loop
from nnodely.layers.roll import Roll

## nnodely pallette
# :root {
#   --color-primary: #6C75DF;   /* modely blue-purple */
#   --color-secondary: #89C1F9; /* light sky blue */
#   --color-accent: #81ACF4;    /* soft gradient blue */
#   --color-text: #000000;      /* logo black */
# }


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
        if node_type == "Output" or node_type == "IntermediateOutput":
            return {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "#ff0000",
            }
        if node_type == "ModelCall":
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

        style = _node_style(node.__class__.__name__)
        dot.node(
            nid,
            label=f"{node.name}\n{node.__class__.__name__}",
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
    for node in model.order:
        _add_node(node)

    # Ensure top-level inputs/outputs are always present
    for node in model.inputs:
        _add_node(node)
    for node in model.outputs:
        _add_node(node)

    # Add normal DAG edges
    for node in model.order:
        for pred in getattr(node, "preds", []) or []:
            _add_node(pred)
            shape_label = str(pred.shape.tuple) if hasattr(pred, "shape") else ""
            _add_edge(pred.name, node.name, label=shape_label)

    # Add minimizers
    if include_minimizers:
        for i, m in enumerate(model.minimizers):
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


def _copy_logo(out_path: Path) -> str | None:
    """Place the header logo next to the pages, returning its relative URL."""
    source = Path(__file__).resolve().parents[3] / "imgs" / "logo_info.png"
    if not source.is_file():
        return None
    target = out_path / "imgs" / "logo_info.png"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return "imgs/logo_info.png"


def _block_ports(node):
    """The model a block node wraps, plus how that body is bound to this graph.

    A block hides a whole model behind one node, so the drill-down is only
    readable if both pages name the same ports: which body input every
    predecessor feeds, which body output every successor reads, and which
    outputs are fed back inside the block.
    """
    if isinstance(node, ModelCall):
        return node.model, {
            "block": "ModelCall",
            "detail": {},
            "inputs": [
                {"body": body.name, "outer": outer.name, "role": "input"}
                for body, outer in node.inputs_map.items()
            ],
            "outputs": [
                {"body": body.name, "outer": outer.name}
                for outer, body in node.outputs_map.items()
            ],
            "feedback": [],
        }

    if isinstance(node, Loop):
        return node.f, {
            "block": "Loop",
            "detail": {"steps": node.horizon, "collect": node.collect},
            "inputs": [
                {"body": body.name, "outer": outer.name, "role": "initial"}
                for body, outer in zip(node.callback_inputs, node.initial_values)
            ]
            + [
                {"body": body.name, "outer": outer.name, "role": "input"}
                for body, outer in zip(node.static_inputs, node.static_sources)
            ],
            "outputs": [
                {"body": body.name, "outer": outer.name}
                for body, outer in zip(node.f.outputs, node.loop_outputs)
            ],
            "feedback": [
                {"from": output.name, "to": input_node.name}
                for input_node, output in zip(
                    node.callback_inputs, node.callback_outputs
                )
            ],
        }

    if isinstance(node, Roll):
        # Roll wires the body's own inputs, so a port's outer name is its own.
        return node.f, {
            "block": "Roll",
            "detail": {"steps": node.steps},
            "inputs": [
                {
                    "body": node.callback_input.name,
                    "outer": node.callback_input.name,
                    "role": "state",
                }
            ]
            + [
                {"body": body.name, "outer": body.name, "role": "input"}
                for body in node.static_inputs
            ],
            "outputs": [{"body": node.callback_output.name, "outer": node.name}],
            "feedback": [
                {"from": node.callback_output.name, "to": node.callback_input.name}
            ],
        }

    return None


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

    ``out_dir`` is the folder the pages are written to; a path ending in
    ``.html`` names the root page instead and its parent becomes the folder.

    Every block that wraps a model (``ModelCall``, ``Loop``, ``Roll``) is
    exported as its own page, linked from the block node and annotated with the
    ports that bind the body to the graph above it.
    """
    out_path = Path(out_dir)
    if out_path.suffix.lower() == ".html":
        filename = filename or out_path.stem
        out_path = out_path.parent
    out_path.mkdir(parents=True, exist_ok=True)
    logo_rel = _copy_logo(out_path)

    visited_pages: set[tuple[int, str, bool]] = set()

    def _slug(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[^a-z0-9._-]+", "-", s)
        s = re.sub(r"-{2,}", "-", s).strip("-")
        return s or "graph"

    def _node_color(kind: str) -> str:
        if kind == "Input":
            return "#2ecc71"
        if kind == "Output" or kind == "IntermediateOutput":
            return "#e74c3c"
        if kind in ("ModelCall", "Loop", "Roll"):
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
        }

        for attr in ("seq", "time", "dim", "shape"):
            if hasattr(node, attr):
                try:
                    attrs[attr] = str(getattr(node, attr))
                except Exception:
                    pass

        if hasattr(node, "_properties"):
            try:
                attrs["properties"] = dict(getattr(node, "_properties"))
            except Exception:
                pass

        return attrs

    def _collect_graph(model_obj, *, inline=False, prefix="", alias=None):
        """Nodes, edges and feedback arrows of a graph, blocks optionally spliced.

        With ``inline`` every block is replaced by the body it wraps, so the
        flattened view answers what the model actually computes. ``Modely.flatten``
        cannot do this itself: ``build`` runs the very same pass, and a Loop or a
        Roll has to survive it as one node to become a single Keras layer. A
        rollout is spliced once and its recurrence drawn as a feedback arrow, so
        the picture stays the same size whatever the horizon.

        Body ids are prefixed with the block that owns them, which keeps two
        calls of one model apart and says on the label where a node came from.
        """
        nodes: list = []
        edges: list = []
        feedbacks: list = []
        seen_nodes: set[str] = set()
        seen_edges: set[tuple[str, str]] = set()
        alias = alias or {}

        def outer_id(name: str) -> str:
            """Display id of a node of this graph, by name."""
            return alias.get(name, prefix + name)

        def ident(node) -> str:
            """Display id of a node of this graph."""
            return outer_id(getattr(node, "name", str(node)))

        def add_edge(pred, src: str, dst: str, role: str | None = None) -> None:
            if (src, dst) in seen_edges:
                return
            edges.append((pred, src, dst, role))
            seen_edges.add((src, dst))

        blocks = {}
        # How each spliced block hands its results on: the selector nodes that
        # read one output each, and the body output the block node itself
        # stands for when something reads it as a value instead.
        block_results: dict[str, dict] = {}
        present = {getattr(node, "name", None) for node in model_obj.order}
        if inline:
            blocks = {
                node.name: _block_ports(node)
                for node in model_obj.order
                if _block_ports(node) is not None
            }

        for node in model_obj.order:
            name = getattr(node, "name", str(node))
            ports = blocks.get(name)

            if ports is not None:
                submodel, port_map = ports
                body_prefix = f"{prefix}{name}/"

                # A Roll wires the body's own inputs, so its ports name the very
                # same node on both sides: alias those instead of duplicating them.
                body_alias = {
                    port["body"]: outer_id(port["outer"])
                    for port in port_map["inputs"]
                    if port["body"] == port["outer"]
                }
                body_nodes, body_edges, body_feedbacks = _collect_graph(
                    submodel, inline=True, prefix=body_prefix, alias=body_alias
                )

                def body_id(body_name: str) -> str:
                    return body_alias.get(body_name, body_prefix + body_name)

                for body_node, nid, kind, attrs in body_nodes:
                    if nid in seen_nodes:
                        continue
                    attrs["inlined_in"] = prefix + name
                    nodes.append((body_node, nid, kind, attrs))
                    seen_nodes.add(nid)
                for body_pred, src, dst, role in body_edges:
                    add_edge(body_pred, src, dst, role)
                feedbacks.extend(body_feedbacks)

                # Bind the enclosing graph to the body it now contains.
                for port in port_map["inputs"]:
                    outer, body = outer_id(port["outer"]), body_id(port["body"])
                    if outer != body:
                        add_edge(None, outer, body, port["role"])
                block_results[name] = {
                    "selectors": {port["outer"] for port in port_map["outputs"]},
                    "primary": body_id(port_map["outputs"][0]["body"]),
                }
                for port in port_map["outputs"]:
                    # A selector absent from this graph means the block is read
                    # as a value: a single-output Loop, or any Roll.
                    if port["outer"] == name or port["outer"] not in present:
                        continue
                    outer, body = outer_id(port["outer"]), body_id(port["body"])
                    if outer != body:
                        add_edge(None, body, outer, "output")

                steps = port_map["detail"].get("steps")
                for pair in port_map["feedback"]:
                    feedbacks.append(
                        (
                            body_id(pair["from"]),
                            body_id(pair["to"]),
                            f"feedback (x{steps} steps)" if steps else "feedback",
                            {
                                "kind": "feedback",
                                "block": port_map["block"],
                                "block_node": prefix + name,
                                "feedback_stream": pair["from"],
                                "feedback_input": pair["to"],
                                **port_map["detail"],
                            },
                        )
                    )
                continue

            nid = ident(node)
            if nid in seen_nodes:
                continue
            nodes.append((node, nid, node.__class__.__name__, _safe_attrs(node)))
            seen_nodes.add(nid)

        for node in model_obj.order:
            if getattr(node, "name", None) in blocks:
                continue
            for pred in getattr(node, "preds", []) or []:
                pred_name = getattr(pred, "name", None)
                if pred_name in blocks:
                    # A consumer reads the block either through one of its
                    # selector nodes, already bound above, or straight off the
                    # block node, which now resolves to the body output.
                    results = block_results[pred_name]
                    if getattr(node, "name", None) not in results["selectors"]:
                        add_edge(pred, results["primary"], ident(node))
                    continue
                add_edge(pred, ident(pred), ident(node))

        return nodes, edges, feedbacks

    def _levels(graph_nodes, graph_edges) -> dict[str, int]:
        """Longest-path layer of every node, so the flow reads left to right.

        The levels are assigned here instead of by vis-network because feedback
        arrows close cycles, and its own ``directed`` sort collapses the whole
        graph into a single column as soon as one edge points backwards. Only
        the acyclic edges are passed in, so the recurrence cannot skew them.
        """
        incoming: dict[str, list[str]] = {}
        consumers: dict[str, list[str]] = {}
        for _, src, dst, _ in graph_edges:
            incoming.setdefault(dst, []).append(src)
            consumers.setdefault(src, []).append(dst)

        # The collected nodes are already in topological order, so one forward
        # pass is enough to give each its longest distance from a source.
        level: dict[str, int] = {}
        for _, nid, _, _ in graph_nodes:
            level[nid] = max(
                (level.get(src, 0) + 1 for src in incoming.get(nid, ())), default=0
            )

        # Parameters and constants have no predecessors, so the pass above parks
        # them all in the first column and makes it as tall as the whole graph.
        # Each one belongs beside the node that reads it instead.
        for node, nid, _, _ in graph_nodes:
            if incoming.get(nid) or isinstance(node, Input):
                continue
            read_by = consumers.get(nid)
            if read_by:
                level[nid] = min(level[name] for name in read_by) - 1
        return level

    def _feedback_edge(src: str, dst: str, label: str, attrs: dict) -> dict:
        """A recurrence arrow, drawn apart from the ordinary data flow."""
        return {
            "from": src,
            "to": dst,
            "arrows": {"to": {"enabled": True, "scaleFactor": 1.2}},
            "label": label,
            "width": 3,
            "color": {
                "color": "#e74c3c",
                "highlight": "#c0392b",
                "hover": "#c0392b",
            },
            "font": {
                "align": "middle",
                "size": 12,
                "color": "#c0392b",
                "background": "#ffffff",
            },
            "smooth": {
                "enabled": True,
                "type": "curvedCW",
                "roundness": 0.35,
            },
            "title": json.dumps(attrs, ensure_ascii=False, indent=2, default=str),
        }

    def _export_one(
        model_obj,
        html_name: str,
        page_title: str,
        *,
        parent_rel: str | None = None,
        flattened: bool = False,
        standard_rel: str | None = None,
        flattened_rel: str | None = None,
        boundary: dict | None = None,
    ) -> str:
        page_key = (id(model_obj), html_name, flattened)
        file_path = out_path / html_name

        if page_key in visited_pages and file_path.exists():
            return str(file_path)
        visited_pages.add(page_key)

        render_model = model_obj.flatten() if flattened else model_obj
        graph_nodes, graph_edges, inlined_feedbacks = _collect_graph(
            render_model, inline=flattened
        )
        levels = _levels(graph_nodes, graph_edges)

        # Flattening inlines ModelCall bodies but leaves Loop and Roll intact,
        # so a flattened page still has blocks worth drilling into.
        url_map: dict[str, str] = {}
        for node, nid, kind, attrs in graph_nodes:
            ports = _block_ports(node)
            if ports is None:
                continue
            submodel, port_map = ports
            attrs["nested_model"] = submodel.name
            attrs["block"] = port_map

            sub_name = f"{_slug(page_title)}__{_slug(nid)}.html"
            sub_title = f"{page_title} :: {nid}"
            parent_for_child = os.path.relpath(file_path, start=out_path)

            sub_path = _export_one(
                submodel,
                sub_name,
                sub_title,
                parent_rel=parent_for_child,
                boundary={
                    "node": nid,
                    "parent": page_title,
                    "parent_rel": parent_for_child,
                    "ports": port_map,
                },
            )
            url_map[nid] = os.path.relpath(sub_path, start=out_path)

        # Ports of the block this page is the body of, so the reader can see
        # where the graph above enters and leaves.
        entry_ports: dict[str, dict] = {}
        exit_ports: dict[str, dict] = {}
        if boundary is not None:
            entry_ports = {port["body"]: port for port in boundary["ports"]["inputs"]}
            exit_ports = {port["body"]: port for port in boundary["ports"]["outputs"]}

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
                "title": f"{nid}\n{kind} {attrs.get('shape', '')}",
                "level": levels.get(nid, 0),
            }

            if nid in url_map:
                rec["shape"] = "diamond"
                rec["size"] = 18
                rec["color"]["background"] = "#3498db"
                rec["url"] = url_map[nid]
                rec["title"] = f"{rec['title']}\ndouble-click to open the body"

            port = entry_ports.get(nid) or exit_ports.get(nid)
            if port is not None:
                rec["borderWidth"] = 4
                rec["color"]["border"] = "#f39c12"
                bound = (
                    f"{nid} ← {port['outer']} ({port['role']})"
                    if nid in entry_ports
                    else f"{nid} → {port['outer']}"
                )
                rec["title"] = f"{rec['title']}\nblock port: {bound}"

            vis_nodes.append(rec)

        vis_edges = []
        for pred, src, dst, role in graph_edges:
            edge_attrs = {
                "from": src,
                "to": dst,
                "shape": str(getattr(pred, "shape", "")),
            }
            if role is not None:
                edge_attrs["block port"] = role

            record = {
                "from": src,
                "to": dst,
                "arrows": "to",
                "label": role if role is not None else str(getattr(pred, "shape", "")),
                "font": {"align": "middle", "size": 10},
                "title": json.dumps(
                    edge_attrs, ensure_ascii=False, indent=2, default=str
                ),
                "color": {"color": "#7f8c8d"},
            }
            if role is not None:
                # A binding edge crosses a block boundary that no longer has a
                # node of its own, so it is the only thing marking the seam.
                record["color"] = {"color": "#3498db", "highlight": "#2563eb"}
                record["dashes"] = True
                record["font"] = {"align": "middle", "size": 10, "color": "#2563eb"}
            vis_edges.append(record)

        visible_node_ids = {record["id"] for record in vis_nodes}
        for src, dst, label, attrs in inlined_feedbacks:
            if src in visible_node_ids and dst in visible_node_ids:
                vis_edges.append(_feedback_edge(src, dst, label, attrs))

        for input_node, stream in getattr(model_obj, "_roll_callbacks", {}).items():
            src = stream.name
            dst = input_node.name
            if src not in visible_node_ids or dst not in visible_node_ids:
                continue
            steps = getattr(model_obj, "_roll_steps", None)
            vis_edges.append(
                _feedback_edge(
                    src,
                    dst,
                    f"roll ({steps} steps)",
                    {
                        "kind": "roll",
                        "feedback_stream": src,
                        "feedback_input": dst,
                        "steps": steps,
                    },
                )
            )

        # The recurrence the enclosing block closes over this body.
        if boundary is not None:
            detail = boundary["ports"]["detail"]
            steps = detail.get("steps")
            for pair in boundary["ports"]["feedback"]:
                if (
                    pair["from"] not in visible_node_ids
                    or pair["to"] not in visible_node_ids
                ):
                    continue
                vis_edges.append(
                    _feedback_edge(
                        pair["from"],
                        pair["to"],
                        f"feedback ({steps} steps)" if steps else "feedback",
                        {
                            "kind": "feedback",
                            "block": boundary["ports"]["block"],
                            "block_node": boundary["node"],
                            "feedback_stream": pair["from"],
                            "feedback_input": pair["to"],
                            **detail,
                        },
                    )
                )

        # Training objectives, so the loss wiring is visible here too.
        for index, minimizer in enumerate(getattr(model_obj, "minimizers", [])):
            min_name = minimizer.get("name", f"loss_{index}")
            loss = minimizer.get("loss")
            loss_label = (
                getattr(loss, "name", None)
                or getattr(loss, "__name__", None)
                or type(loss).__name__
            )
            loss_id = f"__min_{min_name}"

            vis_nodes.append(
                {
                    "id": loss_id,
                    "label": f"{min_name}\n{loss_label}",
                    "shape": "hexagon",
                    "size": 18,
                    "color": {
                        "background": "#8e44ad",
                        "border": "#6c3483",
                        "highlight": {"background": "#a569bd", "border": "#6c3483"},
                    },
                    "font": {"color": "#6c3483", "bold": {"color": "#6c3483"}},
                    "title": f"minimize {min_name}\nloss: {loss_label}",
                    "level": max(
                        levels.get(
                            getattr(minimizer.get("source"), "name", ""), 0
                        ),
                        levels.get(
                            getattr(minimizer.get("target"), "name", ""), 0
                        ),
                    )
                    + 1,
                }
            )
            node_details[loss_id] = {
                "class": "Minimizer",
                "name": min_name,
                "loss": loss_label,
                "source": getattr(minimizer.get("source"), "name", None),
                "target": getattr(minimizer.get("target"), "name", None),
            }

            for role, endpoint in (
                ("source", minimizer.get("source")),
                ("target", minimizer.get("target")),
            ):
                name = getattr(endpoint, "name", None)
                if name is None:
                    continue
                # A target is usually an Input that no output depends on, so it
                # is absent from the traversal and has to be drawn here.
                if name not in visible_node_ids:
                    kind = type(endpoint).__name__
                    vis_nodes.append(
                        {
                            "id": name,
                            "label": name,
                            "shape": "dot",
                            "size": 14,
                            "color": {
                                "background": _node_color(kind),
                                "border": "#8e44ad",
                                "highlight": {
                                    "background": "#f1c40f",
                                    "border": "#8e44ad",
                                },
                            },
                            "font": {"color": "#111111"},
                            "title": f"{name}\n{kind} (training target)",
                            "level": levels.get(name, 0),
                        }
                    )
                    node_details[name] = _safe_attrs(endpoint)
                    visible_node_ids.add(name)

                vis_edges.append(
                    {
                        "from": name,
                        "to": loss_id,
                        "arrows": "to",
                        "label": role,
                        "dashes": role == "target",
                        "width": 2,
                        "font": {"align": "middle", "size": 10, "color": "#6c3483"},
                        "color": {"color": "#8e44ad", "highlight": "#6c3483"},
                    }
                )

        target = "_blank" if open_subgraph_in_new_tab else "_self"
        # A spring layout turns a large body into a hairball, so those pages
        # open on the deterministic hierarchical one instead.
        hierarchical = len(vis_nodes) > 60

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

        logo_html = ""
        if logo_rel is not None:
            logo_html = (
                f'<img src="./{logo_rel}" alt="nnodely logo" width="120" height="40"/>'
            )

        boundary_html = ""
        if boundary is not None:
            boundary_html = """
        <section id="boundary">
            <h2>Block boundary</h2>
            <div id="boundaryContent"></div>
        </section>
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
    .roll-arrow {{
        color: #e74c3c;
        display: inline-block;
        font-size: 17px;
        font-weight: 700;
        line-height: 10px;
        margin-right: 5px;
        vertical-align: -1px;
    }}
    .toolbar {{
        display: flex;
        gap: 8px;
        align-items: center;
        padding: 8px 16px;
        border-bottom: 1px solid #eee;
        background: #fbfbfb;
        font-size: 12px;
        flex-wrap: wrap;
    }}
    #search {{
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 5px 9px;
        font-size: 12px;
        width: 220px;
    }}
    #searchInfo {{
        color: #666;
    }}
    .main {{
        display: flex;
        width: 100%;
        height: 86vh;
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
    #boundary {{
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 12px;
        margin-bottom: 14px;
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
    .ports {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        margin: 4px 0 10px 0;
    }}
    .ports td {{
        border: 1px solid #e5e7eb;
        padding: 4px 6px;
        word-break: break-word;
    }}
    .ports .role {{
        color: #666;
        white-space: nowrap;
        width: 1%;
    }}
    .ports-title {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .04em;
        color: #666;
        margin-top: 10px;
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
        {logo_html}
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
        <span><span class="swatch" style="background:#8e44ad"></span>Loss</span>
        <span><span class="roll-arrow">→</span>Roll</span>
    </div>
    <div class="header-right">
        <div class="bottom-bar">
            {flat_button_html}
        </div>
    </div>
</header>

<div class="toolbar">
    <input id="search" type="search" placeholder="Find node by name…" autocomplete="off"/>
    <span id="searchInfo"></span>
    <button id="layoutBtn" class="btn"></button>
    <button id="fitBtn" class="btn">Fit</button>
    <span id="graphInfo"></span>
</div>

<div class="main">
    <div id="network"></div>
    <aside id="inspector">
        {boundary_html}
        <h2>Node details</h2>
        <div id="inspectorContent" class="empty">
            Click a node to inspect its attributes.
        </div>
    </aside>
</div>

<script>
    const nodes = new vis.DataSet({json.dumps(vis_nodes, ensure_ascii=False)});
    const edges = new vis.DataSet({json.dumps(vis_edges, ensure_ascii=False)});
    const nodeDetails = {json.dumps(node_details, ensure_ascii=False, default=str)};
    const boundary = {json.dumps(boundary, ensure_ascii=False, default=str)};

    const container = document.getElementById('network');
    const inspectorContent = document.getElementById('inspectorContent');
    const data = {{ nodes, edges }};

    let hierarchical = {str(hierarchical).lower()};
    const springPhysics = {str(physics).lower()};

    function layoutOptions() {{
        return {{
            layout: {{
                improvedLayout: !hierarchical,
                hierarchical: hierarchical
                    ? {{
                        enabled: true,
                        direction: 'LR',
                        sortMethod: 'directed',
                        nodeSpacing: 110,
                        levelSeparation: 190,
                        treeSpacing: 140,
                        shakeTowards: 'roots'
                    }}
                    : {{ enabled: false }}
            }},
            physics: hierarchical ? false : springPhysics,
            edges: {{
                smooth: hierarchical ? {{ enabled: true, type: 'cubicBezier',
                    forceDirection: 'horizontal', roundness: 0.4 }} : springPhysics,
                shadow: hierarchical ? false : springPhysics
            }}
        }};
    }}

    const options = Object.assign({{
        autoResize: true,
        interaction: {{
            hover: true,
            tooltipDelay: 80,
            multiselect: true,
            navigationButtons: true,
            keyboard: true
        }},
        nodes: {{
            borderWidth: 1,
            shadow: true
        }}
    }}, layoutOptions());
    options.edges.font = {{ size: 10, align: "middle" }};

    const network = new vis.Network(container, data, options);

    document.getElementById('graphInfo').textContent =
        nodes.length + ' nodes · ' + edges.length + ' edges';

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

    function focusNode(nodeId) {{
        if (!nodes.get(nodeId)) return;
        network.selectNodes([nodeId]);
        network.focus(nodeId, {{ scale: 1.1, animation: true }});
        renderInspector(nodeId);
    }}

    function portRows(rows) {{
        return rows.map(row => `
            <tr>
                <td><a href="#" data-node="${{escapeHtml(row.node)}}">${{escapeHtml(row.node)}}</a></td>
                <td>${{escapeHtml(row.text)}}</td>
                <td class="role">${{escapeHtml(row.role || "")}}</td>
            </tr>`).join("");
    }}

    function portsHtml(ports) {{
        let html = "";
        const detail = Object.entries(ports.detail || {{}})
            .map(([key, value]) => `${{escapeHtml(key)}}: ${{escapeHtml(String(value))}}`)
            .join(" · ");
        html += `<div class="ports-title">${{escapeHtml(ports.block)}}${{detail ? " — " + detail : ""}}</div>`;

        if (ports.inputs.length) {{
            html += '<div class="ports-title">Inputs</div><table class="ports">' +
                portRows(ports.inputs.map(p => (
                    {{ node: p.body, text: "\\u2190 " + p.outer, role: p.role }}
                ))) + "</table>";
        }}
        if (ports.feedback.length) {{
            html += '<div class="ports-title">Feedback</div><table class="ports">' +
                portRows(ports.feedback.map(p => (
                    {{ node: p.from, text: "\\u2192 " + p.to, role: "closed" }}
                ))) + "</table>";
        }}
        if (ports.outputs.length) {{
            html += '<div class="ports-title">Outputs</div><table class="ports">' +
                portRows(ports.outputs.map(p => (
                    {{ node: p.body, text: "\\u2192 " + p.outer, role: "" }}
                ))) + "</table>";
        }}
        return html;
    }}

    function bindPortLinks(root) {{
        root.querySelectorAll('a[data-node]').forEach(link => {{
            link.addEventListener('click', event => {{
                event.preventDefault();
                focusNode(link.getAttribute('data-node'));
            }});
        }});
    }}

    if (boundary) {{
        const el = document.getElementById('boundaryContent');
        el.innerHTML =
            `<div class="empty">Body of <b>${{escapeHtml(boundary.node)}}</b> in ` +
            `<a href="${{escapeHtml(boundary.parent_rel)}}">${{escapeHtml(boundary.parent)}}</a>.</div>` +
            portsHtml(boundary.ports);
        bindPortLinks(el);
    }}

    function renderInspector(nodeId) {{
        const details = nodeDetails[nodeId];
        const n = nodes.get(nodeId);

        if (!details) {{
            inspectorContent.innerHTML = '<div class="empty">No details available for this node.</div>';
            return;
        }}

        const rows = Object.entries(details)
            .filter(([key]) => key !== "block")
            .map(([key, value]) => {{
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
            ${{details.block ? portsHtml(details.block) : ""}}
            ${{extra}}
        `;
        inspectorContent.querySelectorAll('a[data-node]').forEach(link => {{
            link.addEventListener('click', event => {{
                event.preventDefault();
                focusNode(link.getAttribute('data-node'));
            }});
        }});
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

    const searchBox = document.getElementById("search");
    const searchInfo = document.getElementById("searchInfo");
    searchBox.addEventListener("input", () => {{
        const query = searchBox.value.trim().toLowerCase();
        if (!query) {{
            searchInfo.textContent = "";
            network.unselectAll();
            return;
        }}
        const matches = nodes.getIds().filter(
            id => String(id).toLowerCase().includes(query));
        searchInfo.textContent = matches.length + (matches.length === 1 ? " match" : " matches");
        if (matches.length) {{
            network.selectNodes(matches);
            network.fit({{ nodes: matches, animation: true }});
            if (matches.length === 1) renderInspector(matches[0]);
        }}
    }});

    const layoutBtn = document.getElementById("layoutBtn");
    function refreshLayoutBtn() {{
        layoutBtn.textContent = hierarchical ? "Layout: hierarchical" : "Layout: force";
    }}
    refreshLayoutBtn();
    layoutBtn.addEventListener("click", () => {{
        hierarchical = !hierarchical;
        refreshLayoutBtn();
        const next = layoutOptions();
        next.edges.font = {{ size: 10, align: "middle" }};
        network.setOptions(next);
        network.fit({{ animation: true }});
    }});

    document.getElementById("fitBtn").addEventListener("click", () => {{
        network.fit({{ animation: true }});
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

    # The root goes first so that a body page shared by both views points its
    # breadcrumb at the standard graph rather than the flattened one.
    root_path = _export_one(
        model,
        root_name,
        model.name,
        flattened=False,
        flattened_rel=flat_name,
    )
    _export_one(
        model,
        flat_name,
        model.name,
        flattened=True,
        standard_rel=root_name,
    )
    return root_path
