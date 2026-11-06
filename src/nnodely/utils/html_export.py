"""An interactive HTML page of a Modely graph, drawn with vis-network.

Every block that wraps a model gets a page of its own. The page itself is
the template ``templates/graph.html``, filled in with the graph's data.
"""

import os
import re
import json
import shutil
from html import escape
from pathlib import Path
from nnodely.core.modely import Modely, ModelCall

from nnodely.layers.input import Input
from nnodely.layers.loop import Loop

## nnodely pallette
# :root {
#   --color-primary: #6C75DF;   /* modely blue-purple */
#   --color-secondary: #89C1F9; /* light sky blue */
#   --color-accent: #81ACF4;    /* soft gradient blue */
#   --color-text: #000000;      /* logo black */
# }


_TEMPLATE = Path(__file__).with_name("templates") / "graph.html"
_PLACEHOLDER = re.compile(r"\{\{ (\w+) \}\}")


def _script_json(value, **kwargs) -> str:
    """``value`` as JSON that can sit inside a ``<script>``: a name holding
    ``</script>`` would otherwise end the script there."""
    text = json.dumps(value, ensure_ascii=False, **kwargs)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _render_page(**values: str) -> str:
    """The page of one graph: the template with every ``{{ name }}`` filled in.

    One pass over the template, so a value is never searched for placeholders
    of its own - a node may well be named after one.
    """
    template = _TEMPLATE.read_text(encoding="utf-8")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def _copy_logo(out_path: Path) -> str | None:
    """Place the header logo next to the pages, returning its relative URL."""
    source = Path(__file__).with_name("templates") / "logo.png"
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
        return node._model, {
            "block": "ModelCall",
            "detail": {},
            "inputs": [
                {"body": body.name, "outer": outer.name, "role": "input"}
                for body, outer in node._inputs_map.items()
            ],
            "outputs": [
                {"body": body.name, "outer": outer.name}
                for outer, body in node._outputs_map.items()
            ],
            "feedback": [],
        }

    if isinstance(node, Loop):
        return node._f, {
            "block": "Loop",
            "detail": {
                "steps": node._horizon,
                "collect": node._collect,
                # A rollout as long as its data is built with a placeholder.
                "steps_from_data": node._length is None
                and node._horizon_index is not None,
            },
            "inputs": [
                {"body": body.name, "outer": outer.name, "role": "initial"}
                for body, outer in zip(node._feedback_inputs, node._initial_values)
            ]
            + [
                {"body": body.name, "outer": outer.name, "role": "input"}
                for body, outer in zip(node._static_inputs, node._static_sources)
            ],
            "outputs": [
                {"body": body.name, "outer": outer.name}
                for body, outer in zip(node._returns, node._loop_outputs)
            ],
            "feedback": [
                {"from": output.name, "to": input_node.name}
                for input_node, output in zip(
                    node._feedback_inputs, node._feedback_outputs
                )
            ],
        }

    return None


def _safe_attrs(node) -> dict:
    attrs = {
        "class": type(node).__name__,
        "name": getattr(node, "name", None),
    }

    for attr in ("time", "dim", "shape"):
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

    # The axes one by one, so the page can say what each of them is.
    try:
        shape = node.shape
        attrs["axes"] = {
            "seq": ["dyn" if size is None else size for size in shape.seq],
            "time": shape.time,
            "dim": list(shape.dim),
        }
    except Exception:
        pass
    about = _about(node)
    if about:
        attrs["about"] = about

    return attrs


def _about(node) -> str | None:
    """The first sentence of what the node's class says it is."""
    doc = type(node).__doc__
    if not doc:
        return None
    paragraph = " ".join(doc.strip().split("\n\n", 1)[0].split())
    sentence = re.split(r"(?<=[.!?])\s", paragraph, maxsplit=1)[0]
    sentence = sentence.rstrip(":").rstrip()
    if not sentence:
        return None
    if len(sentence) > 240:
        sentence = sentence[:237].rstrip() + "..."
    return sentence if sentence.endswith((".", "!", "?", "...")) else sentence + "."


def _collect_graph(model_obj, *, inline=False, prefix="", alias=None):
    """Nodes, edges and feedback arrows of a graph, blocks optionally spliced.

    With ``inline`` every block is replaced by the body it wraps, so the
    flattened view answers what the model actually computes. ``Modely.flatten``
    cannot do this itself: ``build`` runs the very same pass, and a Loop has
    to survive it as one node to become a single Keras layer. A rollout is
    spliced once and its recurrence drawn as a feedback arrow, so
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
    present = {getattr(node, "name", None) for node in model_obj._order}
    if inline:
        blocks = {
            node.name: _block_ports(node)
            for node in model_obj._order
            if _block_ports(node) is not None
        }

    for node in model_obj._order:
        name = getattr(node, "name", str(node))
        ports = blocks.get(name)

        if ports is not None:
            submodel, port_map = ports
            body_prefix = f"{prefix}{name}/"

            # A block that reads the body's own inputs names the very same
            # node on both sides of a port: alias those instead of duplicating
            # them.
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
                # The innermost block a node was spliced from: a nested body
                # has set it already.
                attrs.setdefault("inlined_in", prefix + name)
                attrs.setdefault(
                    "inlined_block",
                    {
                        "id": prefix + name,
                        "block": port_map["block"],
                        "model": submodel.name,
                        **port_map["detail"],
                    },
                )
                nodes.append((body_node, nid, kind, attrs))
                seen_nodes.add(nid)
            for body_pred, src, dst, role in body_edges:
                add_edge(body_pred, src, dst, role)
            feedbacks.extend(body_feedbacks)

            # Bind the enclosing graph to the body it now contains.
            for port in port_map["inputs"]:
                outer, body = outer_id(port["outer"]), body_id(port["body"])
                # A block can read another block spliced before it - one
                # Loop rolling out another's result - whose node is gone: bind to
                # the body output that node stood for, as its consumers are.
                if port["outer"] in block_results:
                    outer = block_results[port["outer"]]["primary"]
                if outer != body:
                    add_edge(None, outer, body, port["role"])
            block_results[name] = {
                "selectors": {port["outer"] for port in port_map["outputs"]},
                "primary": body_id(port_map["outputs"][0]["body"]),
            }
            for port in port_map["outputs"]:
                # A selector absent from this graph means the block is read
                # as a value: a single-output Loop.
                if port["outer"] == name or port["outer"] not in present:
                    continue
                outer, body = outer_id(port["outer"]), body_id(port["body"])
                if outer != body:
                    add_edge(None, body, outer, "output")

            for pair in port_map["feedback"]:
                feedbacks.append(
                    (
                        body_id(pair["from"]),
                        body_id(pair["to"]),
                        _feedback_label(port_map["detail"], "x"),
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

    for node in model_obj._order:
        if getattr(node, "name", None) in blocks:
            continue
        for pred in getattr(node, "_preds", []) or []:
            pred_name = getattr(pred, "name", None)
            if pred_name in blocks:
                # A consumer reads the block either through one of its
                # selector nodes, already bound above, or straight off the
                # block node, which now resolves to the body output.
                results = block_results[pred_name] if pred_name else {}
                if getattr(node, "name", None) not in results["selectors"]:
                    add_edge(pred, results["primary"], ident(node))
                continue
            add_edge(pred, ident(pred), ident(node))

    return nodes, edges, feedbacks


def _axes_text(shape) -> str:
    """A shape as the page writes it, ``(*seq, time, *dim)``, a dynamic axis
    as ``dyn``."""
    axes = getattr(shape, "tuple", None)
    if axes is None:
        return ""
    return "(" + ", ".join("dyn" if size is None else str(size) for size in axes) + ")"


def _steps_text(node) -> str:
    """How many steps a Loop rolls out, as a page says it."""
    if node._length is None and node._horizon_index is not None:
        return "steps from data"
    steps = node._horizon
    return f"{steps} step" + ("" if steps == 1 else "s")


def _node_role(node, region: str | None, states: set, own_outputs: set, nid: str):
    """What a node is, for the reader: the role the page colours it by."""
    from nnodely.core.modely import IntermediateOutput
    from nnodely.layers.loop import LoopOutput
    from nnodely.layers.output import Output
    from nnodely.layers.parameter import Constant, Parameter

    if isinstance(node, (ModelCall, Loop)):
        return "submodel"
    if isinstance(node, Input):
        # A loop's memory: fed back on this page, or kept by a layer.
        if nid in states or getattr(node, "_hidden", False):
            return "state"
        return "inner_input" if region else "input"
    if type(node) is IntermediateOutput or isinstance(node, LoopOutput):
        return "inner_output"
    if isinstance(node, Output):
        return "output" if not region and node.name in own_outputs else "inner_output"
    if isinstance(node, Parameter):
        return "parameter"
    if isinstance(node, Constant):
        return "constant"
    return "relation"


def _node_subtitle(node, kind: str, role: str) -> str:
    """The second line of a node's card: what it is, in a few words."""
    if isinstance(node, Loop):
        return f"Loop · {_steps_text(node)}"
    if isinstance(node, ModelCall):
        return f"call of {node._model.name}"
    return {
        "input": "input",
        "state": "state",
        "inner_input": "body input",
        "output": "output",
        "inner_output": "body output",
        "parameter": "parameter",
        "constant": "constant",
    }.get(role, kind)


def _feedback_label(detail: dict, times: str = "") -> str:
    """The label of a recurrence arrow: how many steps it closes."""
    if detail.get("steps_from_data"):
        return "feedback (steps from data)"
    steps = detail.get("steps")
    return f"feedback ({times}{steps} steps)" if steps else "feedback"


def _feedback_record(src: str, dst: str, label: str, info: dict) -> dict:
    """A recurrence arrow: the value of a step fed back to a state."""
    return {"from": src, "to": dst, "kind": "feedback", "label": label, "info": info}


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

    Every block that wraps a model (``ModelCall``, ``Loop``) is
    exported as its own page, linked from the block node and annotated with the
    ports that bind the body to the graph above it. Next to the root page a
    flattened one splices every block in, each body inside a shaded region.

    The page says what every node is - its role, coloured the same in the
    graph, the legend and the inspector - and how the graph is read. The
    pages open on the left-to-right hierarchical layout; ``physics`` lets the
    force layout settle the nodes by simulation.
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

        # Flattened, every block is spliced in where it is called - as plot()
        # draws it - so each body keeps the region of the call it came from.
        render_model = model_obj
        graph_nodes, graph_edges, inlined_feedbacks = _collect_graph(
            render_model, inline=flattened
        )
        levels = _levels(graph_nodes, graph_edges)

        # Every block left as one node gets a page of its own; flattened, they
        # are all spliced in, so there is none.
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

        visible = {nid for _, nid, _, _ in graph_nodes}

        # The recurrences: the ones the spliced bodies close, and the one the
        # enclosing block closes over this body.
        feedbacks = [
            _feedback_record(src, dst, label, info)
            for src, dst, label, info in inlined_feedbacks
            if src in visible and dst in visible
        ]
        if boundary is not None:
            detail = boundary["ports"]["detail"]
            for pair in boundary["ports"]["feedback"]:
                if pair["from"] not in visible or pair["to"] not in visible:
                    continue
                feedbacks.append(
                    _feedback_record(
                        pair["from"],
                        pair["to"],
                        _feedback_label(detail),
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
        states = {record["to"] for record in feedbacks}
        own_outputs = {output.name for output in render_model._outputs}

        vis_nodes: list[dict] = []
        node_details: dict[str, dict] = {}
        regions: dict[str, dict] = {}
        for node, nid, kind, attrs in graph_nodes:
            region = attrs.get("inlined_in")
            role = _node_role(node, region, states, own_outputs, nid)
            short = nid[len(region) + 1 :] if region and nid.startswith(region + "/") else nid
            record = {
                "id": nid,
                "label": short,
                "role": role,
                "kind": kind,
                "subtitle": _node_subtitle(node, kind, role),
                "level": levels.get(nid, 0),
            }
            if region:
                record["region"] = region
                regions.setdefault(region, attrs.get("inlined_block", {"id": region}))
            if nid in url_map:
                record["url"] = url_map[nid]
            if nid in entry_ports:
                port = entry_ports[nid]
                record["port"] = {"side": "in", "outer": port["outer"], "role": port["role"]}
            elif nid in exit_ports:
                record["port"] = {"side": "out", "outer": exit_ports[nid]["outer"]}
            attrs["role"] = role
            node_details[nid] = attrs
            vis_nodes.append(record)

        # A region nested in another one is drawn inside it, even when the
        # outer body holds no node of its own.
        for region in list(regions):
            parts = region.split("/")
            for depth in range(1, len(parts)):
                regions.setdefault("/".join(parts[:depth]), {"id": "/".join(parts[:depth])})

        vis_edges: list[dict] = []
        for pred, src, dst, role in graph_edges:
            # A multi-output Loop has no one shape: an AttributeError.
            axes = _axes_text(getattr(pred, "shape", None))
            record = {
                "from": src,
                "to": dst,
                "kind": "data" if role is None else "binding",
                # The arrow carries the axes; a binding names the port it is.
                "label": role if role is not None else axes,
                "shape": axes,
            }
            vis_edges.append(record)
        vis_edges.extend(feedbacks)

        # Training objectives, so the loss wiring is visible here too.
        for index, minimizer in enumerate(getattr(model_obj, "_minimizers", [])):
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
                    "label": min_name,
                    "role": "loss",
                    "kind": "Minimizer",
                    "subtitle": f"loss · {loss_label}",
                    "level": max(
                        levels.get(getattr(minimizer.get("source"), "name", ""), 0),
                        levels.get(getattr(minimizer.get("target"), "name", ""), 0),
                    )
                    + 1,
                }
            )
            node_details[loss_id] = {
                "class": "Minimizer",
                "name": min_name,
                "role": "loss",
                "loss": loss_label,
                "source": getattr(minimizer.get("source"), "name", None),
                "target": getattr(minimizer.get("target"), "name", None),
            }

            for side, endpoint in (
                ("source", minimizer.get("source")),
                ("target", minimizer.get("target")),
            ):
                name = getattr(endpoint, "name", None)
                if name is None:
                    continue
                # A target is usually an Input that no output depends on, so it
                # is absent from the traversal and has to be drawn here.
                if name not in visible:
                    vis_nodes.append(
                        {
                            "id": name,
                            "label": name,
                            "role": "target",
                            "kind": type(endpoint).__name__,
                            "subtitle": "training target",
                            "level": levels.get(name, 0),
                        }
                    )
                    node_details[name] = {**_safe_attrs(endpoint), "role": "target"}
                    visible.add(name)
                vis_edges.append(
                    {
                        "from": name,
                        "to": loss_id,
                        "kind": "loss",
                        "label": side,
                        "shape": _axes_text(getattr(endpoint, "shape", None)),
                    }
                )

        logo_html = ""
        if logo_rel is not None:
            logo_html = (
                f'<img src="./{logo_rel}" alt="nnodely" width="96" height="32"/>'
            )

        page = {
            "title": page_title,
            "flattened": flattened,
            "parent_rel": parent_rel,
            "standard_rel": standard_rel,
            "flattened_rel": flattened_rel,
            "link_target": "_blank" if open_subgraph_in_new_tab else "_self",
            "physics": bool(physics),
        }
        html = _render_page(
            title=escape(page_title),
            title_suffix=" [flattened]" if flattened else "",
            logo=logo_html,
            nodes=_script_json(vis_nodes),
            edges=_script_json(vis_edges),
            node_details=_script_json(node_details, default=str),
            boundary=_script_json(boundary, default=str),
            regions=_script_json(list(regions.values()), default=str),
            page=_script_json(page),
        )
        file_path.write_text(html, encoding="utf-8")
        return str(file_path)

    root_base = _slug(filename if filename else model.name)
    if root_base.endswith(".html"):
        root_base = root_base[: -len(".html")]
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
