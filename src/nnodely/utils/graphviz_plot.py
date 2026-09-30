"""A static drawing of a Modely graph, rendered by graphviz."""

import warnings
from pathlib import Path

from nnodely.core.modely import Modely
from nnodely.core.validation import loss_name


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

    # The suffix names the format; a dot in a folder's name is no suffix.
    target = Path(to_file)
    fmt = target.suffix[1:] or "png"
    outpath = target.with_suffix("") if target.suffix else target

    dot = graphviz.Digraph(name=model.name, format=fmt)
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
            loss_label = loss_name(m.get("loss"))
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
                label=f"{min_name}\n{loss_label}",
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
        dot.render(filename=str(outpath), cleanup=True)
    except graphviz.ExecutableNotFound:
        source_file = outpath.with_name(f"{outpath.name}.gv")
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(dot.source, encoding="utf-8")
        warnings.warn(
            f"Graphviz's dot program was not found, so {str(target)!r} was not "
            f"drawn: the graph's DOT source is written to {str(source_file)!r}. "
            "Install Graphviz (https://graphviz.org/download/) to render it.",
            UserWarning,
            stacklevel=3,
        )
    return dot
