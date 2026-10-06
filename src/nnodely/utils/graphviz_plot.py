"""Static drawings of a Modely graph, rendered by Graphviz.

Every image carries the logo, the model's name and a small legend. A node is
a card holding its name, coloured by what it is; an arrow carries the shape
that flows along it. The recurrence of a Loop, a Roll or a model's rollback
is a coloured arrow back to the input it feeds.

Without ``flatten`` every block that wraps a model (a sub-model call, a
``Loop``, a ``Roll``) is one sub-model node, and its body is drawn in an image
of its own, recursively, named ``<filename>_<body name>``. With ``flatten``
everything is spliced into one image, each sub-model call's body inside a box.

The graph is walked by the traversal the HTML export uses, so both drawings
show the same nodes and the same bindings.
"""

from __future__ import annotations

import os
import re
import warnings
from html import escape
from pathlib import Path

from nnodely.core.modely import ModelCall, Modely
from nnodely.layers.input import Input
from nnodely.layers.loop import Loop
from nnodely.layers.output import Output
from nnodely.layers.parameter import Constant, Parameter
from nnodely.layers.roll import Roll
from nnodely.layers.time_ops import SampleWindow
from nnodely.utils.html_export import _block_ports, _collect_graph

_LOGO = Path(__file__).with_name("templates") / "logo.png"

#: Formats a suffix may name; anything else Graphviz renders is passed as
#: ``format=``.
_SUFFIX_FORMATS = {"png", "svg", "pdf", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"}

_INK = "#1F2335"
_MUTED = "#6B7280"

# Inputs green and outputs red, the rest in the nnodely palette.
_ROLES = {
    "input": {
        "fill": "#2ECC71",
        "border": "#239B56",
        "text": "#FFFFFF",
        "label": "Input",
    },
    "output": {
        "fill": "#E74C3C",
        "border": "#B03A2E",
        "text": "#FFFFFF",
        "label": "Output",
    },
    "parameter": {
        "fill": "#FFE9C7",
        "border": "#E9A23B",
        "text": _INK,
        "label": "Parameter",
    },
    "constant": {
        "fill": "#E0FFFF",
        "border": "#5FB8C2",
        "text": _INK,
        "label": "Constant",
    },
    "submodel": {
        "fill": "#6C75DF",
        "border": "#4B53B8",
        "text": "#FFFFFF",
        "label": "SubModel",
    },
    "loss": {
        "fill": "#9B6BD3",
        "border": "#7A4DB3",
        "text": "#FFFFFF",
        "label": "Loss",
    },
    "relation": {
        "fill": "#FFFFFF",
        "border": "#81ACF4",
        "text": _INK,
        "label": "Relation",
    },
}
_LEGEND_ROLES = ("input", "output", "parameter", "constant", "submodel", "loss")

_ARROWS = {
    "data": {"color": "#8A94A6"},
    "loop": {"color": "#F28C28", "penwidth": "2.4", "label": "loop"},
    "roll": {"color": "#17A398", "penwidth": "2.4", "label": "roll"},
    "loss": {"color": "#9B6BD3", "penwidth": "1.4"},
}


def _role(node) -> str:
    if isinstance(node, (ModelCall, Loop, Roll)):
        return "submodel"
    if isinstance(node, Input):
        return "input"
    if isinstance(node, Output):
        return "output"
    if isinstance(node, Parameter):
        return "parameter"
    if isinstance(node, Constant):
        return "constant"
    return "relation"


def _shape(node) -> str:
    """The shape as ``(*dim, time, *seq)``, a dynamic axis written ``dyn``."""
    shape = getattr(node, "shape", None)
    if shape is None:
        return ""
    return (
        "("
        + ", ".join("dyn" if axis is None else str(axis) for axis in shape.tuple)
        + ")"
    )


def _recurrence(block) -> tuple[str, str]:
    """The arrow kind and label of the recurrence a block closes."""
    if isinstance(block, Roll):
        return "roll", f"roll (steps={block.steps})"
    return "loop", "loop"


def _font(text: str, size: int, color: str, bold: bool = False) -> str:
    body = escape(text)
    if bold:
        body = f"<B>{body}</B>"
    return f'<FONT POINT-SIZE="{size}" COLOR="{color}">{body}</FONT>'


def _card(name: str, role: str) -> str:
    """A node: its name on a rounded card of its role's colour."""
    style = _ROLES[role]
    return (
        f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="6" '
        f'STYLE="ROUNDED" BGCOLOR="{style["fill"]}" COLOR="{style["border"]}">'
        f"<TR><TD>{_font(name, 11, style['text'], bold=True)}</TD></TR></TABLE>>"
    )


def _header(title: str) -> str:
    """The banner of every image: logo, title and the legend."""
    swatches = "".join(
        f'<TD BGCOLOR="{_ROLES[role]["fill"]}" COLOR="{_ROLES[role]["border"]}" '
        f'BORDER="1" WIDTH="16" HEIGHT="10" FIXEDSIZE="TRUE"></TD>'
        f'<TD ALIGN="LEFT">{_font(_ROLES[role]["label"], 9, _INK)}</TD>'
        for role in _LEGEND_ROLES
    )
    arrows = "".join(
        f"<TD>{_font('→', 12, _ARROWS[kind]['color'], bold=True)}</TD>"
        f'<TD ALIGN="LEFT">{_font(_ARROWS[kind]["label"], 9, _INK)}</TD>'
        for kind in ("loop", "roll")
    )
    logo = (
        f'<TD FIXEDSIZE="TRUE" WIDTH="120" HEIGHT="35">'
        f'<IMG SRC="{escape(str(_LOGO))}" SCALE="TRUE"/></TD>'
        if _LOGO.is_file()
        else ""
    )
    return (
        '<<TABLE BORDER="0" CELLSPACING="2" CELLPADDING="2">'
        f'<TR>{logo}<TD ALIGN="LEFT">{_font(title, 18, _INK, bold=True)}</TD></TR>'
        '<TR><TD COLSPAN="2" ALIGN="LEFT"><TABLE BORDER="0" CELLSPACING="3" CELLPADDING="1">'
        f"<TR>{swatches}{arrows}</TR></TABLE></TD></TR>"
        "</TABLE>>"
    )


class _Drawing:
    """One Graphviz graph: the header, the cards and the arrows."""

    def __init__(self, graphviz, title: str, fmt: str):
        self.dot = graphviz.Digraph(name=title, format=fmt)
        self.dot.attr(
            "graph",
            rankdir="LR",
            bgcolor="white",
            fontname="Helvetica",
            label=_header(title),
            labelloc="t",
            labeljust="l",
            nodesep="0.35",
            ranksep="0.6",
            pad="0.3",
            splines="spline",
        )
        if fmt in ("png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"):
            self.dot.attr("graph", dpi="144")
        self.dot.attr("node", shape="plain", fontname="Helvetica")
        self.dot.attr(
            "edge",
            fontname="Helvetica",
            fontsize="8",
            fontcolor=_MUTED,
            arrowsize="0.7",
            color=_ARROWS["data"]["color"],
        )
        self.nodes: set[str] = set()

    def node(self, nid: str, name: str, role: str, graph=None) -> None:
        if nid in self.nodes:
            return
        (graph or self.dot).node(nid, label=_card(name, role))
        self.nodes.add(nid)

    def data(self, src: str, dst: str, label: str = "") -> None:
        self.dot.edge(src, dst, label=label)

    def recurrence(self, src: str, dst: str, kind: str, label: str) -> None:
        # A recurrence closes the graph on itself: it must not pull the layout
        # backwards, and its label is as loud as the arrow.
        color = _ARROWS[kind]["color"]
        self.dot.edge(
            src,
            dst,
            label=f"<<B>{escape(label)}</B>>",
            color=color,
            fontcolor=color,
            fontsize="10",
            penwidth=_ARROWS[kind]["penwidth"],
            constraint="false",
        )


def _minimizers(drawing: _Drawing, model: Modely) -> None:
    style = _ARROWS["loss"]
    for index, minimizer in enumerate(model.minimizers):
        name = minimizer.get("name", f"loss_{index}")
        loss_id = f"__loss_{name}"
        drawing.node(loss_id, name, "loss")
        for endpoint, dashed in (
            (minimizer.get("source"), False),
            (minimizer.get("target"), True),
        ):
            if getattr(endpoint, "name", None) is None:
                continue
            shown = endpoint
            # A target is usually a window of an Input no output reads, so the
            # traversal never reached it: the Input it reads stands for it.
            if (
                endpoint.name not in drawing.nodes
                and isinstance(endpoint, SampleWindow)
                and endpoint.preds
            ):
                shown = endpoint.preds[0]
            if shown.name not in drawing.nodes:
                drawing.node(shown.name, shown.name, _role(shown))
            drawing.dot.edge(
                shown.name,
                loss_id,
                color=style["color"],
                penwidth=style["penwidth"],
                **({"style": "dashed"} if dashed else {}),
            )


def _rollback(drawing: _Drawing, model: Modely) -> None:
    steps = getattr(model, "_roll_steps", None)
    for input_node, stream in getattr(model, "_roll_callbacks", {}).items():
        src, dst = stream.name, input_node.name
        if src in drawing.nodes and dst in drawing.nodes:
            drawing.recurrence(src, dst, "roll", f"roll (steps={steps})")


def _draw_model(
    graphviz, model: Modely, fmt: str, *, include_minimizers: bool, block=None
) -> _Drawing:
    """A model as declared: each block one sub-model node."""
    drawing = _Drawing(graphviz, model.name, fmt)
    nodes, edges, _ = _collect_graph(model, inline=False)
    for node, nid, _, _ in nodes:
        drawing.node(nid, nid, _role(node))
    for pred, src, dst, _ in edges:
        drawing.data(src, dst, _shape(pred))

    # The recurrence of the Loop or Roll this model is the body of.
    if block is not None:
        node, port_map = block
        kind, label = _recurrence(node)
        for pair in port_map["feedback"]:
            if pair["from"] in drawing.nodes and pair["to"] in drawing.nodes:
                drawing.recurrence(pair["from"], pair["to"], kind, label)

    _rollback(drawing, model)
    if include_minimizers:
        _minimizers(drawing, model)
    return drawing


def _block_index(model: Modely, prefix: str = "") -> dict[str, tuple]:
    """Every block of the tree, by the path its body's nodes are prefixed with."""
    index: dict[str, tuple] = {}
    for node in model.order:
        ports = _block_ports(node)
        if ports is None:
            continue
        path = prefix + node.name
        index[path] = (node, ports[0])
        index.update(_block_index(ports[0], prefix=path + "/"))
    return index


def _draw_flattened(
    graphviz, model: Modely, fmt: str, *, include_minimizers: bool
) -> _Drawing:
    """The whole tree in one image, each sub-model call's body boxed."""
    drawing = _Drawing(graphviz, model.name, fmt)
    nodes, edges, feedbacks = _collect_graph(model, inline=True)
    blocks = _block_index(model)
    # A Loop or a Roll is told by its arrow, not by a box: only the bodies of
    # sub-model calls are boxed.
    boxes = {
        path for path, (block, _) in blocks.items() if isinstance(block, ModelCall)
    }

    def box_of(path: str) -> str:
        while path and path not in boxes:
            path = path.rsplit("/", 1)[0] if "/" in path else ""
        return path

    def owner(nid: str) -> str:
        return box_of(nid.rsplit("/", 1)[0] if "/" in nid else "")

    children: dict[str, list[str]] = {}
    for path in sorted(boxes):
        children.setdefault(owner(path), []).append(path)
    members: dict[str, list] = {}
    shapes: dict[str, str] = {}
    for node, nid, _, _ in nodes:
        members.setdefault(owner(nid), []).append((node, nid))
        shapes[nid] = _shape(node)

    def fill(graph, path: str) -> None:
        for node, nid in members.get(path, []):
            drawing.node(nid, nid.rsplit("/", 1)[-1], _role(node), graph=graph)
        for child in children.get(path, []):
            style = _ROLES["submodel"]
            with graph.subgraph(name=f"cluster_{_stem(child)}") as box:
                box.attr(
                    label=f"<{_font(blocks[child][1].name, 10, style['border'], bold=True)}>",
                    labeljust="l",
                    style="rounded,filled",
                    fillcolor="#E7E8FB",
                    color=style["border"],
                    penwidth="1.4",
                )
                fill(box, child)

    fill(drawing.dot, "")

    for pred, src, dst, _ in edges:
        # A binding edge has no predecessor of its own: it carries what the
        # node it leaves holds.
        drawing.data(
            src, dst, _shape(pred) if pred is not None else shapes.get(src, "")
        )

    for src, dst, _, attrs in feedbacks:
        if src in drawing.nodes and dst in drawing.nodes:
            block = blocks.get(attrs.get("block_node", ""), (None,))[0]
            kind, label = _recurrence(block)
            drawing.recurrence(src, dst, kind, label)

    _rollback(drawing, model)
    if include_minimizers:
        _minimizers(drawing, model)
    return drawing


def _stem(name: str) -> str:
    """A model name made safe for a file name."""
    return re.sub(r"[^\w.-]+", "_", str(name)).strip("_") or "graph"


def _render(drawing: _Drawing, path: Path, undrawn: list[Path]) -> Path:
    import graphviz

    try:
        drawing.dot.render(filename=str(path), cleanup=True)
        return path.with_name(f"{path.name}.{drawing.dot.format}")
    except graphviz.ExecutableNotFound:
        source = path.with_name(f"{path.name}.gv")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(drawing.dot.source, encoding="utf-8")
        undrawn.append(source)
        return source


def plot_graphviz(
    model: Modely,
    out_dir: str | os.PathLike,
    filename: str | None = None,
    *,
    flatten: bool = False,
    include_minimizers: bool = True,
    format: str = "png",
) -> list[Path]:
    """Draw ``model`` into ``out_dir``, returning the files written.

    The root image is ``filename`` (the model's name by default) in ``format``;
    a suffix on ``filename``, or on ``out_dir`` itself, names the format
    instead. Without ``flatten`` every block's body is drawn as
    ``<filename>_<body name>``, recursively; with it the root is the only image.
    """
    try:
        import graphviz
    except ImportError as error:  # pragma: no cover - graphviz is a dependency
        raise ImportError(
            "Modely.plot() needs the graphviz Python package: pip install graphviz."
        ) from error

    out_path = Path(out_dir)
    if out_path.suffix[1:].lower() in _SUFFIX_FORMATS:
        filename = filename or out_path.name
        out_path = out_path.parent
    stem = filename or model.name
    suffix = Path(stem).suffix[1:].lower()
    if suffix in _SUFFIX_FORMATS:
        format, stem = suffix, Path(stem).stem
    stem = _stem(stem)
    out_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    undrawn: list[Path] = []

    if flatten:
        drawing = _draw_flattened(
            graphviz, model, format, include_minimizers=include_minimizers
        )
        written.append(_render(drawing, out_path / stem, undrawn))
    else:
        # A body is drawn once, however many blocks wrap it; two bodies that
        # share a name get files of their own.
        drawn: set[int] = set()
        taken: set[str] = set()

        def draw(sub: Modely, sub_stem: str, block) -> None:
            if id(sub) in drawn:
                return
            name, index = sub_stem, 2
            while name in taken:
                name, index = f"{sub_stem}_{index}", index + 1
            drawn.add(id(sub))
            taken.add(name)

            drawing = _draw_model(
                graphviz,
                sub,
                format,
                include_minimizers=include_minimizers,
                block=block,
            )
            written.append(_render(drawing, out_path / name, undrawn))
            for node in sub.order:
                ports = _block_ports(node)
                if ports is not None:
                    body, port_map = ports
                    draw(body, f"{name}_{_stem(body.name)}", (node, port_map))

        draw(model, stem, None)

    if undrawn:
        warnings.warn(
            "Graphviz's dot program was not found, so no image was drawn: the "
            f"DOT source of each is written instead ({', '.join(str(p) for p in undrawn)}). "
            "Install Graphviz (https://graphviz.org/download/) to render them.",
            UserWarning,
            stacklevel=3,
        )
    return written
