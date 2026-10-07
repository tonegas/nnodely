import pytest
import json
import re
import shutil
from pathlib import Path

from nnodely import (
    Input,
    Output,
    Fir,
    Linear,
    Loop,
    Modely,
    Parameter,
    Constant,
    Cos,
    Sin,
    Roll,
    Integrate,
)


def test_plot_and_export_html(tmp_path):
    x = Input("x", dim=1)
    out = Output("out", Fir(out_features=1)([x.sw(3)]))
    model = Modely("plot_model", inputs=[x], outputs=[out])
    model.build()

    model.export_html(out_dir=tmp_path, filename="plot_model")
    assert (tmp_path / "plot_model.html").exists()


def test_model_composed_visualization(tmp_path):
    # ------- Model definition and building -------
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    x_stream = x.sw(10)
    y_stream = y.sw(10)

    fir = Fir(out_features=2)
    result_fir = fir([x_stream + y_stream])

    x_out = Output("x_pred", result_fir)
    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    # ------- Model composition -------
    z = Input("z", dim=1)
    z_stream = z.sw(10)
    z_fir = Fir(out_features=1)([model1([z_stream, z_stream])])
    z_out = Output("z_pred", z_fir)
    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    # ------- Model visualization -------
    model1.plot(tmp_path, "model1")
    model2.plot(tmp_path, "model2")

    # ------- Model export to HTML -------
    model1.export_html(out_dir=tmp_path, filename="model1")
    model2.export_html(out_dir=tmp_path, filename="model2")


def test_multiple_model_composed_visualization(tmp_path):
    # ------- Model Flatten and Visualization -------
    x = Input("x", dim=1)
    param = Parameter("param1", dim=1)
    x_param = x.sw(5) * param
    x_out = Output("x_out", x_param)
    model1 = Modely("model1", inputs=[x], outputs=[x_out])
    model1.build()

    y = Input("y", dim=1)
    y_fir = Fir(out_features=1)([model1([y.sw(5)])])
    y_out = Output("y_out", y_fir)
    model2 = Modely("model2", inputs=[y], outputs=[y_out])
    model2.build()

    z = Input("z", dim=1)
    const_z = Constant("const_z", value=[1.0, 2.0, 3.0, 4.0, 5.0])
    z_fir = Fir(out_features=1)([model2([z.sw(5) + const_z])])
    z_out = Output("z_out", z_fir)
    model3 = Modely("model3", inputs=[z], outputs=[z_out])
    model3.build()

    # ------- Visualize the Flatten model -------
    model3.export_html(out_dir=tmp_path, filename="model3")
    model3.plot(tmp_path, "model3_standard")
    model3.plot(tmp_path, "model3_flattened", flatten=True)


def test_visualize_roll(tmp_path):
    # ------- Model with roll connections -------
    x = Input(name="x", dim=1)
    y = Input(name="y", dim=1)
    r1 = x.sw(1) + y.sw(1)
    out1 = Output("out1", r1)
    model1 = Modely(name="model1", inputs=[x, y], outputs=[out1])
    model1.build()

    z = Input(name="z", dim=1)
    r2 = Fir(out_features=1, use_bias=False)(z.sw(5))
    roll_body_output = Output("roll_body_output", r2)
    roll_body = Modely(name="roll_body", inputs=[z], outputs=[roll_body_output]).build()
    roll_fn = Roll(
        f=roll_body,
        callback={z: roll_body_output},
        name="roll_fn",
    )
    out = Output("out", roll_fn)
    model = Modely(name="model", inputs=[z], outputs=[out])
    model.build()

    # ------- Model visualization -------
    model1.plot(tmp_path, "model1")
    model.plot(tmp_path, "model2")

    # ------- Model export to HTML -------
    model1.export_html(out_dir=tmp_path, filename="model1")
    model.export_html(out_dir=tmp_path, filename="model2")

    html = (tmp_path / "model2.html").read_text(encoding="utf-8")
    assert '"nested_model": "roll_body"' in html
    assert len(list(tmp_path.glob("*roll_fn.html"))) == 1


def test_visualize_model_roll(tmp_path):
    x = Input("closed_x")
    feedback = Fir(out_features=1, use_bias=False, name="closed_fir")(x.sw(5))
    output = Output("closed_out", feedback + x.last())
    model = Modely("closed_model", inputs=[x], outputs=[output])
    model.rollback({x: feedback}, steps=3)
    model.build()

    model.export_html(out_dir=tmp_path, filename="closed_model")
    html = (tmp_path / "closed_model.html").read_text(encoding="utf-8")
    flattened_html = (tmp_path / "closed_model__flattened.html").read_text(
        encoding="utf-8"
    )

    for page in (html, flattened_html):
        assert '"from": "closed_fir", "to": "closed_x"' in page
        assert '"label": "roll (3 steps)"' in page
        assert '\\"kind\\": \\"roll\\"' in page
        assert '"color": "#e74c3c"' in page


def test_visualize_high_level_blocks(tmp_path):
    class Tangent:
        def __init__(self, name: str | None = "Tangent"):
            self.name = name

        def __call__(self, inputs):
            ret = Sin()(inputs) / Cos()(inputs)
            out = Output(name=f"{self.name}_out", stream=ret)
            return Modely(name=f"{self.name}", inputs=inputs, outputs=[out])(inputs)

    input = Input("input")
    tan = Tangent(name="tangent")([input])
    output = Output("output", tan)

    model = Modely(name="model", inputs=[input], outputs=[output])
    model_flat = model.flatten()

    model.plot(tmp_path, "model_tangent")
    model_flat.plot(tmp_path, "model_tangent_flat")

    model.export_html(out_dir=tmp_path, filename="model_tangent")


def test_export_html_accepts_a_file_path(tmp_path):
    x = Input("x", dim=1)
    out = Output("out", Fir(out_features=1)([x.sw(3)]))
    model = Modely("path_model", inputs=[x], outputs=[out])
    model.build()

    model.export_html(tmp_path / "named_page.html")

    assert (tmp_path / "named_page.html").is_file()
    assert (tmp_path / "named_page__flattened.html").is_file()


def test_export_html_describes_the_loop_boundary(tmp_path):
    seed_input = Input("in1")
    body_output = Output("body_out", Linear(out_features=1)(seed_input.last()))
    body = Modely("loop_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=5)
    loop = Loop(
        f=body,
        callback={seed_input: body_output},
        name="loop_block",
        init={seed_input: seed},
    )()
    model = Modely("loop_model", inputs=[seed], outputs=[Output("out1", loop)])
    model.minimize("err", source=model.outputs[0], target=Input("target", seq=5))
    model.build()

    model.export_html(out_dir=tmp_path, filename="loop_model")
    page = (tmp_path / "loop_model.html").read_text(encoding="utf-8")
    body_page = (tmp_path / "loop_model__loop_block.html").read_text(encoding="utf-8")

    # The block node carries the ports that bind the body to this graph.
    assert '"nested_model": "loop_body"' in page
    assert '"body": "in1", "outer": "in1_seq", "role": "initial"' in page
    assert '"from": "body_out", "to": "in1"' in page
    assert '"steps": 5' in page

    # The body page names the same ports and draws the recurrence it is under.
    assert '"parent": "loop_model"' in body_page
    assert '"label": "feedback (5 steps)"' in body_page

    # Training objectives are visible, and an unreferenced target is drawn.
    assert '"__min_err"' in page
    assert '"from": "target", "to": "__min_err"' in page


def test_export_html_layout_survives_feedback_cycles(tmp_path):
    seed_input = Input("in1")
    body_output = Output("body_out", Linear(out_features=1)(seed_input.last()))
    body = Modely("cyc_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=5)
    loop = Loop(
        f=body,
        callback={seed_input: body_output},
        name="cyc_block",
        init={seed_input: seed},
    )()
    model = Modely("cyc_model", inputs=[seed], outputs=[Output("out1", loop)])
    model.build()

    model.export_html(out_dir=tmp_path, filename="cyc_model")
    body_page = (tmp_path / "cyc_model__cyc_block.html").read_text(encoding="utf-8")

    # Levels are assigned from the acyclic edges only: vis-network's own
    # "directed" sort collapses every node into one column once an edge
    # points backwards, which the Loop feedback arrow does.
    nodes = json.loads(
        re.search(
            r"const nodes = new vis\.DataSet\((\[.*?\])\);\n", body_page, re.S
        ).group(1)  # type: ignore
    )
    # The body is a straight chain, so a working layout gives every node its
    # own level; the collapse showed up as every node sharing level 0.
    assert sorted(node["level"] for node in nodes) == list(range(len(nodes)))


def _graph(page):
    """Node ids, edges and levels of an exported page."""
    nodes = json.loads(
        re.search(r"const nodes = new vis\.DataSet\((\[.*?\])\);\n", page, re.S).group(  # type: ignore
            1
        )
    )
    edges = json.loads(
        re.search(r"const edges = new vis\.DataSet\((\[.*?\])\);\n", page, re.S).group(  # type: ignore
            1
        )
    )
    return nodes, edges


def _assert_no_dangling(page):
    nodes, edges = _graph(page)
    ids = {node["id"] for node in nodes}
    assert [(e["from"], e["to"]) for e in edges if {e["from"], e["to"]} - ids] == []


def test_flattened_page_inlines_a_loop_body(tmp_path):
    seed_input = Input("in1")
    body_output = Output("body_out", Linear(out_features=1)(seed_input.last()))
    body = Modely("inline_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=4)
    loop = Loop(
        f=body,
        callback={seed_input: body_output},
        name="inline_block",
        init={seed_input: seed},
    )()
    model = Modely("inline_model", inputs=[seed], outputs=[Output("out1", loop)])
    model.build()
    model.export_html(out_dir=tmp_path, filename="inline_model")

    standard = (tmp_path / "inline_model.html").read_text(encoding="utf-8")
    flattened = (tmp_path / "inline_model__flattened.html").read_text(encoding="utf-8")

    # Modely.flatten leaves the Loop intact because build() runs the same pass
    # and it has to survive as one Keras layer, so the splice is display-only.
    assert len(model.flatten().order) == len(model.order)

    standard_ids = {node["id"] for node in _graph(standard)[0]}
    flattened_ids = {node["id"] for node in _graph(flattened)[0]}
    assert "inline_block" in standard_ids
    assert "inline_block" not in flattened_ids
    assert {"inline_block/in1", "inline_block/body_out"} <= flattened_ids

    labels = {(e["from"], e["to"]): e["label"] for e in _graph(flattened)[1]}
    assert labels[("in1_seq", "inline_block/in1")] == "initial"
    assert (
        labels[("inline_block/body_out", "inline_block/in1")] == "feedback (x4 steps)"
    )
    _assert_no_dangling(flattened)


def test_flattened_page_inlines_a_roll_body(tmp_path):
    z = Input("z2")
    roll_out = Output("roll_out", Fir(out_features=1, use_bias=False)(z.sw(5)))
    roll_body = Modely("inline_roll_body", inputs=[z], outputs=[roll_out]).build()
    roll = Roll(f=roll_body, callback={z: roll_out}, name="inline_roll")
    model = Modely("inline_roll_model", inputs=[z], outputs=[Output("out", roll)])
    model.build()
    model.export_html(out_dir=tmp_path, filename="inline_roll_model")

    flattened = (tmp_path / "inline_roll_model__flattened.html").read_text(
        encoding="utf-8"
    )
    nodes, edges = _graph(flattened)
    ids = {node["id"] for node in nodes}

    # A Roll node is itself the result, so removing it must reconnect its
    # consumers to the body output rather than leave them dangling.
    assert "inline_roll" not in ids
    assert ("inline_roll/roll_out", "out") in {(e["from"], e["to"]) for e in edges}
    _assert_no_dangling(flattened)


def _minimized_plot_model(name):
    x = Input(f"{name}_x", dim=1)
    y = Input(f"{name}_y", dim=1)
    out = Output(f"{name}_out", Fir(out_features=1)([x.sw(3)]))
    model = Modely(name, inputs=[x, y], outputs=[out])
    model.minimize(f"{name}_fit", out, y.last())
    return model.build()


def _dot_sources(model, out_dir, monkeypatch, **kwargs) -> dict[str, str]:
    """The DOT source of every image ``plot`` draws, by file stem.

    The ``dot`` program is made unavailable, so each image is written as its
    DOT source - exactly what would have been drawn - without Graphviz.
    """
    import graphviz

    def no_dot(*args, **kwargs):
        raise graphviz.ExecutableNotFound(["dot"])

    monkeypatch.setattr(graphviz.Digraph, "render", no_dot)
    with pytest.warns(UserWarning, match="dot program was not found"):
        model.plot(out_dir, **kwargs)
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(Path(out_dir).glob("*.gv"))
    }


def _card_colour(source: str, node: str) -> str:
    """The fill colour of the card drawn for ``node``."""
    match = re.search(
        rf'^\s*"?{re.escape(node)}"? \[label=<<TABLE[^>]*BGCOLOR="([^"]+)"',
        source,
        re.M,
    )
    assert match is not None, f"no card for {node!r}"
    return match.group(1)


requires_dot = pytest.mark.skipif(
    shutil.which("dot") is None,
    reason="drawing an image needs Graphviz's dot program, which is not installed",
)


def test_plot_names_a_minimizer_after_itself(tmp_path, monkeypatch):
    # Once labelled with the loss function's repr, memory address included.
    source = _dot_sources(_minimized_plot_model("labelled"), tmp_path, monkeypatch)[
        "labelled"
    ]

    assert ">labelled_fit</B>" in source
    assert " at 0x" not in source


@requires_dot
def test_plot_draws_into_a_folder_with_a_dot_in_its_name(tmp_path):
    model = _minimized_plot_model("dotted")

    model.plot(tmp_path / "run.1")

    assert (tmp_path / "run.1" / "dotted.png").read_bytes().startswith(b"\x89PNG")


def test_plot_without_graphviz_writes_the_dot_source_and_warns(tmp_path, monkeypatch):
    # Once wrote the DOT text into the .png itself, without a word.
    sources = _dot_sources(_minimized_plot_model("undrawn"), tmp_path, monkeypatch)

    assert not (tmp_path / "undrawn.png").exists()
    assert sources["undrawn"].startswith("digraph")


def test_export_html_escapes_the_names_it_shows(tmp_path):
    # Once inserted raw: a name could close the page's <script>, or run its own.
    x = Input("escaped_x")
    out = Output("o</script><script>alert(2)//", Fir(out_features=1)([x.sw(2)]))
    model = Modely("<img src=x onerror=alert(1)>", inputs=[x], outputs=[out]).build()

    model.export_html(tmp_path, filename="escaped")
    page = (tmp_path / "escaped.html").read_text(encoding="utf-8")

    assert "<img src=x" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "</script><script>alert(2)" not in page
    assert "o\\u003c/script\\u003e\\u003cscript\\u003ealert(2)//" in page


def test_export_html_loads_a_pinned_vis_network(tmp_path):
    x = Input("pinned_x")
    model = Modely(
        "pinned",
        inputs=[x],
        outputs=[Output("pinned_out", Fir(out_features=1)([x.sw(2)]))],
    ).build()

    model.export_html(tmp_path)
    page = (tmp_path / "pinned.html").read_text(encoding="utf-8")

    assert (
        'src="https://unpkg.com/vis-network@10.1.2/standalone/umd/vis-network.min.js"'
        in page
    )
    assert 'integrity="sha384-' in page


# ---------------------------------------------------------------------------
# plot(): one image per model, or one flattened image, in the nnodely style
# ---------------------------------------------------------------------------


def _nested_models():
    """model3 calls model2, which calls model1."""
    x = Input("nest_x")
    model1 = Modely(
        "model1",
        inputs=[x],
        outputs=[Output("nest_x_out", x.sw(5) * Parameter("nest_p"))],
    ).build()
    y = Input("nest_y")
    model2 = Modely(
        "model2",
        inputs=[y],
        outputs=[Output("nest_y_out", Fir(out_features=1)([model1([y.sw(5)])]))],
    ).build()
    z = Input("nest_z")
    return Modely(
        "model3",
        inputs=[z],
        outputs=[Output("nest_z_out", Fir(out_features=1)([model2([z.sw(5)])]))],
    ).build()


def _looped_model():
    state, force = Input("loop_state"), Input("loop_force")
    nxt = Output("loop_next", Linear(use_bias=False)(state.last()) + force.last())
    body = Modely("step_body", inputs=[state, force], outputs=[nxt]).build()
    seed, forces = Input("loop_seed"), Input("loop_forces", seq=-1)
    trajectory = Loop(
        f=body, callback={state: nxt}, name="sim_loop", init={state: seed}
    )({force: forces})
    return Modely(
        "simulator", inputs=[seed, forces], outputs=[Output("loop_traj", trajectory)]
    ).build()


def test_plot_names_the_root_image_after_the_model(tmp_path, monkeypatch):
    sources = _dot_sources(
        _minimized_plot_model("named"), tmp_path / "figs", monkeypatch
    )
    assert list(sources) == ["named"]

    renamed = _dot_sources(
        _minimized_plot_model("renamed"),
        tmp_path / "other",
        monkeypatch,
        filename="chosen",
    )
    assert list(renamed) == ["chosen"]


def test_plot_draws_every_nested_body_in_an_image_of_its_own(tmp_path, monkeypatch):
    sources = _dot_sources(_nested_models(), tmp_path, monkeypatch, filename="graph")

    assert set(sources) == {"graph", "graph_model2", "graph_model2_model1"}
    # The call of a sub-model is one SubModel node of the graph that calls it.
    assert _card_colour(sources["graph"], "model2_call") == "#6C75DF"
    assert _card_colour(sources["graph_model2"], "model1_call") == "#6C75DF"


def test_plot_draws_a_body_called_twice_once(tmp_path, monkeypatch):
    x = Input("twice_x")
    body = Modely(
        "twice_body", inputs=[x], outputs=[Output("twice_out", Sin(x.last()))]
    ).build()
    y = Input("twice_y")
    model = Modely(
        "twice",
        inputs=[y],
        outputs=[Output("twice_sum", body([y.last()]) + body([y.last()]))],
    ).build()

    sources = _dot_sources(model, tmp_path, monkeypatch)

    assert set(sources) == {"twice", "twice_twice_body"}


def test_flattened_plot_is_one_image_with_a_box_per_sub_model(tmp_path, monkeypatch):
    sources = _dot_sources(_nested_models(), tmp_path, monkeypatch, flatten=True)

    assert list(sources) == ["model3"]
    source = sources["model3"]
    assert source.count("subgraph cluster_") == 2
    # model1's box, labelled with its name, is drawn inside model2's.
    assert source.index(">model2</B>") < source.index(">model1</B>")
    assert "model2_call" not in re.findall(r'^\s*"?([\w/]+)"? \[label', source, re.M)


def test_plot_draws_a_loop_as_a_sub_model_and_its_recurrence_as_an_arrow(
    tmp_path, monkeypatch
):
    sources = _dot_sources(_looped_model(), tmp_path, monkeypatch)

    assert set(sources) == {"simulator", "simulator_step_body"}
    # A Loop has a body, so it is a SubModel node: no Loop block of its own.
    assert _card_colour(sources["simulator"], "sim_loop") == "#6C75DF"
    # Its body draws the feedback the Loop closes, as an orange "loop" arrow.
    feedback = re.search(
        r"loop_next -> loop_state \[([^\]]*)\]", sources["simulator_step_body"]
    )
    assert feedback is not None
    assert "<B>loop</B>" in feedback.group(1)
    assert "#F28C28" in feedback.group(1) and "constraint=false" in feedback.group(1)


def test_flattened_plot_draws_a_loop_by_its_arrow_alone(tmp_path, monkeypatch):
    source = _dot_sources(_looped_model(), tmp_path, monkeypatch, flatten=True)[
        "simulator"
    ]

    assert "subgraph cluster_" not in source  # no box around a Loop body
    assert re.search(
        r'"sim_loop/loop_next" -> "sim_loop/loop_state" \[[^\]]*<B>loop</B>', source
    )


def test_plot_draws_roll_and_rollback_recurrences(tmp_path, monkeypatch):
    z = Input("rolled_z")
    out = Output("rolled_out", Fir(out_features=1, use_bias=False)(z.sw(5)))
    body = Modely("rolled_body", inputs=[z], outputs=[out]).build()
    rolled = Modely(
        "rolled",
        inputs=[z],
        outputs=[Output("rolled_result", Roll(f=body, callback={z: out}))],
    ).build()
    x = Input("closed_x")
    fir = Fir(out_features=1, use_bias=False, name="closed_fir")(x.sw(5))
    closed = Modely(
        "closed", inputs=[x], outputs=[Output("closed_out", fir + x.last())]
    )
    closed.rollback({x: fir}, steps=3)
    closed.build()

    roll_sources = _dot_sources(rolled, tmp_path / "roll", monkeypatch)
    rollback_source = _dot_sources(closed, tmp_path / "rollback", monkeypatch)["closed"]

    assert "<B>roll (steps=5)</B>" in roll_sources["rolled_rolled_body"]
    assert "#17A398" in roll_sources["rolled_rolled_body"]
    assert re.search(
        r"closed_fir -> closed_x \[[^\]]*roll \(steps=3\)", rollback_source
    )


def test_plot_writes_the_shape_on_the_arrows_and_only_the_name_on_the_cards(
    tmp_path, monkeypatch
):
    acc = Input("dyn_acc", seq=-1)
    model = Modely(
        "dynamic", inputs=[acc], outputs=[Output("dyn_vel", Integrate(dt=0.1)(acc))]
    ).build()

    root = _dot_sources(model, tmp_path, monkeypatch)["dynamic"]

    assert re.search(r'dyn_acc -> \w+ \[label="\(1, 1, dyn\)"\]', root)
    # A card holds its name, and nothing else: no class, no shape.
    for card in re.findall(r"<TABLE BORDER=\"1\".*?</TABLE>", root):
        assert card.count("<TD>") == 1
    assert "dim" not in root


def test_every_image_carries_the_logo_and_the_legend(tmp_path, monkeypatch):
    for source in _dot_sources(_looped_model(), tmp_path, monkeypatch).values():
        assert "logo.png" in source
        legend = re.findall(
            r'<TD ALIGN="LEFT"><FONT POINT-SIZE="9"[^>]*>([^<]+)</FONT>', source
        )
        assert legend == [
            "Input", "Output", "Parameter", "Constant", "SubModel", "Loss", "loop", "roll"
        ]  # fmt: skip


def test_plot_can_leave_the_minimizers_out(tmp_path, monkeypatch):
    with_loss = _dot_sources(
        _minimized_plot_model("with_loss"), tmp_path / "a", monkeypatch
    )
    without = _dot_sources(
        _minimized_plot_model("without_loss"),
        tmp_path / "b",
        monkeypatch,
        include_minimizers=False,
    )

    assert "__loss_with_loss_fit" in with_loss["with_loss"]
    assert "__loss_" not in without["without_loss"]


def test_plot_colours_inputs_green_and_outputs_red(tmp_path, monkeypatch):
    source = _dot_sources(_minimized_plot_model("coloured"), tmp_path, monkeypatch)[
        "coloured"
    ]

    assert _card_colour(source, "coloured_x") == "#2ECC71"
    assert _card_colour(source, "coloured_out") == "#E74C3C"
    # A target no output reads is drawn as the Input it is a window of.
    assert _card_colour(source, "coloured_y") == "#2ECC71"


@requires_dot
def test_plot_writes_the_format_its_path_names(tmp_path):
    model = _minimized_plot_model("formats")

    model.plot(tmp_path / "graph.svg")
    model.plot(tmp_path, "document", format="pdf")

    assert b"<svg" in (tmp_path / "graph.svg").read_bytes()
    assert (tmp_path / "document.pdf").read_bytes().startswith(b"%PDF")


@requires_dot
def test_plot_draws_names_graphviz_would_misread(tmp_path):
    # Names go into HTML-like labels, where <, > and & would break the image.
    x = Input("in<put>&")
    model = Modely(
        "odd<&>names",
        inputs=[x],
        outputs=[Output("out&<1>", Fir(out_features=1)([x.sw(2)]))],
    ).build()

    model.plot(tmp_path, "odd")

    assert (tmp_path / "odd.png").read_bytes().startswith(b"\x89PNG")


@requires_dot
def test_plot_draws_every_image_of_a_nested_model(tmp_path):
    _nested_models().plot(tmp_path, "nested")
    _looped_model().plot(tmp_path, "looped", flatten=True)

    for name in ("nested", "nested_model2", "nested_model2_model1", "looped"):
        assert (tmp_path / f"{name}.png").read_bytes().startswith(b"\x89PNG")
