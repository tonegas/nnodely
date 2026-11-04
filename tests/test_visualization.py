import pytest
import json
import re
import shutil
from pathlib import Path

from nnodely import (
    Input,
    Output,
    EquationLearner,
    Fir,
    Linear,
    Loop,
    Modely,
    Parameter,
    Constant,
    Cos,
    Sin,
)


def test_plot_and_export_html(tmp_path):
    x = Input("x", dim=1, sw=3)
    out = Output("out", Fir(features=1)([x]))
    model = Modely("plot_model", inputs=[x], outputs=[out])
    model.build()

    model.export_html(out_dir=tmp_path, filename="plot_model")
    assert (tmp_path / "plot_model.html").exists()


def test_model_composed_visualization(tmp_path):
    # ------- Model definition and building -------
    x = Input("x", dim=1, sw=10)
    y = Input("y", dim=1, sw=10)

    x_stream = x
    y_stream = y

    fir = Fir(features=2)
    result_fir = fir([x_stream + y_stream])

    x_out = Output("x_pred", result_fir)
    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    # ------- Model composition -------
    z = Input("z", dim=1, sw=10)
    z_stream = z
    z_fir = Fir(features=1)([model1([z_stream, z_stream])])
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
    x = Input("x", dim=1, sw=5)
    param = Parameter("param1", dim=1)
    x_param = x * param
    x_out = Output("x_out", x_param)
    model1 = Modely("model1", inputs=[x], outputs=[x_out])
    model1.build()

    y = Input("y", dim=1, sw=5)
    y_fir = Fir(features=1)([model1([y])])
    y_out = Output("y_out", y_fir)
    model2 = Modely("model2", inputs=[y], outputs=[y_out])
    model2.build()

    z = Input("z", dim=1, sw=5)
    const_z = Constant("const_z", value=[[1.0, 2.0, 3.0, 4.0, 5.0]], time=1)
    z_fir = Fir(features=1)([model2([z + const_z])])
    z_out = Output("z_out", z_fir)
    model3 = Modely("model3", inputs=[z], outputs=[z_out])
    model3.build()

    # ------- Visualize the Flatten model -------
    model3.export_html(out_dir=tmp_path, filename="model3")
    model3.plot(tmp_path, "model3_standard")
    model3.plot(tmp_path, "model3_flattened", flatten=True)


def test_visualize_high_level_blocks(tmp_path):
    class Tangent:
        def __init__(self, name: str | None = "Tangent"):
            self.name = name

        def __call__(self, inputs):
            ret = Sin()(inputs) / Cos()(inputs)
            out = Output(name=f"{self.name}_out", stream=ret)
            return Modely(name=f"{self.name}", inputs=inputs, outputs=[out])(inputs)

    input = Input("input", sw=1)
    tan = Tangent(name="tangent")([input])
    output = Output("output", tan)

    model = Modely(name="model", inputs=[input], outputs=[output])
    model_flat = model.flatten()

    model.plot(tmp_path, "model_tangent")
    model_flat.plot(tmp_path, "model_tangent_flat")

    model.export_html(out_dir=tmp_path, filename="model_tangent")


def test_export_html_accepts_a_file_path(tmp_path):
    x = Input("x", dim=1, sw=3)
    out = Output("out", Fir(features=1)([x]))
    model = Modely("path_model", inputs=[x], outputs=[out])
    model.build()

    model.export_html(tmp_path / "named_page.html")

    assert (tmp_path / "named_page.html").is_file()
    assert (tmp_path / "named_page__flattened.html").is_file()


def test_export_html_describes_the_loop_boundary(tmp_path):
    seed_input = Input("in1", sw=1)
    body_output = Output("body_out", Linear(features=1)(seed_input))
    body = Modely("loop_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=5, sw=1)
    seed_window = seed
    loop = Loop(
        f=body,
        feedback={seed_input: body_output},
        name="loop_block",
        init={seed_input: seed_window},
    )()
    model = Modely("loop_model", inputs=[seed], outputs=[Output("out1", loop)])
    target = Input("target", seq=5, sw=1)
    model.minimize("err", source=model.outputs[0], target=target)
    model.build()

    model.export_html(out_dir=tmp_path, filename="loop_model")
    page = (tmp_path / "loop_model.html").read_text(encoding="utf-8")
    body_page = (tmp_path / "loop_model__loop_block.html").read_text(encoding="utf-8")

    # The block node carries the ports that bind the body to this graph.
    assert '"nested_model": "loop_body"' in page
    # The seed is the Input itself.
    outer = seed_window.name
    assert f'"body": "in1", "outer": "{outer}", "role": "initial"' in page
    assert '"from": "body_out", "to": "in1"' in page
    assert '"steps": 5' in page

    # The body page names the same ports and draws the recurrence it is under.
    assert '"parent": "loop_model"' in body_page
    assert '"label": "feedback (5 steps)"' in body_page

    # Training objectives are visible, and an unreferenced target is drawn.
    assert '"__min_err"' in page
    assert f'"from": "{target.name}", "to": "__min_err"' in page


def test_export_html_layout_survives_feedback_cycles(tmp_path):
    seed_input = Input("in1", sw=1)
    # Without weights: their Parameters would branch off the chain.
    body_output = Output("body_out", Sin(seed_input))
    body = Modely("cyc_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=5, sw=1)
    loop = Loop(
        f=body,
        feedback={seed_input: body_output},
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
    seed_input = Input("in1", sw=1)
    body_output = Output("body_out", Linear(features=1)(seed_input))
    body = Modely("inline_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=4, sw=1)
    seed_window = seed
    loop = Loop(
        f=body,
        feedback={seed_input: body_output},
        name="inline_block",
        init={seed_input: seed_window},
    )()
    model = Modely("inline_model", inputs=[seed], outputs=[Output("out1", loop)])
    model.build()
    model.export_html(out_dir=tmp_path, filename="inline_model")

    standard = (tmp_path / "inline_model.html").read_text(encoding="utf-8")
    flattened = (tmp_path / "inline_model__flattened.html").read_text(encoding="utf-8")

    # Modely.flatten leaves the Loop intact because build() runs the same pass
    # and it has to survive as one Keras layer, so the splice is display-only.
    assert len(model.flatten()._order) == len(model._order)

    standard_ids = {node["id"] for node in _graph(standard)[0]}
    flattened_ids = {node["id"] for node in _graph(flattened)[0]}
    assert "inline_block" in standard_ids
    assert "inline_block" not in flattened_ids
    assert {"inline_block/in1", "inline_block/body_out"} <= flattened_ids

    labels = {(e["from"], e["to"]): e["label"] for e in _graph(flattened)[1]}
    assert labels[(seed_window.name, "inline_block/in1")] == "initial"
    assert (
        labels[("inline_block/body_out", "inline_block/in1")] == "feedback (x4 steps)"
    )
    _assert_no_dangling(flattened)


def _minimized_plot_model(name):
    x = Input(f"{name}_x", dim=1, sw=3)
    y = Input(f"{name}_y", dim=1, sw=1)
    out = Output(f"{name}_out", Fir(features=1)([x]))
    model = Modely(name, inputs=[x, y], outputs=[out])
    model.minimize(f"{name}_fit", out, y)
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
    x = Input("escaped_x", sw=2)
    out = Output("o</script><script>alert(2)//", Fir(features=1)([x]))
    model = Modely("<img src=x onerror=alert(1)>", inputs=[x], outputs=[out]).build()

    model.export_html(tmp_path, filename="escaped")
    page = (tmp_path / "escaped.html").read_text(encoding="utf-8")

    assert "<img src=x" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "</script><script>alert(2)" not in page
    assert "o\\u003c/script\\u003e\\u003cscript\\u003ealert(2)//" in page


def test_export_html_loads_a_pinned_vis_network(tmp_path):
    x = Input("pinned_x", sw=2)
    model = Modely(
        "pinned",
        inputs=[x],
        outputs=[Output("pinned_out", Fir(features=1)([x]))],
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
    x = Input("nest_x", sw=5)
    model1 = Modely(
        "model1",
        inputs=[x],
        outputs=[Output("nest_x_out", x * Parameter("nest_p"))],
    ).build()
    y = Input("nest_y", sw=5)
    model2 = Modely(
        "model2",
        inputs=[y],
        outputs=[Output("nest_y_out", Fir(features=1)([model1([y])]))],
    ).build()
    z = Input("nest_z", sw=5)
    return Modely(
        "model3",
        inputs=[z],
        outputs=[Output("nest_z_out", Fir(features=1)([model2([z])]))],
    ).build()


def _looped_model():
    state, force = Input("loop_state", sw=1), Input("loop_force", sw=1)
    nxt = Output("loop_next", Linear(bias=False)(state) + force)
    body = Modely("step_body", inputs=[state, force], outputs=[nxt]).build()
    seed, forces = Input("loop_seed", sw=1), Input("loop_forces", seq=-1, sw=1)
    trajectory = Loop(
        f=body, feedback={state: nxt}, name="sim_loop", init={state: seed}
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
    x = Input("twice_x", sw=1)
    body = Modely(
        "twice_body", inputs=[x], outputs=[Output("twice_out", Sin(x))]
    ).build()
    y = Input("twice_y", sw=1)
    model = Modely(
        "twice",
        inputs=[y],
        outputs=[Output("twice_sum", body([y]) + body([y]))],
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


def test_plot_writes_the_shape_on_the_arrows_and_only_the_name_on_the_cards(
    tmp_path, monkeypatch
):
    acc = Input("dyn_acc", seq=-1, sw=1)
    model = Modely(
        "dynamic",
        inputs=[acc],
        outputs=[Output("dyn_vel", Sin()([acc]))],
    ).build()

    root = _dot_sources(model, tmp_path, monkeypatch)["dynamic"]

    assert re.search(r'dyn_acc -> \w+ \[label="\(dyn, 1, 1\)"\]', root)
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
            "Input", "Output", "Parameter", "Constant", "SubModel", "Loss", "loop"
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
    x = Input("in<put>&", sw=2)
    model = Modely(
        "odd<&>names",
        inputs=[x],
        outputs=[Output("out&<1>", Fir(features=1)([x]))],
    ).build()

    model.plot(tmp_path, "odd")

    assert (tmp_path / "odd.png").read_bytes().startswith(b"\x89PNG")


@requires_dot
def test_plot_draws_every_image_of_a_nested_model(tmp_path):
    _nested_models().plot(tmp_path, "nested")
    _looped_model().plot(tmp_path, "looped", flatten=True)

    for name in ("nested", "nested_model2", "nested_model2_model1", "looped"):
        assert (tmp_path / f"{name}.png").read_bytes().startswith(b"\x89PNG")


def _equation_learner(name, functions, features, linear_out=None):
    """An EquationLearner on x, every argument x, built into a model."""
    x = Input("x", sw=1)
    learner = EquationLearner(
        functions,
        linear_in=Linear(features=features, kernel="ones", bias=False),
        linear_out=linear_out,
        name=name,
    )
    Modely(
        f"{name}_model", inputs=[x], outputs=[Output(f"{name}_y", learner(x))]
    ).build()
    return learner


def test_equation_learner_plot_returns_the_equation_it_draws():
    import matplotlib.pyplot as plt
    import sympy

    x, v = Input("x", sw=1), Input("v", dim=2, sw=1)
    learner = EquationLearner(
        ["identity", Sin, (lambda a, b: a * b, 2)],
        linear_in=Linear(
            features=4,
            kernel=Parameter(
                "drawn_in",
                value=[
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0, 1.0],
                    [0.0, 0.004, 0.0, 1.0],
                ],
            ),
            bias=Parameter("drawn_in_bias", value=[0.0, 0.5, 0.0, 0.0]),
        ),
        linear_out=Linear(
            features=1,
            kernel=Parameter("drawn_out", value=[[2.0], [3.0], [0.123456]]),
            bias=False,
        ),
        name="drawn",
    )
    Modely(
        "drawn_model", inputs=[x, v], outputs=[Output("drawn_y", learner([x, v]))]
    ).build()

    # Named after the streams, a vector by feature, and expanded: the
    # function of two arguments computes on symbols, a product.
    sx, v0, v1 = sympy.symbols("x v_0 v_1")
    c = sympy.Float(0.1235)
    expected = 2 * sx + c * sx * v0 + c * sx * v1
    figures = len(plt.get_fignums())
    equation = learner.plot()
    assert equation == sympy.Eq(
        sympy.Symbol("y"),
        expected + 3 * sympy.sin(2 * v0 + sympy.Float(0.004) * v1 + sympy.Float(0.5)),  # type:ignore
    )
    # Drawn as LaTeX on a figure of its own.
    assert len(plt.get_fignums()) == figures + 1
    drawn = plt.gcf().axes[0].texts[0].get_text()
    assert drawn.startswith("$y = ") and drawn.endswith("$")
    assert r"\sin(2 v_{0} + 0.004 v_{1} + 0.5" in drawn

    # The coefficients below the threshold are left out, 0.004 v_1.
    learner.threshold = 0.01
    assert learner.plot().rhs == expected + 3 * sympy.sin(2 * v0 + sympy.Float(0.5))  # type:ignore
    # Rounded to the significant digits asked for.
    assert sympy.Float(0.12) in learner.plot(precision=2).rhs.atoms(sympy.Float)
    plt.close("all")


def test_equation_learner_plot_writes_the_basis_without_linear_out():
    import matplotlib.pyplot as plt
    import sympy

    learner = _equation_learner(
        "basis_drawn", ["sin", "sigmoid", "log10", "relu", "atan2"], features=6
    )
    equations = learner.plot()
    x = sympy.Symbol("x")

    assert [equation.lhs for equation in equations] == list(sympy.symbols("f_0:5"))
    assert equations[0].rhs == sympy.sin(x)
    # Written in mathematical notation where there is one, the other layers
    # by name.
    assert [sympy.latex(equation.rhs) for equation in equations[1:]] == [
        r"\sigma{\left(x \right)}",
        r"\log_{10}{\left(x \right)}",
        r"\operatorname{ReLU}{\left(x \right)}",
        r"\operatorname{atan2}{\left(x, x \right)}",
    ]
    # One line each.
    assert plt.gcf().axes[0].texts[0].get_text().count("\n") == 4
    plt.close("all")


def test_equation_learner_plot_writes_the_new_math_functions():
    import matplotlib.pyplot as plt
    import sympy

    learner = _equation_learner(
        "math_drawn", ["sinh", "log2", "log1p", "square", "round"], features=5
    )
    assert [sympy.latex(equation.rhs) for equation in learner.plot()] == [
        r"\sinh{\left(x \right)}",
        r"\log_{2}{\left(x \right)}",
        r"\log{\left(x + 1 \right)}",
        "x^{2}",
        r"\operatorname{round}{\left(x \right)}",
    ]

    x, v = Input("x", sw=1), Input("v", sw=1)
    binary = EquationLearner(
        ["maximum", "minimum", "mod"],
        linear_in=Linear(
            features=6,
            kernel=Parameter(
                "binary_drawn_in",
                value=[[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]],
            ),
            bias=False,
        ),
        name="binary_drawn",
    )
    Modely(
        "binary_drawn_model",
        inputs=[x, v],
        outputs=[Output("binary_y", binary([x, v]))],
    ).build()
    x, v = sympy.symbols("x v")
    assert [equation.rhs for equation in binary.plot()] == [
        sympy.Max(x, v),
        sympy.Min(x, v),
        sympy.Mod(x, v),
    ]
    plt.close("all")


def test_equation_learner_plot_writes_several_outputs_and_python_functions():
    import matplotlib.pyplot as plt
    import sympy

    from nnodely import Tanh

    def scaled_tanh(stream):
        return Tanh()(stream) * 2.0

    learner = _equation_learner(
        "python_drawn",
        # A function computing on streams is written by name, a lambda as
        # f_index; one that computes on symbols too, as what it computes.
        [scaled_tanh, lambda stream: Tanh()(stream), lambda a: a * a],
        features=3,
        linear_out=Linear(features=2, kernel="ones", bias=False),
    )
    equations = learner.plot()

    assert [equation.lhs for equation in equations] == list(sympy.symbols("y_0:2"))
    written = sympy.latex(equations[0].rhs)
    assert r"\operatorname{scaled\_tanh}" in written
    assert "f_{1}" in written
    assert "x^{2}" in written
    assert equations[0].rhs == equations[1].rhs
    plt.close("all")


def test_equation_learner_plot_draws_on_the_axes_given():
    import matplotlib.pyplot as plt

    learner = _equation_learner(
        "axes_drawn", ["sin"], features=1, linear_out=Linear(features=1)
    )
    figure, ax = plt.subplots()
    figures = plt.get_fignums()
    learner.plot(ax=ax)

    assert plt.get_fignums() == figures
    assert ax.texts[0].get_text().startswith("$y = ")
    assert not ax.axison
    plt.close("all")


def test_equation_learner_plot_breaks_a_long_equation_into_lines():
    import matplotlib.pyplot as plt

    # With the arguments of the default projection, each one its own.
    x = Input("x", sw=1)
    learner = EquationLearner(
        ["sin"] * 12, linear_out=Linear(features=1), name="long_drawn"
    )
    Modely("long_drawn_model", inputs=[x], outputs=[Output("y", learner(x))]).build()
    learner.plot()

    lines = plt.gcf().axes[0].texts[0].get_text().split("\n")
    assert len(lines) > 1
    assert all(line.startswith(r"$\qquad") for line in lines[1:])
    plt.close("all")


def test_equation_learner_plot_needs_a_built_model():
    learner = EquationLearner(["sin"], name="unbuilt_drawn")
    with pytest.raises(RuntimeError, match="build the model first"):
        learner.plot()

    learner(Input("unbuilt_drawn_x", sw=1))
    with pytest.raises(RuntimeError, match="build the model first"):
        learner.plot()
