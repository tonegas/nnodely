import pytest
import json
import re

from nnodely import (
    Input,
    Integrate,
    ModelyLoop,
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


def test_export_html_writes_the_root_page(tmp_path):
    x = Input("x", dim=1, tw=3)
    out = Output("out", Fir(features=1)([x]))
    model = Modely("plot_model", inputs=[x], outputs=[out])
    model.build()

    model.export_html(out_dir=tmp_path, filename="plot_model")
    assert (tmp_path / "plot_model.html").exists()


def test_model_composed_visualization(tmp_path):
    # ------- Model definition and building -------
    x = Input("x", dim=1, tw=10)
    y = Input("y", dim=1, tw=10)

    x_stream = x
    y_stream = y

    fir = Fir(features=2)
    result_fir = fir([x_stream + y_stream])

    x_out = Output("x_pred", result_fir)
    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    # ------- Model composition -------
    z = Input("z", dim=1, tw=10)
    z_stream = z
    z_fir = Fir(features=1)([model1([z_stream, z_stream])])
    z_out = Output("z_pred", z_fir)
    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    # ------- Model export to HTML -------
    model1.export_html(out_dir=tmp_path, filename="model1")
    model2.export_html(out_dir=tmp_path, filename="model2")


def test_multiple_model_composed_visualization(tmp_path):
    # ------- Model Flatten and Visualization -------
    x = Input("x", dim=1, tw=5)
    param = Parameter("param1", dim=1)
    x_param = x * param
    x_out = Output("x_out", x_param)
    model1 = Modely("model1", inputs=[x], outputs=[x_out])
    model1.build()

    y = Input("y", dim=1, tw=5)
    y_fir = Fir(features=1)([model1([y])])
    y_out = Output("y_out", y_fir)
    model2 = Modely("model2", inputs=[y], outputs=[y_out])
    model2.build()

    z = Input("z", dim=1, tw=5)
    const_z = Constant("const_z", value=[1.0, 2.0, 3.0, 4.0, 5.0], tw=5)
    z_fir = Fir(features=1)([model2([z + const_z])])
    z_out = Output("z_out", z_fir)
    model3 = Modely("model3", inputs=[z], outputs=[z_out])
    model3.build()

    # ------- Visualize the Flatten model -------
    model3.export_html(out_dir=tmp_path, filename="model3")


def test_visualize_high_level_blocks(tmp_path):
    class Tangent:
        def __init__(self, name: str | None = "Tangent"):
            self.name = name

        def __call__(self, inputs):
            ret = Sin()(inputs) / Cos()(inputs)
            out = Output(name=f"{self.name}_out", stream=ret)
            return Modely(name=f"{self.name}", inputs=inputs, outputs=[out])(inputs)

    input = Input("input", tw=1)
    tan = Tangent(name="tangent")([input])
    output = Output("output", tan)

    model = Modely(name="model", inputs=[input], outputs=[output])
    model_flat = model.flatten()

    model.export_html(out_dir=tmp_path, filename="model_tangent")
    model_flat.export_html(out_dir=tmp_path, filename="model_tangent_flat")


def test_export_html_accepts_a_file_path(tmp_path):
    x = Input("x", dim=1, tw=3)
    out = Output("out", Fir(features=1)([x]))
    model = Modely("path_model", inputs=[x], outputs=[out])
    model.build()

    model.export_html(tmp_path / "named_page.html")

    assert (tmp_path / "named_page.html").is_file()
    assert (tmp_path / "named_page__flattened.html").is_file()


def test_export_html_describes_the_loop_boundary(tmp_path):
    seed_input = Input("in1", tw=1)
    body_output = Output("body_out", Linear(features=1)(seed_input))
    body = Modely("loop_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=5, tw=1)
    seed_window = seed
    loop = Loop(
        f=body,
        feedback={seed_input: body_output},
        name="loop_block",
        init={seed_input: seed_window},
    )()
    model = Modely("loop_model", inputs=[seed], outputs=[Output("out1", loop)])
    target = Input("target", seq=5, tw=1)
    model.minimize("err", source=model.outputs[0], target=target)
    model.build()

    model.export_html(out_dir=tmp_path, filename="loop_model")
    page = (tmp_path / "loop_model.html").read_text(encoding="utf-8")
    body_page = (tmp_path / "loop_model__loop_block.html").read_text(encoding="utf-8")

    # The block node carries the ports that bind the body to this graph.
    assert '"nested_model": "loop_body"' in page
    # The seed is the Input itself.
    outer = seed_window.name
    assert f'"body_label": "in1", "outer": "n0", "outer_label": "{outer}"' in page
    assert '"from_label": "body_out", "to_label": "in1"' in page
    assert '"steps": 5' in page

    # The body page names the same ports and draws the recurrence it is under.
    assert '"parent": "loop_model"' in body_page
    assert '"label": "feedback (5 steps)"' in body_page

    # Training objectives are visible, and an unreferenced target is drawn.
    names = _names(page)
    edges = {(names[e["from"]], names[e["to"]]) for e in _graph(page)[1]}
    assert (target.name, "err") in edges


def test_export_html_layout_survives_feedback_cycles(tmp_path):
    seed_input = Input("in1", tw=1)
    # Without weights: their Parameters would branch off the chain.
    body_output = Output("body_out", Sin(seed_input))
    body = Modely("cyc_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=5, tw=1)
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


def _names(page):
    """What the page shows for every node id: its label, after the names of
    the regions it is spliced in - ``block/body/x`` - as a reader finds it.
    The ids themselves are the page's own, n0, n1, ..."""
    nodes, _ = _graph(page)
    regions = json.loads(
        re.search(r"const REGIONS = (\[.*?\]);\n", page, re.S).group(1)  # type: ignore
    )
    region_names = {region["id"]: region["name"] for region in regions}
    names = {}
    for node in nodes:
        parts = node["id"].split("/")
        path = ["/".join(parts[: depth + 1]) for depth in range(len(parts) - 1)]
        names[node["id"]] = "/".join([region_names[r] for r in path] + [node["label"]])
    return names


def _assert_no_dangling(page):
    nodes, edges = _graph(page)
    ids = {node["id"] for node in nodes}
    assert [(e["from"], e["to"]) for e in edges if {e["from"], e["to"]} - ids] == []


def test_flattened_page_inlines_a_loop_body(tmp_path):
    seed_input = Input("in1", tw=1)
    body_output = Output("body_out", Linear(features=1)(seed_input))
    body = Modely("inline_body", inputs=[seed_input], outputs=[body_output]).build()

    seed = Input("in1_seq", seq=4, tw=1)
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

    standard_names = set(_names(standard).values())
    names = _names(flattened)
    assert "inline_block" in standard_names
    assert "inline_block" not in names.values()
    assert {"inline_block/in1", "inline_block/body_out"} <= set(names.values())

    labels = {
        (names[e["from"]], names[e["to"]]): e["label"] for e in _graph(flattened)[1]
    }
    assert labels[(seed_window.name, "inline_block/in1")] == "initial"
    assert (
        labels[("inline_block/body_out", "inline_block/in1")] == "feedback (x4 steps)"
    )
    _assert_no_dangling(flattened)


def test_a_loop_inlined_from_a_called_model_reads_the_streams_the_call_binds(
    tmp_path,
):
    # Inlined in the step of the outer loop, the inner Loop reads k, which
    # the call binds to its x; it was once drawn reading x, and k was left
    # alone on the flattened page.
    x = Input("bound_x", tw=1)
    inner = ModelyLoop(
        "bound_inner", inputs=[x], outputs=[Output("bound_xx", Integrate(dt=1)(x))]
    ).build()
    k = Input("bound_k", tw=1, seq=-1)
    outer = ModelyLoop(
        "bound_outer",
        inputs=[k],
        outputs=[Output("bound_kk", Integrate(dt=1)(inner(k)))],
    ).build()
    outer.export_html(out_dir=tmp_path, filename="bound_outer")

    page = (tmp_path / "bound_outer__flattened.html").read_text(encoding="utf-8")
    nodes, edges = _graph(page)
    linked = {e["from"] for e in edges} | {e["to"] for e in edges}
    assert [node["id"] for node in nodes if node["id"] not in linked] == []
    names = _names(page)
    assert ("bound_k", "bound_outer_loop/bound_inner_loop/bound_x") in {
        (names[e["from"]], names[e["to"]]) for e in edges
    }
    _assert_no_dangling(page)

    # The state of a called loop starts from the stream the call binds.
    y0 = Input("bound_y0", tw=1)
    seeded = ModelyLoop(
        "bound_seeded",
        inputs=[x, y0],
        outputs=[Output("bound_xs", Integrate(dt=1, init=y0)(x))],
    ).build()
    k0 = Input("bound_k0", tw=1, seq=-1)
    caller = Modely(
        "bound_caller", inputs=[k, k0], outputs=[Output("bound_o", seeded([k, k0]))]
    ).build()
    table = str(caller.inputs)
    assert "bound_k0, first step" in table
    assert "bound_y0" not in table


def test_export_html_escapes_the_names_it_shows(tmp_path):
    # Once inserted raw: a name could close the page's <script>, or run its own.
    x = Input("escaped_x", tw=2)
    out = Output("o</script><script>alert(2)//", Fir(features=1)([x]))
    model = Modely("<img src=x onerror=alert(1)>", inputs=[x], outputs=[out]).build()

    model.export_html(tmp_path, filename="escaped")
    page = (tmp_path / "escaped.html").read_text(encoding="utf-8")

    assert "<img src=x" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "</script><script>alert(2)" not in page
    assert "o\\u003c/script\\u003e\\u003cscript\\u003ealert(2)//" in page


def test_export_html_loads_a_pinned_vis_network(tmp_path):
    x = Input("pinned_x", tw=2)
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


def _equation_learner(name, functions, features, linear_out=None):
    """An EquationLearner on x, every argument x, built into a model."""
    x = Input("x", tw=1)
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

    x, v = Input("x", tw=1), Input("v", dim=2, tw=1)
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


def test_equation_learner_plot_gives_every_stream_a_variable_of_its_own():
    import matplotlib.pyplot as plt
    import sympy

    def learner_on(streams, inputs, name):
        learner = EquationLearner(
            ["identity"],
            linear_in=Linear(
                features=1,
                kernel=Parameter(f"{name}_k", value=[[1.0], [2.0]]),
                bias=False,
            ),
            name=name,
        )
        Modely(
            f"{name}_model",
            inputs=inputs,
            outputs=[Output(f"{name}_y", learner(streams))],
        ).build()
        return learner

    # Two streams of one name are two variables: sympy took two symbols of
    # one name for one, and added up their terms into 3 p.
    p = Input("sym_p", tw=1)
    q = p * 2.0
    q.name = "sym_p"
    learner = learner_on([p, q], [p], "sym_twins")
    first, second = sympy.symbols("sym_p sym_p_2")
    assert learner.plot().rhs == first + 2 * second  # type: ignore[union-attr]

    # The same stream twice is one variable.
    learner = learner_on([p, p], [p], "sym_same")
    assert learner.plot().rhs == 3 * first  # type: ignore[union-attr]

    # Named as the streams are now: renamed after the call, the new name.
    r = Input("sym_r", tw=1)
    s = Input("sym_s", tw=1)
    learner = learner_on([r, s], [r, s], "sym_renamed")
    r.name = "sym_speed"
    assert learner.plot().rhs == sympy.Symbol("sym_speed") + 2 * sympy.Symbol("sym_s")  # type: ignore[union-attr]

    # A left side nnodely names yields to a variable of the user's: f_0.
    f = Input("f_0", tw=1)
    learner = learner_on([f, s], [f, s], "sym_side")
    equation = learner.plot()
    assert equation.lhs == sympy.Symbol("f_0_2")  # type: ignore[union-attr]
    assert equation.rhs == sympy.Symbol("f_0") + 2 * sympy.Symbol("sym_s")  # type: ignore[union-attr]
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

    x, v = Input("x", tw=1), Input("v", tw=1)
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
    x = Input("x", tw=1)
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

    learner(Input("unbuilt_drawn_x", tw=1))
    with pytest.raises(RuntimeError, match="build the model first"):
        learner.plot()


# ---------------------------------------------------------------------------
# Ids are the page's own; labels are the names, never changed
# ---------------------------------------------------------------------------


def _pages(model, tmp_path):
    model.export_html(out_dir=tmp_path, filename=model.name)
    standard = (tmp_path / f"{model.name}.html").read_text(encoding="utf-8")
    flattened = (tmp_path / f"{model.name}__flattened.html").read_text(encoding="utf-8")
    return standard, flattened


def test_a_name_like_a_path_does_not_hide_a_node(tmp_path):
    # Once, a node of the graph named like the id of a spliced body node -
    # "block/x" - took its id and vanished from the flattened page.
    x = Input("path_x", tw=1)
    block = Modely("path_block", inputs=[x], outputs=[Output("path_y", x * 2.0)])
    u = Input("path_u", tw=1)
    named = u * 3.0
    named.name = "path_block/path_x"
    model = Modely(
        "path_model", inputs=[u], outputs=[Output("p", block([u])), Output("q", named)]
    )

    _, flattened = _pages(model, tmp_path)
    labels = [node["label"] for node in _graph(flattened)[0]]
    assert "path_block/path_x" in labels
    assert "path_x" in labels
    _assert_no_dangling(flattened)


def test_a_node_named_like_a_loss_is_drawn_apart(tmp_path):
    x = Input("loss_x", tw=1)
    y = Output("loss_y", Linear(features=1)(x))
    named = x * 1.0
    named.name = "__min_fit"
    model = Modely("loss_model", inputs=[x], outputs=[y, Output("loss_z", named)])
    model.minimize("fit", source=y, target=Input("loss_t", tw=1))

    standard, _ = _pages(model, tmp_path)
    nodes = _graph(standard)[0]
    assert len({node["id"] for node in nodes}) == len(nodes)
    assert {"__min_fit", "fit"} <= {node["label"] for node in nodes}
    _assert_no_dangling(standard)


def test_a_renamed_modelyloop_state_is_drawn_once(tmp_path):
    # The rollout Input a ModelyLoop makes of its state is bound to it by
    # object: renamed, they are still one node - once two, matched by name.
    x, u = Input("ren_x", tw=1), Input("ren_u", tw=1)
    x.feedback(x + u)
    sim = ModelyLoop("ren_sim", inputs=[x, u], outputs=[Output("ren_out", x)])
    sim.inputs["ren_x"].name = "ren_position"

    _, flattened = _pages(sim, tmp_path)
    labels = [node["label"] for node in _graph(flattened)[0]]
    assert labels.count("ren_position") == 1
    assert "ren_x" not in labels


def test_two_inputs_of_one_name_bound_by_a_loop_are_two_nodes(tmp_path):
    # Bound to the body's v, the outer v is another Input that only shares
    # its name: drawn apart, as it would be under any other name.
    s, v = Input("hom_s", tw=1), Input("hom_v", tw=1)
    s.feedback(s + v)
    body = Modely("hom_body", inputs=[s, v], outputs=[Output("hom_o", s * 1.0)])
    outer_v, start = Input("hom_v", seq=4, tw=1), Input("hom_s0", tw=1)
    rolled = Loop(f=body, init={s: start})({v: outer_v})
    model = Modely(
        "hom_model", inputs=[start, outer_v], outputs=[Output("hom_y", rolled)]
    )

    _, flattened = _pages(model, tmp_path)
    labels = [node["label"] for node in _graph(flattened)[0]]
    assert labels.count("hom_v") == 2
    _assert_no_dangling(flattened)


def test_blocks_named_alike_but_for_case_get_a_page_each(tmp_path):
    x = Input("case_x", tw=1)
    upper = Modely("Case", inputs=[x], outputs=[Output("case_a", x * 2.0)])
    lower = Modely("case", inputs=[x], outputs=[Output("case_b", x * 3.0)])
    u = Input("case_u", tw=1)
    model = Modely(
        "case_model", inputs=[u], outputs=[Output("p", upper([u]) + lower([u]))]
    )

    _pages(model, tmp_path)
    pages = sorted(p.name for p in tmp_path.glob("case_model__case*.html"))
    assert pages == ["case_model__case.html", "case_model__case_2.html"]
