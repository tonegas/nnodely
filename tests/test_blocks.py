from nnodely import (
    Input,
    Output,
    Modely,
    ReLU,
    Linear,
    EquationLearner,
    Sin,
    Cos,
    Fir,
    Select,
    Tanh,
    IntegrateStep,
    Integrate,
    Loop,
    DataLoader,
    Parameter,
)

from nnodely.core.layer import Add, Identity
from nnodely.layers.fir import FirImpl
from nnodely.layers.localmodel import LocalModel
from nnodely.layers.fuzzify import Fuzzify
import numpy as np
from conftest import to_numpy

import pytest


def test_fuzzify():
    x = Input("x", dim=1)
    fuzzy_rectangular = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")
    fuzzy_triangular = Fuzzify(centers=[0.0, 0.5, 1.0], function="triangular")
    x_out_rectangular = Output("x_pred_rectangular", fuzzy_rectangular([x.sw(1)]))
    x_out_triangular = Output("x_pred_triangular", fuzzy_triangular([x.sw(1)]))
    model1 = Modely("model1", inputs=[x], outputs=[x_out_rectangular, x_out_triangular])
    model1.build()

    # ------- Model inference -------
    dummy_input_x = np.array([[[-7]], [[0.5]], [[0.8]], [[5.0]]], dtype=np.float32)
    result1 = model1({"x": dummy_input_x})

    assert "x_pred_rectangular" in result1
    assert result1["x_pred_rectangular"].shape == (4, 3, 1)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_rectangular"]),
        np.array(
            [
                [[0.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
                [[0.0], [0.0], [1.0]],
                [[0.0], [0.0], [0.0]],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "x_pred_triangular" in result1
    assert result1["x_pred_triangular"].shape == (4, 3, 1)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_triangular"]),
        np.array(
            [
                [[0.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
                [[0.0], [0.4], [0.6]],
                [[0.0], [0.0], [0.0]],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_model_with_user_functions():
    # ------- Local Model with user functions (one explicit cell per center) ----
    x = Input("x", dim=1)
    k = Input("k", dim=1)

    fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")([k.sw(1)])
    local_model = LocalModel(
        input_function=lambda x: Identity()(x),
        output_function=lambda x: ReLU()(x),
        name="local_model",
    )([x.sw(1)], [fuzzy_k])

    out = Output("out", local_model)
    model = Modely("model_with_local", inputs=[x, k], outputs=[out])
    model.build()

    # ------- Model inference -------
    dummy_input_x = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]], dtype=np.float32)
    dummy_input_k = np.array([[[-7]], [[0.5]], [[0.8]], [[5.0]]], dtype=np.float32)

    result = model({"x": dummy_input_x, "k": dummy_input_k})
    assert "out" in result
    assert result["out"].shape == (4, 1, 1)

    np.testing.assert_allclose(
        to_numpy(result["out"]),
        np.array([[[0.0]], [[2.0]], [[3.0]], [[0.0]]]),
        rtol=1e-5,
        atol=1e-5,
    )


def _cell_weights(model, layer_name, cells):
    """Read the per-cell Fir weights of an explicitly built local model."""
    layers = {layer.name: layer for layer in model.model.layers}
    kernel = np.stack(
        [to_numpy(layers[f"{layer_name}{i}"].kernel) for i in range(cells)]
    )
    bias = np.stack([to_numpy(layers[f"{layer_name}{i}"].bias) for i in range(cells)])
    return kernel, bias


def test_local_model_matches_explicit_cells():
    # ------- The fused matmul must equal one Fir per membership -------
    x = Input("x_local", dim=1)
    k = Input("k_local", dim=1)
    centers = [0.0, 1.0, 2.0]

    activation = Fuzzify(centers=centers, function="Triangular")([k])
    fused = LocalModel(Fir(out_features=2), name="fused_local")([x.sw(4)], [activation])
    fused_model = Modely(
        "fused_local_model", inputs=[x, k], outputs=[Output("fused", fused)]
    ).build()

    cells = [
        Fir(out_features=2, name=f"cell{i}")([x.sw(4)]) * Select(idx=i)([activation])
        for i in range(len(centers))
    ]
    explicit_model = Modely(
        "explicit_local_model",
        inputs=[x, k],
        outputs=[Output("explicit", cells[0] + cells[1] + cells[2])],
    ).build()

    kernel, bias = _cell_weights(explicit_model, "cell", len(centers))
    assert fused.kernel is not None and fused.bias is not None
    fused.kernel.assign(kernel)
    fused.bias.assign(bias)

    dummy_input = {
        "x_local": np.arange(20, dtype=np.float32).reshape(5, 1, 4) / 10.0,
        "k_local": np.linspace(-0.5, 2.5, 5, dtype=np.float32).reshape(5, 1, 1),
    }
    fused_result = fused_model(dict(dummy_input))["fused"]
    explicit_result = explicit_model(dict(dummy_input))["explicit"]

    assert fused_result.shape == (5, 2, 1)
    np.testing.assert_allclose(
        to_numpy(fused_result), to_numpy(explicit_result), rtol=1e-5, atol=1e-5
    )


def _two_fuzzy_inputs(prefix):
    x = Input(f"{prefix}_x", dim=1)
    j = Input(f"{prefix}_j", dim=1)
    k = Input(f"{prefix}_k", dim=1)
    activation_j = Fuzzify(centers=[0.0, 1.0, 2.0, 3.0], function="Triangular")([j])
    activation_k = Fuzzify(centers=[0.0, 1.0, 2.0, 3.0, 4.0], function="Triangular")(
        [k]
    )
    data = {
        f"{prefix}_x": np.arange(30, dtype=np.float32).reshape(6, 1, 5) / 10.0 - 1.0,
        f"{prefix}_j": np.linspace(-0.5, 3.5, 6, dtype=np.float32).reshape(6, 1, 1),
        f"{prefix}_k": np.linspace(0.3, 4.2, 6, dtype=np.float32).reshape(6, 1, 1),
    }
    return x, j, k, activation_j, activation_k, data


def test_local_model_multiplies_fuzzifications_row_major():
    # ------- Two fuzzifications with 4 and 5 centers give 20 cells -------
    x, j, k, activation_j, activation_k, data = _two_fuzzy_inputs("product")

    fused = LocalModel(Fir(out_features=2), name="product_local")(
        [x.sw(5)], [activation_j, activation_k]
    )
    fused_model = Modely(
        "product_local_model", inputs=[x, j, k], outputs=[Output("fused", fused)]
    ).build()
    assert tuple(fused.kernel.shape) == (20, 5, 2)
    assert tuple(fused.bias.shape) == (20, 2)

    # Cell (a, b) is number a * 5 + b and weighs mu_j[a] * mu_k[b].
    cells = [
        Fir(out_features=2, name=f"product_cell{a * 5 + b}")([x.sw(5)])
        * Select(idx=a)([activation_j])
        * Select(idx=b)([activation_k])
        for a in range(4)
        for b in range(5)
    ]
    explicit_model = Modely(
        "product_explicit_model",
        inputs=[x, j, k],
        outputs=[Output("explicit", Add()(cells))],
    ).build()
    kernel, bias = _cell_weights(explicit_model, "product_cell", 20)
    assert fused.kernel is not None and fused.bias is not None
    fused.kernel.assign(kernel)
    fused.bias.assign(bias)

    np.testing.assert_allclose(
        to_numpy(fused_model(dict(data))["fused"]),
        to_numpy(explicit_model(dict(data))["explicit"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_model_generic_and_elementwise_paths_agree():
    # ------- Batched and per-cell evaluation must give the same model -------
    x, j, k, activation_j, activation_k, data = _two_fuzzy_inputs("paths")
    activations = [activation_j, activation_k]

    fused = LocalModel(Fir(out_features=2), output_function=Tanh(), name="p_fused")(
        [x.sw(5)], activations
    )
    stacked = LocalModel(
        lambda s: Fir(out_features=2)(s), output_function=Tanh(), name="p_stacked"
    )([x.sw(5)], activations)
    per_cell = LocalModel(
        Fir(out_features=2), output_function=lambda s: Tanh()(s), name="p_cell"
    )([x.sw(5)], activations)
    model = Modely(
        "paths_model",
        inputs=[x, j, k],
        outputs=[
            Output("fused", fused),
            Output("stacked", stacked),
            Output("per_cell", per_cell),
        ],
    ).build()

    # Give the explicit cells the fused weights.
    assert model.model is not None
    layers = {layer.name: layer for layer in model.model.layers}
    kernel = to_numpy(layers["p_fused_cells"].kernel)
    bias = to_numpy(layers["p_fused_cells"].bias)
    # The stacked cells are the only auto-named Firs, in creation order.
    stacked_names = sorted(
        (
            name
            for name, layer in layers.items()
            if isinstance(layer, FirImpl) and not name.startswith("p_cell")
        ),
        key=lambda name: int(name[3:]),
    )
    assert len(stacked_names) == 20
    for names in (stacked_names, [f"p_cell_in{i}" for i in range(20)]):
        for i, name in enumerate(names):
            layers[name].kernel.assign(kernel[i])
            layers[name].bias.assign(bias[i])

    result = model(dict(data))
    assert result["fused"].shape == (6, 2, 1)
    for name in ("stacked", "per_cell"):
        np.testing.assert_allclose(
            to_numpy(result[name]), to_numpy(result["fused"]), rtol=1e-5, atol=1e-5
        )


def test_local_model_passes_cell_index():
    # ------- A factory builds each cell from its (i_j, i_k) index -------
    x = Input("index_x", dim=1)
    j = Input("index_j", dim=1)
    k = Input("index_k", dim=1)
    activation_j = Fuzzify(centers=[0.0, 1.0], function="Rectangular")([j])
    activation_k = Fuzzify(centers=[0.0, 1.0, 2.0], function="Rectangular")([k])

    seen = []

    def factory(index):
        seen.append(index)
        return lambda s: s[0] * float(10 * index[0] + index[1])

    by_index = LocalModel(factory, pass_index=True, name="index_local")(
        [x.last()], [activation_j, activation_k]
    )
    by_list = LocalModel(
        [lambda s, c=c: s[0] * float(c) for c in (0, 1, 2, 10, 11, 12)],
        name="list_local",
    )([x.last()], [activation_j, activation_k])
    model = Modely(
        "index_model",
        inputs=[x, j, k],
        outputs=[Output("by_index", by_index), Output("by_list", by_list)],
    ).build()
    assert seen == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    # Rectangular memberships pick exactly one cell.
    result = model(
        {
            "index_x": np.full((3, 1, 1), 2.0, dtype=np.float32),
            "index_j": np.array([0.0, 1.0, 1.0], dtype=np.float32).reshape(3, 1, 1),
            "index_k": np.array([2.0, 0.0, 1.0], dtype=np.float32).reshape(3, 1, 1),
        }
    )
    expected = np.array([2.0 * 2, 2.0 * 10, 2.0 * 11]).reshape(3, 1, 1)
    for name in ("by_index", "by_list"):
        np.testing.assert_allclose(to_numpy(result[name]), expected, atol=1e-5)


def test_local_model_scalar_input_one_fuzzy_function_by_hand():
    # ------- y = x * sum_i mu_i(g) * gain_i, centers 0, 1, 2 -------
    x = Input("x_hand")
    g = Input("g_hand")
    mu = Fuzzify(centers=[0.0, 1.0, 2.0], function="Triangular")(g.last())
    y = LocalModel(Fir(out_features=1))([x.last()], [mu])
    model = Modely(
        "local_hand", inputs=[x, g], outputs=[Output("y", y), Output("mu", mu)]
    ).build()

    gains = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    assert y.kernel is not None and y.bias is not None
    y.kernel.assign(gains.reshape(3, 1, 1))
    y.bias.assign(np.zeros((3, 1), dtype=np.float32))

    g_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    result = model(
        {
            "x_hand": np.full((5, 1, 1), 2.0, dtype=np.float32),
            "g_hand": g_values.reshape(-1, 1, 1),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["mu"]).reshape(5, 3),
        [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5], [0, 0, 1]],
        atol=1e-6,
    )
    # e.g. g = 0.5: y = 2 * (0.5 * 1 + 0.5 * 10) = 11
    np.testing.assert_allclose(
        to_numpy(result["y"]).ravel(), [2.0, 11.0, 20.0, 110.0, 200.0], rtol=1e-5
    )


def test_local_model_matrix_input_two_fuzzy_functions_by_hand():
    # ------- 3 x 2 = 6 cells over a 2x2 matrix, cell (i, j) is i * 2 + j -------
    X = Input("X_hand", dim=(2, 2))
    a = Input("a_hand")
    b = Input("b_hand")
    mu_a = Fuzzify(centers=[0.0, 1.0, 2.0])(a.last())
    mu_b = Fuzzify(centers=[0.0, 1.0])(b.last())
    # A Fir filters every entry of X alone, so its kernel holds one tap per
    # entry. Cell c weighs every entry by c + 1: its local model is (c + 1) * X.
    cells = [
        Fir(
            out_features=1,
            kernel=Parameter(f"hand_cell{c}", value=np.full((2, 2), c + 1.0)),
            bias=False,
        )
        for c in range(6)
    ]
    Y = LocalModel(cells, name="local_matrix_hand")([X.last()], [mu_a, mu_b])
    model = Modely(
        "local_matrix_hand",
        inputs=[X, a, b],
        outputs=[Output("Y", Y), Output("mu_a", mu_a), Output("mu_b", mu_b)],
    ).build()
    assert Y.dim == (2, 2)

    result = model(
        {
            "X_hand": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).reshape(
                1, 2, 2, 1
            ),
            "a_hand": np.array([[[0.5]]], dtype=np.float32),
            "b_hand": np.array([[[0.25]]], dtype=np.float32),
        }
    )
    membership_a = to_numpy(result["mu_a"]).ravel()
    membership_b = to_numpy(result["mu_b"]).ravel()
    np.testing.assert_allclose(membership_a, [0.5, 0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(membership_b, [0.75, 0.25], atol=1e-6)
    np.testing.assert_allclose(
        np.outer(membership_a, membership_b).ravel(),
        [0.375, 0.125, 0.375, 0.125, 0.0, 0.0],
        atol=1e-6,
    )
    # Y = (0.375*1 + 0.125*2 + 0.375*3 + 0.125*4) * X = 2.25 * X
    np.testing.assert_allclose(
        to_numpy(result["Y"]).ravel(), [2.25, 4.5, 6.75, 9.0], rtol=1e-5
    )


def test_local_model_cells_share_a_fir_given_its_kernel_as_a_parameter():
    # Each cell copies the Fir, Parameter included: every cell filters with
    # the same kernel, and memberships summing to one give that Fir back.
    x = Input("x_shared_w")
    g = Input("g_shared_w")
    w = Parameter("shared_w", value=[[1.0], [2.0], [3.0]])
    mu = Fuzzify(centers=[0.0, 1.0, 2.0], function="Triangular")(g.last())
    blended = LocalModel(Fir(out_features=1, kernel=w, bias=False))([x.sw(3)], [mu])
    model = Modely(
        "local_shared_w", inputs=[x, g], outputs=[Output("blended", blended)]
    ).build()

    result = model(
        {
            "x_shared_w": np.array([[[1.0, 1.0, 2.0]]], dtype=np.float32),
            "g_shared_w": np.array([[[0.7]]], dtype=np.float32),
        }
    )
    # 1 * 1 + 2 * 1 + 3 * 2 = 9
    np.testing.assert_allclose(to_numpy(result["blended"]).ravel(), [9.0], rtol=1e-5)


def test_local_model_rejects_mismatched_inputs():
    x = Input("x_invalid", dim=1)
    k = Input("k_invalid", dim=1)
    activation = Fuzzify(centers=[0.0, 1.0], function="Triangular")([k.sw(1)])

    with pytest.raises(ValueError, match="got 3 input functions for 2 cells"):
        LocalModel([Fir(out_features=1)] * 3, name="unpaired_local")(
            [x.sw(2)], [activation]
        )

    with pytest.raises(ValueError, match="same shape"):
        LocalModel([Fir(out_features=1), Fir(out_features=2)], name="uneven_local")(
            [x.sw(2)], [activation]
        )

    with pytest.raises(ValueError, match="every output function must return"):
        LocalModel(
            output_function=[lambda s: s[0], lambda s: Fir(out_features=2)(s)],
            name="uneven_output_local",
        )([x.sw(2)], [activation])

    windowed_activation = Fuzzify(centers=[0.0, 1.0], function="Triangular")([k.sw(2)])
    with pytest.raises(ValueError, match="activation 0 must have shape"):
        LocalModel(name="windowed_local")([x.sw(2)], [windowed_activation])


def test_equation_learner_composes_symbolic_functions_and_multiple_inputs():
    x = Input("equation_x")
    y = Input("equation_y")
    equation = EquationLearner(
        functions=["identity", (lambda left, right: left * right, 2), Sin],
        linear_in=Linear(
            out_features=4,
            kernel=Parameter(
                "equation_in_matrix", value=[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
            ),
            bias=False,
            name="equation_linear_in",
        ),
        linear_out=Linear(
            out_features=1,
            kernel=Parameter("equation_out_weights", value=[[2.0], [3.0], [4.0]]),
            bias=False,
            name="equation_linear_out",
        ),
        name="equation",
    )
    learned = equation([x.last(), y.last()])
    model = Modely(
        "equation_model",
        inputs=[x, y],
        outputs=[Output("result", learned)],
    ).build()

    inputs = {
        "equation_x": np.array([[[2.0]]], dtype=np.float32),
        "equation_y": np.array([[[0.5]]], dtype=np.float32),
    }
    result = to_numpy(model(inputs)["result"])
    expected = 2.0 * 2.0 + 3.0 * 2.0 * 0.5 + 4.0 * np.sin(0.5)

    assert result.shape == (1, 1, 1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert equation.model is not None
    internal_types = {type(node).__name__ for node in equation.model.order}
    assert {"Linear", "Select", "Multiply", "Sin", "Concatenate"} <= internal_types


def test_equation_learner_supports_layer_classes_and_basis_output():
    x = Input("basis_x")
    equation = EquationLearner(
        functions=[Sin, Cos, "add"],
        linear_in=Linear(out_features=4, kernel="ones", bias=False),
        name="basis_equation",
    )
    basis = equation(x.last())
    model = Modely(
        "basis_model",
        inputs=[x],
        outputs=[Output("basis", basis)],
    ).build()

    value = np.array([[[0.25]]], dtype=np.float32)
    result = to_numpy(model({"basis_x": value})["basis"])
    expected = np.array(
        [[[np.sin(0.25)], [np.cos(0.25)], [0.5]]],
        dtype=np.float32,
    ).reshape((1, 3, 1))

    assert basis.dim == (3,)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_equation_learner_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="at least one function"):
        EquationLearner([])

    with pytest.raises(ValueError, match="total number of function arguments"):
        EquationLearner(
            [Sin, "add"],
            linear_in=Linear(out_features=2),
        )

    with pytest.raises(ValueError, match="Unknown EquationLearner function"):
        EquationLearner(["not_a_function"])


def test_integrate():
    # make a simple forward Euler integrator step
    x = Input("x")
    x0 = Input("x0")
    integrate = IntegrateStep(solver="euler", dt=0.1, init=x0)(x)
    out = Output("x_hat", integrate)
    model = Modely(name="integrator", inputs=[x, x0], outputs=[out])
    model.build()

    # make a loop that rolls out the integrator along a sequence of rates
    x_seq = Input("x_seq", seq=-1)
    x0_loop = Input("x0_loop")
    loop = Loop(f=model, callback={x0: out}, init={x0: x0_loop})({x: x_seq})
    out_loop = Output("x_hat_loop", loop)
    model_loop = Modely(
        name="integrator_loop", inputs=[x_seq, x0_loop], outputs=[out_loop]
    ).build()

    dummy = np.array([1.0, 2.0, 3.0]).reshape(1, 1, 1, 3)
    start = np.array(0.0).reshape(1, 1, 1)
    by_hand = to_numpy(model_loop({"x_seq": dummy, "x0_loop": start})["x_hat_loop"])
    np.testing.assert_allclose(by_hand, [[[[0.1, 0.3, 0.6]]]], rtol=1e-6)

    # The block builds the same loop: the horizon is the rate's sequence axis.
    y = Input("y", seq=-1)
    y0 = Input("y0")
    new_integrate = Integrate(solver="euler", dt=0.1, init=y0)(y)
    out_int = Output("y_hat", new_integrate)
    new_model = Modely(name="new_integrator", inputs=[y, y0], outputs=[out_int]).build()

    result = to_numpy(new_model({"y": dummy, "y0": start})["y_hat"])
    assert result.shape == (1, 1, 1, 3)
    np.testing.assert_allclose(result, by_hand, rtol=1e-6)


# ---------------------------------------------------------------------------
# Integrate: IntegrateStep rolled out by a Loop along the rate's horizon
# ---------------------------------------------------------------------------


def _integrate_model(name, seq: int | tuple[int, ...] = -1, dim=1, **kwargs):
    rate = Input(f"{name}_rate", dim=dim, seq=seq)
    out = Output(f"{name}_out", Integrate(**kwargs)(rate))
    return Modely(name, inputs=[rate], outputs=[out]).build()


@pytest.mark.parametrize("solver", ["euler", "rectangular", "trapezoidal"])
def test_integrate_matches_integrate_step_over_the_same_samples(solver):
    # Along a horizon or along a time window, the same rule on the same samples.
    samples = np.array([1.0, -2.0, 4.0, 0.5, 3.0], dtype=np.float32)
    horizon = _integrate_model(f"along_{solver}", solver=solver, dt=0.1, init=0.7)
    w = Input(f"window_{solver}")
    window = Modely(
        f"window_{solver}",
        inputs=[w],
        outputs=[
            Output(
                f"window_{solver}_out",
                IntegrateStep(solver=solver, dt=0.1, init=0.7)(w.sw(5)),
            )
        ],
    ).build()

    along = to_numpy(
        horizon({f"along_{solver}_rate": samples.reshape(1, 1, 1, 5)})[
            f"along_{solver}_out"
        ]
    )
    over = to_numpy(
        window({f"window_{solver}": samples.reshape(1, 1, 5)})[f"window_{solver}_out"]
    )
    np.testing.assert_allclose(along.ravel(), over.ravel(), rtol=1e-5, atol=1e-6)


def test_integrate_follows_a_dynamic_horizon_without_rebuilding():
    model = _integrate_model("dynamic", solver="euler", dt=0.5)
    for steps in (1, 4, 9):
        rate = np.ones((2, 1, 1, steps), dtype=np.float32)
        result = to_numpy(model({"dynamic_rate": rate})["dynamic_out"])
        assert result.shape == (2, 1, 1, steps)
        np.testing.assert_allclose(result[0].ravel(), 0.5 * np.arange(1, steps + 1))


def test_integrate_over_a_fixed_horizon():
    model = _integrate_model("fixed", seq=4, solver="euler", dt=1.0)
    result = to_numpy(
        model({"fixed_rate": np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 1, 4)})[
            "fixed_out"
        ]
    )
    np.testing.assert_allclose(result.ravel(), [1.0, 3.0, 6.0, 10.0])


def test_integrate_starts_from_none_a_number_or_a_stream():
    rate = np.full((1, 1, 1, 3), 2.0, dtype=np.float32)
    zero = _integrate_model("from_zero", dt=0.1)
    number = _integrate_model("from_number", dt=0.1, init=5.0)
    r, r0 = Input("from_stream_rate", seq=-1), Input("from_stream_init")
    stream = Modely(
        "from_stream",
        inputs=[r, r0],
        outputs=[Output("from_stream_out", Integrate(dt=0.1, init=r0)(r))],
    ).build()

    np.testing.assert_allclose(
        to_numpy(zero({"from_zero_rate": rate})["from_zero_out"]).ravel(),
        [0.2, 0.4, 0.6],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(number({"from_number_rate": rate})["from_number_out"]).ravel(),
        [5.2, 5.4, 5.6],
        rtol=1e-6,
    )
    # One initial value per sample of the batch.
    result = to_numpy(
        stream(
            {
                "from_stream_rate": np.concatenate([rate, rate]),
                "from_stream_init": np.array([1.0, -1.0]).reshape(2, 1, 1),
            }
        )["from_stream_out"]
    )
    np.testing.assert_allclose(result[0].ravel(), [1.2, 1.4, 1.6], rtol=1e-6)
    np.testing.assert_allclose(result[1].ravel(), [-0.8, -0.6, -0.4], rtol=1e-6)


def test_integrate_each_feature_of_a_vector_rate():
    rate, start = Input("vector_rate", dim=2, seq=-1), Input("vector_init", dim=2)
    scalar_start = Input("vector_scalar_init")
    model = Modely(
        "vector",
        inputs=[rate, start, scalar_start],
        outputs=[
            Output("vector_out", Integrate(dt=1.0, init=start)(rate)),
            Output("vector_scalar_out", Integrate(dt=1.0, init=scalar_start)(rate)),
        ],
    ).build()
    values = np.array([[1.0, 1.0, 1.0], [10.0, 20.0, 30.0]], dtype=np.float32)

    result = model(
        {
            "vector_rate": values.reshape(1, 2, 1, 3),
            "vector_init": np.array([0.0, 100.0]).reshape(1, 2, 1),
            "vector_scalar_init": np.array([1.0]).reshape(1, 1, 1),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["vector_out"]).reshape(2, 3), [[1, 2, 3], [110, 130, 160]]
    )
    # A scalar initial value starts every feature.
    np.testing.assert_allclose(
        to_numpy(result["vector_scalar_out"]).reshape(2, 3), [[2, 3, 4], [11, 31, 61]]
    )


def test_integrate_a_rate_with_an_extra_sequence_axis():
    # The horizon is the last sequence axis; the others are integrated apart.
    model = _integrate_model("extra_seq", seq=(2, -1), dt=1.0)
    rate = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 2.0]], dtype=np.float32)
    result = to_numpy(
        model({"extra_seq_rate": rate.reshape(1, 1, 1, 2, 3)})["extra_seq_out"]
    )
    assert result.shape == (1, 1, 1, 2, 3)
    np.testing.assert_allclose(result.reshape(2, 3), [[1, 2, 3], [2, 2, 4]])


def test_integrate_twice_from_acceleration_to_position():
    # Constant acceleration: the trapezoid of a linear velocity is exact.
    dt, steps, a, v0, x0 = 0.1, 20, 2.0, 1.0, 3.0
    acc = Input("acc", seq=-1)
    vel = Integrate(solver="trapezoidal", dt=dt, init=v0)(acc)
    pos = Integrate(solver="trapezoidal", dt=dt, init=x0)(vel)
    model = Modely(
        "kinematics", inputs=[acc], outputs=[Output("vel", vel), Output("pos", pos)]
    ).build()

    result = model({"acc": np.full((1, 1, 1, steps), a, dtype=np.float32)})
    t = dt * np.arange(1, steps + 1)
    np.testing.assert_allclose(to_numpy(result["vel"]).ravel(), v0 + a * t, rtol=1e-5)
    # The first velocity step is a rectangle, so the position trails the exact
    # parabola by the area that rectangle adds: dt/2 * (v(dt) - v0) = a*dt^2/2.
    expected = x0 + v0 * t + a * t**2 / 2 + a * dt**2 / 2
    np.testing.assert_allclose(to_numpy(result["pos"]).ravel(), expected, rtol=1e-5)


def test_integrate_is_trained_through():
    # x = Integrate(k * u): the gain of the rate is learnt from the trajectory.
    # (A fixed horizon: arithmetic on a dynamic sequence axis cannot be declared.)
    steps = 8
    u = Input("trained_u", seq=steps)
    k = Parameter("trained_k", value=0.5)
    x = Output("trained_x", Integrate(dt=0.1)(u * k))
    model = Modely("trained", inputs=[u], outputs=[x])
    model.minimize("fit", x, Input("trained_target", seq=steps))
    model.build()
    rng = np.random.default_rng(0)
    rates = rng.uniform(-1.0, 1.0, (64, steps)).astype(np.float32)
    targets = 0.1 * np.cumsum(2.0 * rates, axis=1)
    data = DataLoader(
        model,
        source=[
            {"trained_u": rates[i], "trained_target": targets[i]}
            for i in range(len(rates))
        ],
    )
    assert len(data) == len(rates)  # one trajectory per simulation

    model.train(data, epochs=200, batch_size=16, lr=0.05, printer=None)

    np.testing.assert_allclose(to_numpy(k.param).ravel(), [2.0], atol=1e-2)


def test_integrate_round_trips_through_save_and_the_keras_file(tmp_path):
    import keras

    rate, start = Input("saved_rate", seq=-1), Input("saved_init")
    model = Modely(
        "saved_integrator",
        inputs=[rate, start],
        outputs=[
            Output(
                "saved_out", Integrate(solver="trapezoidal", dt=0.2, init=start)(rate)
            )
        ],
    ).build()
    data = {
        "saved_rate": np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float32).reshape(
            1, 1, 1, 4
        ),
        "saved_init": np.array([0.5], dtype=np.float32).reshape(1, 1, 1),
    }
    expected = to_numpy(model(data)["saved_out"])

    model.save(tmp_path / "saved")
    restored = Modely.load(tmp_path / "saved")
    np.testing.assert_allclose(
        to_numpy(restored(data)["saved_out"]), expected, rtol=1e-6
    )

    model.export_keras(tmp_path)
    exported = Modely.import_keras(str(tmp_path / "saved_integrator.keras"))
    assert exported is not None
    np.testing.assert_allclose(
        to_numpy(exported(data)["saved_out"]),  # type: ignore
        expected,
        rtol=1e-6,
    )
    assert isinstance(exported, keras.Model)


def test_integrate_returns_the_loop_for_further_relations():
    # The result is the Loop itself, a stream to compute with like any other.
    acc = Input("further_acc", seq=5)
    vel = Integrate(dt=0.1)(acc)
    assert isinstance(vel, Loop)
    gain = Parameter("further_gain", value=3.0)
    model = Modely(
        "further",
        inputs=[acc],
        outputs=[Output("further_out", Sin()([vel]) * gain + Integrate(dt=0.1)(vel))],
    ).build()

    result = to_numpy(
        model({"further_acc": np.ones((1, 1, 1, 5), dtype=np.float32)})["further_out"]
    )
    v = 0.1 * np.arange(1, 6)
    np.testing.assert_allclose(
        result.ravel(), 3.0 * np.sin(v) + 0.1 * np.cumsum(v), rtol=1e-5
    )


def test_a_dynamic_integrate_feeds_any_relation():
    # On a dynamic horizon the result is still a stream to compute with.
    acc = Input("dynamic_further_acc", seq=-1)
    vel = Integrate(dt=0.1)(acc)
    assert vel.seq == (None,)
    gain = Parameter("dynamic_further_gain", value=3.0)
    model = Modely(
        "dynamic_further",
        inputs=[acc],
        outputs=[Output("dynamic_further_out", Sin()([vel]) * gain + vel * 2.0)],
    ).build()

    for steps in (3, 7):
        result = to_numpy(
            model({"dynamic_further_acc": np.ones((2, 1, 1, steps), dtype=np.float32)})[
                "dynamic_further_out"
            ]
        )
        v = 0.1 * np.arange(1, steps + 1)
        assert result.shape == (2, 1, 1, steps)
        np.testing.assert_allclose(
            result[1].ravel(), 3.0 * np.sin(v) + 2.0 * v, rtol=1e-5
        )


def test_a_dynamic_integrate_trains_on_simulations_of_different_lengths():
    # Every simulation is one trajectory of its own length; the loader pads the
    # shorter ones, and the padded steps are left out of the loss.
    u = Input("dynamic_trained_u", seq=-1)
    k = Parameter("dynamic_trained_k", value=0.5)
    x = Output("dynamic_trained_x", Integrate(dt=0.1)(u * k))
    model = Modely("dynamic_trained", inputs=[u], outputs=[x])
    model.minimize("fit", x, Input("dynamic_trained_target", seq=-1))
    model.build()
    rng = np.random.default_rng(1)
    simulations = []
    for length in (4, 9, 6, 12) * 8:
        rates = rng.uniform(-1.0, 1.0, length).astype(np.float32)
        simulations.append(
            {
                "dynamic_trained_u": rates,
                "dynamic_trained_target": 0.1 * np.cumsum(2.0 * rates),
            }
        )
    data = DataLoader(model, source=simulations, seq_length="full")
    assert len(data) == len(simulations)

    model.train(data, epochs=200, batch_size=8, lr=0.05, printer=None)

    np.testing.assert_allclose(to_numpy(k.param).ravel(), [2.0], atol=1e-2)


def test_integrate_rejects_what_it_cannot_integrate():
    with pytest.raises(ValueError, match="has none"):
        Integrate(dt=0.1)(Input("no_seq"))
    with pytest.raises(ValueError, match="one rate sample per step"):
        Integrate(dt=0.1)(Input("windowed", seq=-1).sw(3))
    with pytest.raises(ValueError, match="solver must be one of"):
        Integrate(solver="rk4", dt=0.1)
    with pytest.raises(ValueError, match="dt"):
        Integrate()
    with pytest.raises(TypeError, match="configured first"):
        Integrate(Input("misused", seq=-1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integrates one rate"):
        Integrate(dt=0.1)([Input("first", seq=-1), Input("second", seq=-1)])
    rate = Input("checked_rate", dim=2, seq=-1)
    with pytest.raises(ValueError, match="single sample"):
        Integrate(dt=0.1, init=Input("long_init", dim=2).sw(2))(rate)
    with pytest.raises(ValueError, match="init has dim"):
        Integrate(dt=0.1, init=Input("wide_init", dim=3))(rate)
    with pytest.raises(ValueError, match="init has seq"):
        Integrate(dt=0.1, init=Input("seq_init", dim=2, seq=4))(rate)
