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

from nnodely.core.layer import Add
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
    # ------- No input function: the input itself, weighted cell by cell ----
    x = Input("x", dim=1)
    k = Input("k", dim=1)

    fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")([k.sw(1)])
    local_model = LocalModel(output_function=ReLU(), name="local_model")(
        [x.sw(1)], [fuzzy_k]
    )
 
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


def _fir_factory(kernel, bias, sizes):
    """A factory of Firs whose weights are the Parameters ``kernel[c]`` and
    ``bias[c]``, ``c`` the row-major number of the cell."""

    def factory(index):
        cell = int(np.ravel_multi_index(index, sizes))
        return Fir(
            features=kernel.shape[-1],
            kernel=Parameter(value=kernel[cell]),
            bias=Parameter(value=bias[cell]),
        )

    return factory


def test_local_model_matches_explicit_cells():
    # ------- One Fir per membership, with the weights given as Parameters -------
    x = Input("x_local", dim=1)
    k = Input("k_local", dim=1)
    centers = [0.0, 1.0, 2.0]
    rng = np.random.default_rng(0)
    kernel = rng.normal(size=(3, 4, 2)).astype(np.float32)  # (cell, time, features)
    bias = rng.normal(size=(3, 2)).astype(np.float32)

    activation = Fuzzify(centers=centers, function="Triangular")([k])
    local = LocalModel(_fir_factory(kernel, bias, (3,)), name="cells_local")(
        [x.sw(4)], [activation]
    )
    cells = [
        Fir(
            features=2,
            kernel=Parameter(value=kernel[i]),
            bias=Parameter(value=bias[i]),
        )([x.sw(4)])
        * Select(idx=i)([activation])
        for i in range(len(centers))
    ]
    model = Modely(
        "cells_local_model",
        inputs=[x, k],
        outputs=[Output("local", local), Output("explicit", Add()(cells))],
    ).build()

    result = model(
        {
            "x_local": np.arange(20, dtype=np.float32).reshape(5, 1, 4) / 10.0,
            "k_local": np.linspace(-0.5, 2.5, 5, dtype=np.float32).reshape(5, 1, 1),
        }
    )
    assert result["local"].shape == (5, 2, 1)
    np.testing.assert_allclose(
        to_numpy(result["local"]), to_numpy(result["explicit"]), rtol=1e-5, atol=1e-5
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
    rng = np.random.default_rng(1)
    kernel = rng.normal(size=(20, 5, 2)).astype(np.float32)
    bias = rng.normal(size=(20, 2)).astype(np.float32)

    local = LocalModel(_fir_factory(kernel, bias, (4, 5)), name="product_local")(
        [x.sw(5)], [activation_j, activation_k]
    )
    # Cell (a, b) is number a * 5 + b and weighs mu_j[a] * mu_k[b].
    cells = [
        Fir(
            features=2,
            kernel=Parameter(value=kernel[a * 5 + b]),
            bias=Parameter(value=bias[a * 5 + b]),
        )([x.sw(5)])
        * Select(idx=a)([activation_j])
        * Select(idx=b)([activation_k])
        for a in range(4)
        for b in range(5)
    ]
    model = Modely(
        "product_local_model",
        inputs=[x, j, k],
        outputs=[Output("local", local), Output("explicit", Add()(cells))],
    ).build()

    result = model(dict(data))
    np.testing.assert_allclose(
        to_numpy(result["local"]), to_numpy(result["explicit"]), rtol=1e-5, atol=1e-5
    )


def test_local_model_per_cell_once_and_vectorized_paths_agree():
    # ------- Per-cell factories and one vectorized call: one model -------
    x, j, k, activation_j, activation_k, data = _two_fuzzy_inputs("paths")
    activations = [activation_j, activation_k]
    rng = np.random.default_rng(2)
    kernel = rng.normal(size=(20, 5, 1)).astype(np.float32)
    bias = rng.normal(size=(20, 1)).astype(np.float32)

    once = LocalModel(
        _fir_factory(kernel, bias, (4, 5)), output_function=Tanh(), name="p_once"
    )([x.sw(5)], activations)
    per_cell = LocalModel(
        _fir_factory(kernel, bias, (4, 5)),
        output_function=lambda index: Tanh(),
        name="p_per_cell",
    )([x.sw(5)], activations)
    # One Fir with a feature per cell: column c is the kernel of cell c.
    vectorized = LocalModel(
        lambda streams: Fir(
            features=20,
            kernel=Parameter(value=kernel[:, :, 0].T),
            bias=Parameter(value=bias[:, 0]),
        )(streams),
        output_function=lambda streams: Tanh()(streams),
        vectorized=True,
        name="p_vectorized",
    )([x.sw(5)], activations)
    # The same Fir given as a layer: with vectorized=True it is called once too.
    vectorized_layer = LocalModel(
        Fir(
            features=20,
            kernel=Parameter(value=kernel[:, :, 0].T),
            bias=Parameter(value=bias[:, 0]),
        ),
        output_function=Tanh(),
        vectorized=True,
        name="p_vectorized_layer",
    )([x.sw(5)], activations)
    model = Modely(
        "paths_model",
        inputs=[x, j, k],
        outputs=[
            Output("once", once),
            Output("per_cell", per_cell),
            Output("vectorized", vectorized),
            Output("vectorized_layer", vectorized_layer),
        ],
    ).build()

    result = model(dict(data))
    assert result["once"].shape == (6, 1, 1)
    for name in ("per_cell", "vectorized", "vectorized_layer"):
        np.testing.assert_allclose(
            to_numpy(result[name]), to_numpy(result["once"]), rtol=1e-5, atol=1e-5
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

    local = LocalModel(factory, name="index_local")
    first = local([x.last()], [activation_j, activation_k])
    # A second call applies the cells of the first: the factory is not called again.
    second = local([x.last()], [activation_j, activation_k])
    model = Modely(
        "index_model",
        inputs=[x, j, k],
        outputs=[Output("first", first), Output("second", second)],
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
    for name in ("first", "second"):
        np.testing.assert_allclose(to_numpy(result[name]), expected, atol=1e-5)


def test_local_model_repeated_call_shares_weights():
    x = Input("shared_local_x")
    g = Input("shared_local_g")
    mu = Fuzzify(centers=[0.0, 1.0])(g.last())
    local = LocalModel(lambda index: Fir(features=1), name="shared_local")
    model = Modely(
        "shared_local_model",
        inputs=[x, g],
        outputs=[
            Output("first", local([x.sw(3)], [mu])),
            Output("second", local([x.sw(3)], [mu])),
        ],
    ).build()

    assert model.model is not None
    assert sum(isinstance(layer, FirImpl) for layer in model.model.layers) == 2
    result = model(
        {
            "shared_local_x": np.random.rand(4, 1, 3).astype(np.float32),
            "shared_local_g": np.random.rand(4, 1, 1).astype(np.float32),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["first"]), to_numpy(result["second"]), atol=1e-6
    )


def test_local_model_scalar_input_one_fuzzy_function_by_hand():
    # ------- y = x * sum_i mu_i(g) * gain_i, centers 0, 1, 2 -------
    x = Input("x_hand")
    g = Input("g_hand")
    mu = Fuzzify(centers=[0.0, 1.0, 2.0], function="Triangular")(g.last())
    gains = [1.0, 10.0, 100.0]
    y = LocalModel(
        lambda index: Fir(
            features=1, kernel=Parameter(value=[[gains[index[0]]]]), bias=False
        ),
        name="hand_local",
    )([x.last()], [mu])
    model = Modely(
        "local_hand", inputs=[x, g], outputs=[Output("y", y), Output("mu", mu)]
    ).build()

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
    # Cell c filters every entry of X with the gain c + 1, entry by entry.
    Y = LocalModel(
        lambda index: Fir(
            features=1,
            kernel=Parameter(value=np.full((2, 2, 1, 1), index[0] * 2 + index[1] + 1)),
            bias=False,
        ),
        name="matrix_hand_local",
    )([X.last()], [mu_a, mu_b])
    model = Modely(
        "local_matrix_hand",
        inputs=[X, a, b],
        outputs=[Output("Y", Y), Output("mu_a", mu_a), Output("mu_b", mu_b)],
    ).build()

    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = model(
        {
            "X_hand": values.reshape(1, 2, 2, 1),
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
    assert result["Y"].shape == (1, 2, 2, 1)
    np.testing.assert_allclose(
        to_numpy(result["Y"]).reshape(2, 2), 2.25 * values, rtol=1e-5
    )


def test_local_model_matrix_input_without_input_function_by_hand():
    # ------- Y = sum_c sin(mu_c * X), the output applied to the (6, 2, 2) stack ----
    X = Input("X_plain", dim=(2, 2))
    a = Input("a_plain")
    b = Input("b_plain")
    mu_a = Fuzzify(centers=[0.0, 1.0, 2.0])(a.last())
    mu_b = Fuzzify(centers=[0.0, 1.0])(b.last())
    weighted = LocalModel()([X.sw(3)], [mu_a, mu_b])
    vectorized = LocalModel(
        output_function=lambda streams: Sin()(streams), vectorized=True
    )([X.sw(3)], [mu_a, mu_b])
    model = Modely(
        "local_matrix_plain",
        inputs=[X, a, b],
        outputs=[Output("weighted", weighted), Output("vectorized", vectorized)],
    ).build()

    values = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) / 4.0
    result = model(
        {
            "X_plain": values,
            "a_plain": np.array([[[0.5]]], dtype=np.float32),
            "b_plain": np.array([[[0.25]]], dtype=np.float32),
        }
    )
    mu = np.array([0.375, 0.125, 0.375, 0.125, 0.0, 0.0], dtype=np.float32)
    assert result["vectorized"].shape == (1, 2, 2, 3)
    # The memberships sum to one, so weighting alone gives the input back.
    np.testing.assert_allclose(to_numpy(result["weighted"]), values, rtol=1e-5)
    np.testing.assert_allclose(
        to_numpy(result["vectorized"]),
        np.sin(mu.reshape(6, 1, 1, 1, 1) * values).sum(axis=0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_local_model_memberships_along_the_window_by_hand():
    # ------- y[t] = x[t] * sum_(a, b) mu_a[a, t] * mu_b[b] * (a * 2 + b + 1) -------
    x = Input("x_window")
    a = Input("a_window")
    b = Input("b_window")
    mu_a = Fuzzify(centers=[0.0, 1.0, 2.0])(a.sw(3))  # one membership per sample
    mu_b = Fuzzify(centers=[0.0, 1.0])(b.last())  # the same for every sample
    y = LocalModel(
        lambda index: lambda streams: streams[0] * float(index[0] * 2 + index[1] + 1),
        name="window_local",
    )([x.sw(3)], [mu_a, mu_b])
    model = Modely(
        "local_window",
        inputs=[x, a, b],
        outputs=[Output("y", y), Output("mu_a", mu_a), Output("mu_b", mu_b)],
    ).build()

    x_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = model(
        {
            "x_window": x_values.reshape(1, 1, 3),
            "a_window": np.array([0.0, 0.5, 2.0], dtype=np.float32).reshape(1, 1, 3),
            "b_window": np.array([[[0.25]]], dtype=np.float32),
        }
    )
    membership_a = to_numpy(result["mu_a"]).reshape(3, 3)  # (center, sample)
    membership_b = to_numpy(result["mu_b"]).ravel()
    gains = np.arange(1.0, 7.0).reshape(3, 2)
    expected = x_values * np.einsum("at,b,ab->t", membership_a, membership_b, gains)
    assert result["y"].shape == (1, 1, 3)
    np.testing.assert_allclose(to_numpy(result["y"]).ravel(), expected, rtol=1e-5)


def test_local_model_rejects_mismatched_inputs():
    x = Input("x_invalid", dim=1)
    k = Input("k_invalid", dim=1)
    activation = Fuzzify(centers=[0.0, 1.0], function="Triangular")([k.sw(1)])

    with pytest.raises(TypeError, match=r"pass a factory f\(index\)"):
        LocalModel([Fir(features=1)] * 2, name="list_local")

    with pytest.raises(ValueError, match="every cell must have the same shape"):
        LocalModel(lambda index: Fir(features=index[0] + 1), name="uneven_local")(
            [x.sw(2)], [activation]
        )

    with pytest.raises(ValueError, match="every cell must have the same shape"):
        LocalModel(
            output_function=lambda index: Fir(features=index[0] + 1),
            name="uneven_output_local",
        )([x.sw(2)], [activation])

    with pytest.raises(ValueError, match="one entry per cell"):
        LocalModel(
            lambda streams: Fir(features=3)(streams),
            vectorized=True,
            name="vectorized_local",
        )([x.sw(2)], [activation])

    with pytest.raises(ValueError, match="there must be one"):
        LocalModel(name="two_inputs_local")([x.sw(2), k.sw(2)], [activation])

    windowed_activation = Fuzzify(centers=[0.0, 1.0], function="Triangular")([k.sw(3)])
    with pytest.raises(ValueError, match="must match, or be 1"):
        LocalModel(name="windowed_local")([x.sw(2)], [windowed_activation])

    sequence = Input("k_invalid_seq", dim=1, seq=-1)
    sequence_activation = Fuzzify(centers=[0.0, 1.0])([sequence.last()])
    with pytest.raises(ValueError, match="activation 0 must have shape"):
        LocalModel(name="sequence_local")([x.sw(2)], [sequence_activation])


def test_equation_learner_composes_symbolic_functions_and_multiple_inputs():
    x = Input("equation_x")
    y = Input("equation_y")
    equation = EquationLearner(
        functions=["identity", (lambda left, right: left * right, 2), Sin],
        linear_in=Linear(
            features=4,
            kernel=Parameter(
                "equation_in_matrix", value=[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
            ),
            bias=False,
            name="equation_linear_in",
        ),
        linear_out=Linear(
            features=1,
            kernel=Parameter("equation_out_weights", value=[2.0, 3.0, 4.0]),
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
        linear_in=Linear(features=4, kernel="ones", bias=False),
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
            linear_in=Linear(features=2),
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


@pytest.mark.parametrize("layer", [Fir, Linear])
def test_integrate_a_rate_from_a_layer_with_parameter_weights(layer):
    # rate = 2u + 1 on [1, 2, 3, 4] is [3, 5, 7, 9]; Euler with dt = 1 sums it.
    name = f"prate_{layer.__name__.lower()}"
    u = Input(f"{name}_u", seq=4)
    gain = Parameter(f"{name}_gain", value=[2.0])
    offset = Parameter(f"{name}_offset", value=[1.0])
    rate = layer(features=1, kernel=gain, bias=offset)(u.last())
    model = Modely(
        name,
        inputs=[u],
        outputs=[Output(f"{name}_out", Integrate(solver="euler", dt=1.0)(rate))],
    ).build()

    values = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 1, 4)
    result = to_numpy(model({f"{name}_u": values})[f"{name}_out"])
    np.testing.assert_allclose(result.ravel(), [3.0, 8.0, 15.0, 24.0])


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

def test_my_localmodel():
    x = Input("x_local", dim=(2, 2)).sw(4)
    g = Input("g_local", dim=1)
    k = Input("k_local", dim=1)
    mu = Fuzzify(centers=[0.0, 1.0, 2.0])(g.last())
    mu_k = Fuzzify(centers=[0.0, 1.0])(k.last())
    local = LocalModel(Linear(features=6), Fir(features=1), name="local_model", vectorized=True)
    local_output = local([x], [mu, mu_k])

    model = Modely(
        "local_model_test",
        inputs=[x, g, k],
        outputs=[Output("local_output", local_output)],
    ).build()

    dummy_x = np.ones([1, 2, 2, 4]).astype(np.float32)
    dummy_g = np.array([1, 1, 2, 3]).reshape([1, 1, 4]).astype(np.float32)
    dummy_k = np.array([0, 1, 0, 1]).reshape([1, 1, 4]).astype(np.float32)
    result = model({"x_local": dummy_x, "g_local": dummy_g, "k_local": dummy_k})
    assert result["local_output"].shape == (1, 1, 1)
