from nnodely import Input, Output, Modely, ReLU, Linear, EquationLearner, Sin, Cos, Fir, Select

from nnodely.core.layer import Identity
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
        [to_numpy(layers[f"{layer_name}{i}"].proj.kernel) for i in range(cells)]
    )
    bias = np.stack(
        [to_numpy(layers[f"{layer_name}{i}"].proj.bias) for i in range(cells)]
    )
    return kernel, bias


def test_local_model_matches_explicit_cells():
    # ------- The fused matmul must equal one Fir per membership -------
    x = Input("x_local", dim=1)
    k = Input("k_local", dim=1)
    centers = [0.0, 1.0, 2.0]

    activation = Fuzzify(centers=centers, function="Triangular")([k])
    fused = LocalModel(out_features=2, name="fused_local")([x.sw(4)], [activation])
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


def test_local_model_sums_input_activation_pairs():
    # ------- Multi-input: every input is scheduled by its own activation -----
    x = Input("x_pair", dim=1)
    y = Input("y_pair", dim=3)
    j = Input("j_pair", dim=1)
    k = Input("k_pair", dim=1)

    activation_x = Fuzzify(centers=[0.0, 1.0, 2.0], function="Triangular")([j])
    activation_y = Fuzzify(centers=[0.0, 1.0], function="Triangular")([k])

    paired = LocalModel(out_features=2, name="paired_local")(
        [x.sw(4), y.sw(2)], [activation_x, activation_y]
    )
    paired_model = Modely(
        "paired_local_model",
        inputs=[x, y, j, k],
        outputs=[Output("paired", paired)],
    ).build()

    # 3 cells over 1x4 features and 2 cells over 3x2 features, biases included.
    assert [tuple(kernel.shape) for kernel in paired.kernel] == [(3, 4, 2), (2, 6, 2)]
    assert [tuple(bias.shape) for bias in paired.bias] == [(3, 2), (2, 2)]

    single_x = LocalModel(out_features=2, name="single_x")([x.sw(4)], [activation_x])
    single_y = LocalModel(out_features=2, name="single_y")([y.sw(2)], [activation_y])
    single_model = Modely(
        "single_local_models",
        inputs=[x, y, j, k],
        outputs=[Output("single_x", single_x), Output("single_y", single_y)],
    ).build()

    for single, kernel, bias in zip(
        (single_x, single_y), paired.kernel, paired.bias
    ):
        single.kernel.assign(to_numpy(kernel))
        single.bias.assign(to_numpy(bias))

    dummy_input = {
        "x_pair": np.arange(20, dtype=np.float32).reshape(5, 1, 4) / 10.0,
        "y_pair": np.arange(30, dtype=np.float32).reshape(5, 3, 2) / 10.0,
        "j_pair": np.linspace(-0.5, 2.5, 5, dtype=np.float32).reshape(5, 1, 1),
        "k_pair": np.linspace(0.0, 1.0, 5, dtype=np.float32).reshape(5, 1, 1),
    }
    paired_result = paired_model(dict(dummy_input))["paired"]
    single_result = single_model(dict(dummy_input))

    assert paired_result.shape == (5, 2, 1)
    np.testing.assert_allclose(
        to_numpy(paired_result),
        to_numpy(single_result["single_x"]) + to_numpy(single_result["single_y"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_model_rejects_mismatched_inputs():
    x = Input("x_invalid", dim=1)
    k = Input("k_invalid", dim=1)
    activation = Fuzzify(centers=[0.0, 1.0], function="Triangular")([k.sw(1)])

    with pytest.raises(ValueError, match="own activation"):
        LocalModel(name="unpaired_local")([x.sw(2), x.sw(3)], [activation])

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
            use_bias=False,
            initializer="zeros",
            name="equation_linear_in",
        ),
        linear_out=Linear(
            out_features=1,
            use_bias=False,
            initializer="zeros",
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

    assert equation.linear_in is not None
    assert equation.linear_out is not None
    assert equation.linear_in.kernel is not None
    assert equation.linear_out.kernel is not None
    equation.linear_in.kernel.assign(
        np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    )
    equation.linear_out.kernel.assign(np.array([[2.0], [3.0], [4.0]], dtype=np.float32))

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
        linear_in=Linear(
            out_features=4,
            use_bias=False,
            initializer="ones",
        ),
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
