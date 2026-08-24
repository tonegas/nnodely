from nnodely import Input, Output, Modely, ReLU

from nnodely.core.layer import Identity
from nnodely.layers.localmodel import LocalModel
from nnodely.layers.fuzzify import Fuzzify
import numpy as np
from conftest import to_numpy


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


def test_local_model():
    # ------- High-level Blocks (Local Models) with Multi-inputs -------
    x = Input("x", dim=1)
    k = Input("k", dim=1)

    fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")([k.sw(1)])
    local_model = LocalModel(
        input_function=lambda x: Identity()(x),
        output_function=lambda x: ReLU()(x),
        name="local_model",
    )(activation=fuzzy_k)

    out = Output("out", local_model([x.sw(1)]))
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
