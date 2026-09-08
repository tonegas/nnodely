from nnodely import (
    Acos,
    Asin,
    Atan,
    Constant,
    Concatenate,
    Cos,
    ELU,
    EquationLearner,
    Fuzzify,
    GELU,
    Interpolation,
    Modely,
    Input,
    Output,
    Fir,
    LeakyReLU,
    Linear,
    LocalModel,
    PReLU,
    Parameter,
    ReLU,
    Sigmoid,
    Sin,
    Softmax,
    Softplus,
    Swish,
    Tan,
    Tanh,
    TimeConcatenate,
)
from nnodely.core.layer import Identity
from nnodely.layers.time_ops import Select
from conftest import to_numpy
import numpy as np
from pathlib import Path
import pytest


def graph_signature(model):
    flat = model.flatten()

    return [
        {
            "name": node.name,
            "type": type(node).__name__,
            "shape": repr(node.shape),
            "preds": [(p.name, repr(p.shape)) for p in node.preds],
            "config": node.get_config(),
        }
        for node in flat.order
    ]


def test_save_load_simple_model(tmp_path):
    ## simple model
    x = Input(name="x")
    y = Input(name="y")
    x_fir = Fir(out_features=1, use_bias=True, name="fir_x")([x.sw(5)])
    y_fir = Fir(out_features=1, use_bias=True, name="fir_y")([y.sw(5)])
    out = Output("accelleration", x_fir + y_fir)
    model = Modely(name="vehicle", inputs=[x, y], outputs=[out])
    model.build()

    ## dummy input
    x_data = np.random.randn(1, 5)
    y_data = np.random.randn(1, 5)

    pred = model({"x": x_data, "y": y_data})
    assert pred["accelleration"].shape == (1, 1, 1)

    model.save(tmp_path / "simple_save")

    new_model = Modely.load(tmp_path / "simple_save")

    new_pred = new_model({"x": x_data, "y": y_data})
    assert new_pred["accelleration"].shape == (1, 1, 1)
    np.testing.assert_allclose(
        to_numpy(pred["accelleration"]),
        to_numpy(new_pred["accelleration"]),
        rtol=1e-5,
        atol=1e-5,
    )


def _single_input_model(name, transform, *, dim=1, time=3):
    input_node = Input(f"{name}_input", dim=dim)
    stream = input_node.sw(time)
    output = Output(f"{name}_output", transform(stream))
    return Modely(name, inputs=[input_node], outputs=[output]).build()


def _arithmetic_model():
    parameter = Parameter("onnx_parameter", value=[[2.0]])
    constant = Constant("onnx_constant", value=[[0.5]])
    return _single_input_model(
        "onnx_arithmetic",
        lambda stream: (((stream * parameter) + constant - 0.25) / 2.0) ** 2,
        time=1,
    )


def _activation_model():
    return _single_input_model(
        "onnx_activations",
        lambda stream: Softplus()(
            GELU()(
                Swish()(
                    Tanh()(
                        Sigmoid()(
                            Softmax(axis=-1)(
                                PReLU()(ELU()(LeakyReLU()(ReLU()(stream))))
                            )
                        )
                    )
                )
            )
        ),
    )


def _trigonometric_model():
    return _single_input_model(
        "onnx_trigonometric",
        lambda stream: Atan()(Tan()(Acos()(Cos()(Asin()(Sin()(stream)))))),
        time=1,
    )


def _interpolation_model(mode):
    return _single_input_model(
        f"export_interpolation_{mode}",
        lambda stream: Interpolation(
            x_points=[-2.0, -1.0, 0.0, 1.0, 2.0],
            y_points=[3.0, 2.0, 3.0, 6.0, 11.0],
            mode=mode,
            name=f"{mode}_interpolation",
        )(stream),
        time=5,
    )


def _equation_learner_model():
    x = Input("equation_export_input")
    equation = EquationLearner(
        functions=["identity", Sin, Cos],
        linear_in=Linear(
            out_features=3,
            use_bias=False,
            initializer="ones",
            name="equation_export_linear_in",
        ),
        linear_out=Linear(
            out_features=1,
            use_bias=False,
            initializer="zeros",
            name="equation_export_linear_out",
        ),
        name="equation_export",
    )
    relation = equation(x.last())
    model = Modely(
        "onnx_equation_learner",
        inputs=[x],
        outputs=[Output("equation_export_output", relation)],
    ).build()
    assert equation.linear_out is not None
    assert equation.linear_out.kernel is not None
    equation.linear_out.kernel.assign(np.array([[2.0], [3.0], [4.0]], dtype=np.float32))
    return model


def _concatenate_model():
    x = Input("onnx_concat_x", dim=1)
    y = Input("onnx_concat_y", dim=1)
    concatenated = Concatenate(axis=0, name="export_dim_concatenate")(
        [x.sw(2), y.sw(2)]
    )
    concatenated = TimeConcatenate(name="export_time_concatenate")(
        [concatenated, concatenated]
    )
    return Modely(
        "onnx_concatenate",
        inputs=[x, y],
        outputs=[Output("onnx_concatenate_output", concatenated)],
    ).build()


def _local_model():
    x = Input("onnx_local_x", dim=1)
    activation_input = Input("onnx_local_activation", dim=1)
    activations = Fuzzify(centers=[0.0, 0.5, 1.0], function="Rectangular")(
        activation_input.sw(1)
    )
    local_function = LocalModel(
        input_function=lambda inputs: Identity()(inputs),
        output_function=lambda inputs: ReLU()(inputs),
        name="onnx_local_function",
    )(activation=activations)
    result = local_function([x.sw(1)])
    return Modely(
        "onnx_local_model",
        inputs=[x, activation_input],
        outputs=[Output("onnx_local_output", result)],
    ).build()


@pytest.mark.parametrize(
    ("model_factory", "inputs"),
    [
        (_arithmetic_model, {"onnx_arithmetic_input": [[[1.0]]]}),
        (
            lambda: _single_input_model(
                "onnx_fir", lambda stream: Fir(out_features=2)(stream)
            ),
            {"onnx_fir_input": [[[1.0, 2.0, 3.0]]]},
        ),
        (
            lambda: _single_input_model(
                "onnx_linear", lambda stream: Linear(out_features=2)(stream)
            ),
            {"onnx_linear_input": [[[1.0, 2.0, 3.0]]]},
        ),
        (_activation_model, {"onnx_activations_input": [[[-0.5, 0.0, 0.5]]]}),
        (_trigonometric_model, {"onnx_trigonometric_input": [[[0.25]]]}),
        (
            lambda: _interpolation_model("linear"),
            {"export_interpolation_linear_input": [[[-3.0, -0.5, 0.0, 1.5, 3.0]]]},
        ),
        (
            lambda: _interpolation_model("polynomial"),
            {"export_interpolation_polynomial_input": [[[-3.0, -0.5, 0.0, 1.5, 3.0]]]},
        ),
        (
            lambda: _single_input_model(
                "onnx_fuzzify",
                lambda stream: Fuzzify(centers=[-1.0, 0.0, 1.0], function="Gaussian")(
                    stream
                ),
                time=1,
            ),
            {"onnx_fuzzify_input": [[[0.25]]]},
        ),
        (
            lambda: _single_input_model(
                "onnx_select",
                lambda stream: Identity()(Select(idx=1, axis=0)(stream)),
                dim=3,
                time=1,
            ),
            {"onnx_select_input": [[[1.0], [2.0], [3.0]]]},
        ),
        (
            _concatenate_model,
            {
                "onnx_concat_x": [[[1.0, 2.0]]],
                "onnx_concat_y": [[[3.0, 4.0]]],
            },
        ),
        (
            _equation_learner_model,
            {"equation_export_input": [[[0.25]]]},
        ),
        (
            _local_model,
            {
                "onnx_local_x": [[[2.0]]],
                "onnx_local_activation": [[[0.5]]],
            },
        ),
    ],
)
def test_export_onnx_feedforward_blocks(tmp_path, model_factory, inputs):
    pytest.importorskip("onnxruntime")
    model = model_factory()
    inputs = {
        name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()
    }

    expected = model(inputs)
    path = model.export_onnx(tmp_path / model.name)
    actual = Modely.validate_onnx(path, inputs, return_dict=True)

    assert path.is_file()
    assert list(actual) == list(expected)
    for name in expected:
        np.testing.assert_allclose(
            to_numpy(actual[name]),
            to_numpy(expected[name]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_validate_onnx_rejects_missing_input(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _single_input_model("onnx_missing", lambda stream: Identity()(stream))
    path = model.export_onnx(tmp_path / "missing.onnx")

    with pytest.raises(ValueError, match="Missing ONNX inputs"):
        Modely.validate_onnx(path, {})


@pytest.mark.parametrize("mode", ["linear", "polynomial"])
def test_interpolation_save_keras_and_html(tmp_path, mode):
    model = _interpolation_model(mode)
    input_name = f"export_interpolation_{mode}_input"
    output_name = f"export_interpolation_{mode}_output"
    inputs = {input_name: np.array([[[-3.0, -0.5, 0.0, 1.5, 3.0]]], dtype=np.float32)}
    expected = model(inputs)[output_name]

    nnodely_path = tmp_path / f"interpolation_{mode}.nnodely"
    model.save(nnodely_path)
    restored = Modely.load(nnodely_path)
    np.testing.assert_allclose(
        to_numpy(restored(inputs)[output_name]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    restored_layer = next(
        node for node in restored.flatten().order if isinstance(node, Interpolation)
    )
    assert restored_layer.get_config() == {
        "x_points": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "y_points": [3.0, 2.0, 3.0, 6.0, 11.0],
        "mode": mode,
    }

    keras_path = tmp_path / f"interpolation_{mode}.keras"
    model.export_keras(keras_path)
    keras_model = Modely.import_keras(keras_path)
    keras_result = keras_model(inputs, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result[output_name]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    html_path = model.export_html(
        tmp_path,
        filename=f"interpolation_{mode}.html",
        physics=False,
    )
    html = Path(html_path).read_text(encoding="utf-8")
    assert f"{mode}_interpolation" in html
    assert '"class": "Interpolation"' in html
    assert '"x_points"' in html
    assert f'"mode": "{mode}"' in html


def test_concatenate_save_keras_and_html(tmp_path):
    model = _concatenate_model()
    inputs = {
        "onnx_concat_x": np.array([[[1.0, 2.0]]], dtype=np.float32),
        "onnx_concat_y": np.array([[[3.0, 4.0]]], dtype=np.float32),
    }
    expected = model(inputs)["onnx_concatenate_output"]

    nnodely_path = tmp_path / "concatenate.nnodely"
    model.save(nnodely_path)
    restored = Modely.load(nnodely_path)
    np.testing.assert_allclose(
        to_numpy(restored(inputs)["onnx_concatenate_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    keras_path = tmp_path / "concatenate.keras"
    model.export_keras(keras_path)
    keras_model = Modely.import_keras(keras_path)
    keras_result = keras_model(inputs, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result["onnx_concatenate_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    html_path = model.export_html(
        tmp_path,
        filename="concatenate.html",
        physics=False,
    )
    html = Path(html_path).read_text(encoding="utf-8")
    assert "export_dim_concatenate" in html
    assert '"class": "Concatenate"' in html
    assert '"axis": 0' in html
    assert "export_time_concatenate" in html
    assert '"class": "TimeConcatenate"' in html


def test_equation_learner_save_keras_and_html(tmp_path):
    model = _equation_learner_model()
    inputs = {
        "equation_export_input": np.array([[[0.25]]], dtype=np.float32),
    }
    expected = model(inputs)["equation_export_output"]

    nnodely_path = tmp_path / "equation_learner.nnodely"
    model.save(nnodely_path)
    restored = Modely.load(nnodely_path)
    np.testing.assert_allclose(
        to_numpy(restored(inputs)["equation_export_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    keras_path = tmp_path / "equation_learner.keras"
    model.export_keras(keras_path)
    keras_model = Modely.import_keras(keras_path)
    keras_result = keras_model(inputs, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result["equation_export_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    html_path = model.export_html(
        tmp_path,
        filename="equation_learner",
        physics=False,
    )
    html = Path(html_path).read_text(encoding="utf-8")
    assert "equation_export_call" in html
    assert '"nested_model": "equation_export"' in html

    nested_pages = list(
        tmp_path.glob("onnx_equation_learner__equation_export_call.html")
    )
    assert len(nested_pages) == 1
    nested_html = nested_pages[0].read_text(encoding="utf-8")
    assert "equation_export_linear_in" in nested_html
    assert "equation_export_identity_0" in nested_html
    assert "equation_export_sin_1" in nested_html
    assert "equation_export_cos_2" in nested_html
    assert "equation_export_linear_out" in nested_html


def _roll_model():
    x = Input("roll_x")
    fir = Fir(out_features=1, use_bias=False, name="roll_fir")(x.sw(5))
    output = Output("roll_out", fir + x.last())
    model = Modely("roll_model", inputs=[x], outputs=[output])
    model.rollback({x: fir}, steps=3, name="rollout")
    model.build()
    fir.kernel.assign(np.ones((5, 1), dtype=np.float32))
    return model


def _assert_roll_result(result):
    assert result["roll_out"].shape == (1, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["roll_out"]),
        np.array([[[85.0]]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_save_load_roll_model(tmp_path):
    model = _roll_model()
    inputs = {"roll_x": np.arange(1, 6, dtype=np.float32)}
    _assert_roll_result(model(inputs))

    path = tmp_path / "roll_model.nnodely"
    model.save(path)
    restored = Modely.load(path)

    assert restored._roll_steps == 3
    assert {
        input_node.name: stream.name
        for input_node, stream in restored._roll_callbacks.items()
    } == {"roll_x": "roll_fir"}
    _assert_roll_result(restored(inputs))


def test_export_keras_roll_model(tmp_path):
    model = _roll_model()
    inputs = {"roll_x": np.arange(1, 6, dtype=np.float32).reshape((1, 1, 5))}
    path = tmp_path / "roll_model.keras"

    model.export_keras(path)
    restored = Modely.import_keras(str(path.with_suffix("")))

    assert path.is_file()
    _assert_roll_result(restored(inputs))  # type: ignore


def test_export_onnx_roll_model(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _roll_model()
    inputs = {"roll_x": np.arange(1, 6, dtype=np.float32).reshape((1, 1, 5))}

    path = model.export_onnx(tmp_path / "roll.onnx")
    result = Modely.validate_onnx(path, inputs, return_dict=True)

    assert path.is_file()
    _assert_roll_result(result)
