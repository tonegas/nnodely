from nnodely import (
    Acos,
    Asin,
    Atan,
    Constant,
    Concatenate,
    Cos,
    ELU,
    Fuzzify,
    GELU,
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
)
from nnodely.core.layer import Identity
from nnodely.layers.concatenate import TimeConcatenate
from nnodely.layers.time_ops import Select
from conftest import to_numpy
import numpy as np
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


def _concatenate_model():
    x = Input("onnx_concat_x", dim=1)
    y = Input("onnx_concat_y", dim=1)
    concatenated = Concatenate(axis=0)([x.sw(2), y.sw(2)])
    concatenated = TimeConcatenate()([concatenated, concatenated])
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
