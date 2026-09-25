from nnodely import (
    Abs,
    Acos,
    BatchNorm,
    Asin,
    Atan,
    Ceil,
    Clamp,
    Constant,
    Concatenate,
    Cos,
    Deg2Rad,
    Derivative,
    ELU,
    EquationLearner,
    Exp,
    Floor,
    Fuzzify,
    GELU,
    Integrate,
    Interpolation,
    Log,
    Log10,
    Loop,
    Modely,
    Input,
    Negative,
    Ode,
    OdeNet,
    Output,
    Fir,
    LeakyReLU,
    Linear,
    LocalModel,
    PReLU,
    Parameter,
    Range,
    ReLU,
    Roll,
    Sigmoid,
    Sign,
    Sin,
    Softmax,
    Softplus,
    Sqrt,
    Sum,
    Swish,
    Tan,
    Tanh,
    TimeConcatenate,
    TimeRange,
    TimeSelect,
)
from nnodely.core.layer import Identity
from nnodely.layers.time_ops import Select
from conftest import requires_onnx_export, to_numpy
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
    result = LocalModel(out_features=1, name="onnx_local_function")(
        [x.sw(1)], [activations]
    )
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
@requires_onnx_export
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


@requires_onnx_export
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
        "name": f"{mode}_interpolation",
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


def test_local_model_save_keras_and_html(tmp_path):
    model = _local_model()
    inputs = {
        "onnx_local_x": np.array([[[2.0]]], dtype=np.float32),
        "onnx_local_activation": np.array([[[0.5]]], dtype=np.float32),
    }
    expected = model(inputs)["onnx_local_output"]

    nnodely_path = tmp_path / "local_model.nnodely"
    model.save(nnodely_path)
    restored = Modely.load(nnodely_path)
    np.testing.assert_allclose(
        to_numpy(restored(inputs)["onnx_local_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    keras_path = tmp_path / "local_model.keras"
    model.export_keras(keras_path)
    keras_model = Modely.import_keras(keras_path)
    keras_result = keras_model(inputs, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result["onnx_local_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    html_path = model.export_html(tmp_path, filename="local_model", physics=False)
    html = Path(html_path).read_text(encoding="utf-8")
    assert "onnx_local_function" in html


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


@requires_onnx_export
def test_export_onnx_roll_model(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _roll_model()
    inputs = {"roll_x": np.arange(1, 6, dtype=np.float32).reshape((1, 1, 5))}

    path = model.export_onnx(tmp_path / "roll.onnx")
    result = Modely.validate_onnx(path, inputs, return_dict=True)

    assert path.is_file()
    _assert_roll_result(result)


def test_save_load_batchnorm_model(tmp_path):
    x = Input(name="batchnorm_save_input", dim=2)
    normalized = BatchNorm(name="batchnorm_save")([x.sw(3)])
    model = Modely(
        name="batchnorm_save_model",
        inputs=[x],
        outputs=[Output("batchnorm_save_output", normalized)],
    ).build()
    assert model.model is not None
    # Move the moving statistics away from their initial values, so a wrong
    # weight roundtrip changes the prediction.
    values = np.random.rand(4, 2, 3).astype(np.float32) * 10.0
    model.model({"batchnorm_save_input": values}, training=True)

    pred = model({"batchnorm_save_input": values})
    model.save(tmp_path / "batchnorm_save")

    new_model = Modely.load(tmp_path / "batchnorm_save")
    assert graph_signature(new_model) == graph_signature(model)

    new_pred = new_model({"batchnorm_save_input": values})
    np.testing.assert_allclose(
        to_numpy(pred["batchnorm_save_output"]),
        to_numpy(new_pred["batchnorm_save_output"]),
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Save/load round trips across the layer suite
#
# A reloaded model has to behave like the one that was saved: the same outputs
# on the same batch, the same graph, and it has to survive being saved again.
# ---------------------------------------------------------------------------

_BATCH = 3


def _randomize_trainable_weights(model):
    """Move every trainable weight off its initial value.

    A weight that is not restored then changes the prediction whatever its
    initializer, so a reload that only rebuilds the graph cannot pass.
    """
    assert model.model is not None
    rng = np.random.default_rng(0)
    for weight in model.model.trainable_weights:
        weight.assign(rng.uniform(-1.0, 1.0, size=weight.shape).astype(np.float32))
    return model


def _random_inputs(model):
    rng = np.random.default_rng(1)
    return {
        node.name: rng.uniform(0.1, 1.0, size=(_BATCH, *node.shape.tuple)).astype(
            np.float32
        )
        for node in model.train_inputs
    }


def _assert_same_outputs(actual, expected):
    assert sorted(actual) == sorted(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(
            to_numpy(actual[name]), value, rtol=1e-5, atol=1e-5, err_msg=name
        )


def _assert_round_trip(model, inputs, path):
    """Save ``model``, load it back and check the copy behaves the same.

    The copy is saved and loaded once more: a reloaded model is saved again as
    soon as it is fine-tuned, so it has to be as serializable as the original.
    """
    expected = {name: to_numpy(value) for name, value in model(dict(inputs)).items()}

    model.save(path / "saved")
    restored = Modely.load(path / "saved")
    _assert_same_outputs(restored(dict(inputs)), expected)
    assert graph_signature(restored) == graph_signature(model)
    # A layer shared by several parts of the model is still one layer.
    assert restored.model is not None
    assert len(restored.model.weights) == len(model.model.weights)

    restored.save(path / "resaved")
    _assert_same_outputs(Modely.load(path / "resaved")(dict(inputs)), expected)


def _unary_arithmetic_model():
    x = Input("unary_x", dim=2)
    window = x.sw(3)
    relations = {
        "unary_exp_log": Log()(Exp()(window)),
        "unary_log10_sqrt": Log10()(Sqrt()(window) + 1.0),
        "unary_rounding": Floor()(window * 10.0) + Ceil()(window * 10.0),
        "unary_sign_abs": Sign()(Negative()(window)) * Abs()(window - 0.5),
        "unary_deg2rad": Deg2Rad()(window),
        "unary_clamp": Clamp(min=0.3, max=0.7)(window),
        "unary_sum": Sum()(window),
    }
    return Modely(
        "unary_arithmetic",
        inputs=[x],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()


def _weighted_layers_model():
    x = Input("weighted_x", dim=1)
    y = Input("weighted_y", dim=3)
    # One Fir applied to two windows: both applications share its weights.
    shared_fir = Fir(out_features=2, name="weighted_shared_fir")
    past = shared_fir(x.sw(4))
    around = shared_fir(x.sw([3, 1]))
    projected = Linear(out_features=2, name="weighted_linear")(y.sw(2))
    mixed = Fir(out_features=2, use_bias=False, name="weighted_fir")(projected)
    gain = Parameter("weighted_gain", dim=2)
    offset = Constant("weighted_offset", value=[[1.0], [2.0]])
    return Modely(
        "weighted_layers",
        inputs=[x, y],
        outputs=[
            Output("weighted_out", (past - around + mixed) * gain + offset),
            Output("weighted_projected", projected),
        ],
    ).build()


def _shared_activation_model():
    x = Input("shared_activation_x", dim=1)
    y = Input("shared_activation_y", dim=1)
    # Layers are shared by name, so both applications must keep one alpha.
    activation = PReLU(name="shared_activation_prelu")
    return Modely(
        "shared_activation",
        inputs=[x, y],
        outputs=[
            Output(
                "shared_activation_out",
                activation(x.sw(2) - 0.5) + activation(y.sw(2) - 0.5),
            )
        ],
    ).build()


def _time_ops_model():
    x = Input("time_ops_x", dim=3)
    window = x.sw([3, 2])
    newest = TimeSelect(idx=-1, name="time_ops_time_select")(window)
    middle = TimeRange(start=1, end=4, name="time_ops_time_range")(window)
    third = Select(idx=2, axis=0, name="time_ops_select")(window)
    first_two = Range(start=0, end=2, axis=0, name="time_ops_range")(window)
    return Modely(
        "time_ops",
        inputs=[x],
        outputs=[
            Output(
                "time_ops_dims",
                Concatenate(axis=0, name="time_ops_concat")([first_two, third]),
            ),
            Output(
                "time_ops_times",
                TimeConcatenate(name="time_ops_time_concat")([newest, middle]),
            ),
        ],
    ).build()


def _fuzzy_local_model():
    x = Input("fuzzy_x", dim=1)
    gear = Input("fuzzy_gear", dim=1)
    speed = Input("fuzzy_speed", dim=1)
    triangular = Fuzzify(
        centers=[0.0, 0.5, 1.0], function="Triangular", name="fuzzy_triangular"
    )(gear.last())
    rectangular = Fuzzify(
        centers=[0.0, 0.5, 1.0], function="Rectangular", name="fuzzy_rectangular"
    )(speed.last())
    gaussian = Fuzzify(centers=[0.2, 0.8], function="Gaussian", name="fuzzy_gaussian")(
        speed.last()
    )
    # Two inputs, each scheduled by its own activation.
    affine = LocalModel(out_features=2, name="fuzzy_affine")(
        [x.sw(3), speed.sw(2)], [triangular, gaussian]
    )
    # One explicit cell per membership, built from ordinary layers.
    expanded = LocalModel(
        input_function=lambda stream: Fir(out_features=1)(stream),
        output_function=lambda stream: ReLU()(stream),
        name="fuzzy_expanded",
    )([x.sw(3)], [rectangular])
    return Modely(
        "fuzzy_local",
        inputs=[x, gear, speed],
        outputs=[
            Output("fuzzy_affine_out", affine),
            Output("fuzzy_expanded_out", expanded),
            Output("fuzzy_triangular_out", triangular),
        ],
    ).build()


def _derivative_model():
    dt = 0.1
    x = Input("derivative_x", dim=2)
    x0 = Input("derivative_x0", dim=2)
    u = Input("derivative_u", dim=1)
    window = x.sw(4)
    relation = Sin()(Linear(out_features=1, name="derivative_linear")(u.sw(3)))
    mixed = Linear(out_features=2, name="derivative_mix")(window)
    relations = {
        "d_init_stream": Derivative(order=1, respect_to=dt, init=x0.last())(window),
        "d_init_number": Derivative(order=2, respect_to=dt, init=0.5)(window),
        "d_smooth": Derivative(order=1, respect_to=dt, window=3, poly_order=2)(window),
        "d_of_layer": Derivative(order=1, respect_to=dt)(mixed),
        "d_wrt_input": Derivative(order=1, respect_to=u)(relation),
        "d2_wrt_input": Derivative(order=2, respect_to=u)(relation),
    }
    return Modely(
        "derivative_suite",
        inputs=[x, x0, u],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()


def _integrate_model():
    dt = 0.1
    rate = Input("integrate_rate", dim=2)
    state = Input("integrate_state", dim=2)
    signal = Input("integrate_signal", dim=1)
    x0 = Input("integrate_x0", dim=1)
    acceleration = Input("integrate_acceleration", dim=1)
    window = rate.sw(3)
    velocity = Integrate(dt=dt, init=0.2)(acceleration.sw(4))
    relations = {
        "i_euler": Integrate(solver="euler", dt=dt)(window),
        "i_trapezoidal": Integrate(solver="trapezoidal", dt=dt / 2, init=state.last())(
            window
        ),
        "i_rectangular": Integrate(solver="rectangular", dt=dt, init=1.5)(window),
        "i_position": Integrate(solver="trapezoidal", dt=dt, init=-0.3)(velocity),
        # Integrating the backward difference with the same initial condition
        # gives the signal back.
        "i_inverse": Integrate(dt=dt, init=x0.last())(
            Derivative(respect_to=dt, init=x0.last())(signal.sw(4))
        ),
    }
    return Modely(
        "integrate_suite",
        inputs=[rate, state, signal, x0, acceleration],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()


def _ode_step_model():
    position = Input("ode_step_p", dim=1)
    velocity = Input("ode_step_v", dim=1)
    stiffness = Parameter("ode_step_k", value=[[2.0]])
    damping = Constant("ode_step_c", value=[[0.3]])

    def field(p, v):
        return [v, -1.0 * stiffness * p - damping * v]

    p_next, v_next = Ode(
        field, [position.last(), velocity.last()], dt=0.1, method="rk4"
    )
    return Modely(
        "ode_step",
        inputs=[position, velocity],
        outputs=[Output("ode_step_p_next", p_next), Output("ode_step_v_next", v_next)],
    ).build()


def _rollback_mechanical_model():
    dt = 0.1
    position = Input("rollback_position", dim=1)
    velocity = Input("rollback_velocity", dim=1)
    force = Input("rollback_force", dim=1)
    stiffness = Parameter("rollback_stiffness", value=[[1.0]])
    # The damping reads the velocity estimated from the position window.
    estimated_velocity = TimeSelect(idx=-1)(Derivative(respect_to=dt)(position.sw(3)))
    acceleration = (
        Fir(out_features=1, name="rollback_force_fir")(force.sw(3))
        - stiffness * position.last()
        - Linear(out_features=1, use_bias=False, name="rollback_damping")(
            estimated_velocity
        )
    )
    velocity_next = Integrate(dt=dt, init=velocity.last(), name="rollback_v_next")(
        acceleration
    )
    position_next = Integrate(dt=dt, init=position.last(), name="rollback_p_next")(
        velocity_next
    )
    model = Modely(
        "rollback_mechanical",
        inputs=[position, velocity, force],
        outputs=[
            Output("rollback_position_out", position_next),
            Output("rollback_velocity_out", velocity_next),
        ],
    )
    model.rollback(
        {position: position_next, velocity: velocity_next},
        steps=4,
        name="rollback_mechanical_roll",
    )
    return model.build()


def _composition_model():
    signal = Input("composition_signal", dim=1)
    rate = Fir(out_features=1, name="composition_fir")(signal.sw(3))
    block = Modely(
        "composition_block",
        inputs=[signal],
        outputs=[
            Output("composition_state", Integrate(dt=0.1, init=signal.last())(rate))
        ],
    ).build()

    a = Input("composition_a", dim=1)
    b = Input("composition_b", dim=1)
    # The block is called twice, and both calls run with the same weights.
    return Modely(
        "composition",
        inputs=[a, b],
        outputs=[
            Output("composition_first", block([a.sw(3)])),
            Output("composition_second", block([b.sw(3)])),
        ],
    ).build()


def _loop_final_state_model():
    state = Input("loop_final_state", dim=1)
    force = Input("loop_final_force", dim=1)
    next_state = Output(
        "loop_final_next",
        Linear(out_features=1, name="loop_final_linear")(state.last()) + force.last(),
    )
    body = Modely(
        "loop_final_body", inputs=[state, force], outputs=[next_state]
    ).build()

    seed = Input("loop_final_seed", dim=1, seq=4)
    loop = Loop(
        f=body,
        callback={state: next_state},
        initial={state: seed},
        collect=False,
        name="loop_final",
    )
    return Modely(
        "loop_final",
        inputs=[seed, force],
        outputs=[Output("loop_final_out", loop)],
    ).build()


def _loop_trajectory_model():
    dt, horizon = 0.1, 5
    position = Input("loop_traj_position", dim=1)
    velocity = Input("loop_traj_velocity", dim=1)
    force = Input("loop_traj_force", dim=1)
    window = position.sw(2)
    acceleration = Fir(out_features=1, name="loop_traj_force_fir")(
        force.last()
    ) - Parameter("loop_traj_damping", value=[[0.5]]) * TimeSelect(idx=-1)(
        Derivative(respect_to=dt)(window)
    )
    velocity_step = Integrate(dt=dt, init=velocity.last())(acceleration)
    position_step = Integrate(
        solver="trapezoidal", dt=dt, init=TimeSelect(idx=-1)(window)
    )(velocity_step)
    position_next = Output("loop_traj_position_next", position_step)
    velocity_next = Output("loop_traj_velocity_next", velocity_step)
    body = Modely(
        "loop_traj_body",
        inputs=[position, velocity, force],
        outputs=[position_next, velocity_next],
    ).build()

    # The position window is shifted, the velocity seeded by a constant and the
    # force read one step at a time from its own sequence.
    p0 = Input("loop_traj_p0", dim=1, seq=horizon)
    force_sequence = Input("loop_traj_force_sequence", dim=1, seq=horizon)
    position_trajectory, velocity_trajectory = Loop(
        f=body,
        callback={position: position_next, velocity: velocity_next},
        initial={position: p0.sw(2), velocity: 0.25},
        inputs={force: force_sequence},
        name="loop_traj",
    )
    return Modely(
        "loop_traj",
        inputs=[p0, force_sequence],
        outputs=[
            Output("loop_traj_positions", position_trajectory),
            Output("loop_traj_velocities", velocity_trajectory),
        ],
    ).build()


def _nested_loop_model():
    x = Input("nested_save_input", dim=1, seq=(5, 3))
    gain = Constant("nested_save_gain", value=2.0)

    inner = Input("nested_save_inner", dim=1, seq=5)
    first_output = Output("nested_save_first_out", inner * gain)
    first_body = Modely(
        "nested_save_first_body", inputs=[inner], outputs=[first_output]
    ).build()
    first_loop = Loop(
        f=first_body,
        callback={inner: first_output},
        initial={inner: x},
        name="nested_save_first_loop",
        collect=False,
    )

    second = Input("nested_save_second", dim=1)
    second_output = Output("nested_save_second_out", second * gain)
    second_body = Modely(
        "nested_save_second_body", inputs=[second], outputs=[second_output]
    ).build()
    second_loop = Loop(
        f=second_body,
        callback={second: second_output},
        initial={second: first_loop},
        name="nested_save_second_loop",
        collect=False,
    )
    return Modely(
        "nested_save_loop",
        inputs=[x],
        outputs=[Output("nested_save_out", second_loop)],
    ).build()


def _shared_body_model():
    state = Input("shared_body_state", dim=1)
    gain = Parameter("shared_body_gain", value=[[0.5]])
    next_state = Output(
        "shared_body_next",
        Linear(out_features=1, name="shared_body_linear")(state.last()) * gain,
    )
    body = Modely("shared_body", inputs=[state], outputs=[next_state]).build()

    # One body rolled out by two loops, and its gain read outside it as well:
    # all three places use the same weights.
    short = Input("shared_body_short", dim=1, seq=3)
    long = Input("shared_body_long", dim=1, seq=5)
    first = Loop(
        f=body,
        callback={state: next_state},
        initial={state: short},
        collect=False,
        name="shared_body_first",
    )
    second = Loop(
        f=body,
        callback={state: next_state},
        initial={state: long},
        collect=False,
        name="shared_body_second",
    )
    return Modely(
        "shared_body_model",
        inputs=[short, long],
        outputs=[
            Output("shared_body_first_out", first),
            Output("shared_body_second_out", second + gain),
        ],
    ).build()


def _ode_loop_model():
    x = Input("ode_loop_x", dim=1)
    rate = Parameter("ode_loop_rate", value=[[1.0]])
    body_output = Output(
        "ode_loop_next",
        Ode(lambda state: -1.0 * rate * state, x.last(), 0.05, method="rk4"),
    )
    body = Modely("ode_loop_body", inputs=[x], outputs=[body_output]).build()

    seed = Input("ode_loop_seed", dim=1, seq=6)
    loop = Loop(f=body, callback={x: body_output}, initial={x: seed}, name="ode_loop")
    return Modely(
        "ode_loop", inputs=[seed], outputs=[Output("ode_loop_out", loop)]
    ).build()


def _roll_layer_model():
    x = Input("roll_layer_x", dim=1)
    u = Input("roll_layer_u", dim=1)
    feedback = Output(
        "roll_layer_next",
        Fir(out_features=1, name="roll_layer_fir")(x.sw(4))
        + Linear(out_features=1, name="roll_layer_linear")(u.last()),
    )
    body = Modely("roll_layer_body", inputs=[x, u], outputs=[feedback]).build()
    roll = Roll(f=body, callback={x: feedback}, steps=3, name="roll_layer")
    return Modely(
        "roll_layer", inputs=[x, u], outputs=[Output("roll_layer_out", roll)]
    ).build()


def _complete_vehicle_model():
    """Most layer families at once, wired like a real mechanical relation."""
    dt = 0.05
    speed = Input("vehicle_speed", dim=1)
    throttle = Input("vehicle_throttle", dim=1)
    gear = Input("vehicle_gear", dim=1)
    wheels = Input("vehicle_wheels", dim=4)

    gear_membership = Fuzzify(centers=[0.0, 0.5, 1.0], name="vehicle_gear_fuzzy")(
        gear.last()
    )
    engine = LocalModel(out_features=1, name="vehicle_engine")(
        [throttle.sw(5)], [gear_membership]
    )
    grip = Sigmoid()(
        PReLU()(
            Linear(out_features=1, name="vehicle_grip")(
                BatchNorm(name="vehicle_wheels_norm")(wheels.last())
            )
        )
    )
    slope = Interpolation(
        x_points=[0.0, 0.5, 1.0], y_points=[0.0, 0.2, 0.1], name="vehicle_slope"
    )(speed.last())
    drag = Parameter("vehicle_drag", value=[[0.1]]) * speed.last() ** 2
    gravity = Constant("vehicle_gravity", value=[[9.81]])
    acceleration = (
        engine * grip
        - drag
        - gravity * Sin()(slope)
        + Fir(out_features=1, name="vehicle_throttle_fir")(throttle.sw(3))
    )
    return Modely(
        "vehicle",
        inputs=[speed, throttle, gear, wheels],
        outputs=[
            Output(
                "vehicle_speed_next",
                Integrate(dt=dt, init=speed.last())(acceleration),
            ),
            Output(
                "vehicle_acceleration_estimate",
                Derivative(respect_to=dt, window=3, poly_order=2)(speed.sw(3)),
            ),
            Output("vehicle_grip_out", grip),
        ],
    ).build()


@pytest.mark.parametrize(
    "model_factory",
    [
        _arithmetic_model,
        _unary_arithmetic_model,
        _activation_model,
        _trigonometric_model,
        _weighted_layers_model,
        _shared_activation_model,
        _time_ops_model,
        _fuzzy_local_model,
        _derivative_model,
        _integrate_model,
        _ode_step_model,
        _rollback_mechanical_model,
        _composition_model,
        _loop_final_state_model,
        _loop_trajectory_model,
        _nested_loop_model,
        _shared_body_model,
        _ode_loop_model,
        _roll_layer_model,
        _complete_vehicle_model,
    ],
    ids=lambda factory: factory.__name__.strip("_").removesuffix("_model"),
)
def test_save_load_behaves_like_original(tmp_path, model_factory):
    model = _randomize_trainable_weights(model_factory())
    _assert_round_trip(model, _random_inputs(model), tmp_path)


@pytest.mark.parametrize("method", ["rk4", "dopri5"])
def test_save_load_odenet(tmp_path, method):
    points = 4
    x = Input(f"odenet_save_{method}_x", dim=2)
    field = Modely(
        f"odenet_save_{method}_field",
        inputs=[x],
        outputs=[
            Output(
                f"odenet_save_{method}_dx",
                Linear(out_features=2, use_bias=False)(x.last()),
            )
        ],
    ).build()
    t = Input(f"odenet_save_{method}_t", dim=1, seq=points)
    trajectory = OdeNet(
        f=field,
        states={x: f"odenet_save_{method}_dx"},
        t=t,
        method=method,
        steps=2,
        name=f"odenet_save_{method}",
    )
    model = Modely(
        f"odenet_save_{method}_model",
        inputs=[x, t],
        outputs=[Output(f"odenet_save_{method}_y", trajectory)],
    ).build()
    _randomize_trainable_weights(model)

    times = np.linspace(0.0, 1.0, points, dtype=np.float32).reshape(1, 1, 1, points)
    inputs = {
        x.name: np.random.default_rng(1)
        .uniform(-1.0, 1.0, size=(_BATCH, 2, 1))
        .astype(np.float32),
        t.name: np.tile(times, (_BATCH, 1, 1, 1)),
    }
    _assert_round_trip(model, inputs, tmp_path)


def test_save_load_odenet_with_event(tmp_path):
    # A ball dropped on a floor: the event and the reset are part of the model.
    position = Input("odenet_event_p", dim=1)
    velocity = Input("odenet_event_v", dim=1)
    gravity = Parameter("odenet_event_g", value=[9.8])
    floor = Parameter("odenet_event_floor", value=[0.0])
    restitution = Parameter("odenet_event_e", value=[0.8])
    field = Modely(
        "odenet_event_field",
        inputs=[position, velocity],
        outputs=[
            Output("odenet_event_dp", velocity.last()),
            Output("odenet_event_dv", 0.0 * position.last() - gravity),
            Output("odenet_event_guard", position.last() - floor),
            Output("odenet_event_p_plus", position.last()),
            Output("odenet_event_v_plus", -1.0 * restitution * velocity.last()),
        ],
    ).build()
    t = Input("odenet_event_t", dim=1, seq=4)
    position_trajectory, velocity_trajectory = OdeNet(
        f=field,
        states={position: "odenet_event_dp", velocity: "odenet_event_dv"},
        t=t,
        steps=20,
        event="odenet_event_guard",
        reset={position: "odenet_event_p_plus", velocity: "odenet_event_v_plus"},
        name="odenet_event",
    )
    model = Modely(
        "odenet_event_model",
        inputs=[position, velocity, t],
        outputs=[
            Output("odenet_event_positions", position_trajectory),
            Output("odenet_event_velocities", velocity_trajectory),
        ],
    ).build()

    times = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32).reshape(1, 1, 1, 4)
    inputs = {
        position.name: np.array([0.5, 1.0, 1.5], dtype=np.float32).reshape(3, 1, 1),
        velocity.name: np.zeros((_BATCH, 1, 1), dtype=np.float32),
        t.name: np.tile(times, (_BATCH, 1, 1, 1)),
    }
    _assert_round_trip(model, inputs, tmp_path)
