from nnodely import (
    Gaussian,
    Rectangular,
    Triangular,
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
    DataLoader,
    Deg2Rad,
    DerivativeStep,
    Differentiate,
    ELU,
    EquationLearner,
    Exp,
    Floor,
    Fuzzify,
    GELU,
    IntegrateStep,
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
    TimeConcatenate,
    TimeRange,
    TimeSelect,
    SeqConcatenate,
    SeqRange,
    SeqReverse,
    SeqSelect,
    SeqStack,
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
)
from nnodely.core.layer import Identity
from nnodely.layers.partitions import Select
from conftest import requires_onnx_export, to_numpy
import json
import keras
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
    x = Input(name="x", sw=5)
    y = Input(name="y", sw=5)
    x_fir = Fir(features=1, bias=True, name="fir_x")([x])
    y_fir = Fir(features=1, bias=True, name="fir_y")([y])
    out = Output("accelleration", x_fir + y_fir)
    model = Modely(name="vehicle", inputs=[x, y], outputs=[out])
    model.build()

    ## dummy input
    x_data = np.random.randn(1, 5, 1)
    y_data = np.random.randn(1, 5, 1)

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
    input_node = Input(f"{name}_input", dim=dim, sw=time)
    stream = input_node
    output = Output(f"{name}_output", transform(stream))
    return Modely(name, inputs=[input_node], outputs=[output]).build()


def _arithmetic_model():
    parameter = Parameter("onnx_parameter", value=[2.0])
    constant = Constant("onnx_constant", value=[0.5])
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
                            Softmax(axis=1)(PReLU()(ELU()(LeakyReLU()(ReLU()(stream)))))
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
    x = Input("equation_export_input", sw=1)
    equation = EquationLearner(
        functions=["identity", Sin, Cos],
        linear_in=Linear(
            features=3, kernel="ones", bias=False, name="equation_export_linear_in"
        ),
        linear_out=Linear(
            features=1,
            kernel=Parameter("equation_export_weights", value=[2.0, 3.0, 4.0]),
            bias=False,
            name="equation_export_linear_out",
        ),
        name="equation_export",
    )
    relation = equation(x)
    model = Modely(
        "onnx_equation_learner",
        inputs=[x],
        outputs=[Output("equation_export_output", relation)],
    ).build()
    return model


def _concatenate_model():
    x = Input("onnx_concat_x", dim=1, sw=2)
    y = Input("onnx_concat_y", dim=1, sw=2)
    concatenated = Concatenate(axis=0, name="export_dim_concatenate")([x, y])
    concatenated = TimeConcatenate(name="export_time_concatenate")(
        [concatenated, concatenated]
    )
    return Modely(
        "onnx_concatenate",
        inputs=[x, y],
        outputs=[Output("onnx_concatenate_output", concatenated)],
    ).build()


def _local_model():
    x = Input("onnx_local_x", dim=1, sw=1)
    activation_input = Input("onnx_local_activation", dim=1, sw=1)
    activations = Fuzzify(centers=[0.0, 0.5, 1.0], function=Rectangular)(
        activation_input
    )
    result = LocalModel(name="onnx_local_function")([x], [activations])
    return Modely(
        "onnx_local_model",
        inputs=[x, activation_input],
        outputs=[Output("onnx_local_output", result)],
    ).build()


def _seq_ops_model():
    x = Input("seq_ops_x", dim=2, seq=(3, 4), sw=2)
    window = x
    return Modely(
        "seq_ops",
        inputs=[x],
        # Declared in name order until the TODO in _set_onnx_io_names is done.
        outputs=[
            Output(
                "seq_ops_joined",
                SeqConcatenate(axis=1, name="seq_ops_concat")(
                    [
                        SeqRange(1, 3, axis=1, name="seq_ops_range")([window]),
                        SeqReverse(axis=1, name="seq_ops_reverse")([window]),
                    ]
                ),
            ),
            Output(
                "seq_ops_last",
                SeqSelect(-1, name="seq_ops_select")([window]),
            ),
            Output(
                "seq_ops_stacked",
                SeqStack(axis=0, name="seq_ops_stack")(
                    [
                        SeqSelect(0)([window]),
                        SeqSelect(-1)([window]),
                    ]
                ),
            ),
        ],
    ).build()


@pytest.mark.parametrize(
    ("model_factory", "inputs"),
    [
        (_arithmetic_model, {"onnx_arithmetic_input": [[[1.0]]]}),
        (
            lambda: _single_input_model(
                "onnx_fir", lambda stream: Fir(features=2)(stream)
            ),
            {"onnx_fir_input": [[[1.0], [2.0], [3.0]]]},
        ),
        (
            lambda: _single_input_model(
                "onnx_linear", lambda stream: Linear(features=2)(stream)
            ),
            {"onnx_linear_input": [[[1.0], [2.0], [3.0]]]},
        ),
        (_activation_model, {"onnx_activations_input": [[[-0.5], [0.0], [0.5]]]}),
        (_trigonometric_model, {"onnx_trigonometric_input": [[[0.25]]]}),
        (
            lambda: _interpolation_model("linear"),
            {
                "export_interpolation_linear_input": [
                    [[-3.0], [-0.5], [0.0], [1.5], [3.0]]
                ]
            },
        ),
        (
            lambda: _interpolation_model("polynomial"),
            {
                "export_interpolation_polynomial_input": [
                    [[-3.0], [-0.5], [0.0], [1.5], [3.0]]
                ]
            },
        ),
        (
            lambda: _single_input_model(
                "onnx_fuzzify",
                lambda stream: Fuzzify(centers=[-1.0, 0.0, 1.0], function=Gaussian)(
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
            {"onnx_select_input": [[[1.0, 2.0, 3.0]]]},
        ),
        (
            _concatenate_model,
            {
                "onnx_concat_x": [[[1.0], [2.0]]],
                "onnx_concat_y": [[[3.0], [4.0]]],
            },
        ),
        (
            _equation_learner_model,
            {"equation_export_input": [[[0.25]]]},
        ),
        (
            _seq_ops_model,
            {
                "seq_ops_x": np.arange(48, dtype=np.float32)
                .reshape(1, 2, 2, 4, 3)
                .transpose(0, 4, 3, 2, 1)
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
@requires_onnx_export
def test_export_onnx_feedforward_blocks(tmp_path, model_factory, inputs):
    pytest.importorskip("onnxruntime")
    model = model_factory()
    inputs = {
        name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()
    }

    expected = model(inputs)
    model.export_onnx(tmp_path)
    path = tmp_path / f"{model.name}.onnx"
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


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="the jax backend is the one without an ONNX exporter",
)
def test_export_onnx_on_jax_says_it_is_unavailable(tmp_path):
    # Every other ONNX test is skipped on jax: this is what a jax user sees.
    model = _single_input_model("onnx_on_jax", lambda stream: Identity()(stream))

    with pytest.raises(NotImplementedError, match="not available on the jax backend"):
        model.export_onnx(tmp_path, "onnx_on_jax")
    assert not (tmp_path / "onnx_on_jax.onnx").exists()


@requires_onnx_export
def test_validate_onnx_rejects_missing_input(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _single_input_model("onnx_missing", lambda stream: Identity()(stream))
    model.export_onnx(tmp_path, "missing")
    path = tmp_path / "missing.onnx"

    with pytest.raises(ValueError, match="Missing ONNX inputs"):
        Modely.validate_onnx(path, {})


@pytest.mark.parametrize("mode", ["linear", "polynomial"])
def test_interpolation_save_keras_and_html(tmp_path, mode):
    model = _interpolation_model(mode)
    input_name = f"export_interpolation_{mode}_input"
    output_name = f"export_interpolation_{mode}_output"
    inputs = {
        input_name: np.array([[[-3.0], [-0.5], [0.0], [1.5], [3.0]]], dtype=np.float32)
    }
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
    model.export_keras(tmp_path, f"interpolation_{mode}")
    keras_model = Modely.import_keras(keras_path)
    keras_result = keras_model(inputs, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result[output_name]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    model.export_html(
        tmp_path,
        filename=f"interpolation_{mode}.html",
        physics=False,
    )
    html = (tmp_path / f"interpolation_{mode}.html").read_text(encoding="utf-8")
    assert f"{mode}_interpolation" in html
    assert '"class": "Interpolation"' in html
    assert '"x_points"' in html
    assert f'"mode": "{mode}"' in html


def test_concatenate_save_keras_and_html(tmp_path):
    model = _concatenate_model()
    inputs = {
        "onnx_concat_x": np.array([[[1.0], [2.0]]], dtype=np.float32),
        "onnx_concat_y": np.array([[[3.0], [4.0]]], dtype=np.float32),
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
    model.export_keras(tmp_path, "concatenate")
    keras_model = Modely.import_keras(keras_path)
    keras_result = keras_model(inputs, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result["onnx_concatenate_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    model.export_html(
        tmp_path,
        filename="concatenate.html",
        physics=False,
    )
    html = (tmp_path / "concatenate.html").read_text(encoding="utf-8")
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
    model.export_keras(tmp_path, "equation_learner")
    keras_model = Modely.import_keras(keras_path)
    keras_result = keras_model(inputs, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result["equation_export_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    model.export_html(
        tmp_path,
        filename="equation_learner",
        physics=False,
    )
    html = (tmp_path / "equation_learner.html").read_text(encoding="utf-8")
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
    model.export_keras(tmp_path, "local_model")
    keras_model = Modely.import_keras(keras_path)
    keras_result = keras_model(inputs, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result["onnx_local_output"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    model.export_html(tmp_path, filename="local_model", physics=False)
    html = (tmp_path / "local_model.html").read_text(encoding="utf-8")
    assert "onnx_local_function" in html


def test_local_model_with_fuzzy_product_round_trips(tmp_path):
    x = Input("save_local_x", dim=1, sw=3)
    j = Input("save_local_j", dim=1, sw=1)
    k = Input("save_local_k", dim=1, sw=1)
    activations = [
        Fuzzify(centers=[0.0, 1.0, 2.0], function=Triangular)([j]),
        Fuzzify(centers=[0.0, 1.0], function=Triangular)([k]),
    ]
    cloned = LocalModel(Fir(features=2), output_function=Tanh(), name="save_cloned")(
        [x], activations
    )
    factory = LocalModel(lambda index: Fir(features=2), name="save_factory")(
        [x], activations
    )
    vectorized = LocalModel(
        lambda streams: Fir(features=6)(streams),
        output_function=lambda streams: Tanh()(streams),
        vectorized=True,
        name="save_vectorized",
    )([x], activations)
    model = Modely(
        "save_local_model",
        inputs=[x, j, k],
        outputs=[
            Output("save_cloned_out", cloned),
            Output("save_factory_out", factory),
            Output("save_vectorized_out", vectorized),
        ],
    ).build()
    inputs = {
        "save_local_x": np.arange(12, dtype=np.float32).reshape(4, 3, 1) / 5.0,
        "save_local_j": np.linspace(0.0, 2.0, 4, dtype=np.float32).reshape(4, 1, 1),
        "save_local_k": np.linspace(0.2, 0.9, 4, dtype=np.float32).reshape(4, 1, 1),
    }
    expected = model(dict(inputs))

    path = tmp_path / "local_product.nnodely"
    model.save(path)
    restored = Modely.load(path)(dict(inputs))
    for name in ("save_cloned_out", "save_factory_out", "save_vectorized_out"):
        np.testing.assert_allclose(
            to_numpy(restored[name]), to_numpy(expected[name]), rtol=1e-5, atol=1e-5
        )


def _roll_model():
    x = Input("roll_x", sw=5)
    taps = Parameter("roll_taps", value=np.ones(5))
    fir = Fir(features=1, kernel=taps, bias=False, name="roll_fir")(x)
    output = Output("roll_out", fir + TimeSelect(-1)(x))
    model = Modely("roll_model", inputs=[x], outputs=[output])
    model.rollback({x: fir}, steps=3, name="rollout")
    model.build()
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
    inputs = {"roll_x": np.arange(1, 6, dtype=np.float32).reshape((1, 5, 1))}
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
    inputs = {"roll_x": np.arange(1, 6, dtype=np.float32).reshape((1, 5, 1))}
    path = tmp_path / "roll_model.keras"

    model.export_keras(tmp_path, "roll_model")
    restored = Modely.import_keras(str(path.with_suffix("")))

    assert path.is_file()
    _assert_roll_result(restored(inputs))  # type: ignore


@requires_onnx_export
def test_export_onnx_roll_model(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _roll_model()
    inputs = {"roll_x": np.arange(1, 6, dtype=np.float32).reshape((1, 5, 1))}

    model.export_onnx(tmp_path, "roll")
    path = tmp_path / "roll.onnx"
    result = Modely.validate_onnx(path, inputs, return_dict=True)

    assert path.is_file()
    _assert_roll_result(result)


def test_save_load_batchnorm_model(tmp_path):
    x = Input(name="batchnorm_save_input", dim=2, sw=3)
    normalized = BatchNorm(name="batchnorm_save")([x])
    model = Modely(
        name="batchnorm_save_model",
        inputs=[x],
        outputs=[Output("batchnorm_save_output", normalized)],
    ).build()
    assert model.model is not None
    # Move the moving statistics away from their initial values, so a wrong
    # weight roundtrip changes the prediction.
    values = np.random.rand(4, 3, 2).astype(np.float32) * 10.0
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
    assert restored.inference_model is not None
    assert len(restored.inference_model.weights) == len(model.inference_model.weights)

    restored.save(path / "resaved")
    _assert_same_outputs(Modely.load(path / "resaved")(dict(inputs)), expected)


def _unary_arithmetic_model():
    x = Input("unary_x", dim=2, sw=3)
    window = x
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
    x = Input("weighted_x", dim=1, sw=(4, 1))
    y = Input("weighted_y", dim=3, sw=2)
    # One Fir applied to two windows: both applications share its weights.
    shared_fir = Fir(features=2, name="weighted_shared_fir")
    past = shared_fir(TimeRange(0, 4)(x))
    around = shared_fir(TimeRange(1, 5)(x))
    projected = Linear(features=2, name="weighted_linear")(y)
    mixed = Fir(features=2, bias=False, name="weighted_fir")(projected)
    gain = Parameter("weighted_gain", dim=2)
    offset = Constant("weighted_offset", value=[1.0, 2.0])
    return Modely(
        "weighted_layers",
        inputs=[x, y],
        # The Fir of the projection has dim (2, 2), which a vector of dim (2,)
        # is never added to: it is an output of its own.
        outputs=[
            Output("weighted_out", (past - around) * gain + offset),
            Output("weighted_mixed", mixed),
            Output("weighted_projected", projected),
        ],
    ).build()


def _shared_activation_model():
    x = Input("shared_activation_x", dim=1, sw=2)
    y = Input("shared_activation_y", dim=1, sw=2)
    # Layers are shared by name, so both applications must keep one alpha.
    activation = PReLU(name="shared_activation_prelu")
    return Modely(
        "shared_activation",
        inputs=[x, y],
        outputs=[
            Output(
                "shared_activation_out",
                activation(x - 0.5) + activation(y - 0.5),
            )
        ],
    ).build()


def _time_ops_model():
    x = Input("time_ops_x", dim=3, sw=(3, 2))
    window = x
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
    x = Input("fuzzy_x", dim=1, sw=3)
    gear = Input("fuzzy_gear", dim=1, sw=1)
    speed = Input("fuzzy_speed", dim=1, sw=1)
    triangular = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=Triangular, name="fuzzy_triangular"
    )(gear)
    rectangular = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=Rectangular, name="fuzzy_rectangular"
    )(speed)
    gaussian = Fuzzify(centers=[0.2, 0.8], function=Gaussian, name="fuzzy_gaussian")(
        speed
    )
    # Two fuzzifications combined into joint memberships, one Fir for every cell.
    affine = LocalModel(Fir(features=2), name="fuzzy_affine")(
        [x], [triangular, gaussian]
    )
    # One explicit cell per membership, built from ordinary layers.
    expanded = LocalModel(
        input_function=lambda index: Fir(features=1),
        output_function=lambda index: ReLU(),
        name="fuzzy_expanded",
    )([x], [rectangular])
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
    x = Input("derivative_x", dim=2, sw=4)
    x0 = Input("derivative_x0", dim=2, sw=1)
    u = Input("derivative_u", dim=1, sw=3)
    window = x
    relation = Sin()(Linear(features=1, name="derivative_linear")(u))
    mixed = Linear(features=2, name="derivative_mix")(window)
    relations = {
        "d_init_stream": DerivativeStep(order=1, dt=dt)(x0, window),
        "d_init_number": DerivativeStep(order=2, dt=dt)(0.5, window),
        "d_smooth": DerivativeStep(order=1, dt=dt, window=3, poly_order=2)(window),
        "d_of_layer": DerivativeStep(order=1, dt=dt)(mixed),
        "d_wrt_input": Differentiate(order=1, respect_to=u)(relation),
        "d2_wrt_input": Differentiate(order=2, respect_to=u)(relation),
    }
    return Modely(
        "derivative_suite",
        inputs=[x, x0, u],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()


def _integrate_model():
    dt = 0.1
    rate = Input("integrate_rate", dim=2, sw=3)
    state = Input("integrate_state", dim=2, sw=1)
    signal = Input("integrate_signal", dim=1, sw=4)
    x0 = Input("integrate_x0", dim=1, sw=1)
    acceleration = Input("integrate_acceleration", dim=1, sw=4)
    window = rate
    velocity = IntegrateStep(dt=dt)(0.2, acceleration)
    relations = {
        "i_euler": IntegrateStep(solver="euler", dt=dt)(window),
        "i_trapezoidal": IntegrateStep(solver="trapezoidal", dt=dt / 2)(state, window),
        "i_rectangular": IntegrateStep(solver="rectangular", dt=dt)(1.5, window),
        "i_position": IntegrateStep(solver="trapezoidal", dt=dt)(-0.3, velocity),
        # Integrating the backward difference with the same initial condition
        # gives the signal back.
        "i_inverse": IntegrateStep(dt=dt)(x0, DerivativeStep(dt=dt)(x0, signal)),
    }
    return Modely(
        "integrate_suite",
        inputs=[rate, state, signal, x0, acceleration],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()


def _ode_step_model():
    position = Input("ode_step_p", dim=1, sw=1)
    velocity = Input("ode_step_v", dim=1, sw=1)
    stiffness = Parameter("ode_step_k", value=[2.0])
    damping = Constant("ode_step_c", value=[0.3])

    def field(p, v):
        return [v, -1.0 * stiffness * p - damping * v]

    p_next, v_next = Ode(field, [position, velocity], dt=0.1, method="rk4")
    return Modely(
        "ode_step",
        inputs=[position, velocity],
        outputs=[Output("ode_step_p_next", p_next), Output("ode_step_v_next", v_next)],
    ).build()


def _rollback_mechanical_model():
    dt = 0.1
    position = Input("rollback_position", dim=1, sw=3)
    velocity = Input("rollback_velocity", dim=1, sw=1)
    force = Input("rollback_force", dim=1, sw=3)
    stiffness = Parameter("rollback_stiffness", value=[1.0])
    # The damping reads the velocity estimated from the position window.
    estimated_velocity = TimeSelect(idx=-1)(DerivativeStep(dt=dt)(position))
    acceleration = (
        Fir(features=1, name="rollback_force_fir")(force)
        - stiffness * TimeSelect(-1)(position)
        - Linear(features=1, bias=False, name="rollback_damping")(estimated_velocity)
    )
    velocity_next = IntegrateStep(dt=dt, name="rollback_v_next")(velocity, acceleration)
    position_next = IntegrateStep(dt=dt, name="rollback_p_next")(
        TimeSelect(-1)(position), velocity_next
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
    signal = Input("composition_signal", dim=1, sw=3)
    rate = Fir(features=1, name="composition_fir")(signal)
    block = Modely(
        "composition_block",
        inputs=[signal],
        outputs=[
            Output(
                "composition_state",
                IntegrateStep(dt=0.1)(TimeSelect(-1)(signal), rate),
            )
        ],
    ).build()

    a = Input("composition_a", dim=1, sw=3)
    b = Input("composition_b", dim=1, sw=3)
    # The block is called twice, and both calls run with the same weights.
    return Modely(
        "composition",
        inputs=[a, b],
        outputs=[
            Output("composition_first", block([a])),
            Output("composition_second", block([b])),
        ],
    ).build()


def _loop_final_state_model():
    state = Input("loop_final_state", dim=1, sw=1)
    force = Input("loop_final_force", dim=1, sw=1)
    next_state = Output(
        "loop_final_next",
        Linear(features=1, name="loop_final_linear")(state) + force,
    )
    body = Modely(
        "loop_final_body", inputs=[state, force], outputs=[next_state]
    ).build()

    seed = Input("loop_final_seed", dim=1, seq=4, sw=1)
    loop = Loop(
        f=body,
        callback={state: next_state},
        collect=False,
        name="loop_final",
        init={state: seed},
    )()
    return Modely(
        "loop_final",
        inputs=[seed, force],
        outputs=[Output("loop_final_out", loop)],
    ).build()


def _loop_trajectory_model():
    dt, horizon = 0.1, 5
    position = Input("loop_traj_position", dim=1, sw=2)
    velocity = Input("loop_traj_velocity", dim=1, sw=1)
    force = Input("loop_traj_force", dim=1, sw=1)
    window = position
    acceleration = Fir(features=1, name="loop_traj_force_fir")(force) - Parameter(
        "loop_traj_damping", value=[0.5]
    ) * TimeSelect(idx=-1)(DerivativeStep(dt=dt)(window))
    velocity_step = IntegrateStep(dt=dt)(velocity, acceleration)
    position_step = IntegrateStep(solver="trapezoidal", dt=dt)(
        TimeSelect(idx=-1)(window), velocity_step
    )
    position_next = Output("loop_traj_position_next", position_step)
    velocity_next = Output("loop_traj_velocity_next", velocity_step)
    body = Modely(
        "loop_traj_body",
        inputs=[position, velocity, force],
        outputs=[position_next, velocity_next],
    ).build()

    # The position window is shifted, the velocity seeded by a constant and the
    # force read one step at a time from its own sequence.
    p0 = Input("loop_traj_p0", dim=1, seq=horizon, sw=2)
    force_sequence = Input("loop_traj_force_sequence", dim=1, seq=horizon, sw=1)
    position_trajectory, velocity_trajectory = Loop(
        f=body,
        callback={position: position_next, velocity: velocity_next},
        name="loop_traj",
        init={position: p0, velocity: 0.25},
    )({force: force_sequence})
    return Modely(
        "loop_traj",
        inputs=[p0, force_sequence],
        outputs=[
            Output("loop_traj_positions", position_trajectory),
            Output("loop_traj_velocities", velocity_trajectory),
        ],
    ).build()


def _nested_loop_model():
    x = Input("nested_save_input", dim=1, seq=(3, 5), sw=1)
    gain = Constant("nested_save_gain", value=2.0)

    inner = Input("nested_save_inner", dim=1, seq=5, sw=1)
    first_output = Output("nested_save_first_out", inner * gain)
    first_body = Modely(
        "nested_save_first_body", inputs=[inner], outputs=[first_output]
    ).build()
    first_loop = Loop(
        f=first_body,
        callback={inner: first_output},
        name="nested_save_first_loop",
        collect=False,
        init={inner: x},
    )()

    second = Input("nested_save_second", dim=1, sw=1)
    second_output = Output("nested_save_second_out", second * gain)
    second_body = Modely(
        "nested_save_second_body", inputs=[second], outputs=[second_output]
    ).build()
    second_loop = Loop(
        f=second_body,
        callback={second: second_output},
        name="nested_save_second_loop",
        collect=False,
        init={second: first_loop},
    )()
    return Modely(
        "nested_save_loop",
        inputs=[x],
        outputs=[Output("nested_save_out", second_loop)],
    ).build()


def _shared_body_model():
    state = Input("shared_body_state", dim=1, sw=1)
    gain = Parameter("shared_body_gain", value=[0.5])
    next_state = Output(
        "shared_body_next",
        Linear(features=1, name="shared_body_linear")(state) * gain,
    )
    body = Modely("shared_body", inputs=[state], outputs=[next_state]).build()

    # One body rolled out by two loops, and its gain read outside it as well:
    # all three places use the same weights.
    short = Input("shared_body_short", dim=1, seq=3, sw=1)
    long = Input("shared_body_long", dim=1, seq=5, sw=1)
    first = Loop(
        f=body,
        callback={state: next_state},
        collect=False,
        name="shared_body_first",
        init={state: short},
    )()
    second = Loop(
        f=body,
        callback={state: next_state},
        collect=False,
        name="shared_body_second",
        init={state: long},
    )()
    return Modely(
        "shared_body_model",
        inputs=[short, long],
        outputs=[
            Output("shared_body_first_out", first),
            Output("shared_body_second_out", second + gain),
        ],
    ).build()


def _ode_loop_model():
    x = Input("ode_loop_x", dim=1, sw=1)
    rate = Parameter("ode_loop_rate", value=[1.0])
    body_output = Output(
        "ode_loop_next",
        Ode(lambda state: -1.0 * rate * state, x, 0.05, method="rk4"),
    )
    body = Modely("ode_loop_body", inputs=[x], outputs=[body_output]).build()

    seed = Input("ode_loop_seed", dim=1, seq=6, sw=1)
    loop = Loop(f=body, callback={x: body_output}, name="ode_loop", init={x: seed})()
    return Modely(
        "ode_loop", inputs=[seed], outputs=[Output("ode_loop_out", loop)]
    ).build()


def _roll_layer_model():
    x = Input("roll_layer_x", dim=1, sw=4)
    u = Input("roll_layer_u", dim=1, sw=1)
    feedback = Output(
        "roll_layer_next",
        Fir(features=1, name="roll_layer_fir")(x)
        + Linear(features=1, name="roll_layer_linear")(u),
    )
    body = Modely("roll_layer_body", inputs=[x, u], outputs=[feedback]).build()
    roll = Roll(f=body, callback={x: feedback}, steps=3, name="roll_layer")
    return Modely(
        "roll_layer", inputs=[x, u], outputs=[Output("roll_layer_out", roll)]
    ).build()


def _complete_vehicle_model():
    """Most layer families at once, wired like a real mechanical relation."""
    dt = 0.05
    speed = Input("vehicle_speed", dim=1, sw=3)
    throttle = Input("vehicle_throttle", dim=1, sw=5)
    gear = Input("vehicle_gear", dim=1, sw=1)
    wheels = Input("vehicle_wheels", dim=4, sw=1)

    gear_membership = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=Triangular, name="vehicle_gear_fuzzy"
    )(gear)
    engine = LocalModel(Fir(features=1), name="vehicle_engine")(
        [throttle], [gear_membership]
    )
    grip = Sigmoid()(
        PReLU()(
            Linear(features=1, name="vehicle_grip")(
                BatchNorm(name="vehicle_wheels_norm")(wheels)
            )
        )
    )
    slope = Interpolation(
        x_points=[0.0, 0.5, 1.0], y_points=[0.0, 0.2, 0.1], name="vehicle_slope"
    )(TimeSelect(-1)(speed))
    drag = Parameter("vehicle_drag", value=[0.1]) * TimeSelect(-1)(speed) ** 2
    gravity = Constant("vehicle_gravity", value=[9.81])
    acceleration = (
        engine * grip
        - drag
        - gravity * Sin()(slope)
        + Fir(features=1, name="vehicle_throttle_fir")(TimeRange(2, 5)(throttle))
    )
    return Modely(
        "vehicle",
        inputs=[speed, throttle, gear, wheels],
        outputs=[
            Output(
                "vehicle_speed_next",
                IntegrateStep(dt=dt)(TimeSelect(-1)(speed), acceleration),
            ),
            Output(
                "vehicle_acceleration_estimate",
                DerivativeStep(dt=dt, window=3, poly_order=2)(speed),
            ),
            Output("vehicle_grip_out", grip),
        ],
    ).build()


_LAYER_SUITE = pytest.mark.parametrize(
    "model_factory",
    [
        _arithmetic_model,
        _unary_arithmetic_model,
        _activation_model,
        _trigonometric_model,
        _weighted_layers_model,
        _shared_activation_model,
        _time_ops_model,
        _seq_ops_model,
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


@pytest.mark.parametrize(
    "model_factory",
    [_loop_final_state_model, _loop_trajectory_model],
    ids=["loop_final_state", "loop_trajectory"],
)
@requires_onnx_export
def test_export_onnx_loop_model(tmp_path, model_factory):
    pytest.importorskip("onnxruntime")
    model = model_factory()
    # The torch exporter fixes the batch axis to the one it traces with.
    inputs = {name: value[:1] for name, value in _random_inputs(model).items()}

    expected = model(inputs)
    model.export_onnx(tmp_path)
    path = tmp_path / f"{model.name}.onnx"
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


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="only the tensorflow exporter keeps the rollout axis dynamic",
)
@requires_onnx_export
def test_export_onnx_dynamic_loop_follows_the_data(tmp_path):
    # Exported with an unknown width, the rollout reads its length from the
    # data instead of the length it was declared with.
    pytest.importorskip("onnxruntime")
    x, u = Input("dyn_onnx_x", dim=1, sw=1), Input("dyn_onnx_u", dim=1, sw=1)
    body_output = Output("dyn_onnx_next", x * 0.5 + u)
    body = Modely("dyn_onnx_body", inputs=[x, u], outputs=[body_output]).build()
    x_seq = Input("dyn_onnx_x_seq", dim=1, seq=-1, sw=1)
    u_seq = Input("dyn_onnx_u_seq", dim=1, seq=-1, sw=1)
    loop = Loop(f=body, callback={x: body_output}, init={x: x_seq})({u: u_seq})
    model = Modely(
        "dyn_onnx_model",
        inputs=[x_seq, u_seq],
        outputs=[Output("dyn_onnx_out", loop)],
    ).build()
    model.export_onnx(tmp_path)

    for steps in (1, 7):
        inputs = {
            "dyn_onnx_x_seq": np.ones((1, steps, 1, 1), dtype=np.float32),
            "dyn_onnx_u_seq": np.arange(steps, dtype=np.float32).reshape(
                1, steps, 1, 1
            ),
        }
        expected = to_numpy(model(inputs)["dyn_onnx_out"])
        actual = Modely.validate_onnx(
            tmp_path / f"{model.name}.onnx", inputs, return_dict=True
        )["dyn_onnx_out"]  # type: ignore
        assert expected.shape == (1, steps, 1, 1)
        np.testing.assert_allclose(to_numpy(actual), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="only the tensorflow exporter keeps the sequence axis dynamic",
)
@requires_onnx_export
def test_export_onnx_seq_ops_follow_the_data(tmp_path):
    # Counted from the end, the selected elements are those of the data, not
    # of the length the graph was declared with.
    pytest.importorskip("onnxruntime")
    x = Input("dyn_seq_ops_x", dim=1, seq=-1, sw=1)
    sequence = x
    model = Modely(
        "dyn_seq_ops",
        inputs=[x],
        outputs=[
            Output(
                "dyn_seq_ops_last",
                SeqSelect(-1)([sequence]),
            ),
            Output(
                "dyn_seq_ops_tail",
                SeqReverse()(SeqRange(-2, None)([sequence])),
            ),
        ],
    ).build()
    model.export_onnx(tmp_path)

    for steps in (2, 6):
        inputs = {
            "dyn_seq_ops_x": np.arange(steps, dtype=np.float32).reshape(1, steps, 1, 1)
        }
        expected = model(inputs)
        actual = Modely.validate_onnx(
            tmp_path / f"{model.name}.onnx", inputs, return_dict=True
        )
        for name in expected:
            np.testing.assert_allclose(to_numpy(actual[name]), to_numpy(expected[name]))
        np.testing.assert_allclose(
            to_numpy(expected["dyn_seq_ops_tail"]).ravel(), [steps - 1, steps - 2]
        )


@_LAYER_SUITE
def test_save_load_behaves_like_original(tmp_path, model_factory):
    model = _randomize_trainable_weights(model_factory())
    _assert_round_trip(model, _random_inputs(model), tmp_path)


def _target_for(output):
    """A dataset input shaped like ``output``, the way a user adds a target."""
    seq = tuple(-1 if size is None else size for size in output.shape.seq)
    return Input(f"{output.name}_target", dim=output.dim, seq=seq, sw=output.shape.time)


@_LAYER_SUITE
def test_loaded_model_rebuilt_with_new_minimizers_keeps_its_weights(
    tmp_path, model_factory
):
    model = _randomize_trainable_weights(model_factory())
    inputs = _random_inputs(model)
    model.save(tmp_path / "pretrained")

    loaded = Modely.load(tmp_path / "pretrained")
    assert loaded.inference_model is not None
    expected = {name: to_numpy(value) for name, value in loaded(dict(inputs)).items()}
    weight_count = len(loaded.inference_model.weights)

    # The user's own minimizers bring new inputs and new graph nodes, so the
    # model is built again - around the blocks it loaded, not fresh ones.
    for output in loaded.outputs:
        loaded.minimize(f"{output.name}_fit", source=output, target=_target_for(output))
    loaded.build()

    assert loaded.inference_model is not None
    assert len(loaded.inference_model.weights) == weight_count
    _assert_same_outputs(loaded(dict(inputs)), expected)


def test_fine_tune_a_loaded_model_with_its_own_minimizers(tmp_path):
    samples = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    x = Input("fine_tune_x", dim=1, sw=3)
    prediction = Output("fine_tune_out", Fir(features=1, name="fine_tune_fir")(x))
    model = Modely("fine_tune", inputs=[x], outputs=[prediction])
    model.minimize(
        "pretrain", source=prediction, target=Input("pretrain_target", dim=1, sw=1)
    )
    model.build()
    model.train(
        train_data=DataLoader(
            model, source={"fine_tune_x": samples, "pretrain_target": 2.0 * samples}
        ),
        epochs=5,
        batch_size=8,
        lr=1e-2,
        printer=None,
    )
    model.save(tmp_path / "pretrained")

    windows = np.random.default_rng(1).uniform(size=(_BATCH, 3, 1))
    inputs = {"fine_tune_x": windows.astype(np.float32)}
    pretrained = to_numpy(model(inputs)["fine_tune_out"])

    # A loaded model is known by the names of its streams, not by their nodes.
    loaded = Modely.load(tmp_path / "pretrained")
    loaded.minimize(
        "fine_tune",
        source="fine_tune_out",
        target=Input("fine_tune_target", dim=1, sw=1),
    )
    loaded.build()
    np.testing.assert_allclose(
        to_numpy(loaded(inputs)["fine_tune_out"]), pretrained, rtol=1e-6
    )

    # Fine-tuning starts from the loaded weights and moves them, and inference
    # runs with the weights the training left.
    loaded.train(
        train_data=DataLoader(
            loaded, source={"fine_tune_x": samples, "fine_tune_target": -samples}
        ),
        epochs=5,
        batch_size=8,
        lr=1e-2,
        printer=None,
    )
    tuned = to_numpy(loaded(inputs)["fine_tune_out"])
    assert not np.allclose(tuned, pretrained)
    assert loaded.model is not None
    trained = loaded.model(
        {**inputs, "fine_tune_target": np.zeros((_BATCH, 1, 1), dtype=np.float32)},
        training=False,
    )
    np.testing.assert_allclose(tuned, to_numpy(trained["fine_tune_out"]), rtol=1e-6)


@pytest.mark.parametrize("method", ["rk4", "dopri5"])
def test_save_load_odenet(tmp_path, method):
    points = 4
    x = Input(f"odenet_save_{method}_x", dim=2, sw=1)
    field = Modely(
        f"odenet_save_{method}_field",
        inputs=[x],
        outputs=[
            Output(
                f"odenet_save_{method}_dx",
                Linear(features=2, bias=False)(x),
            )
        ],
    ).build()
    t = Input(f"odenet_save_{method}_t", dim=1, seq=points, sw=1)
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

    times = np.linspace(0.0, 1.0, points, dtype=np.float32).reshape(1, points, 1, 1)
    inputs = {
        x.name: np.random.default_rng(1)
        .uniform(-1.0, 1.0, size=(_BATCH, 1, 2))
        .astype(np.float32),
        t.name: np.tile(times, (_BATCH, 1, 1, 1)),
    }
    _assert_round_trip(model, inputs, tmp_path)


@pytest.mark.slow
def test_save_load_odenet_with_event(tmp_path):
    # A ball dropped on a floor: the event and the reset are part of the model.
    position = Input("odenet_event_p", dim=1, sw=1)
    velocity = Input("odenet_event_v", dim=1, sw=1)
    gravity = Parameter("odenet_event_g", value=[9.8])
    floor = Parameter("odenet_event_floor", value=[0.0])
    restitution = Parameter("odenet_event_e", value=[0.8])
    field = Modely(
        "odenet_event_field",
        inputs=[position, velocity],
        outputs=[
            Output("odenet_event_dp", velocity),
            Output("odenet_event_dv", 0.0 * position - gravity),
            Output("odenet_event_guard", position - floor),
            Output("odenet_event_p_plus", position),
            Output("odenet_event_v_plus", -1.0 * restitution * velocity),
        ],
    ).build()
    t = Input("odenet_event_t", dim=1, seq=4, sw=1)
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

    times = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32).reshape(1, 4, 1, 1)
    inputs = {
        position.name: np.array([0.5, 1.0, 1.5], dtype=np.float32).reshape(3, 1, 1),
        velocity.name: np.zeros((_BATCH, 1, 1), dtype=np.float32),
        t.name: np.tile(times, (_BATCH, 1, 1, 1)),
    }
    _assert_round_trip(model, inputs, tmp_path)


def _minimized_model():
    """A model whose minimizers read a target and a layer of their own."""
    x = Input("minimized_x", dim=1, sw=3)
    target = Input("minimized_target", dim=1, sw=1)
    prediction = Output("minimized_out", Fir(features=1, name="minimized_fir")(x))
    model = Modely("minimized", inputs=[x, target], outputs=[prediction])
    model.minimize("minimized_fit", source=prediction, target=target)
    # A stream only a minimizer reads, computed by a weight of its own.
    auxiliary = Linear(features=1, name="minimized_auxiliary")(TimeSelect(-1)(x))
    model.minimize("minimized_auxiliary_fit", source=auxiliary, target=target)
    return _randomize_trainable_weights(model.build())


def _minimized_inputs():
    # Only the model's own input: nothing exported or saved asks for the target.
    values = np.random.default_rng(1).uniform(size=(_BATCH, 3, 1))
    return {"minimized_x": values.astype(np.float32)}


def test_save_load_leaves_minimizers_out(tmp_path):
    model = _minimized_model()
    _assert_round_trip(model, _minimized_inputs(), tmp_path)

    restored = Modely.load(tmp_path / "saved")
    assert restored.minimizers == []
    assert [node.name for node in restored.train_inputs] == ["minimized_x"]
    saved = json.loads((tmp_path / "saved" / "model.json").read_text(encoding="utf-8"))
    saved_names = {node["config"]["name"] for node in saved["nodes"]}
    assert "minimized_target" not in saved_names
    assert "minimized_auxiliary" not in saved_names
    # The auxiliary layer trains with the model, but is not part of it.
    assert model.model is not None and model.inference_model is not None
    assert len(model.model.weights) > len(model.inference_model.weights)


def test_export_keras_leaves_minimizers_out(tmp_path):
    model = _minimized_model()
    inputs = _minimized_inputs()
    expected = model(dict(inputs))["minimized_out"]

    path = tmp_path / "minimized.keras"
    model.export_keras(tmp_path, "minimized")
    restored = Modely.import_keras(path)

    assert [tensor.name for tensor in restored.inputs] == ["minimized_x"]  # type: ignore
    result = restored(inputs, training=False)  # type: ignore
    assert set(result) == {"minimized_out"}
    np.testing.assert_allclose(
        to_numpy(result["minimized_out"]), to_numpy(expected), rtol=1e-5, atol=1e-5
    )


@requires_onnx_export
def test_export_onnx_leaves_minimizers_out(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _minimized_model()
    # One sample: the exported graph is run one sample at a time.
    inputs = {name: value[:1] for name, value in _minimized_inputs().items()}
    expected = model(dict(inputs))["minimized_out"]

    model.export_onnx(tmp_path, "minimized")
    path = tmp_path / "minimized.onnx"
    # validate_onnx feeds every input of the graph: the target is not one of them.
    result = Modely.validate_onnx(path, inputs, return_dict=True)

    assert list(result) == ["minimized_out"]
    np.testing.assert_allclose(
        to_numpy(result["minimized_out"]),  # type: ignore
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,  # type: ignore
    )


# ---------------------------------------------------------------------------
# Names and weights of a loaded model
# ---------------------------------------------------------------------------


def _fir_model(name, gain):
    """``out = Fir(x)``, ``x`` a window of 2, with every kernel entry equal to ``gain``."""
    x = Input(f"{name}_x", sw=2)
    fir = Fir(features=1, bias=False)(x)  # left unnamed
    model = Modely(name, inputs=[x], outputs=[Output(f"{name}_out", fir)]).build()
    assert fir.kernel is not None
    fir.kernel.assign(np.full((2, 1), gain, dtype=np.float32))
    return model, fir


_WINDOW_OF_ONES = np.ones((1, 2, 1), dtype=np.float32)


def test_a_loaded_layer_keeps_its_name_and_its_own_weights(tmp_path):
    model, ours = _fir_model("apart", 3.0)
    model.save(tmp_path / "apart")

    loaded = Modely.load(tmp_path / "apart")
    loaded_fir = next(node for node in loaded.order if isinstance(node, Fir))
    # One name, two layers: a layer is told apart by the object it comes from.
    assert loaded_fir.name == ours.name
    assert ours.kernel is not None
    ours.kernel.assign(np.ones((2, 1), dtype=np.float32))

    x = model.inputs[0]
    both = Modely(
        "apart_both", inputs=[x], outputs=[Output("apart_sum", ours + loaded([x]))]
    ).build()
    assert both.inference_model is not None
    assert len(both.inference_model.weights) == 2
    # Keras names the layers of one model uniquely, so one of them is suffixed.
    keras_names = [layer.name for layer in both.inference_model.layers]
    assert len(keras_names) == len(set(keras_names))
    result = both({"apart_x": _WINDOW_OF_ONES})["apart_sum"]
    # Ours sums the window of ones, the loaded Fir keeps its 3.0 per sample.
    np.testing.assert_allclose(to_numpy(result), [[[2.0 + 6.0]]])


def test_a_reloaded_model_stays_apart_from_its_original(tmp_path):
    # Once tied by name: composed with its original, the copy computed with
    # the original's layer from then on, and saved the original's weights.
    original, fir = _fir_model("reload", 1.0)
    original.save(tmp_path / "reload")
    copy = Modely.load(tmp_path / "reload")
    assert fir.kernel is not None
    fir.kernel.assign(np.full((2, 1), 2.0, dtype=np.float32))  # trained on

    z = Input("reload_z", sw=2)
    both = Modely(
        "reload_both",
        inputs=[z],
        outputs=[
            Output("reload_original", original([z])),
            Output("reload_copy", copy([z])),
        ],
    ).build()
    result = both({"reload_z": _WINDOW_OF_ONES})

    np.testing.assert_allclose(to_numpy(result["reload_original"]), [[[4.0]]])
    np.testing.assert_allclose(to_numpy(result["reload_copy"]), [[[2.0]]])
    np.testing.assert_allclose(
        to_numpy(copy({"reload_x": _WINDOW_OF_ONES})["reload_out"]), [[[2.0]]]
    )
    copy.build()
    np.testing.assert_allclose(
        to_numpy(copy({"reload_x": _WINDOW_OF_ONES})["reload_out"]), [[[2.0]]]
    )
    copy.save(tmp_path / "copy")
    again = Modely.load(tmp_path / "copy")
    np.testing.assert_allclose(
        to_numpy(again({"reload_x": _WINDOW_OF_ONES})["reload_out"]), [[[2.0]]]
    )


def test_one_file_loaded_twice_gives_two_models(tmp_path):
    model, _ = _fir_model("twice", 1.0)
    model.save(tmp_path / "twice")
    first, second = Modely.load(tmp_path / "twice"), Modely.load(tmp_path / "twice")
    second_fir = next(node for node in second.order if isinstance(node, Fir))
    assert second_fir.kernel is not None
    second_fir.kernel.assign(np.full((2, 1), 5.0, dtype=np.float32))

    np.testing.assert_allclose(
        to_numpy(first({"twice_x": _WINDOW_OF_ONES})["twice_out"]), [[[2.0]]]
    )
    np.testing.assert_allclose(
        to_numpy(second({"twice_x": _WINDOW_OF_ONES})["twice_out"]), [[[10.0]]]
    )


def test_a_loop_closed_over_a_generated_layer_reloads_beside_its_original(tmp_path):
    # Once a failure: a load renamed the body's layer while the Loop still
    # named it as the output it feeds back.
    s = Input("bare_s", sw=1)
    gain = Parameter("bare_gain", value=[2.0])
    # A body output, unnamed.
    bare = Fir(features=1, kernel=gain, bias=False)(s)
    body = Modely("bare_body", inputs=[s], outputs=[bare]).build()
    seed = Input("bare_seed", seq=4, sw=1)
    loop = Loop(f=body, callback={s: bare}, collect=False, init={s: seed})()
    model = Modely(
        "bare_top", inputs=[seed], outputs=[Output("bare_out", loop)]
    ).build()
    seed_values = np.zeros((1, 4, 1, 1), dtype=np.float32)
    seed_values[:, 0] = 1.0
    expected = to_numpy(model({"bare_seed": seed_values})["bare_out"])
    model.save(tmp_path / "bare")

    restored = Modely.load(tmp_path / "bare")

    np.testing.assert_allclose(
        to_numpy(restored({"bare_seed": seed_values})["bare_out"]), expected
    )
    np.testing.assert_allclose(expected, [[[16.0]]])


def test_load_without_weights_initializes_the_saved_architecture(tmp_path):
    model, _ = _fir_model("fresh", 7.0)
    model.save(tmp_path / "fresh")

    restored = Modely.load(tmp_path / "fresh", weights=False)
    restored_fir = next(node for node in restored.order if isinstance(node, Fir))

    assert (tmp_path / "fresh" / "model.weights.h5").exists()
    assert not np.allclose(to_numpy(restored_fir.kernel), 7.0)
    reloaded = Modely.load(tmp_path / "fresh")
    reloaded_fir = next(node for node in reloaded.order if isinstance(node, Fir))
    np.testing.assert_allclose(to_numpy(reloaded_fir.kernel), 7.0)


def test_names_generated_after_a_load_avoid_the_loaded_ones(tmp_path):
    from nnodely.core import dag

    x = Input("upcoming_x", sw=2)
    fir = Fir(features=1, name="upcoming_fir")(x)
    model = Modely("upcoming", inputs=[x], outputs=[Output("upcoming_out", fir)])
    model.build().save(tmp_path / "upcoming")
    # Saved by a process that ran further: its Fir has the name this one makes next.
    upcoming = f"Fir{dag._node_counter + 1}"
    config = tmp_path / "upcoming" / "model.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    for node in data["nodes"]:
        if node["config"]["name"] == "upcoming_fir":
            node["config"]["name"] = upcoming
    config.write_text(json.dumps(data), encoding="utf-8")

    loaded = Modely.load(tmp_path / "upcoming")

    assert upcoming in {node.name for node in loaded.order}
    assert Fir(features=1).name != upcoming


def test_a_model_saved_without_weights_loads_as_freshly_built(tmp_path):
    x = Input("architecture_x", sw=2)
    gain = Parameter("architecture_gain", value=[2.0])
    fir = Fir(features=1, bias=False, name="architecture_fir")(x)
    model = Modely(
        "architecture", inputs=[x], outputs=[Output("architecture_out", fir * gain)]
    ).build()
    assert fir.kernel is not None and gain.param is not None
    fir.kernel.assign(np.full((2, 1), 5.0, dtype=np.float32))  # as if trained
    gain.param.assign([7.0])

    path = tmp_path / "architecture"
    model.save(path)
    model.save(path, weights=False)  # the architecture alone, from now on
    assert not (path / "model.weights.h5").exists()

    restored = {node.name: node for node in Modely.load(path).order}
    restored_gain, restored_fir = (
        restored["architecture_gain"],
        restored["architecture_fir"],
    )
    assert isinstance(restored_gain, Parameter) and isinstance(restored_fir, Fir)
    # As declared again: the parameter from its value, the Fir from a new draw.
    np.testing.assert_allclose(to_numpy(restored_gain.param), [2.0])
    assert not np.allclose(to_numpy(restored_fir.kernel), 5.0)


def test_a_linear_initializer_object_is_saved_as_its_config(tmp_path):
    x = Input("initializer_x", sw=1)
    linear = Linear(
        features=2,
        kernel=keras.initializers.Constant(0.5),
        bias=keras.initializers.Constant(-1.0),
        name="initializer_linear",
    )([x])
    model = Modely(
        "initializer", inputs=[x], outputs=[Output("initializer_out", linear)]
    ).build()
    model.save(tmp_path / "initializer", weights=False)

    restored = Modely.load(tmp_path / "initializer")
    restored_linear = next(node for node in restored.order if isinstance(node, Linear))

    np.testing.assert_allclose(to_numpy(restored_linear.kernel), np.full((1, 2), 0.5))
    np.testing.assert_allclose(to_numpy(restored_linear.bias), [-1.0, -1.0])


# ---------------------------------------------------------------------------
# Fir and Linear computing with Parameters as kernel and bias
# ---------------------------------------------------------------------------


def _parameter_weights_model():
    """A Fir and a Linear whose kernels and biases are all Parameters."""
    x = Input("weights_x", dim=2, sw=3)
    parameters = [
        Parameter("weights_fir_kernel", value=[[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]),
        Parameter("weights_fir_bias", value=[0.1, -0.2]),
        Parameter("weights_linear_kernel", value=[[1.0, 2.0], [3.0, 4.0]]),
        Parameter("weights_linear_bias", value=[0.5, -0.5]),
    ]
    fir = Fir(features=1, kernel=parameters[0], bias=parameters[1], name="weights_fir")(
        [x]
    )
    linear = Linear(
        features=2,
        kernel=parameters[2],
        bias=parameters[3],
        name="weights_linear",
    )([fir])
    model = Modely(
        "weights_model", inputs=[x], outputs=[Output("weights_out", linear)]
    ).build()
    return model, parameters


# Two elements of three samples each: [[0, 0.25, 0.5], [0.75, 1, 1.25]].
_WEIGHTS_INPUTS = {
    "weights_x": np.arange(6, dtype=np.float32).reshape(1, 2, 3).transpose(0, 2, 1)
    / 4.0
}


def _as_if_trained(parameters):
    """Move every Parameter away from its declared value, as training would."""
    for parameter in parameters:
        assert parameter.param is not None
        parameter.param.assign(to_numpy(parameter.param) * 2.0 + 0.1)


def test_fir_and_linear_computing_with_parameters_save_load_and_export(tmp_path):
    model, parameters = _parameter_weights_model()
    declared = to_numpy(model(_WEIGHTS_INPUTS)["weights_out"])
    # Fir: [2.0 + 0.1, 2.25 - 0.2]; Linear: [2.1, 2.05] @ [[1, 2], [3, 4]] + bias.
    np.testing.assert_allclose(declared, [[[8.75, 11.9]]], rtol=1e-5)
    _as_if_trained(parameters)
    trained = to_numpy(model(_WEIGHTS_INPUTS)["weights_out"])
    assert not np.allclose(trained, declared)

    model.save(tmp_path / "weights")
    restored = Modely.load(tmp_path / "weights")
    assert graph_signature(restored) == graph_signature(model)
    np.testing.assert_allclose(
        to_numpy(restored(_WEIGHTS_INPUTS)["weights_out"]),
        trained,
        rtol=1e-5,
        atol=1e-5,
    )

    # Saved as the architecture alone, the Parameters come back as declared.
    model.save(tmp_path / "weights_architecture", weights=False)
    fresh = Modely.load(tmp_path / "weights_architecture")
    np.testing.assert_allclose(
        to_numpy(fresh(_WEIGHTS_INPUTS)["weights_out"]),
        declared,
        rtol=1e-5,
        atol=1e-5,
    )

    model.export_keras(tmp_path, "weights")
    imported = Modely.import_keras(tmp_path / "weights.keras")
    np.testing.assert_allclose(
        to_numpy(imported(_WEIGHTS_INPUTS)["weights_out"]),  # type: ignore
        trained,
        rtol=1e-5,
        atol=1e-5,
    )


@requires_onnx_export
def test_fir_and_linear_computing_with_parameters_export_to_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    model, parameters = _parameter_weights_model()
    _as_if_trained(parameters)
    expected = to_numpy(model(_WEIGHTS_INPUTS)["weights_out"])

    model.export_onnx(tmp_path, "weights")
    result = Modely.validate_onnx(
        tmp_path / "weights.onnx", _WEIGHTS_INPUTS, return_dict=True
    )

    assert isinstance(result, dict)
    np.testing.assert_allclose(
        to_numpy(result["weights_out"]), expected, rtol=1e-5, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Fuzzify with any layer, and the membership functions on their own
# ---------------------------------------------------------------------------


def _generalized_fuzzify_model():
    # Custom functions defined here, as a user would, not registered by hand.
    class Square(keras.layers.Layer):
        def call(self, x):
            return x * x

    x = Input("general_fuzzy_x", sw=1)
    squares = Fuzzify([0.0, 1.0], function=Square, name="general_fuzzy_square")([x])
    mixed = Fuzzify(
        [0.0, 0.5, 1.0],
        function=[Rectangular, Triangular, Gaussian],
        bounds=True,
        name="general_fuzzy_mixed",
    )([x])
    waves = Fuzzify([0.0, 1.0], function=Sin, name="general_fuzzy_sin")([x])
    bell = Gaussian(sigma=0.3, name="general_bell")([x - 0.2])
    leaning = Triangular(radius=(0.5, 1.0), name="general_leaning")([x - 0.5])
    return Modely(
        "general_fuzzy",
        inputs=[x],
        outputs=[
            Output("general_mixed", mixed),
            Output("general_waves", waves),
            Output("general_bell_out", bell),
            Output("general_leaning_out", leaning),
            Output("general_square", squares),
        ],
    ).build()


_GENERAL_OUTPUTS = (
    "general_mixed",
    "general_waves",
    "general_bell_out",
    "general_leaning_out",
    "general_square",
)


def test_generalized_fuzzify_saves_loads_and_exports_to_keras(tmp_path):
    model = _generalized_fuzzify_model()
    data = {
        "general_fuzzy_x": np.linspace(-0.5, 1.5, 9, dtype=np.float32).reshape(
            -1, 1, 1
        ),
    }
    before = model(data)

    model.save(tmp_path / "general_fuzzy")
    restored = Modely.load(tmp_path / "general_fuzzy")
    assert graph_signature(restored) == graph_signature(model)
    after = restored(data)
    model.export_keras(tmp_path, "general_fuzzy")
    imported = Modely.import_keras(tmp_path / "general_fuzzy.keras")
    exported = imported(data)  # type: ignore
    for name in _GENERAL_OUTPUTS:
        for result in (after, exported):
            np.testing.assert_allclose(
                to_numpy(result[name]), to_numpy(before[name]), rtol=1e-6, atol=1e-7
            )


@requires_onnx_export
def test_generalized_fuzzify_exports_to_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _generalized_fuzzify_model()
    data = {
        "general_fuzzy_x": np.full((1, 1, 1), 0.8, dtype=np.float32),
    }
    before = model(data)

    model.export_onnx(tmp_path, "general_fuzzy")
    result = Modely.validate_onnx(
        tmp_path / "general_fuzzy.onnx", data, return_dict=True
    )

    assert isinstance(result, dict)
    for name in _GENERAL_OUTPUTS:
        np.testing.assert_allclose(
            to_numpy(result[name]), to_numpy(before[name]), rtol=1e-5, atol=1e-6
        )


def test_fuzzify_sharing_a_keras_layer_saves_and_loads(tmp_path):
    # The instance was wrapped once per center, and the saved model could not
    # be loaded: "expected 1 variables, but received 0".
    class Scaled(keras.layers.Layer):
        def build(self, input_shape):
            self.gain = self.add_weight(shape=(), initializer="ones", name="gain")

        def call(self, x):
            return self.gain * x

    x = Input("scaled_fuzzy_x", sw=1)
    scaled = Scaled()
    model = Modely(
        "scaled_fuzzy",
        inputs=[x],
        outputs=[
            Output(
                "scaled_fuzzy_out",
                Fuzzify([0.0, 1.0, 2.0], function=scaled, normalize=False)([x]),
            )
        ],
    ).build()
    scaled.gain.assign(3.0)
    data = {"scaled_fuzzy_x": np.full((1, 1, 1), 2.0, dtype=np.float32)}

    model.save(tmp_path / "scaled_fuzzy")
    restored = Modely.load(tmp_path / "scaled_fuzzy")
    np.testing.assert_allclose(
        to_numpy(restored(data)["scaled_fuzzy_out"]).ravel(), [6.0, 3.0, 0.0]
    )


def _moving_centers_model():
    x = Input("moving_x", sw=1)
    u = Input("moving_u", sw=2)
    centers = Parameter("moving_centers", value=[0.0, 0.5, 1.0])
    computed = Fir(features=3, bias=False)(u)  # each sample its own
    middle = Parameter("moving_middle", value=[0.4])
    # Widths trained with the model, as Parameters.
    width = Parameter("moving_width", value=[0.3, 0.6])
    sigma = Parameter("moving_sigma", value=[0.4])
    model = Modely(
        "moving",
        inputs=[x, u],
        outputs=[
            Output("moving_triangles", Fuzzify(centers, function=Triangular)([x])),
            Output("moving_bells", Fuzzify(centers, function=Gaussian)([x])),
            Output("moving_waves", Fuzzify(computed, function=Sin)([x])),
            Output(
                "moving_listed", Fuzzify([0.0, middle, 1.0], function=Triangular)([x])
            ),
            Output(
                "moving_bounded",
                Fuzzify(computed, function=Triangular, bounds=True)([x]),
            ),
            Output(
                "moving_inner",
                Fuzzify([0.0, middle, 1.0], function=Gaussian, bounds=(0.2, 0.9))([x]),
            ),
            Output(
                "moving_widths",
                Fuzzify(centers, function=Triangular(radius=width))([x]),
            ),
            Output("moving_sigma_out", Gaussian(sigma=sigma)([x - 0.5])),
        ],
    ).build()
    # As if trained: the centers and the widths moved from where they started.
    centers.param.assign(np.array([-0.2, 0.6, 1.3], dtype=np.float32))  # type: ignore
    width.param.assign(np.array([0.5, 0.8], dtype=np.float32))  # type: ignore
    sigma.param.assign(np.array([0.7], dtype=np.float32))  # type: ignore
    return model


_MOVING_OUTPUTS = (
    "moving_triangles",
    "moving_bells",
    "moving_waves",
    "moving_listed",
    "moving_bounded",
    "moving_inner",
    "moving_widths",
    "moving_sigma_out",
)
_MOVING_DATA = {
    "moving_x": np.linspace(-0.5, 1.5, 5, dtype=np.float32).reshape(-1, 1, 1),
    "moving_u": np.linspace(-1.0, 1.0, 10, dtype=np.float32).reshape(5, 2, 1),
}


def test_moving_fuzzify_centers_save_load_and_export_to_keras(tmp_path):
    model = _moving_centers_model()
    before = model(_MOVING_DATA)

    model.save(tmp_path / "moving")
    restored = Modely.load(tmp_path / "moving")
    assert graph_signature(restored) == graph_signature(model)
    model.export_keras(tmp_path, "moving")
    imported = Modely.import_keras(tmp_path / "moving.keras")
    for result in (restored(_MOVING_DATA), imported(_MOVING_DATA)):  # type: ignore
        for name in _MOVING_OUTPUTS:
            np.testing.assert_allclose(
                to_numpy(result[name]), to_numpy(before[name]), rtol=1e-6, atol=1e-7
            )


@pytest.mark.slow
@requires_onnx_export
def test_moving_fuzzify_centers_export_to_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _moving_centers_model()
    data = {name: values[1:2] for name, values in _MOVING_DATA.items()}
    before = model(data)

    model.export_onnx(tmp_path, "moving")
    result = Modely.validate_onnx(tmp_path / "moving.onnx", data, return_dict=True)

    assert isinstance(result, dict)
    for name in _MOVING_OUTPUTS:
        np.testing.assert_allclose(
            to_numpy(result[name]), to_numpy(before[name]), rtol=1e-5, atol=1e-6
        )


@requires_onnx_export
def test_onnx_outputs_of_one_shape_keep_their_names(tmp_path):
    # Once swapped on torch: its exporter drops the names and writes the
    # outputs in the order of their names, which were paired off in the order
    # they were declared.
    pytest.importorskip("onnxruntime")
    x = Input("same_shape_x", sw=1)
    names = ["zeta", "alpha", "mid", "beta"]
    model = Modely(
        "same_shape",
        inputs=[x],
        outputs=[
            Output(name, x * float(index + 1)) for index, name in enumerate(names)
        ],
    ).build()

    model.export_onnx(tmp_path, "same_shape")
    result = Modely.validate_onnx(
        tmp_path / "same_shape.onnx",
        {"same_shape_x": np.ones((1, 1, 1), dtype=np.float32)},
        return_dict=True,
    )

    assert isinstance(result, dict)
    for index, name in enumerate(names):
        np.testing.assert_allclose(to_numpy(result[name]).ravel(), [index + 1.0])

    # Listed as model.outputs lists them: the exporters sort them by name.
    listed = Modely.validate_onnx(
        tmp_path / "same_shape.onnx",
        {"same_shape_x": np.ones((1, 1, 1), dtype=np.float32)},
    )
    assert [float(to_numpy(value).ravel()[0]) for value in listed] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_a_loaded_model_reads_the_window_the_saved_one_read(tmp_path):
    # The saved graph has the outputs alone, which read 3 samples; the model
    # was built reading 5 for its minimizer, and the loaded one reads the same
    # data as it did.
    x = Input("saved_window_x", sw=5)
    out = Output("saved_window_out", Fir(features=1)(TimeRange(2, 5)(x)))
    model = Modely("saved_window", inputs=[x], outputs=[out])
    model.minimize("fit", out, Fir(features=1)(x))
    model.build()
    data = {
        "saved_window_x": np.random.default_rng(0)
        .normal(size=(2, 5, 1))
        .astype(np.float32)
    }

    model.save(tmp_path / "saved_window")
    restored = Modely.load(tmp_path / "saved_window")
    assert restored.train_inputs[0].past == 5
    np.testing.assert_allclose(
        to_numpy(restored(data)["saved_window_out"]),
        to_numpy(model(data)["saved_window_out"]),
        rtol=1e-6,
    )


def test_a_parameter_read_by_several_models_loads_as_one(tmp_path):
    # Read by a Loop body, by a block and by the outer model, the gain - and
    # the kernel of the Fir both use - is one object, written in the file of
    # each model; it was loaded as one object per file.
    gain = Parameter("one_gain", value=2.0)
    fir = Fir(features=1, bias=False, name="one_fir")
    s = Input("one_s", sw=1)
    body = Modely(
        "one_body",
        inputs=[s],
        outputs=[Output("one_next", s * gain + fir(s))],
    ).build()
    a, c = Input("one_a", seq=3, sw=1), Input("one_c", sw=1)
    loop = Loop(f=body, callback={s: "one_next"}, init={s: a})()
    model = Modely(
        "one_model",
        inputs=[a, c],
        outputs=[
            Output("one_loop", loop),
            Output("one_block", body([c])),
            Output("one_out", c * gain + fir(c)),
        ],
    ).build()
    assert [p.name for p in model.parameters] == ["one_gain", "one_fir_kernel"]
    data = {
        "one_a": np.full((2, 3, 1, 1), 0.5, dtype=np.float32),
        "one_c": np.full((2, 1, 1), 1.5, dtype=np.float32),
    }

    model.save(tmp_path / "one")
    restored = Modely.load(tmp_path / "one")
    assert [p.name for p in restored.parameters] == ["one_gain", "one_fir_kernel"]
    loaded_gain = restored.parameters[0]
    loaded_loop = next(node for node in restored.order if isinstance(node, Loop))
    assert any(node is loaded_gain for node in loaded_loop.f.order)
    for name in ("one_loop", "one_block", "one_out"):
        np.testing.assert_allclose(
            to_numpy(restored(data)[name]), to_numpy(model(data)[name]), rtol=1e-6
        )

    # Frozen once, it is frozen everywhere, and saved so.
    loaded_gain.freeze()
    restored.save(tmp_path / "two")
    again = Modely.load(tmp_path / "two")
    assert [p.trainable for p in again.parameters] == [False, True]


# ---------------------------------------------------------------------------
# Keras export of models whose weights are reached from several Keras graphs
# ---------------------------------------------------------------------------


def _keras_two_loops():
    s = Input("ks2_s", sw=1)
    nxt = Output("ks2_next", Linear(features=1, name="ks2_lin")(s))
    body = Modely("ks2_body", inputs=[s], outputs=[nxt]).build()
    a = Input("ks2_a", seq=3, sw=1)
    rolled = Loop(f=body, callback={s: nxt}, collect=False, init={s: a})()
    fixed = Loop(f=body, callback={s: nxt}, collect=False, init={s: a}, length=2)()
    model = Modely(
        "ks2", inputs=[a], outputs=[Output("ks2_o1", rolled), Output("ks2_o2", fixed)]
    ).build()
    return model, {"ks2_a": (2, 3, 1, 1)}


def _keras_parameter_inside_and_outside():
    gain = Parameter("ksp_gain", value=1.5)
    s = Input("ksp_s", sw=1)
    body = Modely(
        "ksp_body", inputs=[s], outputs=[Output("ksp_next", s * gain)]
    ).build()
    a = Input("ksp_a", seq=3, sw=1)
    loop = Loop(f=body, callback={s: "ksp_next"}, init={s: a})()
    model = Modely(
        "ksp",
        inputs=[a],
        outputs=[Output("ksp_loop", loop), Output("ksp_out", a * gain)],
    ).build()
    return model, {"ksp_a": (2, 3, 1, 1)}


def _keras_block_and_loop():
    s = Input("ksb_s", sw=1)
    nxt = Output("ksb_next", Linear(features=1, name="ksb_lin")(s))
    body = Modely("ksb_body", inputs=[s], outputs=[nxt]).build()
    a, c = Input("ksb_a", seq=3, sw=1), Input("ksb_c", sw=1)
    loop = Loop(f=body, callback={s: nxt}, init={s: a})()
    model = Modely(
        "ksb",
        inputs=[a, c],
        outputs=[Output("ksb_loop", loop), Output("ksb_block", body([c]))],
    ).build()
    return model, {"ksb_a": (2, 3, 1, 1), "ksb_c": (2, 1, 1)}


def _keras_two_differentiate():
    x = Input("ksd_x", sw=1)
    relation = Linear(features=1, name="ksd_lin")(x) ** 2
    model = Modely(
        "ksd",
        inputs=[x],
        outputs=[
            Output("ksd_first", Differentiate(respect_to=x)(relation)),
            Output("ksd_second", Differentiate(order=2, respect_to=x)(relation)),
        ],
    ).build()
    return model, {"ksd_x": (2, 1, 1)}


def _keras_loop_in_a_body():
    gain = Parameter("ksn_gain", value=0.9)
    si = Input("ksn_si", sw=1)
    inner = Modely(
        "ksn_inner", inputs=[si], outputs=[Output("ksn_inner_next", si * gain)]
    ).build()
    m = Input("ksn_m", sw=1)
    inner_loop = Loop(
        f=inner,
        callback={si: "ksn_inner_next"},
        collect=False,
        init={si: m},
        length=3,
    )()
    mid = Modely(
        "ksn_mid", inputs=[m], outputs=[Output("ksn_mid_out", inner_loop)]
    ).build()
    m0 = Input("ksn_m0", sw=1)
    outer = Loop(f=mid, callback={m: "ksn_mid_out"}, init={m: m0}, length=2)()
    model = Modely(
        "ksn",
        inputs=[m0],
        outputs=[Output("ksn_out", outer), Output("ksn_direct", m0 * gain)],
    ).build()
    return model, {"ksn_m0": (2, 1, 1)}


def _keras_stateful_and_custom_body():
    from nnodely import KerasFunction

    class Scale(keras.layers.Layer):
        def build(self, input_shape):
            self.k = self.add_weight(shape=(), initializer="ones", name="k")

        def call(self, x):
            return self.k * x

    scale = Scale()
    s, u = Input("ksk_s", sw=1), Input("ksk_u", sw=1)
    body = Modely(
        "ksk_body",
        inputs=[s, u],
        outputs=[
            Output(
                "ksk_next",
                KerasFunction(scale)([BatchNorm()(s)]) + u,
            )
        ],
    ).build()
    s0, us = Input("ksk_s0", sw=1), Input("ksk_us", seq=4, sw=1)
    loop = Loop(f=body, callback={s: "ksk_next"}, init={s: s0})({u: us})
    model = Modely(
        "ksk",
        inputs=[s0, us],
        outputs=[
            Output("ksk_loop", loop),
            Output("ksk_scaled", KerasFunction(scale)([s0])),
        ],
    ).build()
    return model, {"ksk_s0": (2, 1, 1), "ksk_us": (2, 4, 1, 1)}


@pytest.mark.parametrize(
    "make",
    [
        _keras_two_loops,
        _keras_parameter_inside_and_outside,
        _keras_block_and_loop,
        _keras_two_differentiate,
        _keras_loop_in_a_body,
        _keras_stateful_and_custom_body,
    ],
)
def test_keras_export_of_weights_reached_from_several_graphs(make, tmp_path):
    # A Keras file once saved such a weight once but could not rebuild it:
    # "expected 1 variables, but received 0", or a broken graph at the call.
    model, shapes = make()
    rng = np.random.default_rng(0)
    # Values unlike any initializer - moving statistics included - so a
    # weight left at its initial value after the import would show. Constants
    # are left alone: a number's Constant is shared by every model.
    assert model.inference_model is not None
    for weight in model.inference_model.weights:
        if weight.trainable or "moving" in weight.name:
            weight.assign(rng.uniform(0.5, 1.5, size=weight.shape).astype("float32"))
    data = {
        name: rng.normal(size=shape).astype(np.float32)
        for name, shape in shapes.items()
    }
    before = model(data)

    model.export_keras(tmp_path, "shared")
    imported = Modely.import_keras(tmp_path / "shared.keras")
    after = imported(data)  # type: ignore[operator]
    for name, value in before.items():
        np.testing.assert_allclose(
            to_numpy(after[name]), to_numpy(value), rtol=1e-5, atol=1e-6
        )

    # Exported again from its nnodely save, it gives the same file back.
    model.save(tmp_path / "saved")
    Modely.load(tmp_path / "saved").export_keras(tmp_path, "reloaded")
    again = Modely.import_keras(tmp_path / "reloaded.keras")(data)  # type: ignore[operator]
    for name, value in before.items():
        np.testing.assert_allclose(
            to_numpy(again[name]), to_numpy(value), rtol=1e-5, atol=1e-6
        )


def test_keras_export_follows_a_dynamic_rollout(tmp_path):
    s, u = Input("ksdyn_s", sw=1), Input("ksdyn_u", sw=1)
    body = Modely(
        "ksdyn_body",
        inputs=[s, u],
        outputs=[Output("ksdyn_next", s * 0.5 + u)],
    ).build()
    s0, us = Input("ksdyn_s0", sw=1), Input("ksdyn_us", seq=-1, sw=1)
    loop = Loop(f=body, callback={s: "ksdyn_next"}, init={s: s0})({u: us})
    model = Modely(
        "ksdyn", inputs=[s0, us], outputs=[Output("ksdyn_out", loop)]
    ).build()

    model.export_keras(tmp_path, "ksdyn")
    imported = Modely.import_keras(tmp_path / "ksdyn.keras")
    for steps in (2, 7):
        data = {
            "ksdyn_s0": np.ones((1, 1, 1), dtype=np.float32),
            "ksdyn_us": np.ones((1, steps, 1, 1), dtype=np.float32),
        }
        expected = to_numpy(model(data)["ksdyn_out"])
        assert expected.shape == (1, steps, 1, 1)
        np.testing.assert_allclose(
            to_numpy(imported(data)["ksdyn_out"]),  # type: ignore[operator]
            expected,
            rtol=1e-6,
        )


def test_an_input_read_directly_and_through_a_window_round_trips(tmp_path):
    ## Read directly x is its newest sample, whatever window the data holds
    x = Input("rt_direct_x", sw=3)
    taps = Parameter("rt_direct_taps", value=[[1.0], [10.0], [100.0]])
    filtered = Fir(features=1, kernel=taps, bias=False)([x])
    model = Modely(
        "rt_direct",
        inputs=[x],
        outputs=[Output("rt_direct_out", TimeSelect(-1)(x) * 2.0 + filtered)],
    ).build()
    data = {"rt_direct_x": np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)}
    expected = [[[3.0 * 2.0 + 1.0 + 20.0 + 300.0]]]
    np.testing.assert_allclose(to_numpy(model(data)["rt_direct_out"]), expected)

    model.save(tmp_path / "saved")
    loaded = Modely.load(tmp_path / "saved")
    np.testing.assert_allclose(to_numpy(loaded(data)["rt_direct_out"]), expected)

    model.export_keras(tmp_path, "rt_direct")
    imported = Modely.import_keras(str(tmp_path / "rt_direct.keras"))
    np.testing.assert_allclose(
        to_numpy(imported(data)["rt_direct_out"]),  # type: ignore[index]
        expected,
    )


def test_a_parameter_declared_by_its_sizes_round_trips(tmp_path):
    ## Without time, its variable holds the value as declared, (2, 3)
    x = Input("rt_sizes_x", sw=1)
    gain = Parameter("rt_sizes_gain", dim=(2, 3))
    model = Modely(
        "rt_sizes", inputs=[x], outputs=[Output("rt_sizes_out", x * gain)]
    ).build()
    assert gain.shape.dimensions == ((), None, (2, 3))
    assert gain.param is not None and tuple(gain.param.shape) == (2, 3)

    model.save(tmp_path / "saved")
    loaded = Modely.load(tmp_path / "saved")
    data = {"rt_sizes_x": np.full((2, 1, 1), 3.0, dtype=np.float32)}
    np.testing.assert_allclose(
        to_numpy(loaded(data)["rt_sizes_out"]),
        to_numpy(model(data)["rt_sizes_out"]),
    )
