import keras
import numpy as np
import pytest

from conftest import to_numpy
from nnodely import DataLoader, Modely, Input, Output, Fir, Linear, Loop


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def window_size():
    return 5


def test_basic_model_build_and_inference(batch_size, window_size):
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    x_stream = x.sw(window_size)
    y_stream = y.sw(window_size)

    result_fir = Fir(out_features=2)([x_stream + y_stream])
    x_out = Output("x_pred", result_fir)

    model = Modely("model1", inputs=[x, y], outputs=[x_out])
    model.build()

    dummy_x = np.ones((batch_size, 1, window_size), dtype=np.float32)
    dummy_y = np.ones((batch_size, 1, window_size), dtype=np.float32)

    result = model([dummy_x, dummy_y])

    assert "x_pred" in result
    assert result["x_pred"].shape == (batch_size, 2, 1)


def test_model_inference_with_composed_model(batch_size, window_size):
    # ------- Model definition and building -------
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    x_stream = x.sw(window_size)
    y_stream = y.sw(window_size)

    fir = Fir(out_features=2)
    result_fir = fir([x_stream + y_stream])

    x_out = Output("x_pred", result_fir)
    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    # ------- Model composition -------
    z = Input("z", dim=1)
    z_stream = z.sw(window_size)
    z_fir = Fir(out_features=1)(model1([z_stream, z_stream]))
    z_out = Output("z_pred", z_fir)
    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    # ------- Model inference -------
    dummy_input_x = np.ones((batch_size, 1, window_size), dtype=np.float32)
    dummy_input_y = np.ones((batch_size, 1, window_size), dtype=np.float32)
    dummy_input_z = np.ones((batch_size, 1, window_size), dtype=np.float32)

    result1 = model1([dummy_input_x, dummy_input_y])
    assert "x_pred" in result1
    assert result1["x_pred"].shape == (batch_size, 2, 1)

    result2 = model2([dummy_input_z])
    assert "z_pred" in result2
    assert result2["z_pred"].shape == (batch_size, 1, 1)


def test_inference_adds_batch_axis_to_backend_tensor():
    x = Input("backend_tensor_x", dim=1)
    output = Output("backend_tensor_out", x * 2.0)
    model = Modely("backend_tensor_model", inputs=[x], outputs=[output]).build()

    value = keras.ops.convert_to_tensor(np.array([[3.0]], dtype=np.float32))
    result = model({"backend_tensor_x": value})

    assert result["backend_tensor_out"].shape == (1, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["backend_tensor_out"]),
        np.array([[[6.0]]], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Inference runs the model as declared, without its minimizers
# ---------------------------------------------------------------------------


def _minimized_model(name):
    """A model whose minimizers bring inputs and outputs of their own.

    The target is an input only the loss reads, and the second minimizer
    regularizes a stream that no declared output exposes.
    """
    x = Input(f"{name}_x", dim=1)
    target = Input(f"{name}_target", dim=1)
    hidden = Linear(out_features=1, name=f"{name}_hidden")(x.last())
    prediction = Output(
        f"{name}_pred", Fir(out_features=1, name=f"{name}_fir")(x.sw(3)) + hidden
    )
    model = Modely(name, inputs=[x, target], outputs=[prediction])
    model.minimize(f"{name}_fit", source=prediction, target=target.last())
    model.minimize(f"{name}_small", source=hidden, target=0.0)
    return model.build()


def _windows(batch=4):
    return np.random.default_rng(0).uniform(size=(batch, 1, 3)).astype(np.float32)


def test_inference_does_not_need_inputs_only_minimizers_read():
    model = _minimized_model("skip_target")
    x = _windows()
    target = np.ones((4, 1, 1), dtype=np.float32)

    assert {node.name for node in model.train_inputs} == {
        "skip_target_x",
        "skip_target_target",
    }
    assert [node.name for node in model.inference_inputs] == ["skip_target_x"]

    result = model({"skip_target_x": x})
    assert model.model is not None
    trained = model.model(
        {"skip_target_x": x, "skip_target_target": target}, training=False
    )
    np.testing.assert_allclose(
        to_numpy(result["skip_target_pred"]),
        to_numpy(trained["skip_target_pred"]),
        rtol=1e-6,
    )
    # A target passed anyway is ignored.
    again = model({"skip_target_x": x, "skip_target_target": target})
    np.testing.assert_allclose(
        to_numpy(again["skip_target_pred"]),
        to_numpy(result["skip_target_pred"]),
        rtol=1e-6,
    )


def test_inference_returns_only_declared_outputs():
    model = _minimized_model("declared_only")
    x = _windows()
    target = np.ones((4, 1, 1), dtype=np.float32)

    # Training exposes both sides of every minimizer...
    assert model.model is not None
    trained = model.model(
        {"declared_only_x": x, "declared_only_target": target}, training=False
    )
    assert set(trained) > {"declared_only_pred"}
    # ...inference only what the model declared.
    assert set(model({"declared_only_x": x})) == {"declared_only_pred"}


def test_positional_inference_takes_only_the_inputs_the_outputs_read():
    model = _minimized_model("positional")

    result = model([_windows()])

    assert set(result) == {"positional_pred"}
    assert result["positional_pred"].shape == (4, 1, 1)


def test_inference_keeps_an_input_that_outputs_and_minimizers_both_read():
    # A reconstruction: the input is both what the model reads and its target.
    x = Input("reconstruct_x", dim=1)
    rebuilt = Output("reconstruct_out", Linear(out_features=1)(x.last()))
    model = Modely("reconstruct", inputs=[x], outputs=[rebuilt])
    model.minimize("reconstruct_fit", source=rebuilt, target=x.last())
    model.build()

    assert [node.name for node in model.inference_inputs] == ["reconstruct_x"]
    result = model({"reconstruct_x": np.ones((2, 1, 1), dtype=np.float32)})
    assert set(result) == {"reconstruct_out"}


def test_inference_runs_with_the_trained_weights():
    model = _minimized_model("trained")
    samples = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    data = DataLoader(
        model, source={"trained_x": samples, "trained_target": 2.0 * samples}
    )
    x = _windows()
    before = to_numpy(model({"trained_x": x})["trained_pred"])

    model.train(train_data=data, epochs=3, batch_size=8, lr=1e-2, printer=None)

    after = to_numpy(model({"trained_x": x})["trained_pred"])
    assert model.model is not None
    trained = model.model(
        {"trained_x": x, "trained_target": np.zeros((4, 1, 1), dtype=np.float32)},
        training=False,
    )
    # Inference and training share the weights, so training moves both.
    assert not np.allclose(after, before)
    np.testing.assert_allclose(after, to_numpy(trained["trained_pred"]), rtol=1e-6)


def test_rollback_inference_does_not_need_minimizer_inputs():
    x = Input("rollback_inference_x", dim=1)
    target = Input("rollback_inference_target", dim=1)
    fir = Fir(out_features=1, use_bias=False, name="rollback_inference_fir")(x.sw(5))
    output = Output("rollback_inference_out", fir + x.last())
    model = Modely("rollback_inference", inputs=[x, target], outputs=[output])
    model.minimize("rollback_inference_fit", source=output, target=target.last())
    model.rollback({x: fir}, steps=3)
    model.build()
    fir.kernel.assign(np.ones((5, 1), dtype=np.float32))

    result = model(
        {"rollback_inference_x": np.arange(1, 6, dtype=np.float32).reshape(1, 1, 5)}
    )

    assert set(result) == {"rollback_inference_out"}
    # 1..5 -> 15, 29, 56 fed back; the last step reads 56 + x.last() = 29 + 56.
    np.testing.assert_allclose(
        to_numpy(result["rollback_inference_out"]), np.array([[[85.0]]])
    )


def test_loop_rolls_out_a_body_trained_with_its_own_minimizers():
    state = Input("minimized_body_state", dim=1)
    relation = Linear(out_features=1, use_bias=False, name="minimized_body_linear")(
        state.last()
    )
    next_state = Output("minimized_body_next", relation)
    body = Modely("minimized_body", inputs=[state], outputs=[next_state])
    body.minimize(
        "minimized_body_fit",
        source=next_state,
        target=Input("minimized_body_target", dim=1).last(),
    )
    body.build()
    assert relation.kernel is not None
    relation.kernel.assign(np.full((1, 1), 2.0, dtype=np.float32))

    # The loop runs the body as declared: its target is not an input of the loop.
    seed = Input("minimized_body_seed", dim=1, seq=4)
    loop = Loop(
        f=body, callback={state: next_state}, initial={state: seed}, collect=False
    )
    model = Modely(
        "minimized_loop", inputs=[seed], outputs=[Output("minimized_loop_out", loop)]
    ).build()

    seed_values = np.zeros((1, 1, 4), dtype=np.float32)
    seed_values[..., 0] = 1.0
    result = model({"minimized_body_seed": seed_values})
    np.testing.assert_allclose(
        to_numpy(result["minimized_loop_out"]), np.full((1, 1, 1), 16.0)
    )
