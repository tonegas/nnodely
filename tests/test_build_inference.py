import keras
import numpy as np
import pytest

from conftest import to_numpy
from nnodely import DataLoader, Modely, Input, Output, Fir, Linear, Loop, Parameter
from nnodely import TimeSelect


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def window_size():
    return 5


def test_basic_model_build_and_inference(batch_size, window_size):
    x = Input("x", dim=1, tw=window_size)
    y = Input("y", dim=1, tw=window_size)

    x_stream = x
    y_stream = y

    result_fir = Fir(features=2)([x_stream + y_stream])
    x_out = Output("x_pred", result_fir)

    model = Modely("model1", inputs=[x, y], outputs=[x_out])
    model.build()

    dummy_x = np.ones((batch_size, window_size, 1), dtype=np.float32)
    dummy_y = np.ones((batch_size, window_size, 1), dtype=np.float32)

    result = model({"x": dummy_x, "y": dummy_y})

    assert "x_pred" in result
    assert result["x_pred"].shape == (batch_size, 1, 2)


def test_model_inference_with_composed_model(batch_size, window_size):
    # ------- Model definition and building -------
    x = Input("x", dim=1, tw=window_size)
    y = Input("y", dim=1, tw=window_size)

    x_stream = x
    y_stream = y

    fir = Fir(features=2)
    result_fir = fir([x_stream + y_stream])

    x_out = Output("x_pred", result_fir)
    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    # ------- Model composition -------
    z = Input("z", dim=1, tw=window_size)
    z_stream = z
    z_fir = Fir(features=1)(model1([z_stream, z_stream]))
    z_out = Output("z_pred", z_fir)
    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    # ------- Model inference -------
    dummy_input_x = np.ones((batch_size, window_size, 1), dtype=np.float32)
    dummy_input_y = np.ones((batch_size, window_size, 1), dtype=np.float32)
    dummy_input_z = np.ones((batch_size, window_size, 1), dtype=np.float32)

    result1 = model1({"x": dummy_input_x, "y": dummy_input_y})
    assert "x_pred" in result1
    assert result1["x_pred"].shape == (batch_size, 1, 2)

    result2 = model2({"z": dummy_input_z})
    assert "z_pred" in result2
    # The outer Fir filters each of the 2 features of model1 alone.
    assert result2["z_pred"].shape == (batch_size, 1, 2)


def test_inference_accepts_a_backend_tensor():
    x = Input("backend_tensor_x", dim=1, tw=1)
    output = Output("backend_tensor_out", x * 2.0)
    model = Modely("backend_tensor_model", inputs=[x], outputs=[output]).build()

    value = keras.ops.convert_to_tensor(np.array([[[3.0]]], dtype=np.float32))
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
    x = Input(f"{name}_x", dim=1, tw=3)
    target = Input(f"{name}_target", dim=1, tw=1)
    hidden = Linear(features=1, name=f"{name}_hidden")(TimeSelect(-1)(x))
    prediction = Output(f"{name}_pred", Fir(features=1, name=f"{name}_fir")(x) + hidden)
    model = Modely(name, inputs=[x, target], outputs=[prediction])
    model.minimize(f"{name}_fit", source=prediction, target=target)
    model.minimize(f"{name}_small", source=hidden, target=0.0)
    return model.build()


def _windows(batch=4):
    return np.random.default_rng(0).uniform(size=(batch, 3, 1)).astype(np.float32)


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


def test_inference_keeps_an_input_that_outputs_and_minimizers_both_read():
    # A reconstruction: the input is both what the model reads and its target.
    x = Input("reconstruct_x", dim=1, tw=1)
    rebuilt = Output("reconstruct_out", Linear(features=1)(x))
    model = Modely("reconstruct", inputs=[x], outputs=[rebuilt])
    model.minimize("reconstruct_fit", source=rebuilt, target=x)
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


def test_loop_rolls_out_a_body_trained_with_its_own_minimizers():
    state = Input("minimized_body_state", dim=1, tw=1)
    gain = Parameter("minimized_body_gain", value=[2.0])
    relation = Linear(
        features=1, kernel=gain, bias=False, name="minimized_body_linear"
    )(state)
    next_state = Output("minimized_body_next", relation)
    body = Modely("minimized_body", inputs=[state], outputs=[next_state])
    body.minimize(
        "minimized_body_fit",
        source=next_state,
        target=Input("minimized_body_target", dim=1, tw=1),
    )
    body.build()

    # The loop runs the body as declared: its target is not an input of the loop.
    seed = Input("minimized_body_seed", dim=1, seq=4, tw=1)
    loop = Loop(
        f=body,
        feedback={state: next_state},
        collect=False,
        init={state: seed},
    )()
    model = Modely(
        "minimized_loop", inputs=[seed], outputs=[Output("minimized_loop_out", loop)]
    ).build()

    seed_values = np.zeros((1, 4, 1, 1), dtype=np.float32)
    seed_values[:, 0] = 1.0
    result = model({"minimized_body_seed": seed_values})
    np.testing.assert_allclose(
        to_numpy(result["minimized_loop_out"]), np.full((1, 1, 1), 16.0)
    )


def test_inference_names_the_inputs_missing_from_the_data():
    model = _minimized_model("missing_input")

    # The target is only read by a minimizer, so it is not what is missing.
    with pytest.raises(ValueError, match=r"\['missing_input_x'\]"):
        model({"missing_input_target": np.ones((4, 1, 1), dtype=np.float32)})


def test_inference_takes_its_arrays_as_a_block_takes_its_streams():
    model = _minimized_model("keyed_inference")
    x = model._inputs[0]
    windows = _windows()
    target = np.ones((4, 1, 1), dtype=np.float32)
    expected = to_numpy(model({"keyed_inference_x": windows})["keyed_inference_pred"])

    # Keyed by Input or by name, or listed in the order of the inputs - the
    # target too, which the outputs do not read: the same run.
    for given in (
        {x: windows},
        {x: windows, "keyed_inference_unknown": target},
        [windows, target],
        (windows, target),
    ):
        np.testing.assert_allclose(
            to_numpy(model(given)["keyed_inference_pred"]), expected
        )

    with pytest.raises(ValueError, match="2 inputs"):
        model([windows])
    with pytest.raises(ValueError, match="two values for its input"):
        model({x: windows, "keyed_inference_x": windows})
    with pytest.raises(TypeError, match="streams and arrays together"):
        model({x: windows, "keyed_inference_target": Input("keyed_other", tw=1)})


def test_a_model_that_reads_no_input_is_refused_at_build():
    # Once a bare Keras "All outputs values must be KerasTensors".
    from nnodely import Parameter

    only_values = Modely(
        "only_values",
        inputs=[],
        outputs=[Output("only_values_out", Parameter("only_gain", value=2.0) * 3.0)],
    )
    with pytest.raises(ValueError, match="reads no Input"):
        only_values.build()

    state = Input("numbers_state", tw=1)
    step = Output("numbers_next", Linear(bias=False)(state))
    body = Modely("numbers_body", inputs=[state], outputs=[step]).build()
    driven_by_numbers = Loop(f=body, feedback={state: step}, seq=4, init={state: 1.0})()
    looped = Modely(
        "numbers_loop", inputs=[], outputs=[Output("numbers_traj", driven_by_numbers)]
    )
    with pytest.raises(ValueError, match="'numbers_loop' reads no Input"):
        looped.build()


def test_a_model_declares_every_input_it_reads():
    # An Input read without being declared was an input all the same, the
    # init of an Integrate too: a block read it from the data of its caller.
    # Only a hidden state, or an Input only a minimizer reads, is left out.
    from nnodely import Integrate

    x, y, x0 = Input("decl_x", tw=1), Input("decl_y", tw=1), Input("decl_x0", tw=1)
    with pytest.raises(ValueError, match=r"'decl_sum' reads the Inputs \['decl_y'\]"):
        Modely("decl_sum", inputs=[x], outputs=[Output("decl_sum_o", x + y)])
    integral = Integrate(dt=1.0, init=x0)(x)
    with pytest.raises(ValueError, match=r"'decl_int' reads the Inputs \['decl_x0'\]"):
        Modely("decl_int", inputs=[x], outputs=[Output("decl_int_o", integral)])

    # A caller declares what it gives a block.
    block = Modely(
        "decl_block", inputs=[x, x0], outputs=[Output("decl_block_o", integral)]
    )
    u = Input("decl_u", tw=1)
    with pytest.raises(
        ValueError, match=r"'decl_caller' reads the Inputs \['decl_x0'\]"
    ):
        Modely("decl_caller", inputs=[u], outputs=[Output("decl_co", block([u, x0]))])
    caller = Modely(
        "decl_caller", inputs=[u, x0], outputs=[Output("decl_co", block([u, x0]))]
    ).build()
    result = caller({"decl_u": np.ones((1, 1, 1)), "decl_x0": np.full((1, 1, 1), 10.0)})
    np.testing.assert_allclose(to_numpy(result["decl_co"]).ravel(), [11.0])

    # A fed-back Input too: it was read from the data with a warning.
    state, target = Input("decl_state", tw=1), Input("decl_target", tw=1)
    state.feedback(state + x)
    with pytest.raises(
        ValueError, match=r"'decl_left_out' reads the Inputs \['decl_state'\]"
    ):
        Modely("decl_left_out", inputs=[x], outputs=[Output("decl_lo", state + x)])
    model = Modely(
        "decl_left_in", inputs=[x, state], outputs=[Output("decl_lo", state + x)]
    )
    model.minimize("decl_error", "decl_lo", target)
    model.build()
    assert {node.name for node in model.train_inputs} == {
        "decl_x",
        "decl_state",
        "decl_target",
    }


def test_two_inputs_of_one_name_are_refused():
    # The data is keyed by input name: two Inputs named alike read one array,
    # and a model with them once ran silently on the wrong values.
    a, b = Input("twin_name", tw=1), Input("twin_name", tw=1)
    model = Modely("twin_names", inputs=[a, b], outputs=[Output("twin_out", a + b)])
    with pytest.raises(ValueError, match="two different Inputs named 'twin_name'"):
        model.build()


def test_a_build_that_fails_leaves_the_model_as_it_was():
    # A failed build once left the model with the outputs of the build it
    # attempted and the Keras model of the one before: with the minimizer
    # that broke it removed, training failed on a bare KeyError.
    y = Input("failed_build_y", tw=4)
    target = Input("failed_build_target", tw=1)
    out = Output("failed_build_out", Fir(features=1)([y]))
    model = Modely("failed_build", inputs=[y], outputs=[out])
    model.minimize("fit", out, target)
    model.build()

    def state():
        return [
            id(model.model),
            id(model.inference_model),
            *map(id, model.train_inputs),
            *map(id, model.inference_inputs),
            *map(id, model._train_outputs),
            *model._minimizer_outputs.values(),
        ]

    built = state()
    # Named like y after it is minimized: the build reads two Inputs of one
    # name.
    twin = Input("failed_build_twin", tw=1)
    model.minimize("clash", out, twin)
    twin.name = "failed_build_y"
    with pytest.raises(ValueError, match="two different Inputs named"):
        model.build()
    assert state() == built

    model.remove_minimizer("clash")
    data = DataLoader(
        model,
        source={"failed_build_y": np.zeros(12), "failed_build_target": np.zeros(12)},
    )
    model.train(data, epochs=1, printer=None)


def test_a_file_that_is_not_an_nnodely_2_model_is_refused(tmp_path):
    import json

    x = Input("file_check_x", tw=1)
    Modely(
        "file_check", inputs=[x], outputs=[Output("file_check_out", x * 2.0)]
    ).build().save(tmp_path / "model")
    config = tmp_path / "model" / "model.json"
    saved = json.loads(config.read_text())

    for edit, message in (
        ({"version": 99}, "format version 99"),
        ({"format": "keras"}, "not a model saved by nnodely 2"),
    ):
        config.write_text(json.dumps({**saved, **edit}))
        with pytest.raises(ValueError, match=message):
            Modely.load(tmp_path / "model")

    unknown = {**saved, "nodes": [{**saved["nodes"][0], "class_name": "Unknown"}]}
    config.write_text(json.dumps(unknown))
    with pytest.raises(ValueError, match="'Unknown' node"):
        Modely.load(tmp_path / "model")
    config.write_text("{ not json")
    with pytest.raises(ValueError, match="not a valid model file"):
        Modely.load(tmp_path / "model")


def test_a_model_keeps_its_own_lists_of_inputs_and_outputs():
    # Once a model kept the very lists it was declared with: one reused for a
    # second model changed the first, which then returned an output it was
    # never declared with and read an Input it never declared.
    a, b = Input("ls_a", tw=3), Input("ls_b", tw=3)
    inputs = [a]
    outs = [Output("ls_ya", Fir(features=1)([a]))]
    m1 = Modely("ls_m1", inputs=inputs, outputs=outs)
    inputs.append(b)
    outs.append(Output("ls_yb", Fir(features=1)([b])))  # reused for a second model
    m2 = Modely("ls_m2", inputs=inputs, outputs=outs)

    m1.build()
    assert [node.name for node in m1.outputs] == ["ls_ya"]
    assert [node.name for node in m1.inputs] == ["ls_a"]
    assert [node.name for node in m1.inference_inputs] == ["ls_a"]
    window = np.ones((1, 3, 1), np.float32)
    assert set(m1({"ls_a": window})) == {"ls_ya"}

    m2.build()
    assert [node.name for node in m2.outputs] == ["ls_ya", "ls_yb"]
    assert set(m2({"ls_a": window, "ls_b": window})) == {"ls_ya", "ls_yb"}
