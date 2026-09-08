import os
from typing import cast

import numpy as np

from conftest import to_numpy
from nnodely import (
    Constant,
    Parameter,
    Input,
    Linear,
    Loop,
    Roll,
    Modely,
    Output,
    Fir,
    DataLoader,
)
import pytest


def test_loop(tmp_path):
    input1 = Input("in1")
    relation = Linear(
        out_features=1,
        use_bias=False,
        initializer="ones",
    )(input1.last())
    body_output = Output("body_out", relation)
    body = Modely("loop_body", inputs=[input1], outputs=[body_output]).build()
    assert relation.kernel is not None
    relation.kernel.assign(np.full((1, 1), 2.0, dtype=np.float32))

    # The rollout axis lives outside the body: five steps, seeded by in1_seq[0].
    seed = Input("in1_seq", seq=5)
    loop = Loop(
        f=body,
        callback={input1: body_output},
        initial={input1: seed},
        collect=False,
    )
    output1 = Output("out1", loop)

    model = Modely(
        "test_closed_loop",
        inputs=[seed],
        outputs=[output1],
    ).build()
    assert loop.f is body
    model.export_html(out_dir=tmp_path, filename="test_closed_loop")

    seed_values = np.zeros((1, 1, 5), dtype=np.float32)
    seed_values[..., 0] = 1.0
    result = model({"in1_seq": seed_values})
    output = to_numpy(result["out1"])[0]

    assert output.shape == (1, 1)
    np.testing.assert_allclose(
        output,
        np.full((1, 1), 32.0, dtype=np.float32),
    )


def test_nested_loop(tmp_path):
    # The rollout axis is the last one, so the outer three steps come last.
    x = Input("nested_input", dim=1, seq=(5, 3))
    gain = Constant("nested_gain", value=2.0)

    inner = Input("inner_input", dim=1, seq=5)
    first_relation = inner * gain
    first_body_output = Output("first_body_output", first_relation)
    first_body = Modely(
        "first_body", inputs=[inner], outputs=[first_body_output]
    ).build()
    first_loop = Loop(
        f=first_body,
        callback={inner: first_body_output},
        initial={inner: x},
        name="first_loop",
        collect=False,
    )
    assert first_loop.shape.dimensions == ((1,), 1, (5,))

    second_input = Input("second_input", dim=1)
    second_relation = second_input * gain
    second_body_output = Output("second_body_output", second_relation)
    second_body = Modely(
        "second_body", inputs=[second_input], outputs=[second_body_output]
    ).build()
    second_loop = Loop(
        f=second_body,
        callback={second_input: second_body_output},
        initial={second_input: first_loop},
        name="second_loop",
        collect=False,
    )
    output = Output("nested_output", second_loop)
    model = Modely("nested_loop_model", inputs=[x], outputs=[output]).build()
    model.export_html(out_dir=tmp_path, filename="test_nested_loop")

    nested_values = np.zeros((1, 1, 5, 3), dtype=np.float32)
    nested_values[..., 0] = 1.0
    result = model({"nested_input": nested_values})
    output = to_numpy(result["nested_output"])[0]

    assert second_loop.shape.dimensions == ((1,), 1, ())
    assert output.shape == (1, 1)
    np.testing.assert_allclose(
        output,
        np.full((1, 1), 256.0, dtype=np.float32),
    )


def test_loop_mechanical_modely(tmp_path):
    state = Input("mechanical_state", dim=1)
    external_force = Input("external_force", dim=1)
    state_transition = Linear(
        out_features=1,
        initializer="ones",
        bias_initializer="zeros",
    )(state.last())
    next_state_stream = state_transition + 1.0
    next_state = Output("next_state", next_state_stream)
    diagnostic = Output("diagnostic", external_force + 10.0)
    mechanical_model = Modely(
        "mechanical_model",
        inputs=[state, external_force],
        outputs=[next_state, diagnostic],
    ).build()
    mechanical_model.export_html(out_dir=tmp_path, filename="mechanical_model")
    assert state_transition.kernel is not None
    state_transition.kernel.assign(np.full((1, 1), 2.0, dtype=np.float32))

    state_seq = Input("mechanical_state_seq", dim=1, seq=5)
    loop = Loop(
        f=mechanical_model,
        callback={"mechanical_state": "next_state"},
        initial={"mechanical_state": state_seq},
        name="mechanical_loop",
        collect=False,
    )
    assert loop.shape.dimensions == ((1,), 1, ())
    model = Modely(
        "closed_mechanical_model",
        inputs=[state_seq, external_force],
        outputs=[Output("closed_state", loop)],
    ).build()
    model.export_html(out_dir=tmp_path, filename="closed_mechanical_model")

    state_values = np.zeros((1, 1, 5), dtype=np.float32)
    state_values[..., 0] = 1.0
    result = model(
        {
            "mechanical_state_seq": state_values,
            "external_force": np.ones((1, 1), dtype=np.float32),
        }
    )
    output = to_numpy(result["closed_state"])[0]

    assert output.shape == (1, 1)
    np.testing.assert_allclose(output, np.full((1, 1), 63.0, dtype=np.float32))


def test_model_roll():
    x = Input("x")
    fir = Fir(out_features=1, use_bias=False)(x.sw(5))
    out = Output("out", fir)
    test = Modely("body", inputs=[x], outputs=[out])
    test.build()
    fir.kernel.assign(np.full((5, 1), 1.0, dtype=np.float32))
    roll = Roll(f=test, callback={x: out}, name="roll")
    roll_3 = Roll(f=test, callback={x: out}, steps=3, name="roll_3")
    assert roll.steps == x.time == 5
    assert roll.shape.tuple == x.shape.tuple
    output = Output("res", roll)
    output_3 = Output("res_3", roll_3)
    model = Modely("test_roll_multi_input", inputs=[x], outputs=[output, output_3])
    model.build()

    result = model(inputs={"x": [1, 2, 3, 4, 5]})
    assert result["res"].shape == (1, 1, 5)
    assert result["res_3"].shape == (1, 1, 5)
    np.testing.assert_allclose(
        to_numpy(result["res"]),
        np.array([15.0, 29.0, 56.0, 109.0, 214.0]).reshape((1, 1, 5)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["res_3"]),
        np.array([4.0, 5.0, 15.0, 29.0, 56.0]).reshape((1, 1, 5)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_model_rollback():
    x = Input("x")
    fir = Fir(out_features=1, use_bias=False, name="fir")(x.sw(5))
    out = Output("out", fir + x.last())
    test = Modely("body", inputs=[x], outputs=[out])
    test.rollback({"x": "fir"}, steps=3)
    test.build()
    fir.kernel.assign(np.full((5, 1), 1.0, dtype=np.float32))

    result = test(inputs={"x": [1, 2, 3, 4, 5]})
    assert result["out"].shape == (1, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        np.array([[[85.0]]]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_model_multi_rollback(tmp_path):
    x = Input("x")
    y = Input("y")
    fir = Fir(out_features=1, use_bias=False, name="fir")(x.sw(5))
    out = Output("out", fir + y.last())
    test = Modely("body", inputs=[x, y], outputs=[out])
    test.rollback({"x": "fir", "y": "out"}, steps=3)
    test.build()
    fir.kernel.assign(np.full((5, 1), 1.0, dtype=np.float32))

    result = test(inputs={"x": [1, 2, 3, 4, 5], "y": [10]})
    assert result["out"].shape == (1, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        np.array([[[110.0]]]),
        rtol=1e-5,
        atol=1e-5,
    )
    test.export_html(out_dir=tmp_path, filename="test_model_multi_rollback")


def dummy_input(shape, method="random"):
    if method == "random":
        return np.random.rand(*shape).astype(np.float32)
    elif method == "ones":
        return np.ones(shape, dtype=np.float32)
    elif method == "sequential":
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    else:
        return np.zeros(shape, dtype=np.float32)


def test_nested_closed_loop(tmp_path):
    # ------- Model with closed loop connections -------
    batch_size = 10
    # Define a simple model to be used in the loop
    x = Input(name="x", dim=1)
    y = Input(name="y", dim=1)
    c = Parameter("c", dim=1, value=1.0)
    d = Parameter("d", dim=1, value=1.0)
    r1 = x * c + y * d
    out1 = Output("out1", r1)
    model_add = Modely(name="model1", inputs=[x, y], outputs=[out1])
    model_add.build()

    # The rollout axis lives on the outer streams: x is seeded from x_seq[0] and
    # closed onto out1, while y is consumed one step at a time.
    x_seq = Input(name="x_seq", dim=1, seq=4)
    y_seq = Input(name="y_seq", dim=1, seq=4)
    loop_fn = Loop(
        f=model_add,
        callback={"x": "out1"},
        initial={"x": x_seq},
        inputs={"y": y_seq},
        name="loop_model_add",
    )
    out = Output("out", loop_fn)
    model_in = Modely(name="model", inputs=[x_seq, y_seq], outputs=[out])
    model_in.build()

    # Create a nested loop model
    z = Input(name="z", dim=1, seq=(4, 2))
    loop_fn2 = Loop(
        f=model_in,
        callback={"y_seq": "out"},
        initial={"y_seq": z},
        name="loop_model_in",
    )
    out_w = Output("out_w", loop_fn2)
    model_out = Modely(name="model_with_loop_w", inputs=[x_seq, z], outputs=[out_w])

    model_out.minimize(
        "error",
        source=out_w,
        target=Input("w_target", dim=1, seq=(4, 2)).sw(1),
        loss="mse",
    )

    model_out.build()
    model_in.plot(to_file=os.path.join(tmp_path, "model_with_loop.png"))
    model_out.plot(to_file=os.path.join(tmp_path, "model_with_loop_w.png"))
    model_out.export_html(tmp_path / "model_with_loop_w.html")

    # ------- Model inference -------
    batch_size = 1
    dummy_input_x = dummy_input((batch_size, 1, 1, 4), method="ones")
    dummy_input_y = dummy_input((batch_size, 1, 1, 4), method="sequential")

    result_in = model_in({"x_seq": dummy_input_x, "y_seq": dummy_input_y})
    assert "out" in result_in
    assert result_in["out"].shape == (batch_size, 1, 1, 4)
    np.testing.assert_allclose(
        to_numpy(result_in["out"]),
        np.array([2.0, 4.0, 7.0, 11.0], dtype=np.float32).reshape(1, 1, 1, 4),
    )
    dummy_input_z = dummy_input((batch_size, 1, 1, 4, 2), method="sequential")
    result_out = model_out(
        {
            "x_seq": dummy_input_x,
            "z": dummy_input_z,
            "w_target": dummy_input_z,
        }
    )
    assert "out_w" in result_out
    assert result_out["out_w"].shape == (batch_size, 1, 1, 4, 2)
    np.testing.assert_allclose(
        to_numpy(result_out["out_w"]),
        np.array(
            [[2.0, 3.0], [5.0, 8.0], [10.0, 18.0], [17.0, 35.0]],
            dtype=np.float32,
        ).reshape(1, 1, 1, 4, 2),
    )


@pytest.mark.slow
def test_simple_model_loop(tmp_path):
    # Define a simple model to be used in the loop
    x = Input(name="x", dim=1)
    z = Input(name="z", dim=1)
    w = Parameter("w", dim=1, value=3.5)
    out1 = Output("y", x + w + z)
    model_add = Modely(name="model1", inputs=[x, z], outputs=[out1])
    model_add.build()

    # x is seeded from x_seq[0] and rolled out over its four steps
    x_seq = Input(name="x_seq", dim=1, seq=4)
    loop_fn = Loop(
        f=model_add,
        callback={"x": "y"},
        initial={"x": x_seq},
        name="loop_model_add",
    )
    out = Output("out", loop_fn)
    model_in = Modely(name="simple_loop_model", inputs=[x_seq, z], outputs=[out])

    model_in.minimize(
        "error",
        source=out,
        target=Input("x_target", dim=1, seq=4),
        loss="mse",
    )
    model_in.build()
    model_in.export_html(os.path.join(tmp_path, "simple_loop_model.html"))

    # DataLoader creates seq=4 windows from these raw time series. Consecutive
    # x values differ by 5, matching the desired recurrence x[t+1] = x[t] + w + z
    # for w=4 and z=1.
    xs = 1.0 + 5.0 * np.arange(3000, dtype=np.float32)
    zs = np.ones(3000, dtype=np.float32)
    dataset = {"x_seq": xs, "z": zs, "x_target": xs + 5.0}

    res = model_in(
        {
            "x_seq": dummy_input((1, 1, 4), method="ones") + 7,
            "z": dummy_input((1, 1), method="ones"),
            "x_target": dummy_input((1, 1, 4), method="sequential"),
        }
    )
    print("Result of simple model:", res["out"])
    data_train = DataLoader(model_in, source=dataset)
    model_in.train(train_data=data_train, epochs=100, batch_size=64, lr=5e-3)
    if model_in.model is not None:
        print("Weights of simple model:", model_in.model.get_weights())

    res = model_in(
        {
            "x_seq": dummy_input((1, 1, 4), method="ones") + 7,
            "z": dummy_input((1, 1), method="ones"),
            "x_target": dummy_input((1, 1, 4), method="sequential"),
        }
    )
    print("Result of simple model:", res["out"])

    if model_in.model is None:
        raise ValueError(
            "Model weights are not available. call model_in.build() before training."
        )
    assert model_in.model.get_weights()[0] == pytest.approx(4.0, rel=1e-2)


@pytest.mark.slow
def test_simple_model2(tmp_path):
    # Define a simple model to be used in the loop
    x = Input(name="x", dim=1)
    z = Input(name="z", dim=1)
    w = Parameter("w", dim=1)
    k = Parameter("k", dim=1)
    out1 = Output("y", x + w)
    out2 = Output("y2", z - k)
    model_add = Modely(name="model1", inputs=[x, z], outputs=[out1, out2])
    model_add.build()

    # both states are seeded from the first step of their own outer stream
    x_seq = Input(name="x_seq", dim=1, seq=4)
    z_seq = Input(name="z_seq", dim=1, seq=4)
    loop_out1, loop_out2 = Loop(
        f=model_add,
        callback={"x": "y", "z": "y2"},
        initial={"x": x_seq, "z": z_seq},
        name="loop_model_add",
    )

    out1 = Output("out1", loop_out1)
    out2 = Output("out2", loop_out2)
    model_in = Modely(
        name="simple_loop_model", inputs=[x_seq, z_seq], outputs=[out1, out2]
    )

    model_in.minimize(
        "error_x",
        source=out1,
        target=Input("x_target", dim=1, seq=4),
        loss="mse",
    )
    model_in.minimize(
        "error_z",
        source=out2,
        target=Input("z_target", dim=1, seq=4),
        loss="mse",
    )
    model_in.build()
    model_in.export_html(tmp_path / "simple_loop_model.html")

    # DataLoader creates the seq=4 windows. These raw trajectories exactly
    # follow x[t+1] = x[t] + 4 and z[t+1] = z[t] - 2.
    samples = np.arange(3000, dtype=np.float32)
    xs = 1.0 + 4.0 * samples
    zs = 10000.0 - 2.0 * samples
    dataset = {
        "x_seq": xs,
        "z_seq": zs,
        "x_target": xs + 4.0,
        "z_target": zs - 2.0,
    }
    data_train = DataLoader(model_in, source=dataset)

    if model_in.model is None:
        raise ValueError(
            "Model weights are not available. call model_in.build() before training."
        )
    print("Weights of simple model:", model_in.model.get_weights())
    model_in.train(train_data=data_train, epochs=100, batch_size=64, lr=5e-3)

    res = model_in(
        {
            "x_seq": dummy_input((1, 1, 4), method="ones") + 7,
            "z_seq": dummy_input((1, 1, 4), method="ones"),
            "x_target": dummy_input((1, 1, 4), method="sequential"),
            "z_target": dummy_input((1, 1, 4), method="sequential"),
        }
    )
    print("Result of simple model:", res["out1"], res["out2"])

    print("Weights of simple model:", model_in.model.get_weights())
    assert model_in.model.get_weights()[0] == pytest.approx(4.0, rel=1e-2)
    assert model_in.model.get_weights()[1] == pytest.approx(2.0, rel=1e-2)


def test_loop_bound_inputs(tmp_path):
    # One-step body: the rollout axis lives on the outer streams only.
    x = Input("sliced_x", dim=1)
    u = Input("sliced_u", dim=1)
    body_output = Output("sliced_next", x.last() * 2.0 + u.last())
    body = Modely("sliced_body", inputs=[x, u], outputs=[body_output]).build()

    initial = Input("sliced_x0", dim=1, seq=3)
    driver = Input("sliced_u_seq", dim=1, seq=3)
    loop = Loop(
        f=body,
        callback={x: body_output},
        initial={x: initial},
        inputs={u: driver},
        name="sliced_loop",
    )
    assert loop.horizon == 3
    assert loop.shape.dimensions == ((1,), 1, (3,))

    model = Modely(
        "sliced_model",
        inputs=[initial, driver],
        outputs=[Output("sliced_out", loop)],
    ).build()
    model.export_html(out_dir=tmp_path, filename="sliced_model")

    result = model(
        {
            "sliced_x0": np.array([[[[1.0, 0.0, 0.0]]]], dtype=np.float32),
            "sliced_u_seq": np.array([[[[10.0, 20.0, 30.0]]]], dtype=np.float32),
        }
    )
    # x0 = 1 -> 2*1 + 10 = 12 -> 2*12 + 20 = 44 -> 2*44 + 30 = 118
    expected = np.array([12.0, 44.0, 118.0], dtype=np.float32).reshape(1, 1, 1, 3)
    np.testing.assert_allclose(to_numpy(result["sliced_out"]), expected)

    path = os.path.join(tmp_path, "sliced_model.keras")
    model.export_keras(path)
    reloaded = cast(Modely, Modely.import_keras(path))
    np.testing.assert_allclose(
        to_numpy(
            reloaded(
                {
                    "sliced_x0": np.array([[[[1.0, 0.0, 0.0]]]], dtype=np.float32),
                    "sliced_u_seq": np.array(
                        [[[[10.0, 20.0, 30.0]]]], dtype=np.float32
                    ),
                }
            )["sliced_out"]
        ),
        expected,
    )


def test_loop_dynamic_length():
    x = Input("dyn_x", dim=1)
    u = Input("dyn_u", dim=1)
    body_output = Output("dyn_next", x.last() * 2.0 + u.last())
    body = Modely("dyn_body", inputs=[x, u], outputs=[body_output]).build()

    initial = Input("dyn_x0", dim=1, seq=(None,))
    driver = Input("dyn_u_seq", dim=1, seq=(None,))

    with pytest.raises(ValueError, match="cannot determine the rollout length"):
        Loop(
            f=body,
            callback={x: body_output},
            initial={x: initial},
            inputs={u: driver},
            name="dyn_loop_no_length",
        )

    loop = Loop(
        f=body,
        callback={x: body_output},
        initial={x: initial},
        inputs={u: driver},
        name="dyn_loop",
        length=3,
    )
    assert loop.horizon == 3
    assert loop.shape.dimensions == ((1,), 1, (3,))

    model = Modely(
        "dyn_model",
        inputs=[initial, driver],
        outputs=[Output("dyn_out", loop)],
    ).build()

    # The declared length drives the training-time shapes ...
    short = model(
        {
            "dyn_x0": np.array([[[[1.0, 0.0, 0.0]]]], dtype=np.float32),
            "dyn_u_seq": np.array([[[[10.0, 20.0, 30.0]]]], dtype=np.float32),
        }
    )
    np.testing.assert_allclose(
        to_numpy(short["dyn_out"]),
        np.array([12.0, 44.0, 118.0], dtype=np.float32).reshape(1, 1, 1, 3),
    )

    # ... a longer sequence rolls out further.
    long = model(
        {
            "dyn_x0": np.zeros((1, 1, 1, 5), dtype=np.float32)
            + np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "dyn_u_seq": np.array(
                [[[[10.0, 20.0, 30.0, 40.0, 50.0]]]], dtype=np.float32
            ),
        }
    )
    np.testing.assert_allclose(
        to_numpy(long["dyn_out"]),
        np.array([12.0, 44.0, 118.0, 276.0, 602.0], dtype=np.float32).reshape(
            1, 1, 1, 5
        ),
    )


def test_loop_window_feedback():
    # A feedback input with a time window is closed by shifting the window.
    x = Input("window_x", dim=1)
    taps = Fir(out_features=1, use_bias=False)([x.sw(3)])
    body_output = Output("window_next", taps)
    body = Modely("window_body", inputs=[x], outputs=[body_output])

    initial = Input("window_x0", dim=1, seq=3)
    loop = Loop(
        f=body,
        callback={x: body_output},
        initial={x: initial.sw(3)},
        name="window_loop",
    )
    # Loop builds the body when it is handed an unbuilt one
    assert body.built
    assert taps.kernel is not None
    taps.kernel.assign(np.ones((3, 1), dtype=np.float32))
    assert loop.callback_shift_axes == [2]
    assert loop.shape.dimensions == ((1,), 1, (3,))

    model = Modely(
        "window_model",
        inputs=[initial],
        outputs=[Output("window_out", loop)],
    ).build()

    values = np.zeros((1, 1, 3, 3), dtype=np.float32)
    values[0, 0, :, 0] = [1.0, 2.0, 3.0]
    result = model({"window_x0": values})
    # [1,2,3] -> 6 -> [2,3,6] -> 11 -> [3,6,11] -> 20
    np.testing.assert_allclose(
        to_numpy(result["window_out"]),
        np.array([6.0, 11.0, 20.0], dtype=np.float32).reshape(1, 1, 1, 3),
    )


def test_loop_multi_output():
    position = Input("int_p", dim=1)
    velocity = Input("int_v", dim=1)
    acceleration = Input("int_a", dim=1)
    position_next = Output("int_p_next", position.last() + velocity.last())
    velocity_next = Output("int_v_next", velocity.last() + acceleration.last())
    body = Modely(
        "integrator",
        inputs=[position, velocity, acceleration],
        outputs=[position_next, velocity_next],
    ).build()

    p0 = Input("int_p0", dim=1, seq=(None,))
    v0 = Input("int_v0", dim=1, seq=(None,))
    a_seq = Input("int_a_seq", dim=1, seq=(None,))
    position_out, velocity_out = Loop(
        f=body,
        callback={position: position_next, velocity: velocity_next},
        initial={position: p0, velocity: v0},
        inputs={acceleration: a_seq},
        name="integrator_loop",
        length=3,
    )
    model = Modely(
        "integrator_model",
        inputs=[p0, v0, a_seq],
        outputs=[Output("p_traj", position_out), Output("v_traj", velocity_out)],
    ).build()

    zeros = np.zeros((1, 1, 1, 3), dtype=np.float32)
    result = model(
        {
            "int_p0": zeros,
            "int_v0": zeros,
            "int_a_seq": np.ones((1, 1, 1, 3), dtype=np.float32),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["p_traj"]),
        np.array([0.0, 1.0, 3.0], dtype=np.float32).reshape(1, 1, 1, 3),
    )
    np.testing.assert_allclose(
        to_numpy(result["v_traj"]),
        np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 1, 1, 3),
    )


def test_loop_collects_output_that_is_not_fed_back():
    x = Input("extra_x", dim=1)
    state = Output("extra_state", x * 2.0)
    diagnostic = Output("extra_diag", x + 100.0)
    body = Modely("extra_body", inputs=[x], outputs=[state, diagnostic]).build()

    seed = Input("extra_x0", dim=1, seq=4)
    loop = Loop(f=body, callback={x: state}, initial={x: seed}, name="extra_loop")
    state_out, diagnostic_out = loop

    model = Modely(
        "extra_model",
        inputs=[seed],
        outputs=[Output("s", state_out), Output("d", diagnostic_out)],
    ).build()

    result = model({"extra_x0": np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)})
    # the state starts at x[0] = 1 and doubles; the diagnostic reports each state
    np.testing.assert_allclose(
        to_numpy(result["s"]),
        np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32).reshape(1, 1, 1, 4),
    )
    np.testing.assert_allclose(
        to_numpy(result["d"]),
        np.array([101.0, 102.0, 104.0, 108.0], dtype=np.float32).reshape(1, 1, 1, 4),
    )


def test_loop_constant_initial_batched():
    # A constant initial value has no batch axis; the carry still has to match
    # the batched body output.
    x = Input("cst_x", dim=1)
    u = Input("cst_u", dim=1)
    body_output = Output("cst_next", x.last() * 2.0 + u.last())
    body = Modely("cst_body", inputs=[x, u], outputs=[body_output]).build()

    driver = Input("cst_u_seq", dim=1, seq=3)
    loop = Loop(f=body, callback={x: body_output}, inputs={u: driver}, name="cst_loop")

    model = Modely(
        "cst_model", inputs=[driver], outputs=[Output("cst_out", loop)]
    ).build()

    values = np.tile(np.array([[[[1.0, 2.0, 3.0]]]], dtype=np.float32), (2, 1, 1, 1))
    if model.model is None:
        raise ValueError(
            "Model weights are not available. call model.build() before inference."
        )
    result = model.model({"cst_u_seq": values})
    # x starts at the default 0.0: 0*2 + 1 = 1 -> 2*1 + 2 = 4 -> 2*4 + 3 = 11
    np.testing.assert_allclose(
        to_numpy(result["cst_out"]),
        np.tile(
            np.array([1.0, 4.0, 11.0], dtype=np.float32).reshape(1, 1, 1, 3),
            (2, 1, 1, 1),
        ),
    )
