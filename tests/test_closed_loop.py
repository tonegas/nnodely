import os

import numpy as np

from conftest import to_numpy
from nnodely import (
    Constant,
    Parameter,
    Input,
    Linear,
    Scan,
    Roll,
    Modely,
    Output,
    Fir,
    DataLoader,
)
import pytest


def test_scan(tmp_path):
    input1 = Input("in1", seq=5)
    init_constant = Constant("init", value=1.0)
    relation = Linear(
        out_features=1,
        use_bias=False,
        initializer="ones",
    )(input1.last())
    body_output = Output("body_out", relation)
    body = Modely("scan_body", inputs=[input1], outputs=[body_output]).build()
    assert relation.kernel is not None
    relation.kernel.assign(np.full((1, 1), 2.0, dtype=np.float32))

    scan = Scan(
        f=body,
        callback={input1: body_output},
        initial=init_constant,
    )
    output1 = Output("out1", scan)

    model = Modely(
        "test_closed_scan",
        inputs=[input1],
        outputs=[output1],
    ).build()
    assert scan.f is body
    model.export_html(out_dir=tmp_path, filename="test_closed_scan")

    result = model({"in1": np.zeros((1, 1, 5), dtype=np.float32)})
    output = to_numpy(result["out1"])[0]

    assert output.shape == (1, 1)
    np.testing.assert_allclose(
        output,
        np.full((1, 1), 32.0, dtype=np.float32),
    )


def test_nested_scan(tmp_path):
    x = Input("nested_input", dim=1, seq=(3, 5))
    gain = Constant("nested_gain", value=2.0)
    initial_vector = Constant(
        "initial_vector", value=np.ones((1, 1, 5), dtype=np.float32)
    )

    first_relation = x * gain
    first_body_output = Output("first_body_output", first_relation)
    first_body = Modely("first_body", inputs=[x], outputs=[first_body_output]).build()
    first_scan = Scan(
        f=first_body,
        callback={x: first_body_output},
        initial=initial_vector,
        name="first_scan",
    )
    assert first_scan.shape.dimensions == ((1,), 1, (5,))

    second_input = Input("second_input", dim=1, seq=5)
    second_relation = second_input * gain
    second_body_output = Output("second_body_output", second_relation)
    second_body = Modely(
        "second_body", inputs=[second_input], outputs=[second_body_output]
    ).build()
    second_scan = Scan(
        f=second_body,
        callback={second_input: second_body_output},
        initial=first_scan,
        name="second_scan",
    )
    output = Output("nested_output", second_scan)
    model = Modely(
        "nested_scan_model", inputs=[x, second_input], outputs=[output]
    ).build()
    model.export_html(out_dir=tmp_path, filename="test_nested_scan")

    result = model(
        {
            "nested_input": np.zeros((1, 1, 3, 5), dtype=np.float32),
            "second_input": np.zeros((1, 1, 5), dtype=np.float32),
        }
    )
    output = to_numpy(result["nested_output"])[0]

    assert second_scan.shape.dimensions == ((1,), 1, ())
    assert output.shape == (1, 1)
    np.testing.assert_allclose(
        output,
        np.full((1, 1), 256.0, dtype=np.float32),
    )


def test_scan_mechanical_modely(tmp_path):
    state = Input("mechanical_state", dim=1, seq=5)
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

    scan = Scan(
        f=mechanical_model,
        callback={"mechanical_state": "next_state"},
        initial=1.0,
        name="mechanical_scan",
    )
    assert scan.remaining_seq == ()
    model = Modely(
        "closed_mechanical_model",
        inputs=[state, external_force],
        outputs=[Output("closed_state", scan)],
    ).build()
    model.export_html(out_dir=tmp_path, filename="closed_mechanical_model")

    result = model(
        {
            "mechanical_state": np.zeros((1, 1, 5), dtype=np.float32),
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


def test_closed_loop_with_scan(tmp_path):
    # ------- Model with closed loop connections -------
    batch_size = 10
    # Define a simple model to be used in the loop
    x = Input(name="x", dim=1, seq=4)
    y = Input(name="y", dim=1, seq=4)
    c = Parameter("c", dim=1, value=1.0)
    d = Parameter("d", dim=1, value=1.0)
    r1 = x * c + y * d
    out1 = Output("out1", r1)
    model_add = Modely(name="model1", inputs=[x, y], outputs=[out1])
    model_add.build()

    # Option 1: use Loop layer directly in the output definition
    loop_fn = Scan(
        f=model_add,
        callback={"x": "out1"},
        initial=x,
        name="loop_model_add",
    )  # add initial condition
    out = Output("out", loop_fn)
    model_in = Modely(name="model", inputs=[x, y], outputs=[out])
    model_in.build()

    # Create a nested loop model
    # w = Input(name="w", dim=1)
    z = Input(name="z", dim=1, seq=(4, 2))
    loop_fn2 = Scan(
        f=model_in,
        callback={"y": "out"},
        initial=z,
        name="loop_model_in",
    )
    out_w = Output("out_w", loop_fn2)
    model_out = Modely(name="model_with_loop_w", inputs=[x, y, z], outputs=[out_w])

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

    result_in = model_in({"x": dummy_input_x, "y": dummy_input_y})
    assert "out" in result_in
    assert result_in["out"].shape == (batch_size, 1, 1, 4)
    np.testing.assert_allclose(
        to_numpy(result_in["out"]),
        np.array([2.0, 4.0, 7.0, 11.0], dtype=np.float32).reshape(1, 1, 1, 4),
    )
    dummy_input_z = dummy_input((batch_size, 1, 1, 4, 2), method="sequential")
    result_out = model_out(
        {
            "x": dummy_input_x,
            "y": dummy_input_y,
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
def test_simple_model_scan(tmp_path):
    # Define a simple model to be used in the loop
    x = Input(name="x", dim=1, seq=4)
    z = Input(name="z", dim=1)
    w = Parameter("w", dim=1, value=3.5)
    out1 = Output("y", x + w + z)
    model_add = Modely(name="model1", inputs=[x, z], outputs=[out1])
    model_add.build()

    # Option 1: use Loop layer directly in the output definition
    loop_fn = Scan(
        f=model_add,
        callback={"x": "y"},
        initial=x,
        name="loop_model_add",
    )
    out = Output("out", loop_fn)
    model_in = Modely(name="simple_loop_model", inputs=[x, z], outputs=[out])

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
    dataset = {"x": xs, "z": zs, "x_target": xs + 5.0}

    res = model_in(
        {
            "x": dummy_input((1, 1, 4), method="ones") + 7,
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
            "x": dummy_input((1, 1, 4), method="ones") + 7,
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
    x = Input(name="x", dim=1, seq=4)
    z = Input(name="z", dim=1, seq=4)
    w = Parameter("w", dim=1)
    k = Parameter("k", dim=1)
    out1 = Output("y", x + w)
    out2 = Output("y2", z - k)
    model_add = Modely(name="model1", inputs=[x, z], outputs=[out1, out2])
    model_add.build()

    # Option 1: use Loop layer directly in the output definition
    loop_out1, loop_out2 = Scan(
        f=model_add,
        callback={"x": "y", "z": "y2"},
        initial={"x": x, "z": z},
        name="loop_model_add",
    )

    out1 = Output("out1", loop_out1)
    out2 = Output("out2", loop_out2)
    model_in = Modely(name="simple_loop_model", inputs=[x, z], outputs=[out1, out2])

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
        "x": xs,
        "z": zs,
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
            "x": dummy_input((1, 1, 4), method="ones") + 7,
            "z": dummy_input((1, 1, 4), method="ones"),
            "x_target": dummy_input((1, 1, 4), method="sequential"),
            "z_target": dummy_input((1, 1, 4), method="sequential"),
        }
    )
    print("Result of simple model:", res["out1"], res["out2"])

    print("Weights of simple model:", model_in.model.get_weights())
    assert model_in.model.get_weights()[0] == pytest.approx(4.0, rel=1e-2)
    assert model_in.model.get_weights()[1] == pytest.approx(2.0, rel=1e-2)
