import numpy as np

from conftest import to_numpy
from nnodely import Constant, Input, Linear, Scan, Loop, Modely, Output, Fir


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


def test_model_loop():
    x = Input("x")
    fir = Fir(out_features=1, use_bias=False)(x.sw(5))
    out = Output("out", fir)
    test = Modely("body", inputs=[x], outputs=[out])
    test.build()
    fir.kernel.assign(np.full((5, 1), 1.0, dtype=np.float32))
    loop = Loop(f=test, callback={x: out}, name="loop")
    loop_3 = Loop(f=test, callback={x: out}, steps=3, name="loop_3")
    assert loop.steps == x.time == 5
    assert loop.shape.tuple == x.shape.tuple
    output = Output("res", loop)
    output_3 = Output("res_3", loop_3)
    model = Modely("test_loop_multi_input", inputs=[x], outputs=[output, output_3])
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


def test_model_closed_loop():
    x = Input("x")
    fir = Fir(out_features=1, use_bias=False, name="fir")(x.sw(5))
    out = Output("out", fir + x.last())
    test = Modely("body", inputs=[x], outputs=[out])
    test.closed_loop({"x": "fir"}, steps=3)
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


def test_model_multi_closed_loop(tmp_path):
    x = Input("x")
    y = Input("y")
    fir = Fir(out_features=1, use_bias=False, name="fir")(x.sw(5))
    out = Output("out", fir + y.last())
    test = Modely("body", inputs=[x, y], outputs=[out])
    test.closed_loop({"x": "fir", "y": "out"}, steps=3)
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
    test.export_html(out_dir=tmp_path, filename="test_model_multi_closed_loop")
