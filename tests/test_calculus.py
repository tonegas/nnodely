import tempfile

import keras
import numpy as np
import pytest

from conftest import to_numpy
from nnodely import (
    DataLoader,
    Derivative,
    Fir,
    Input,
    Integrate,
    Modely,
    Output,
    Sin,
)
from nnodely.core.layer import Add
from nnodely.core.modely import _traces_backward_pass


def test_derivate_wrt_input_and_time():
    x = Input("x", dim=1)
    y = Input("y", dim=1)
    x_last = x.last()
    y_last = y.last()
    fun = Sin()(x_last) + y_last**2
    out_der_x = Derivative(order=1, respect_to=x)(fun)
    out_der_y = Derivative(order=1, respect_to=y)(fun)
    out_der_time = Derivative(order=1, respect_to=0.1)(fun)

    out = Output("out", fun)
    outx = Output("outx", out_der_x)
    outy = Output("outy", out_der_y)
    out_time = Output("out_time", out_der_time)

    m = Modely(name="test", inputs=[x, y], outputs=[out, outx, outy, out_time])
    m.build()

    result = m({"x": [[np.pi / 2]], "y": [[3]]})
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        np.array(10.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["outx"]),
        np.array(0.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["outy"]),
        np.array(6.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["out_time"]),
        np.array(100.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    # results = []
    # for xi, yi in zip([90, 0], [3, 6]):
    #     results.append(m({"x": [xi], "y": [yi]}))
    # np.testing.assert_allclose(
    #     to_numpy(results[0]["out"]),
    #     np.array(10.0, dtype=np.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     to_numpy(results[0]["outx"]),
    #     np.array(0.0, dtype=np.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     to_numpy(results[0]["outy"]),
    #     np.array(6.0, dtype=np.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     to_numpy(results[0]["out_time"]),
    #     np.array(1.0, dtype=np.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     to_numpy(results[1]["out"]),
    #     np.array(36.0, dtype=np.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     to_numpy(results[1]["outx"]),
    #     np.array(1.0, dtype=np.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     to_numpy(results[1]["outy"]),
    #     np.array(12.0, dtype=np.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     to_numpy(results[1]["out_time"]),
    #     np.array(260, dtype=np.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )


def test_derivative_wrt_input_matches_analytic_over_a_batch():
    """Every sample of the batch is differentiated on its own: reverse mode
    sums over a relation's own axes, never across independent samples."""
    x = Input("batch_x", dim=1)
    y = Input("batch_y", dim=1)
    fun = Sin()(x.last()) + y.last() ** 2

    model = Modely(
        "derivative_batch_model",
        inputs=[x, y],
        outputs=[
            Output("fun", fun),
            Output("dx", Derivative(order=1, respect_to=x)(fun)),
            Output("dy", Derivative(order=1, respect_to=y)(fun)),
        ],
    ).build()

    x_values = np.array([0.0, np.pi / 2, 1.0, -2.0], dtype=np.float32).reshape(4, 1, 1)
    y_values = np.array([3.0, -1.0, 0.5, 2.0], dtype=np.float32).reshape(4, 1, 1)
    result = model({"batch_x": x_values, "batch_y": y_values})

    np.testing.assert_allclose(
        to_numpy(result["fun"]), np.sin(x_values) + y_values**2, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(result["dx"]), np.cos(x_values), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(result["dy"]), 2.0 * y_values, rtol=1e-5, atol=1e-5
    )


def test_derivative_wrt_input_second_order():
    x = Input("second_x", dim=1)
    fun = Sin()(x.last()) + x.last() ** 3

    model = Modely(
        "derivative_second_order_model",
        inputs=[x],
        outputs=[
            Output("first", Derivative(order=1, respect_to=x)(fun)),
            Output("second", Derivative(order=2, respect_to=x)(fun)),
        ],
    ).build()

    values = np.array([0.5, 1.5, -1.0], dtype=np.float32).reshape(3, 1, 1)
    result = model({"second_x": values})

    np.testing.assert_allclose(
        to_numpy(result["first"]),
        np.cos(values) + 3.0 * values**2,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["second"]),
        -np.sin(values) + 6.0 * values,
        rtol=1e-4,
        atol=1e-4,
    )


def test_derivative_wrt_input_nested_equals_second_order():
    """A derivative is itself a differentiable relation, so it can be fed
    back into another Derivative."""
    x = Input("nested_x", dim=1)
    fun = x.last() ** 3
    first = Derivative(order=1, respect_to=x)(fun)

    model = Modely(
        "derivative_nested_model",
        inputs=[x],
        outputs=[
            Output("nested", Derivative(order=1, respect_to=x)(first)),
            Output("direct", Derivative(order=2, respect_to=x)(fun)),
        ],
    ).build()

    values = np.array([[[2.0]]], dtype=np.float32)
    result = model({"nested_x": values})

    np.testing.assert_allclose(
        to_numpy(result["nested"]), np.full((1, 1, 1), 12.0), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        to_numpy(result["nested"]), to_numpy(result["direct"]), rtol=1e-4, atol=1e-4
    )


def test_derivative_wrt_input_through_a_trainable_layer():
    """Differentiation goes through the weights, not around them: a Fir over
    a window is a weighted sum, so its derivative is the weight of each sample
    and the result has the shape of the input's window."""
    x = Input("fir_x", dim=1)
    fun = Fir(out_features=1)([x.sw(3)])
    derivative = Derivative(order=1, respect_to=x)(fun)

    assert derivative.shape.tuple == (1, 3)

    model = Modely(
        "derivative_through_fir_model",
        inputs=[x],
        outputs=[Output("dx", derivative)],
    ).build()

    values = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    result = to_numpy(model({"fir_x": values})["dx"])
    weights = to_numpy(fun.kernel).reshape(1, 1, 3)

    assert result.shape == (1, 1, 3)
    np.testing.assert_allclose(result, weights, rtol=1e-5, atol=1e-5)


def test_derivative_wrt_input_the_relation_does_not_read_raises():
    x = Input("unrelated_x", dim=1)
    y = Input("unrelated_y", dim=1)
    fun = Sin()(x.last())

    with pytest.raises(ValueError, match="does not depend on input"):
        Modely(
            "derivative_unrelated_model",
            inputs=[x, y],
            outputs=[Output("dy", Derivative(order=1, respect_to=y)(fun))],
        ).build()


def test_derivative_wrt_time_keeps_the_window_length():
    """One derivative per sample of the window, each read from the samples up
    to it, so the result stays aligned with the signal."""
    dt = 0.1
    v = Input("time_window_v", dim=1)
    fun = v.sw(4)
    derivative = Derivative(order=1, respect_to=dt)(fun)

    assert derivative.shape.tuple == (1, 4)

    model = Modely(
        "derivative_time_window_model",
        inputs=[v],
        outputs=[Output("dt", derivative)],
    ).build()

    samples = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    result = to_numpy(model({"time_window_v": samples.reshape(1, 1, 4)})["dt"])

    expected = np.diff(np.concatenate([[0.0], samples])) / dt
    assert result.shape == (1, 1, 4)
    np.testing.assert_allclose(result, expected.reshape(1, 1, 4), rtol=1e-5)


def test_derivative_wrt_time_over_a_relation_and_a_batch():
    """f is any Stream, and every sample of the batch is differentiated on
    its own timeline."""
    dt = 0.5
    v = Input("time_batch_v", dim=1)
    fun = v.sw(3) * 2.0

    model = Modely(
        "derivative_time_batch_model",
        inputs=[v],
        outputs=[Output("dt", Derivative(order=1, respect_to=dt)(fun))],
    ).build()

    samples = np.array([[1.0, 2.0, 4.0], [0.0, -1.0, -3.0]], dtype=np.float32)
    values = samples.reshape(2, 1, 3)
    result = to_numpy(model({"time_batch_v": values})["dt"])

    scaled = 2.0 * samples
    expected = np.diff(np.pad(scaled, ((0, 0), (1, 0))), axis=1) / dt
    np.testing.assert_allclose(result, expected.reshape(2, 1, 3), rtol=1e-5)


def test_derivative_wrt_time_takes_a_missing_past_as_zero():
    """With no initial condition the sample before the window is zero, so a
    single-sample relation differentiates to f / dt."""
    dt = 0.1
    v = Input("time_single_v", dim=1)

    model = Modely(
        "derivative_time_single_model",
        inputs=[v],
        outputs=[Output("dt", Derivative(order=1, respect_to=dt)(v.last()))],
    ).build()

    values = np.array([[[2.0]]], dtype=np.float32)
    result = to_numpy(model({"time_single_v": values})["dt"])
    np.testing.assert_allclose(result, np.full((1, 1, 1), 20.0), rtol=1e-5)


def test_derivative_wrt_time_reads_a_number_as_the_initial_condition():
    dt = 0.1
    v = Input("time_init_number_v", dim=1)

    model = Modely(
        "derivative_time_init_number_model",
        inputs=[v],
        outputs=[
            Output("dt", Derivative(order=1, respect_to=dt, init=1.5)(v.sw(3))),
        ],
    ).build()

    samples = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    result = to_numpy(model({"time_init_number_v": samples.reshape(1, 1, 3)})["dt"])

    expected = np.diff(np.concatenate([[1.5], samples])) / dt
    np.testing.assert_allclose(result, expected.reshape(1, 1, 3), rtol=1e-5)


def test_derivative_wrt_time_reads_a_stream_as_the_initial_condition():
    """The initial condition is a Stream like any other - here the state the
    window continues, so consecutive chunks of a trajectory join up instead
    of restarting from zero."""
    dt = 0.1
    v = Input("time_init_v", dim=1)
    previous = Input("time_init_previous", dim=1)

    model = Modely(
        "derivative_time_init_stream_model",
        inputs=[v, previous],
        outputs=[
            Output(
                "dt",
                Derivative(order=1, respect_to=dt, init=previous.last())(v.sw(3)),
            )
        ],
    ).build()

    samples = np.array([[2.0, 3.0, 5.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    initial = np.array([[1.0], [4.0]], dtype=np.float32)
    result = to_numpy(
        model(
            {
                "time_init_v": samples.reshape(2, 1, 3),
                "time_init_previous": initial.reshape(2, 1, 1),
            }
        )["dt"]
    )

    expected = np.diff(np.concatenate([initial, samples], axis=1), axis=1) / dt
    np.testing.assert_allclose(result, expected.reshape(2, 1, 3), rtol=1e-5)


def test_derivative_wrt_time_second_order():
    dt = 0.1
    v = Input("time_second_v", dim=1)

    model = Modely(
        "derivative_time_second_model",
        inputs=[v],
        outputs=[Output("dt2", Derivative(order=2, respect_to=dt)(v.sw(3)))],
    ).build()

    samples = np.array([1.0, 3.0, 8.0], dtype=np.float32)
    result = to_numpy(model({"time_second_v": samples.reshape(1, 1, 3)})["dt2"])

    padded = np.concatenate([[0.0, 0.0], samples])
    expected = (
        np.array([padded[i] - 2.0 * padded[i + 1] + padded[i + 2] for i in range(3)])
        / dt**2
    )
    np.testing.assert_allclose(result, expected.reshape(1, 1, 3), rtol=1e-5)


def test_derivative_wrt_time_second_order_repeats_a_single_init_sample():
    """A second derivative reads two samples of history; one initial
    condition stands for both, which is the same as starting at rest."""
    dt = 0.1
    v = Input("time_second_init_v", dim=1)

    model = Modely(
        "derivative_time_second_init_model",
        inputs=[v],
        outputs=[
            Output("dt2", Derivative(order=2, respect_to=dt, init=2.0)(v.sw(2))),
        ],
    ).build()

    samples = np.array([2.0, 5.0], dtype=np.float32)
    result = to_numpy(model({"time_second_init_v": samples.reshape(1, 1, 2)})["dt2"])

    padded = np.concatenate([[2.0, 2.0], samples])
    expected = (
        np.array([padded[i] - 2.0 * padded[i + 1] + padded[i + 2] for i in range(2)])
        / dt**2
    )
    np.testing.assert_allclose(result, expected.reshape(1, 1, 2), rtol=1e-5)


def test_derivative_wrt_time_a_higher_poly_order_is_more_accurate():
    """poly_order=2 fits a parabola instead of a line, so it differentiates a
    quadratic trajectory exactly where the two-sample difference lags by half
    a step - same causality, one more sample of history."""
    dt = 0.1
    v = Input("time_accuracy_v", dim=1)

    model = Modely(
        "derivative_time_accuracy_model",
        inputs=[v],
        outputs=[
            Output("two", Derivative(order=1, respect_to=dt)(v.sw(4))),
            Output(
                "bdf2",
                Derivative(order=1, respect_to=dt, window=3, poly_order=2)(v.sw(4)),
            ),
        ],
    ).build()

    # x(t) = t^2 on the window, so the exact derivative is 2t.
    times = np.arange(4, dtype=np.float32) * dt
    samples = times**2
    result = model({"time_accuracy_v": samples.reshape(1, 1, 4)})

    exact = 2.0 * times
    two = to_numpy(result["two"]).reshape(4)
    bdf2 = to_numpy(result["bdf2"]).reshape(4)

    # Only the samples with a full history are comparable; the first ones read
    # the zero initial condition.
    np.testing.assert_allclose(bdf2[2:], exact[2:], rtol=1e-4, atol=1e-4)
    assert np.abs(two[2:] - exact[2:]).max() > 0.09  # a half step of lag


def test_derivative_wrt_time_longer_window_smooths_noise():
    """The other reason to widen the window: the two-sample difference
    multiplies sensor noise by sqrt(2)/dt, while a least-squares fit over
    more samples divides that gain down by roughly window^-3/2."""
    dt = 0.01
    samples = 64
    v = Input("time_noise_v", dim=1)

    model = Modely(
        "derivative_time_noise_model",
        inputs=[v],
        outputs=[
            Output("sharp", Derivative(order=1, respect_to=dt)(v.sw(samples))),
            Output(
                "smooth",
                Derivative(order=1, respect_to=dt, window=9)(v.sw(samples)),
            ),
        ],
    ).build()

    # Slow compared with the 9-sample window, so the fit's delay stays small
    # next to the noise it removes - the regime this knob is meant for.
    times = np.arange(samples, dtype=np.float32) * dt
    clean = np.sin(2.0 * np.pi * 0.2 * times)
    noise = np.random.default_rng(0).normal(0.0, 1e-3, size=samples).astype(np.float32)
    exact = 2.0 * np.pi * 0.2 * np.cos(2.0 * np.pi * 0.2 * times)

    result = model({"time_noise_v": (clean + noise).reshape(1, 1, samples)})
    settled = slice(9, None)
    sharp_error = np.abs(
        to_numpy(result["sharp"]).reshape(samples)[settled] - exact[settled]
    ).mean()
    smooth_error = np.abs(
        to_numpy(result["smooth"]).reshape(samples)[settled] - exact[settled]
    ).mean()
    assert smooth_error < sharp_error / 3.0


def test_derivative_wrt_time_inverts_the_cumulative_integral():
    """Differentiating and integrating are exact inverses on a window: the
    rectangular rule sums exactly the increments the backward difference
    produced, so a physics residual written with both stays consistent."""
    dt = 0.1
    v = Input("time_inverse_v", dim=1)
    window = v.sw(5)
    derivative = Derivative(order=1, respect_to=dt)(window)

    model = Modely(
        "derivative_time_inverse_model",
        inputs=[v],
        outputs=[
            Output("roundtrip", Integrate(solver="rectangular", dt=dt)(derivative))
        ],
    ).build()

    samples = np.array([2.0, 3.5, 3.0, 7.0, 11.0], dtype=np.float32)
    result = to_numpy(model({"time_inverse_v": samples.reshape(1, 1, 5)})["roundtrip"])
    np.testing.assert_allclose(
        result, (samples - samples[0]).reshape(1, 1, 5), rtol=1e-4, atol=1e-4
    )


def test_derivative_wrt_time_ignores_sample_time():
    """The float passed to respect_to *is* dt; an Input's sample_time is
    deliberately not consulted, so the step is always explicit."""
    v = Input("time_ignored_v", dim=1, sample_time=0.5)

    model = Modely(
        "derivative_time_explicit_model",
        inputs=[v],
        outputs=[Output("dt", Derivative(order=1, respect_to=0.1)(v.sw(2)))],
    ).build()

    values = np.array([[[1.0, 2.0]]], dtype=np.float32)
    result = to_numpy(model({"time_ignored_v": values})["dt"])
    np.testing.assert_allclose(
        result, np.array([[[10.0, 10.0]]], dtype=np.float32), rtol=1e-5
    )


def test_derivative_wrt_time_in_a_multi_dimensional_relation():
    """Each feature is differentiated along its own time axis, and a scalar
    initial condition applies to all of them."""
    dt = 0.1
    x = Input("time_multi_x", dim=3)

    model = Modely(
        "derivative_time_multi_model",
        inputs=[x],
        outputs=[Output("dt", Derivative(order=1, respect_to=dt, init=1.0)(x.sw(2)))],
    ).build()

    samples = np.array([[1.0, 2.0], [3.0, 3.0], [-1.0, 0.0]], dtype=np.float32)
    result = to_numpy(model({"time_multi_x": samples.reshape(1, 3, 2)})["dt"])

    expected = np.diff(np.pad(samples, ((0, 0), (1, 0)), constant_values=1.0), axis=1)
    assert result.shape == (1, 3, 2)
    np.testing.assert_allclose(result, expected.reshape(1, 3, 2) / dt, rtol=1e-5)


def test_derivative_wrt_time_rejects_invalid_arguments():
    dt = 0.1
    v = Input("time_invalid_v", dim=1)
    x = Input("time_invalid_x", dim=1)

    with pytest.raises(ValueError, match="reads at least"):
        Derivative(order=2, respect_to=dt, window=2)
    with pytest.raises(ValueError, match="poly_order"):
        Derivative(order=1, respect_to=dt, window=3, poly_order=3)
    with pytest.raises(ValueError, match="poly_order"):
        Derivative(order=2, respect_to=dt, window=4, poly_order=1)
    with pytest.raises(ValueError, match="init, window and poly_order"):
        Derivative(order=1, respect_to=x, init=0.0)
    with pytest.raises(ValueError, match="init, window and poly_order"):
        Derivative(order=1, respect_to=x, window=3)

    with pytest.raises(ValueError, match="init must carry"):
        Derivative(order=1, respect_to=dt, init=v.sw(2))(v.sw(3))
    with pytest.raises(ValueError, match="init has dim"):
        Derivative(order=1, respect_to=dt, init=Input("time_invalid_i", dim=2).last())(
            v.sw(3)
        )


def test_derivative_wrt_input_in_a_multi_dimensional_relation():
    """dim > 1 inputs keep their own axis: one derivative per feature."""
    x = Input("multi_x", dim=3)
    fun = x.last() ** 2

    model = Modely(
        "derivative_multi_dim_model",
        inputs=[x],
        outputs=[Output("dx", Derivative(order=1, respect_to=x)(fun))],
    ).build()

    values = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    result = to_numpy(model({"multi_x": values})["dx"])

    assert result.shape == (1, 3, 1)
    np.testing.assert_allclose(result, 2.0 * values, rtol=1e-5, atol=1e-5)


def _build_mixed_derivative_model(name):
    """One model exercising both modes at once - including the time
    derivative's configuration (init, window, poly_order), which has to
    survive a round trip like any other layer state."""
    x = Input(f"{name}_x", dim=1)
    init = Input(f"{name}_init", dim=1)
    fir = Fir(out_features=1, name=f"{name}_fir")
    fun = Sin()(fir([x.sw(2)]) * x.sw(2))
    return (
        x,
        fir,
        Modely(
            name,
            inputs=[x, init],
            outputs=[
                Output("fun", fun),
                Output("dx", Derivative(order=1, respect_to=x)(fun)),
                Output("dx2", Derivative(order=2, respect_to=x)(fun)),
                Output(
                    "dt",
                    Derivative(order=1, respect_to=0.1, init=init.last())(fun),
                ),
                Output(
                    "dt_smooth",
                    Derivative(order=1, respect_to=0.1, window=2, poly_order=1)(fun),
                ),
            ],
        ),
    )


def test_derivative_save_load_round_trip(tmp_path):
    _, _, model = _build_mixed_derivative_model("derivative_save_model")
    model.build()

    values = np.array([[[0.4, 1.3]]], dtype=np.float32)
    inputs = {
        "derivative_save_model_x": values,
        "derivative_save_model_init": np.array([[[0.7]]], dtype=np.float32),
    }
    expected = {name: to_numpy(value) for name, value in model(inputs).items()}

    path = tmp_path / "derivative_model"
    model.save(path)
    restored = Modely.load(path)

    result = restored(inputs)
    for name, value in expected.items():
        np.testing.assert_allclose(to_numpy(result[name]), value, rtol=1e-5, atol=1e-5)


def test_derivative_export_keras_round_trip(tmp_path):
    _, _, model = _build_mixed_derivative_model("derivative_keras_model")
    model.build()

    values = np.array([[[0.4, 1.3]]], dtype=np.float32)
    inputs = {
        "derivative_keras_model_x": values,
        "derivative_keras_model_init": np.array([[[0.7]]], dtype=np.float32),
    }
    expected = {name: to_numpy(value) for name, value in model(inputs).items()}

    path = tmp_path / "derivative_export.keras"
    model.export_keras(path)
    restored = Modely.import_keras(path)

    result = restored(inputs, training=False)  # type: ignore
    for name, value in expected.items():
        np.testing.assert_allclose(to_numpy(result[name]), value, rtol=1e-5, atol=1e-5)


def test_derivative_export_onnx(tmp_path):
    """A differentiated graph carries its backward pass into ONNX; the batch
    axis is fixed for it, which is what the exporter can convert."""
    _, _, model = _build_mixed_derivative_model("derivative_onnx_model")
    model.build()

    values = np.array([[[0.4, 1.3]]], dtype=np.float32)
    inputs = {
        "derivative_onnx_model_x": values,
        "derivative_onnx_model_init": np.array([[[0.7]]], dtype=np.float32),
    }
    expected = {name: to_numpy(value) for name, value in model(inputs).items()}

    if keras.backend.backend() == "torch":
        # Torch traces a forward pass only, so it cannot record the backward
        # pass this graph evaluates - and says so instead of writing a model
        # whose gradient operators no runtime implements.
        with pytest.raises(NotImplementedError, match="torch"):
            model.export_onnx(tmp_path / "derivative_export.onnx")
        return

    path = model.export_onnx(tmp_path / "derivative_export.onnx")
    result = Modely.validate_onnx(str(path), inputs, return_dict=True)
    for name, value in expected.items():
        np.testing.assert_allclose(result[name], value, rtol=1e-4, atol=1e-4)  # type: ignore


def test_derivative_wrt_time_needs_no_fixed_batch_to_export():
    """Only automatic differentiation constrains the export: the finite
    difference is ordinary arithmetic, so the time mode leaves the batch axis
    exactly as the backend's exporter would."""
    v = Input("onnx_time_v", dim=1)
    time_model = Modely(
        "derivative_time_onnx_model",
        inputs=[v],
        outputs=[Output("dt", Derivative(order=1, respect_to=0.1)(v.sw(2)))],
    ).build()

    x = Input("onnx_grad_x", dim=1)
    fun = Sin()(x.last())
    gradient_model = Modely(
        "derivative_grad_onnx_model",
        inputs=[x],
        outputs=[Output("dx", Derivative(order=1, respect_to=x)(fun))],
    ).build()

    assert not _traces_backward_pass(time_model.model)
    assert _traces_backward_pass(gradient_model.model)

    with tempfile.TemporaryDirectory() as folder:
        path = time_model.export_onnx(f"{folder}/derivative_time.onnx")
        values = np.array([[[1.0, 2.0]]], dtype=np.float32)
        result = Modely.validate_onnx(
            str(path), {"onnx_time_v": values}, return_dict=True
        )
        np.testing.assert_allclose(
            result["dt"],  # type: ignore
            np.array([[[10.0, 10.0]]], dtype=np.float32),
            rtol=1e-5,
        )


def test_derivative_inside_a_rollback_model():
    """A derivative is an ordinary block of the graph, so unrolling the model
    re-evaluates it at every step with that step's own state."""
    x = Input("roll_x", dim=1)
    fun = x.last() ** 2
    derivative = Derivative(order=1, respect_to=x)(fun)  # 2x
    next_x = x.last() + derivative * 0.1  # 1.2 x

    model = Modely(
        "derivative_rollback_model",
        inputs=[x],
        outputs=[Output("next_x", next_x)],
    )
    model.rollback({x.name: "next_x"}, steps=3)
    model.build()

    result = to_numpy(
        model({"roll_x": np.array([[[1.0]]], dtype=np.float32)})["next_x"]
    )
    np.testing.assert_allclose(result, np.full((1, 1, 1), 1.2**3), rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_train_through_a_derivative():
    """Gradients reach the weights through the differentiation itself: the
    loss is written on d fun / d x, and only the Fir weight can move it.
    For a single-sample Fir the derivative *is* the weight, so it must
    converge to the target."""
    x = Input("dtrain_x", dim=1)
    fun = Fir(out_features=1, use_bias=False, name="dtrain_fir")([x.last()])
    derivative = Output("dx", Derivative(order=1, respect_to=x)(fun))

    model = Modely("derivative_train_model", inputs=[x], outputs=[derivative])
    model.minimize("slope", source=derivative, target=3.0, loss="mse")
    model.build()

    values = np.array([[[1.0]]], dtype=np.float32)
    before = float(to_numpy(model({"dtrain_x": values})["dx"]).ravel()[0])
    assert abs(before - 3.0) > 1e-3

    data = DataLoader(model, source={"dtrain_x": np.ones((16, 1), dtype=np.float32)})
    history = model.train(
        train_data=data, epochs=40, batch_size=4, lr=0.1, printer=None
    )

    after = float(to_numpy(model({"dtrain_x": values})["dx"]).ravel()[0])
    assert abs(after - 3.0) < abs(before - 3.0)
    np.testing.assert_allclose(after, 3.0, rtol=1e-2, atol=1e-2)
    assert history["loss"][-1] < history["loss"][0]


def _build_pos_vel_integrators(name_suffix, dt, mass):
    """F=ma -> integrate once for velocity, again for position, entirely as
    blocks of one model - no separate rate Modely, no state input needed.

    Integrate(f, solver=...) only returns the increment (dt-weighted rate),
    so the state update is composed explicitly with '+', same as the
    finite-difference Derivative/Integrate primitives.
    """
    force = Input(f"force_{name_suffix}", dim=1, sample_time=dt)
    vel = Input(f"vel_{name_suffix}", dim=1, sample_time=dt)
    pos = Input(f"pos_{name_suffix}", dim=1, sample_time=dt)

    acc_rate = force.last() / mass
    vel_increment = Integrate(acc_rate, solver="euler")
    vel_next = Add(name=f"vel_next_{name_suffix}")([vel.last(), vel_increment])

    # d(pos)/dt = vel (the current velocity, before this step's own update -
    # explicit Euler, not semi-implicit).
    pos_increment = Integrate(vel.last(), solver="euler")
    pos_next = Add(name=f"pos_next_{name_suffix}")([pos.last(), pos_increment])

    return force, vel, pos, vel_next, pos_next


def test_integrate_pos_vel():
    """A single Euler step of both integrators against hand-computed values."""
    dt, mass = 0.1, 2.0
    force, vel, pos, vel_next, pos_next = _build_pos_vel_integrators("single", dt, mass)

    model = Modely(
        "pos_vel_model",
        inputs=[vel, pos, force],
        outputs=[Output("vel_out", vel_next), Output("pos_out", pos_next)],
    ).build()

    vel0, pos0, force0 = 1.0, 0.0, 4.0
    result = model({vel.name: [vel0], pos.name: [pos0], force.name: [force0]})

    acc0 = force0 / mass
    expected_vel = vel0 + dt * acc0
    expected_pos = pos0 + dt * vel0  # uses vel0 (old vel), matching explicit Euler

    np.testing.assert_allclose(
        to_numpy(result["vel_out"]),
        np.full((1, 1, 1), expected_vel, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["pos_out"]),
        np.full((1, 1, 1), expected_pos, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_integrate_pos_vel_multi_step_rollback():
    """Roll both coupled integrators forward under a constant force and
    compare against the same discrete Euler recurrence computed in Python."""
    dt, mass, steps = 0.1, 2.0, 10
    force, vel, pos, vel_next, pos_next = _build_pos_vel_integrators("multi", dt, mass)

    model = Modely(
        "pos_vel_rollout_model",
        inputs=[vel, pos, force],
        outputs=[Output("vel_out", vel_next), Output("pos_out", pos_next)],
    )
    model.rollback({vel.name: vel_next.name, pos.name: pos_next.name}, steps=steps)
    model.build()

    vel0, pos0, force0 = 1.0, 0.0, 4.0
    result = model({vel.name: [vel0], pos.name: [pos0], force.name: [force0]})

    acc0 = force0 / mass
    v, x = vel0, pos0
    for _ in range(steps):
        x = x + dt * v  # uses the old v, same order as the model's own rollback
        v = v + dt * acc0

    np.testing.assert_allclose(
        to_numpy(result["vel_out"]),
        np.full((1, 1, 1), v, dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        to_numpy(result["pos_out"]),
        np.full((1, 1, 1), x, dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )


def test_integrate_step_euler():
    dt = 0.1
    v = Input("v", dim=1, sample_time=dt)
    increment = Integrate(v.last(), solver="euler")
    model = Modely(
        "integrate_euler_model", inputs=[v], outputs=[Output("increment", increment)]
    ).build()

    values = np.array([[[3.0]]], dtype=np.float32)
    result = to_numpy(model({"v": values})["increment"])
    expected_val = dt * 3.0
    np.testing.assert_allclose(
        result, np.array([[[expected_val]]], dtype=np.float32), rtol=1e-5, atol=1e-5
    )


def test_integrate_step_trapezoidal():
    dt = 0.1
    v = Input("v", dim=1, sample_time=dt)
    increment = Integrate(v.sw(2), solver="trapezoidal")
    model = Modely(
        "integrate_trap_model", inputs=[v], outputs=[Output("increment", increment)]
    ).build()

    values = np.array([[[2.0, 4.0]]], dtype=np.float32)
    result = to_numpy(model({"v": values})["increment"])
    expected_val = dt / 2.0 * (2.0 + 4.0)
    np.testing.assert_allclose(
        result, np.array([[[expected_val]]], dtype=np.float32), rtol=1e-5, atol=1e-5
    )


def test_integrate_step_heun_matches_trapezoidal_weights():
    """ "heun" is an alias for the same 2-sample trapezoidal quadrature."""
    dt = 0.1
    v = Input("v", dim=1, sample_time=dt)
    heun_increment = Integrate(v.sw(2), solver="heun")
    model = Modely(
        "integrate_heun_model",
        inputs=[v],
        outputs=[Output("increment", heun_increment)],
    ).build()

    values = np.array([[[2.0, 4.0]]], dtype=np.float32)
    result = to_numpy(model({"v": values})["increment"])
    expected_val = dt / 2.0 * (2.0 + 4.0)
    np.testing.assert_allclose(
        result, np.array([[[expected_val]]], dtype=np.float32), rtol=1e-5, atol=1e-5
    )


def test_integrate_step_over_arbitrary_layer_output():
    """f can be any Stream, not only a raw Input window - e.g. a relation
    computed over one."""
    dt = 0.1
    v = Input("v", dim=1, sample_time=dt)
    scaled = v.sw(2) * 2.0
    increment = Integrate(scaled, solver="trapezoidal")
    model = Modely(
        "integrate_arbitrary_layer_model",
        inputs=[v],
        outputs=[Output("increment", increment)],
    ).build()

    values = np.array([[[2.0, 4.0]]], dtype=np.float32)
    result = to_numpy(model({"v": values})["increment"])
    expected_val = dt / 2.0 * (2.0 * 2.0 + 2.0 * 4.0)
    np.testing.assert_allclose(
        result, np.array([[[expected_val]]], dtype=np.float32), rtol=1e-5, atol=1e-5
    )


def test_integrate_step_wrong_window_size_raises():
    v = Input("v", dim=1, sample_time=0.1)
    with pytest.raises(ValueError):
        Integrate(v.sw(2), solver="euler")
    with pytest.raises(ValueError):
        Integrate(v.last(), solver="trapezoidal")


def test_integrate_step_unknown_solver_raises():
    v = Input("v", dim=1, sample_time=0.1)
    with pytest.raises(ValueError):
        Integrate(v.last(), solver="rk4")


def test_integrate_step_requires_sample_time_or_explicit_dt():
    v = Input("v", dim=1)  # no sample_time set

    with pytest.raises(ValueError):
        Integrate(v.last(), solver="euler")

    increment = Integrate(v.last(), solver="euler", dt=0.5)
    model = Modely(
        "integrate_explicit_dt_model",
        inputs=[v],
        outputs=[Output("increment", increment)],
    ).build()

    values = np.array([[[2.0]]], dtype=np.float32)
    result = to_numpy(model({"v": values})["increment"])
    np.testing.assert_allclose(result, np.array([[[1.0]]], dtype=np.float32))


def test_integrate_step_save_load_round_trip(tmp_path):
    """f is now just a Stream/pred, part of this node's own graph - unlike
    the earlier Modely-embedding design, native save/load works directly."""
    dt = 0.1
    v = Input("v", dim=1, sample_time=dt)
    increment = Integrate(v.sw(2), solver="trapezoidal")
    model = Modely(
        "integrate_step_save_model",
        inputs=[v],
        outputs=[Output("increment", increment)],
    ).build()

    values = np.array([[[2.0, 4.0]]], dtype=np.float32)
    expected = model({"v": values})["increment"]

    path = tmp_path / "integrate_step_model"
    model.save(path)
    restored = Modely.load(path)

    np.testing.assert_allclose(
        to_numpy(restored({"v": values})["increment"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_integrate_step_export_keras_and_onnx(tmp_path):
    dt = 0.1
    v = Input("v", dim=1, sample_time=dt)
    increment = Integrate(v.last(), solver="euler")
    model = Modely(
        "integrate_step_export_model",
        inputs=[v],
        outputs=[Output("increment", increment)],
    ).build()

    values = np.array([[[3.0]]], dtype=np.float32)
    expected = model({"v": values})["increment"]

    keras_path = tmp_path / "integrate_step_export.keras"
    model.export_keras(keras_path)
    restored = Modely.import_keras(keras_path)
    np.testing.assert_allclose(
        to_numpy(restored({"v": values}, training=False)["increment"]),  # type: ignore
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    onnx_path = model.export_onnx(tmp_path / "integrate_step_export.onnx")
    onnx_result = Modely.validate_onnx(str(onnx_path), {"v": values}, return_dict=True)
    np.testing.assert_allclose(
        onnx_result["increment"],  # type: ignore
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )  # type: ignore


def test_integrate_cumulative():
    dt = 0.1
    v = Input("v", dim=1, sample_time=dt)
    cum = Integrate(solver="trapezoidal")(v.sw(4))
    model = Modely(
        "integrate_cum_model", inputs=[v], outputs=[Output("cum", cum)]
    ).build()

    samples = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
    values = samples.reshape(1, 1, 4)
    result = to_numpy(model({"v": values})["cum"])

    expected = np.zeros(4, dtype=np.float32)
    for i in range(1, 4):
        expected[i] = expected[i - 1] + dt / 2.0 * (samples[i - 1] + samples[i])
    expected = expected.reshape(1, 1, 4)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_integrate_cumulative_solver_aliases():
    """ "euler"/"heun" are valid cumulative-mode solvers too, aliasing the
    same "rectangular"/"trapezoidal" rules - one solver vocabulary for both
    forms of Integrate."""
    dt = 0.1
    v = Input("v", dim=1, sample_time=dt)
    cum_euler = Integrate(solver="euler")(v.sw(4))
    cum_rectangular = Integrate(solver="rectangular")(v.sw(4))
    cum_heun = Integrate(solver="heun")(v.sw(4))
    cum_trapezoidal = Integrate(solver="trapezoidal")(v.sw(4))
    model = Modely(
        "integrate_cum_alias_model",
        inputs=[v],
        outputs=[
            Output("euler", cum_euler),
            Output("rectangular", cum_rectangular),
            Output("heun", cum_heun),
            Output("trapezoidal", cum_trapezoidal),
        ],
    ).build()

    values = np.array([[[1.0, 2.0, 3.0, 5.0]]], dtype=np.float32)
    result = model({"v": values})
    np.testing.assert_allclose(
        to_numpy(result["euler"]), to_numpy(result["rectangular"]), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        to_numpy(result["heun"]), to_numpy(result["trapezoidal"]), rtol=1e-6, atol=1e-6
    )
