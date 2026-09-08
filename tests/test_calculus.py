import numpy as np
import pytest

from conftest import to_numpy
from nnodely import (
    Derivative,
    Input,
    Integrate,
    Modely,
    Output,
)
from nnodely.core.layer import Add


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


def test_derivative_order1():
    dt = 0.1
    x = Input("x", dim=1, sample_time=dt)
    dx = Derivative(order=1)(x.sw(2))
    model = Modely("deriv1_model", inputs=[x], outputs=[Output("dx", dx)]).build()

    values = np.array([[[2.0, 5.0]]], dtype=np.float32)
    result = to_numpy(model({"x": values})["dx"])
    expected = np.array([[[(5.0 - 2.0) / dt]]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_derivative_order2():
    dt = 0.1
    x = Input("x", dim=1, sample_time=dt)
    d2x = Derivative(order=2)(x.sw(3))
    model = Modely("deriv2_model", inputs=[x], outputs=[Output("d2x", d2x)]).build()

    x0, x1, x2 = 1.0, 2.0, 4.0
    values = np.array([[[x0, x1, x2]]], dtype=np.float32)
    result = to_numpy(model({"x": values})["d2x"])
    expected_val = (x2 - 2 * x1 + x0) / dt**2
    expected = np.array([[[expected_val]]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_derivative_requires_sample_time_or_explicit_dt():
    x = Input("x", dim=1)  # no sample_time set

    with pytest.raises(ValueError):
        Derivative(order=1)(x.sw(2))

    # An explicit dt overrides the missing Input.sample_time.
    dx = Derivative(order=1, dt=0.5)(x.sw(2))
    model = Modely(
        "deriv_explicit_dt_model", inputs=[x], outputs=[Output("dx", dx)]
    ).build()

    values = np.array([[[1.0, 2.0]]], dtype=np.float32)
    result = to_numpy(model({"x": values})["dx"])
    np.testing.assert_allclose(result, np.array([[[2.0]]], dtype=np.float32))


def test_derivative_wrong_window_size_raises():
    x = Input("x", dim=1, sample_time=0.1)
    with pytest.raises(ValueError):
        Derivative(order=1)(x.sw(3))


def test_derivative_over_arbitrary_layer_output():
    """Derivative no longer requires a bare Input window - any Stream with the
    right window length works, e.g. one produced by an upstream relation."""
    dt = 0.1
    x = Input("x", dim=1, sample_time=dt)
    shifted = x.sw(2) + 1.0  # an arbitrary relation, not a raw SampleWindow
    dx = Derivative(order=1)(shifted)
    model = Modely(
        "deriv_arbitrary_layer_model", inputs=[x], outputs=[Output("dx", dx)]
    ).build()

    values = np.array([[[2.0, 5.0]]], dtype=np.float32)
    result = to_numpy(model({"x": values})["dx"])
    # the constant +1 shift vanishes under finite differencing
    expected = np.array([[[(5.0 - 2.0) / dt]]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


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
    cum = Integrate(method="trapezoidal")(v.sw(4))
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


def test_derivative_save_load_round_trip(tmp_path):
    dt = 0.1
    x = Input("x", dim=1, sample_time=dt)
    dx = Derivative(order=1)(x.sw(2))
    model = Modely("deriv_save_model", inputs=[x], outputs=[Output("dx", dx)]).build()

    values = np.array([[[2.0, 5.0]]], dtype=np.float32)
    expected = model({"x": values})["dx"]

    path = tmp_path / "derivative_model"
    model.save(path)
    restored = Modely.load(path)

    np.testing.assert_allclose(
        to_numpy(restored({"x": values})["dx"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    restored_input = next(
        node for node in restored.flatten().order if isinstance(node, Input)
    )
    assert restored_input.sample_time == dt


def test_derivative_export_keras_and_onnx(tmp_path):
    """Derivative is a plain weight-based block of the same model - no
    separate keras model to rebuild - so it exports cleanly like every
    other layer, unlike the old autodiff-mode design it replaced."""
    dt = 0.1
    x = Input("x", dim=1, sample_time=dt)
    dx = Derivative(order=1)(x.sw(2))
    model = Modely("deriv_export_model", inputs=[x], outputs=[Output("dx", dx)]).build()

    values = np.array([[[2.0, 5.0]]], dtype=np.float32)
    expected = model({"x": values})["dx"]

    keras_path = tmp_path / "derivative_export.keras"
    model.export_keras(keras_path)
    restored = Modely.import_keras(keras_path)
    np.testing.assert_allclose(
        to_numpy(restored({"x": values}, training=False)["dx"]),  # type: ignore
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )

    onnx_path = model.export_onnx(tmp_path / "derivative_export.onnx")
    onnx_result = Modely.validate_onnx(str(onnx_path), {"x": values}, return_dict=True)
    np.testing.assert_allclose(
        onnx_result["dx"],  # type: ignore
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )  # type: ignore


def test_derivative_reflects_sample_time_changed_after_reload(tmp_path):
    """Changing Input.sample_time on a reloaded Modely and rebuilding must
    change Derivative's dt - the coefficients are re-resolved fresh from
    self.preds every build_layer() call, never cached at save time."""
    dt = 0.1
    x = Input("x", dim=1, sample_time=dt)
    dx = Derivative(order=1)(x.sw(2))
    model = Modely(
        "deriv_dt_reload_model", inputs=[x], outputs=[Output("dx", dx)]
    ).build()

    values = np.array([[[2.0, 5.0]]], dtype=np.float32)

    path = tmp_path / "deriv_dt_reload"
    model.save(path)
    restored = Modely.load(path)

    new_dt = 0.2
    restored.inputs[0].sample_time = new_dt
    restored.build()

    result = to_numpy(restored({"x": values})["dx"])
    expected = np.array([[[(5.0 - 2.0) / new_dt]]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
