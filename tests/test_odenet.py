import os

import numpy as np
import pytest

from nnodely import (
    Constant,
    DataLoader,
    Input,
    Linear,
    Modely,
    Ode,
    OdeNet,
    Output,
    Parameter,
)
from conftest import to_numpy

# The field from torchdiffeq's ode_demo.py: dy/dt = y @ A. Because
# A = -0.1 * I + 2 * J, the solution from y0 = [2, 0] is known in closed form.
TRUE_A = np.array([[-0.1, 2.0], [-2.0, -0.1]], dtype=np.float32)
Y0 = np.array([[[2.0], [0.0]]], dtype=np.float32)  # (batch, dim=2, time=1)


def _analytic(times):
    return np.exp(-0.1 * times)[:, None] * np.stack(
        [2 * np.cos(2 * times), 2 * np.sin(2 * times)], axis=-1
    )


def _spiral_field(name):
    x = Input(f"x_{name}", dim=2)
    relation = Linear(out_features=2, use_bias=False)(x.last())
    field = Modely(
        f"field_{name}", inputs=[x], outputs=[Output(f"dx_{name}", relation)]
    ).build()
    relation.kernel.assign(TRUE_A)
    return x, field


def _spiral_model(name, points, **kwargs):
    x, field = _spiral_field(name)
    t = Input(f"t_{name}", dim=1, seq=points)
    # The helper builds the adaptive solver; the fixed-step tests override it.
    kwargs.setdefault("method", "dopri5")
    trajectory = OdeNet(f=field, states={x: f"dx_{name}"}, t=t, **kwargs)
    model = Modely(
        f"odenet_{name}", inputs=[x, t], outputs=[Output(f"y_{name}", trajectory)]
    ).build()
    return model, trajectory, (x.name, t.name, f"y_{name}")


def _solve(model, names, times):
    x_name, t_name, y_name = names
    points = times.shape[0]
    result = model(
        {
            x_name: Y0,
            t_name: times.reshape(1, 1, 1, points).astype(np.float32),
        }
    )
    return to_numpy(result[y_name])[0, :, 0, :].T  # (points, dim)


def test_odenet_matches_analytic_solution():
    points = 6
    model, trajectory, names = _spiral_model("basic", points)
    assert trajectory.shape.dimensions == ((2,), 1, (points,))

    times = np.linspace(0.0, 5.0, points, dtype=np.float32)
    solved = _solve(model, names, times)

    assert solved.shape == (points, 2)
    # The first reported point is the initial condition, as in torchdiffeq.
    np.testing.assert_allclose(solved[0], Y0[0, :, 0], atol=1e-6)
    np.testing.assert_allclose(solved, _analytic(times), atol=1e-4)


def test_odenet_integrates_arbitrary_horizon_without_rebuild():
    # One built model, three horizons chosen at call time: the reported times are
    # tensor values, so the trained field is not tied to the training step size.
    points = 6
    model, _, names = _spiral_model("horizon", points)

    for horizon in (1.0, 5.0, 25.0):
        times = np.linspace(0.0, horizon, points, dtype=np.float32)
        solved = _solve(model, names, times)
        np.testing.assert_allclose(solved, _analytic(times), atol=1e-4)


def test_odenet_tolerance_controls_error():
    # Tightening the tolerance must actually buy accuracy, which is what
    # distinguishes an error-controlled march from a fixed step.
    points = 3
    times = np.linspace(0.0, 10.0, points, dtype=np.float32)
    errors = []
    for index, rtol in enumerate((1e-2, 1e-5)):
        model, _, names = _spiral_model(f"tol{index}", points, rtol=rtol, atol=rtol)
        solved = _solve(model, names, times)
        errors.append(np.abs(solved - _analytic(times)).max())

    assert errors[0] > 10 * errors[1]
    assert errors[1] < 1e-4


def _rotation_field():
    """dp/dt = q, dq/dt = -p: pure arithmetic, so no weights need sharing."""
    return lambda p, q: [q, -1.0 * p]


def test_odenet_beats_coarse_fixed_step():
    # The point of adaptivity: over a long horizon a coarse fixed step accumulates
    # phase error while a controlled march holds the tolerance. Same field both
    # times, written with arithmetic only so the two graphs cannot diverge.
    horizon, steps = 10.0, 10
    exact = np.array([np.cos(horizon), -np.sin(horizon)])

    p = Input("p_cmp", dim=1)
    q = Input("q_cmp", dim=1)
    field = Modely(
        "rotation_cmp",
        inputs=[p, q],
        outputs=[Output("dp_cmp", q.last()), Output("dq_cmp", -1.0 * p.last())],
    ).build()
    t = Input("t_cmp", dim=1, seq=2)
    p_traj, q_traj = OdeNet(
        f=field,
        states={p: "dp_cmp", q: "dq_cmp"},
        t=t,
        method="dopri5",
        rtol=1e-7,
        atol=1e-9,
    )
    adaptive = Modely(
        "odenet_cmp",
        inputs=[p, q, t],
        outputs=[Output("p_a", p_traj), Output("q_a", q_traj)],
    ).build()
    seed = {
        "p_cmp": np.ones((1, 1, 1), dtype=np.float32),
        "q_cmp": np.zeros((1, 1, 1), dtype=np.float32),
        "t_cmp": np.array([0.0, horizon], dtype=np.float32).reshape(1, 1, 1, 2),
    }
    result = adaptive(seed)
    adaptive_error = np.abs(
        np.array(
            [
                to_numpy(result["p_a"])[0, 0, 0, -1],
                to_numpy(result["q_a"])[0, 0, 0, -1],
            ]
        )
        - exact
    ).max()

    # Same field, ten fixed RK4 steps of 1.0s.
    p_fixed = Input("p_fix", dim=1)
    q_fixed = Input("q_fix", dim=1)
    states = [p_fixed.last(), q_fixed.last()]
    for _ in range(steps):
        states = Ode(_rotation_field(), states, horizon / steps, method="rk4")
    fixed = Modely(
        "ode_fixed_cmp",
        inputs=[p_fixed, q_fixed],
        outputs=[Output("p_f", states[0]), Output("q_f", states[1])],
    ).build()
    values = fixed(
        {
            "p_fix": np.ones((1, 1, 1), dtype=np.float32),
            "q_fix": np.zeros((1, 1, 1), dtype=np.float32),
        }
    )
    fixed_error = np.abs(
        np.array([to_numpy(values["p_f"])[0, 0, 0], to_numpy(values["q_f"])[0, 0, 0]])
        - exact
    ).max()

    print(f"adaptive={adaptive_error:.2e}  fixed rk4 dt=1.0={fixed_error:.2e}")
    assert adaptive_error < 1e-4
    assert fixed_error > 100 * adaptive_error


def test_odenet_multi_state():
    # Two scalar states with a rotation field: dp/dt = q, dq/dt = -p, so the
    # solution from (1, 0) is (cos t, -sin t).
    p = Input("p", dim=1)
    q = Input("q", dim=1)
    field = Modely(
        "rotation",
        inputs=[p, q],
        outputs=[Output("dp", q.last()), Output("dq", -1.0 * p.last())],
    ).build()

    points = 4
    t = Input("t_multi", dim=1, seq=points)
    p_traj, q_traj = OdeNet(
        f=field, states={p: "dp", q: "dq"}, t=t, method="dopri5", rtol=1e-7, atol=1e-9
    )
    model = Modely(
        "odenet_multi",
        inputs=[p, q, t],
        outputs=[Output("p_out", p_traj), Output("q_out", q_traj)],
    ).build()

    times = np.linspace(0.0, 3.0, points, dtype=np.float32)
    result = model(
        {
            "p": np.ones((1, 1, 1), dtype=np.float32),
            "q": np.zeros((1, 1, 1), dtype=np.float32),
            "t_multi": times.reshape(1, 1, 1, points).astype(np.float32),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["p_out"])[0, 0, 0, :], np.cos(times), atol=1e-4
    )
    np.testing.assert_allclose(
        to_numpy(result["q_out"])[0, 0, 0, :], -np.sin(times), atol=1e-4
    )


def test_odenet_rejects_driver_inputs():
    x = Input("x_driver", dim=1)
    u = Input("u_driver", dim=1)
    field = Modely(
        "driven",
        inputs=[x, u],
        outputs=[Output("dx_driver", x.last() + u.last())],
    ).build()
    t = Input("t_driver", dim=1, seq=3)
    with pytest.raises(ValueError, match="autonomous field"):
        OdeNet(f=field, states={x: "dx_driver"}, t=t)


def test_odenet_requires_static_time_count():
    x, field = _spiral_field("dynamic")
    t = Input("t_dynamic", dim=1, seq=(None,))
    with pytest.raises(ValueError, match="at least two reported times"):
        OdeNet(f=field, states={x: "dx_dynamic"}, t=t)


def test_odenet_keras_round_trip(tmp_path):
    points = 4
    model, _, names = _spiral_model("save", points)
    x_name, t_name, y_name = names
    times = np.linspace(0.0, 5.0, points, dtype=np.float32)
    before = _solve(model, names, times)

    path = str(tmp_path / "odenet")
    model.export_keras(path + ".keras")
    reloaded = Modely.import_keras(path, safe_mode=False)
    assert reloaded is not None

    after = to_numpy(
        reloaded(
            {
                x_name: Y0,
                t_name: times.reshape(1, 1, 1, points).astype(np.float32),
            }
        )[y_name]
    )[0, :, 0, :].T
    np.testing.assert_allclose(after, before, atol=1e-6)


def _linear_field(prefix, values):
    """Shared parameters expressing dy/dt = y @ A, started at `values`.

    The parameters are the shared nodes rather than a reused layer, because a
    weight-bearing layer instance cannot be applied once per Runge-Kutta stage.
    """
    entries = {
        key: Parameter(name=f"{prefix}_{key}", value=[value])
        for key, value in values.items()
    }

    def field(p, q):
        return [
            entries["a11"] * p + entries["a21"] * q,
            entries["a12"] * p + entries["a22"] * q,
        ]

    return field, entries


def test_odenet_trained_field_matches_analytic_solution():
    dt, span, points = 0.05, 10.0, 11
    sample_times = np.arange(0.0, span + dt / 2, dt, dtype=np.float32)
    reference = _analytic(sample_times)

    # One model for both halves of the job: the field is fitted through the
    # unrolled rk4 march, which carries gradients, and the very same built graph
    # is switched to the adaptive march once the weights are learned.
    field, _ = _linear_field("fit", dict.fromkeys(("a11", "a12", "a21", "a22"), 0.0))
    p = Input("p_fit", dim=1)
    q = Input("q_fit", dim=1)
    derivatives = field(p.last(), q.last())
    body = Modely(
        "fit_field",
        inputs=[p, q],
        outputs=[Output("dp_fit", derivatives[0]), Output("dq_fit", derivatives[1])],
    ).build()

    t = Input("t_fit", dim=1, seq=points)
    p_traj, q_traj = OdeNet(
        f=body,
        states={p: "dp_fit", q: "dq_fit"},
        t=t,
        method="rk4",
        steps=1,
        rtol=1e-7,
        atol=1e-9,
        name="fit_ode",
    )
    predicted_p = Output("p_fit_pred", p_traj)
    predicted_q = Output("q_fit_pred", q_traj)

    trainer = Modely(
        "spiral_fit", inputs=[p, q, t], outputs=[predicted_p, predicted_q]
    )
    trainer.minimize(
        "err_p",
        source=predicted_p,
        target=Input("p_fit_target", dim=1, seq=points),
        loss="mse",
    )
    trainer.minimize(
        "err_q",
        source=predicted_q,
        target=Input("q_fit_target", dim=1, seq=points),
        loss="mse",
    )
    trainer.build()
    trainer.export_html(os.path.join("html", "spiral_fit.html"))

    # The loader reads a seq-less input at the end of the window it aligns with,
    # so the seeds are shifted by the window to start the trajectory they predict.
    seed = np.vstack(
        [np.repeat(reference[:1], points - 1, axis=0), reference[: 1 - points]]
    )
    data = DataLoader(
        trainer,
        source={
            "p_fit": seed[:, 0],
            "q_fit": seed[:, 1],
            "t_fit": sample_times,
            "p_fit_target": reference[:, 0],
            "q_fit_target": reference[:, 1],
        },
    )
    np.testing.assert_allclose(
        to_numpy(data.dataset["p_fit"])[:, 0, 0],
        to_numpy(data.dataset["p_fit_target"])[:, 0, 0, 0],
        atol=1e-6,
    )

    trainer.train(train_data=data, epochs=200, batch_size=64, lr=0.05, optimizer="adam")

    assert body.model is not None
    learned = {
        layer.name.removeprefix("fit_"): float(
            np.asarray(layer.get_weights()[0]).ravel()[0]
        )
        for layer in body.model.layers
        if layer.name.startswith("fit_a") and layer.get_weights()
    }
    # dy/dt = y @ A, so the state equations read off the columns of A.
    expected = {
        "a11": TRUE_A[0, 0],
        "a21": TRUE_A[1, 0],
        "a12": TRUE_A[0, 1],
        "a22": TRUE_A[1, 1],
    }
    assert set(learned) == set(expected)
    for key, value in expected.items():
        assert learned[key] == pytest.approx(value, abs=1e-3)

    # The tableau is the only thing that changes: the same model and the same
    # weights, now marching adaptively over twice the horizon it was fitted on.
    trainer.model.get_layer("fit_ode").set_method("dopri5")
    grid = np.linspace(0.0, 2 * span, points, dtype=np.float32)
    # The losses are part of the trained model's signature, so the targets have
    # to be present even here, where only the integrated outputs are read.
    unused = np.zeros((1, 1, 1, points), dtype=np.float32)
    result = trainer(
        {
            "p_fit": Y0[:, 0:1, :],
            "q_fit": Y0[:, 1:2, :],
            "t_fit": grid.reshape(1, 1, 1, points).astype(np.float32),
            "p_fit_target": unused,
            "q_fit_target": unused,
        }
    )
    solved_trajectory = np.stack(
        [
            to_numpy(result["p_fit_pred"])[0, 0, 0, :],
            to_numpy(result["q_fit_pred"])[0, 0, 0, :],
        ],
        axis=-1,
    )
    np.testing.assert_allclose(solved_trajectory, _analytic(grid), atol=1e-3)

    # # plot 2d solved vs analytic
    # try:
    #     import matplotlib.pyplot as plt

    #     plt.figure()
    #     plt.plot(solved_trajectory[:, 0], solved_trajectory[:, 1], "o-", label="OdeNet")
    #     plt.plot(reference[:, 0], reference[:, 1], "x--", label="Analytic")
    #     plt.xlabel("p")
    #     plt.ylabel("q")
    #     plt.title("OdeNet vs Analytic Solution")
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    # except ImportError:
    #     pass


def test_odenet_fixed_step_matches_analytic_solution():
    # A fixed tableau covers each reported interval with `steps` equal substeps,
    # so the march is unrolled instead of controlled and still hits the solution.
    points = 6
    model, trajectory, names = _spiral_model("fixed", points, method="rk4", steps=20)
    assert trajectory.shape.dimensions == ((2,), 1, (points,))

    times = np.linspace(0.0, 5.0, points, dtype=np.float32)
    solved = _solve(model, names, times)

    np.testing.assert_allclose(solved[0], Y0[0, :, 0], atol=1e-6)
    np.testing.assert_allclose(solved, _analytic(times), atol=1e-4)


def test_odenet_set_method_switches_a_built_layer():
    # The tableau is read on every call, so a field fitted with a fixed step is
    # integrated adaptively afterwards without rebuilding the graph.
    points = 3
    times = np.linspace(0.0, 10.0, points, dtype=np.float32)
    model, trajectory, names = _spiral_model("switch", points, method="euler", steps=5)
    coarse = np.abs(_solve(model, names, times) - _analytic(times)).max()

    model.model.get_layer(trajectory.name).set_method("dopri5")
    adaptive = np.abs(_solve(model, names, times) - _analytic(times)).max()

    assert adaptive < 1e-4
    assert coarse > 100 * adaptive


def test_odenet_rejects_unknown_method():
    x, field = _spiral_field("badmethod")
    t = Input("t_badmethod", dim=1, seq=3)
    with pytest.raises(ValueError, match="is not available"):
        OdeNet(f=field, states={x: "dx_badmethod"}, t=t, method="dopri8")


def test_odenet_trains_a_field_holding_a_shared_layer():
    # The training path `Ode` cannot take: the field is a built Modely, so the
    # one Linear is applied at every Runge-Kutta stage with the same weights.
    dt, span = 0.05, 10.0
    sample_times = np.arange(0.0, span + dt / 2, dt, dtype=np.float32)
    reference = _analytic(sample_times)

    x = Input("x_shared", dim=2)
    relation = Linear(out_features=2, use_bias=False)(x.last())
    field = Modely(
        "field_shared", inputs=[x], outputs=[Output("dx_shared", relation)]
    ).build()

    # Two reported times one sample apart: the trajectory is [x(t), x(t + dt)].
    t = Constant("t_shared", value=np.array([[[0.0, dt]]], dtype=np.float32))
    trajectory = OdeNet(
        f=field, states={x: "dx_shared"}, t=t, method="rk4", steps=1, name="shared_ode"
    )
    predicted = Output("x_shared_pred", trajectory)
    target = Input("x_shared_target", dim=2, seq=2)

    trainer = Modely("odenet_train", inputs=[x, target], outputs=[predicted])
    trainer.minimize("err", source=predicted, target=target, loss="mse")
    trainer.build()

    # The loader reads a seq-less input at the end of the window it aligns with,
    # so x carries the previous sample and seeds the window it starts.
    data = DataLoader(
        trainer,
        source={
            "x_shared": np.vstack([reference[:1], reference[:-1]]),
            "x_shared_target": reference,
        },
    )
    np.testing.assert_allclose(
        to_numpy(data.dataset["x_shared"])[:, :, 0],
        to_numpy(data.dataset["x_shared_target"])[:, :, 0, 0],
        atol=1e-6,
    )

    history = trainer.train(
        train_data=data, epochs=200, batch_size=64, lr=0.05, optimizer="adam"
    )
    assert history["loss"][-1] < history["loss"][0]
    np.testing.assert_allclose(to_numpy(relation.kernel), TRUE_A, atol=1e-3)

    # The trained layer keeps its weights across the swap, so the same model
    # integrates adaptively once the fitting is done.
    trainer.model.get_layer("shared_ode").set_method("dopri5")
    rolled = to_numpy(trainer(data.as_dict())["x_shared_pred"])[:, :, 0, -1]
    np.testing.assert_allclose(
        rolled, to_numpy(data.dataset["x_shared_target"])[:, :, 0, -1], atol=1e-3
    )


# The bouncing ball from torchdiffeq's bouncing_ball.py: free fall until the
# ball reaches the floor, then the velocity flips and loses a fixed fraction.
# Between two impacts the motion is a parabola, so the trajectory and its
# sensitivity to gravity are both known in closed form.
DROP, FLOOR, RESTITUTION, GRAVITY = 10.0, 0.2, 0.8, 9.8


def _exact_bounces(
    times, height=DROP, gravity=GRAVITY, floor=FLOOR, restitution=RESTITUTION
):
    """Position and velocity of the ideal ball, one parabola per bounce."""
    position, velocity = np.empty_like(times), np.empty_like(times)
    start, speed = 0.0, 0.0
    while True:
        # When this parabola reaches the floor: height + speed*s - g/2 s^2 = floor.
        fall = (speed + np.sqrt(speed**2 + 2 * gravity * (height - floor))) / gravity
        inside = (times >= start) & (times <= start + fall)
        elapsed = times[inside] - start
        position[inside] = height + speed * elapsed - 0.5 * gravity * elapsed**2
        velocity[inside] = speed - gravity * elapsed
        if start + fall > times[-1]:
            return position, velocity
        # The impact itself: the next parabola starts on the floor, going up.
        start, height = start + fall, floor
        speed = restitution * (gravity * fall - speed)


def _exact_ball(time, gravity=GRAVITY):
    """Position at a single `time`."""
    return _exact_bounces(np.array([time], dtype=np.float64), gravity=gravity)[0][0]


def _ball_field(name, gravity=GRAVITY, floor=FLOOR, restitution=RESTITUTION):
    position = Input(f"p_{name}", dim=1)
    velocity = Input(f"v_{name}", dim=1)
    gravity = Parameter(name=f"g_{name}", value=[gravity])
    floor = Parameter(name=f"r_{name}", value=[floor])
    restitution = Parameter(name=f"e_{name}", value=[restitution])
    field = Modely(
        f"ball_{name}",
        inputs=[position, velocity],
        outputs=[
            Output(f"dp_{name}", velocity.last()),
            # The state carries the batch axis the parameter alone does not.
            Output(f"dv_{name}", 0.0 * position.last() - gravity),
            Output(f"floor_{name}", position.last() - floor),
            Output(f"p_plus_{name}", position.last()),
            Output(f"v_plus_{name}", -1.0 * restitution * velocity.last()),
        ],
    ).build()
    return position, velocity, field


def _ball_model(name, points, steps, **kwargs):
    position, velocity, field = _ball_field(name)
    t = Input(f"t_{name}", dim=1, seq=points)
    trajectory, _ = OdeNet(
        f=field,
        states={position: f"dp_{name}", velocity: f"dv_{name}"},
        t=t,
        steps=steps,
        event=f"floor_{name}",
        reset={position: f"p_plus_{name}", velocity: f"v_plus_{name}"},
        name=f"bounce_{name}",
        **kwargs,
    )
    model = Modely(
        f"drop_{name}",
        inputs=[position, velocity, t],
        outputs=[Output(f"pos_{name}", trajectory)],
    ).build()
    return model, field, (position.name, velocity.name, t.name, f"pos_{name}")


def _drop(model, names, times):
    p_name, v_name, t_name, pos_name = names
    result = model(
        {
            p_name: np.full((1, 1, 1), DROP, dtype=np.float32),
            v_name: np.zeros((1, 1, 1), dtype=np.float32),
            t_name: times.reshape(1, 1, 1, -1).astype(np.float32),
        }
    )
    return to_numpy(result[pos_name]).ravel()


def test_odenet_event_bounces_off_the_floor():
    # Three reported times spanning the first impact at sqrt(2 * 9.8 / 9.8).
    times = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    model, _, names = _ball_model("bounce", points=times.size, steps=100)
    positions = _drop(model, names, times)

    assert positions.min() >= FLOOR - 1e-4
    np.testing.assert_allclose(
        positions, [_exact_ball(time) for time in times], atol=1e-3
    )


def test_odenet_event_time_reaches_the_gradient():
    # The point of locating the crossing inside the step: gravity moves the
    # impact time, and that term has to show up in the gradient. Resetting at
    # the end of whichever step crosses leaves a bias of about a third here,
    # and refining the step does not remove it.
    import tensorflow as tf  # the suite pins the tensorflow backend

    horizon = 3.0
    times = np.array([0.0, horizon], dtype=np.float32)
    model, field, names = _ball_model("grad", points=times.size, steps=200)
    p_name, v_name, t_name, pos_name = names
    gravity = next(
        variable
        for variable in field.model.trainable_variables
        if "g_grad" in variable.path
    )

    with tf.GradientTape() as tape:
        final = model.model(
            {
                p_name: tf.constant(np.full((1, 1, 1), DROP, dtype=np.float32)),
                v_name: tf.constant(np.zeros((1, 1, 1), dtype=np.float32)),
                t_name: tf.constant(times.reshape(1, 1, 1, -1)),
            }
        )[pos_name][0, 0, 0, -1]
    computed = np.ravel(tape.gradient(final, gravity).numpy())[0]

    step = 1e-3
    expected = (
        _exact_ball(horizon, GRAVITY + step) - _exact_ball(horizon, GRAVITY - step)
    ) / (2 * step)
    np.testing.assert_allclose(computed, expected, rtol=1e-2)


@pytest.mark.slow
def test_odenet_fits_the_event_along_with_the_field():
    """The counterpart of torchdiffeq's bouncing_ball.py, fitted end to end.

    Three numbers describe the ball, and each sits in a different part of the
    hybrid model: `gravity` in the vector field, `floor` in the event function,
    `restitution` in the reset. All three are parameters of one Modely, so a
    single train call fits the continuous dynamics and the discrete jump
    together. The floor only moves because the crossing is located inside the
    step - freeze the impact on the step grid and its gradient dies with it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt, span, points, height = 0.02, 4.0, 11, 3.0
    times = np.arange(0.0, span + dt / 2, dt, dtype=np.float32)
    reference = np.stack(_exact_bounces(times, height=height), axis=-1)

    # The guard has to be crossed for the floor to receive a gradient, so the
    # guess sits above the true floor: the ball then bounces too early, which is
    # an error the fit can see. A floor guessed far below the lowest point the
    # ball ever reaches would never trigger the event, and would never move.
    guess = {"g_fit": 7.5, "r_fit": 0.45, "e_fit": 0.5}
    truth = {"g_fit": GRAVITY, "r_fit": FLOOR, "e_fit": RESTITUTION}
    position, velocity, field = _ball_field(
        "fit",
        gravity=guess["g_fit"],
        floor=guess["r_fit"],
        restitution=guess["e_fit"],
    )

    window = Input("t_fit", dim=1, seq=points)
    predicted_p, _ = OdeNet(
        f=field,
        states={position: "dp_fit", velocity: "dv_fit"},
        t=window,
        method="rk4",
        steps=1,
        event="floor_fit",
        reset={position: "p_plus_fit", velocity: "v_plus_fit"},
        name="fit_bounce",
    )
    p_out = Output("p_fit_pred", predicted_p)

    # The loss is on the position alone, which stays continuous through the
    # bounce. Velocity jumps there, so a timing error off by a fraction of a
    # step costs twice the impact speed, and that spike swamps every other
    # gradient: adding it to the loss sends the fit to a negative restitution.
    trainer = Modely("ball_fit", inputs=[position, velocity, window], outputs=[p_out])
    trainer.minimize(
        "err_p",
        source=p_out,
        target=Input("p_fit_target", dim=1, seq=points),
        loss="mse",
    )
    trainer.build()

    # A seq-less input is read at the end of the window it aligns with, so the
    # seeds are shifted by the window to start the trajectory they predict.
    seed = np.vstack(
        [np.repeat(reference[:1], points - 1, axis=0), reference[: 1 - points]]
    )
    data = DataLoader(
        trainer,
        source={
            "p_fit": seed[:, 0],
            "v_fit": seed[:, 1],
            "t_fit": times,
            "p_fit_target": reference[:, 0],
        },
    )
    history = trainer.train(
        train_data=data, epochs=500, batch_size=32, lr=5e-3, optimizer="adam"
    )
    assert history["loss"][-1] < history["loss"][0] / 100

    learned = {
        name: float(np.ravel(field.model.get_layer(name).get_weights()[0])[0])
        for name in guess
    }
    for name, value in truth.items():
        assert learned[name] == pytest.approx(value, abs=2e-2)

    # The whole span from the drop alone, so the fitted bounces have to line up
    # with the exact ones and not just with the window they were fitted on.
    grid = Input("t_full", dim=1, seq=times.size)
    full_p, full_v = OdeNet(
        f=field,
        states={position: "dp_fit", velocity: "dv_fit"},
        t=grid,
        method="rk4",
        steps=1,
        event="floor_fit",
        reset={position: "p_plus_fit", velocity: "v_plus_fit"},
        name="rollout_bounce",
    )
    rollout = Modely(
        "ball_rollout",
        inputs=[position, velocity, grid],
        outputs=[Output("p_full", full_p), Output("v_full", full_v)],
    ).build()
    result = rollout(
        {
            "p_fit": np.full((1, 1, 1), height, dtype=np.float32),
            "v_fit": np.zeros((1, 1, 1), dtype=np.float32),
            "t_full": times.reshape(1, 1, 1, -1),
        }
    )
    predicted = (to_numpy(result["p_full"]).ravel(), to_numpy(result["v_full"]).ravel())
    assert np.abs(predicted[0] - reference[:, 0]).max() < 0.1

    figure, (upper, lower) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    upper.plot(times, reference[:, 0], color="black", label="exact")
    upper.plot(times, predicted[0], "--", color="tab:red", label="fitted OdeNet")
    upper.axhline(FLOOR, color="black", lw=0.6, alpha=0.4)
    upper.axhline(
        learned["r_fit"], ls=":", color="tab:red", lw=1.0, label="learned floor"
    )
    upper.set_ylabel("position [m]")
    upper.legend(loc="upper right")
    lower.plot(times, reference[:, 1], color="black")
    lower.plot(times, predicted[1], "--", color="tab:red")
    lower.set_xlabel("time [s]")
    lower.set_ylabel("velocity [m/s]")
    figure.suptitle(
        f"gravity {learned['g_fit']:.3f}  floor {learned['r_fit']:.3f}  "
        f"restitution {learned['e_fit']:.3f}   "
        f"(true {GRAVITY}, {FLOOR}, {RESTITUTION})"
    )
    figure.tight_layout()
    os.makedirs("html", exist_ok=True)
    figure.savefig(os.path.join("html", "bouncing_ball.png"), dpi=130)
    plt.close(figure)


def test_odenet_event_rejects_the_adaptive_march():
    position, velocity, field = _ball_field("adaptive")
    t = Input("t_adaptive", dim=1, seq=2)
    with pytest.raises(ValueError, match="events need a fixed tableau"):
        OdeNet(
            f=field,
            states={position: "dp_adaptive", velocity: "dv_adaptive"},
            t=t,
            method="dopri5",
            event="floor_adaptive",
            reset={position: "p_plus_adaptive", velocity: "v_plus_adaptive"},
        )


def test_odenet_event_blocks_the_method_swap():
    # Swapping a trained field to the adaptive march is the documented move, but
    # that march has no event handling, so it must not go through silently.
    times = np.array([0.0, 1.0], dtype=np.float32)
    model, _, _ = _ball_model("swap", points=times.size, steps=2)
    with pytest.raises(ValueError, match="cannot switch to 'dopri5'"):
        model.model.get_layer("bounce_swap").set_method("dopri5")


def test_odenet_event_requires_a_reset():
    position, velocity, field = _ball_field("noreset")
    t = Input("t_noreset", dim=1, seq=2)
    with pytest.raises(ValueError, match="event and reset go together"):
        OdeNet(
            f=field,
            states={position: "dp_noreset", velocity: "dv_noreset"},
            t=t,
            event="floor_noreset",
        )


def test_odenet_reset_must_cover_every_state():
    position, velocity, field = _ball_field("partial")
    t = Input("t_partial", dim=1, seq=2)
    with pytest.raises(ValueError, match="reset must map every state"):
        OdeNet(
            f=field,
            states={position: "dp_partial", velocity: "dv_partial"},
            t=t,
            event="floor_partial",
            reset={velocity: "v_plus_partial"},
        )


if __name__ == "__main__":
    test_odenet_fits_the_event_along_with_the_field()
