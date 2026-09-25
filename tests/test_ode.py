import numpy as np
import pytest

from nnodely import (
    Constant,
    Cos,
    Input,
    Loop,
    Modely,
    Ode,
    Output,
    Parameter,
    Sin,
)
from conftest import to_numpy


def _decay(state):
    """dx/dt = -x, whose solution from x(0) = 1 is exp(-t)."""
    return -1.0 * state


def _integrate_decay(method, steps, horizon=1.0):
    """Integrate the decay over `horizon` with `steps` steps of the method."""
    x0 = Input(f"x0_{method}_{steps}", dim=1)
    state = x0.last()
    for _ in range(steps):
        state = Ode(_decay, state, horizon / steps, method=method)

    model = Modely(
        f"decay_{method}_{steps}",
        inputs=[x0],
        outputs=[Output(f"x_{method}_{steps}", state)],
    ).build()
    result = model({x0.name: np.ones((1, 1, 1), dtype=np.float32)})
    return float(to_numpy(result[f"x_{method}_{steps}"]).ravel()[0])


@pytest.mark.parametrize("method, order", [("euler", 1), ("midpoint", 2), ("heun", 2)])
def test_ode_convergence_order(method, order):
    # Halving the step must shrink the error by 2**order: that is what pins each
    # tableau to the method it claims to be.
    coarse = abs(_integrate_decay(method, 10) - np.exp(-1.0))
    fine = abs(_integrate_decay(method, 20) - np.exp(-1.0))
    assert np.log2(coarse / fine) == pytest.approx(order, abs=0.2)


def test_ode_rk4_accuracy():
    # Fourth order over ten steps already sits at the float32 noise floor, so the
    # order estimate is meaningless here and the absolute error is the check.
    assert _integrate_decay("rk4", 10) == pytest.approx(np.exp(-1.0), abs=1e-5)


def test_ode_unknown_method():
    with pytest.raises(ValueError, match="is not available"):
        Ode(_decay, Input("x_bad_method", dim=1).last(), 0.1, method="dopri5")


def test_ode_derivative_count_mismatch():
    x = Input("x_mismatch", dim=1).last()
    y = Input("y_mismatch", dim=1).last()
    with pytest.raises(ValueError, match="one derivative per state"):
        Ode(lambda a, b: [-1.0 * a], [x, y], 0.1)


@pytest.mark.slow
def test_ode_matches_hand_written_rk4():
    pos = Input(name="Xpos", dim=1)
    vel = Input(name="Xvelocity", dim=1)
    angle = Input(name="Xangle", dim=1)
    ang_vel = Input(name="Xangular_velocity", dim=1)
    force = Input(name="action", dim=1)

    g = Constant(name="g", value=9.81)
    dt = Constant(name="dt", value=0.02)

    # Shared parameters, so both branches integrate exactly the same dynamics.
    gear, m1, m2, len, b, d, Inertia = (
        Parameter(name=name, value=[value])
        for name, value in zip(
            "gear m1 m2 l b d I".split(),
            [1.3, 0.9, 0.4, 0.6, 0.15, 0.05, 0.08],
        )
    )

    def inv_pend(p, v, alpha, omega, u):
        sin_theta = Sin()(alpha)
        cos_theta = Cos()(alpha)
        I_eff = Inertia + m2 * len**2
        denom = (m1 + m2) * I_eff - (m2 * len * cos_theta) ** 2
        F = gear * u
        omega_dot = (
            (m1 + m2) * m2 * g * len * sin_theta
            - m2**2 * len**2 * omega**2 * sin_theta * cos_theta
            - (m1 + m2) * d * omega
            + m2 * len * b * v * cos_theta
            - m2 * len * cos_theta * F
        ) / denom
        v_dot = (
            I_eff * m2 * len * omega**2 * sin_theta
            - I_eff * b * v
            - m2**2 * len**2 * g * sin_theta * cos_theta
            + m2 * len * d * omega * cos_theta
            + F * I_eff
        ) / denom
        return [v, v_dot, omega, omega_dot]

    # The four stages spelled out by hand, as an application would write them.
    k1 = inv_pend(pos, vel, angle, ang_vel, force)
    k2 = inv_pend(
        pos + k1[0] * dt / 2,
        vel + k1[1] * dt / 2,
        angle + k1[2] * dt / 2,
        ang_vel + k1[3] * dt / 2,
        force,
    )
    k3 = inv_pend(
        pos + k2[0] * dt / 2,
        vel + k2[1] * dt / 2,
        angle + k2[2] * dt / 2,
        ang_vel + k2[3] * dt / 2,
        force,
    )
    k4 = inv_pend(
        pos + k3[0] * dt,
        vel + k3[1] * dt,
        angle + k3[2] * dt,
        ang_vel + k3[3] * dt,
        force,
    )
    hand = [
        state + (dt / 6) * (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index])
        for index, state in enumerate((pos, vel, angle, ang_vel))
    ]

    auto = Ode(
        inv_pend,
        [pos, vel, angle, ang_vel],
        dt,
        method="rk4",
        args=(force,),
    )

    names = ["pos", "vel", "angle", "ang_vel"]
    model = Modely(
        name="inv_pend_ode",
        inputs=[pos, vel, angle, ang_vel, force],
        outputs=[Output(f"hand_{n}", s) for n, s in zip(names, hand)]
        + [Output(f"auto_{n}", s) for n, s in zip(names, auto)],
    ).build()

    rng = np.random.default_rng(0)
    feed = {
        name: rng.normal(size=(4, 1, 1)).astype(np.float32)
        for name in ("Xpos", "Xvelocity", "Xangle", "Xangular_velocity", "action")
    }
    result = model(feed)
    for name in names:
        np.testing.assert_allclose(
            to_numpy(result[f"auto_{name}"]),
            to_numpy(result[f"hand_{name}"]),
            atol=1e-6,
        )


def test_ode_in_loop():
    steps, dt = 10, 0.05

    x = Input("x", dim=1)
    body_output = Output("x_next", Ode(_decay, x.last(), dt, method="rk4"))
    body = Modely("ode_body", inputs=[x], outputs=[body_output]).build()

    seed = Input("x_seq", dim=1, seq=steps)
    loop = Loop(f=body, callback={x: body_output}, initial={x: seed}, collect=False)
    model = Modely("ode_loop", inputs=[seed], outputs=[Output("x_end", loop)]).build()

    seed_values = np.zeros((1, 1, steps), dtype=np.float32)
    seed_values[..., 0] = 1.0
    result = to_numpy(model({"x_seq": seed_values})["x_end"])

    assert result.shape == (1, 1, 1)
    # Ten RK4 steps of 0.05 reach t = 0.5 with error far below the tolerance.
    assert float(result.ravel()[0]) == pytest.approx(np.exp(-steps * dt), abs=1e-5)
