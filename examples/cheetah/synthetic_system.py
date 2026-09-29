"""A synthetic stand-in for the HalfCheetah/MuJoCo dynamics, used until the
dataset comes from the real simulator.

It is structurally similar to HalfCheetah (same shapes, same physics roles)
and follows the constrained mechanical system equation the model is built on:

    q_ddot = M(q)^-1 (tau + J^T lambda(q, q_dot) - h(q, q_dot))

    q, q_dot in R^9   (dof 0-2: an unactuated "root" - x, z/height, pitch;
                       dof 3-8: 6 actuated "leg" joints, matching real
                       HalfCheetah's actuator count)
    tau in R^9        (zero at the unactuated root dof, a rhythmic gait-like
                       pattern at the leg dof)
    M(q)              a coupled, positive-definite 9x9 mass matrix: a fixed
                      SPD base plus a rank-1, q-dependent PSD perturbation
    h(q, q_dot)       Coriolis-like, spring, damping and gravity terms, in the
                      role of MuJoCo's qfrc_bias - qfrc_passive
    J^T lambda        a ReLU-gated ground-contact force at the height dof and
                      the two "foot" dof, active only when the base dips below
                      ground level - the quantity the network has to learn,
                      in the role of MuJoCo's qfrc_constraint

Integration is semi-implicit Euler, MuJoCo's Euler integrator:

    q_dot_next = q_dot + dt * q_ddot
    q_next     = q + dt * q_dot_next
"""

from __future__ import annotations

import numpy as np

N_DOF = 9
N_UNACTUATED = 3  # dof 0,1,2: root x, height (z), pitch
HEIGHT_DOF = 1
FOOT_DOF = (5, 8)  # "bfoot"/"ffoot"-like dof, matching real HalfCheetah's layout

DT = 0.01
GROUND_LEVEL = 0.0
CONTACT_STIFFNESS = 400.0

# Fixed SPD base mass matrix: bigger inertia at the root, lighter at the legs.
_BASE_MASS_DIAG = np.array(
    [2.5, 2.5, 0.5, 0.3, 0.25, 0.2, 0.3, 0.25, 0.2], dtype=np.float64
)
_M_BASE = np.diag(_BASE_MASS_DIAG)
_MASS_COUPLING = 0.15  # strength of the rank-1, q-dependent perturbation

_BIAS_GAIN = 0.4
_SPRING_K = np.array([0.0, 0.0, 0.0, 2.0, 1.5, 1.0, 2.0, 1.5, 1.0], dtype=np.float64)
_PASSIVE_DAMP = np.array(
    [0.5, 0.5, 0.5, 0.8, 0.6, 0.4, 0.8, 0.6, 0.4], dtype=np.float64
)
_DOF_DAMPING = np.array([0.2, 0.2, 0.1, 0.3, 0.3, 0.2, 0.3, 0.3, 0.2], dtype=np.float64)

# Gravity pulls the free-floating root down, so the "cheetah" actually falls
# toward the ground and triggers contact_force() - without this the height
# dof just free-floats (no restoring/attracting force at all) and the
# contact task never activates.
_GRAVITY = 9.8

_CONTACT_SHARE = {HEIGHT_DOF: 1.0, FOOT_DOF[0]: 0.5, FOOT_DOF[1]: 0.5}


def mass_matrix(q: np.ndarray) -> np.ndarray:
    """M(q): fixed SPD base + a rank-1, q-dependent PSD perturbation."""
    v_q = np.sin(q)
    return _M_BASE + _MASS_COUPLING * np.outer(v_q, v_q)


def bias_force(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """h(q, q_dot): a velocity-coupled ("Coriolis-like") term, per-dof
    springs and dampers, and gravity pulling the root down."""
    coriolis = _BIAS_GAIN * v * np.abs(v).sum()
    force = coriolis + _SPRING_K * q + (_PASSIVE_DAMP + _DOF_DAMPING) * v
    force[HEIGHT_DOF] += _GRAVITY * _BASE_MASS_DIAG[HEIGHT_DOF]
    return force


def contact_force(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """J^T lambda: ReLU-gated ground contact at the height/foot dof."""
    depth = max(GROUND_LEVEL - q[HEIGHT_DOF], 0.0)
    lam = np.zeros(N_DOF, dtype=np.float64)
    if depth > 0.0:
        push = CONTACT_STIFFNESS * depth
        for dof, share in _CONTACT_SHARE.items():
            lam[dof] += push * share
        # Contact also damps vertical velocity a little, like a real impact.
        lam[HEIGHT_DOF] -= 5.0 * min(v[HEIGHT_DOF], 0.0)
    return lam


def gait_tau(t: float, phase: np.ndarray, freq: float, amplitude: float) -> np.ndarray:
    """A rhythmic, gait-like control signal at the actuated (leg) dof only."""
    tau = np.zeros(N_DOF, dtype=np.float64)
    tau[N_UNACTUATED:] = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    return tau


def step(q, v, tau):
    """One semi-implicit Euler step of the constrained dynamics."""
    M = mass_matrix(q)
    h = bias_force(q, v)
    lam = contact_force(q, v)

    qdd = np.linalg.solve(M, tau + lam - h)
    v_next = v + DT * qdd
    q_next = q + DT * v_next
    return q_next, v_next, qdd, lam, M, h


def simulate_run(rng: np.random.Generator, n_steps: int) -> dict:
    """One episode: random initial state, a randomized gait, full physics log.

    Row t holds the state at time t and every quantity the step t -> t+1
    uses, so q[t+1] = q[t] + DT * qd[t+1] and qd[t+1] = qd[t] + DT * qdd[t].
    """
    q = rng.normal(0.0, 0.1, size=N_DOF)
    q[HEIGHT_DOF] = rng.uniform(0.05, 0.25)  # start just above ground
    v = rng.normal(0.0, 0.1, size=N_DOF)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=N_DOF - N_UNACTUATED)
    freq = rng.uniform(0.5, 1.5)
    amplitude = rng.uniform(2.0, 5.0)

    log = {
        name: np.empty((n_steps, *shape), dtype=np.float32)
        for name, shape in [
            ("q", (N_DOF,)),
            ("qd", (N_DOF,)),
            ("tau", (N_DOF,)),
            ("M", (N_DOF, N_DOF)),
            ("h", (N_DOF,)),
            ("JT_lambda", (N_DOF,)),
            ("qdd", (N_DOF,)),
        ]
    }

    for t in range(n_steps):
        tau = gait_tau(t * DT, phase, freq, amplitude)
        q_next, v_next, qdd, lam, M, h = step(q, v, tau)
        log["q"][t] = q
        log["qd"][t] = v
        log["tau"][t] = tau
        log["M"][t] = M
        log["h"][t] = h
        log["JT_lambda"][t] = lam
        log["qdd"][t] = qdd
        q, v = q_next, v_next

    return log


def generate_runs(n_runs: int, n_steps: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [simulate_run(rng, n_steps) for _ in range(n_runs)]
