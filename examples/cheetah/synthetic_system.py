"""A synthetic stand-in for the HalfCheetah/MuJoCo dynamics used by the
PyTorch reference in this folder (model.py/data.py/conf.py/utils.py/main.py).

We deliberately do NOT call MuJoCo: the reference's use_mujoco_dynamics=True
path calls MuJoCo's C++ physics engine directly inside training through a
hand-written finite-difference gradient, which isn't practical to replicate
in nnodely (no MuJoCo bindings, no generic "wrap an external non-
differentiable engine" mechanism). Instead this fabricates a structurally
similar - same shapes, same qualitative physics roles - 9-DOF system:

    q, v in R^9   (dof 0-2: an unactuated "root" - x, z/height, pitch;
                   dof 3-8: 6 actuated "leg" joints, matching real
                   HalfCheetah's actuator count)
    tau in R^9    (zero at the unactuated root dof, a rhythmic gait-like
                   pattern at the leg dof)
    M(q)          a genuinely coupled (not just diagonal), positive-definite
                   9x9 mass matrix: a fixed SPD base plus a rank-1,
                   q-dependent perturbation (v_q @ v_q.T is always PSD, so
                   the sum stays SPD by construction)
    bias(q, v)    a simple velocity-coupled ("Coriolis-like") term
    passive(q, v) per-dof spring-damper forces
    lam(q, v)     a ReLU-gated ground-contact force at the height dof and
                  the two "foot" dof, active only when height dips below
                  ground level - the nonlinear, state-dependent quantity the
                  network has to learn to predict, playing the same role as
                  the reference's neural fit to MuJoCo's contact solver.

Integration matches the reference's own scheme exactly (semi-implicit Euler
with implicitly-integrated joint damping):

    dv = (M(q) + dt*diag(damping))^-1 @ (tau + lam(q,v) + passive(q,v) - bias(q,v))
    v_next = v + dt*dv
    q_next = q + dt*v_next
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
    """Simple velocity-coupled ("Coriolis-like") term."""
    return _BIAS_GAIN * v * np.abs(v).sum()


def passive_force(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-dof spring-damper, plus gravity pulling the root down."""
    force = -_SPRING_K * q - _PASSIVE_DAMP * v
    force[HEIGHT_DOF] -= _GRAVITY * _BASE_MASS_DIAG[HEIGHT_DOF]
    return force


def contact_force(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ReLU-gated ground contact at the height/foot dof."""
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
    """One semi-implicit Euler step, matching the reference's own scheme."""
    M = mass_matrix(q)
    bias = bias_force(q, v)
    passive = passive_force(q, v)
    lam = contact_force(q, v)

    rhs = tau + lam + passive - bias
    lhs = M + DT * np.diag(_DOF_DAMPING)
    dv = np.linalg.solve(lhs, rhs)
    v_next = v + DT * dv
    q_next = q + DT * v_next
    return q_next, v_next, lam, M, bias, passive


def simulate_run(rng: np.random.Generator, n_steps: int) -> dict:
    """One episode: random initial state, a randomized gait, full physics log."""
    q = rng.normal(0.0, 0.1, size=N_DOF)
    q[HEIGHT_DOF] = rng.uniform(0.05, 0.25)  # start just above ground
    v = rng.normal(0.0, 0.1, size=N_DOF)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=N_DOF - N_UNACTUATED)
    freq = rng.uniform(0.5, 1.5)
    amplitude = rng.uniform(2.0, 5.0)

    state = np.empty((n_steps, 2 * N_DOF), dtype=np.float32)
    tau_log = np.empty((n_steps, N_DOF), dtype=np.float32)
    lam_log = np.empty((n_steps, N_DOF), dtype=np.float32)
    mass_log = np.empty((n_steps, N_DOF, N_DOF), dtype=np.float32)
    bias_log = np.empty((n_steps, N_DOF), dtype=np.float32)
    passive_log = np.empty((n_steps, N_DOF), dtype=np.float32)

    for t in range(n_steps):
        state[t, :N_DOF] = q
        state[t, N_DOF:] = v
        tau = gait_tau(t * DT, phase, freq, amplitude)
        tau_log[t] = tau

        q_next, v_next, lam, M, bias, passive = step(q, v, tau)
        lam_log[t] = lam
        mass_log[t] = M
        bias_log[t] = bias
        passive_log[t] = passive

        q, v = q_next, v_next

    return {
        "state": state,
        "tau": tau_log,
        "JT_lambda": lam_log,
        "M": mass_log,
        "bias": bias_log,
        "passive": passive_log,
    }


def generate_runs(n_runs: int, n_steps: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [simulate_run(rng, n_steps) for _ in range(n_runs)]
