"""Cheetah quadruped: a learned contact force inside the equation of a
constrained mechanical system, rolled out in closed loop.

One step of the model (build_step):

    q, q_dot  (last `window` samples, normalized)
      -> GRU encoder run over the window         (a Loop nested in the step)
      -> MLP on the last encoder state           -> J^T lambda
      -> q_ddot = M^-1 (tau + J^T lambda - h)     (LinearSolve)
      -> Integrate(q_ddot) -> q_dot*, Integrate(q_dot*) -> q*   (Euler)

The simulator (build_simulator) is an outer Loop over that step: q* and q_dot*
are fed back into the q and q_dot windows (the oldest sample is dropped, the
prediction appended), while tau, M and h are read from the dataset at every
step - they are what the simulator provides, and the model only has to learn
the contact term. Integrating the rate with Euler and then the new velocity
with Euler again is semi-implicit Euler, MuJoCo's Euler integrator.

Every Loop wraps its body's Keras model, so the simulator trains the weights
of the step itself, and the step called on its own uses them too. The
simulator rolls out a fixed `horizon`; a longer prediction chains calls, each
seeded with the last predicted q/q_dot windows - the whole state a step
carries, since the encoder starts from zero at every step.
"""

from __future__ import annotations

import keras
import numpy as np

from nnodely import (
    Concatenate,
    Constant,
    Input,
    Integrate,
    Linear,
    Loop,
    Modely,
    Output,
    Range,
    Sigmoid,
    Swish,
    Tanh,
    TimeSelect,
)
from nnodely.core.layer import Layer

N_DOF = 9


@keras.saving.register_keras_serializable(package="cheetah_example")
class LinearSolveImpl(keras.layers.Layer):
    """x = A^-1 b, with A [batch, n, n, time] and b [batch, n, time]."""

    def call(self, inputs):
        a, b = inputs
        a = keras.ops.transpose(a, (0, 3, 1, 2))
        b = keras.ops.expand_dims(keras.ops.transpose(b, (0, 2, 1)), axis=-1)
        x = keras.ops.squeeze(keras.ops.solve(a, b), axis=-1)
        return keras.ops.transpose(x, (0, 2, 1))


class LinearSolve(Layer):
    """Solves A x = b per sample, for A of dim (n, n) and b of dim n: the
    M^-1 of the dynamics, without forming the inverse."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def output_shape(self, *inputs):
        # The default probes the layer with zeros, which makes A singular.
        b = inputs[1]
        return tuple(b.dim), b.time, tuple(b.seq)

    def build_layer(self):
        return LinearSolveImpl(name=self.name)

    def get_config(self):
        return {"name": self.name}


def build_encoder_cell(n_features: int, window: int, hidden: int) -> Modely:
    """One GRU step, run over a window of `window` samples by a Loop.

    The cell reads the oldest sample of the window and also returns it as a
    callback output: the Loop drops the oldest sample and appends the returned
    one, so the window rotates and step k reads sample k, oldest first.
    """
    x = Input("enc_x", dim=n_features)
    state = Input("enc_state", dim=hidden)
    x_k = TimeSelect(0)(x.sw(window))
    s = state.last()

    gates = Sigmoid()(Linear(out_features=2 * hidden)([Concatenate()([x_k, s])]))
    update = Range(0, hidden)(gates)
    reset = Range(hidden, 2 * hidden)(gates)
    candidate = Tanh()(
        Linear(out_features=hidden)([x_k])
        + Linear(out_features=hidden, use_bias=False)([reset * s])
    )
    state_next = s + update * (candidate - s)

    return Modely(
        "gru_cell",
        inputs=[x, state],
        outputs=[
            Output("enc_state_next", state_next),
            Output("enc_x_oldest", x_k),
        ],
    ).build()


def build_step(
    dt: float,
    stats: dict,
    window: int = 5,
    hidden: int = 32,
    mlp_hidden: int = 64,
    mlp_layers: int = 2,
    name: str = "cheetah_step",
) -> Modely:
    """One step of the constrained dynamics with a learned contact term.

    Inputs: q, qd (windows of `window` samples), tau, M, h (current sample).
    Outputs: q_next, qd_next, JT_lambda, qdd. `stats` holds the mean and std
    (numpy arrays) of q, qd and JT_lambda, used to normalize the encoder input
    and to scale the MLP output to physical units.
    """
    q = Input("q", dim=N_DOF)
    qd = Input("qd", dim=N_DOF)
    tau = Input("tau", dim=N_DOF)
    mass = Input("M", dim=(N_DOF, N_DOF))
    h = Input("h", dim=N_DOF)

    q_n = (q.sw(window) - Constant("q_mean", value=stats["q_mean"].tolist())) * (
        Constant("q_inv_std", value=(1.0 / stats["q_std"]).tolist())
    )
    qd_n = (qd.sw(window) - Constant("qd_mean", value=stats["qd_mean"].tolist())) * (
        Constant("qd_inv_std", value=(1.0 / stats["qd_std"]).tolist())
    )

    cell = build_encoder_cell(2 * N_DOF, window, hidden)
    enc_state = Loop(
        f=cell,
        callback={"enc_state": "enc_state_next", "enc_x": "enc_x_oldest"},
        initial={"enc_state": 0.0, "enc_x": Concatenate()([q_n, qd_n])},
        length=window,
        collect=False,
        name="gru_encoder",
    )

    x = enc_state
    for _ in range(mlp_layers):
        x = Swish()(Linear(out_features=mlp_hidden)([x]))
    jt_lambda = Linear(out_features=N_DOF)([x]) * Constant(
        "lam_std", value=stats["lam_std"].tolist()
    ) + Constant("lam_mean", value=stats["lam_mean"].tolist())

    qdd = LinearSolve(name="mass_solve")(
        [mass.last(), tau.last() + jt_lambda - h.last()]
    )
    qd_next = Integrate(solver="euler", dt=dt, init=qd.last())(qdd)
    q_next = Integrate(solver="euler", dt=dt, init=q.last())(qd_next)

    return Modely(
        name,
        inputs=[q, qd, tau, mass, h],
        outputs=[
            Output("q_next", q_next),
            Output("qd_next", qd_next),
            Output("JT_lambda", jt_lambda),
            Output("qdd", qdd),
        ],
    ).build()


def build_simulator(
    step: Modely, horizon: int, stats: dict, name: str = "cheetah_sim"
) -> Modely:
    """Closed-loop rollout of `step` over `horizon` steps, with its objectives.

    Data, one row per time t of a run (see train.py): the q/qd windows seed the
    rollout at the first step, tau/M/h drive every step, and the targets are
    the next state and the contact force of each step. Each objective compares
    the whole trajectory, per dof in units of that dof's std.
    """
    window = next(node for node in step.inputs if node.name == "q").time

    q_seq = Input("q_seq", dim=N_DOF, seq=horizon)
    qd_seq = Input("qd_seq", dim=N_DOF, seq=horizon)
    tau_seq = Input("tau_seq", dim=N_DOF, seq=horizon)
    mass_seq = Input("M_seq", dim=(N_DOF, N_DOF), seq=horizon)
    h_seq = Input("h_seq", dim=N_DOF, seq=horizon)

    q_traj, qd_traj, jt_lambda_traj, _ = Loop(
        f=step,
        callback={"q": "q_next", "qd": "qd_next"},
        initial={"q": q_seq.sw(window), "qd": qd_seq.sw(window)},
        inputs={"tau": tau_seq.last(), "M": mass_seq.last(), "h": h_seq.last()},
        name="cheetah_loop",
    )

    q_out = Output("q_traj", q_traj)
    qd_out = Output("qd_traj", qd_traj)
    jt_lambda_out = Output("JT_lambda_traj", jt_lambda_traj)
    sim = Modely(
        name,
        inputs=[q_seq, qd_seq, tau_seq, mass_seq, h_seq],
        outputs=[q_out, qd_out, jt_lambda_out],
    )

    q_scale = Constant("q_loss_scale", value=(1.0 / stats["q_std"]).tolist())
    qd_scale = Constant("qd_loss_scale", value=(1.0 / stats["qd_std"]).tolist())
    lam_scale = Constant("lam_loss_scale", value=(1.0 / stats["lam_std"]).tolist())
    q_target = Input("q_next_seq", dim=N_DOF, seq=horizon)
    qd_target = Input("qd_next_seq", dim=N_DOF, seq=horizon)
    jt_lambda_target = Input("JT_lambda_seq", dim=N_DOF, seq=horizon)

    sim.minimize(
        "position",
        Output("q_traj_n", q_out * q_scale),
        Output("q_target_n", q_target.last() * q_scale),
    )
    sim.minimize(
        "velocity",
        Output("qd_traj_n", qd_out * qd_scale),
        Output("qd_target_n", qd_target.last() * qd_scale),
    )
    sim.minimize(
        "contact_force",
        Output("JT_lambda_traj_n", jt_lambda_out * lam_scale),
        Output("JT_lambda_target_n", jt_lambda_target.last() * lam_scale),
    )
    return sim.build()


def stats_from_runs(runs: list[dict]) -> dict[str, np.ndarray]:
    """Per-dof mean and std of q, qd and J^T lambda. A dof that never sees a
    contact force has zero std: flooring it keeps the MLP output there at
    ~0 and its normalized loss finite."""

    def moments(key):
        values = np.concatenate([run[key] for run in runs], axis=0)
        std = np.maximum(values.std(axis=0), 1e-6)
        return values.mean(axis=0).astype(np.float32), std.astype(np.float32)

    stats = {}
    for key, prefix in [("q", "q"), ("qd", "qd"), ("JT_lambda", "lam")]:
        stats[f"{prefix}_mean"], stats[f"{prefix}_std"] = moments(key)
    return stats
