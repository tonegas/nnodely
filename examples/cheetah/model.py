"""nnodely port of the PyTorch ContactModel (../model.py) - a neural
contact-force predictor coupled to a physics integrator, trained
autoregressively.

Deliberate departures from the reference, each forced by a real gap between
what the reference needs and what nnodely currently offers:

- Physics runs on the dataset's cached mass/bias/passive (the reference's
  own use_mujoco_dynamics=False path), not live MuJoCo calls - no MuJoCo
  bindings exist in nnodely, and wrapping MuJoCo in a custom differentiable
  op is a much bigger undertaking than this example warrants.
- No LSTM history encoder: nnodely has no recurrent layer. The contact-force
  predictor here uses only the current (state, tau) features - a real
  architectural simplification, not a like-for-like swap.
- The 9x9 linear solve (dv = (M + dt*diag(damping))^-1 @ rhs) has no nnodely
  layer either, so it's one small custom keras.layers.Layer here (not added
  to nnodely's own layers/ - this is example-specific, not a general
  framework feature). It also does the q/v split internally via plain
  Keras slicing, matching DynamicsStepImpl's other internal splits.
- Roll (used for the autoregressive rollout, exactly as validated in
  examples/dualtank/system_id.py) holds every non-callback input fixed
  across all rolled steps - so mass/bias/passive, which really depend on
  the evolving q (and v for bias), are held at their seed-window value for
  the whole rollout. The reference's own use_mujoco_dynamics=False path has
  a related (if less severe, since it at least uses a fresh cached value
  per step) acknowledged approximation - see conf.py's comment on it. Since
  this makes a genuinely time-varying tau/mass/bias/passive rollout
  unrealistic to train against, training itself uses a single teacher step
  (build_training_model, rollout_steps=1 in practice) and Roll is reserved
  for qualitative multi-step assessment after training.
- nnodely's minimize() has no per-loss weight argument, so the reference's
  differential position_loss_weight=1.0/velocity_loss_weight=0.1 is folded
  directly into the velocity stream (scaled by sqrt(0.1) before an
  otherwise-equal-weight MSE) rather than passed as a loss weight.
- minimize()'s target must resolve to a dataset column through a chain of
  single-pred nodes (Modely.train's _resolve_label_name) - a Subtract/
  Divide normalization chain (2 preds each) can't satisfy that, so targets
  are fed pre-normalized(-and-scaled) as their own Inputs (train.py's
  dataset prep does the normalizing, with this same `stats`) rather than
  derived inline from a raw target Input.
- Calling an already-built Modely with fresh Streams (as in
  test_model_composition.py) does not reuse its trained Keras layers - it
  rebuilds a fresh, freshly-initialized copy of the composed graph (verified
  empirically: the composed layers get distinct auto-numbered names and
  distinct Variable objects from the original). So training instead builds
  its own structurally-identical copy of the step graph (_build_step_graph,
  called twice) and copies the trained weights across positionally
  afterwards (train.py, via trainable_weights + assign()) - this lines up
  because both graphs are built by the exact same helper and so contain the
  same trainable layers in the same construction order.
"""

from __future__ import annotations

import keras
import numpy as np

from nnodely import Input, Linear, Modely, Output, Parameter, Range, Swish
from nnodely.core.layer import Layer
from nnodely.layers.roll import Roll
from nnodely.layers.time_ops import Concatenate

N_DOF = 9
STATE_DIM = 2 * N_DOF


@keras.saving.register_keras_serializable(package="cheetah_example")
class DynamicsStepImpl(keras.layers.Layer):
    """One semi-implicit Euler step from (state, tau, lam, M, bias, passive)
    to a combined next state [q_next, v_next] - matching the reference
    dynamics_step's math exactly, for frame_skip=1 (no MuJoCo substepping).
    Splits state into q/v internally via plain slicing."""

    def __init__(self, dt, damping, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dt = float(dt)
        self.damping = tuple(float(d) for d in damping)

    def call(self, inputs):
        state, tau, lam, mass, bias, passive = inputs
        # nnodely tensor convention: [batch, dim..., time]; time=1 is always
        # the trailing axis regardless of dim's rank, so squeeze it off
        # uniformly (mass has dim=(9,9), everything else dim=(9,) or (18,)).
        state = keras.ops.squeeze(state, axis=-1)
        tau = keras.ops.squeeze(tau, axis=-1)
        lam = keras.ops.squeeze(lam, axis=-1)
        mass = keras.ops.squeeze(mass, axis=-1)
        bias = keras.ops.squeeze(bias, axis=-1)
        passive = keras.ops.squeeze(passive, axis=-1)

        q = state[:, :N_DOF]
        v = state[:, N_DOF:]

        damping = keras.ops.convert_to_tensor(self.damping, dtype=mass.dtype)
        lhs = mass + self.dt * keras.ops.diag(damping)[None, :, :]
        rhs = tau + lam + passive - bias
        dv = keras.ops.solve(lhs, rhs)

        v_next = v + self.dt * dv
        q_next = q + self.dt * v_next
        state_next = keras.ops.concatenate([q_next, v_next], axis=-1)
        return keras.ops.expand_dims(state_next, axis=-1)

    def compute_output_shape(self, input_shapes):
        batch = input_shapes[0][0]
        return (batch, STATE_DIM, 1)

    def get_config(self):
        config = super().get_config()
        config.update({"dt": self.dt, "damping": self.damping})
        return config


class DynamicsStep(Layer):
    """nnodely Layer wrapper around DynamicsStepImpl."""

    def __init__(self, dt: float, damping, name=None):
        self.dt = float(dt)
        self.damping = tuple(float(d) for d in damping)
        super().__init__(name=name, dt=self.dt, damping=self.damping)

    def build_layer(self):
        return DynamicsStepImpl(dt=self.dt, damping=self.damping, name=self.name)

    def get_config(self):
        return {"name": self.name, "dt": self.dt, "damping": list(self.damping)}


def _mlp(x, hidden_dim, out_dim, n_hidden_layers, name_prefix):
    for i in range(n_hidden_layers):
        x = Swish(name=f"{name_prefix}_act{i}")(
            Linear(out_features=hidden_dim, name=f"{name_prefix}_lin{i}")(x)
        )
    return Linear(out_features=out_dim, name=f"{name_prefix}_out")(x)


def _build_step_graph(
    dt: float,
    damping: np.ndarray,
    stats: dict,
    hidden_dim: int,
    mlp_layers: int,
    velocity_loss_weight: float,
):
    """Builds one fresh copy of the step architecture's symbolic graph -
    called once for the plain inference/rollout body (build_contact_step)
    and once more (a structurally-identical but Keras-object-distinct copy)
    for the training body (build_training_model), so trained weights can
    later be copied across positionally via trainable_weights + assign()."""
    state = Input("state", dim=STATE_DIM, sample_time=dt)
    tau = Input("tau", dim=N_DOF, sample_time=dt)
    mass = Input("mass", dim=(N_DOF, N_DOF), sample_time=dt)
    bias = Input("bias", dim=N_DOF, sample_time=dt)
    passive = Input("passive", dim=N_DOF, sample_time=dt)

    state_mean = Parameter("state_mean", value=stats["state_mean"].tolist())
    state_std = Parameter("state_std", value=stats["state_std"].tolist())
    tau_mean = Parameter("tau_mean", value=stats["tau_mean"].tolist())
    tau_std = Parameter("tau_std", value=stats["tau_std"].tolist())
    lam_mean = Parameter("lam_mean", value=stats["lam_mean"].tolist())
    lam_std = Parameter("lam_std", value=stats["lam_std"].tolist())

    state_last = state.last()
    tau_last = tau.last()

    state_n = (state_last - state_mean) / state_std
    tau_n = (tau_last - tau_mean) / tau_std
    features = Concatenate(axis=0, name="features")([state_n, tau_n])

    lam_n = _mlp(features, hidden_dim, N_DOF, mlp_layers, "force_head")
    lam = lam_n * lam_std + lam_mean

    state_next = DynamicsStep(dt=dt, damping=damping, name="dynamics_step")(
        [state_last, tau_last, lam, mass.last(), bias.last(), passive.last()]
    )

    state_next_n = (state_next - state_mean) / state_std
    vel_scale = float(np.sqrt(velocity_loss_weight))
    pos_pred_n = Range(0, N_DOF, name="pos_pred_n")(state_next_n)
    vel_pred_n = Range(N_DOF, STATE_DIM, name="vel_pred_n")(state_next_n) * vel_scale

    return {
        "state": state,
        "tau": tau,
        "mass": mass,
        "bias": bias,
        "passive": passive,
        "state_next": state_next,
        "lam_n": lam_n,
        "pos_pred_n": pos_pred_n,
        "vel_pred_n": vel_pred_n,
    }


def build_contact_step(
    dt: float,
    damping: np.ndarray,
    stats: dict,
    hidden_dim: int = 64,
    mlp_layers: int = 2,
    velocity_loss_weight: float = 0.1,
    name: str = "cheetah_step",
) -> Modely:
    """Plain inference/rollout body: predicts the contact force from the
    current (state, tau), then integrates the physics one dt forward.
    `stats` supplies state/tau/lam mean+std (numpy arrays) for normalization.
    No training targets - safe to wrap in Roll."""
    g = _build_step_graph(
        dt, damping, stats, hidden_dim, mlp_layers, velocity_loss_weight
    )

    lam_n_out = Output("lam_n", g["lam_n"])
    state_next_out = Output("state_next", g["state_next"])
    pos_pred_out = Output("pos_pred_n", g["pos_pred_n"])
    vel_pred_out = Output("vel_pred_n", g["vel_pred_n"])

    step_body = Modely(
        name,
        inputs=[g["state"], g["tau"], g["mass"], g["bias"], g["passive"]],
        outputs=[state_next_out, lam_n_out, pos_pred_out, vel_pred_out],
    )
    step_body.build()
    return step_body


def build_training_model(
    dt: float,
    damping: np.ndarray,
    stats: dict,
    hidden_dim: int = 64,
    mlp_layers: int = 2,
    velocity_loss_weight: float = 0.1,
    name: str = "cheetah_train",
) -> Modely:
    """A second, structurally-identical copy of the step graph (see
    _build_step_graph), plus minimize() calls for one-step supervised
    training: force loss on the normalized contact force, position loss on
    the normalized next position, velocity loss on the normalized next
    velocity (pre-scaled by sqrt(velocity_loss_weight) inside the graph
    itself, since minimize() has no per-loss weight argument - an otherwise-
    equal-weight MSE on the scaled stream reproduces the reference's
    velocity_loss_weight exactly, as MSE(a*s, b*s) = s^2 * MSE(a, b)).

    Targets are fed as already-normalized(-and-scaled) Inputs rather than
    derived inline from a raw target Input: minimize()'s target must resolve
    to a dataset column through a chain of single-pred nodes (Modely.train's
    _resolve_label_name), which a Subtract/Divide normalization chain (each
    node has 2 preds) can't satisfy. train.py's dataset prep normalizes the
    targets with this same `stats`/`velocity_loss_weight`.
    """
    g = _build_step_graph(
        dt, damping, stats, hidden_dim, mlp_layers, velocity_loss_weight
    )

    lam_target_n = Input("lam_target_n", dim=N_DOF, sample_time=dt)
    pos_target_n = Input("pos_target_n", dim=N_DOF, sample_time=dt)
    vel_target_n = Input("vel_target_n", dim=N_DOF, sample_time=dt)

    lam_n_out = Output("lam_n_train", g["lam_n"])
    lam_target_n_out = Output("lam_target_n_out", lam_target_n.last())
    pos_pred_out = Output("pos_pred_n_train", g["pos_pred_n"])
    pos_target_out = Output("pos_target_n_out", pos_target_n.last())
    vel_pred_out = Output("vel_pred_n_train", g["vel_pred_n"])
    vel_target_out = Output("vel_target_n_out", vel_target_n.last())

    training_body = Modely(
        name,
        inputs=[
            g["state"],
            g["tau"],
            g["mass"],
            g["bias"],
            g["passive"],
            lam_target_n,
            pos_target_n,
            vel_target_n,
        ],
        outputs=[
            lam_n_out,
            lam_target_n_out,
            pos_pred_out,
            pos_target_out,
            vel_pred_out,
            vel_target_out,
        ],
    )
    training_body.minimize("force_loss", lam_n_out, lam_target_n_out, loss="mse")
    training_body.minimize("position_loss", pos_pred_out, pos_target_out, loss="mse")
    training_body.minimize("velocity_loss", vel_pred_out, vel_target_out, loss="mse")
    training_body.build()
    return training_body


def build_rollout_model(step_body: Modely, rollout_steps: int) -> Modely:
    """Chains step_body's own predictions `rollout_steps` times via Roll -
    genuine closed-loop/free-running multi-step simulation, matching
    examples/dualtank/system_id.py's validated approach."""
    state = next(n for n in step_body.inputs if n.name == "state")
    tau = next(n for n in step_body.inputs if n.name == "tau")
    mass = next(n for n in step_body.inputs if n.name == "mass")
    bias = next(n for n in step_body.inputs if n.name == "bias")
    passive = next(n for n in step_body.inputs if n.name == "passive")
    state_next_out = next(o for o in step_body.outputs if o.name == "state_next")

    rolled_state = Roll(
        f=step_body,
        callback={state: state_next_out},
        steps=rollout_steps,
        name="cheetah_roll",
    )
    rolled_state_out = Output("rolled_state", rolled_state)

    rollout_model = Modely(
        "cheetah_rollout",
        inputs=[state, tau, mass, bias, passive],
        outputs=[rolled_state_out],
    )
    return rollout_model
