"""Dataset generation, training and assessment for the nnodely-ported
cheetah contact model (model.py), using the synthetic stand-in physics in
synthetic_system.py in place of the reference's MuJoCo dataset.

Pipeline:
  1. Generate independent synthetic runs, split by run into train/val (no
     leakage across a trajectory).
  2. Flatten each run into one-step (t -> t+1) transitions and compute
     normalization stats from the train split only.
  3. Train build_training_model() (one-step supervised: force + position +
     velocity loss, weights matching conf.py: 1.0/1.0/0.1) via nnodely's
     Modely.train()/validate().
  4. Copy the trained weights into a plain build_contact_step() body (see
     model.py's docstring for why this can't just be shared directly).
  5. Assess: one-step validation metrics/plots (from validate()) plus a
     genuine multi-step closed-loop Roll rollout, compared against the true
     trajectory - the rollout's mass/bias/passive are held fixed at their
     seed-step value (Roll's own limitation, see model.py), so the rollout
     plot doubles as a direct, visible measurement of how much that
     approximation costs as the horizon grows.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from nnodely import DataLoader

from examples.cheetah.model import (
    N_DOF,
    STATE_DIM,
    build_contact_step,
    build_rollout_model,
    build_training_model,
)
from examples.cheetah.synthetic_system import (
    DT,
    HEIGHT_DOF,
    FOOT_DOF,
    _DOF_DAMPING,
    generate_runs,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
VALIDATION_DIR = os.path.join(SCRIPT_DIR, "validation")

SEED = 0
N_RUNS = 40
N_STEPS_PER_RUN = 150
VAL_RUN_FRACTION = 0.2

HIDDEN_DIM = 64
MLP_LAYERS = 2
VELOCITY_LOSS_WEIGHT = 0.1

EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

ROLLOUT_STEPS = 30


def _transitions_from_runs(runs: list[dict]) -> dict[str, np.ndarray]:
    """Flattens a list of runs into one-step (t -> t+1) transitions, dropping
    each run's last step (no t+1 target for it)."""
    state, tau, mass, bias, passive, next_state, lam = [], [], [], [], [], [], []
    for run in runs:
        n = run["state"].shape[0] - 1
        state.append(run["state"][:n])
        tau.append(run["tau"][:n])
        mass.append(run["M"][:n])
        bias.append(run["bias"][:n])
        passive.append(run["passive"][:n])
        next_state.append(run["state"][1 : n + 1])
        lam.append(run["JT_lambda"][:n])
    return {
        "state": np.concatenate(state, axis=0),
        "tau": np.concatenate(tau, axis=0),
        "mass": np.concatenate(mass, axis=0),
        "bias": np.concatenate(bias, axis=0),
        "passive": np.concatenate(passive, axis=0),
        "next_state": np.concatenate(next_state, axis=0),
        "lam": np.concatenate(lam, axis=0),
    }


def _compute_stats(transitions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    eps = 1e-6
    return {
        "state_mean": transitions["state"].mean(axis=0).astype(np.float32),
        "state_std": (transitions["state"].std(axis=0) + eps).astype(np.float32),
        "tau_mean": transitions["tau"].mean(axis=0).astype(np.float32),
        "tau_std": (transitions["tau"].std(axis=0) + eps).astype(np.float32),
        "lam_mean": transitions["lam"].mean(axis=0).astype(np.float32),
        "lam_std": (transitions["lam"].std(axis=0) + eps).astype(np.float32),
    }


def _build_training_dataset(
    transitions: dict[str, np.ndarray], stats: dict, velocity_loss_weight: float
) -> dict[str, np.ndarray]:
    """(T, dim...)-shaped arrays for nnodely's DataLoader.

    Every input build_training_model() declares is consumed via `.last()`
    (past=1, future=0) - state, tau, mass, bias, passive and the three
    targets alike - so DataLoader's sliding window degenerates to "one
    independent row per transition": there's no window to blend across the
    run boundaries _transitions_from_runs() already flattened out, unlike
    the dual-tank Roll rollout's real multi-sample windows (see
    examples/dualtank/system_id.py). DataLoader also does its own
    [time, dim...] -> [dim..., time] reshape per input, so these arrays are
    handed over exactly as logged, with no manual reshaping.
    """
    lam_target_n = (transitions["lam"] - stats["lam_mean"]) / stats["lam_std"]
    next_state_n = (transitions["next_state"] - stats["state_mean"]) / stats[
        "state_std"
    ]
    pos_target_n = next_state_n[:, :N_DOF]
    vel_scale = float(np.sqrt(velocity_loss_weight))
    vel_target_n = next_state_n[:, N_DOF:] * vel_scale

    return {
        "state": transitions["state"],
        "tau": transitions["tau"],
        "mass": transitions["mass"],
        "bias": transitions["bias"],
        "passive": transitions["passive"],
        "lam_target_n": lam_target_n,
        "pos_target_n": pos_target_n,
        "vel_target_n": vel_target_n,
    }


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    runs = generate_runs(n_runs=N_RUNS, n_steps=N_STEPS_PER_RUN, seed=SEED)
    n_val_runs = max(1, int(round(N_RUNS * VAL_RUN_FRACTION)))
    train_runs, val_runs = runs[:-n_val_runs], runs[-n_val_runs:]
    print(
        f"{len(train_runs)} train runs, {len(val_runs)} val runs, "
        f"{N_STEPS_PER_RUN} steps each (dt={DT}s)"
    )

    train_transitions = _transitions_from_runs(train_runs)
    val_transitions = _transitions_from_runs(val_runs)
    stats = _compute_stats(train_transitions)

    # Built before the DataLoaders below: DataLoader inspects the model's
    # declared inputs (name, past/future) to know how to window each column.
    training_body = build_training_model(
        dt=DT,
        damping=_DOF_DAMPING,
        stats=stats,
        hidden_dim=HIDDEN_DIM,
        mlp_layers=MLP_LAYERS,
        velocity_loss_weight=VELOCITY_LOSS_WEIGHT,
    )

    train_data = DataLoader(
        model=training_body,
        source=_build_training_dataset(train_transitions, stats, VELOCITY_LOSS_WEIGHT),
    )
    val_data = DataLoader(
        model=training_body,
        source=_build_training_dataset(val_transitions, stats, VELOCITY_LOSS_WEIGHT),
    )
    print(f"{len(train_data)} train transitions, {len(val_data)} val transitions")

    print(f"\nTraining for {EPOCHS} epochs...")
    training_body.train(
        train_data=train_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        optimizer="adamw",
        lr=LEARNING_RATE,
    )

    print("\nOne-step validation:")
    val_result = training_body.validate(
        val_data=val_data, batch_size=BATCH_SIZE, out_dir=VALIDATION_DIR
    )

    # ------------------------------------------------------------------
    # Copy trained weights into the plain inference/rollout body - see
    # model.py's module docstring for why the two are separate graphs.
    # ------------------------------------------------------------------
    step_body = build_contact_step(
        dt=DT,
        damping=_DOF_DAMPING,
        stats=stats,
        hidden_dim=HIDDEN_DIM,
        mlp_layers=MLP_LAYERS,
        velocity_loss_weight=VELOCITY_LOSS_WEIGHT,
    )
    if step_body.model is None or training_body.model is None:
        raise RuntimeError("step_body.model is None, build() must be called first")
    for dst, src in zip(
        step_body.model.trainable_weights, training_body.model.trainable_weights
    ):
        assert dst.shape == src.shape, (dst.path, src.path)
        dst.assign(src.numpy())

    # ------------------------------------------------------------------
    # Multi-step closed-loop rollout assessment.
    # ------------------------------------------------------------------
    rollout_model = build_rollout_model(step_body, rollout_steps=ROLLOUT_STEPS)
    rollout_model.build()
    rollout_model.export_html(os.path.join(PLOTS_DIR, "rollout_model.html"))

    print(
        f"\nRunning {ROLLOUT_STEPS}-step closed-loop rollout on {len(val_runs)} val runs..."
    )
    run_errors_pos, run_errors_vel = [], []
    example_run = val_runs[0]
    example_pred_states = None
    for run in val_runs:
        n = run["state"].shape[0]
        if n <= ROLLOUT_STEPS:
            continue
        batch = {
            "state": run["state"][0:1].reshape(1, STATE_DIM, 1),
            "tau": run["tau"][0:1].reshape(1, N_DOF, 1),
            "mass": run["M"][0:1].reshape(1, N_DOF, N_DOF, 1),
            "bias": run["bias"][0:1].reshape(1, N_DOF, 1),
            "passive": run["passive"][0:1].reshape(1, N_DOF, 1),
        }
        out = rollout_model(batch)
        pred_final = np.asarray(out["rolled_state"])[0].squeeze(-1)
        true_final = run["state"][ROLLOUT_STEPS]
        run_errors_pos.append(pred_final[:N_DOF] - true_final[:N_DOF])
        run_errors_vel.append(pred_final[N_DOF:] - true_final[N_DOF:])
        if run is example_run:
            # Re-run step by step (small ROLLOUT_STEPS) to also get a full
            # trajectory for the example plot, not just the final state.
            pred_states = [run["state"][0]]
            for k in range(1, ROLLOUT_STEPS + 1):
                sub_out = build_rollout_model(step_body, rollout_steps=k)
                sub_out.build()
                sub_batch = {
                    "state": run["state"][0:1].reshape(1, STATE_DIM, 1),
                    "tau": run["tau"][0:1].reshape(1, N_DOF, 1),
                    "mass": run["M"][0:1].reshape(1, N_DOF, N_DOF, 1),
                    "bias": run["bias"][0:1].reshape(1, N_DOF, 1),
                    "passive": run["passive"][0:1].reshape(1, N_DOF, 1),
                }
                sub_pred = np.asarray(sub_out(sub_batch)["rolled_state"])[0].squeeze(-1)
                pred_states.append(sub_pred)
            example_pred_states = np.stack(pred_states, axis=0)

    run_errors_pos = np.stack(run_errors_pos, axis=0)
    run_errors_vel = np.stack(run_errors_vel, axis=0)
    rollout_pos_rmse = float(np.sqrt(np.mean(run_errors_pos**2)))
    rollout_vel_rmse = float(np.sqrt(np.mean(run_errors_vel**2)))
    print(f"  {ROLLOUT_STEPS}-step rollout position RMSE: {rollout_pos_rmse:.4f}")
    print(f"  {ROLLOUT_STEPS}-step rollout velocity RMSE: {rollout_vel_rmse:.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    true_traj = example_run["state"][: ROLLOUT_STEPS + 1]
    time = np.arange(ROLLOUT_STEPS + 1) * DT

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, dof, label in zip(
        axes,
        [HEIGHT_DOF, FOOT_DOF[0], FOOT_DOF[1]],
        ["height (dof 1)", "bfoot (dof 5)", "ffoot (dof 8)"],
    ):
        ax.plot(time, true_traj[:, dof], label="true", linewidth=2)
        ax.plot(
            time,
            example_pred_states[:, dof],  # type: ignore
            label="rollout (predicted)",
            linestyle="--",
        )
        ax.set_ylabel(f"q[{label}]")
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Closed-loop rollout vs. ground truth ({ROLLOUT_STEPS} steps)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rollout_trajectory.png"), dpi=160)
    plt.close(fig)

    step_pos_err = np.sqrt(
        np.mean((example_pred_states[:, :N_DOF] - true_traj[:, :N_DOF]) ** 2, axis=1)  # type: ignore
    )
    step_vel_err = np.sqrt(
        np.mean((example_pred_states[:, N_DOF:] - true_traj[:, N_DOF:]) ** 2, axis=1)  # type: ignore
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, step_pos_err, label="position RMSE")
    ax.plot(time, step_vel_err, label="velocity RMSE")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMSE")
    ax.set_title("Rollout error growth (mass/bias/passive held at seed value)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rollout_error_growth.png"), dpi=160)
    plt.close(fig)

    print("\nSummary")
    print("=======")
    for name, m in val_result["metrics"].items():
        print(f"  one-step {name:16s} MSE={m['mse']:.4e}  R2={m['r2']:.4f}")
    print(f"  {ROLLOUT_STEPS}-step rollout position RMSE: {rollout_pos_rmse:.4f}")
    print(f"  {ROLLOUT_STEPS}-step rollout velocity RMSE: {rollout_vel_rmse:.4f}")
    print(f"\nPlots saved to {PLOTS_DIR}")
    print(f"Per-loss validation plots saved to {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
