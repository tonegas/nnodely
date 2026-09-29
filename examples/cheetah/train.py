"""Dataset generation, training and assessment of the cheetah model
(model.py), on the synthetic stand-in physics of synthetic_system.py.

Pipeline:
  1. Generate independent synthetic runs, split by run into train/val.
  2. Train the simulator (build_simulator) on HORIZON-step closed-loop
     rollouts: every loss compares a whole trajectory, so the contact model
     is trained through the dynamics it drives.
  3. Run inference with the trained simulator on EVAL_HORIZON-step windows
     of the val runs, chaining EVAL_HORIZON // HORIZON calls of it, and
     compare the predicted q and q_dot against the ground truth and against
     the physics with no contact force, checking that the graph integrates
     exactly the equation it states.

To move to the real simulator, replace generate_runs() with runs logged from
it, one row per physics step (keys as in synthetic_system.simulate_run).
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import matplotlib.pyplot as plt
import numpy as np

from nnodely import DataLoader, set_seed

from examples.cheetah.model import (
    build_simulator,
    build_step,
    stats_from_runs,
)
from examples.cheetah.synthetic_system import DT, FOOT_DOF, HEIGHT_DOF, generate_runs

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
VALIDATION_DIR = os.path.join(SCRIPT_DIR, "validation")

SEED = 0
N_RUNS = 100
N_STEPS_PER_RUN = 200
VAL_RUN_FRACTION = 0.2

WINDOW = 5  # samples of (q, qd) the recurrent encoder reads
HIDDEN = 32
MLP_HIDDEN = 64
MLP_LAYERS = 2

HORIZON = 20  # closed-loop steps per training sample
EVAL_HORIZON = 100

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

TRUTH_COLOR = "#52514e"
MODEL_COLOR = "#2a78d6"
BASELINE_COLOR = "#eb6834"


def sequence_source(runs: list[dict]) -> list[dict[str, np.ndarray]]:
    """One DataLoader simulation per run, so no window crosses two runs.

    Row t holds the state at t and what the step t -> t+1 uses (tau, M, h,
    J^T lambda), with the state at t+1 as target. A sample of HORIZON steps
    then seeds the rollout with the q/qd window ending at its first step.
    """
    return [
        {
            "q_seq": run["q"][:-1],
            "qd_seq": run["qd"][:-1],
            "tau_seq": run["tau"][:-1],
            "M_seq": run["M"][:-1],
            "h_seq": run["h"][:-1],
            "JT_lambda_seq": run["JT_lambda"][:-1],
            "q_next_seq": run["q"][1:],
            "qd_next_seq": run["qd"][1:],
        }
        for run in runs
    ]


def long_windows(runs: list[dict], horizon: int, stride: int) -> dict[str, np.ndarray]:
    """`horizon`-step windows of the runs, one every `stride` steps, in
    nnodely's layout (batch, dim..., time, seq): q0/qd0 hold the WINDOW samples
    ending at the first step, the *_seq arrays the columns sequence_source
    gives the DataLoader, for every step of the window."""

    def steps(values, start):
        return np.moveaxis(values[start : start + horizon], 0, -1)[..., None, :]

    samples = []
    for run in runs:
        for t0 in range(WINDOW - 1, len(run["q"]) - horizon, stride):
            seed = slice(t0 - WINDOW + 1, t0 + 1)
            samples.append(
                {
                    "q0": run["q"][seed].T,
                    "qd0": run["qd"][seed].T,
                    "tau_seq": steps(run["tau"], t0),
                    "M_seq": steps(run["M"], t0),
                    "h_seq": steps(run["h"], t0),
                    "JT_lambda_seq": steps(run["JT_lambda"], t0),
                    "q_next_seq": steps(run["q"], t0 + 1),
                    "qd_next_seq": steps(run["qd"], t0 + 1),
                }
            )
    return {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}


def chained_rollout(sim, windows: dict, horizon: int) -> dict[str, np.ndarray]:
    """Predict `horizon` steps with `sim`, which rolls out HORIZON steps, by
    calling it horizon // HORIZON times. The q/qd window is the whole state a
    step carries (the encoder starts from zero at every step), so seeding a
    call with the last WINDOW predictions continues the previous call exactly.
    """
    assert horizon % HORIZON == 0, (horizon, HORIZON)
    q_win, qd_win = windows["q0"], windows["qd0"]
    chunks = []
    for start in range(0, horizon, HORIZON):
        steps = slice(start, start + HORIZON)
        feed = {
            # The Loop seeds the rollout from the first element of the
            # sequence axis only.
            "q_seq": np.repeat(q_win[..., None], HORIZON, axis=-1),
            "qd_seq": np.repeat(qd_win[..., None], HORIZON, axis=-1),
            "tau_seq": windows["tau_seq"][..., steps],
            "M_seq": windows["M_seq"][..., steps],
            "h_seq": windows["h_seq"][..., steps],
        }
        out = {
            name: np.asarray(keras.ops.convert_to_numpy(value))
            for name, value in sim(feed).items()
        }
        chunks.append(out)
        q_win = np.concatenate([q_win, out["q_traj"][:, :, 0]], axis=-1)[..., -WINDOW:]
        qd_win = np.concatenate([qd_win, out["qd_traj"][:, :, 0]], axis=-1)[
            ..., -WINDOW:
        ]
    return {
        name: np.concatenate([chunk[name] for chunk in chunks], axis=-1)
        for name in chunks[0]
    }


def euler_rollout(data: dict, jt_lambda: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q_ddot = M^-1 (tau + J^T lambda - h), integrated in numpy from the seed
    state of every window of `data` (long_windows) with the given J^T lambda
    trajectory. Arrays follow nnodely's layout: (batch, dim..., time, seq)."""
    q = data["q0"][:, :, -1].astype(np.float64)
    qd = data["qd0"][:, :, -1].astype(np.float64)
    q_traj, qd_traj = [], []
    for k in range(data["tau_seq"].shape[-1]):
        rhs = (
            data["tau_seq"][:, :, 0, k]
            + jt_lambda[:, :, 0, k]
            - data["h_seq"][:, :, 0, k]
        )
        qdd = np.linalg.solve(data["M_seq"][:, :, :, 0, k], rhs[..., None])[..., 0]
        qd = qd + DT * qdd
        q = q + DT * qd
        q_traj.append(q)
        qd_traj.append(qd)
    return np.stack(q_traj, axis=-1)[:, :, None], np.stack(qd_traj, axis=-1)[:, :, None]


def rmse_per_step(pred: np.ndarray, true: np.ndarray, std: np.ndarray) -> np.ndarray:
    """RMSE over samples and dof at every rollout step, in units of each
    dof's std."""
    err = (pred - true) / std[None, :, None, None]
    return np.sqrt(np.mean(err**2, axis=(0, 1, 2)))


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    set_seed(SEED)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    runs = generate_runs(n_runs=N_RUNS, n_steps=N_STEPS_PER_RUN, seed=SEED)
    n_val_runs = max(1, int(round(N_RUNS * VAL_RUN_FRACTION)))
    train_runs, val_runs = runs[:-n_val_runs], runs[-n_val_runs:]
    print(
        f"{len(train_runs)} train runs, {len(val_runs)} val runs, "
        f"{N_STEPS_PER_RUN} steps each (dt={DT}s)"
    )

    stats = stats_from_runs(train_runs)
    step = build_step(
        DT,
        stats,
        window=WINDOW,
        hidden=HIDDEN,
        mlp_hidden=MLP_HIDDEN,
        mlp_layers=MLP_LAYERS,
    )
    print("\nStep model Summary")
    step.summary()

    sim = build_simulator(step, HORIZON, stats)
    sim.export_html(os.path.join(PLOTS_DIR, "model_html"))

    train_data = DataLoader(model=sim, source=sequence_source(train_runs))
    val_data = DataLoader(model=sim, source=sequence_source(val_runs))
    print(f"{len(train_data)} train / {len(val_data)} val rollouts of {HORIZON} steps")
    print(train_data)
    print(val_data)

    print(f"\nTraining for {EPOCHS} epochs...")
    history = sim.train(
        train_data=train_data,
        val_data=val_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        optimizer="adamw",
        lr=LEARNING_RATE,
        optimizer_kwargs={"global_clipnorm": 1.0},
        printer="nnodely",
    )

    print(f"\n{HORIZON}-step validation:")
    short_result = sim.validate(
        val_data=val_data,
        out_dir=os.path.join(VALIDATION_DIR, f"horizon_{HORIZON}"),
        history=history,
    )

    # ------------------------------------------------------------------
    # Inference: the trained sim predicts q and q_dot over EVAL_HORIZON-step
    # windows of the val runs, from the seed q/qd window and the tau/M/h of
    # every step, in EVAL_HORIZON // HORIZON chained calls.
    # ------------------------------------------------------------------
    data = long_windows(val_runs, EVAL_HORIZON, stride=EVAL_HORIZON // 2)
    n_windows = len(data["q0"])
    pred = chained_rollout(sim, data, EVAL_HORIZON)
    print(
        f"\nInference on {n_windows} val windows, {EVAL_HORIZON // HORIZON} "
        f"calls of sim each: q_traj {pred['q_traj'].shape}, "
        f"qd_traj {pred['qd_traj'].shape}"
    )

    ## example data inferred
    sample = val_data.get_samples(1)
    print(f"Sample data\n: {sample}")
    prediction = sim(sample)
    print(
        f"\nExample inference on one sample: q_traj {prediction['q_traj'].shape}, "
        f"qd_traj {prediction['qd_traj'].shape}"
    )
    print("values q_traj:")
    print(prediction["q_traj"])
    print("values qd_traj:")
    print(prediction["qd_traj"])

    # The graph must integrate exactly the equation it states: numpy Euler
    # with the model's own J^T lambda reproduces its rollout.
    q_check, qd_check = euler_rollout(data, pred["JT_lambda_traj"])
    wiring_error = max(
        np.max(np.abs(q_check - pred["q_traj"])),
        np.max(np.abs(qd_check - pred["qd_traj"])),
    )
    print(f"\nPhysics wiring check, max |numpy - model|: {wiring_error:.2e}")
    assert wiring_error < 1e-3, wiring_error

    q_free, qd_free = euler_rollout(data, np.zeros_like(pred["JT_lambda_traj"]))
    rmse = {
        "q": rmse_per_step(pred["q_traj"], data["q_next_seq"], stats["q_std"]),
        "qd": rmse_per_step(pred["qd_traj"], data["qd_next_seq"], stats["qd_std"]),
        "q_free": rmse_per_step(q_free, data["q_next_seq"], stats["q_std"]),
        "qd_free": rmse_per_step(qd_free, data["qd_next_seq"], stats["qd_std"]),
    }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    time = np.arange(1, EVAL_HORIZON + 1) * DT
    sample = 0
    panels = [
        (data["q_next_seq"], pred["q_traj"], q_free, HEIGHT_DOF, "height q[1]"),
        (data["q_next_seq"], pred["q_traj"], q_free, FOOT_DOF[0], "bfoot q[5]"),
        (
            data["JT_lambda_seq"],
            pred["JT_lambda_traj"],
            None,
            HEIGHT_DOF,
            r"$J^T\lambda$ [1]",
        ),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, (true, model, free, dof, label) in zip(axes, panels):
        ax.plot(
            time,
            true[sample, dof, 0],
            color=TRUTH_COLOR,
            linewidth=2,
            label="ground truth",
        )
        ax.plot(
            time,
            model[sample, dof, 0],
            color=MODEL_COLOR,
            linewidth=2,
            linestyle="--",
            label="model",
        )
        if free is not None:
            ax.plot(
                time,
                free[sample, dof, 0],
                color=BASELINE_COLOR,
                linewidth=2,
                linestyle=":",
                label="no contact force",
            )
        style(ax, label)
    axes[0].legend(frameon=False, loc="lower left")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Closed-loop rollout on a val run ({EVAL_HORIZON} steps)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rollout_trajectory.png"), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, key, label in zip(
        axes, ("q", "qd"), ("position RMSE (std)", "velocity RMSE (std)")
    ):
        ax.plot(time, rmse[key], color=MODEL_COLOR, linewidth=2, label="model")
        ax.plot(
            time,
            rmse[f"{key}_free"],
            color=BASELINE_COLOR,
            linewidth=2,
            linestyle=":",
            label="no contact force",
        )
        style(ax, label)
        ax.set_yscale("log")
    axes[0].legend(frameon=False, loc="lower right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Rollout error growth, {n_windows} val rollouts")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rollout_error_growth.png"), dpi=160)
    plt.close(fig)

    print("\nSummary")
    print("=======")
    for name, m in short_result.metrics().items():
        print(
            f"  {HORIZON:3d}-step {name:14s} R2={m['r2']:.4f}  fit={m['fit_pct']:.1f}%"
        )
    for k in (1, HORIZON, EVAL_HORIZON):
        print(
            f"  after {k:3d} steps: position RMSE {rmse['q'][k - 1]:.4f} std "
            f"(no contact {rmse['q_free'][k - 1]:.4f}), velocity RMSE "
            f"{rmse['qd'][k - 1]:.4f} std (no contact {rmse['qd_free'][k - 1]:.4f})"
        )
    print(f"\nPlots saved to {PLOTS_DIR}")
    print(f"Per-loss validation plots saved to {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
