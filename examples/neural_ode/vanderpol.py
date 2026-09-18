# """Learn the Van der Pol oscillator with nnodely's NeuralODE.

# This example follows Neuromancer's ``Part_1_NODE`` tutorial:

# 1. generate train, validation, and test trajectories from the known system;
# 2. replace the unknown right-hand side with a three-layer ReLU network;
# 3. integrate that network with differentiable RK4 over short training horizons;
# 4. train with trajectory and finite-difference losses; and
# 5. compare a long free-running NeuralODE trajectory with unseen reference data.

# Run from the repository root::

#     uv run python examples/neural_ode/vanderpol.py

# For a faster smoke run::

#     uv run python examples/neural_ode/vanderpol.py \
#         --epochs 30 --nsim 200 --test-steps 100 --hidden-size 32

# The plots and interactive nnodely graph are written next to this file under
# ``outputs/vanderpol`` unless ``--output-dir`` is provided.
# """

# from __future__ import annotations

# import argparse
# from pathlib import Path

# import keras
# import matplotlib.pyplot as plt
# import numpy as np

# from nnodely import DataLoader, Input, Linear, Modely, NeuralODE, Output, ReLU


# SEED = 0
# MU = 1.0
# DT = 0.1
# TRAIN_HORIZON = 2
# NSIM = 600
# TEST_STEPS = 600
# HIDDEN_SIZE = 60
# HIDDEN_LAYERS = 3
# EPOCHS = 500
# BATCH_SIZE = 100
# LEARNING_RATE = 1e-3

# SCRIPT_DIR = Path(__file__).resolve().parent
# DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "vanderpol"


# def vanderpol_rate(state: np.ndarray, mu: float = MU) -> np.ndarray:
#     """Van der Pol dynamics used by the Neuromancer tutorial."""
#     x1, x2 = state
#     return np.array(
#         [
#             mu * (x1 - x1**3 / 3.0 - x2),
#             x1 / mu,
#         ],
#         dtype=np.float64,
#     )


# def simulate_vanderpol(
#     initial: tuple[float, float] | np.ndarray,
#     steps: int,
#     dt: float = DT,
#     mu: float = MU,
# ) -> np.ndarray:
#     """Generate a high-quality reference trajectory with classical RK4."""
#     trajectory = np.empty((steps + 1, 2), dtype=np.float64)
#     trajectory[0] = np.asarray(initial, dtype=np.float64)

#     for index in range(steps):
#         state = trajectory[index]
#         k1 = vanderpol_rate(state, mu)
#         k2 = vanderpol_rate(state + 0.5 * dt * k1, mu)
#         k3 = vanderpol_rate(state + 0.5 * dt * k2, mu)
#         k4 = vanderpol_rate(state + dt * k3, mu)
#         trajectory[index + 1] = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

#     return trajectory.astype(np.float32)


# def trajectory_and_difference_loss(y_true, y_pred):
#     """Neuromancer-style state tracking plus finite-difference matching."""
#     reference_error = keras.ops.mean(keras.ops.square(y_true - y_pred))
#     true_difference = y_true[..., 1:] - y_true[..., :-1]
#     predicted_difference = y_pred[..., 1:] - y_pred[..., :-1]
#     difference_error = keras.ops.mean(
#         keras.ops.square(true_difference - predicted_difference)
#     )
#     return reference_error + 2.0 * difference_error


# def build_rhs(
#     dt: float = DT,
#     hidden_size: int = HIDDEN_SIZE,
#     hidden_layers: int = HIDDEN_LAYERS,
# ):
#     """Build the neural approximation of dx/dt and expose its state/output."""
#     state = Input("rhs_state", dim=2)
#     hidden = state.last()
#     for index in range(hidden_layers):
#         hidden = Linear(
#             out_features=hidden_size,
#             name=f"rhs_linear_{index + 1}",
#         )(hidden)
#         hidden = ReLU(name=f"rhs_relu_{index + 1}")(hidden)
#     derivative = Linear(out_features=2, name="rhs_derivative_linear")(hidden)
#     derivative_output = Output("state_derivative", derivative)
#     rhs = Modely(
#         "vanderpol_neural_rhs",
#         inputs=[state],
#         outputs=[derivative_output],
#     ).build()
#     return rhs, state, derivative_output


# def build_training_model(
#     rhs: Modely,
#     rhs_state: Input,
#     derivative_output: Output,
#     *,
#     dt: float = DT,
#     horizon: int = TRAIN_HORIZON,
# ) -> tuple[Modely, Input, Input]:
#     """Create a short-horizon NODE model for supervised trajectory fitting."""
#     initial = Input("initial_state", dim=2)
#     target = Input("target_state", dim=2, seq=horizon)
#     prediction = NeuralODE(
#         f=rhs,
#         derivatives={rhs_state: derivative_output},
#         initial=initial,
#         steps=horizon,
#         dt=dt,
#         method="rk4",
#         name="vanderpol_node",
#     )
#     prediction_output = Output("predicted_state", prediction)
#     model = Modely(
#         "vanderpol_system_identification",
#         inputs=[initial],
#         outputs=[prediction_output],
#     )
#     model.minimize(
#         "trajectory_error",
#         source=prediction_output,
#         target=target,
#         loss=trajectory_and_difference_loss,
#     )
#     model.build()
#     return model, initial, target


# def build_trajectory_data(
#     model: Modely,
#     trajectory: np.ndarray,
#     *,
#     initial_name: str,
#     target_name: str,
#     horizon: int = TRAIN_HORIZON,
# ) -> DataLoader:
#     """Create aligned ``x(k) -> [x(k+1), ..., x(k+horizon)]`` samples.

#     ``DataLoader`` aligns an input without ``seq`` to the final element of the
#     largest sequence window. Shifting the raw initial-state column by
#     ``horizon - 1`` therefore gives exactly the desired current-state/future-
#     trajectory pairs. Taking every ``horizon``-th window matches Neuromancer's
#     non-overlapping trajectory batches.
#     """
#     if len(trajectory) <= horizon:
#         raise ValueError("trajectory must contain more samples than horizon.")

#     targets = trajectory[1:]
#     initials = np.empty_like(targets)
#     initials[: horizon - 1] = trajectory[0]
#     initials[horizon - 1 :] = trajectory[: len(targets) - horizon + 1]

#     data = DataLoader(
#         model,
#         source={
#             initial_name: initials,
#             target_name: targets,
#         },
#     )
#     data.dataset = {name: values[::horizon] for name, values in data.dataset.items()}
#     data._num_steps = data._infer_num_steps()
#     return data


# def predict_trajectory(
#     rhs: Modely,
#     rhs_state: Input,
#     derivative_output: Output,
#     initial_state: np.ndarray,
#     *,
#     steps: int,
#     dt: float = DT,
# ) -> np.ndarray:
#     """Build a long-horizon inference graph sharing the trained RHS weights."""
#     initial = Input("test_initial_state", dim=2)
#     prediction = NeuralODE(
#         f=rhs,
#         derivatives={rhs_state: derivative_output},
#         initial=initial,
#         steps=steps,
#         dt=dt,
#         method="rk4",
#         name="vanderpol_test_node",
#     )
#     model = Modely(
#         "vanderpol_test_rollout",
#         inputs=[initial],
#         outputs=[Output("predicted_trajectory", prediction)],
#     ).build()
#     result = model({initial.name: initial_state})["predicted_trajectory"]
#     future = np.asarray(keras.ops.convert_to_numpy(result))[0, :, 0, :].T
#     return np.concatenate([initial_state.reshape(1, 2), future], axis=0)


# def plot_reference(trajectory: np.ndarray, dt: float, output_dir: Path) -> None:
#     """Plot the reference time series and phase portrait (pltOL/pltPhase)."""
#     time = np.arange(len(trajectory)) * dt

#     fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
#     for index, axis in enumerate(axes):
#         axis.plot(time, trajectory[:, index], color="tab:cyan", linewidth=2.0)
#         axis.set_ylabel(rf"$x_{index + 1}$")
#         axis.grid(alpha=0.25)
#     axes[-1].set_xlabel("time")
#     fig.suptitle("Van der Pol reference trajectory")
#     fig.tight_layout()
#     fig.savefig(output_dir / "reference_timeseries.png", dpi=160)
#     plt.close(fig)

#     fig, axis = plt.subplots(figsize=(7, 7))
#     axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:cyan", linewidth=2.0)
#     axis.scatter(*trajectory[0], color="black", s=35, label="initial state")
#     axis.set_xlabel(r"$x_1$")
#     axis.set_ylabel(r"$x_2$")
#     axis.set_title("Van der Pol phase portrait")
#     axis.grid(alpha=0.25)
#     axis.legend()
#     fig.tight_layout()
#     fig.savefig(output_dir / "reference_phase.png", dpi=160)
#     plt.close(fig)


# def plot_prediction(
#     reference: np.ndarray,
#     prediction: np.ndarray,
#     dt: float,
#     output_dir: Path,
# ) -> None:
#     """Plot true/predicted states in the style of the Neuromancer tutorial."""
#     time = np.arange(len(reference)) * dt
#     fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
#     for index, axis in enumerate(axes):
#         axis.plot(
#             time,
#             reference[:, index],
#             color="c",
#             linewidth=2.5,
#             label="True",
#         )
#         axis.plot(
#             time,
#             prediction[:, index],
#             color="m",
#             linestyle="--",
#             linewidth=2.5,
#             label="Pred",
#         )
#         axis.set_ylabel(rf"$x_{index + 1}$", rotation=0, labelpad=20)
#         axis.grid(alpha=0.25)
#         axis.legend()
#     axes[-1].set_xlabel("time")
#     fig.suptitle("NeuralODE long-horizon prediction")
#     fig.tight_layout()
#     fig.savefig(output_dir / "trajectory_prediction.png", dpi=160)
#     plt.close(fig)

#     fig, axis = plt.subplots(figsize=(7, 7))
#     axis.plot(
#         reference[:, 0],
#         reference[:, 1],
#         color="c",
#         linewidth=2.5,
#         label="True",
#     )
#     axis.plot(
#         prediction[:, 0],
#         prediction[:, 1],
#         color="m",
#         linestyle="--",
#         linewidth=2.5,
#         label="Pred",
#     )
#     axis.set_xlabel(r"$x_1$")
#     axis.set_ylabel(r"$x_2$")
#     axis.set_title("Learned Van der Pol phase portrait")
#     axis.grid(alpha=0.25)
#     axis.legend()
#     fig.tight_layout()
#     fig.savefig(output_dir / "phase_prediction.png", dpi=160)
#     plt.close(fig)


# def plot_loss(loss: list[float], output_dir: Path) -> None:
#     fig, axis = plt.subplots(figsize=(8, 5))
#     axis.semilogy(np.arange(1, len(loss) + 1), loss, color="tab:blue")
#     axis.set_xlabel("epoch")
#     axis.set_ylabel("training loss")
#     axis.set_title("Van der Pol NeuralODE training")
#     axis.grid(alpha=0.25)
#     fig.tight_layout()
#     fig.savefig(output_dir / "training_loss.png", dpi=160)
#     plt.close(fig)


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--epochs", type=int, default=EPOCHS)
#     parser.add_argument("--nsim", type=int, default=NSIM)
#     parser.add_argument("--test-steps", type=int, default=TEST_STEPS)
#     parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
#     parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
#     parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
#     parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     keras.utils.set_random_seed(SEED)
#     args.output_dir.mkdir(parents=True, exist_ok=True)

#     # Separate trajectories test generalization from distinct initial states.
#     train_reference = simulate_vanderpol((1.5, -0.5), args.nsim, DT)
#     dev_reference = simulate_vanderpol((-1.0, 1.0), args.nsim, DT)
#     test_reference = simulate_vanderpol((2.0, 0.0), args.test_steps, DT)

#     plot_reference(train_reference, DT, args.output_dir)

#     rhs, rhs_state, derivative_output = build_rhs(
#         dt=DT,
#         hidden_size=args.hidden_size,
#     )
#     model, initial, target = build_training_model(
#         rhs,
#         rhs_state,
#         derivative_output,
#         dt=DT,
#     )
#     model.export_html(
#         args.output_dir,
#         filename="vanderpol_neural_ode",
#         physics=False,
#     )

#     train_data = build_trajectory_data(
#         model,
#         train_reference,
#         initial_name=initial.name,
#         target_name=target.name,
#     )
#     dev_data = build_trajectory_data(
#         model,
#         dev_reference,
#         initial_name=initial.name,
#         target_name=target.name,
#     )
#     print(f"Training on {len(train_data)} trajectories of {TRAIN_HORIZON} RK4 steps...")
#     history = model.train(
#         train_data=train_data,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         optimizer="adam",
#         lr=args.learning_rate,
#     )
#     plot_loss(history["loss"], args.output_dir)

#     train_prediction = model(train_data.as_dict())["predicted_state"]
#     dev_prediction = model(dev_data.as_dict())["predicted_state"]
#     train_mse = float(
#         np.mean(
#             (
#                 np.asarray(keras.ops.convert_to_numpy(train_prediction))
#                 - train_data.dataset[target.name]
#             )
#             ** 2
#         )
#     )
#     dev_mse = float(
#         np.mean(
#             (
#                 np.asarray(keras.ops.convert_to_numpy(dev_prediction))
#                 - dev_data.dataset[target.name]
#             )
#             ** 2
#         )
#     )

#     test_prediction = predict_trajectory(
#         rhs,
#         rhs_state,
#         derivative_output,
#         test_reference[0],
#         steps=args.test_steps,
#         dt=DT,
#     )
#     rollout_rmse = float(np.sqrt(np.mean((test_prediction - test_reference) ** 2)))
#     plot_prediction(test_reference, test_prediction, DT, args.output_dir)

#     print(f"Final training loss: {history['loss'][-1]:.3e}")
#     print(f"Short-horizon train MSE: {train_mse:.3e}")
#     print(f"Short-horizon validation MSE: {dev_mse:.3e}")
#     print(f"Long-horizon test RMSE: {rollout_rmse:.3e}")
#     print(f"Plots and graph written to: {args.output_dir}")


# if __name__ == "__main__":
#     main()
