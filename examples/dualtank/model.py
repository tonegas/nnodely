"""Dual tank level control via Differentiable Predictive Control (DPC).

System: nonlinear two-tank (Neuromancer's psl.TwoTank), matching the DPC
formulation pasted in the task -- u_k = [pump, valve], both controlled:

    dh1/dt = c1*(1 - valve)*pump - c2*sqrt(h1)
    dh2/dt = c1*valve*pump + c2*sqrt(h1) - c2*sqrt(h2)

    h1, h2, pump, valve in [0, 1]   (states/inputs are all normalized)
    c1 = 0.08, c2 = 0.04, sample time = 1.0   (Neuromancer psl.TwoTank defaults)

This first pass builds the DPC *architecture* only:
  - the differentiable system model (the ODESolve(f(x,u)) box)
  - the neural control policy pi_theta(x_k, R)
  - their closed-loop composition into one Euler step
  - a multi-step rollout of that closed loop via Modely.rollback()

Not yet included (deliberately, to discuss next): the full DPC loss summed
over every step of the prediction horizon. rollback() only exposes the
*final* rolled step's outputs (see ModelRollImpl in layers/roll.py), so a
running sum of per-step tracking error needs either a seq-axis reduction
primitive nnodely doesn't have yet, or driving the horizon with Scan's
collect_outputs mode instead. What's below trains against the terminal
tracking cost only, which already exercises the whole architecture end to
end.
"""

import os

import numpy as np

import time

from nnodely import (
    Input,
    Output,
    Modely,
    Linear,
    ReLU,
    Sigmoid,
    Integrate,
)
from nnodely.layers.time_ops import Select, Concatenate
from nnodely.core.dataloader import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

N_SAMPLES = 5000
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
SEED = 42

rng = np.random.default_rng(SEED)


def sample_scenarios(n_samples: int, rng: np.random.Generator) -> dict:
    """n independent scenarios: a random initial state (h1_0, h2_0) and a
    random reference (r1, r2), both uniform in [0, 1) - matching
    Neuromancer's `torch.rand(n_samples, 1, nx)` / `torch.rand(1, 1)`.
    """
    return {
        "h1": rng.uniform(0.0, 1.0, size=n_samples).astype(np.float32),
        "h2": rng.uniform(0.0, 1.0, size=n_samples).astype(np.float32),
        "r1": rng.uniform(0.0, 1.0, size=n_samples).astype(np.float32),
        "r2": rng.uniform(0.0, 1.0, size=n_samples).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C1 = 0.08
C2 = 0.04
DT = 1.0
HORIZON = 10  # prediction horizon N for the closed-loop rollout (Neuromancer's nsteps)

# ---------------------------------------------------------------------------
# Inputs: current states and the (held-constant-over-horizon) references
# ---------------------------------------------------------------------------
h1 = Input("h1", dim=1, sample_time=DT)
h2 = Input("h2", dim=1, sample_time=DT)
r1 = Input("r1", dim=1, sample_time=DT)
r2 = Input("r2", dim=1, sample_time=DT)

# ---------------------------------------------------------------------------
# Neural control policy: u_k = pi_theta(x_k, R) = pi(h1, h2, r1, r2)
# Sigmoid bounds pump/valve to [0, 1], matching the DPC box constraint on u.
# ---------------------------------------------------------------------------
state_and_ref = Concatenate(axis=0)([h1.last(), h2.last(), r1.last(), r2.last()])
policy_hidden = ReLU()(Linear(out_features=16)(state_and_ref))
policy_out = Sigmoid()(Linear(out_features=2)(policy_hidden))

pump = Select(idx=0, axis=0)(policy_out)
valve = Select(idx=1, axis=0)(policy_out)

# ---------------------------------------------------------------------------
# Differentiable system model: ODESolve(f(x_k, u_k)), one explicit Euler step
# ---------------------------------------------------------------------------
sqrt_h1 = h1.last() ** 0.5
sqrt_h2 = h2.last() ** 0.5

rate_h1 = C1 * (1.0 - valve) * pump - C2 * sqrt_h1
rate_h2 = C1 * valve * pump + C2 * sqrt_h1 - C2 * sqrt_h2

h1_increment = Integrate(rate_h1, solver="euler", dt=DT, name="h1_increment")
h2_increment = Integrate(rate_h2, solver="euler", dt=DT, name="h2_increment")
h1_next = h1.last() + h1_increment
h2_next = h2.last() + h2_increment

# ---------------------------------------------------------------------------
# Close the loop: one Modely holds policy + plant for a single step; rollback
# unrolls it for HORIZON steps, feeding each step's next state back in as the
# following step's current state (state.last() above) - same pattern as
# tests/test_applications.py::test_recurrent_vehicle_longitudinal_dynamics.
# ---------------------------------------------------------------------------
h1_next_out = Output("h1_next", h1_next)
h2_next_out = Output("h2_next", h2_next)

dual_tank = Modely(
    "dual_tank_dpc",
    inputs=[h1, h2, r1, r2],
    outputs=[
        h1_next_out,
        h2_next_out,
        Output("pump", pump),
        Output("valve", valve),
    ],
)
dual_tank.rollback(
    {"h1": "h1_next", "h2": "h2_next"}, steps=HORIZON, name="dpc_rollout"
)

# Terminal tracking cost (placeholder for the full DPC objective - see the
# module docstring): penalize the state reached after HORIZON steps against
# the reference.
dual_tank.minimize("h1_tracking", h1_next_out, r1.last(), loss="mse")
dual_tank.minimize("h2_tracking", h2_next_out, r2.last(), loss="mse")

dual_tank.build()

# ---------------------------------------------------------------------------
# Visualize the architecture
# ---------------------------------------------------------------------------
dual_tank.export_html(os.path.join("html", "dual_tank_dpc.html"))
dual_tank.plot(
    os.path.join("html", "dual_tank_dpc.png"), include_minimizers=True, flatten=True
)

# ---------------------------------------------------------------------------
# Inference before training
# ---------------------------------------------------------------------------
dummy = {
    "h1": np.array([0.5], dtype=np.float32),
    "h2": np.array([0.5], dtype=np.float32),
    "r1": np.array([0.8], dtype=np.float32),
    "r2": np.array([0.6], dtype=np.float32),
}
result = dual_tank(dummy)
print("Closed-loop rollout result after", HORIZON, "steps: (before training)")
for name, value in result.items():
    print(f"  {name}: {np.asarray(value).squeeze()}")

# ---------------------------------------------------------------------------
# Load Datasets and train the model
# ---------------------------------------------------------------------------
train_data = DataLoader(model=dual_tank, source=sample_scenarios(N_SAMPLES, rng))
dev_data = DataLoader(model=dual_tank, source=sample_scenarios(N_SAMPLES // 50, rng))


start = time.time()
history = dual_tank.train(
    train_data=train_data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
)
end = time.time()
print(f"Training completed in {end - start:.2f} seconds.")
print("Final training loss:", history["loss"][-1])

metrics = dual_tank.validate(
    val_data=dev_data,
    batch_size=BATCH_SIZE,
    out_dir=os.path.join(SCRIPT_DIR, "validation"),
)
print("Validation metrics:", metrics["metrics"])

# ---------------------------------------------------------------------------
# Inference after training
# ---------------------------------------------------------------------------
result = dual_tank(dummy)
print("Closed-loop rollout result after", HORIZON, "steps: (after training)")
for name, value in result.items():
    print(f"  {name}: {np.asarray(value).squeeze()}")
