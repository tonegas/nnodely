"""System identification for the dual-tank ODE: learn c1, c2 with nnodely,
via a genuinely RECURRENT Roll-based rollout.

Builds on the earlier TimeSelect-based static multi-step unroll (see git
history), replacing it with Roll: a "step" Modely computes one Euler step
from a windowed (h1, h2) state, and Roll unrolls it WINDOW times via its
own sliding-window feedback mechanism. The key difference from the static
TimeSelect version: that one always recomputed the rate from the *ground
truth* h1[t]/h2[t] at every step (teacher forcing, i.e. one-step-ahead
prediction error). Roll instead chains its OWN predictions step to step -
a true closed-loop/free-running multi-step simulation - which control
theory generally credits with better parameter-identifiability properties
than one-step-ahead prediction error, since it's sensitive to how errors
*compound* over the horizon rather than being reset to ground truth every
step.

Two structural notes worth keeping in mind:

1. Roll supports exactly one recurrent callback pair (unlike rollback(),
   which supports several). The coupled (h1, h2) ODE state is therefore
   represented as a single dim=2 Input here, not two separate Inputs -
   TimeSelect + Select unpack/repack it inside the step model.
2. Roll's non-callback inputs (pump, valve here) are held fixed across
   every rolled step (same limitation already noted for rollback() in
   model.py) - so the ground truth is generated with one constant
   (pump, valve) pair per scenario, matching what the rolled model assumes;
   a mismatch here (e.g. genuinely time-varying pump/valve during the
   rollout) would show up as irreducible model-mismatch error unrelated to
   c1/c2's correctness.

Data loading uses nnodely's own DataLoader, but not by handing it one big
concatenated array: each training scenario needs a WINDOW-length seed *and*
a WINDOW-length target, independent of every other scenario, and
DataLoader's dict source builds sliding windows over ONE continuous array -
which would blend adjacent, unrelated scenarios into the same window. So
instead each scenario is sized to exactly WINDOW rows per input (see
`simulate_scenario_dict`), which makes DataLoader's own windowing produce
precisely the one window that scenario is meant to provide, and
`build_dataloader` combines the per-scenario DataLoader results by
concatenating their already-built `.dataset` arrays - the same thing
DataLoader does internally for a folder of CSV files (one window set per
file, then concatenated across files), just driven from the caller's side
since state/target_state need to stay combined dim=2 inputs (see the Roll
single-callback-pair note above) rather than split into per-column scalar
inputs the way CSV loading would require.
"""

import os

import numpy as np

from nnodely import Input, Integrate, Modely, Output, Parameter, DataLoader
from nnodely.layers.roll import Roll
from nnodely.layers.time_ops import Concatenate, Select, TimeSelect

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DT = 1.0
WINDOW = 5  # seed window length AND number of chained Roll steps
N_SAMPLES = 2000
BATCH_SIZE = 200
EPOCHS = 100
LEARNING_RATE = 0.002
SEED = 0

TRUE_C1 = 0.08
TRUE_C2 = 0.04
GUESS_C1 = 0.2  # deliberately wrong initial guesses - this is the mismatch
GUESS_C2 = 0.01  # gradient descent is supposed to correct.

PUMP_VALVE_MAX = 0.3
INITIAL_STATE_RANGE = (0.2, 0.8)


def simulate_scenario_window(rng, c1, c2):
    """One independent scenario: WINDOW seed samples + WINDOW future
    (target) samples, all under one constant (pump, valve) - matching
    what Roll's rolled-out model assumes (pump/valve held fixed)."""
    total = 2 * WINDOW
    h1 = np.empty(total, dtype=np.float64)
    h2 = np.empty(total, dtype=np.float64)
    pump = rng.uniform(0.0, PUMP_VALVE_MAX)
    valve = rng.uniform(0.0, PUMP_VALVE_MAX)
    h1[0] = rng.uniform(*INITIAL_STATE_RANGE)
    h2[0] = rng.uniform(*INITIAL_STATE_RANGE)
    for t in range(total - 1):
        rate_h1 = c1 * (1.0 - valve) * pump - c2 * np.sqrt(h1[t])
        rate_h2 = c1 * valve * pump + c2 * np.sqrt(h1[t]) - c2 * np.sqrt(h2[t])
        h1[t + 1] = h1[t] + DT * rate_h1
        h2[t + 1] = h2[t] + DT * rate_h2
    return h1, h2, pump, valve


def simulate_scenario_dict(rng, c1: float, c2: float) -> dict:
    """One independent scenario, shaped for nnodely's DataLoader: a
    WINDOW-row 'state' seed and a WINDOW-row 'target_state' continuation,
    under one constant (pump, valve) pair - matching what Roll's rolled-out
    model assumes (pump/valve held fixed). Every input is exactly WINDOW
    rows long (pump/valve repeat their one constant value WINDOW times) so
    DataLoader's own windowing has nothing to slide over: it can only ever
    produce the single window this scenario is meant to provide."""
    h1, h2, pump, valve = simulate_scenario_window(rng, c1, c2)
    return {
        "state": np.stack([h1[:WINDOW], h2[:WINDOW]], axis=1),
        "target_state": np.stack([h1[WINDOW:], h2[WINDOW:]], axis=1),
        "pump": np.full(WINDOW, pump, dtype=np.float32),
        "valve": np.full(WINDOW, valve, dtype=np.float32),
    }


def build_dataloader(
    model: Modely,
    n_samples: int,
    rng: np.random.Generator,
    c1: float,
    c2: float,
) -> DataLoader:
    """Combine n_samples independent scenarios into one nnodely DataLoader.

    Each scenario is handed to its own DataLoader call (see
    simulate_scenario_dict for why that call can only ever yield the one
    window the scenario is meant to provide), and the resulting per-scenario
    datasets are concatenated afterwards. This is the same combination
    DataLoader already does internally for a folder of CSV files - it's just
    driven from here instead, since state/target_state need to stay combined
    dim=2 inputs rather than split into per-file scalar columns.
    """
    loaders = [
        DataLoader(model=model, source=simulate_scenario_dict(rng, c1, c2))
        for _ in range(n_samples)
    ]
    merged = loaders[0]
    merged.dataset = {
        name: np.concatenate([loader.dataset[name] for loader in loaders], axis=0)
        for name in merged.dataset
    }
    merged._num_steps = merged._infer_num_steps()
    return merged


# ---------------------------------------------------------------------------
# Step model: one Euler step from the CURRENT (last) sample of a windowed,
# combined (h1, h2) state.
# ---------------------------------------------------------------------------
c1 = Parameter("c1", value=[GUESS_C1])
c2 = Parameter("c2", value=[GUESS_C2])

state = Input("state", dim=2, sample_time=DT)
pump = Input("pump", dim=1, sample_time=DT)
valve = Input("valve", dim=1, sample_time=DT)

state_window = state.sw(WINDOW)
current = TimeSelect(idx=WINDOW - 1, name="state_current")(state_window)
h1_cur = Select(idx=0, axis=0, name="h1_current")(current)
h2_cur = Select(idx=1, axis=0, name="h2_current")(current)

sqrt_h1 = h1_cur**0.5
sqrt_h2 = h2_cur**0.5
rate_h1 = c1 * (1.0 - valve.last()) * pump.last() - c2 * sqrt_h1
rate_h2 = c1 * valve.last() * pump.last() + c2 * sqrt_h1 - c2 * sqrt_h2

inc_h1 = Integrate(rate_h1, solver="euler", dt=DT, name="roll_h1_increment")
inc_h2 = Integrate(rate_h2, solver="euler", dt=DT, name="roll_h2_increment")
next_h1 = h1_cur + inc_h1
next_h2 = h2_cur + inc_h2
next_state = Concatenate(axis=0, name="next_state_combined")([next_h1, next_h2])

next_state_out = Output("next_state", next_state)
step_body = Modely(
    "dual_tank_step_body", inputs=[state, pump, valve], outputs=[next_state_out]
).build()

# ---------------------------------------------------------------------------
# Outer model: Roll unrolls step_body WINDOW times, chaining its own
# predictions (free-running / closed-loop, not teacher-forced).
# ---------------------------------------------------------------------------
rolled_state = Roll(
    f=step_body, callback={state: next_state_out}, steps=WINDOW, name="dpc_roll"
)
rolled_state_out = Output("rolled_state", rolled_state)

target_state = Input("target_state", dim=2, sample_time=DT)

dual_tank_roll_id = Modely(
    "dual_tank_roll_id",
    inputs=[state, pump, valve],
    outputs=[rolled_state_out],
)
dual_tank_roll_id.minimize(
    "rollout_error", rolled_state_out, target_state.sw(WINDOW), loss="mse"
)
dual_tank_roll_id.build()


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    train_data = build_dataloader(dual_tank_roll_id, N_SAMPLES, rng, TRUE_C1, TRUE_C2)
    dev_data = build_dataloader(
        dual_tank_roll_id, N_SAMPLES // 5, rng, TRUE_C1, TRUE_C2
    )

    print(
        f"Before training: c1={c1.value_numpy} (true {TRUE_C1}), "
        f"c2={c2.value_numpy} (true {TRUE_C2})"
    )

    history = dual_tank_roll_id.train(
        train_data=train_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        optimizer="adamw",
        lr=LEARNING_RATE,
    )
    print("Final training loss:", history["loss"][-1])

    metrics = dual_tank_roll_id.validate(
        val_data=dev_data,
        batch_size=BATCH_SIZE,
        out_dir=os.path.join(SCRIPT_DIR, "roll_id_validation"),
    )
    print("Validation metrics:", metrics["metrics"])

    print(
        f"After training:  c1={c1.value_numpy} (true {TRUE_C1}), "
        f"c2={c2.value_numpy} (true {TRUE_C2})"
    )
