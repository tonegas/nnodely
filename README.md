<a name="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/nnodely_console.svg" alt="The nnodely training console" width="820">
</p>

<p align="center">
  <a href="https://pypi.org/project/nnodely/"><img src="https://img.shields.io/pypi/v/nnodely?color=6F84FE&label=PyPI" alt="PyPI"></a>
  <a href="https://nnodely.readthedocs.io/"><img src="https://img.shields.io/readthedocs/nnodely?color=80BFFF&label=docs" alt="Documentation"></a>
  <a href="https://codecov.io/github/tonegas/nnodely"><img src="https://codecov.io/github/tonegas/nnodely/graph/badge.svg?token=8V6P2PSYT4" alt="codecov"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-5B4BFB" alt="Python 3.10 - 3.13">
  <img src="https://img.shields.io/badge/Keras%203-TensorFlow%20%7C%20PyTorch%20%7C%20JAX-7A3FE8" alt="Keras 3: TensorFlow, PyTorch, JAX">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-A5D8FF" alt="License: MIT"></a>
</p>

<p align="center">
  <b>Model-Structured Neural Networks for the modeling, control and estimation of physical systems.</b>
</p>

<p align="center">
  <a href="https://nnodely.readthedocs.io/"><b>Documentation</b></a> &nbsp;•&nbsp;
  <a href="https://nnodely.readthedocs.io/en/latest/getting_started.html"><b>Getting Started</b></a> &nbsp;•&nbsp;
  <a href="https://github.com/tonegas/nnodely-applications"><b>Applications</b></a> &nbsp;•&nbsp;
  <a href="https://github.com/tonegas/nnodely/blob/main/NNODELY_AI_GUIDE.md"><b>Guide for AI assistants</b></a>
</p>

<img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/rule.svg" width="100%" height="4" alt="">

**nnodely** (read the *nn* as an *m*: *Modely*) builds neural networks whose
structure *is* the physics. Instead of a black box, you write the model the way
you would write the equations: FIR filters on past samples, local models
scheduled by a fuzzy variable, derivatives, integrators, ODE solvers. The data
only has to find the coefficients.

- **Structure first.** Every block has a physical meaning, so the trained
  network stays interpretable.
- **Little data.** Structural priors do the heavy lifting, so a few recordings
  are often enough.
- **Generalization.** A model that respects the physics behaves in scenarios it
  never saw during training.
- **Real time.** Small networks, exported to Keras or ONNX for deployment.
- **Any backend.** Built on Keras 3: the same model runs on TensorFlow,
  PyTorch or JAX.

nnodely is not a replacement for general-purpose deep-learning frameworks. It
is a **structured layer on top of them**, purpose-built for physical systems.

## Install

```sh
pip install "nnodely[torch]"        # or [tensorflow], or [jax]
pip install "nnodely[torch,onnx]"   # + ONNX export
```

Pick the backend with `KERAS_BACKEND` (TensorFlow if unset) before importing nnodely:

```sh
export KERAS_BACKEND=torch
```

## Hello, world

A mass-spring-damper, *m ẍ = −k x − c ẋ + F*, learned as its own
discrete-time structure: the next position is a filter over the last positions
plus a gain on the current force.

```python
import numpy as np
from nnodely import DataLoader, Fir, Input, Modely, Output

# Simulate the system: in practice, these are your measurements
dt, m, k, c = 0.05, 1.0, 2.0, 0.5
force = np.repeat(np.random.uniform(-1.0, 1.0, 60), 50)
position, velocity = np.zeros(3000), np.zeros(3000)
for t in range(2999):
    velocity[t + 1] = velocity[t] + dt * (-k * position[t] - c * velocity[t] + force[t]) / m
    position[t + 1] = position[t] + dt * velocity[t + 1]

# The structure: x[t+1] = FIR(x[t-4..t]) + FIR(F[t])
x, F = Input("x"), Input("F")
x_next = Output("x_next", Fir(out_features=1)([x.sw(5)]) + Fir(out_features=1)([F.last()]))

model = Modely("mass_spring_damper", inputs=[x, F], outputs=[x_next])
model.minimize("next_position", x_next, x.next())
model.build()

# Train with the nnodely console, then score the model
data = DataLoader(model, source={"x": position, "F": force})
model.train(data, epochs=100, batch_size=64, lr=1e-2, printer="nnodely")
model.validate(data)
```

`validate()` reports the loss next to RMSE, FIT, R² and the other indicators a
system-identification report is read for, and can plot them.

## What's in the box

| | Blocks |
|---|---|
| **Signals** | `Input` with sample windows (`sw`, `last`, `next`), `Output`, `Parameter`, `Constant`, arithmetic on streams |
| **Structured layers** | `Fir` [[1]](#1), `Linear`, `LocalModel` [[1]](#1) [[3]](#3) [[4]](#4) [[5]](#5), `Fuzzify` [[2]](#2), `EquationLearner` [[6]](#6), `Interpolation`, `BatchNorm` |
| **Calculus** | `Derivative` w.r.t. inputs or time, for physics-informed [[7]](#7) and Sobolev [[8]](#8) training · `Integrate` · `Ode` (Euler, midpoint, Heun, RK4) · `OdeNet` (neural ODEs, adaptive Dormand-Prince, hybrid events) |
| **Recurrence** | `rollback` for multi-step prediction, `Loop` and `Roll` for rollouts over whole trajectories |
| **Data** | `DataLoader` from dicts, DataFrames or CSV folders, multiple simulations, sequences, masks, normalization |
| **Workflow** | `train` with any Keras optimizer and loss · `validate` with system-identification metrics and plots · model composition · `save`/`load`, Keras and ONNX export, interactive HTML graphs |

Every block is documented, with runnable examples, in the
[User Guide](https://nnodely.readthedocs.io/en/latest/guide/index.html).

<details>
<summary><b>Structure of the repository</b></summary>

```bash
nnodely/
├── src/nnodely/
│   ├── core/       # Modely, DataLoader, streams, graph, validation
│   ├── layers/     # every block of the table above
│   └── utils/      # printers, plots, seeding
├── docs/           # documentation, built on Read the Docs
├── tests/          # unit and integration tests, run on all three backends
├── imgs/           # images used in the README and the documentation
└── NNODELY_AI_GUIDE.md  # reference for AI coding assistants
```

</details>

## Writing nnodely code with an AI assistant

The repository ships
[`NNODELY_AI_GUIDE.md`](https://github.com/tonegas/nnodely/blob/main/NNODELY_AI_GUIDE.md),
a reference written for LLMs and coding agents rather than for people. It is
dense on purpose and covers what an assistant needs to produce working code on
the first attempt:

- the hard rules of the library: lifecycle order, how objectives and targets
  work, naming and weight sharing, and the behaviors that fail silently;
- a table that maps physical concepts and equations to nnodely blocks;
- the shape convention `(*dim, time, *seq)` and how windows and sequences are
  aligned by the `DataLoader`;
- exact signatures of every block, of `Modely` and of `DataLoader`;
- when to use `rollback`, `Loop`, `Roll`, `Ode` or `OdeNet`, and how to align
  their training data;
- which models can be saved or exported to Keras and ONNX, on which backend;
- common error messages with their fix, runnable recipes, and a checklist to
  review generated code.

To use it, give the assistant the guide together with a description of the
system you want to model:

```text
<contents of NNODELY_AI_GUIDE.md>

Using nnodely, write a model of a DC motor: the angular velocity w follows
J dw/dt = K i - b w, with J, K and b unknown. I have CSV files with columns
time, current, omega sampled at 1 kHz. Identify J, K and b.
```

Agents that can read URLs can fetch the raw file directly from
`https://raw.githubusercontent.com/tonegas/nnodely/main/NNODELY_AI_GUIDE.md`.
Every behavior the guide marks as verified was checked against the code, so
keep it in sync when the API changes.

## Contributing

Contributions and collaborations are welcome: open an issue for questions and
ideas, or a pull request for a new feature or a fix. See
[CONTRIBUTING.md](CONTRIBUTING.md) to set up the development environment.

## License

nnodely is released under the [MIT License](https://opensource.org/licenses/MIT).

<img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/rule.svg" width="100%" height="4" alt="">

<a name="references"></a>
## References

<a id="1">[1]</a>
Mauro Da Lio, Daniele Bortoluzzi, Gastone Pietro Rosati Papini. (2019).
Modelling longitudinal vehicle dynamics with neural networks.
Vehicle System Dynamics. https://doi.org/10.1080/00423114.2019.1638947 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/model_longit_vehicle_dynamics/model_longit_vehicle_dynamics.py))

<a id="2">[2]</a>
Alice Plebe, Mauro Da Lio, Daniele Bortoluzzi. (2019).
On Reliable Neural Network Sensorimotor Control in Autonomous Vehicles.
IEEE Transaction on Intelligent Transportation System. https://doi.org/10.1109/TITS.2019.2896375

<a id="3">[3]</a>
Mauro Da Lio, Riccardo Donà, Gastone Pietro Rosati Papini, Francesco Biral, Henrik Svensson. (2020).
A Mental Simulation Approach for Learning Neural-Network Predictive Control (in Self-Driving Cars).
IEEE Access. https://doi.org/10.1109/ACCESS.2020.3032780 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/model_lateral_vehicle_dynamics/model_lateral_vehicle_dynamics.ipynb))

<a id="4">[4]</a>
Edoardo Pagot, Mattia Piccinini, Enrico Bertolazzi, Francesco Biral. (2023).
Fast Planning and Tracking of Complex Autonomous Parking Maneuvers With Optimal Control and Pseudo-Neural Networks.
IEEE Access. https://doi.org/10.1109/ACCESS.2023.3330431 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/control_steer_car_parking/control_steer_car_parking.ipynb))

<a id="5">[5]</a>
Mattia Piccinini, Sebastiano Taddei, Matteo Larcher, Mattia Piazza, Francesco Biral. (2023).
A Physics-Driven Artificial Agent for Online Time-Optimal Vehicle Motion Planning and Control.
IEEE Access. https://doi.org/10.1109/ACCESS.2023.3274836 (look [[code basic]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/control_steer_artificial_race_driver/control_steer_artificial_race_driver.ipynb)
and [[code extended]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/control_steer_artificial_race_driver_extended/control_steer_artificial_race_driver_extended.ipynb))

<a id="6">[6]</a>
Hector Perez-Villeda, Justus Piater, Matteo Saveriano. (2023).
Learning and extrapolation of robotic skills using task-parameterized equation learner networks.
Robotics and Autonomous Systems. https://doi.org/10.1016/j.robot.2022.104309 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/equation_learner/equation_learner.ipynb))

<a id="7">[7]</a>
M. Raissi. P. Perdikaris b, G.E. Karniadakis a. (2019).
Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
Journal of Computational Physics. https://doi.org/10.1016/j.jcp.2018.10.045 (look the [[example Burger's equation]](https://github.com/tonegas/nnodely-applications/blob/main/pinn/pinn_Burgers_equation.ipynb))

<a id="8">[8]</a>
Wojciech Marian Czarnecki, Simon Osindero, Max Jaderberg, Grzegorz Świrszcz, Razvan Pascanu. (2017).
Sobolev Training for Neural Networks.
arXiv. https://doi.org/10.48550/arXiv.1706.04859 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/sobolev/Sobolev_learning.ipynb))

<a id="9">[9]</a>
Mattia Piccinini, Matteo Zumerle, Johannes Betz, Gastone Pietro Rosati Papini. (2025).
A Road Friction-Aware Anti-Lock Braking System Based on Model-Structured Neural Networks.
IEEE Open Journal of Intelligent Transportation Systems. https://doi.org/10.1109/OJITS.2025.3563347 (look at the [[code]](https://github.com/tonegas/nnodely-applications/tree/main/vehicle/road_friction_aware_ABS))

<a id="10">[10]</a>
Mauro Da Lio, Mattia Piccinini, Francesco Biral. (2023).
Robust and Sample-Efficient Estimation of Vehicle Lateral Velocity Using Neural Networks With Explainable Structure Informed by Kinematic Principles.
IEEE Transactions on Intelligent Transportation Systems. https://doi.org/10.1109/TITS.2023.3303776

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!--
<a name="cite-us"></a>
## Cite Us

> TODO: Possiamo aggiungere DOI di repo con zenodo e mettere la citazione di quello [guida](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
-->
