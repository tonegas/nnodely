<a name="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/logo_white_info.png" alt="The nnodely training console" width="820">
</p>

<p align="center">
  <a href="https://pypi.org/project/nnodely/"><img src="https://img.shields.io/pypi/v/nnodely?color=6F84FE&label=PyPI" alt="PyPI"></a>
  <a href="https://nnodely.readthedocs.io/"><img src="https://img.shields.io/readthedocs/nnodely?color=80BFFF&label=docs" alt="Documentation"></a>
  <a href="https://codecov.io/github/tonegas/nnodely"><img src="https://codecov.io/github/tonegas/nnodely/graph/badge.svg?token=8V6P2PSYT4" alt="codecov"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-5B4BFB" alt="Python 3.10 - 3.13">
  <img src="https://img.shields.io/badge/Keras%203-TensorFlow%20%7C%20PyTorch%20%7C%20JAX-7A3FE8" alt="Keras 3: TensorFlow, PyTorch, JAX">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-A5D8FF" alt="License: MIT"></a>
</p>

<img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/rule.svg" width="100%" height="4" alt="">


# Neural Network Framework for Modelling, Control, and Estimation of Physical Systems

Modeling, control, and estimation of physical systems are central to many engineering disciplines. While data-driven methods like neural networks offer powerful tools, they often struggle to **incorporate prior domain knowledge**, limiting their interpretability, generalizability, and safety.

We present ***nnodely*** (where "nn" can be read as "m," forming *Modely*) — a framework that facilitates the creation and deployment of **Model-Structured Neural Networks** (**MS-NNs**).
MS-NNs combine the learning capabilities of neural networks with structural **priors** grounded in **physics, control, and estimation theory**, enabling:

- **Encoding Physics** at the architectural level
- **Reduced training data** requirements
- **Generalization** to unseen scenarios
- **Real time** deployment in real-world applications
- **Multi-backend** TensorFlow, PyTorch or JAX.

In short:

nnodely is not a replacement for a general purpose deep learning frameworks — it is a **structured layer on top of them**, purpose-built for physical systems.


<p align="center">
  📖 <a href="https://nnodely.readthedocs.io/"><b>Documentation</b></a> •
  🚀 <a href="https://github.com/tonegas/nnodely-applications"><b>Applications</b></a> •
  🤖 <a href="https://github.com/tonegas/nnodely/blob/main/NNODELY_AI_GUIDE.md"><b>Guide for AI assistants</b></a>
</p>

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

To check that nnodely is installed, teach it the Fibonacci rule: two
parameters, *x[t+1] = A x[t−1] + B x[t]*, trained until the next value is the
sum of the last two, then run closed loop to generate the series.

```python
import numpy as np
from nnodely import *

set_seed(42)

x = Input("x")
window = x.sw(2)  # [x[t-1], x[t]]
prev, cur = TimeSelect(0)(window), TimeSelect(1)(window)
out = Output("out", Parameter("A") * prev + Parameter("B") * cur)

model = Modely("Fibonacci", inputs=[x], outputs=[out])
model.minimize("target", out, prev + cur)  # Add loss
model.build()

data = DataLoader(model, source={"x": np.random.uniform(-1.0, 1.0, 200)})
model.train(data, epochs=50, batch_size=32, lr=0.05)  # Train the model

# Run it closed loop: each prediction is fed back into x, for 10 steps
loop = Loop(f=model, callback={x: out}, length=10)({x: window})
fib = Output("fib", loop)
generator = Modely("FibonacciGenerator", inputs=[x], outputs=[fib]).build()

start = np.array([[[0.0, 1.0]]], dtype=np.float32)
print(generator({"x": start}))  # Make inference

generator.save("path/to/fibonacci_model")  # Save the nnodely model
```


## What's in the box

| | Blocks |
|---|---|
| **Signals** | `Input` with sample windows (`sw`, `last`, `next`), `Output`, `Parameter`, `Constant`, arithmetic on streams |
| **Structured layers** | `Fir` [[1]](#1), `Linear`, `LocalModel` [[1]](#1) [[3]](#3) [[4]](#4) [[5]](#5), `Fuzzify` [[2]](#2), `EquationLearner` [[6]](#6), `Interpolation`, `BatchNorm` |
| **Calculus** | `Differentiate` w.r.t. inputs and `Derivate` w.r.t. time, for physics-informed [[7]](#7) and Sobolev [[8]](#8) training · `IntegrateStep`, and `Integrate` along a horizon · `Ode` (Euler, midpoint, Heun, RK4) · `OdeNet` (neural ODEs, adaptive Dormand-Prince, hybrid events) |
| **Recurrence** | `rollback` for multi-step prediction, `Loop` and `Roll` for rollouts over whole trajectories |
| **Data** | `DataLoader` from dicts, DataFrames or CSV folders, multiple simulations, sequences, masks, normalization |
| **Workflow** | `train` with any Keras optimizer and loss · `validate` with system-identification metrics and plots · model composition · `save`/`load`, Keras and ONNX export, interactive HTML graphs |

Every block is documented, with runnable examples, in the
[User Guide](https://nnodely.readthedocs.io/en/stable/guide/index.html).

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

The repository ships [`NNODELY_AI_GUIDE.md`](https://github.com/tonegas/nnodely/blob/main/NNODELY_AI_GUIDE.md),
a reference written for LLMs and coding agents rather than for people. It is dense on purpose and covers what an assistant needs to produce working code on the first attempt:

- the hard rules of the library: lifecycle order, how objectives and targets work, naming and weight sharing.
- a table that maps physical concepts and equations to nnodely blocks;
- the shape convention `(*dim, time, *seq)` and how windows and sequences are aligned by the `DataLoader`;
- exact signatures of every block, of `Modely` and of `DataLoader`;
- when to use `rollback`, `Loop`, `Roll`, `Ode` or `OdeNet`, and how to align their training data;
- which models can be saved or exported to Keras and ONNX, on which backend;
- common error messages with their fix, runnable recipes, and a checklist to review generated code.

To use it, give the assistant the guide together with a description of the system you want to model:

```text
<contents of NNODELY_AI_GUIDE.md>

Using nnodely, write a model of a DC motor: the angular velocity w follows
J dw/dt = K i - b w, with J, K and b unknown. I have CSV files with columns
time, current, omega sampled at 1 kHz. Identify J, K and b.
```

Agents that can read URLs can fetch the raw file directly from
`https://raw.githubusercontent.com/tonegas/nnodely/main/NNODELY_AI_GUIDE.md`.
Every behavior the guide marks as verified was checked against the code, so keep it in sync when the API changes.

## Coming from nnodely 1.x

nnodely 2.0 is a rewrite on Keras 3: the model is a `Modely` built from its inputs and outputs, trained on a `DataLoader`, and runs on TensorFlow, PyTorch or JAX. The [CHANGELOG](https://github.com/tonegas/nnodely/blob/main/CHANGELOG.md#migrating-from-1x) maps every 1.x call to its 2.0 counterpart. Models saved with 1.x cannot be loaded by 2.0; to keep using them, pin `pip install "nnodely<2"`.

## Contributing

Contributions and collaborations are welcome: open an issue for questions and ideas, or a pull request for a new feature or a fix. See [CONTRIBUTING.md](https://github.com/tonegas/nnodely/blob/main/CONTRIBUTING.md) to set up the development environment.

## License

nnodely is released under the [MIT License](https://opensource.org/licenses/MIT).

<img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/rule.svg" width="100%" height="4" alt="">

<a name="references"></a>
## References

<a id="1">[1]</a> Mauro Da Lio, Daniele Bortoluzzi, Gastone Pietro Rosati Papini. (2019).
Modelling longitudinal vehicle dynamics with neural networks.
Vehicle System Dynamics. https://doi.org/10.1080/00423114.2019.1638947 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/model_longit_vehicle_dynamics/model_longit_vehicle_dynamics.py))

<a id="2">[2]</a> Alice Plebe, Mauro Da Lio, Daniele Bortoluzzi. (2019).
On Reliable Neural Network Sensorimotor Control in Autonomous Vehicles.
IEEE Transaction on Intelligent Transportation System. https://doi.org/10.1109/TITS.2019.2896375

<a id="3">[3]</a> Mauro Da Lio, Riccardo Donà, Gastone Pietro Rosati Papini, Francesco Biral, Henrik Svensson. (2020).
A Mental Simulation Approach for Learning Neural-Network Predictive Control (in Self-Driving Cars).
IEEE Access. https://doi.org/10.1109/ACCESS.2020.3032780 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/model_lateral_vehicle_dynamics/model_lateral_vehicle_dynamics.ipynb))

<a id="4">[4]</a> Edoardo Pagot, Mattia Piccinini, Enrico Bertolazzi, Francesco Biral. (2023).
Fast Planning and Tracking of Complex Autonomous Parking Maneuvers With Optimal Control and Pseudo-Neural Networks.
IEEE Access. https://doi.org/10.1109/ACCESS.2023.3330431 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/control_steer_car_parking/control_steer_car_parking.ipynb))

<a id="5">[5]</a> Mattia Piccinini, Sebastiano Taddei, Matteo Larcher, Mattia Piazza, Francesco Biral. (2023).
A Physics-Driven Artificial Agent for Online Time-Optimal Vehicle Motion Planning and Control.
IEEE Access. https://doi.org/10.1109/ACCESS.2023.3274836 (look [[code basic]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/control_steer_artificial_race_driver/control_steer_artificial_race_driver.ipynb)
and [[code extended]](https://github.com/tonegas/nnodely-applications/blob/main/vehicle/control_steer_artificial_race_driver_extended/control_steer_artificial_race_driver_extended.ipynb))

<a id="6">[6]</a> Hector Perez-Villeda, Justus Piater, Matteo Saveriano. (2023).
Learning and extrapolation of robotic skills using task-parameterized equation learner networks.
Robotics and Autonomous Systems. https://doi.org/10.1016/j.robot.2022.104309 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/equation_learner/equation_learner.ipynb))

<a id="7">[7]</a> M. Raissi. P. Perdikaris b, G.E. Karniadakis a. (2019).
Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
Journal of Computational Physics. https://doi.org/10.1016/j.jcp.2018.10.045 (look the [[example Burger's equation]](https://github.com/tonegas/nnodely-applications/blob/main/pinn/pinn_Burgers_equation.ipynb))

<a id="8">[8]</a> Wojciech Marian Czarnecki, Simon Osindero, Max Jaderberg, Grzegorz Świrszcz, Razvan Pascanu. (2017).
Sobolev Training for Neural Networks.
arXiv. https://doi.org/10.48550/arXiv.1706.04859 (look the [[code]](https://github.com/tonegas/nnodely-applications/blob/main/sobolev/Sobolev_learning.ipynb))

<a id="9">[9]</a> Mattia Piccinini, Matteo Zumerle, Johannes Betz, Gastone Pietro Rosati Papini. (2025).
A Road Friction-Aware Anti-Lock Braking System Based on Model-Structured Neural Networks.
IEEE Open Journal of Intelligent Transportation Systems. https://doi.org/10.1109/OJITS.2025.3563347 (look at the [[code]](https://github.com/tonegas/nnodely-applications/tree/main/vehicle/road_friction_aware_ABS))

<a id="10">[10]</a> Mauro Da Lio, Mattia Piccinini, Francesco Biral. (2023).
Robust and Sample-Efficient Estimation of Vehicle Lateral Velocity Using Neural Networks With Explainable Structure Informed by Kinematic Principles.
IEEE Transactions on Intelligent Transportation Systems. https://doi.org/10.1109/TITS.2023.3303776

<p align="right">(<a href="#readme-top">back to top</a>)</p>
