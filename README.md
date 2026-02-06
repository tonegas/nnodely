<p align="center">
<img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/logo_white_info.png" alt="logo" >
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/tonegas/nnodely/badge.svg?branch=main)](https://coveralls.io/github/tonegas/nnodely?branch=main)
[![Documentation](https://readthedocs.org/projects/nnodely/badge/?version=latest&style=default)](https://nnodely.readthedocs.io/en/latest/)

<a name="readme-top"></a>
# nnodely – Model-Structured Neural Networks

Modeling, control, and estimation of physical systems are central to many engineering disciplines. While data-driven methods like neural networks offer powerful tools, they often struggle to **incorporate prior domain knowledge**, limiting their interpretability, generalizability, and safety.

To bridge this gap, we present ***nnodely*** (where "nn" can be read as "m," forming *Modely*) — a framework that facilitates the creation and deployment of **Model-Structured Neural Networks** (**MS-NNs**).  
MS-NNs combine the learning capabilities of neural networks with structural **priors** grounded in **physics, control, and estimation theory**, enabling:

- **Reduced training data** requirements  
- **Generalization** to unseen scenarios  
- **Real-time** deployment in real-world applications  

<h2>Table of Contents</h2>
<ol>
  <li><a href="#whyuse">Why use nnodely?</a></li>
  <li><a href="#gettingstarted">Getting Started</a></li>
  <ul>
      <li><a href="#installation">Installation</a></li>
      <li><a href="#hellow">Hello, World!</a></li>
      <li><a href="#contribute">How to contribute</a></li>
      <li><a href="#examples">Examples </a></li>
  </ul>
  <li><a href="#basicexample">Basic Example</a></li>
  <ul>
      <li><a href="#structuredneuralmodel">Build the neural model</a></li>
      <li><a href="#neuralizemodel">Neuralize the neural model</a></li>
      <li><a href="#loaddataset">Load the dataset</a></li>
      <li><a href="#trainmodel">Train the neural model</a></li>
      <li><a href="#testmodel">Test the neural model</a></li>
  </ul>
  <li><a href="#fonlderstructure">Structure of the Repository</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#references">References</a></li>
</ol>

<a name="whyuse"></a>
## Why use nnodely?

### 1. The Challenge

Designing neural models for physical systems is fundamentally different from designing models for images or text.

In engineering applications, models must often:

- Respect known physical laws or constraints
- Operate in real-time
- Be interpretable
- Integrate within feedback control loops
- Generalize outside the training distribution
- Work with limited datasets

Standard deep learning frameworks are powerful, but they are not designed to:
- Naturally express time-structured operators
- Encode signal-processing concepts
- Embed control architectures
- Separate known and unknown dynamics

As a result, implementing Model-Structured Neural Networks (MS-NNs) from scratch can become complex, error-prone, and inefficient.

---

### 2. The nnodely Methodology

nnodely is built around a simple but powerful idea:

> Instead of learning everything, learn only what is unknown — inside a structured model.

The framework enables users to:

- Define **Inputs**, **Outputs**, and **Parameters**
- Compose models using signal operators (FIR, time windows, local models, parametric functions)
- Explicitly manipulate time (delays, shifts, derivatives)
- Embed classical system-theoretic structures into neural architectures
- Train only the necessary degrees of freedom

The resulting models are:

- Structurally constrained
- Physically meaningful
- Data-efficient
- Compatible with control and estimation pipelines

This approach aligns with methodologies such as:
- Structured system identification
- Physics-informed learning
- Local model networks
- Hybrid modeling (gray-box modeling)

---

### 3. Core Capabilities

nnodely supports:

#### ✔ Structured Modeling
Design neural models using:
- FIR filters
- Linear operators
- Local models
- Parametric functions
- Equation learners
- Time-domain operations

#### ✔ Modeling, Control, and Estimation
MS-NNs can be used to:
- Identify unknown dynamics
- Design adaptive controllers
- Implement observers and estimators
- Embed learned components into closed-loop systems

#### ✔ Explicit Time-Domain Formulation
Unlike many generic frameworks, nnodely treats time as a first-class concept:
- Time windows
- Delays
- Forward shifts (Z-domain interpretation)
- Derivatives (enabling PINN-style formulations)

#### ✔ Compositional Architecture
Multiple neural components (models, controllers, estimators) can be combined into a unified architecture and trained jointly.

#### ✔ Deployment-Ready
Trained MS-NNs can be exported:
- As standalone PyTorch models
- In ONNX format for real-time or embedded deployment

---

### 4. Structured Workflow

nnodely guides users through a clear development pipeline:
<p align="center">
<img src="https://raw.githubusercontent.com/tonegas/nnodely/readme/imgs/phases.png" alt="phases" width="50%" >
</p>

1. **Neural Model Definition**  
   Define structured components using modular operators.

2. **Dataset Integration**  
   Bind signals to structured inputs with minimal preprocessing overhead.

3. **Training & Optimization**  
   Train only the unknown parameters with user-defined losses.

4. **Validation & Analysis**  
   Evaluate model performance and reliability.

5. **Export & Deployment**  
   Convert the MS-NN into production-ready formats.

6. **Model Composition**  
   Combine multiple structured elements into a complete architecture.

---

### 5. Who is it for?

nnodely is particularly suited for:

- Control engineers
- System identification researchers
- Robotics developers
- Energy systems engineers
- Automotive and aerospace applications
- Researchers working on physics-informed or hybrid neural models

If your problem involves **structured dynamics**, **time-series modeling**, or **feedback systems**, nnodely provides abstractions that standard deep learning libraries do not.

---

In short:

nnodely is not a replacement for deep learning frameworks —  
it is a **structured modeling layer on top of them**, purpose-built for physical systems.


<a name="gettingstarted"></a>
## Getting Started

<a name="installation"></a>
### Installation

You can install nnodely from PyPI via:

```sh
pip install nnodely
```

Alternatively, you can build it from source by first cloning the repository and installing the requirements and the nnodely library:

```sh
git clone https://github.com/tonegas/nnodely.git
cd nnodely
pip install -r requirements.txt
pip install .
```
<a name="hellow"></a>
### Hello, World!
To check if `nnodely` is installed correctly try running the following script

```python
from nnodely import *

x = Input('x')
x_out = Output('x_out', Fir(x.last()))
model = nnodely()
model.addModel('x_out', x_out)
print("nnodely installed correctly!")
```


<a name="contribute"></a>
### How to Contribute

To contribute to the nnodely framework, you can:

- Open a pull request if you have a new feature or bug fix.  
- Open an issue if you have a question or suggestion.  

We welcome contributions and collaborations.

<a name="examples"></a>
### Examples

Some **examples of applications** of nnodely in different fields are collected in the following open-source repository:   [nnodely-applications](https://github.com/tonegas/nnodely-applications)

The complete **documentation** is available [here](https://nnodely.readthedocs.io/en/latest/).


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<a name="basicexample"></a>
## Basic Example

This example shows how to use nnodely to create a Model-Structured Neural Network (MS-NN) for a simple **mass-spring-damper mechanical system**.

<a name="structuredneuralmodel"></a>
### Build the Neural Model

<p align="center">
<img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/massspringdamper.png" width="250" alt="linearsys" >
</p>

The system to be modeled is defined by:

```math
M \ddot x = - k x - c \dot x + F
```

Suppose we want to **estimate the future position** of the mass given the current position and the external force.

The estimator is defined as:

```python
x = Input('x')
F = Input('F')
x_z_est = Output('x_z_est', Fir(x.tw(1)) + Fir(F.last()))
```

Input variables are created using the `Input` class.  
In this system, we define two inputs: the mass position `x` and the external force `F`.

The `Output` class defines the model output. It takes two arguments:
1. The name of the output.
2. The structure of the estimator.

Explanation of the methods used:

1. `tw(...)` extracts a time window from a signal. Here, we extract a time window $T_w$ of 1 second.
2. `last()` returns the most recent force sample.
3. `Fir(...)` builds an FIR (finite impulse response) filter with one learnable parameter.

This creates an estimator for the next time-step value of `x` using:

```math
x[1] = \sum_{k=0}^{N_x-1} x[-k]\cdot h_x[(N_x-1)-k] + F[0]\cdot h_F
```

where:

- $x[1]$ is the next position,
- $F[0]$ is the latest force sample,
- $N_x$ is the number of samples in the time window,
- $h_x$ and $h_F$ are learnable parameters.

With $T_w = 1$ second, $N_x = T_w/T_s$, where $T_s$ is the sampling time.

For specific parameter choices ($N_x = 3$, $h_x$ equal to the system’s characteristic polynomial, and $h_F = T_s^2/m$), the MS-NN becomes equivalent to the discrete-time system obtained using the Forward–Euler method.

More generally, the formulation can better adapt to model mismatches and noise by increasing $N_x$.



<a name="neuralizemodel"></a>
### Neuralize the Model

```python
mass_spring_damper = nnodely()
mass_spring_damper.addModel('x_z_est', x_z_est)
mass_spring_damper.addMinimize('next-pos', x.z(-1), x_z_est, 'mse')
mass_spring_damper.neuralizeModel(0.2)
```

- `addModel` adds the defined output to the model.
- `addMinimize` defines the loss function.
- `z(-1)` applies a one-step forward shift (Z-transform notation), equivalent to the `next()` operator.
- `neuralizeModel(0.2)` builds the discrete-time MS-NN with sampling time $T_s = 0.2$ seconds.



<a name="loaddataset"></a>
### Load the Dataset

nnodely requires two pieces of information:
1. The data structure.
2. The dataset location.

```python
data_struct = ['time','x','dx','F']
data_folder = './tutorials/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(
    name='mass_spring_dataset',
    source=data_folder,
    format=data_struct,
    delimiter=';'
)
```

This binds:
- Column 1 → `time`
- Column 2 → `x`
- Column 3 → `dx`
- Column 4 → `F`



<a name="trainmodel"></a>
### Train the Neural Network

```python
mass_spring_damper.trainModel()
```



<a name="testmodel"></a>
### Test the Neural Model

```python
sample = {'F':[0.5], 'x':[0.25, 0.26, 0.27, 0.28, 0.29]}
results = mass_spring_damper(sample)
print(results)
```

Output:

```shell
{'x_z_est':[0.4]}
```



<a name="fonlderstructure"></a>
## Structure of the Repository

```bash
nnodely/
├── docs/
├── examples/
├── mplplots/
├── nnodely/
│   ├── basic/        # core low-level classes
│   ├── exporter/     # model export utilities
│   ├── layers/       # supported layers
│   ├── operators/    # core operators
│   ├── support/      # utility functions
│   └── visualizer/   # visualization tools
└── tests/            # unit tests
```

<details>
<summary>More info about repository structure</summary>
<a name="nnodelyfolder"></a>

### nnodely Folder
This folder contains all the nnodely library files with relative references.

The `nnodely` main class defined in __nnodely.py__, it contains all the main properties of the nnodely object and it derives from five main operators:
1. __composer.py__ contains all the functions to build the networks: `addModel`, `neuralizeModel`, `addConnection`, `addClosedLoop` etc..
2. __loader.py__ contains the function for managing the dataset, the main function is `dataLoad`.
3. __trainer.py__ contains the function for training the network as the `trainModel`.
4. __exporter.py__ contains all the function for import and export: `saveModel`, `loadModel`, `exportONNX` etc..
5. __validator.py__ contains all the function for validate the model and the `resultsAnalysis`.
All the operators derive from `Network`defined in __network.py__, that contains the shared support functions for all the operators.

The folder basic contains the main classes for the low level functionalities: 
1. __model.py__ containts the pytorch template model for the structured network.
2. __modeldef.py__ containts the operation for work with the json model definition.
3. __loss.py__ contains the loss functions.
4. __optimizer.py__ contains the optimizer calss.
6. __relation.py__ contains all the main classes from which all the layers are derived.

The other folders are:
1. exporter that contains the classes for the export functions.
2. support for the support functions.
3. visualizer that contains all the classes related to the visualization.
4. And finally the layers folder.

The layers folder contains all the layers that can be used in the MSNN.
In particular, the model structured NN is defined by `Inputs`, `Outputs` and `Parameters`:
1. __input.py__ contains the Input class used for create an input for the network.
2. __output.py__ contains the Output class used for create an output for the network.
3. __parameter.py__ contains the logic for create a generic parameters and constants.

The main basic layers without parameters are:
1. __activation.py__ this file contains all the activation functions. The activation are mainly based on the pytorch functions.
2. __arithmetic.py__ this file contains the aritmetic functions as: +, -, /, *., **.
3. __trigonometric.py__ this file contains all the trigonometric functions.
4. __part.py__ are used for selecting part of the data.
5. __fuzzify.py__ contains the operation for the fuzzification of a variable, 
commonly used in the local model as activation function as in [[1]](#1) with rectangular activation functions or in [[3]](#3), [[4]](#4) and [[5]](#5) with triangular activation function activation functions.
Using fuzzification it is also possible create a channel coding as presented in [[2]](#2).

The main basic layers with parameters are:
1. __fir.py__ this file contains the finite impulse response filter function. It is a linear operation on the time dimension (second dimension). 
This filter was introduced in [[1]](#1).
2. __linear.py__ this file contains the linear function. Typical Linear operation `W*x+b` operated on the space dimension (third dimension). 
This operation is presented in [[1]](#1).
3. __localmodel.py__ this file contains the logic for build a local model. This operation is presented in [[1]](#1), [[3]](#3), [[4]](#4) and [[5]](#5).
4. __parametricfunction.py__ are the user custom function. The function can use the pytorch syntax. A parametric function is presented in [[3]](#3), [[4]](#4), [[5]](#5).
5. __equationlearner.py__ contains the logic for the equation learner. The equation learner is used for learn a relation input outpur following a list of activation functions. The first implementation is presented in [[6]](#6).
6. __timeoperation.py__ contains the time operation functions. The time operation are used for extract a time window from a signal. The derivative operation can be used to implement Physics-informed neural network [[7]](#7) Sobolev learning [[8]](#8).

<a name="testsfolder"></a>
### Tests Folder
This folder contains the unit tests of the library. Each file tests a specific functionality.

<a name="examplesfolder"></a>
### Examples Folder
The files in the examples folder are a collection of the functionality of the library.
Each file presents a specific functionality of the framework.
This folder is useful to understand the flexibility and capability of the framework.

</details>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="license"></a>
## License
This project is released under the license [License: MIT](https://opensource.org/licenses/MIT).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
