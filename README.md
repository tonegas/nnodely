<p align="center">
<img src="https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/logo_white_info.png" alt="logo" >
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/tonegas/nnodely/badge.svg?branch=main)](https://coveralls.io/github/tonegas/nnodely?branch=main)
[![Documentation](https://readthedocs.org/projects/nnodely/badge/?version=docs-update&style=default)](https://nnodely.readthedocs.io/en/docs-update/)

<a name="readme-top"></a>
# nnodely – Model-Structured Neural Networks

:::note
**Full documentation** of the code is available at the following [link](https://nnodely.readthedocs.io/en/docs-update/)
:::

Modeling, control, and estimation of physical systems are central to many engineering disciplines. While data-driven methods like neural networks offer powerful tools, they often struggle to **incorporate prior domain knowledge**, limiting their interpretability, generalizability, and safety.

To bridge this gap, we present ***nnodely*** (where "nn" can be read as "m," forming *Modely*) — a framework that facilitates the creation and deployment of **Model-Structured Neural Networks** (**MS-NNs**).  
MS-NNs combine the learning capabilities of neural networks with structural **priors** grounded in **physics, control, and estimation theory**, enabling:

- **Reduced training data** requirements  
- **Generalization** to unseen scenarios  
- **Real-time** deployment in real-world applications  

In short:

nnodely is not a replacement for deep learning frameworks —  
it is a **structured modeling layer on top of them**, purpose-built for physical systems.

<h2>Table of Contents</h2>
<ol>
  <li><a href="#gettingstarted">Getting Started</a></li>
  <ul>
      <li><a href="#installation">Installation</a></li>
      <li><a href="#hellow">Hello, World!</a></li>
      <li><a href="#contribute">How to contribute</a></li>
      <li><a href="#examples">Examples </a></li>
  </ul>
  <li><a href="#fonlderstructure">Structure of the Repository</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#cite-us">Cite Us</a></li>
</ol>


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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

<a name="cite-us"></a>
## Cite Us

> TODO: Possiamo aggiungere DOI di repo con zenodo e mettere la citazione di quello [guida](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)

<details>
<summary>Featured works</summary>

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
</details>


<p align="right">(<a href="#readme-top">back to top</a>)</p>
