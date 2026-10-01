# Case Studies

This folder contains the examples presented in the paper, updated to **nnodely version 1.5.3**.

The subfolders are organized as follows:

## 1. `lateral_dynamics`

Contains three Python notebooks (plus all the other relative files) relative to the lateral vehicle dynamics model:

- `lateral_dynamics_model.ipynp`: Design of the lateral vehicle dynamics model as presented in the paper
- `lateral_dynamics_control.ipynp`: Design and training of the lateral controller as presented in the paper
- `lateral_dynamics_model_torch.ipynp`: A comparative implementation of the lateral dynamics model developed in native **PyTorch**

## 2. `mass_spring_damper`

Contains the file related to the model and control of the mass-spring-damper system:

- `mass_spring_damper.py` The mass–spring–damper (MSD) system dynamics model and PID controller as the code presented in the paper
- A subfolder containing notebooks that compare the implementation of the same system in **nnodely** and in native **PyTorch**

## 3. `pinn`

Provides an example implementation of a Physics-Informed Neural Network (PINN), inspired by the reference article (available at https://doi.org/10.1016/j.jcp.2018.10.045).

## 4. `neuralODE`

Contains a preliminary implementation of Neural ODEs within the nnodely framework, illustrating continuous-time modeling via parametric functions and integration operators for the mass-spring-damper-system.
