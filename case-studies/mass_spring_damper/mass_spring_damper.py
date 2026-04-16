from nnodely import *


# This example shows how to fit a simple linear model.
# The model is a mass spring damper.
# The data was created previously and loaded from file.
# The data are the position of the mass and the force applied.
# The neural model mirrors the structure of the physical model.
# The network estimates the future position of the mass.

# The time history for x
T_x = 0.1

# Define the neural model
x = Input("x")  # MSNN input (Mass position)
F = Input("F")  # MSNN input (Force)
x_free = Fir(
    W_init="init_negexp",
    W_init_params={"first_value": 0.1, "size_index": 0, "lambda": 3},
)(x.tw(T_x))
x_force = Fir(F.tw(T_x))
x_n = Output("x_n", x_free + x_force)

# Add the neural models to the nnodely structure
msd = Modely(seed=42, workspace="saved")
msd.addModel("neural_msd", x_n)

# These functions are used to impose the minimization objectives.
# Here it is minimized the error between the future position of x get from the dataset
# and the estimator designed using the neural network.
# The minimization is imposed via MSE error.
x_t = Input("x_t")  # Real position
msd.addMinimize("x[t]", x_t.next(), x_n)

# Nauralize the model and getting the neural network.
# The sampling time depends on the datasets.
msd.neuralizeModel(0.01)

# Data load carica i file CSV e costruisce automaticamente il dataset compatibile con la struttura della rete.
data_struct = ["time", ("x", "x_t"), "", "F"]
msd.loadData(
    name="simulations", source="msd-data/data", format=data_struct, delimiter=";"
)

# Neural network train
default_par = {
    "num_of_epochs": 80,
    "train_batch_size": 128,
    "lr": 0.0005,
    "splits": [70, 20, 10],
}
msd.trainModel(training_params=default_par)

# Save the neural model in json format
msd.saveModel(name="msd_preliminary")

# Show the network performance on the test dataset
msd.analyzeModel(splits=[70, 20, 10])
vis = MPLVisualizer()
vis.setModely(msd)
vis.showResult("simulations_test")

# Refine weights with recurrent train and analyze the model showing the performance (mse, FVU, AIC).
msd.trainAndAnalyze(
    num_of_epochs=10,
    prediction_samples=1500,
    step=500,
    lr=0.00001,
    closed_loop={"x": "x_n"},
    training_params=default_par,
)

# Show the network performance on the test dataset on recurrent
vis.showResult("simulations_test")
# Save the neural model in json format
msd.saveModel(name="msd_final")

# Definition of the PID controller network
x_m = Input("x_m")  # measured position
kp = Parameter("P", values=0.5)
ki = Parameter("I", values=0.5)
kd = Parameter("D", values=0.5)
e = x_t.next() - x_m.next()
c = e * kp + Integrate(e) * ki + Differentiate(e) * kd
controlForce = Output("F_PID", c)
msd.addModel("PID", controlForce)

# Neuralization of the whole models
msd.neuralizeModel()

# Train the PID controller
msd.trainModel(
    models="PID",
    closed_loop={"x": "x_n", "x_m": "x_n"},
    connect={"F": "F_PID"},
    prediction_samples=500,
    step=500,
    num_of_epochs=20,
    lr=0.05,
    training_params=default_par,
)

# Print the parameter of the PID
print(
    f"The PID controller kp = {msd.parameters['P']} ki = {msd.parameters['I']} kd = {msd.parameters['D']}"
)

# Test the controller on step and triangular signal
import numpy as np

tt = np.linspace(0, 5, int(5 / 0.01), endpoint=False)
data_target = np.concatenate(
    [
        np.ones(511, dtype=np.float32) * 1.0,  # Step
        -2 * np.ones(500, dtype=np.float32) * 1.0,
        2 * np.abs(2 * (tt * 1 / 5 - np.floor(tt * 1 / 5 + 0.5))) - 1,
        2 * np.abs(2 * (tt * 1 / 2.5 - np.floor(tt * 1 / 2.5 + 0.5))) - 1,
    ]
)
msd.loadData("test_control", {"x_t": data_target})
msd.analyzeModel(
    "test_control",
    prediction_samples=2000,
    closed_loop={"x": "x_n", "x_m": "x_n"},
    connect={"F": "F_PID"},
    batch_size=1,
)
vis.showResult("test_control")
