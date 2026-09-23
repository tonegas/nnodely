# ------- Test Model Longitudinal Vehicle Dynamics -------
import os

import time

import numpy as np

from nnodely import (
    Input,
    Output,
    Modely,
    DataLoader,
    Linear,
    Fir,
    ReLU,
    Fuzzify,
    LocalModel,
    set_seed,
)

set_seed(42)

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
SAVE_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
n = 25
na = 1  # na = 21

# Create neural model inputs
velocity = Input("vel")
brake = Input("brk")
gear = Input("gear")
torque = Input("trq")
altitude = Input("alt", dim=na)

# Create neural network relations
air_drag_force = velocity.last()  # * velocity.last()
air_drag_force = Linear(
    out_features=1,
    use_bias=True,
    initializer="ones",
    bias_initializer="zeros",
)([air_drag_force])

breaking_force = Fir(out_features=1)([brake.sw(n)])
breaking_force = ReLU()([breaking_force])
breaking_force = -1 * breaking_force

gravity_force = Linear(out_features=1)([altitude.last()])

fuzzi_gear = Fuzzify(centers=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], function="Rectangular")(
    [gear.last()]
)

local_model = LocalModel(name="local_model")
engine_force = local_model(inputs=torque.sw(n), activations=fuzzi_gear)

# Create neural network output
out = Output(
    "accelleration",
    air_drag_force + breaking_force + gravity_force + engine_force,
)

# Create the nnodely model
vehicle = Modely(
    "vehicle",
    inputs=[velocity, brake, gear, torque, altitude],
    outputs=[out],
)
vehicle.rollback({"vel": "accelleration"}, steps=10, name="vehicle_roll")

# Define the training objective
vehicle.minimize("acc_error", out, Input("acc").last(), loss="mse")

# Build the model
vehicle.build()
assert air_drag_force.kernel is not None
air_drag_force.kernel.assign(np.full((1, 1), -1e-3, dtype=np.float32))

# Plot the model
vehicle.export_html(os.path.join(SAVE_FOLDER, "recurrent_vehicle_model.html"))

# Load the training and the validation dataset
data_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "datasets",
    "train",
)
data_train = DataLoader(
    model=vehicle,
    format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
    source=data_folder,
)
data_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "datasets",
    "validation",
)
data_val = DataLoader(
    model=vehicle,
    format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
    source=data_folder,
)

# Make inference before training
dummy_input = data_val[0]
result = vehicle(dummy_input)
print("dummy input: ", dummy_input)
print("Inference result before training:", result)
print("Expected acceleration:", dummy_input["acc"])

## Train the model and validate
start_time = time.time()
history = vehicle.train(
    train_data=data_train,
    epochs=60,
    batch_size=128,
    lr=0.001,
    optimizer_kwargs={"global_clipnorm": 1.0},
)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

## Validate the model.
report = vehicle.validate(
    val_data=data_val,
    out_dir=os.path.join(SAVE_FOLDER, "recurrent_vehicle_validation"),
    history=history,
)

## Everything printed above is also available as data. `report.metrics()` keeps
## the numbers, `report["acc_error"]` the scored signals themselves.
acc = report.metrics()["acc_error"]
print(f"\nRMSE {acc['rmse']:.4f} m/s^2, fit {acc['fit_pct']:.1f}%")
if acc["non_finite"]:
    raise SystemExit("the rollout did not stay numerically stable - not saving")

## Inference after training
result = vehicle(dummy_input)
print("Inference result after training:", result)
print("Expected acceleration:", dummy_input["acc"])

## Save the model weights
vehicle.save(os.path.join(SAVE_FOLDER, "recurrent_vehicle"))

## Load the model weights
loaded_vehicle = Modely.load(os.path.join(SAVE_FOLDER, "recurrent_vehicle"))

## inference after loading the model weights
new_result = loaded_vehicle(dummy_input)
print("Inference result with loaded model:", new_result)

## plot htlm with the loaded model
loaded_vehicle.export_html(
    os.path.join(SAVE_FOLDER, "recurrent_vehicle_model_loaded.html")
)

## Export the model in keras format
vehicle.export_keras(os.path.join(SAVE_FOLDER, "recurrent_vehicle_model.keras"))

## Load keras model
loaded_keras_model = Modely.import_keras(
    os.path.join(SAVE_FOLDER, "recurrent_vehicle_model.keras"), safe_mode=False
)

## keras inference
keras_result = loaded_keras_model(dummy_input)  # type: ignore
print("Keras imported inference result:", keras_result)

## Export the model in onnx format
vehicle.export_onnx(os.path.join(SAVE_FOLDER, "recurrent_vehicle_model.onnx"))

## onnx validation
onnx_result = Modely.validate_onnx(
    os.path.join(SAVE_FOLDER, "recurrent_vehicle_model.onnx"), dummy_input
)
print("ONNX validation result:", onnx_result)
