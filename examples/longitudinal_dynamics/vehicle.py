# ------- Test Model Longitudinal Vehicle Dynamics -------

import os

import time

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
air_drag_force = velocity.last() * velocity.last()
air_drag_force = Linear(out_features=1, use_bias=True)([air_drag_force])

breaking_force = Fir(out_features=1)([brake.sw(n)])
breaking_force = ReLU()([breaking_force])
breaking_force = -1 * breaking_force

gravity_force = Linear(out_features=1)([altitude.last()])

fuzzi_gear = Fuzzify(centers=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], function="Rectangular")(
    [gear.last()]
)

local_model = LocalModel(
    input_function=lambda x: Fir(out_features=1)(x), name="local_model"
)
engine_force = local_model(activation=fuzzi_gear)([torque.sw(n)])

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

# Define the training objective
vehicle.minimize("acc_error", out, Input("acc").last(), loss="mse")

# Build the model
vehicle.build()

# Print the model summary
vehicle.summary()

# Plot the model and save inside the local folder
vehicle.export_html(os.path.join(SAVE_FOLDER, "vehicle_model.html"))

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

# Print Dataset information
print(data_train)
print(data_val)

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
    epochs=200,
    batch_size=128,
    lr=0.003,
    printer="nnodely",
)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

start_time = time.time()
vehicle.validate(
    val_data=data_val,
    out_dir=os.path.join(SAVE_FOLDER, "vehicle_validation"),
    history=history,
)
end_time = time.time()
print(f"validation completed in {end_time - start_time:.2f} seconds.")

## Inference after training
result = vehicle(dummy_input)
print("Inference result after training:", result)
print("Expected acceleration:", dummy_input["acc"])

## Save the model weights
vehicle.save(os.path.join(SAVE_FOLDER, "vehicle"))

## Load the model weights
loaded_vehicle = Modely.load(os.path.join(SAVE_FOLDER, "vehicle"))

## inference after loading the model weights
new_result = loaded_vehicle(dummy_input)
print("Inference result with loaded model:", new_result)

## plot htlm with the loaded model
loaded_vehicle.export_html(os.path.join(SAVE_FOLDER, "vehicle_model_loaded.html"))

## Export the model in keras format
vehicle.export_keras(os.path.join(SAVE_FOLDER, "vehicle_model.keras"))

## Load keras model
loaded_keras_model = Modely.import_keras(
    os.path.join(SAVE_FOLDER, "vehicle_model"), safe_mode=False
)

## keras inference
keras_result = loaded_keras_model(dummy_input)  # type: ignore
print("Keras imported inference result:", keras_result)


## Export the model in onnx format
vehicle.export_onnx(os.path.join(SAVE_FOLDER, "vehicle_model"))

## onnx validation
onnx_result = Modely.validate_onnx(
    os.path.join(SAVE_FOLDER, "vehicle_model"), dummy_input
)
print("ONNX validation result:", onnx_result)
