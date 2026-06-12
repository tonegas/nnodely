import os

import numpy as np
import pytest

from nnodely import (
    Input,
    Fir,
    Linear,
    ReLU,
    Modely,
    DataLoader,
    Fuzzify,
    LocalModel,
    Output,
)

os.environ.setdefault("KERAS_BACKEND", "tensorflow")


@pytest.mark.slow
def test_vehicle_longitudinal_dynamics():
    # ------- Test Model Longitudinal Vehicle Dynamics -------
    n = 25
    na = 1  # na = 21

    # Create neural model inputs
    velocity = Input("vel")
    brake = Input("brk")
    gear = Input("gear")
    torque = Input("trq")
    altitude = Input("alt", dim=na)

    # Create neural network relations
    air_drag_force = Linear(out_features=1, use_bias=True)([velocity.last() ** 2])

    breaking_force = Fir(out_features=1)([brake.sw(n)])
    breaking_force = ReLU()([breaking_force])
    breaking_force = -1 * breaking_force

    gravity_force = Linear(out_features=1)([altitude.last()])

    fuzzi_gear = Fuzzify(
        centers=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], function="Rectangular"
    )([gear.last()])
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

    # Plot the model
    vehicle.plot(to_file="html/vehicle_model.png")
    vehicle.plot(
        to_file="html/vehicle_model_flatten.png", include_minimizers=False, flatten=True
    )
    vehicle.export_html("html/vehicle_model.html")

    # Make inference before training
    dummy_input = {
        "vel": np.ones((4, 1, 1), dtype=np.float32),
        "trq": np.ones((4, 1, n), dtype=np.float32),
        "brk": np.ones((4, 1, n), dtype=np.float32),
        "gear": np.ones((4, 1, 1), dtype=np.float32),
        "alt": np.ones((4, na, 1), dtype=np.float32),
    }
    result = vehicle(dummy_input)
    assert "accelleration" in result
    assert result["accelleration"].shape == (4, 1, 1)

    # Load the training and the validation dataset
    # data_folder = os.path.join(
    #     os.path.dirname(os.path.realpath(__file__)),
    #     "datasets",
    #     "vehicle_long_dyn",
    #     "train",
    # )
    # data_train = DataLoader(
    #     model=vehicle,
    #     format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
    #     source=data_folder,
    # )
    data_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "datasets",
        "vehicle_long_dyn",
        "validation",
    )
    data_val = DataLoader(
        model=vehicle,
        format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
        source=data_folder,
    )

    ## Train the model
    vehicle.train(data_val, epochs=3, batch_size=32, lr=0.001)

    ## Validation

    ## Inference after training

    ## Save the model weights

    ## Load the model weights

    ## inference after loading the model weights

    ## Export the model in onnx format

    ## Load onnx model

    ## onnx inference


if __name__ == "__main__":
    test_vehicle_longitudinal_dynamics()
