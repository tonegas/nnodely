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
    air_drag_force = velocity.last() * velocity.last()
    air_drag_force = Linear(out_features=1, use_bias=True)([air_drag_force])

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

    # def from_config(config):
    #     from nnodely.core.stream import Node
    #     tmp = []
    #     for child in config['preds']:
    #         tmp.append(from_config(child))

    #     return Node(name=config['name'], preds=tmp)

    # for output in vehicle.outputs:
    #     config = output.get_config()

    #     new_output = from_config(config)
    #     print(f"Original output: {output}")
    #     print(f"Reconstructed output: {new_output}")

    # new_model = Modely(
    #     "vehicle_reconstructed",
    #     inputs=vehicle.inputs,
    #     outputs=[new_output],
    # )
    # pprint(f"Original model order: {[node.name for node in vehicle.order]}")
    # pprint(f"new model order: {[node.name for node in new_model.order]}")

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

    # Load the training and the validation dataset
    data_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "datasets",
        "vehicle_long_dyn",
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
        "vehicle_long_dyn",
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
    assert "accelleration" in result
    assert result["accelleration"].shape == (1, 1, 1)

    ## Train the model and validate
    import time

    start_time = time.time()
    vehicle.train(
        train_data=data_train,
        epochs=3,
        batch_size=256,
        lr=0.001,
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    vehicle.validate(
        val_data=data_val, batch_size=32, out_dir="html/vehicle_validation"
    )
    end_time = time.time()
    print(f"validation completed in {end_time - start_time:.2f} seconds.")

    ## Inference after training
    result = vehicle(dummy_input)
    print("Inference result after training:", result)
    print("Expected acceleration:", dummy_input["acc"])
    assert "accelleration" in result
    assert result["accelleration"].shape == (1, 1, 1)

    ## Save the model weights
    # vehicle.save("html/vehicle_model_weights.keras")

    ## Load the model weights
    # new_vehicle = Modely.load("html/vehicle_model_weights.keras")

    ## inference after loading the model weights
    # new_result = new_vehicle(dummy_input)
    # assert "accelleration" in new_result
    # assert new_result["accelleration"].shape == (4, 1, 1)
    # assert np.allclose(result["accelleration"], new_result["accelleration"], atol=1e-4)

    ## Export the model in keras format
    vehicle.export_keras("html/vehicle_model.keras")

    ## Load keras model
    loaded_keras_model = Modely.import_keras("html/vehicle_model", safe_mode=False)
    if loaded_keras_model is None:
        raise ValueError("Failed to load the Keras model.")

    ## keras inference
    keras_result = loaded_keras_model(dummy_input)
    print("Inference result after training:", result)
    print("Keras imported inference result:", keras_result)
    print("Expected acceleration:", dummy_input["acc"])
    assert "accelleration" in keras_result
    assert keras_result["accelleration"].shape == (1, 1, 1)
    assert np.allclose(
        result["accelleration"].cpu().detach().numpy(),
        keras_result["accelleration"].cpu().detach().numpy(),
        atol=1e-4,
    )

    ## Export the model in onnx format
    # vehicle.export_onnx("html/vehicle_model")

    ## onnx validation
    # onnx_result = Modely.validate_onnx("html/vehicle_model.onnx", dummy_input)
    # print("ONNX validation result:", onnx_result)
    # assert np.allclose(result["accelleration"], onnx_result[1], atol=1e-4)


# def test_vehicle_longitudinal_dynamics_recurrent():
#     # ------- Test Model Longitudinal Vehicle Dynamics -------
#     from nnodely import Loop
#     n = 25
#     na = 1  # na = 21

#     # Create neural model inputs
#     velocity = Input("vel", seq=20)
#     brake = Input("brk")
#     gear = Input("gear")
#     torque = Input("trq")
#     altitude = Input("alt", dim=na)

#     # Create neural network relations
#     air_drag_force = velocity.last() * velocity.last()
#     air_drag_force = Linear(out_features=1, use_bias=True)([air_drag_force])

#     breaking_force = Fir(out_features=1)([brake.sw(n)])
#     breaking_force = ReLU()([breaking_force])
#     breaking_force = -1 * breaking_force

#     gravity_force = Linear(out_features=1)([altitude.last()])

#     fuzzi_gear = Fuzzify(
#         centers=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], function="Rectangular"
#     )([gear.last()])

#     local_model = LocalModel(
#         input_function=lambda x: Fir(out_features=1)(x), name="local_model"
#     )
#     engine_force = local_model(activation=fuzzi_gear)([torque.sw(n)])

#     # Create neural network output
#     out = Output(
#         "accelleration",
#         air_drag_force + breaking_force + gravity_force + engine_force,
#     )

#     # Create the nnodely model
#     vehicle = Modely(
#         "vehicle",
#         inputs=[velocity, brake, gear, torque, altitude],
#         outputs=[out],
#     )
#     loop_fn = Loop(vehicle, closed_loop={"vel": "accelleration"}, initial_values={"vel": velocity.last()}, name="loop_vehicle")
#     out_recurrent = Output("latest", loop_fn([velocity, brake, gear, torque, altitude]))
#     vehicle_recurrent = Modely(
#         "vehicle_recurrent",
#         inputs=[velocity, brake, gear, torque, altitude],
#         outputs=[out_recurrent],
#     )
#     vehicle_recurrent.build()

#     # Define the training objective
#     vehicle_recurrent.minimize("acc_error", out, Input("acc").last(), loss="mse")

#     # Plot the model
#     vehicle_recurrent.plot(to_file="html/vehicle_model.png")
#     vehicle_recurrent.plot(
#         to_file="html/vehicle_model_flatten.png", include_minimizers=False, flatten=True
#     )
#     vehicle_recurrent.export_html("html/vehicle_model.html")

#     # Load the training and the validation dataset
#     data_folder = os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         "datasets",
#         "vehicle_long_dyn",
#         "train",
#     )
#     data_train = DataLoader(
#         model=vehicle_recurrent,
#         format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
#         source=data_folder,
#     )
#     data_folder = os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         "datasets",
#         "vehicle_long_dyn",
#         "validation",
#     )
#     data_val = DataLoader(
#         model=vehicle_recurrent,
#         format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
#         source=data_folder,
#     )

#     # Make inference before training
#     dummy_input = data_val[0]
#     result = vehicle_recurrent(dummy_input)
#     assert "accelleration" in result
#     assert result["accelleration"].shape == (1, 1, 1)

#     ## Train the model and validate
#     import time

#     start_time = time.time()
#     vehicle_recurrent.train(
#         train_data=data_train,
#         epochs=3,
#         batch_size=256,
#         lr=0.001,
#     )
#     end_time = time.time()
#     print(f"Training completed in {end_time - start_time:.2f} seconds.")

#     start_time = time.time()
#     vehicle_recurrent.validate(val_data=data_val, batch_size=32, out_dir="html/vehicle_validation", show=True)
#     end_time = time.time()
#     print(f"validation completed in {end_time - start_time:.2f} seconds.")

#     ## Inference after training
#     result = vehicle_recurrent(dummy_input)
#     print("Inference result after training:", result)
#     print("Expected acceleration:", dummy_input["acc"])
#     assert "accelleration" in result
#     assert result["accelleration"].shape == (1, 1, 1)

#     ## Export the model in keras format
#     vehicle_recurrent.export_keras("html/vehicle_model.keras")

#     ## Load keras model
#     loaded_keras_model = Modely.import_keras("html/vehicle_model", safe_mode=False)
#     if loaded_keras_model is None:
#         raise ValueError("Failed to load the Keras model.")

#     ## keras inference
#     keras_result = loaded_keras_model(dummy_input)
#     print("Inference result after training:", result)
#     print("Keras imported inference result:", keras_result)
#     print("Expected acceleration:", dummy_input["acc"])
#     assert "accelleration" in keras_result
#     assert keras_result["accelleration"].shape == (1, 1, 1)
#     assert np.allclose(
#         result["accelleration"].cpu().detach().numpy(),
#         keras_result["accelleration"].cpu().detach().numpy(),
#         atol=1e-4,
#     )

## Export the model in onnx format
# vehicle.export_onnx("html/vehicle_model")

## onnx validation
# onnx_result = Modely.validate_onnx("html/vehicle_model.onnx", dummy_input)
# print("ONNX validation result:", onnx_result)
# assert np.allclose(result["accelleration"], onnx_result[1], atol=1e-4)
