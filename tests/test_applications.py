import os

import numpy as np
import pytest

from nnodely import (
    Modely,
    Input,
    Output,
    Fir,
    DataLoader,
    Linear,
    ReLU,
    Fuzzify,
    LocalModel,
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


# import os

# os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# import pytest
# from nnodely import Input, Output, Modely, Loop, Parameter, Constant, DataLoader
# import numpy as np
# from nnodely.layers.trigonometric import Cos, Sin

# def dummy_input(shape, method="random"):
#     if method == "random":
#         return np.random.rand(*shape).astype(np.float32)
#     elif method == "zeros":
#         return np.zeros(shape, dtype=np.float32)
#     elif method == "ones":
#         return np.ones(shape, dtype=np.float32)
#     elif method == "sequential":
#         return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1

# def test_inv_pend(tmp_path):
#     # Define inputs
#     pos = Input(name="Xpos", dim=1)
#     vel = Input(name="Xvelocity", dim=1)
#     angle = Input(name="Xangle", dim=1)
#     ang_vel = Input(name="Xangular_velocity", dim=1)
#     force = Input(name="action", dim=1)

#     # Define constants
#     g = Constant(name="g", value=9.81)      # acceleration due to gravity
#     dt = Constant(name="dt", value=0.02)    # time step

#     # Define parameters
#     gear = Parameter(name="gear", dim=1)    # gear ratio for the motor
#     m1 = Parameter(name="m1", dim=1)        # mass of the cart
#     m2 = Parameter(name="m2", dim=1)        # mass of the pendulum
#     l = Parameter(name="l", dim=1)          # length of the
#     b = Parameter(name="b", dim=1)          # damping coefficient for the cart
#     d = Parameter(name="d", dim=1)          # damping coefficient for the pendulum
#     I = Parameter(name="I", dim=1)          # moment of inertia of the pendulum

#     # Define the equations of motion
#     def inv_pend(p, v, alpha, omega, u):
#         sin_theta = Sin()(alpha)
#         cos_theta = Cos()(alpha)
#         I_eff = I + m2 * l**2
#         denom = (m1 + m2) * I_eff - (m2 * l * cos_theta) ** 2

#         # Input force
#         F = gear * u

#         # Friction
#         friction_cart = I_eff * b * v
#         friction_pend = (m1 + m2) * d * omega

#         # Angular acceleration (omega_dot)
#         omega_dot = (
#             (m1 + m2) * m2 * g * l * sin_theta
#             - m2**2 * l**2 * omega**2 * sin_theta * cos_theta
#             - friction_pend
#             + m2 * l * b * v * cos_theta
#             - m2 * l * cos_theta * F
#         ) / denom

#         # Linear acceleration of the cart (v_dot)
#         v_dot = (
#             I_eff * m2 * l * omega**2 * sin_theta
#             - friction_cart
#             - m2**2 * l**2 * g * sin_theta * cos_theta
#             + m2 * l * d * omega * cos_theta
#             + F * I_eff
#         ) / denom

#         p_dot = v
#         alpha_dot = omega

#         return [p_dot, v_dot, alpha_dot, omega_dot]

#     # Runge-Kutta 4th order method
#     k1 = inv_pend(pos, vel, angle, ang_vel, force)
#     k2 = inv_pend(
#         pos + k1[0] * dt / 2,
#         vel + k1[1] * dt / 2,
#         angle + k1[2] * dt / 2,
#         ang_vel + k1[3] * dt / 2,
#         force,
#     )
#     k3 = inv_pend(
#         pos + k2[0] * dt / 2,
#         vel + k2[1] * dt / 2,
#         angle + k2[2] * dt / 2,
#         ang_vel + k2[3] * dt / 2,
#         force,
#     )
#     k4 = inv_pend(
#         pos + k3[0] * dt,
#         vel + k3[1] * dt,
#         angle + k3[2] * dt,
#         ang_vel + k3[3] * dt,
#         force,
#     )

#     # Update state variables
#     pos_next = pos + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
#     vel_next = vel + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
#     angle_next = angle + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
#     ang_vel_next = ang_vel + (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

#     # Define outputs
#     out_pos = Output(name="Ypos_pred", stream=pos_next)
#     out_vel = Output(name="Yvelocity_pred", stream=vel_next)
#     out_angle = Output(name="Yangle_pred", stream=angle_next)
#     out_ang_vel = Output(name="Yangular_velocity_pred", stream=ang_vel_next)

#     # Create model
#     model = Modely(
#         name="InvertedPendulum",
#         inputs=[pos, vel, angle, ang_vel, force],
#         outputs=[out_pos, out_vel, out_angle, out_ang_vel],
#     )

#     model.minimize("error_pos", source=out_pos, target=Input(name="Ypos", dim=1).sw(1), loss="mse")
#     model.minimize("error_vel", source=out_vel, target=Input(name="Yvelocity", dim=1).sw(1), loss="mse")
#     model.minimize("error_angle", source=out_angle, target=Input(name="Yangle", dim=1).sw(1), loss="mse")
#     model.minimize("error_ang_vel", source=out_ang_vel, target=Input(name="Yangular_velocity", dim=1).sw(1), loss="mse")
#     model.build()

#     # Load data
#     data_struct = {
#         "action": "action",
#         "Xpos": "Xpos",
#         "Xangle": "Xangle",
#         "Xvelocity": "Xvelocity",
#         "Xangular_velocity": "Xangular_velocity",
#         "Ypos": "Ypos",
#         "Yangle": "Yangle",
#         "Yvelocity": "Yvelocity",
#         "Yangular_velocity": "Yangular_velocity",
#     }
#     data_train = DataLoader(
#         model,
#         format=data_struct,
#         source=os.path.join("tests", "datasets", "data_inv_pend"),
#     )
#     # Train the model
#     history = model.train(train_data=data_train, epochs=30, batch_size=128, lr=1e-3)

#     test_data = data_train[0]  # Use the first batch of training data for testing
#     print("Test data:", {k: v for k, v in test_data.items() if k in ["Ypos", "Yvelocity", "Yangle", "Yangular_velocity"]})
#     predictions = model(test_data)

#     # print([f"{key}: model pred {pred}, target {test_data[key.replace('_pred', '')]}" for key, pred in predictions.items()])
#     print("Model predictions:", [f"{key}: model pred {pred}" for key, pred in predictions.items() if "Y" in key])
#     print("Model parameters:", {layer.name: layer.get_weights() for layer in model.model.layers if len(layer.get_weights()) > 0})

# @pytest.mark.slow
# def test_inv_pend_loop(tmp_path):
#     # Define inputs
#     pos = Input(name="Xpos", dim=1)
#     vel = Input(name="Xvelocity", dim=1)
#     angle = Input(name="Xangle", dim=1)
#     ang_vel = Input(name="Xangular_velocity", dim=1)
#     force = Input(name="action", dim=1)

#     # Define constants
#     g = Constant(name="g", value=9.81)  # acceleration due to gravity
#     dt = Constant(name="dt", value=0.02)  # time step

#     init_value = 0.1
#     # Define parameters
#     gear = Parameter(name="gear", dim=1, value=100)  # gear ratio for the motor
#     m1 = Parameter(name="m1", dim=1, value=10)  # mass of the cart
#     m2 = Parameter(name="m2", dim=1, value=5)  # mass of the pendulum
#     l = Parameter(name="l", dim=1, value=0.3)  # length of the
#     b = Parameter(name="b", dim=1, value=1.0)  # damping coefficient for the cart
#     d = Parameter(name="d", dim=1, value=1.0)  # damping coefficient for the pendulum
#     I = Parameter(name="I", dim=1, value=0.19)  # moment of inertia of the pendulum

#     # Define the equations of motion
#     def inv_pend(p, v, alpha, omega, u):
#         sin_theta = Sin()(alpha)
#         cos_theta = Cos()(alpha)
#         I_eff = I + m2 * l**2
#         denom = (m1 + m2) * I_eff - (m2 * l * cos_theta) ** 2

#         # Input force
#         F = gear * u

#         # Friction
#         friction_cart = I_eff * b * v
#         friction_pend = (m1 + m2) * d * omega

#         # Angular acceleration (omega_dot)
#         omega_dot = (
#             (m1 + m2) * m2 * g * l * sin_theta
#             - m2**2 * l**2 * omega**2 * sin_theta * cos_theta
#             - friction_pend
#             + m2 * l * b * v * cos_theta
#             - m2 * l * cos_theta * F
#         ) / denom

#         # Linear acceleration of the cart (v_dot)
#         v_dot = (
#             I_eff * m2 * l * omega**2 * sin_theta
#             - friction_cart
#             - m2**2 * l**2 * g * sin_theta * cos_theta
#             + m2 * l * d * omega * cos_theta
#             + F * I_eff
#         ) / denom

#         p_dot = v
#         alpha_dot = omega

#         return [p_dot, v_dot, alpha_dot, omega_dot]

#     # Runge-Kutta 4th order method
#     k1 = inv_pend(pos, vel, angle, ang_vel, force)
#     k2 = inv_pend(
#         pos + k1[0] * dt / 2,
#         vel + k1[1] * dt / 2,
#         angle + k1[2] * dt / 2,
#         ang_vel + k1[3] * dt / 2,
#         force,
#     )
#     k3 = inv_pend(
#         pos + k2[0] * dt / 2,
#         vel + k2[1] * dt / 2,
#         angle + k2[2] * dt / 2,
#         ang_vel + k2[3] * dt / 2,
#         force,
#     )
#     k4 = inv_pend(
#         pos + k3[0] * dt,
#         vel + k3[1] * dt,
#         angle + k3[2] * dt,
#         ang_vel + k3[3] * dt,
#         force,
#     )

#     # Update state variables
#     pos_next = pos + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
#     vel_next = vel + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
#     angle_next = angle + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
#     ang_vel_next = ang_vel + (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

#     # Define outputs
#     out_pos = Output(name="Ypos_pred", stream=pos_next)
#     out_vel = Output(name="Yvelocity_pred", stream=vel_next)
#     out_angle = Output(name="Yangle_pred", stream=angle_next)
#     out_ang_vel = Output(name="Yangular_velocity_pred", stream=ang_vel_next)

#     # Create model
#     model = Modely(
#         name="InvertedPendulum",
#         inputs=[pos, vel, angle, ang_vel, force],
#         outputs=[out_pos, out_vel, out_angle, out_ang_vel],
#     )
#     model.plot(to_file=os.path.join(tmp_path, "model_inv_pend_initial.png"))
#     model.export_html(os.path.join("html", "model_inv_pend_initial.html"))

#     sequence_length = 3#(None,) #-1
#     pos = Input(name="Xpos_s", dim=1, seq=sequence_length)
#     vel = Input(name="Xvelocity_s", dim=1, seq=sequence_length)
#     angle = Input(name="Xangle_s", dim=1, seq=sequence_length)
#     ang_vel = Input(name="Xangular_velocity_s", dim=1, seq=sequence_length)
#     force = Input(name="action_s", dim=1, seq=sequence_length)

#     loop_fn = Loop(
#         f=model,
#         closed_loop={
#             "Xpos": "Ypos_pred",
#             "Xvelocity": "Yvelocity_pred",
#             "Xangle": "Yangle_pred",
#             "Xangular_velocity": "Yangular_velocity_pred",
#         },
#         initial_values={
#             "Xpos": pos,
#             "Xvelocity": vel,
#             "Xangle": angle,
#             "Xangular_velocity": ang_vel,
#         },
#         name="loop_inv_pend",
#     )

#     pos_pred, vel_pred, angle_pred, ang_vel_pred = loop_fn([pos, vel, angle, ang_vel, force])
#     loop_out_pos = Output(name="Ypos_pred_s", stream=pos_pred)
#     loop_out_vel = Output(name="Yvel_pred_s", stream=vel_pred)
#     loop_out_angle = Output(name="Yang_pred_s", stream=angle_pred)
#     loop_out_ang_vel = Output(name="Yang_vel_s", stream=ang_vel_pred)

#     loop_model = Modely(
#         name="model_with_loop",
#         inputs=[pos, vel, angle, ang_vel, force],
#         outputs=[loop_out_pos, loop_out_vel, loop_out_angle, loop_out_ang_vel],
#     )

#     loop_model.minimize("error_pos", source=loop_out_pos, target=Input(name="Ypos_s", dim=1, seq=sequence_length), loss="mse")
#     loop_model.minimize("error_vel", source=loop_out_vel, target=Input(name="Yvelocity_s", dim=1, seq=sequence_length), loss="mse")
#     loop_model.minimize("error_angle", source=loop_out_angle, target=Input(name="Yangle_s", dim=1, seq=sequence_length), loss="mse")
#     loop_model.minimize("error_ang_vel", source=loop_out_ang_vel, target=Input(name="Yangular_velocity_s", dim=1, seq=sequence_length), loss="mse")
#     loop_model.build()
#     loop_model.plot(to_file=os.path.join(tmp_path, "model_inv_pend.png"))
#     loop_model.export_html(os.path.join("html", "model_inv_pend.html"))

#     # Load data
#     data_struct = {
#         "action_s": "action",
#         "Xpos_s": "Xpos",
#         "Xangle_s": "Xangle",
#         "Xvelocity_s": "Xvelocity",
#         "Xangular_velocity_s": "Xangular_velocity",
#         "Ypos_s": "Ypos",
#         "Yangle_s": "Yangle",
#         "Yvelocity_s": "Yvelocity",
#         "Yangular_velocity_s": "Yangular_velocity",
#     }
#     data_train = DataLoader(
#         loop_model,
#         format=data_struct,
#         source=os.path.join("tests", "datasets", "data_inv_pend"),
#         seq_length=sequence_length
#     )

#     batch = 1
#     # res = loop_model({
#     #     "Xpos": np.zeros((batch, 1, sequence_length)),
#     #     "Xvelocity": np.ones((batch, 1, sequence_length)),
#     #     "Xangle": np.ones((batch, 1, sequence_length))+1,
#     #     "Xangular_velocity": np.ones((batch, 1, sequence_length))+2,
#     #     "action": np.ones((batch, 1, sequence_length))+3,
#     #     "Ypos": np.zeros((batch, 1, sequence_length)),
#     #     "Yvelocity": np.zeros((batch, 1, sequence_length)),
#     #     "Yangle": np.zeros((batch, 1, sequence_length)),
#     #     "Yangular_velocity": np.zeros((batch, 1, sequence_length)),
#     # })
#     #print("Initial prediction loop:", res)

#     # Train the model
#     print("\nDataset size:", len(data_train))
#     print("Starting training...")
#     loop_model.train(train_data=data_train, epochs=300, batch_size=128, lr=3e-3)
#     # # Export the trained model
#     # model.save(os.path.join(tmp_path, "model_inv_pend_exported"))
#     # model.export_keras(os.path.join(tmp_path, "model_inv_pend_keras.h5"))

#     # # Load the exported model and test it
#     # loaded_model = Modely.load(os.path.join(tmp_path, "model_inv_pend_exported"))
#     test_data = data_train[0]  # Use the first batch of training data for testing

#     predictions = loop_model(test_data)
#     print("Outputs:", predictions)
#     print("Target:", {k: v for k, v in test_data.items() if k in ["Ypos", "Yvelocity", "Yangle", "Yangular_velocity"]})
#     print("Predictions shapes:", {k: v.shape for k, v in predictions.items() if "Y" in k})
