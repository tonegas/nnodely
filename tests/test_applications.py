# import os

# import pytest

# from nnodely import (
#     Input,
#     Fir,
#     Linear,
#     ReLU,
#     Modely,
#     DataLoader,
#     Fuzzify,
#     LocalModel,
#     Output,
# )


# @pytest.mark.slow
# def test_vehicle_longitudinal_dynamics():
#     # ------- Test Model Longitudinal Vehicle Dynamics -------
#     n = 25
#     na = 21

#     # Create neural model inputs
#     velocity = Input("vel")
#     brake = Input("brk")
#     gear = Input("gear")
#     torque = Input("trq")
#     altitude = Input("alt", dim=na)
#     acc = Input("acc")

#     # Create neural network relations
#     air_drag_force = Fir(out_features=1, use_bias=True)([velocity.sw(1)])

#     breaking_force = Fir(out_features=1)([brake.sw(n)])
#     breaking_force = ReLU()([breaking_force])
#     breaking_force = -1 * breaking_force

#     gravity_force = Linear(out_features=1)([altitude.sw(na)])
#     gravity_force = Fir(out_features=1, use_bias=True)([altitude.sw(1)])

#     fuzzi_gear = Fuzzify(
#         centers=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], function="Rectangular"
#     )([gear.sw(1)])
#     local_model = LocalModel(input_function=Fir(out_features=1), name="local_model")
#     engine_force = local_model(fuzzi_gear)
#     engine_force = engine_force([torque.sw(n)])

#     # Create neural network output
#     out = Output(
#         "accelleration",
#         air_drag_force + breaking_force + gravity_force + engine_force([torque.sw(n)]),
#     )

#     # Add the neural model to the nnodely structure and neuralization of the model
#     vehicle = Modely(
#         "vehicle",
#         inputs=[velocity, brake, gear, torque, altitude, acc],
#         outputs=[out],
#     )
#     vehicle.minimize("acc_error", acc.sw(1), out, loss="rmse")
#     vehicle.build()
#     vehicle.plot(to_file="html/vehicle_model.png")

#     # Load the training and the validation dataset
#     data_folder = os.path.join(
#         os.path.dirname(os.path.realpath(__file__)), "dataset", "trainingset"
#     )
#     DataLoader(
#         model=vehicle,
#         format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
#         source=data_folder,
#     )
#     data_folder = os.path.join(
#         os.path.dirname(os.path.realpath(__file__)), "dataset", "validationset"
#     )
#     DataLoader(
#         model=vehicle,
#         format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
#         source=data_folder,
#     )
