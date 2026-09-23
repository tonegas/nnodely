import os

import numpy as np
import matplotlib.pyplot as plt

from nnodely import (
    Modely,
    Input,
    Output,
    DataLoader,
    Constant,
    Parameter,
    Sin,
    Cos,
    Loop,
    Ode,
)

os.environ.setdefault("KERAS_BACKEND", "jax")

import nnodely

nnodely.set_seed(42)

SAVE_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")

# Define inputs
pos = Input(name="Xpos", dim=1)
vel = Input(name="Xvelocity", dim=1)
angle = Input(name="Xangle", dim=1)
ang_vel = Input(name="Xangular_velocity", dim=1)
force = Input(name="action", dim=1)

# Define constants
g = Constant(name="g", value=9.81)  # acceleration due to gravity
dt = Constant(name="dt", value=0.02)  # time step

init_value = 0.1
# Define parameters
gear = Parameter(name="gear", dim=1, value=100)  # gear ratio for the motor
m1 = Parameter(name="m1", dim=1, value=10)  # mass of the cart
m2 = Parameter(name="m2", dim=1, value=5)  # mass of the pendulum
length = Parameter(name="l", dim=1, value=0.3)  # half the length of the pendulum
b = Parameter(name="b", dim=1, value=1.0)  # damping coefficient for the cart
d = Parameter(name="d", dim=1, value=1.0)  # damping coefficient for the pendulum
Inertia = Parameter(name="I", dim=1, value=0.19)  # moment of inertia of the pendulum
# gear = Parameter(name="gear", dim=1)        # gear ratio for the motor
# m1 = Parameter(name="m1", dim=1)            # mass of the cart
# m2 = Parameter(name="m2", dim=1)            # mass of the pendulum
# l = Parameter(name="l", dim=1)              # length of the
# b = Parameter(name="b", dim=1)              # damping coefficient for the cart
# d = Parameter(name="d", dim=1)              # damping coefficient for the pendulum
# I = Parameter(name="I", dim=1)              # moment of inertia of the pendulum


# Define the equations of motion
def inv_pend(p, v, alpha, omega, u):
    sin_theta = Sin()(alpha)
    cos_theta = Cos()(alpha)
    I_eff = Inertia + m2 * length**2
    denom = (m1 + m2) * I_eff - (m2 * length * cos_theta) ** 2

    # Input force
    F = gear * u

    # Friction
    friction_cart = I_eff * b * v
    friction_pend = (m1 + m2) * d * omega

    # Angular acceleration (omega_dot)
    omega_dot = (
        (m1 + m2) * m2 * g * length * sin_theta
        - m2**2 * length**2 * omega**2 * sin_theta * cos_theta
        - friction_pend
        + m2 * length * b * v * cos_theta
        - m2 * length * cos_theta * F
    ) / denom

    # Linear acceleration of the cart (v_dot)
    v_dot = (
        I_eff * m2 * length**2 * omega**2 * sin_theta
        - friction_cart
        - m2**2 * length**2 * g * sin_theta * cos_theta
        + m2 * length * d * omega * cos_theta
        + F * I_eff
    ) / denom

    p_dot = v
    alpha_dot = omega

    return [p_dot, v_dot, alpha_dot, omega_dot]


# Runge-Kutta 4th order method
pos_next, vel_next, angle_next, ang_vel_next = Ode(
    inv_pend, [pos, vel, angle, ang_vel], dt, method="rk4", args=(force,)
)

# Define outputs
out_pos = Output(name="Ypos_pred", stream=pos_next)
out_vel = Output(name="Yvelocity_pred", stream=vel_next)
out_angle = Output(name="Yangle_pred", stream=angle_next)
out_ang_vel = Output(name="Yangular_velocity_pred", stream=ang_vel_next)

# Create model
model = Modely(
    name="InvertedPendulum",
    inputs=[pos, vel, angle, ang_vel, force],
    outputs=[out_pos, out_vel, out_angle, out_ang_vel],
)
model.plot(to_file=os.path.join(SAVE_FOLDER, "model_inv_pend_initial.png"))
model.export_html(os.path.join(SAVE_FOLDER, "model_inv_pend_initial.html"))

sequence_length = (None,)  # dynamic loop axis; horizon is pinned by Loop(length=...)
pos = Input(name="Xpos_s", dim=1, seq=sequence_length)
vel = Input(name="Xvelocity_s", dim=1, seq=sequence_length)
angle = Input(name="Xangle_s", dim=1, seq=sequence_length)
ang_vel = Input(name="Xangular_velocity_s", dim=1, seq=sequence_length)
force = Input(name="action_s", dim=1, seq=sequence_length)

train_seq_length = 10
loop_fn = Loop(
    f=model,
    length=train_seq_length,
    callback={
        "Xpos": "Ypos_pred",
        "Xvelocity": "Yvelocity_pred",
        "Xangle": "Yangle_pred",
        "Xangular_velocity": "Yangular_velocity_pred",
    },
    initial={
        "Xpos": pos,
        "Xvelocity": vel,
        "Xangle": angle,
        "Xangular_velocity": ang_vel,
    },
    inputs={"action": force},
    name="loop_inv_pend",
)

pos_pred, vel_pred, angle_pred, ang_vel_pred = loop_fn
loop_out_pos = Output(name="Ypos_pred_s", stream=pos_pred)
loop_out_vel = Output(name="Yvel_pred_s", stream=vel_pred)
loop_out_angle = Output(name="Yang_pred_s", stream=angle_pred)
loop_out_ang_vel = Output(name="Yang_vel_s", stream=ang_vel_pred)

loop_model = Modely(
    name="model_with_loop",
    inputs=[pos, vel, angle, ang_vel, force],
    outputs=[loop_out_pos, loop_out_vel, loop_out_angle, loop_out_ang_vel],
)

loop_model.minimize(
    "error_pos",
    source=loop_out_pos,
    target=Input(name="Ypos_s", dim=1, seq=sequence_length),
    loss="mse",
)
loop_model.minimize(
    "error_vel",
    source=loop_out_vel,
    target=Input(name="Yvelocity_s", dim=1, seq=sequence_length),
    loss="mse",
)
loop_model.minimize(
    "error_angle",
    source=loop_out_angle,
    target=Input(name="Yangle_s", dim=1, seq=sequence_length),
    loss="mse",
)
loop_model.minimize(
    "error_ang_vel",
    source=loop_out_ang_vel,
    target=Input(name="Yangular_velocity_s", dim=1, seq=sequence_length),
    loss="mse",
)
loop_model.build()
# loop_model.plot(to_file=os.path.join(SAVE_FOLDER, "model_inv_pend.png"))
loop_model.export_html(os.path.join(SAVE_FOLDER, "model_inv_pend.html"))

# Load data
data_struct = {
    "action_s": "action",
    "Xpos_s": "Xpos",
    "Xangle_s": "Xangle",
    "Xvelocity_s": "Xvelocity",
    "Xangular_velocity_s": "Xangular_velocity",
    "Ypos_s": "Ypos",
    "Yangle_s": "Yangle",
    "Yvelocity_s": "Yvelocity",
    "Yangular_velocity_s": "Yangular_velocity",
}
data_train = DataLoader(
    loop_model,
    format=data_struct,
    source=os.path.join("examples", "datasets", "data_inv_pend"),
    seq_length=train_seq_length,
)

# Train the model
print("\nDataset size:", len(data_train))
print("Starting training...")
loop_model.train(train_data=data_train, epochs=100, batch_size=128, lr=5e-4)

# Export the trained model and check the reloaded one still rolls out identically
export_path = os.path.join(SAVE_FOLDER, "model_inv_pend_keras")
loop_model.export_keras(export_path + ".keras")
loaded_model = Modely.import_keras(export_path, safe_mode=False)
# Load data
data_struct = {
    "action_s": "action",
    "Xpos_s": "Xpos",
    "Xangle_s": "Xangle",
    "Xvelocity_s": "Xvelocity",
    "Xangular_velocity_s": "Xangular_velocity",
    "Ypos_s": "Ypos",
    "Yangle_s": "Yangle",
    "Yvelocity_s": "Yvelocity",
    "Yangular_velocity_s": "Yangular_velocity",
}
data_train = DataLoader(
    loop_model,
    format=data_struct,
    source=os.path.join("examples", "datasets", "data_inv_pend"),
    seq_length=150,
)
test_data = data_train[0]  # Use the first batch of training data for testing

predictions = loop_model(test_data)
reloaded_predictions = loaded_model(test_data)  # type: ignore
for key in ["Ypos_pred_s", "Yvel_pred_s", "Yang_pred_s", "Yang_vel_s"]:
    assert np.allclose(
        np.asarray(predictions[key]), np.asarray(reloaded_predictions[key]), atol=1e-5
    ), f"reloaded model diverged on {key}"
# print("Outputs:", predictions)
# print("Target:", {k: v for k, v in test_data.items() if k in ["Ypos_s", "Yvelocity_s", "Yangle_s", "Yangular_velocity_s"]})
# print("Predictions shapes:", {k: v.shape for k, v in predictions.items() if "Y" in k})

# make a plot of the predictions vs the target for each output

for key_t, key_p in zip(
    ["Ypos_s", "Yvelocity_s", "Yangle_s", "Yangular_velocity_s"],
    ["Ypos_pred_s", "Yvel_pred_s", "Yang_pred_s", "Yang_vel_s"],
):
    plt.figure()
    plt.plot(predictions[key_p][0, 0, 0, :], label="Prediction")
    plt.plot(test_data[key_t][0, 0, 0, :], label="Target", linestyle=":")
    plt.title(key_t)
    plt.xlabel("Time step")
    plt.ylabel(key_t)
    plt.legend()
    plt.savefig(os.path.join(SAVE_FOLDER, f"{key_t}_prediction.png"))
    plt.close()
