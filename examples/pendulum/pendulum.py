import os

from nnodely import (
    Modely,
    Input,
    Output,
    DataLoader,
    Constant,
    Parameter,
    Sin,
    Cos,
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

# Define parameters
gear = Parameter(name="gear", dim=1)  # gear ratio for the motor
m1 = Parameter(name="m1", dim=1)  # mass of the cart
m2 = Parameter(name="m2", dim=1)  # mass of the pendulum
length = Parameter(name="l", dim=1)  # length of the
b = Parameter(name="b", dim=1)  # damping coefficient for the cart
d = Parameter(name="d", dim=1)  # damping coefficient for the pendulum
Inertia = Parameter(name="I", dim=1)  # moment of inertia of the pendulum


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
        I_eff * m2 * length * omega**2 * sin_theta
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

model.minimize(
    "error_pos", source=out_pos, target=Input(name="Ypos", dim=1), loss="mse"
)
model.minimize(
    "error_vel", source=out_vel, target=Input(name="Yvelocity", dim=1), loss="mse"
)
model.minimize(
    "error_angle", source=out_angle, target=Input(name="Yangle", dim=1), loss="mse"
)
model.minimize(
    "error_ang_vel",
    source=out_ang_vel,
    target=Input(name="Yangular_velocity", dim=1),
    loss="mse",
)
model.build()

# Load data
data_struct = {
    "action": "action",
    "Xpos": "Xpos",
    "Xangle": "Xangle",
    "Xvelocity": "Xvelocity",
    "Xangular_velocity": "Xangular_velocity",
    "Ypos": "Ypos",
    "Yangle": "Yangle",
    "Yvelocity": "Yvelocity",
    "Yangular_velocity": "Yangular_velocity",
}
data_train = DataLoader(
    model,
    format=data_struct,
    source=os.path.join("examples", "datasets", "data_inv_pend"),
)
# Train the model
history = model.train(
    train_data=data_train, epochs=600, batch_size=128, lr=1e-4, optimizer="adam"
)

test_data = data_train[0]  # Use the first batch of training data for testing
print(
    "Test data:",
    {
        k: v
        for k, v in test_data.items()
        if k in ["Ypos", "Yvelocity", "Yangle", "Yangular_velocity"]
    },
)
predictions = model(test_data)

# print([f"{key}: model pred {pred}, target {test_data[key.replace('_pred', '')]}" for key, pred in predictions.items()])
print(
    "Model predictions:",
    [f"{key}: model pred {pred}" for key, pred in predictions.items() if "Y" in key],
)
print(
    "Model parameters:",
    {
        layer.name: layer.get_weights()
        for layer in model.model.layers  # type: ignore
        if len(layer.get_weights()) > 0
    },
)
