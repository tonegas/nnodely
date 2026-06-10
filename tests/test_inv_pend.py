import pytest

from nnodely import Input, Output, Modely, Loop, Parameter, Constant, DataLoader
import numpy as np

import os

from nnodely.layers.trigonometric import Cos, Sin
# os.environ.setdefault("KERAS_BACKEND", "jax")

@pytest.mark.slow
def test_inv_pend(tmp_path):
    # Define inputs
    pos = Input(name="Xpos", dim=1)
    vel = Input(name="Xvelocity", dim=1)
    angle = Input(name="Xangle", dim=1)
    ang_vel = Input(name="Xangular_velocity", dim=1)
    force = Input(name="action", dim=1)

    # Define constants
    g = Constant(name="g", value=9.81)      # acceleration due to gravity
    dt = Constant(name="dt", value=0.02)    # time step
    
    # Define parameters
    gear = Parameter(name="gear", dim=1)    # gear ratio for the motor
    m1 = Parameter(name="m1", dim=1)        # mass of the cart
    m2 = Parameter(name="m2", dim=1)        # mass of the pendulum
    l = Parameter(name="l", dim=1)          # length of the
    b = Parameter(name="b", dim=1)          # damping coefficient for the cart
    d = Parameter(name="d", dim=1)          # damping coefficient for the pendulum
    I = Parameter(name="I", dim=1)          # moment of inertia of the pendulum

    # Define the equations of motion
    def inv_pend(p, v, alpha, omega, u):
        sin_theta = Sin()(alpha)
        cos_theta = Cos()(alpha)
        I_eff = I + m2 * l**2
        denom = (m1+m2)*I_eff - (m2*l*cos_theta)**2

        # Input force
        F = gear * u
        
        # Friction
        friction_cart = I_eff * b*v 
        friction_pend = (m1+m2) * d*omega 
        
        # Angular acceleration (omega_dot)
        omega_dot = ((m1+m2)*m2*g*l*sin_theta
                    - m2**2*l**2*omega**2*sin_theta*cos_theta
                    - friction_pend
                    + m2*l*b*v*cos_theta
                    - m2*l*cos_theta*F
        )/ denom

        # Linear acceleration of the cart (v_dot)
        v_dot = (I_eff*m2*l*omega**2*sin_theta
                - friction_cart
                - m2**2*l**2*g*sin_theta*cos_theta
                + m2*l*d*omega*cos_theta
                + F*I_eff
        )/ denom

        p_dot = v
        alpha_dot = omega

        return [p_dot, v_dot, alpha_dot, omega_dot]
    
    # Runge-Kutta 4th order method
    k1 = inv_pend(pos, vel, angle, ang_vel, force)
    k2 = inv_pend(pos + k1[0]*dt/2, vel + k1[1]*dt/2, angle + k1[2]*dt/2, ang_vel + k1[3]*dt/2, force)
    k3 = inv_pend(pos + k2[0]*dt/2, vel + k2[1]*dt/2, angle + k2[2]*dt/2, ang_vel + k2[3]*dt/2, force)
    k4 = inv_pend(pos + k3[0]*dt, vel + k3[1]*dt, angle + k3[2]*dt, ang_vel + k3[3]*dt, force)

    # Update state variables
    pos_next = pos + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    vel_next = vel + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    angle_next = angle + (dt/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    ang_vel_next = ang_vel + (dt/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    # Define outputs
    out_pos = Output(name="Ypos_pred", stream=pos_next)
    out_vel = Output(name="Yvelocity_pred", stream=vel_next)
    out_angle = Output(name="Yangle_pred", stream=angle_next)
    out_ang_vel = Output(name="Yangular_velocity_pred", stream=ang_vel_next)

    # Create model
    model = Modely(
        name="InvertedPendulum",
        inputs=[pos, vel, angle, ang_vel, force],
        outputs=[out_pos, out_vel, out_angle, out_ang_vel]
    )
    model.build()

    loop_fn = Loop(f=model, closed_loop={"Xpos": "Ypos_pred", "Xvelocity": "Yvelocity_pred", "Xangle": "Yangle_pred", "Xangular_velocity": "Yangular_velocity_pred"}, initial_values={"Xpos": pos, "Xvelocity": vel, "Xangle": angle, "Xangular_velocity": ang_vel}, name="loop_inv_pend")
    out_loop = Output("loop_out", loop_fn([pos, vel, angle, ang_vel, force]))
    model = Modely(name="model_with_loop", inputs=[pos, vel, angle, ang_vel, force], outputs=[out_loop])

    # Add minimizers for each output
    model.minimize("error_pos", source=out_pos, target=Input(name="Ypos", dim=1), loss="mse")
    model.minimize("error_vel", source=out_vel, target=Input(name="Yvelocity", dim=1), loss="mse")
    model.minimize("error_angle", source=out_angle, target=Input(name="Yangle", dim=1), loss="mse")
    model.minimize("error_ang_vel", source=out_ang_vel, target=Input(name="Yangular_velocity", dim=1), loss="mse")
    model.plot(to_file=os.path.join(tmp_path, "model_inv_pend.png"))
    model.build()

    # Load data
    data_struct = {"action": "action", "Xpos": "Xpos", "Xangle": "Xangle", "Xvelocity": "Xvelocity", "Xangular_velocity": "Xangular_velocity", "Ypos": "Ypos", "Yangle": "Yangle", "Yvelocity": "Yvelocity", "Yangular_velocity": "Yangular_velocity"}
    data_train = DataLoader(
        model,
        format=data_struct,
        source=os.path.join("tests", "datasets", "data_inv_pend"),
    )

    # Train the model
    model.train(train_data=data_train, epochs=10, batch_size=64, lr=1e-3)

    # # Export the trained model
    # model.save(os.path.join(tmp_path, "model_inv_pend_exported"))
    # model.export_keras(os.path.join(tmp_path, "model_inv_pend_keras.h5"))

    # # Load the exported model and test it
    # loaded_model = Modely.load(os.path.join(tmp_path, "model_inv_pend_exported"))
    test_data = data_train[0]  # Use the first batch of training data for testing
    predictions = model(test_data)

    assert np.allclose(predictions["Ypos_pred"], test_data["Ypos"], atol=1e-3)
    assert np.allclose(predictions["Yvelocity_pred"], test_data["Yvelocity"], atol=1e-3)
    assert np.allclose(predictions["Yangle_pred"], test_data["Yangle"], atol=1e-3)
    assert np.allclose(predictions["Yangular_velocity_pred"], test_data["Yangular_velocity"], atol=1e-3)
    

if __name__ == "__main__":
    test_inv_pend(os.path.join("html", "model_inv_pend.png"))