Physics-Based Layers
====================

These layers bring calculus into a model: derivatives, integrators and ODE
solvers. They are ordinary *nnodely* layers, so they train, roll out and export
like any other relation. The time step ``dt`` is always explicit and is never
inferred from the data.

.. code-block:: python

   import numpy as np
   from nnodely import (
       DataLoader, Derivative, DerivativeStep, Differentiate, Input, Integrate, IntegrateStep,
       Linear, Loop, Modely,
       Ode, OdeNet, Output, Parameter, set_seed, Sin, Tanh,
   )

   set_seed(0)

Derivatives
-----------

Three layers differentiate a stream: :class:`~nnodely.Differentiate` with respect to an
input, and :class:`~nnodely.DerivativeStep` and :class:`~nnodely.Derivative` with
respect to time, over a window and along a horizon.

**With respect to an input**, ``Differentiate`` computes the derivative by the backend's
automatic differentiation. It accounts for every relation between the input
and the stream, trainable weights included, and can itself be trained through.
This is how a physical law becomes a training objective, as in
physics-informed neural networks. The model below fits
:math:`u(t) = e^{-t}` from data while imposing :math:`\dot u = -u`:

.. code-block:: python

   t = Input("t")              # one value per sample, no window
   u = Linear(features=1)([Tanh()([Linear(features=16)([t])])])
   du_dt = Differentiate(respect_to=t)(u)

   u_out = Output("u", u)
   decay = Modely("decay", inputs=[t], outputs=[u_out])
   decay.minimize("data", u_out, Input("u_meas"))
   decay.minimize("physics", du_dt + u)       # no target: the residual goes to zero
   decay.build()

   times = np.linspace(0.0, 2.0, 200)
   data = DataLoader(decay, source={"t": times, "u_meas": np.exp(-times)})
   decay.train(data, epochs=50, batch_size=32, lr=1e-2, printer=None)

``order=2`` gives the second derivative.

**With respect to time**, ``DerivativeStep`` takes the time step ``dt``, as
``IntegrateStep`` does, and the derivative is a causal finite difference along the
stream's time window. The
window length is preserved. ``init`` is the sample just before the window
(zero by default). ``window`` and ``poly_order`` turn the plain backward
difference into a least-squares polynomial fit, which smooths noisy measured
signals at the price of a delay of about ``(window - 1) / 2`` samples:

.. code-block:: python

   dt = 0.01
   position = Input("position", sw=10)
   velocity = DerivativeStep(dt=dt, init=0.0)(position)
   smooth_velocity = DerivativeStep(dt=dt, window=5)(position)
   print(velocity.shape)   # Shape(S=(), T=10, D=(1,))

Integrators
-----------

:class:`~nnodely.IntegrateStep` integrates a rate along its time window, with the
``"euler"`` (or ``"rectangular"``) or ``"trapezoidal"`` rule. ``init`` is the
value before the window. On a one-sample window it is exactly one integration
step, which is the state update of a recurrent model:

.. code-block:: python

   acceleration = Input("acceleration", sw=1)
   v = Input("v", sw=1)
   v_next = IntegrateStep(dt=dt, init=v)(acceleration)   # v + dt * a

On a longer window it returns the whole integrated trajectory in one pass, so
position can be obtained from acceleration with no rollout:

.. code-block:: python

   acceleration_window = Input("acceleration_window", sw=50)
   v_window = IntegrateStep(dt=dt)(acceleration_window)
   x_window = IntegrateStep(dt=dt)(v_window)
   print(x_window.shape)   # Shape(S=(), T=50, D=(1,))

The Euler rule is the exact inverse of the time derivative with the same
``init``: ``IntegrateStep(dt=dt, init=x0)(DerivativeStep(dt=dt, init=x0)(x))``
returns ``x``.

:class:`~nnodely.Integrate` integrates along a **horizon** instead: the
outermost sequence axis of the rate, one sample per step. It is a block built from an
``IntegrateStep`` and a :class:`~nnodely.Loop` that feeds each integrated
value back as the state of the next step. A dynamic horizon (``seq=-1``)
follows the length of the data at every call, with no rebuild. The rules, and
the results on the same samples, are those of ``IntegrateStep``:

.. code-block:: python

   acc = Input("acc", seq=-1, sw=1)   # one acceleration per step
   v0 = Input("v0", sw=1)
   velocity = Integrate(solver="trapezoidal", dt=dt, init=v0)(acc)
   position = Integrate(solver="trapezoidal", dt=dt, init=0.0)(velocity)
   kinematics = Modely("kinematics", inputs=[acc, v0],
                       outputs=[Output("velocity", velocity), Output("position", position)])
   kinematics.build()
   result = kinematics({"acc": np.ones((1, 30, 1, 1)), "v0": np.zeros((1, 1, 1))})
   print(result["position"].shape)   # (1, 30, 1, 1): one value per step

:class:`~nnodely.Derivative` differentiates along a horizon, as ``Integrate``
integrates: the stencil, its arguments and the results on the same samples
are those of ``DerivativeStep``. It needs no ``Loop``: a derivative reads the
earlier steps of the signal, not of its own result, so every step is computed
at once, whatever the length of the horizon. ``init`` is the value just before
the first step, or the ``window - 1`` values a longer stencil reads, along its
window. With the same ``init``, ``Integrate``'s Euler rule inverts it:

.. code-block:: python

   x = Input("x", seq=-1, sw=1)       # one position per step, e.g. a Loop trajectory
   x0 = Input("x0", sw=1)
   v = Derivative(dt=dt, init=x0)(x)
   x_again = Integrate(dt=dt, init=x0)(v)

``DerivativeStep`` does not differentiate a trajectory: each step of it is a
one-sample window, whose derivative reads only ``init``.

One ODE step
------------

:func:`~nnodely.Ode` advances a set of states by one step of an explicit
Runge-Kutta method: ``"euler"``, ``"midpoint"``, ``"heun"`` or ``"rk4"``. The
dynamics are a Python function that receives one stream per state, followed by
``args``, and returns their derivatives. A trajectory is this step used as the
body of a :class:`~nnodely.Loop`. Here a pendulum with an unknown ratio
:math:`g/l` is simulated for 100 steps:

.. code-block:: python

   def pendulum(theta, omega, g_over_l):
       return omega, -1.0 * g_over_l * Sin()([theta])

   theta = Input("theta", sw=1)
   omega = Input("omega", sw=1)
   g_over_l = Parameter("g_over_l", value=5.0)
   theta_next, omega_next = Ode(
       pendulum, [theta, omega], dt=0.05, method="rk4", args=(g_over_l,)
   )
   step = Modely(
       "pendulum_step",
       inputs=[theta, omega],
       outputs=[Output("theta_next", theta_next), Output("omega_next", omega_next)],
   ).build()

   theta0 = Input("theta0", seq=100, sw=1)
   omega0 = Input("omega0", seq=100, sw=1)
   theta_traj, omega_traj = Loop(
       f=step,
       callback={"theta": "theta_next", "omega": "omega_next"},
       init={"theta": theta0, "omega": omega0},
   )()
   simulator = Modely(
       "pendulum",
       inputs=[theta0, omega0],
       outputs=[Output("theta_traj", theta_traj), Output("omega_traj", omega_traj)],
   ).build()

   result = simulator({"theta0": np.full((1, 100, 1, 1), 0.5), "omega0": np.zeros((1, 100, 1, 1))})
   print(result["theta_traj"].shape)   # (1, 100, 1, 1)

``g_over_l`` is a parameter, so training the simulator on measured angles
identifies it. ``dt`` can also be a stream, read from data or learned.

Neural ODEs
-----------

:class:`~nnodely.OdeNet` integrates a vector field given as a built model.
``states`` maps each state input of the field to the output with its
derivative. ``t`` is a stream with the times to report: the first one is the
initial condition, and the result gets one point per time as its new outer
time. The field must be autonomous (all its inputs are states).

.. code-block:: python

   z = Input("z", dim=2, sw=1)
   field = Modely(
       "field",
       inputs=[z],
       outputs=[Output("dz", Linear(features=2, bias=False)([z]))],
   ).build()

   report_times = Input("report_times", seq=50, sw=1)
   z_traj = OdeNet(
       f=field, states={z: "dz"}, t=report_times, method="rk4", steps=2
   )
   node = Modely("neural_ode", inputs=[z, report_times], outputs=[Output("z_traj", z_traj)]).build()

   result = node({
       "z": np.array([[[2.0, 0.0]]]),
       "report_times": np.linspace(0.0, 5.0, 50).reshape(1, 50, 1, 1),
   })
   print(result["z_traj"].shape)   # (1, 50, 1, 2)

With a fixed method (``"euler"``, ``"midpoint"``, ``"heun"``, ``"rk4"``), each
reported interval is split into ``steps`` equal substeps and the solver is an
unrolled graph that trains and exports like any other layer. ``"dopri5"``
selects an adaptive Dormand-Prince solver controlled by ``rtol``, ``atol`` and
``max_steps``. It cannot be exported to ONNX and cannot be differentiated in
reverse mode on the JAX backend, so it is meant for inference.

``event`` and ``reset`` make the field hybrid, for systems with impacts or
switches. ``event`` names an output of the field that changes sign when the
event happens, and ``reset`` maps every state to the output with its value
right after it. The event time is located inside the step, so its gradient
reaches the parameters.
