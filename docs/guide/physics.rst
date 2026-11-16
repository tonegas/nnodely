Adding Physics: Derivatives and Integrators
===========================================

At the end of this page you will be able to differentiate a stream with
respect to an input or to time, integrate a rate, advance states by one ODE
step, and impose a physical law as a training objective.

**Prerequisites:** :doc:`models`, :doc:`training`; :doc:`recurrent` for the
step-by-step layers.

**Tutorial:** :doc:`../tutorial/11_calculus`.

**The example.** The motion of a body: its position, velocity and
acceleration are linked by derivatives and integrals. A decaying signal shows
a physical law used in training, and a pendulum the ODE step.

These layers are ordinary *nnodely* layers: they train, roll out and export
like any other. The time step ``dt`` is always given, never read from the
data.

.. code-block:: python

   import keras
   import numpy as np
   from nnodely import (
       DataLoader, Derivative, DerivativeStep, Gradient, Input, Integrate, IntegrateStep,
       Linear, Loop, Modely, ModelyLoop, Ode, Output, Parameter, Sin, Tanh, set_seed,
   )

   set_seed(0)

Step 1: derivatives with respect to an input
--------------------------------------------

:class:`~nnodely.Gradient` computes the derivative of a stream with respect
to an Input, by the automatic differentiation of the backend. It accounts for
every path from the Input to the stream, trainable weights included, and can
be trained through. This is how a physical law becomes an objective, as in
physics-informed neural networks.

The model below fits :math:`u(t) = e^{-t}` from data, while imposing the law
:math:`\dot u = -u` as a second objective:

.. code-block:: python

   t = Input("t")              # one value per sample, no window
   u = Linear(features=1)(Tanh()(Linear(features=16)(t)))
   du_dt = Gradient(wrt=t)(u)

   u_out = Output("u", u)
   decay = Modely("decay", inputs=[t], outputs=[u_out])
   decay.minimize("data", u_out, Input("u_meas"))
   decay.minimize("physics", du_dt + u)       # no target: the residual goes to zero
   decay.build()

   times = np.linspace(0.0, 2.0, 200)
   data = DataLoader(decay, source={"t": times, "u_meas": np.exp(-times)})
   history = decay.train(data, epochs=50, batch_size=32, lr=1e-2, printer=None)

``Gradient(order=2, wrt=t)`` gives the second derivative.

Step 2: derivatives with respect to time
----------------------------------------

:class:`~nnodely.DerivativeStep` differentiates a stream along its window,
and keeps its length: one derivative per sample, from that sample and the
ones before it. The value before the window, ``init``, comes first in the
call, ``DerivativeStep(dt=dt)(init, x)``, and is zero when left out. With
``dt=1``, the window ``[0, 1, 4, 9]`` gives ``[0, 1, 3, 5]``:

.. code-block:: python

   x = Input("x", tw=4)
   speed = DerivativeStep(dt=1.0)(0.0, x)
   derivative = Modely("derivative", inputs=[x], outputs=[Output("speed", speed)]).build()
   result = derivative({"x": np.array([0.0, 1.0, 4.0, 9.0]).reshape(1, 4, 1)})
   print(keras.ops.convert_to_numpy(result["speed"]).ravel())

.. code-block:: text
   :class: output

   [0. 1. 3. 5.]

``poly_order`` sets how many samples before each one it reads, and so how
exact it is:

- ``1``, the default, is :math:`(x_i - x_{i-1}) / dt`, exact on a line;
- ``2`` is exact on a parabola, ``3`` on a cubic.

A higher order amplifies noise more: keep ``1`` for measured signals. Applied
twice, ``DerivativeStep`` gives the second derivative.

.. code-block:: python

   dt = 0.01
   position = Input("position", tw=10)
   velocity = DerivativeStep(dt=dt)(0.0, position)
   acceleration = DerivativeStep(dt=dt)(velocity)
   sharper_velocity = DerivativeStep(dt=dt, poly_order=2)(position)
   print(velocity.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=10, D=(1,))

Step 3: integrators over a window
---------------------------------

:class:`~nnodely.IntegrateStep` integrates a rate along its window. Its
``method`` is one of:

- ``"euler"``, or ``"rectangular"``: :math:`y_i = y_{i-1} + dt \, r_i`;
- ``"trapezoidal"``: :math:`y_i = y_{i-1} + dt \, (r_{i-1} + r_i) / 2`.

The calling forms are:

- ``IntegrateStep(dt=dt)(rate)``, which starts from zero;
- ``IntegrateStep(dt=dt)(init, rate)``, which starts from ``init``, the value
  before the window;
- ``IntegrateStep(method="trapezoidal", dt=dt)(init, init_rate, rate)``,
  where ``init_rate`` is the rate before the window, zero when left out.

On a one-sample window it is exactly one integration step, the state update
of a recurrent model:

.. code-block:: python

   a = Input("a", tw=1)
   v = Input("v", tw=1)
   v_next = IntegrateStep(dt=dt)(v, a)   # v + dt * a

On a longer window it returns the whole integrated trajectory in one pass, so
a position is obtained from an acceleration with no rollout:

.. code-block:: python

   acceleration_window = Input("acceleration_window", tw=50)
   v_window = IntegrateStep(dt=dt)(acceleration_window)
   x_window = IntegrateStep(dt=dt)(v_window)
   print(x_window.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=50, D=(1,))

The Euler rule inverts the time derivative with the same ``init``:
``IntegrateStep(dt=dt)(x0, DerivativeStep(dt=dt)(x0, x))`` returns ``x``.

Step 4: integrators and derivatives step by step
------------------------------------------------

:class:`~nnodely.Integrate` integrates one step at a time, in a
:class:`~nnodely.ModelyLoop` or in the body of a :class:`~nnodely.Loop`. The
rate is one sample of a step, and the integral is a hidden state, fed back
with the result. The rules, and the results on the same samples, are those of
``IntegrateStep``. Its initial values are settings:

- ``init``, the integral before the first step;
- ``init_rate``, the rate before it, for the trapezoidal rule.

Each is zero without one. An initial value that is an Input reads its first
step, since every Input of a ModelyLoop carries one element per step:

.. code-block:: python

   acc = Input("acc", tw=1)           # one acceleration per step
   v0, a0 = Input("v0", tw=1), Input("a0", tw=1)
   speed = Integrate(dt=dt, method="trapezoidal", init=v0, init_rate=a0)(acc)
   place = Integrate(dt=dt, method="trapezoidal", init=0.0, init_rate=v0)(speed)
   kinematics = ModelyLoop("kinematics", inputs=[acc, v0, a0],
                           outputs=[Output("speed", speed), Output("place", place)])
   kinematics.build()
   result = kinematics({"acc": np.ones((1, 30, 1, 1)), "v0": np.zeros((1, 30, 1, 1)),
                        "a0": np.ones((1, 30, 1, 1))})
   print(keras.ops.convert_to_numpy(result["place"]).shape)   # one value per step

.. code-block:: text
   :class: output

   (1, 30, 1, 1)

:class:`~nnodely.Derivative` differentiates one step at a time, as
``Integrate`` integrates. The samples its rule reads before the step are a
hidden state, fed back with the signal. Its rule and its results on the same
samples are those of ``DerivativeStep``. ``init`` is the value before the
first step, or the values a higher ``poly_order`` reads, oldest first. It is
zero without one: a signal at rest before the start. With the same ``init``,
the Euler rule of ``Integrate`` inverts it:

.. code-block:: python

   p = Input("p", tw=1)               # one position per step
   p0 = Input("p0", tw=1)
   rate = Derivative(dt=dt, init=p0)(p)
   p_again = Integrate(dt=dt, init=p0)(rate)   # p again, in a ModelyLoop

Step 5: one ODE step
--------------------

:func:`~nnodely.Ode` advances a set of states by one step of an explicit
Runge-Kutta method: ``"euler"``, ``"midpoint"``, ``"heun"`` or ``"rk4"``.
Only ``"euler"`` is shared with ``IntegrateStep``. The dynamics are a Python
function that receives one stream per state, followed by ``args``, and
returns their derivatives. A trajectory is this step used as the body of a
:class:`~nnodely.Loop`.

Here a pendulum with an unknown ratio :math:`g/l` is simulated for 100
steps:

.. code-block:: python

   def pendulum(theta, omega, g_over_l):
       return omega, -g_over_l * Sin(theta)

   theta = Input("theta", tw=1)
   omega = Input("omega", tw=1)
   g_over_l = Parameter("g_over_l", value=5.0)
   theta_next, omega_next = Ode(
       pendulum, [theta, omega], dt=0.05, method="rk4", args=(g_over_l,)
   )
   step = Modely(
       "pendulum_step",
       inputs=[theta, omega],
       outputs=[Output("theta_next", theta_next), Output("omega_next", omega_next)],
   ).build()

   theta0 = Input("theta0", seq=100, tw=1)
   omega0 = Input("omega0", seq=100, tw=1)
   theta_traj, omega_traj = Loop(
       f=step,
       feedback={"theta": "theta_next", "omega": "omega_next"},
       init={"theta": theta0, "omega": omega0},
   )()
   simulator = Modely(
       "pendulum",
       inputs=[theta0, omega0],
       outputs=[Output("theta_traj", theta_traj), Output("omega_traj", omega_traj)],
   ).build()

   result = simulator({"theta0": np.full((1, 100, 1, 1), 0.5),
                       "omega0": np.zeros((1, 100, 1, 1))})
   print(keras.ops.convert_to_numpy(result["theta_traj"]).shape)

.. code-block:: text
   :class: output

   (1, 100, 1, 1)

``g_over_l`` is a Parameter, so training the simulator on measured angles
identifies it. ``dt`` can also be a stream, read from the data or learned.

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Error
     - Cause
     - Fix
   * - ``DerivativeStep acts on the window of its stream, and 'x' has none``
     - A time derivative of a stream without a window.
     - Declare the Input with a window, ``tw=n``; step by step, use
       ``Derivative`` in a ModelyLoop.
   * - ``Ode method 'trapezoidal' is not available``
     - ``Ode`` takes only explicit Runge-Kutta methods.
     - Use ``"euler"``, ``"midpoint"``, ``"heun"`` or ``"rk4"``; the
       trapezoidal rule is in ``IntegrateStep`` and ``Integrate``.
   * - A derivative that is very noisy
     - A high ``poly_order`` on a measured signal.
     - Use ``poly_order=1``, or filter the signal first.

See also
--------

:class:`~nnodely.Gradient`, :class:`~nnodely.DerivativeStep`,
:class:`~nnodely.Derivative`, :class:`~nnodely.IntegrateStep`,
:class:`~nnodely.Integrate`, :func:`~nnodely.Ode`, :doc:`recurrent`.
