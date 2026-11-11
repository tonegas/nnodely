Physics-Based Layers
====================

.. seealso::

   Step by step, with runnable examples: the tutorial :doc:`Calculus
   <../tutorial/11_calculus>`.

These layers bring calculus into a model: derivatives, integrators and ODE
solvers. They are ordinary *nnodely* layers, so they train, roll out and export
like any other relation. The time step ``dt`` is always explicit and is never
inferred from the data.

.. code-block:: python

   import numpy as np
   from nnodely import (
       DataLoader, Derivative, DerivativeStep, Gradient, Input, Integrate, IntegrateStep,
       Linear, Loop, Modely, ModelyLoop,
       Ode, Output, Parameter, set_seed, Sin, Tanh,
   )

   set_seed(0)

Derivatives
-----------

Three layers differentiate a stream: :class:`~nnodely.Gradient` with respect to an
input, and :class:`~nnodely.DerivativeStep` and :class:`~nnodely.Derivative` with
respect to time, over a window and step by step in a ModelyLoop.

**With respect to an input**, ``Gradient`` computes the derivative by the backend's
automatic differentiation. It accounts for every relation between the input
and the stream, trainable weights included, and can itself be trained through.
This is how a physical law becomes a training objective, as in
physics-informed neural networks. The model below fits
:math:`u(t) = e^{-t}` from data while imposing :math:`\dot u = -u`:

.. code-block:: python

   t = Input("t")              # one value per sample, no window
   u = Linear(features=1)([Tanh()([Linear(features=16)([t])])])
   du_dt = Gradient(wrt=t)(u)

   u_out = Output("u", u)
   decay = Modely("decay", inputs=[t], outputs=[u_out])
   decay.minimize("data", u_out, Input("u_meas"))
   decay.minimize("physics", du_dt + u)       # no target: the residual goes to zero
   decay.build()

   times = np.linspace(0.0, 2.0, 200)
   data = DataLoader(decay, source={"t": times, "u_meas": np.exp(-times)})
   decay.train(data, epochs=50, batch_size=32, lr=1e-2, printer=None)

``order=2`` gives the second derivative.

**With respect to time**, ``DerivativeStep(dt=dt)`` differentiates a stream
along its window and keeps its length: one derivative per sample, from that
sample and the ones before it. The value before the window, ``init``, comes
first in the call, ``DerivativeStep(dt=dt)(init, x)``, and is zero when left
out. Applied twice, it gives the second derivative.

``poly_order`` sets how many samples before each one it reads, and so how
exact it is: ``1``, the default, is ``(x[i] - x[i-1]) / dt``, exact on a line;
``2`` is exact on a parabola, ``3`` on a cubic. A higher order amplifies noise
more, so keep ``1`` for measured signals. :class:`~nnodely.Derivative` does the
same one step at a time in a ModelyLoop (see :doc:`recurrent`).

.. code-block:: python

   dt = 0.01
   position = Input("position", tw=10)
   velocity = DerivativeStep(dt=dt)(0.0, position)
   acceleration = DerivativeStep(dt=dt)(velocity)
   sharper_velocity = DerivativeStep(dt=dt, poly_order=2)(position)
   print(velocity.shape)   # Shape(S=(), T=10, D=(1,))

Integrators
-----------

:class:`~nnodely.IntegrateStep` integrates a rate along its time window, with the
``method`` ``"euler"`` (or ``"rectangular"``) or ``"trapezoidal"``, the name
:func:`~nnodely.Ode` uses too. The initial condition, the value before the
window, is given first when the layer is called, ``IntegrateStep(dt=dt)(init,
rate)``; called on the rate alone it starts from zero. The trapezoid also
averages the first rate with the one before the window, given after ``init``,
``IntegrateStep(method="trapezoidal", dt=dt)(init, init_rate, rate)``, and
zero when left out, as for a rate at rest before the window. On a one-sample
window it is exactly one integration step, which is the state update of a
recurrent model:

.. code-block:: python

   acceleration = Input("acceleration", tw=1)
   v = Input("v", tw=1)
   v_next = IntegrateStep(dt=dt)(v, acceleration)   # v + dt * a

On a longer window it returns the whole integrated trajectory in one pass, so
position can be obtained from acceleration with no rollout:

.. code-block:: python

   acceleration_window = Input("acceleration_window", tw=50)
   v_window = IntegrateStep(dt=dt)(acceleration_window)
   x_window = IntegrateStep(dt=dt)(v_window)
   print(x_window.shape)   # Shape(S=(), T=50, D=(1,))

The Euler rule is the exact inverse of the time derivative with the same
``init``: ``IntegrateStep(dt=dt)(x0, DerivativeStep(dt=dt)(x0, x))``
returns ``x``.

:class:`~nnodely.Integrate` integrates one step at a time instead, in a
:class:`~nnodely.ModelyLoop`: the rate is one sample of a step, and the
integral is a hidden Input, ``<name>_<rate>``, fed back with the result,
which the ModelyLoop feeds back with all its other fed-back Inputs, in a
single loop - a :class:`~nnodely.Loop` finds it in its body too. The
trapezoid keeps the rate of the previous step in a second one,
``<name>_<rate>_rate``. The rules, and the results on the same samples, are those of
``IntegrateStep``, and its initial values are configured: ``init``, the
integral before the first step, and ``init_rate``, the rate before it, each
zero without one. Every Input of a ModelyLoop carries the horizon, one
element per step, and an initial value that is an Input reads its first
step:

.. code-block:: python

   acc = Input("acc", tw=1)           # one acceleration per step
   v0, a0 = Input("v0", tw=1), Input("a0", tw=1)
   velocity = Integrate(method="trapezoidal", dt=dt, init=v0, init_rate=a0)(acc)
   position = Integrate(method="trapezoidal", dt=dt, init=0.0, init_rate=v0)(velocity)
   kinematics = ModelyLoop("kinematics", inputs=[acc, v0, a0],
                           outputs=[Output("velocity", velocity), Output("position", position)])
   kinematics.build()
   result = kinematics({"acc": np.ones((1, 30, 1, 1)), "v0": np.zeros((1, 30, 1, 1)),
                        "a0": np.ones((1, 30, 1, 1))})
   print(result["position"].shape)   # (1, 30, 1, 1): one value per step

:class:`~nnodely.Derivative` differentiates one step at a time, as
``Integrate`` integrates: the ``window - 1`` samples before the step are a
hidden Input, ``<name>_<signal>``, fed back with the signal, and the stencil, its
arguments and the results on the same samples are those of
``DerivativeStep``. ``init`` is the value just before the first step, or the
``window - 1`` values a longer stencil reads, along its window; zero without
one, a signal at rest before the start. The results of ``Integrate`` and
``Derivative`` are named after the block. With the same ``init``,
``Integrate``'s Euler rule inverts it:

.. code-block:: python

   x = Input("x", tw=1)               # one position per step
   x0 = Input("x0", tw=1)
   v = Derivative(dt=dt, init=x0)(x)
   x_again = Integrate(dt=dt, init=x0)(v)   # x again, in a ModelyLoop on x and x0

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

   result = simulator({"theta0": np.full((1, 100, 1, 1), 0.5), "omega0": np.zeros((1, 100, 1, 1))})
   print(result["theta_traj"].shape)   # (1, 100, 1, 1)

``g_over_l`` is a parameter, so training the simulator on measured angles
identifies it. ``dt`` can also be a stream, read from data or learned.
