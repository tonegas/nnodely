Recurrent Models
================

.. seealso::

   Step by step, with runnable examples: the tutorial :doc:`Recurrent models
   <../tutorial/10_recurrent_loop>`.

A one-step model predicts the next sample from measured ones. Used as a
simulator, it has to run on its own predictions instead. *nnodely* offers two
ways to feed outputs back into inputs:

:class:`~nnodely.Loop`
    rolls a built model out over a **sequence**, and returns the whole
    trajectory. This is the tool for multi-step training.
:class:`~nnodely.ModelyLoop`
    is a model declared one step at a time, whose states are Inputs fed back
    with their next value, :meth:`Input.feedback() <nnodely.Input.feedback>`.
    It returns the trajectory of every output.

The examples use a simulated mass-spring-damper:

.. code-block:: python

   import numpy as np
   from nnodely import DataLoader, Fir, Input, Linear, Loop, Modely, Output, set_seed

   set_seed(0)

   dt, m, k, c = 0.05, 1.0, 2.0, 0.5
   rng = np.random.default_rng(0)
   n = 3000
   force = np.repeat(rng.uniform(-1.0, 1.0, n // 50), 50)
   position = np.zeros(n)
   velocity = np.zeros(n)
   for i in range(n - 1):
       acceleration = (-k * position[i] - c * velocity[i] + force[i]) / m
       velocity[i + 1] = velocity[i] + dt * acceleration
       position[i + 1] = position[i] + dt * velocity[i + 1]

Loop
----

A :class:`~nnodely.Loop` takes a model, the *body* (built here if it is not
yet), and rolls it out along the outermost sequence axis of the streams that
drive it: a stream with one more sequence axis than the body input it feeds,
declared with ``seq=`` on an Input. It is configured with the body
and its recurrence, then called on the streams that drive it,
``Loop(f, feedback=..., init=...)(inputs)``:

``feedback``
    maps a body input to the body output that feeds it at the next step. The
    Inputs the body feeds back itself - with :meth:`Input.feedback()
    <nnodely.Input.feedback>`, or the state of an :class:`~nnodely.Integrate`,
    a :class:`~nnodely.Derivative` or a :class:`~nnodely.Buffer` - are found
    in its graph and need no entry; ``feedback`` adds to them. It is needed
    for a body that is already built or loaded and feeds nothing back, or to
    reuse one body with different feedbacks.
``init``
    maps a fed-back input, a *state*, to the stream that seeds it. Step 0
    reads the first step of that stream's sequence. A state left out starts at
    zero. Only a state takes an ``init``, and a state has one initial value:
    the hidden state of ``Integrate`` and ``Derivative`` takes its own from
    their ``init``.
``inputs`` (the call)
    maps the other body inputs to streams that carry one value per step
    (exogenous signals). Body inputs left out keep a constant value. A state
    is not among them: its initial value goes to ``init``.
``collect``
    ``True`` (the default) returns the whole trajectory, with the rollout as
    its new outermost sequence axis. ``False`` returns only the last step.
``length``
    fixes the number of steps. It is needed only when no initial value or
    input carries a rollout.

The body below is a linear state-space step, :math:`s[t+1] = A s[t] + B F[t]`,
on the state :math:`s = (x, \dot x)`:

.. code-block:: python

   horizon = 20
   state = Input("state", dim=2, tw=1)
   F_step = Input("F_step", tw=1)
   next_state = Output(
       "next_state",
       Linear(features=2, bias=False)([state])
       + Linear(features=2, bias=False)([F_step]),
   )
   body = Modely("step", inputs=[state, F_step], outputs=[next_state]).build()

   s0 = Input("s0", dim=2, seq=horizon, tw=1)
   F_seq = Input("F_seq", seq=horizon, tw=1)
   trajectory = Output(
       "trajectory",
       Loop(f=body, feedback={state: next_state}, init={state: s0})(
           {F_step: F_seq}
       ),
   )

   simulator = Modely("msd_loop", inputs=[s0, F_seq], outputs=[trajectory])
   s_next = Input("s_next", dim=2, seq=horizon, tw=1)
   simulator.minimize("trajectory", trajectory, s_next)
   simulator.build()
   print(trajectory.shape)   # Shape(S=(20,), T=1, D=(2,)): 20 steps of 1 sample of 2 states

The :class:`~nnodely.DataLoader` cuts the signals into sequences of
``horizon`` steps. The target sequence is the state one step ahead:

.. code-block:: python

   states = np.stack([position, velocity], axis=-1)
   data = {"s0": states[:-1], "F_seq": force[:-1], "s_next": states[1:]}
   train_data = DataLoader(simulator, source=data, step=5)
   simulator.train(train_data, epochs=10, batch_size=32, lr=1e-2, printer=None)

The loop returns every output of the body, in its order. With one, the loop
is that output, a stream. With more, it is all of them: unpack them,
``position_traj, velocity_traj = Loop(...)(inputs)``, or pick one,
``loop[1]``; read as one stream - in an ``Output``, an operator or a layer -
it raises. An output the loop adds to the body for an update the body does
not declare, as the state of an ``Integrate``, is not returned. A fed-back
input with a window, ``Input(..., tw=n)``, is fed back by
shifting: after ``n`` steps the window contains only predictions. A fed-back output is therefore a single
sample, and the window of a fed-back input cannot read future samples
(``tw=(p, f)`` with ``f > 0``). A fed-back input without a window is replaced
by an output without one.

With a dynamic sequence, ``Input(..., seq=-1, tw=1)``, the rollout follows the length of the
data it receives, and no ``length`` is needed. Together
with ``seq_length="full"`` in the :class:`~nnodely.DataLoader`, this trains on
whole simulations of different lengths (see :doc:`data`).

A body that feeds its states back itself needs no ``feedback``. Here the
state of an :class:`~nnodely.Integrate` is hidden in the body, and starts from
the ``init`` of the Integrate, an Input of the body: the Loop binds it to a
stream of the outer graph, as it binds the force.

.. code-block:: python

   from nnodely import Integrate

   F_k, v0 = Input("F_k", tw=1), Input("v0", tw=1)
   v_next = Integrate(dt=dt, init=v0)(F_k / m)
   impulse = Modely("impulse", inputs=[F_k, v0], outputs=[Output("v_next", v_next)])

   F_dyn, v_start = Input("F_dyn", seq=-1, tw=1), Input("v_start", tw=1)
   velocities = Loop(f=impulse)({F_k: F_dyn, v0: v_start})

As a Modely, the Loop can also be called with a list, one stream per input in
the order the body declares them, ``Loop(f=impulse)([F_dyn, v_start])``, or
with the stream alone when it has one input.

ModelyLoop
----------

A :class:`~nnodely.ModelyLoop` is declared one step at a time, as the
equations of a discrete-time system are written. A state is an Input fed back
with its next value by :meth:`Input.feedback() <nnodely.Input.feedback>`:
within a step the Input holds the value at the end of the previous step, and
the stream it is fed back with, which ``feedback`` returns, the value at the
end of this one. The Inputs of a step read each other's previous values, so
coupled dynamics are written as they are, one Input per state variable, and
the model feeds every one back in a single rollout - it is a
:class:`~nnodely.Loop` over the step, which builds the Inputs of the rollout
itself. Here is the mass-spring-damper of the data, its stiffness and damping
to be identified:

.. code-block:: python

   from nnodely import Buffer, IntegrateStep, ModelyLoop, Parameter

   F_in = Input("F_in", tw=1)    # the force of this step
   pos = Input("pos", tw=1)      # the position at the end of the previous step
   vel = Input("vel", tw=1)      # the velocity at the end of the previous step
   k_hat = Parameter("k_hat", value=1.0)
   c_hat = Parameter("c_hat", value=1.0)

   acc = (F_in - k_hat * pos - c_hat * vel) / m
   vel_next = vel.feedback(IntegrateStep(dt=dt)(vel, acc))
   pos_next = pos.feedback(IntegrateStep(dt=dt)(pos, vel_next))

   msd = ModelyLoop("msd", inputs=[F_in, pos, vel], outputs=[Output("pos_next", pos_next)])
   msd.minimize("position", "pos_next", Input("pos_meas", tw=1))
   msd.build()

Every Input of a ModelyLoop carries the horizon as its outermost sequence
axis, one value per step, and every output is a trajectory: its element ``k``
is computed at step ``k``. The horizon is as long as the data, or ``length``
steps when it is given. A fed-back Input listed among the inputs, as ``pos``
and ``vel`` here, starts from its first value in the data, and reads zero when
the data leaves it out. One that is not listed starts from zero. The state of
an Integrate, a Derivative or a Buffer starts from their ``init`` (and the
rate a trapezoidal Integrate keeps, from its ``init_rate``): a number, a
Parameter, a Constant, or an Input whose first step is read; zero without
one.
:meth:`~nnodely.ModelyLoop.minimize` compares the trajectories of streams of
the step: an output, or an Input that nothing feeds back, such as the measured
position one step ahead. :attr:`~nnodely.Modely.inputs` tells how each
Input of a model is fed:

.. code-block:: python

   print(msd.inputs)

The :class:`~nnodely.DataLoader` cuts the signals into sequences of
``seq_length`` steps, and the target is the position one step ahead:

.. code-block:: python

   data = {"F_in": force[:-1], "pos": position[:-1], "vel": velocity[:-1], "pos_meas": position[1:]}
   train_data = DataLoader(msd, source=data, seq_length=50, step=10)
   msd.train(train_data, epochs=30, batch_size=32, lr=5e-2, printer=None)
   print(k_hat.value_numpy, c_hat.value_numpy)   # about 2.0 and 0.5, the true k and c

Built as a :class:`~nnodely.Modely`, the same graph is the one-step model:
nothing feeds the states back, and they are read from data at every step, as
any Input. A fed-back Input that the Modely does not declare is read from data
too, with a warning. The state that :class:`~nnodely.Integrate`,
:class:`~nnodely.Derivative` and :class:`~nnodely.Buffer` keep is not an Input
of yours: built as a Modely, it starts from their ``init``, zero without one,
as at the first step of a ModelyLoop.

.. code-block:: python

   one_step = Modely("msd_one_step", inputs=[F_in, pos, vel],
                     outputs=[Output("pos_pred", pos_next)]).build()

A model called as a block in a ModelyLoop keeps the feedback of its own
fed-back Inputs, one state per call. Where it reads a window of a signal that
the step computes, a :class:`~nnodely.Buffer` holds the past values:
``Buffer(tw=3)(u)`` is ``[u[k-2], u[k-1], u[k]]``, its newest sample the value
of this step. Its ``init`` holds the past values before the first step, as the
``init`` of a Derivative holds the samples its stencil reads: one sample,
repeated, or ``tw - 1`` of them, oldest first. Here a proportional controller
drives a plant that reads its last three commands:

.. code-block:: python

   forces = Input("forces", tw=3)
   level = Input("level", tw=1)
   level_next = level.feedback(0.9 * level + Fir(features=1, bias=False)([forces]))
   plant = Modely("plant", inputs=[forces], outputs=[Output("level_next", level_next)])

   reference = Input("reference", tw=1)
   measured = Input("measured", tw=1)
   command = Output("command", (reference - measured) * 0.5)
   measured.feedback(plant([Buffer(tw=3)(command)]))
   control = ModelyLoop("control", inputs=[reference],
                        outputs=[command, Output("level", measured)]).build()
   print(control({"reference": np.ones((1, 20, 1, 1))})["level"].shape)   # (1, 20, 1, 1)

A fed-back Input belongs to the model whose graph reads it. One that a
called model reads without declaring it, and that the calling graph reads as
well, would be two states, and is refused: to read the state of the call, make
it an output of the called model; to give the called model the caller's
state, declare an Input for it in the called model and pass the state in the
call. A model called with a stream for a fed-back Input it declares keeps its
feedback, and reads that stream as its initial value. A model is saved with
its fed-back Inputs, and so are the models it calls: loaded back, they feed
them back as before, and a Loop finds them again.
