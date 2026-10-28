Recurrent Models
================

A one-step model predicts the next sample from measured ones. Used as a
simulator, it has to run on its own predictions instead. *nnodely* offers four
ways to feed outputs back into inputs:

:meth:`Modely.rollback() <nnodely.Modely.rollback>`
    unrolls a whole model for a fixed number of steps, feeding a stream back
    into an input's time window. The result is the prediction after the last step.
:class:`~nnodely.Loop`
    rolls a built model out over a **sequence**, and returns the whole
    trajectory. This is the tool for multi-step training.
:class:`~nnodely.Roll`
    unrolls a built model over the time window of one of its inputs, as a
    block inside a larger graph.
:class:`~nnodely.ModelyLoop`
    is a model declared one step at a time, whose states are Inputs closed
    with their next value, :meth:`Input.close() <nnodely.Input.close>`. It
    returns the trajectory of every output.

The examples use a simulated mass-spring-damper:

.. code-block:: python

   import numpy as np
   from nnodely import DataLoader, Fir, Input, Linear, Loop, Modely, Output, Roll, set_seed

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

Rollback
--------

``rollback({input: stream}, steps)`` evaluates the model ``steps`` times.
After every evaluation, the one-sample result of ``stream`` is appended to the
time window of ``input``, and its oldest sample is dropped. The model's outputs
are those of the last evaluation. It is declared before ``build()``.

.. code-block:: python

   steps = 5
   x = Input("x", sw=5)
   F = Input("F", sw=1)
   x_pred = Output("x_pred", Fir(features=1)([x]) + Fir(features=1)([F]))

   model = Modely("msd_rollback", inputs=[x, F], outputs=[x_pred])
   model.rollback({x: x_pred}, steps=steps)
   model.minimize("ahead", x_pred, Input("x_ahead", sw=1))
   model.build()

Inputs that are not fed back keep the value of their window at every step, and
so does the target. The target is therefore the signal ``steps`` samples ahead,
aligned in the data:

.. code-block:: python

   data = {"x": position[:-steps], "F": force[:-steps], "x_ahead": position[steps:]}
   train_data = DataLoader(model, source=data)
   model.train(train_data, epochs=20, batch_size=64, lr=1e-2, printer=None)

Several inputs can be fed back at once: ``rollback({"x": "x_next", "v":
"v_next"}, steps=10)``. Inputs and streams can be given by name.

A model with a rollback cannot be called as a block, which would evaluate it
once. Its one-step model is a ``Modely`` declared on the same inputs and
outputs without rollback, sharing its weights; to unroll a model inside a
graph, use ``Roll`` or ``Loop``.

Loop
----

A :class:`~nnodely.Loop` takes a model, the *body* (built here if it is not
yet), and rolls it out along the outermost sequence axis of the streams that
drive it: a stream with one more sequence axis than the body input it feeds,
declared with ``seq=`` on an Input. It is configured with the body
and its recurrence, then called on the streams that drive it,
``Loop(f, callback, init=...)(inputs)``:

``callback``
    maps a body input to the body output that feeds it at the next step.
``init``
    maps a fed-back input to the stream that seeds it. Step 0 reads the first
    step of that stream's sequence. A fed-back input left out starts at zero.
``inputs`` (the call)
    maps the other body inputs to streams that carry one value per step
    (exogenous signals). Body inputs left out keep a constant value.
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
   state = Input("state", dim=2, sw=1)
   F_step = Input("F_step", sw=1)
   next_state = Output(
       "next_state",
       Linear(features=2, bias=False)([state])
       + Linear(features=2, bias=False)([F_step]),
   )
   body = Modely("step", inputs=[state, F_step], outputs=[next_state]).build()

   s0 = Input("s0", dim=2, seq=horizon, sw=1)
   F_seq = Input("F_seq", seq=horizon, sw=1)
   trajectory = Output(
       "trajectory",
       Loop(f=body, callback={state: next_state}, init={state: s0})(
           {F_step: F_seq}
       ),
   )

   simulator = Modely("msd_loop", inputs=[s0, F_seq], outputs=[trajectory])
   s_next = Input("s_next", dim=2, seq=horizon, sw=1)
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

A body with several outputs returns one stream per output, which can be
unpacked: ``position_traj, velocity_traj = Loop(...)``. A fed-back input with
a window, ``Input(..., sw=n)``, is closed by shifting: after ``n`` steps the
window contains only predictions. A fed-back output is therefore a single
sample, and the window of a fed-back input cannot read future samples
(``sw=(p, f)`` with ``f > 0``). A fed-back input without a window is replaced
by an output without one.

With a dynamic sequence, ``Input(..., seq=-1, sw=1)``, the rollout follows the length of the
data it receives, and no ``length`` is needed. Together
with ``seq_length="full"`` in the :class:`~nnodely.DataLoader`, this trains on
whole simulations of different lengths (see :doc:`data`).

ModelyLoop
----------

A :class:`~nnodely.ModelyLoop` is declared one step at a time, as the
equations of a discrete-time system are written. A state is an Input closed
with its next value by :meth:`Input.close() <nnodely.Input.close>`: within a
step the Input holds the value at the end of the previous step, and the
stream it is closed with, which ``close`` returns, the value at the end of
this one. The Inputs of a step read each other's previous values, so coupled
dynamics are written as they are, one Input per state variable, and the model
feeds every closed Input back in a single rollout. Here is the
mass-spring-damper of the data, its stiffness and damping to be identified:

.. code-block:: python

   from nnodely import Buffer, IntegrateStep, ModelyLoop, Parameter

   F_in = Input("F_in", sw=1)    # the force of this step
   pos = Input("pos", sw=1)      # the position at the end of the previous step
   vel = Input("vel", sw=1)      # the velocity at the end of the previous step
   k_hat = Parameter("k_hat", value=1.0)
   c_hat = Parameter("c_hat", value=1.0)

   acc = (F_in - k_hat * pos - c_hat * vel) / m
   vel_next = vel.close(IntegrateStep(dt=dt)(vel, acc))
   pos_next = pos.close(IntegrateStep(dt=dt)(pos, vel_next))

   msd = ModelyLoop("msd", inputs=[F_in, pos, vel], outputs=[Output("pos_next", pos_next)])
   msd.minimize("position", "pos_next", Input("pos_meas", sw=1))
   msd.build()

Every Input of a ModelyLoop carries the horizon as its outermost sequence
axis, one value per step, and every output is a trajectory: its element ``k``
is computed at step ``k``. The horizon is as long as the data, or ``length``
steps when it is given. A closed Input listed among the inputs, as ``pos`` and
``vel`` here, starts from its first value in the data, and reads zero when the
data leaves it out. One that is not listed starts from zero, or from the
``init`` it is closed with, ``x.close(stream, init=x0)``: a number, a
Parameter, a Constant, or an Input whose first step is read.
:meth:`~nnodely.ModelyLoop.minimize` compares the trajectories of streams of
the step: an output, or an Input that nothing closes, such as the measured
position one step ahead. :attr:`~nnodely.Modely.all_inputs` tells how each
Input of a model is fed:

.. code-block:: python

   print(msd.all_inputs)

The :class:`~nnodely.DataLoader` cuts the signals into sequences of
``seq_length`` steps, and the target is the position one step ahead:

.. code-block:: python

   data = {"F_in": force[:-1], "pos": position[:-1], "vel": velocity[:-1], "pos_meas": position[1:]}
   train_data = DataLoader(msd, source=data, seq_length=50, step=10)
   msd.train(train_data, epochs=30, batch_size=32, lr=5e-2, printer=None)
   print(k_hat.value_numpy, c_hat.value_numpy)   # about 2.0 and 0.5, the true k and c

Built as a :class:`~nnodely.Modely`, the same graph is the one-step model:
nothing feeds the closed Inputs back, and they are read from data at every
step, as any Input. A closed Input that the Modely does not declare is read
from data too, with a warning.

.. code-block:: python

   one_step = Modely("msd_one_step", inputs=[F_in, pos, vel],
                     outputs=[Output("pos_pred", pos_next)]).build()

A model called as a block in a ModelyLoop keeps the feedback of its own
closed Inputs, one state per call. Where it reads a window of a signal that
the step computes, a :class:`~nnodely.Buffer` holds the past values:
``Buffer(sw=3)(u)`` is ``[u[k-2], u[k-1], u[k]]``, its newest sample the value
of this step. Here a proportional controller drives a plant that reads its
last three commands:

.. code-block:: python

   forces = Input("forces", sw=3)
   level = Input("level", sw=1)
   level_next = level.close(0.9 * level + Fir(features=1, bias=False)([forces]))
   plant = Modely("plant", inputs=[forces], outputs=[Output("level_next", level_next)])

   reference = Input("reference", sw=1)
   measured = Input("measured", sw=1)
   command = Output("command", (reference - measured) * 0.5)
   measured.close(plant([Buffer(sw=3)(command)]))
   control = ModelyLoop("control", inputs=[reference],
                        outputs=[command, Output("level", measured)]).build()
   print(control({"reference": np.ones((1, 20, 1, 1))})["level"].shape)   # (1, 20, 1, 1)

A closed Input belongs to the model whose graph reads it. One that a called
model reads without declaring it, and that the calling graph reads as well,
would be two states, and is refused: to read the state of the call, make it
an output of the called model; to give the called model the caller's state,
declare an Input for it in the called model and pass the state in the call.
A model called with a stream for a closed Input it declares keeps its
feedback, and reads that stream as its initial value. A model is saved with
its closed Inputs: loaded back, it feeds them back as before.

Roll
----

:class:`~nnodely.Roll` unrolls a built model over the window of one of its
inputs, and returns a window of the same length. With the default ``steps``
(the window length) every sample of the result is a prediction:

.. code-block:: python

   x = Input("x_roll", sw=5)
   out = Output("out", Fir(features=1)([x]))
   one_step = Modely("one_step", inputs=[x], outputs=[out]).build()

   rolled = Roll(f=one_step, callback={x: out})
   free_run = Modely("free_run", inputs=[x], outputs=[Output("free", rolled)]).build()
   print(free_run({"x_roll": np.arange(1.0, 6.0).reshape(1, 5, 1)})["free"].shape)   # (1, 5, 1)
