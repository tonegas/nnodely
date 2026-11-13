Simulating Over Many Steps
==========================

At the end of this page you will be able to run a one-step model on its own
predictions over many steps, train it on whole trajectories, and declare a
dynamic system one step at a time, as its equations are written.

**Prerequisites:** :doc:`models`, :doc:`data`, :doc:`training`.

**Tutorial:** :doc:`../tutorial/10_recurrent_loop`.

A one-step model predicts the next sample from measured ones. Used as a
simulator, it has to run on its own predictions instead. *nnodely* has two
tools for this:

:class:`~nnodely.Loop`
    Rolls a built model, the *body*, out over a **sequence**, feeding some of
    its outputs back into its inputs. It returns the whole trajectory, and is
    the tool for multi-step training.
:class:`~nnodely.ModelyLoop`
    A model declared one step at a time. Its states are Inputs fed back with
    their next value by :meth:`Input.feedback() <nnodely.Input.feedback>`. It
    returns the trajectory of every output.

**The example.** A mass-spring-damper driven by a force, whose stiffness
:math:`k` and damping :math:`c` are to be identified:

.. code-block:: python

   import keras
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

Step 1: the timing of a rollout
-------------------------------

A rollout runs the same step ``k = 0, 1, 2, ...``. A **state** is an Input
fed back with a value the step computes: at step ``k`` it holds the value
computed at step ``k - 1``. At step 0 it holds its *initial value*.

The smallest example adds an input to a total at every step. The total starts
from 10, and the inputs are 1, 2 and 3:

.. code-block:: python

   from nnodely import ModelyLoop

   u_step = Input("u_step", tw=1)
   total = Input("total", tw=1)
   total_next = total.feedback(total + u_step)
   counter = ModelyLoop("counter", inputs=[u_step, total],
                        outputs=[Output("total_next", total_next)]).build()

   steps = np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)   # (batch, steps, tw, dim)
   start = np.array([10.0, 0.0, 0.0]).reshape(1, 3, 1, 1)  # only step 0 is read
   result = counter({"u_step": steps, "total": start})
   print(keras.ops.convert_to_numpy(result["total_next"]).ravel())

.. code-block:: text
   :class: output

   [11. 13. 16.]

.. list-table::
   :header-rows: 1

   * - Step ``k``
     - ``u_step``
     - ``total`` (the state)
     - ``total_next``, element ``k`` of the output
   * - 0
     - 1
     - 10, its initial value
     - 11
   * - 1
     - 2
     - 11, fed back from step 0
     - 13
   * - 2
     - 3
     - 13, fed back from step 1
     - 16

Element ``k`` of an output is computed at step ``k``: it is the state of step
``k + 1``. So the target of a trajectory of states is the measured state
**one step ahead**, as in the examples below.

Step 2: a Loop over a built model
---------------------------------

A :class:`~nnodely.Loop` takes a model, the *body*, built here if it is not
yet. It rolls the body out along the outermost sequence axis of the streams
that drive it. It is configured with the body and its recurrence, then
called on the streams that drive it, ``Loop(f, feedback=..., init=...)(inputs)``:

``feedback``
    Maps a body input to the body output that feeds it at the next step.
    The body may feed some Inputs back itself, with
    :meth:`Input.feedback() <nnodely.Input.feedback>` or the state of an
    :class:`~nnodely.Integrate`, a :class:`~nnodely.Derivative` or a
    :class:`~nnodely.Buffer`. Those are found in its graph and need no entry.
``init``
    Maps a state to the stream that gives its initial value. Step 0 reads
    the first step of that stream's sequence. A state left out starts at
    zero.
``inputs``, the call
    Maps the other body inputs to streams with one value per step, such as a
    measured force. A body input left out keeps a constant value. A state is
    not among them: its initial value goes to ``init``.
``collect``
    ``True``, the default, returns the whole trajectory, with the rollout as
    its new outermost sequence axis. ``False`` returns only the last step.
``seq``
    The number of steps. It is needed only when no stream of ``init`` or of
    the call carries a sequence.

The body below is a linear state-space step,
:math:`s[t+1] = A s[t] + B F[t]`, on the state :math:`s = (x, \dot x)`:

.. code-block:: python

   horizon = 20
   state = Input("state", dim=2, tw=1)
   F_step = Input("F_step", tw=1)
   next_state = Output(
       "next_state",
       Linear(features=2, bias=False)(state) + Linear(features=2, bias=False)(F_step),
   )
   body = Modely("step", inputs=[state, F_step], outputs=[next_state]).build()

   s0 = Input("s0", dim=2, seq=horizon, tw=1)
   F_seq = Input("F_seq", seq=horizon, tw=1)
   trajectory = Output(
       "trajectory",
       Loop(f=body, feedback={state: next_state}, init={state: s0})({F_step: F_seq}),
   )

   simulator = Modely("msd_loop", inputs=[s0, F_seq], outputs=[trajectory])
   s_next = Input("s_next", dim=2, seq=horizon, tw=1)
   simulator.minimize("trajectory", trajectory, s_next)
   simulator.build()
   print(trajectory.shape)   # 20 steps of 1 sample of 2 states

.. code-block:: text
   :class: output

   Shape(S=(20,), T=1, D=(2,))

The :class:`~nnodely.DataLoader` cuts the signals into sequences of
``horizon`` steps. The target is the state one step ahead, as Step 1 shows:

.. code-block:: python

   states = np.stack([position, velocity], axis=-1)
   data = {"s0": states[:-1], "F_seq": force[:-1], "s_next": states[1:]}
   train_data = DataLoader(simulator, source=data, step=5)
   history = simulator.train(train_data, epochs=10, batch_size=32, lr=1e-2, printer=None)

Step 3: what a Loop returns
---------------------------

A Loop returns every output of the body, in its order:

- with one output, the Loop is that output, a stream;
- with several, unpack them, ``position, velocity = Loop(...)(inputs)``, or
  pick one by position, ``loop[1]``, or by its name in the body,
  ``loop["velocity"]``. Read as one stream, in an ``Output``, an operator or
  a layer, a Loop of several outputs raises an error.

A Loop does not return the outputs it adds for the hidden states of the
body, such as the state of an ``Integrate``.

A state with a window, ``Input(..., tw=n)``, is fed back by shifting: the
newest sample is the value fed back, and after ``n`` steps the window holds
only predictions. So a fed-back output is one sample, and the window of a
state cannot read future samples, ``tw=(p, f)`` with ``f > 0``. A state
without a window is fed back by an output without one.

With a dynamic sequence, ``Input(..., seq=-1, tw=1)``, the rollout follows
the length of the data, and the Loop needs no ``seq``. With
``seq_length="full"`` in the :class:`~nnodely.DataLoader`, this trains on
whole simulations of different lengths (see :doc:`data`).

A body that feeds its states back itself needs no ``feedback``. Here the
state of an :class:`~nnodely.Integrate` is hidden in the body, and starts
from the ``init`` of the Integrate, an Input of the body. The Loop binds that
Input to a stream of the outer graph, as it binds the force:

.. code-block:: python

   from nnodely import Integrate

   F_k, v0 = Input("F_k", tw=1), Input("v0", tw=1)
   v_next = Integrate(dt=dt, init=v0)(F_k / m)
   impulse = Modely("impulse", inputs=[F_k, v0], outputs=[Output("v_next", v_next)])

   F_dyn, v_start = Input("F_dyn", seq=-1, tw=1), Input("v_start", tw=1)
   velocities = Loop(f=impulse)({F_k: F_dyn, v0: v_start})

Like a model, a Loop can be called with a list, one stream per body input in
the order the body declares them, ``Loop(f=impulse)([F_dyn, v_start])``, or
with the stream alone when the body has one input.

Step 4: a model declared one step at a time
-------------------------------------------

A :class:`~nnodely.ModelyLoop` is declared one step at a time, as the
equations of a discrete-time system are written. A state is an Input fed
back with its next value by :meth:`Input.feedback() <nnodely.Input.feedback>`,
which returns the stream it is given. Within a step:

- the Input holds the value at the end of the previous step;
- the stream it is fed back with holds the value at the end of this step.

The Inputs of a step read each other's previous values. So coupled dynamics
are written as they are, one Input per state variable, and the model feeds
every one back in a single rollout. Here is the mass-spring-damper of the
data, with :math:`k` and :math:`c` to identify:

.. code-block:: python

   from nnodely import IntegrateStep, Parameter

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
is computed at step ``k``. The horizon is as long as the data, or ``seq``
steps when it is given.

A state is one of two kinds:

- **declared** among the inputs, as ``pos`` and ``vel`` here: an input of
  the model. It starts from its first value in the data, and reads zero when
  the data leaves it out.
- **hidden**, ``x.feedback(stream, init=x0, hidden=True)``: the model's own
  memory. It is none of its inputs and cannot be declared, and the data does
  not seed it: it starts from its ``init`` - a number, a Parameter, which
  makes the initial state trainable, a Constant, or a declared Input whose
  first step is read - and from zero without one.

A fed-back Input that is neither is refused when the model is declared. The
state of an Integrate, a Derivative or a Buffer is hidden, and starts from
their ``init`` (and the rate a trapezoidal Integrate keeps, from its
``init_rate``).

The model a ModelyLoop returns is a :class:`~nnodely.Modely` of the rollout,
and the one-step model it rolls out is ``msd.models[0]``. Its
:meth:`~nnodely.Modely.minimize` takes the streams of the step, each standing
for its trajectory: an output, or an Input that nothing feeds back. A target
declared one step at a time, such as the measured position one step ahead,
reads one sample per step. A fed-back Input has no trajectory of its own:
minimize the output it is fed back with. The same holds for the body of any
Loop a model rolls out, and for a model loaded back with
:meth:`~nnodely.Modely.load`. :attr:`~nnodely.Modely.inputs` tells how each
Input of a model is fed:

.. code-block:: python

   print(msd.inputs)

The :class:`~nnodely.DataLoader` cuts the signals into sequences of
``seq_length`` steps, and the target is the position one step ahead:

.. code-block:: python

   data = {"F_in": force[:-1], "pos": position[:-1], "vel": velocity[:-1],
           "pos_meas": position[1:]}
   train_data = DataLoader(msd, source=data, seq_length=50, step=10)
   history = msd.train(train_data, epochs=30, batch_size=32, lr=5e-2, printer=None)
   print(k_hat.value_numpy, c_hat.value_numpy)   # about 2.0 and 0.5, the true k and c

Built as a :class:`~nnodely.Modely`, the same graph is the one-step model.
Nothing feeds the states back: a declared state is read from the data at
every step, as any Input, and a hidden one - the state of an ``Integrate``, a
``Derivative`` or a ``Buffer`` too - reads its ``init``, as at the first step
of a ModelyLoop.

.. code-block:: python

   one_step = Modely("msd_one_step", inputs=[F_in, pos, vel],
                     outputs=[Output("pos_pred", pos_next)]).build()

Step 5: blocks and windows inside a step
----------------------------------------

A model called as a block in a ModelyLoop keeps the feedback of its own
states, one state per call. A hidden state is the block's own: it starts from
its ``init`` in each call, and the call does not give it. When a block reads a
window of a signal that the
step computes, a :class:`~nnodely.Buffer` holds the past values:
``Buffer(tw=3)(u)`` is ``[u[k-2], u[k-1], u[k]]``. Its ``init`` holds the
values before the first step: one sample, repeated, or ``tw - 1`` samples,
oldest first.

Here a proportional controller drives a plant that reads its last three
commands:

.. code-block:: python

   from nnodely import Buffer

   forces = Input("forces", tw=3)
   level = Input("level", tw=1)
   level_next = level.feedback(0.9 * level + Fir(bias=False)(forces), hidden=True)
   plant = Modely("plant", inputs=[forces], outputs=[Output("level_next", level_next)])

   reference = Input("reference", tw=1)
   measured = Input("measured", tw=1)
   command = Output("command", (reference - measured) * 0.5)
   measured.feedback(plant([Buffer(tw=3)(command)]), init=0.0, hidden=True)
   control = ModelyLoop("control", inputs=[reference],
                        outputs=[command, Output("level", measured)]).build()
   result = control({"reference": np.ones((1, 20, 1, 1))})
   print(keras.ops.convert_to_numpy(result["level"]).shape)

.. code-block:: text
   :class: output

   (1, 20, 1, 1)

Both states are hidden: ``level`` is the plant's own, and ``measured`` the
controller's, starting from zero whatever the data holds. Declared among the
inputs of ``control`` instead, ``measured`` would start from its data.

A state belongs to the model whose graph reads it:

- a hidden state of a block that the calling graph reads as well would be
  two states, and is refused. To read the state of the block, make it an
  output of the block. To give the block the state of the caller, declare an
  Input for it in the block and pass the state in the call;
- a block called with a stream for a state it declares keeps its feedback,
  and reads that stream as its initial value.

A model is saved with its states, and so are the models it calls: loaded
back, they feed them back as before, and a Loop finds them again.

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Error
     - Cause
     - Fix
   * - ``Loop input 'x' is fed back: give its initial value with init=``
     - A state was passed among the inputs of the call.
     - Pass it in ``init={x: stream}``.
   * - ``Loop init key 'u' is not fed back``
     - An ``init`` for an Input that is not a state.
     - Pass it among the inputs of the call, ``Loop(...)({u: stream})``.
   * - ``Model 'm' reads the Inputs ['x'], which are not among its inputs``,
       naming ``x`` among the fed-back ones
     - A state the model neither declares nor hides.
     - Declare it among the inputs, to start it from the data, or feed it
       back with ``hidden=True``, to start it from its ``init``.
   * - ``Model 'm' declares the hidden states ['x'] among its inputs``
     - A hidden state is the model's own, and no input.
     - Leave it out of the inputs, or feed it back without ``hidden=True``.
   * - The trajectory is one step late, and the loss does not fall
     - The target is the measured state at the same step.
     - Use the state one step ahead, as Step 1 shows.
   * - A trajectory that diverges in training
     - A rollout too long for the initial weights.
     - Train on short sequences first, then on longer ones.

See also
--------

:class:`~nnodely.Loop`, :class:`~nnodely.ModelyLoop`,
:meth:`Input.feedback <nnodely.Input.feedback>`, :class:`~nnodely.Buffer`,
:class:`~nnodely.Integrate`, :class:`~nnodely.IntegrateStep`, :doc:`physics`.
