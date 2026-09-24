Recurrent Models
================

A one-step model predicts the next sample from measured ones. Used as a
simulator, it has to run on its own predictions instead. *nnodely* offers three
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
   x = Input("x")
   F = Input("F")
   x_pred = Output("x_pred", Fir(out_features=1)([x.sw(5)]) + Fir(out_features=1)([F.last()]))

   model = Modely("msd_rollback", inputs=[x, F], outputs=[x_pred])
   model.rollback({x: x_pred}, steps=steps)
   model.minimize("ahead", x_pred, Input("x_ahead").last())
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

Loop
----

A :class:`~nnodely.Loop` takes a **built** model, the *body*, and rolls it out
along a sequence axis:

``callback``
    maps a body input to the body output that feeds it at the next step.
``initial``
    maps a fed-back input to the stream that seeds it. Step 0 reads the first
    element of that stream's sequence.
``inputs``
    maps the other body inputs to streams that carry one value per step
    (exogenous signals). Body inputs left out keep a constant value.
``collect``
    ``True`` (the default) returns the whole trajectory, with the rollout
    appended as the last sequence axis. ``False`` returns only the last step.
``length``
    fixes the number of steps when no input declares it.

The body below is a linear state-space step, :math:`s[t+1] = A s[t] + B F[t]`,
on the state :math:`s = (x, \dot x)`:

.. code-block:: python

   horizon = 20
   state = Input("state", dim=2)
   F_step = Input("F_step")
   next_state = Output(
       "next_state",
       Linear(out_features=2, use_bias=False)([state.last()])
       + Linear(out_features=2, use_bias=False)([F_step.last()]),
   )
   body = Modely("step", inputs=[state, F_step], outputs=[next_state]).build()

   s0 = Input("s0", dim=2, seq=horizon)
   F_seq = Input("F_seq", seq=horizon)
   trajectory = Output(
       "trajectory",
       Loop(f=body, callback={state: next_state}, initial={state: s0}, inputs={F_step: F_seq}),
   )

   simulator = Modely("msd_loop", inputs=[s0, F_seq], outputs=[trajectory])
   simulator.minimize("trajectory", trajectory, Input("s_next", dim=2, seq=horizon))
   simulator.build()
   print(trajectory.shape)   # (2, 1, 20): 2 states, 1 sample, 20 steps

The :class:`~nnodely.DataLoader` cuts the signals into sequences of
``horizon`` steps. The target sequence is the state one step ahead:

.. code-block:: python

   states = np.stack([position, velocity], axis=-1)
   data = {"s0": states[:-1], "F_seq": force[:-1], "s_next": states[1:]}
   train_data = DataLoader(simulator, source=data, step=5)
   simulator.train(train_data, epochs=10, batch_size=32, lr=1e-2, printer=None)

A body with several outputs returns one stream per output, which can be
unpacked: ``position_traj, velocity_traj = Loop(...)``. A fed-back input with
a time window, ``x.sw(n)``, is closed by shifting: after ``n`` steps the window
contains only predictions.

With a dynamic sequence, ``seq=(None,)``, the rollout follows the length of the
data it receives. A ``length`` still has to be given for building. Together
with ``seq_length="full"`` in the :class:`~nnodely.DataLoader`, this trains on
whole simulations of different lengths (see :doc:`data`).

Roll
----

:class:`~nnodely.Roll` unrolls a built model over the window of one of its
inputs, and returns a window of the same length. With the default ``steps``
(the window length) every sample of the result is a prediction:

.. code-block:: python

   x = Input("x_roll")
   out = Output("out", Fir(out_features=1)([x.sw(5)]))
   one_step = Modely("one_step", inputs=[x], outputs=[out]).build()

   rolled = Roll(f=one_step, callback={x: out})
   free_run = Modely("free_run", inputs=[x], outputs=[Output("free", rolled)]).build()
   print(free_run({"x_roll": [1.0, 2.0, 3.0, 4.0, 5.0]})["free"].shape)   # (1, 1, 5)
