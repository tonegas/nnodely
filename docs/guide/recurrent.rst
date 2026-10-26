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
