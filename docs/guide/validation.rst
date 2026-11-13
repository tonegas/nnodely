Validating a Model
==================

At the end of this page you will be able to score a trained model on a
dataset with the metrics of system identification, read them in code, and
save the plots of what the model did.

**Prerequisites:** :doc:`training`.

**Tutorial:** :doc:`../tutorial/04_model_validation`.

**The example.** A filter that smooths an input signal: the output is the
input convolved with a decaying exponential. The model is a :class:`~nnodely.Fir`
filter of 20 samples, trained on part of the signal and validated on the rest.

Step 1: train the model
-----------------------

.. code-block:: python

   import numpy as np
   from nnodely import DataLoader, Fir, Input, Modely, Output, set_seed

   set_seed(0)
   t = np.arange(2000) * 0.01
   u_data = np.sin(t) + 0.5 * np.sin(3.1 * t)
   y_data = np.convolve(u_data, np.exp(-np.arange(20) / 5.0) / 5.0)[: len(t)]

   u = Input("u", tw=20)
   y = Output("y", Fir()(u))
   model = Modely("filter", inputs=[u], outputs=[y])
   model.minimize("response", y, Input("y_target", tw=1))
   model.build()

   signals = {"u": u_data, "y_target": y_data}
   train_data = DataLoader(model, source={k: v[:1500] for k, v in signals.items()})
   val_data = DataLoader(model, source={k: v[1500:] for k, v in signals.items()})
   trained = model.train(train_data, epochs=40, batch_size=32, lr=1e-2, printer=None)

Step 2: validate
----------------

:meth:`~nnodely.Modely.validate` evaluates every objective of a built model
on a dataset, as training evaluates it:

- the same forward pass, with layers such as ``BatchNorm`` in their
  inference mode;
- the same target, aligned with the prediction as in training;
- the same mask of padded steps.

Its loss is the one :meth:`~nnodely.Modely.train` logs as
``val_loss_<objective>`` for the same data. ``out_dir`` saves the figures:

.. code-block:: python

   result = model.validate(val_data, out_dir="validation", history=trained)

The result is read by its attributes, as the result of ``train``. ``print``
shows a row per objective with its loss, and the total:

.. code-block:: python

   print(result.losses.keys())
   print(result.signals.keys(), result.samples)

.. code-block:: text
   :class: output

   ['loss', 'loss_response']
   ['response'] 481

Step 3: read the signals
------------------------

``result.signals`` holds a :class:`~nnodely.core.validation.Signal` per objective: the two
signals it compared, whole, laid out as the stream,
``(samples, *seq, tw, *dim)``, and the metrics a system-identification report
is read for, with the error :math:`e = \hat y - y`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Meaning
   * - ``rmse``, ``mae``, ``max_error``
     - Root mean square, mean and peak of :math:`\lvert e \rvert`, in the
       units of the signal.
   * - ``bias``, ``std_error``
     - Mean and standard deviation of :math:`e`: the systematic and the
       random part of the error, :math:`\mathrm{RMSE}^2 = \mathrm{bias}^2 +
       \mathrm{std}^2`.
   * - ``nrmse_pct``
     - RMSE as a percentage of the range of the target.
   * - ``fit_pct``
     - :math:`100 \, (1 - \|e\| / \|y - \bar y\|)`, the percentage fit of
       system identification. 100 % is a perfect prediction, 0 % no better
       than the mean of the target.
   * - ``r2``
     - :math:`1 - \sum e^2 / \sum (y - \bar y)^2`, the coefficient of
       determination.
   * - ``vaf_pct``
     - :math:`100 \, (1 - \operatorname{var}(e) / \operatorname{var}(y))`,
       the variance accounted for: R2 without the penalty of a constant
       offset.
   * - ``correlation``
     - The Pearson correlation of prediction and target; 0 for a prediction
       that never moves.
   * - ``non_finite``
     - The values of the prediction that are NaN or infinite.

The metrics are functions. Without arguments they give a number, over the
whole signal; ``seq=``, ``tw=`` and ``dim=`` keep those axes, and give a
value per position along them:

.. code-block:: python

   signal = result.signals["response"]
   print(signal)
   print(signal.shape, signal.scored, signal.missing)
   print(signal.rmse(tw=True).shape)   # a value per step of the window

.. code-block:: text
   :class: output

   ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━...┓
   ┃ metric      ┃ meaning                              ┃ response ...┃
   ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━...┩
   │ rmse        │ root mean square error               │ ...│
   ...
   │ non_finite  │ values of source not finite          │ 0 ...│
   └─────────────┴──────────────────────────────────────┴──...┘
   Shape(S=(), T=1, D=(1,)) 481 0
   (1,)

**Channels.** Each feature of the compared signal is a channel, scored on its
own scale: a position in metres and a velocity in mm/s are each measured
against their own spread. Over the whole signal, the absolute metrics pool
every channel, and the relative ones (``nrmse_pct``, ``fit_pct``, ``r2``,
``vaf_pct``, ``correlation``) are the mean over the channels; ``dim=True``
gives the value of each channel. A channel whose target never moves has no
spread to measure against: its relative metrics are NaN, and left out of the
mean.

**Points.** A point is one sample at one step of the sequence and of the
window: every step is scored. A NaN or infinite prediction is counted by
``non_finite()``, and makes the metrics it is in NaN rather than being
hidden.

**Padded rollouts.** Simulations of different lengths loaded with
``seq_length="full"`` are padded to the longest. A padded step has no target:
it is NaN in both signals, counted in ``missing``, and left out of every
metric, so each simulation is scored on exactly its own steps. The loss stays
the training one, in which a padded step adds zero but still counts in the
mean. ``rmse(seq=True)`` is the error at each step, over the simulations that
reach it: how the error grows with the prediction horizon.

Step 4: plots
-------------

With ``out_dir``, the figures are saved as PNG files named after the model
and, for an objective, after its index and its name:

.. code-block:: python

   print(sorted(str(path) for path in result.figures))

.. code-block:: text
   :class: output

   [...filter_0_response.png...]

There is one figure per objective. It shows the prediction against the
target and the error over the scored points, a parity plot, the distribution
of the error, and, for a signal with a sequence, the RMSE at each step. A
second figure shows the loss curves when ``history``, the result of
:meth:`~nnodely.Modely.train`, is given. ``show=True`` opens the figures in
an interactive Matplotlib window. ``result.plot()`` draws them again, at any
time.

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Problem
     - Cause
     - Fix
   * - FIT, R2 or VAF are NaN for a channel
     - The target of that channel never moves.
     - Read the absolute metrics (``rmse``, ``mae``) for it.
   * - Every metric is NaN
     - The prediction has NaN or infinite values: see ``non_finite()``.
     - Check the model and the data for NaN.
   * - The loss differs from the RMSE squared
     - A rollout padded to the longest simulation: padded steps count in
       the loss, not in the metrics.
     - Read the metrics, which leave the padding out.

See also
--------

:meth:`Modely.validate <nnodely.Modely.validate>`,
:class:`~nnodely.core.results.ValidationResult`,
:class:`~nnodely.core.validation.Signal`, :doc:`data`.
