Validating a Model
==================

At the end of this page you will be able to score a trained model on a
dataset with the indicators of system identification, read the scores in
code, and save the plots of what the model did.

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
   history = model.train(train_data, epochs=40, batch_size=32, lr=1e-2, printer=None)

Step 2: validate
----------------

:meth:`~nnodely.Modely.validate` evaluates every objective of a built model
on a dataset, as training evaluates it:

- the same forward pass, with layers such as ``BatchNorm`` in their
  inference mode;
- the same target, aligned with the prediction as in training;
- the same mask of padded steps.

Its loss is the one :meth:`~nnodely.Modely.train` logs as
``val_<objective>_loss`` for the same data. It prints a summary, and saves
the figures to ``out_dir``:

.. code-block:: python

   result = model.validate(val_data, out_dir="validation", history=history)

Next to the loss come the indicators a system-identification report is read
for, with the error :math:`e = \hat y - y`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Indicator
     - Meaning
   * - Loss
     - The loss the objective is trained with, masking included.
   * - RMSE, MAE, Max error
     - Root mean square, mean and peak of :math:`\lvert e \rvert`, in the
       units of the signal.
   * - Bias, Error std
     - Mean and standard deviation of :math:`e`: the systematic and the
       random part of the error, :math:`\mathrm{RMSE}^2 = \mathrm{bias}^2 +
       \mathrm{std}^2`.
   * - NRMSE
     - RMSE as a percentage of the range of the target.
   * - FIT
     - :math:`100 \, (1 - \|e\| / \|y - \bar y\|)`, the percentage fit of
       system identification. 100 % is a perfect prediction, 0 % no better
       than the mean of the target.
   * - R2
     - :math:`1 - \sum e^2 / \sum (y - \bar y)^2`, the coefficient of
       determination.
   * - VAF
     - :math:`100 \, (1 - \operatorname{var}(e) / \operatorname{var}(y))`,
       the variance accounted for: R2 without the penalty of a constant
       offset.
   * - Correlation
     - The Pearson correlation of prediction and target; 0 for a prediction
       that never moves.

Step 3: read the scores
-----------------------

``validate`` returns a :class:`~nnodely.core.validation.ValidationResult`:
a list of :class:`~nnodely.core.validation.SignalScore`, one per objective,
picked by the name of the objective or by its position. Each score's
``metrics`` is a dictionary of numbers, with the keys ``loss``, ``rmse``,
``mae``, ``max_error``, ``bias``, ``std_error``, ``nrmse_pct``,
``fit_pct``, ``r2``, ``vaf_pct``, ``correlation`` and ``non_finite``, the
count of NaN or infinite errors:

.. code-block:: python

   score = result["response"]
   print(sorted(score.metrics))
   print(score.points, score.padded)        # points scored, padded points left out

.. code-block:: text
   :class: output

   ['bias', 'correlation', 'fit_pct', 'loss', 'mae', 'max_error', 'non_finite', 'nrmse_pct', 'r2', 'rmse', 'std_error', 'vaf_pct']
   481 0

Each score also holds the indicators of every channel, and, for a rollout,
the error at every step:

.. code-block:: python

   fit = score.metrics["fit_pct"]                # the whole signal
   per_channel = score.channel_metrics["fit_pct"]  # one value per channel
   print(score.step_rmse)                        # (steps, channels) for a rollout
   metrics = {s.name: s.metrics for s in result}  # every objective

.. code-block:: text
   :class: output

   None

**Channels.** Each feature of the compared signal is a channel, scored on its
own scale: a position in metres and a velocity in mm/s are each measured
against their own spread. For the whole signal, the absolute indicators pool
every channel, and the relative ones (NRMSE, FIT, R2, VAF, correlation) are
the mean over the channels. A channel whose target never moves has no spread
to measure against: its relative indicators are NaN, and left out of the
mean. The summary gives each channel a column when a signal has up to three.

**Points.** A point is one sample at one time step: every step of a window and
of a rollout is scored. A NaN or infinite error is counted in ``non_finite``,
and makes the indicators of its channel, and of the whole signal, NaN rather
than being hidden.

**Padded rollouts.** Simulations of different lengths loaded with
``seq_length="full"`` are padded to the longest. The indicators leave the
padded steps out, so each simulation is scored on exactly its own steps. The
loss stays the training one, in which a padded step adds zero but still
counts in the mean. For a signal with a rollout, ``step_rmse`` is the RMSE of
each channel at each step, over the simulations that reach it: how the error
grows with the prediction horizon.

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
of the error, and, for a rollout, the RMSE along it. A second figure shows
the loss curves when ``history``, the dictionary returned by
:meth:`~nnodely.Modely.train`, is given. ``show=True`` opens the figures in
an interactive Matplotlib window. ``verbose=False`` validates without
printing the summary.

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
     - Read the absolute indicators (RMSE, MAE) for it.
   * - Every indicator is NaN
     - The prediction has NaN or infinite values: see ``non_finite``.
     - Check the model and the data for NaN.
   * - The loss differs from the RMSE squared
     - A rollout padded to the longest simulation: padded steps count in
       the loss, not in the indicators.
     - Read the indicators, which leave the padding out.

See also
--------

:meth:`Modely.validate <nnodely.Modely.validate>`,
:class:`~nnodely.core.validation.ValidationResult`,
:class:`~nnodely.core.validation.SignalScore`, :doc:`data`.
