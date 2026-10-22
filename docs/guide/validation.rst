Validation
==========

:meth:`~nnodely.Modely.validate` evaluates every objective of a built model on
a dataset, the way training evaluates it: the same forward pass (layers such as
``BatchNorm`` take their inference branch), the same target, aligned with the
prediction as training aligns it, and the same padding mask. Its loss is the
one :meth:`~nnodely.Modely.train` logs as ``val_<objective>_loss`` for the same
data. Next to it come the indicators a system-identification report is read
for, with :math:`e = \hat y - y`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Indicator
     - Meaning
   * - Loss
     - The loss the objective is trained with, ``seq_weights`` and masking
       included.
   * - RMSE, MAE, Max error
     - Root-mean-square, mean and peak of :math:`|e|`, in the signal's units.
   * - Bias, Error std
     - Mean and standard deviation of :math:`e`: the systematic and the random
       part of the error, :math:`\mathrm{RMSE}^2 = \mathrm{bias}^2 +
       \mathrm{std}^2`.
   * - NRMSE
     - RMSE as a percentage of the target's range.
   * - FIT
     - :math:`100 \, (1 - \|e\| / \|y - \bar y\|)`, the percentage fit of
       system identification: 100 % is a perfect prediction, 0 % no better
       than the target's mean.
   * - R2
     - :math:`1 - \sum e^2 / \sum (y - \bar y)^2`, the coefficient of
       determination.
   * - VAF
     - :math:`100 \, (1 - \operatorname{var}(e) / \operatorname{var}(y))`,
       the variance accounted for: R2 that forgives a constant offset.
   * - Correlation
     - Pearson's correlation of prediction and target; 0 for a prediction that
       never moves.

**Channels.** Each feature of the compared signal is a channel, and each
channel is scored on its own: a position in metres and a velocity in mm/s are
each measured against their own spread. For the whole signal, the absolute
indicators pool every channel and the relative ones (NRMSE, FIT, R2, VAF,
correlation) are the mean of the channels' values. A channel whose target never
moves has no spread to measure against: its relative indicators are NaN, and
left out of the mean.

**Points.** A point is one sample at one time step: every step of a window and
of a rollout is scored. A NaN or infinite error is counted in ``non_finite`` and
makes the indicators of its channel, and of the whole signal, NaN rather than
being hidden.

**Padded rollouts.** When simulations of different lengths are loaded with
``seq_length="full"``, the shorter ones are padded to the longest, and the
indicators leave the padded steps out: the model is scored on exactly the steps
each simulation has, as if each had been validated alone. The loss stays the
training one, in which a padded step adds zero but still counts in the mean.
For a signal with a rollout, ``step_rmse`` is the RMSE of each channel at each
step, over the simulations that reach it: how the error grows with the
prediction horizon.

The model below is trained on part of a signal and validated on the rest:

.. code-block:: python

   import numpy as np
   from nnodely import DataLoader, Fir, Input, Modely, Output

   t = np.arange(2000) * 0.01
   u_data = np.sin(t) + 0.5 * np.sin(3.1 * t)
   y_data = np.convolve(u_data, np.exp(-np.arange(20) / 5.0) / 5.0)[: len(t)]

   u = Input("u")
   y = Output("y", Fir(features=1)([u.sw(20)]))
   model = Modely("filter", inputs=[u], outputs=[y])
   model.minimize("response", y, Input("y_target").last())
   model.build()

   signals = {"u": u_data, "y_target": y_data}
   train_data = DataLoader(model, source={k: v[:1500] for k, v in signals.items()})
   val_data = DataLoader(model, source={k: v[1500:] for k, v in signals.items()})
   history = model.train(train_data, epochs=40, batch_size=32, lr=1e-2, printer=None)

   result = model.validate(val_data, out_dir="validation", history=history)

``validate`` prints a summary and returns a
:class:`~nnodely.core.validation.ValidationResult`. The result can be indexed by
objective name, and ``metrics()`` returns every number as a plain dictionary.
The keys are ``loss``, ``rmse``, ``mae``, ``max_error``, ``bias``,
``std_error``, ``nrmse_pct``, ``fit_pct``, ``r2``, ``vaf_pct``,
``correlation`` and ``non_finite`` (the count of NaN or infinite errors). Each
objective's :class:`~nnodely.core.validation.SignalScore` also holds the
indicators of every channel, and the points it scored:

.. code-block:: python

   score = result["response"]
   print(score.metrics["fit_pct"])          # the whole signal
   print(score.channel_metrics["fit_pct"])  # one value per channel
   print(score.points, score.padded)        # points scored, padded points left out
   print(score.step_rmse)                   # (steps, channels) for a rollout, else None
   metrics = result.metrics()               # {"response": {"loss": ..., "rmse": ..., ...}}
   print(result.figures)                    # the PNG files written to out_dir

The summary gives each channel a column when a signal has up to three.

Plots
-----

With ``out_dir``, the figures are saved as PNG files named after the model.
There is one figure per objective, with the prediction against the target and
the error over the scored points, a parity plot and the error distribution,
plus the RMSE along the rollout for a signal that has one. A second
figure shows the loss curves when ``history`` is given (pass the dictionary
returned by :meth:`~nnodely.Modely.train`). With ``show=True``, the figures
open in an interactive Matplotlib window.
