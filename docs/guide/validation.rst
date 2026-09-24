Validation
==========

:meth:`~nnodely.Modely.validate` evaluates every objective of a built model on
a dataset. It reports the training loss together with the indicators a
system-identification report is read for:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Indicator
     - Meaning
   * - Loss
     - The loss the objective was trained with.
   * - RMSE, MAE, Max error
     - Root-mean-square, mean absolute and peak error.
   * - Bias, Error std
     - Mean and standard deviation of the error.
   * - NRMSE
     - RMSE as a percentage of the target's range.
   * - FIT
     - :math:`100 \, (1 - \|y - \hat y\| / \|y - \bar y\|)`, the percentage of
       the target's variation the model explains.
   * - R2, Correlation
     - Coefficient of determination and Pearson correlation.

The model below is trained on part of a signal and validated on the rest:

.. code-block:: python

   import numpy as np
   from nnodely import DataLoader, Fir, Input, Modely, Output

   t = np.arange(2000) * 0.01
   u_data = np.sin(t) + 0.5 * np.sin(3.1 * t)
   y_data = np.convolve(u_data, np.exp(-np.arange(20) / 5.0) / 5.0)[: len(t)]

   u = Input("u")
   y = Output("y", Fir(out_features=1)([u.sw(20)]))
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
``std_error``, ``nrmse_pct``, ``fit_pct``, ``r2``, ``correlation`` and
``non_finite`` (the count of NaN or infinite errors):

.. code-block:: python

   print(result["response"].metrics["fit_pct"])
   metrics = result.metrics()        # {"response": {"loss": ..., "rmse": ..., ...}}
   print(result.figures)             # the PNG files written to out_dir

Plots
-----

With ``out_dir``, the figures are saved as PNG files named after the model.
There is one figure per objective, with the prediction against the target, the
error over the samples, a parity plot and the error distribution. A second
figure shows the loss curves when ``history`` is given (pass the dictionary
returned by :meth:`~nnodely.Modely.train`). With ``show=True``, the figures
open in an interactive Matplotlib window.
