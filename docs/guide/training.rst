Training
========

The examples on this page use a small first-order system, :math:`y[t+1] =
0.9 \, y[t] + 0.5 \, u[t]`, split into a training and a validation set:

.. code-block:: python

   import numpy as np
   import keras
   from nnodely import DataLoader, Fir, Input, Modely, Output, set_seed

   set_seed(42)

   rng = np.random.default_rng(0)
   u_data = rng.uniform(-1.0, 1.0, 1000)
   y_data = np.zeros(1000)
   for t in range(999):
       y_data[t + 1] = 0.9 * y_data[t] + 0.5 * u_data[t]

   y = Input("y")
   u = Input("u")
   y_next = Output("y_next", Fir(out_features=1)([y.last()]) + Fir(out_features=1)([u.last()]))
   model = Modely("first_order", inputs=[y, u], outputs=[y_next])

Objectives
----------

:meth:`~nnodely.Modely.minimize` registers a loss between a *source* and a
*target*. Every objective has a name, which labels its loss during training
and validation. Objectives are declared before :meth:`~nnodely.Modely.build`.

.. code-block:: python

   model.minimize("one_step", y_next, y.next(), loss="mse")
   model.build()

The **target** is one of:

- a window of an input, read from the data: here ``y.next()``, the sample after
  the current one. An input may be read by the model and be a target at once, or
  be declared only as a target, such as ``Input("y_measured").last()``;
- a number, which is compared with every sample of the source;
- nothing: ``minimize(name, residual)`` drives ``residual`` to zero, which is
  how physical constraints are imposed (see :doc:`physics`).

The **loss** is anything Keras accepts: a name (``"mse"``, ``"mae"``,
``"huber"``, ...), a ``keras.losses.Loss`` instance, or a callable
``loss(y_true, y_pred)``. :meth:`~nnodely.Modely.remove_minimizer` removes an
objective by name.

Training
--------

.. code-block:: python

   signals = {"y": y_data, "u": u_data}
   train_data = DataLoader(model, source={k: v[:800] for k, v in signals.items()})
   val_data = DataLoader(model, source={k: v[800:] for k, v in signals.items()})

   history = model.train(
       train_data,
       val_data=val_data,
       epochs=50,
       batch_size=32,
       lr=1e-2,
   )
   print(sorted(history))

:meth:`~nnodely.Modely.train` returns the Keras history: a dictionary of
per-epoch losses, with the validation ones under ``val_`` keys when
``val_data`` is given. The main arguments are:

``epochs``, ``batch_size``, ``shuffle``
    as in Keras.
``optimizer``
    a Keras optimizer name (``"adam"``, the default, ``"sgd"``, ``"adamw"``,
    ``"rmsprop"``, ...), a serialized configuration, or an optimizer instance.
``lr``, ``optimizer_kwargs``
    the learning rate and extra arguments used to create a named optimizer.
``printer``
    how progress is shown: ``"legacy"`` (the default, a per-objective loss
    table), ``"tiny"`` (one compact block), ``"nnodely"`` (an animated
    console), ``None`` (nothing), or any ``keras.callbacks.Callback``.

.. code-block:: python

   history = model.train(
       train_data,
       epochs=10,
       optimizer="adamw",
       lr=1e-3,
       optimizer_kwargs={"weight_decay": 1e-4, "global_clipnorm": 1.0},
       printer="tiny",
   )

   history = model.train(
       train_data,
       epochs=10,
       optimizer=keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9),
       printer=None,
   )

Training can be called several times: every call continues from the current
weights.

Reproducibility
---------------

:func:`~nnodely.set_seed` seeds Python, NumPy and the active backend together.
Call it before creating layers, so the initial weights are reproducible too.
The ``NNODELY_SEED`` environment variable has the same effect and is set by
``set_seed`` so that child processes inherit it. :func:`~nnodely.get_seed`
returns the current seed.
