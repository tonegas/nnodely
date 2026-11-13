Training a Model
================

At the end of this page you will be able to declare what a model minimizes,
train it with any Keras optimizer, stop the training early, freeze some of
its weights, and make the results reproducible.

**Prerequisites:** :doc:`models`, :doc:`data`.

**Tutorial:** :doc:`../tutorial/03_model_training`.

**The example.** A first-order system,
:math:`y[t+1] = 0.9 \, y[t] + 0.5 \, u[t]`. The model learns the two
coefficients from data, split into a training and a validation set.

Step 1: the data and the model
------------------------------

.. code-block:: python

   import numpy as np
   import keras
   from nnodely import DataLoader, Fir, Input, Modely, Output, TimeSelect, set_seed

   set_seed(42)

   rng = np.random.default_rng(0)
   u_data = rng.uniform(-1.0, 1.0, 1000)
   y_data = np.zeros(1000)
   for t in range(999):
       y_data[t + 1] = 0.9 * y_data[t] + 0.5 * u_data[t]

   y = Input("y", tw=(1, 1))   # the current sample and the next one
   u = Input("u", tw=1)
   y_now = TimeSelect(0)(y)
   y_next = Output(
       "y_next",
       Fir(name="fir_y")(y_now) + Fir(name="fir_u")(u),
   )
   model = Modely("first_order", inputs=[y, u], outputs=[y_next])

Step 2: objectives
------------------

:meth:`~nnodely.Modely.minimize` registers an **objective**: a loss between a
*source* and a *target*, which training reduces. The method names call it a
*minimizer*: :attr:`~nnodely.Modely.minimizers` lists them, and
:meth:`~nnodely.Modely.remove_minimizer` removes one. Every objective has a
name, which labels its loss during training and validation. Declare the
objectives before :meth:`~nnodely.Modely.build`:

.. code-block:: python

   model.minimize("one_step", y_next, TimeSelect(-1)(y), loss="mse")
   model.build()

The **target** is one of:

- an Input, or a part of its window, read from the data. Here it is
  ``TimeSelect(-1)(y)``, the sample after the current one. An Input may be
  read by the model and be a target at once, or be only a target, such as
  ``Input("y_measured", tw=1)``;
- a number, compared with every sample of the source;
- nothing: ``minimize(name, residual)`` drives ``residual`` to zero. This is
  how a physical law is imposed (see :doc:`physics`).

The **loss** is anything Keras accepts:

- a name, such as ``"mse"``, ``"mae"`` or ``"huber"``;
- a ``keras.losses.Loss`` instance;
- a function ``loss(y_true, y_pred)``.

``gain`` weighs an objective in the total loss: ``gain=0.5`` gives it half
the importance. The loss it logs stays unweighted.

Step 3: training
----------------

:meth:`~nnodely.Modely.train` trains the model on a
:class:`~nnodely.DataLoader`. With ``val_data``, it also computes the losses
on a validation set at the end of every epoch:

.. code-block:: python

   signals = {"y": y_data, "u": u_data}
   train_data = DataLoader(model, source={k: v[:800] for k, v in signals.items()})
   val_data = DataLoader(model, source={k: v[800:] for k, v in signals.items()})

   trained = model.train(
       train_data,
       val_data=val_data,
       epochs=50,
       batch_size=32,
       lr=1e-2,
       printer=None,
   )
   print(trained.losses.keys())
   print(trained.epochs, trained.losses["val_loss_one_step"].shape)

.. code-block:: text
   :class: output

   ['loss', 'loss_one_step', 'val_loss', 'val_loss_one_step']
   50 (50,)

``train`` returns a :class:`~nnodely.core.results.TrainResult`, read by its
attributes. Its ``losses`` hold a value per epoch under the keys Keras logs:
``loss`` is the total, and ``loss_<objective>`` the loss of each objective,
each with a ``val_`` version when ``val_data`` is given. ``print(trained)``
shows a row per objective with its last losses. The model does not keep the
result: save it in a variable. The two learned coefficients are now close to
0.9 and 0.5.

The arguments, with their defaults:

``epochs=10``, ``batch_size=1``, ``shuffle=True``
    As in Keras. The default batch of one sample is slow: pass ``epochs``
    and ``batch_size`` for your data.
``optimizer=None``
    A Keras optimizer name, a serialized configuration, or an optimizer
    instance. ``None`` is ``"adam"``. The names are ``"adam"``, ``"adamw"``,
    ``"sgd"``, ``"rmsprop"``, ``"adagrad"``, ``"adadelta"``, ``"adamax"``,
    ``"adafactor"``, ``"nadam"``, ``"ftrl"`` and ``"lion"``.
``lr=1e-3``, ``optimizer_kwargs=None``
    The learning rate and the other arguments of an optimizer given by name.
    An optimizer instance keeps its own.
``early_stopping=None``, ``early_stopping_kwargs=None``
    When to stop before ``epochs`` (see Step 5).
``printer="legacy"``
    How the progress is shown:

    - ``"legacy"``: a table of the losses of every objective;
    - ``"tiny"``: one compact block;
    - ``"nnodely"``: an animated console with a chart of the total loss;
    - ``None``: nothing;
    - any ``keras.callbacks.Callback``.

Every call continues from the current weights, so a model can be trained in
several rounds.

Step 4: optimizers
------------------

.. code-block:: python

   trained = model.train(
       train_data,
       epochs=10,
       batch_size=32,
       optimizer="adamw",
       lr=1e-3,
       optimizer_kwargs={"weight_decay": 1e-4, "global_clipnorm": 1.0},
       printer="tiny",
   )

   trained = model.train(
       train_data,
       epochs=10,
       batch_size=32,
       optimizer=keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9),
       printer=None,
   )

Step 5: early stopping
----------------------

``early_stopping`` ends the training before ``epochs`` once a loss stops
improving. It takes one of:

- the **name** of the loss to watch, made into a Keras `EarlyStopping
  <https://keras.io/api/callbacks/early_stopping/>`_ with
  ``early_stopping_kwargs`` as its other arguments;
- a **dict** of the arguments of ``keras.callbacks.EarlyStopping``;
- a **callback**: a configured ``keras.callbacks.EarlyStopping``, or one of
  your own that sets ``self.model.stop_training = True``.

The names are the keys of ``losses``: ``"loss"``, ``"loss_<objective>"``,
and their ``val_`` versions when ``val_data`` is given. Watching a loss that is
not logged raises an error, rather than training to the end.

.. code-block:: python

   trained = model.train(
       train_data,
       val_data=val_data,
       epochs=500,
       batch_size=32,
       lr=1e-2,
       early_stopping="val_loss",
       early_stopping_kwargs={"patience": 10, "restore_best_weights": True},
       printer=None,
   )
   print(trained.epochs < 500)   # it stopped early

.. code-block:: text
   :class: output

   True

.. code-block:: python

   class StopBelow(keras.callbacks.Callback):
       """Stop once the objective's loss falls below a level."""

       def on_epoch_end(self, epoch, logs=None):
           if logs["loss_one_step"] < 1e-4:
               self.model.stop_training = True


   trained = model.train(
       train_data, epochs=500, batch_size=32, early_stopping=StopBelow(), printer=None
   )

Step 6: freezing weights
------------------------

The weights of ``Fir``, ``Linear``, ``PReLU`` and ``BatchNorm`` are
:class:`~nnodely.Parameter` objects, like any other Parameter.
:attr:`~nnodely.Modely.parameters` lists every Parameter the model computes
with, in a table:

.. code-block:: python

   model.freeze("fir_y_kernel")
   print(model.parameters)

.. code-block:: text
   :class: output

   ┏━━━━━┳━━━━━━━━━━━━━━┳━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
   ┃ idx ┃ parameter    ┃ seq ┃ tw ┃ dim    ┃ values ┃ trainable ┃
   ┡━━━━━╇━━━━━━━━━━━━━━╇━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
   │ 0   │ fir_y_kernel │ ()  │ -  │ (1, 1) │ 1      │ no        │
   │ 1   │ fir_y_bias   │ ()  │ -  │ (1,)   │ 1      │ yes       │
   │ 2   │ fir_u_kernel │ ()  │ -  │ (1, 1) │ 1      │ yes       │
   │ 3   │ fir_u_bias   │ ()  │ -  │ (1,)   │ 1      │ yes       │
   ├─────┼──────────────┼─────┼────┼────────┼────────┼───────────┤
   │     │ total        │     │    │        │ 4      │ 3         │
   └─────┴──────────────┴─────┴────┴────────┴────────┴───────────┘

A frozen Parameter keeps its value through training, until it is unfrozen.
:meth:`~nnodely.Modely.freeze` and :meth:`~nnodely.Modely.unfreeze` act on:

- every Parameter of the model, without arguments, including those of the
  models it calls and of its Loop bodies;
- the Parameters named, given a name or a list of names;
- every Parameter of a model the graph reaches, given its name: a block, the
  body of a Loop, or the model of a LocalModel.

A name that matches nothing raises, and nothing is frozen. Both work before
or after ``build()``, and the next ``train()`` follows them. A single
Parameter is frozen with ``p.freeze()``, and found by its name or by its
index:

.. code-block:: python

   model.unfreeze()                              # every Parameter trains again
   model.freeze(["fir_y_kernel", "fir_y_bias"])  # those two only
   model.parameters[2].freeze()                  # fir_u_kernel, by its index
   kernel = model.parameters["fir_y_kernel"].param   # its Keras variable
   print(kernel.shape)
   model.unfreeze()

.. code-block:: text
   :class: output

   (1, 1)

``param`` is the Keras variable the built model computes with, whose value
training changes. ``value`` is the value the Parameter was declared with.

``freeze()`` does not reach two kinds of weights:

- the Keras weights of a :class:`~nnodely.KerasFunction`, which are not
  Parameters: set ``trainable`` on its Keras layer;
- the moving statistics of a :class:`~nnodely.BatchNorm`, which training
  keeps updating.

A block is frozen the same way, ``block.freeze()``, or by its name from the
larger model, ``model.freeze("block")``, to train the rest of the model
around it (see :doc:`composition`).

Step 7: reproducibility
-----------------------

:func:`~nnodely.set_seed` seeds Python, NumPy and the backend together. Call
it before creating the layers, so the initial weights are reproducible too.
It also sets the ``NNODELY_SEED`` environment variable, which child processes
inherit and which has the same effect when set by hand.
:func:`~nnodely.get_seed` returns the current seed.

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Error
     - Cause
     - Fix
   * - ``No minimizers defined. Call minimize() before train().``
     - The model has no objective.
     - Declare one with ``minimize()``, then ``build()``.
   * - ``Early stopping monitors 'val_loss', which training does not log``
     - A validation loss is watched without ``val_data``.
     - Pass ``val_data``, or watch ``"loss"``.
   * - ``Model 'm' reaches no Parameter and no model named 'x'``
     - ``freeze`` or ``unfreeze`` with a name that does not exist.
     - Use a name of the table ``model.parameters``.
   * - ``... at indices [0, 1]`` when freezing by name
     - Several Parameters or models share the name.
     - Freeze one by its index, ``model.parameters[1].freeze()``, or rename
       one (see :doc:`composition`).
   * - The loss does not decrease
     - A weight is frozen, or the learning rate is too low or too high.
     - Check the ``trainable`` column of ``model.parameters``; try another
       ``lr``.

See also
--------

:meth:`Modely.minimize <nnodely.Modely.minimize>`,
:meth:`Modely.train <nnodely.Modely.train>`,
:attr:`Modely.parameters <nnodely.Modely.parameters>`,
:meth:`Modely.freeze <nnodely.Modely.freeze>`, :class:`~nnodely.Parameter`,
:func:`~nnodely.set_seed`, :class:`~nnodely.TinyPrinter`,
:class:`~nnodely.LegacyPrinter`, :class:`~nnodely.NNodelyPrinter`.
