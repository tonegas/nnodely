Getting Started
===============

Installation
------------

*nnodely* runs on `Keras 3 <https://keras.io>`_ and needs one Keras backend.
Install it together with the backend you want to use:

.. code-block:: bash

   pip install "nnodely[tensorflow]"
   # or: pip install "nnodely[torch]"
   # or: pip install "nnodely[jax]"

The ``onnx`` extra adds the packages needed to export and run ONNX models:

.. code-block:: bash

   pip install "nnodely[torch,onnx]"

To install from source, clone the repository and install it with the extras
you need:

.. code-block:: bash

   git clone https://github.com/tonegas/nnodely.git
   cd nnodely
   pip install ".[tensorflow]"

Choosing the backend
~~~~~~~~~~~~~~~~~~~~

Keras reads the backend from the ``KERAS_BACKEND`` environment variable when
it is first imported, and uses TensorFlow if the variable is not set. Set it
before importing *nnodely*, either in the shell:

.. code-block:: bash

   export KERAS_BACKEND=torch

or at the very top of your script:

.. code-block:: python
   :class: skip-test

   import os

   os.environ["KERAS_BACKEND"] = "torch"

   import nnodely

A model is written once and runs on all three backends. The few features that
depend on the backend, mostly ONNX export, are listed in :doc:`guide/export`.

A first model: mass-spring-damper
---------------------------------

.. sidebar:: The system

   A mass :math:`m` on a spring of stiffness :math:`k` and a damper of
   coefficient :math:`c`, driven by a force :math:`F`:

   .. math::

      m \ddot x = - k x - c \dot x + F

The goal is to predict the next position of the mass from its recent positions
and the force currently applied to it.

**The data.** Here the system is simulated with a force that changes value
every 50 samples. In a real application these arrays come from measurements.

.. code-block:: python

   import numpy as np

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

**The model.** An :class:`~nnodely.Input` is a signal the model reads. Its
methods select the samples a relation uses: ``x.sw(5)`` is a window of the last
five samples of ``x``, and ``F.last()`` is the current sample of ``F``.

.. code-block:: python

   from nnodely import DataLoader, Fir, Input, Modely, Output, set_seed

   set_seed(0)

   x = Input("x")
   F = Input("F")
   x_next = Fir(out_features=1)([x.sw(5)]) + Fir(out_features=1)([F.last()])
   x_next_est = Output("x_next_est", x_next)

   model = Modely("mass_spring_damper", inputs=[x, F], outputs=[x_next_est])

A :class:`~nnodely.Fir` filter is a learnable weighted sum over the samples of
its window. The estimator above therefore has the structure

.. math::

   \hat x[t+1] = \sum_{i=0}^{4} h_{x,i} \, x[t-i] + h_F \, F[t] + b

This is the structure of the discretized mass-spring-damper, where only the
coefficients are unknown. Choosing it is what makes this network
*model-structured*.

**The objective.** :meth:`~nnodely.Modely.minimize` declares what training
has to reduce. Here it's the error between the estimate and ``x.next()``, the
next sample of the position. After that,
:meth:`~nnodely.Modely.build` turns the graph into a Keras model.

.. code-block:: python

   model.minimize("next_position", x_next_est, x.next(), loss="mse")
   model.build()

**Training.** A :class:`~nnodely.DataLoader` cuts the signals into the windows
the model asks for. The keys of the data dictionary are the input names.

.. code-block:: python

   data = {"x": position, "F": force}
   train_data = DataLoader(model, source=data)

   history = model.train(train_data, epochs=30, batch_size=64, lr=1e-2)

**Validation and inference.** :meth:`~nnodely.Modely.validate` prints the
losses together with RMSE, FIT, R² and other indicators. Calling the model on
a dictionary of windows runs it.

.. code-block:: python

   result = model.validate(train_data)
   prediction = model(train_data[0])
   print(prediction["x_next_est"])

Where to go next
----------------

- :doc:`guide/concepts` explains streams, shapes and windows.
- :doc:`guide/models` lists the building blocks.
- :doc:`guide/recurrent` and :doc:`guide/physics` cover multi-step prediction
  and physics-based layers.
