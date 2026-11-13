Getting Started
===============

This page installs *nnodely* and builds a first model: it learns to predict
the motion of a mass-spring-damper from data, then runs on new samples. It
takes a few minutes, and assumes only Python 3.10 to 3.13 and NumPy.

Installation
------------

*nnodely* runs on `Keras 3 <https://keras.io>`_, which needs one backend:
TensorFlow, PyTorch or JAX. Install *nnodely* together with the backend you
want:

.. code-block:: bash

   pip install "nnodely[tensorflow]"
   # or: pip install "nnodely[torch]"
   # or: pip install "nnodely[jax]"

The ``onnx`` extra adds what is needed to export and run ONNX models:

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
it is first imported. Without it, Keras uses TensorFlow. Set it before
importing *nnodely*, in the shell:

.. code-block:: bash

   export KERAS_BACKEND=torch

or at the very top of your script:

.. code-block:: python
   :class: skip-test

   import os

   os.environ["KERAS_BACKEND"] = "torch"

   import nnodely

A model is written once and runs on all three backends. The few features that
depend on the backend, mostly the ONNX export, are listed in
:doc:`guide/export`.

A first model: mass-spring-damper
---------------------------------

.. sidebar:: The system

   A mass :math:`m` on a spring of stiffness :math:`k` and a damper of
   coefficient :math:`c`, driven by a force :math:`F`:

   .. math::

      m \ddot x = - k x - c \dot x + F

The goal is to predict the next position of the mass from its recent
positions and from the force applied now.

Step 1: the data
~~~~~~~~~~~~~~~~

The system is simulated here, with a force that changes every 50 samples. In
a real application these arrays come from measurements.

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

Step 2: the model
~~~~~~~~~~~~~~~~~

An :class:`~nnodely.Input` is a signal the model reads. It is declared with
the window of samples the model reads from it:

- ``tw=(5, 1)`` is the last five samples of ``x`` and the next one;
- ``tw=1`` is the current sample of ``F``.

:class:`~nnodely.TimeRange` and :class:`~nnodely.TimeSelect` take a part of a
window, and a :class:`~nnodely.Fir` filter is a learned weighted sum over a
window.

.. code-block:: python

   from nnodely import DataLoader, Fir, Input, Modely, Output, set_seed
   from nnodely import TimeRange, TimeSelect

   set_seed(0)

   x = Input("x", tw=(5, 1))
   F = Input("F", tw=1)
   x_next = Fir()(TimeRange(0, 5)(x)) + Fir()(F)
   x_next_est = Output("x_next_est", x_next)

   model = Modely("mass_spring_damper", inputs=[x, F], outputs=[x_next_est])
   print(x.shape, x_next.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=6, D=(1,)) Shape(S=(), T=1, D=(1,))

A shape lists the sequence axes ``S``, the window ``T`` and the features ``D``
of a stream (see :doc:`guide/concepts`). The estimator has the structure

.. math::

   \hat x[t+1] = \sum_{i=0}^{4} h_{x,i} \, x[t-i] + h_F \, F[t] + b

This is the structure of the discretized mass-spring-damper: only the
coefficients are unknown. Choosing it is what makes this network
*model-structured*.

Step 3: the objective
~~~~~~~~~~~~~~~~~~~~~

:meth:`~nnodely.Modely.minimize` declares what training reduces: here the
error between the estimate and ``TimeSelect(-1)(x)``, the next sample of the
position. Then :meth:`~nnodely.Modely.build` turns the graph into a Keras
model and creates its weights.

.. code-block:: python

   model.minimize("next_position", x_next_est, TimeSelect(-1)(x), loss="mse")
   model.build()

Step 4: training
~~~~~~~~~~~~~~~~

A :class:`~nnodely.DataLoader` cuts the signals into the windows the model
reads. The keys of the data dictionary are the names of the inputs. Each
sample needs 5 past positions and the next one, so 3000 samples give 2995
windows.

.. code-block:: python

   data = {"x": position, "F": force}
   train_data = DataLoader(model, source=data)
   print(len(train_data))

.. code-block:: text
   :class: output

   2995

.. code-block:: python

   history = model.train(train_data, epochs=30, batch_size=64, lr=1e-2)

``train`` prints the loss of every epoch, and returns the losses in a
dictionary.

Step 5: validation and inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~nnodely.Modely.validate` prints the loss together with RMSE, FIT, R²
and other indicators. Calling the model on a dictionary of arrays runs it.
The arrays have a batch axis first, and the result maps every output name to
a tensor of the backend: ``keras.ops.convert_to_numpy`` makes it a NumPy
array.

.. code-block:: python

   result = model.validate(train_data)

.. code-block:: python

   import keras

   prediction = model(train_data.get_samples(1))
   estimate = keras.ops.convert_to_numpy(prediction["x_next_est"])
   print(estimate.shape)   # (batch, T, D)

.. code-block:: text
   :class: output

   (1, 1, 1)

Where to go next
----------------

- The :doc:`tutorials <tutorial/index>` teach each part of *nnodely* with
  runnable notebooks: start from :doc:`tutorial/01_model_design`.
- :doc:`guide/concepts` explains streams, shapes and windows.
- :doc:`guide/models` lists the building blocks, and the
  :doc:`API reference <api/index>` describes each of them.
- :doc:`guide/recurrent` and :doc:`guide/physics` cover multi-step
  prediction, models declared one step at a time, and physics-based layers.
- The :doc:`glossary` defines the terms these pages use.
