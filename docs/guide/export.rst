Saving and Exporting a Model
============================

At the end of this page you will be able to save a model and load it back,
export it as a plain Keras model or as an ONNX file, and draw its graph.

**Prerequisites:** :doc:`models`.

**Tutorial:** :doc:`../tutorial/05_save_load_export`.

**The example.** A filter: a ReLU of a FIR filter over 10 samples. It stands
for any trained model.

A built model can be stored in three formats:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Format
     - Methods
     - Use it to
   * - *nnodely* folder
     - :meth:`~nnodely.Modely.save`, :meth:`~nnodely.Modely.load`
     - Store the model, its graph and its weights, and keep working on it
       with *nnodely*.
   * - Keras file (``.keras``)
     - :meth:`~nnodely.Modely.export_keras`, :meth:`~nnodely.Modely.import_keras`
     - Use the network as a plain ``keras.Model``, for inference.
   * - ONNX file (``.onnx``)
     - :meth:`~nnodely.Modely.export_onnx`, :meth:`~nnodely.Modely.run_onnx`
     - Deploy the network outside Python, with any ONNX runtime.

.. code-block:: python

   import keras
   import numpy as np
   from nnodely import Fir, Input, Modely, Output, ReLU

   x = Input("x", tw=10)
   y = Output("y", ReLU()(Fir()(x)))
   model = Modely("filter", inputs=[x], outputs=[y]).build()
   sample = {"x": np.random.rand(1, 10, 1).astype("float32")}

Step 1: save and load
---------------------

.. code-block:: python

   model.save("filter.nnodely")
   restored = Modely.load("filter.nnodely")
   same = np.allclose(
       keras.ops.convert_to_numpy(model(sample)["y"]),
       keras.ops.convert_to_numpy(restored(sample)["y"]),
   )
   print(same)

.. code-block:: text
   :class: output

   True

A restored model is a full :class:`~nnodely.Modely`: it can be trained,
composed and exported again. Its layers are layers of its own: it never
shares weights with the model it was saved from, or with another model loaded
from the same folder.

Objectives are not saved. Declare them again with ``minimize()``, and call
``build()``, before training a loaded model.

The path is a folder:

- ``model.json`` holds the graph, and ``model.weights.h5`` the weights;
- every model it calls as a block, or rolls out in a :class:`~nnodely.Loop`,
  is saved the same way in a folder of its own inside it, named after that
  model, and nested as the models are.

The restored model calls the same models: :attr:`~nnodely.Modely.models`
lists them, ``freeze`` reaches them by name, and a model called several times
is one model again.

The weights a block computes with are in the file of the model that calls
it. The folder of the block holds its graph, and weights of its own only if
the block was built on its own, as a Loop body always is: a block is not
built to be saved. Each folder loads on its own, a block without weights of
its own with initial weights. To take a block trained inside the model out
of it:

- save the block itself, ``smoother.save("smoother")``, which builds it and
  keeps the weights it was trained with;
- or save the model with ``weights="all"``: every block is built and saves
  its own weights too, so ``Modely.load("controller.nnodely/smoother")`` is
  the trained block.

A model is built once loaded, so ``save`` refuses a model whose Inputs share
a name, and with ``weights="all"`` a block whose Inputs do, before anything
is written. A block whose Inputs share a name, called by position, is saved
as usual.

Weights are saved and loaded by default. ``model.save(path, weights=False)``
saves the architecture alone, to train it later or on another machine, and
``Modely.load(path, weights=False)`` ignores the saved weights. Either way,
the loaded model is initialized as ``build()`` does.

A model that uses a custom Keras layer, through
:class:`~nnodely.KerasFunction` or a custom membership function, loads in
another session only after that layer is defined, decorated with
``@keras.saving.register_keras_serializable()``.

Step 2: export to Keras
-----------------------

.. code-block:: python

   model.export_keras("exports")   # writes exports/filter.keras
   keras_model = Modely.import_keras("exports/filter.keras")
   print(keras.ops.convert_to_numpy(keras_model(sample)["y"]).shape)

.. code-block:: text
   :class: output

   (1, 1, 1)

Both exporters write into the folder they are given, and name the file after
the model unless ``filename`` names it. The ``.keras`` or ``.onnx`` suffix is
added when missing. ``import_keras`` returns a ``keras.Model``, which runs on
any Keras backend.

The ``.keras`` file is a stand-alone copy of the model, made for inference.
In *nnodely* one weight can be read from several places: a body rolled out by
two Loops, a Parameter read inside a Loop body and outside it, a block also
used in a Loop, or the graphs a ``Gradient`` evaluates again. Keras cannot
rebuild a layer that two of its graphs call, so the file holds a copy of such
a weight for each place that reads it, all with the same value. The imported
model gives the same outputs, but trained further as a plain Keras model,
those copies would drift apart. To keep training with the weights shared,
use :meth:`~nnodely.Modely.save` and :meth:`~nnodely.Modely.load`.

Step 3: export to ONNX
----------------------

Every backend exports, with the converter its extra installs:

.. code-block:: python

   model.export_onnx("exports")    # writes exports/filter.onnx
   outputs = Modely.run_onnx("exports/filter.onnx", sample, return_dict=True)
   print(outputs["y"].shape)

.. code-block:: text
   :class: output

   (1, 1, 1)

:meth:`~nnodely.Modely.run_onnx` runs the exported file with ONNX
Runtime, to compare it with the original model. The model is exported as
declared, without its objectives: only its outputs, and the inputs they
read.

Every axis of an ONNX file is fixed:

- the **batch** is one sample by default. ``batch_size=n`` exports a file
  that computes ``n`` samples at every call;
- every **sequence** has its declared length. A sequence of dynamic length,
  ``seq=-1``, such as every Input of a rollout of
  :meth:`~nnodely.Modely.loop` without ``seq``, has the length
  ``seq`` gives; without it the export raises.

The file refuses arrays of other shapes: export it again for another batch or
another rollout length.

.. code-block:: python

   model.export_onnx("exports", "filter_batch", batch_size=4)
   batch = {"x": np.random.rand(4, 10, 1).astype("float32")}
   print(Modely.run_onnx("exports/filter_batch.onnx", batch)[0].shape)

.. code-block:: text
   :class: output

   (4, 1, 1)

A deployed model mostly runs one sample at a time, in a controller or an
estimator. A batch of ``n`` serves the uses that run ``n`` samples at every
call, with ``n`` known when the model is deployed:

- sampling-based predictive control (MPPI, CEM), which rolls ``n``
  candidate control sequences out through the model at every step;
- particle filters and ensemble Kalman filters, which propagate ``n``
  particles;
- one model serving several vehicles, robots or tanks;
- replaying a recorded dataset in another runtime, or Monte Carlo studies
  over initial conditions.

.. list-table:: ONNX support by backend
   :header-rows: 1
   :widths: 40 20 20 20

   * - Model
     - TensorFlow
     - PyTorch
     - JAX
   * - Without a Gradient, a Loop and a rollout of
       :meth:`~nnodely.Modely.loop` included
     - yes
     - yes
     - yes
   * - With a :class:`~nnodely.Gradient` with respect to an Input
     - when tf2onnx converts the gradient of every layer it differentiates
       (``Sin`` yes, ``Tanh`` no)
     - no
     - yes

A derivative with respect to an Input records the backward pass in the
exported graph. JAX exports it whatever it differentiates; the PyTorch
exporter traces a forward pass only, and refuses it. The file computes what
the model computes, up to float32 rounding.

Step 4: draw the graph
----------------------

:meth:`~nnodely.Modely.export_html` draws the graph of a model as interactive
pages, one per block (see :doc:`models`):

.. code-block:: python

   model.export_html("filter_html")

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Error
     - Cause
     - Fix
   * - ``No minimizers defined`` when training a loaded model
     - Objectives are not saved.
     - Declare them again with ``minimize()``, then ``build()``.
   * - A loaded model with a custom layer fails to load
     - The custom Keras class is not defined in this session.
     - Define it first, decorated with
       ``@keras.saving.register_keras_serializable()``.
   * - ``ImportError: ONNX export on the jax backend needs the jax2onnx package``
     - *nnodely* was installed without its backend extra, which brings the
       converter.
     - Install the extra of the backend, ``pip install "nnodely[jax]"``.
   * - ``ValueError: Model 'm' reads 'x' as a sequence of dynamic length``
     - An ONNX file has fixed shapes, and the model reads a sequence of
       dynamic length.
     - Give the length to export, ``export_onnx(..., seq=n)``.
   * - ONNX Runtime: ``Got invalid dimensions for input``
     - The file is run with another batch, or another sequence length, than
       it was exported with.
     - Export it again with ``batch_size=`` or ``seq=``.

See also
--------

:meth:`Modely.save <nnodely.Modely.save>`,
:meth:`Modely.load <nnodely.Modely.load>`,
:meth:`Modely.export_keras <nnodely.Modely.export_keras>`,
:meth:`Modely.export_onnx <nnodely.Modely.export_onnx>`,
:meth:`Modely.run_onnx <nnodely.Modely.run_onnx>`,
:meth:`Modely.export_html <nnodely.Modely.export_html>`.
