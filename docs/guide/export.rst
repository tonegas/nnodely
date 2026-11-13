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
     - :meth:`~nnodely.Modely.export_onnx`, :meth:`~nnodely.Modely.validate_onnx`
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
is one model again. Each folder is a model that loads on its own:
``Modely.load("controller.nnodely/smoother")`` takes a trained block out of
the model that holds it.

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

Exporting to ONNX needs the ``onnx`` extra, ``pip install "nnodely[onnx]"``,
and the **TensorFlow** or **PyTorch** backend. This block does not run on
JAX:

.. code-block:: python
   :class: skip-test

   model.export_onnx("exports")    # writes exports/filter.onnx
   outputs = Modely.validate_onnx("exports/filter.onnx", sample, return_dict=True)
   print(outputs["y"].shape)

:meth:`~nnodely.Modely.validate_onnx` runs the exported file with ONNX
Runtime, to compare it with the original model. The model is exported as
declared, without its objectives: only its outputs, and the inputs they
read. ``batch_size`` fixes the batch axis of the exported graph, and
``input_signature`` fixes the shape of every input.

.. list-table:: ONNX support by backend
   :header-rows: 1
   :widths: 40 20 25 15

   * - Model
     - TensorFlow
     - PyTorch
     - JAX
   * - Without a Loop or a Gradient
     - yes
     - yes
     - no
   * - With a :class:`~nnodely.Loop`, or a :class:`~nnodely.ModelyLoop`
     - yes; a dynamic rollout follows the length of the data
     - yes, with the batch and the rollout of the data it was exported
       with
     - no
   * - With a :class:`~nnodely.Gradient` with respect to an Input
     - yes, with a batch of one
     - no
     - no

A derivative with respect to an Input records the backward pass in the
exported graph. Its shape arithmetic only converts with a fixed batch axis,
so such a model is exported with a batch of one, unless ``batch_size`` or
``input_signature`` says otherwise. JAX cannot export ONNX at all: Keras has
no ONNX exporter for it.

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
   * - ONNX export fails on JAX
     - Keras cannot export ONNX from JAX.
     - Export with ``KERAS_BACKEND=tensorflow`` or ``torch``.

See also
--------

:meth:`Modely.save <nnodely.Modely.save>`,
:meth:`Modely.load <nnodely.Modely.load>`,
:meth:`Modely.export_keras <nnodely.Modely.export_keras>`,
:meth:`Modely.export_onnx <nnodely.Modely.export_onnx>`,
:meth:`Modely.validate_onnx <nnodely.Modely.validate_onnx>`,
:meth:`Modely.export_html <nnodely.Modely.export_html>`.
