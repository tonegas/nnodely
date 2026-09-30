Saving and Export
=================

A built model can be stored in three formats:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Format
     - Methods
     - Use it to
   * - *nnodely* file
     - :meth:`~nnodely.Modely.save`, :meth:`~nnodely.Modely.load`
     - store the whole model (graph, objectives and weights) and keep working
       on it with *nnodely*.
   * - Keras file (``.keras``)
     - :meth:`~nnodely.Modely.export_keras`, :meth:`~nnodely.Modely.import_keras`
     - use the network as a plain ``keras.Model``.
   * - ONNX (``.onnx``)
     - :meth:`~nnodely.Modely.export_onnx`, :meth:`~nnodely.Modely.validate_onnx`
     - deploy the network outside Python, or with any ONNX runtime.

.. code-block:: python

   import numpy as np
   from nnodely import Fir, Input, Modely, Output, ReLU

   x = Input("x")
   y = Output("y", ReLU()([Fir(out_features=1)([x.sw(10)])]))
   model = Modely("filter", inputs=[x], outputs=[y]).build()
   sample = {"x": np.random.rand(1, 1, 10).astype("float32")}

Saving and loading
------------------

.. code-block:: python

   model.save("filter.nnodely")
   restored = Modely.load("filter.nnodely")
   print(np.allclose(model(sample)["y"], restored(sample)["y"]))

A restored model is a full :class:`~nnodely.Modely`: it can be trained,
composed and exported again. Its layers are layers of their own: a layer added
to it, the model it was saved from, or another model loaded from the same file
never share their weights.

Weights are saved and loaded by default. ``model.save(path, weights=False)``
saves the architecture alone, to be trained later or on another machine, and
``Modely.load(path, weights=False)`` ignores the saved weights; either way the
loaded model is initialized as ``build()`` does.

Keras
-----

.. code-block:: python

   model.export_keras("exports")   # writes exports/filter.keras
   keras_model = Modely.import_keras("exports/filter.keras")
   print(keras_model(sample)["y"].shape)

Both exporters write into the folder they are given and name the file after
the model, unless ``filename`` names it; the ``.keras`` or ``.onnx`` suffix is
added when missing. ``import_keras`` returns a ``keras.Model``, which runs on
any Keras backend.

ONNX
----

Exporting to ONNX needs the ``onnx`` extra (``pip install "nnodely[onnx]"``)
and the **TensorFlow** or **PyTorch** backend:

.. code-block:: python
   :class: skip-test

   model.export_onnx("exports")    # writes exports/filter.onnx
   outputs = Modely.validate_onnx("exports/filter.onnx", sample, return_dict=True)
   print(outputs["y"])

:meth:`~nnodely.Modely.validate_onnx` runs the exported file with ONNX Runtime,
to check it against the original model. ``batch_size`` fixes the batch axis of
the exported graph, and ``input_signature`` fixes every input shape.

Backend support
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 15 15 15

   * - Model
     - TensorFlow
     - PyTorch
     - JAX
   * - Any model, except the cases below
     - yes
     - yes
     - no
   * - ``Derivative`` with respect to an input
     - yes (batch 1)
     - no
     - no
   * - ``OdeNet`` with ``method="dopri5"``
     - no
     - no
     - no

A derivative with respect to an input records the backward pass in the
exported graph. Its shape arithmetic only converts with a fixed batch axis, so
such a model is exported with a batch of one unless ``batch_size`` or
``input_signature`` says otherwise.

Visualization
-------------

:meth:`~nnodely.Modely.export_html` and :meth:`~nnodely.Modely.plot` draw the
graph of a model (see :doc:`models`).
