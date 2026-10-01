Model Composition
=================

A :class:`~nnodely.Modely` can be used as a block of another model. Calling it
on a list of streams, one per input and in the order of its ``inputs``, returns
its outputs as streams of the outer graph:

.. code-block:: python

   import numpy as np
   from nnodely import Fir, Input, Linear, Modely, Output, ReLU

   signal = Input("signal")
   filtered = Output("filtered", Fir(out_features=1)([signal.sw(10)]))
   smoother = Modely("smoother", inputs=[signal], outputs=[filtered]).build()

   raw = Input("raw")
   smoothed = smoother([raw.sw(10)])
   command = Output("command", ReLU()([Linear(out_features=1)([smoothed])]))

   controller = Modely("controller", inputs=[raw], outputs=[command]).build()
   print(controller({"raw": np.ones((1, 1, 10))})["command"].shape)   # (1, 1, 1)

A model with several outputs returns them as a list:
``first, second = block([a, b])``.

Weights
-------

A block brings its weights along. A model trained on its own can be inserted in
a larger one, and the larger model starts from the trained weights. Training
the outer model trains the block too. To keep a block fixed, set its Keras
layers to non-trainable before training (``block.model.trainable = False``).

Calling the same block several times shares its weights between the calls, as
reusing a layer does:

.. code-block:: python

   left = Input("left")
   right = Input("right")
   both = Output("both", smoother([left.sw(10)]) + smoother([right.sw(10)]))
   stereo = Modely("stereo", inputs=[left, right], outputs=[both]).build()

Each call returns outputs named after the block's own outputs, so two calls
can't be exposed directly as outputs of the same model. Combine them, as
above, or wrap each one in an :class:`~nnodely.Output` with its own name.

Flattening
----------

:meth:`~nnodely.Modely.flatten` returns an equivalent model in which every
block is inlined, so the graph contains only layers:

.. code-block:: python

   flat = controller.flatten()
   print(flat)
