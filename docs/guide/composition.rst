Model Composition
=================

A :class:`~nnodely.Modely` can be used as a block of another model. Calling it
on a list of streams, one per input and in the order of its ``inputs``, returns
its outputs as streams of the outer graph:

.. code-block:: python

   import numpy as np
   from nnodely import Fir, Input, Linear, Modely, Output, ReLU

   signal = Input("signal", sw=10)
   filtered = Output("filtered", Fir(features=1)([signal]))
   smoother = Modely("smoother", inputs=[signal], outputs=[filtered]).build()

   raw = Input("raw", sw=10)
   smoothed = smoother([raw])
   command = Output("command", ReLU()([Linear(features=1)([smoothed])]))

   controller = Modely("controller", inputs=[raw], outputs=[command]).build()
   print(controller({"raw": np.ones((1, 10, 1))})["command"].shape)   # (1, 1, 1)

A model with several outputs returns them as a list:
``first, second = block([a, b])``.

A block keeps the shapes it was declared with. Every stream it is called on
must have the shape of the block's input - here a window of 10 samples of one
feature - the same dim, sequence and window; another shape is refused rather
than reshaped, so a rollout is not slipped into a block that reads one
sample. A dynamic sequence matches any length. An :class:`~nnodely.Input` of
that shape is given as it is, as ``raw`` above.

An Input that a block reads without declaring it is not replaced by the call:
it is the Input of the outer model too, read from the same data. A model with
a :meth:`~nnodely.Modely.rollback` cannot be a block, which would evaluate it
once: call instead a ``Modely`` declared on its inputs and outputs, its
one-step model, which shares its weights. A closed Input that a block declares
reads the stream given for it at every step; in a
:class:`~nnodely.ModelyLoop` it keeps its feedback instead, and the stream is
its initial value (see :doc:`recurrent`).

Weights
-------

A block brings its weights along. A model trained on its own can be inserted in
a larger one, and the larger model starts from the trained weights - also a
model loaded with :meth:`~nnodely.Modely.load`: building the larger model
initializes only its new layers. Training
the outer model trains the block too. To keep a block fixed, freeze it:
``block.freeze()`` keeps every Parameter it computes with as it is through
training, until ``block.unfreeze()``.

Calling the same block several times shares its weights between the calls, as
reusing a layer does:

.. code-block:: python

   left = Input("left", sw=10)
   right = Input("right", sw=10)
   both = Output("both", smoother([left]) + smoother([right]))
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
