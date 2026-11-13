Composing Models
================

At the end of this page you will be able to use a trained model as a block of
a larger one, keep its weights fixed or train them further, deal with names
that clash, and inline every block with ``flatten()``.

**Prerequisites:** :doc:`models`, :doc:`training`.

**Tutorials:** :doc:`../tutorial/01_model_design`,
:doc:`../tutorial/12_custom_layers`.

**The example.** A sensor signal is smoothed by a filter, and a controller
computes a command from the smoothed signal. The smoother is a model of its
own, used as a block of the controller.

Step 1: call a model on streams
-------------------------------

A :class:`~nnodely.Modely` called on streams is a **block**: it returns its
outputs as streams of the larger graph. The streams are given as a built
model is given its arrays (see :doc:`models`):

- a dictionary keyed by the Input or by its name;
- a list, one stream per input, in the order of the ``inputs`` the block was
  declared with;
- the stream alone, for a block with one input.

.. code-block:: python

   import keras
   import numpy as np
   from nnodely import Fir, Input, Linear, Modely, Output, ReLU

   signal = Input("signal", tw=10)
   filtered = Output("filtered", Fir()(signal))
   smoother = Modely("smoother", inputs=[signal], outputs=[filtered]).build()

   raw = Input("raw", tw=10)
   smoothed = smoother(raw)   # or smoother([raw]), smoother({signal: raw}), smoother({"signal": raw})
   command = Output("command", ReLU()(Linear(features=1)(smoothed)))

   controller = Modely("controller", inputs=[raw], outputs=[command]).build()
   result = controller({"raw": np.ones((1, 10, 1))})
   print(keras.ops.convert_to_numpy(result["command"]).shape)

.. code-block:: text
   :class: output

   (1, 1, 1)

Every declared input of the block needs a stream: one left out is refused. A
key that names no input of the block is ignored. A block with several
outputs returns them as a list: ``first, second = block([a, b])``.

A block keeps the shapes it was declared with. Every stream it is called on
has the shape of the block's input: here a window of 10 samples of one
feature. Another shape is refused rather than reshaped, so a rollout is never
slipped into a block that reads one sample. A dynamic sequence matches any
length.

A block reads no Input but those it declares, so all it reads comes from the
call. To give a block the data of an Input of the larger model, declare that
Input in the larger model and pass it in the call. A state that a block
declares reads the stream given for it at every step. In a
:class:`~nnodely.ModelyLoop` it keeps its feedback instead, and the stream is
its initial value (see :doc:`recurrent`). A hidden state of a block,
``x.feedback(stream, hidden=True)``, is the block's own: the call does not
give it, and it reads its ``init``, at every step or, in a ModelyLoop, before
the first one of each call.

Step 2: weights
---------------

A block brings its weights. A model trained on its own, or loaded with
:meth:`~nnodely.Modely.load`, can be inserted in a larger one, and the larger
model starts from the trained weights. Building the larger model initializes
only its new layers; ``build(keep_weights=False)`` initializes all of them,
the block's too.

Training the larger model trains the block too. To keep a block fixed, freeze
it:

- ``block.freeze()`` keeps every Parameter of the block as it is, until
  ``block.unfreeze()``;
- the larger model freezes it by its name, ``controller.freeze("smoother")``;
- a list of names freezes several blocks, or Parameters, at once.

.. code-block:: python

   controller.freeze("smoother")
   print([p.trainable for p in controller.parameters])
   controller.unfreeze()

.. code-block:: text
   :class: output

   [True, True, False, False]

Calling the same block several times shares its weights between the calls,
as reusing a layer does:

.. code-block:: python

   left = Input("left", tw=10)
   right = Input("right", tw=10)
   both = Output("both", smoother(left) + smoother(right))
   stereo = Modely("stereo", inputs=[left, right], outputs=[both]).build()

Each call returns outputs named after the outputs of the block, so two calls
cannot both be outputs of the same model as they are. Combine them, as above,
or wrap each one in an :class:`~nnodely.Output` with a name of its own.

Saving the larger model keeps its blocks: each is saved in a folder of its
own, with its weights, and the loaded model calls them again (see
:doc:`export`).

Step 3: names that clash
------------------------

Nothing keeps two objects from sharing a name: two layers named alike, a
block and the model that calls it, two copies of one loaded model. The calls
that take a name look it up among the objects the graph reaches, the model's
own first, then those of the models it calls:

- ``parameters[...]``, ``models[...]``, ``inputs[...]``, ``outputs[...]``,
  ``minimizers[...]``;
- the source and the target of ``minimize``;
- ``freeze`` and ``unfreeze``.

A name that one object carries picks it. A name that several carry raises
``ValueError``, which lists them, each with its index and the model it is in.
Every table of a model prints an ``idx`` column: ``model.parameters[2]``
picks the item of that row, the same after saving and loading. An item is
picked by its name or by its index, never both. Renaming one,
``obj.name = "new_name"``, gives each a name of its own:

.. code-block:: python

   from nnodely import Parameter

   u = Input("u", tw=1)
   block = Modely("block", inputs=[u], outputs=[Output("y", u * Parameter("k", value=2.0))])
   x = Input("x", tw=1)
   model = Modely("model", inputs=[x],
                  outputs=[Output("out", block(x * Parameter("k", value=10.0)))])

   print([p.name for p in model.parameters])   # index 0: the model's k, index 1: the block's
   model.parameters[1].freeze()                # the block's k alone
   model.parameters[1].name = "k_block"        # now each is picked by its name
   model.unfreeze("k_block")

.. code-block:: text
   :class: output

   ['k', 'k']

Here ``model.parameters["k"]`` and ``model.freeze("k")`` raise an error that
lists ``[0] k in 'model'; [1] k in 'block'``.

Inside *nnodely* the objects themselves are used, never their names, so an
object can be renamed at any time and is still found. A renamed Parameter or
model is picked, frozen and saved under its new name. A renamed Input is read
from the data by its new name, and a renamed output is returned under it. An
objective is renamed the same way, ``model.minimizers["fit"].name =
"track"``. Training and validation key their data by the names the model was
built with, so they need a new ``build()`` after an Input or an output is
renamed.

Input and output names are the keys of the data and of the results, so they
stay unique within a model:

- two Inputs of one name are refused when the model is built, and, renamed
  alike after the build, when the model is run on data;
- they are still two Inputs: a model saved with them loads them apart, and
  runs once one is renamed, ``loaded.inputs[1].name = "x2"``;
- a block whose inputs share a name is called with a list, which pairs the
  streams with its inputs by position.

The hidden state of an ``Integrate``, a ``Derivative`` or a ``Buffer`` is an
Input named after the layer and its signal. Named like another Input of the
model, it takes a free name. A hidden state the user named,
``x.feedback(stream, hidden=True)``, keeps its name: named like another Input
of the model, it is refused when the model is built.

Step 4: flattening
------------------

:meth:`~nnodely.Modely.flatten` returns an equivalent model in which every
block is inlined, so that the graph contains only layers:

.. code-block:: python

   flat = controller.flatten()
   result = flat.build()({"raw": np.ones((1, 10, 1))})
   print(keras.ops.convert_to_numpy(result["command"]).shape)

.. code-block:: text
   :class: output

   (1, 1, 1)

What the flat model keeps, and what it does not:

- flattened after ``build()``, it computes the same outputs, and shares the
  weights of the original: training one trains the other. Flattened before
  ``build()``, it draws weights of its own;
- the body of a :class:`~nnodely.Loop` stays a body, not inlined;
- objectives are not carried: declare them again with ``minimize()``;
- declared inputs that no output reads are left out, so a call with a list
  takes one array per input the flat model keeps.

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Error
     - Cause
     - Fix
   * - ``Model 'blk' reads its input 'x' as Shape(...), so used as a block it
       takes a stream of that shape``
     - The block is called on a stream of another shape.
     - Give it a stream of the shape of its input, or declare the block on
       the shape it is used with.
   * - ``Model 'blk' has 1 inputs ['x']: used as a block it takes a stream
       for each``
     - An input of the block has no stream in the call.
     - Pass a stream for every declared input.
   * - ``ValueError`` listing ``[0] k in 'model'; [1] k in 'block'``
     - A name that several objects carry.
     - Pick by index, ``model.parameters[1]``, or rename one.

See also
--------

:class:`~nnodely.Modely`, :meth:`Modely.flatten <nnodely.Modely.flatten>`,
:meth:`Modely.freeze <nnodely.Modely.freeze>`,
:attr:`Modely.parameters <nnodely.Modely.parameters>`, :doc:`export`.
