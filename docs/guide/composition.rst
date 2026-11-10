Model Composition
=================

A :class:`~nnodely.Modely` can be used as a block of another model. Calling it
on streams returns its outputs as streams of the outer graph. The streams are
given the way a built model is given its arrays (see :doc:`models`): a dict
keyed by the Input or by its name, or a list with one stream per input in the
order of the ``inputs`` the block was declared with:

.. code-block:: python

   import numpy as np
   from nnodely import Fir, Input, Linear, Modely, Output, ReLU

   signal = Input("signal", tw=10)
   filtered = Output("filtered", Fir(features=1)([signal]))
   smoother = Modely("smoother", inputs=[signal], outputs=[filtered]).build()

   raw = Input("raw", tw=10)
   smoothed = smoother([raw])   # or smoother(raw), smoother({signal: raw}), smoother({"signal": raw})
   command = Output("command", ReLU()([Linear(features=1)([smoothed])]))

   controller = Modely("controller", inputs=[raw], outputs=[command]).build()
   print(controller({"raw": np.ones((1, 10, 1))})["command"].shape)   # (1, 1, 1)

Every declared input needs a stream, and one left out is refused; a key that
names no input of the block is ignored. A model with several outputs returns
them as a list: ``first, second = block([a, b])``.

A block keeps the shapes it was declared with. Every stream it is called on
must have the shape of the block's input - here a window of 10 samples of one
feature - the same dim, sequence and window; another shape is refused rather
than reshaped, so a rollout is not slipped into a block that reads one
sample. A dynamic sequence matches any length. An :class:`~nnodely.Input` of
that shape is given as it is, as ``raw`` above.

An Input that a block reads without declaring it is not replaced by the call:
it is the Input of the outer model too, read from the same data. A fed-back
Input that a block declares reads the stream given for it at every step; in a
:class:`~nnodely.ModelyLoop` it keeps its feedback instead, and the stream is
its initial value (see :doc:`recurrent`).

Weights
-------

A block brings its weights along. A model trained on its own can be inserted in
a larger one, and the larger model starts from the trained weights - also a
model loaded with :meth:`~nnodely.Modely.load`: building the larger model
initializes only its new layers, and ``build(keep_weights=False)`` all of
them, the block's too. Training the outer model trains the block too. To keep a block fixed, freeze it:
``block.freeze()`` keeps every Parameter it computes with as it is through
training, until ``block.unfreeze()``. The outer model freezes it by its name
as well, ``controller.freeze("smoother")``, and a list of names freezes
several blocks, or Parameters, at once.

Saving the larger model keeps its blocks: each is saved in a folder of its own,
with its weights, and the loaded model calls them again (see :doc:`export`).

Names that clash
----------------

Nothing keeps two objects from sharing a name: two layers named alike, a
block and the model calling it, two copies of one loaded model - a loaded
model can only be reached by the names it was saved with. A call that takes a
name - ``parameters[...]``, ``models[...]``, ``inputs[...]``,
``outputs[...]``, ``minimizers[...]``, the sides of ``minimize``, ``freeze``
and ``unfreeze`` - looks it up among the objects the graph reaches, in that
order: the model's own first, then those of the models it calls. A name one
object carries picks it. A name several carry raises ``ValueError`` listing
them, each with its index and the model it is in. An item is picked by its
name or by its index - never both: every table of a model - ``parameters``,
``models``, ``inputs``, ``outputs``, ``minimizers`` - prints an ``index``
column, and ``model.parameters[2]`` picks the item of that row, the same
after saving and loading; renaming one - ``obj.name = "new_name"`` - gives
each a name of its own. ``freeze`` and ``unfreeze`` take names; one of
several of a name is frozen by itself, ``model.parameters[2].freeze()``.

Inside nnodely the objects themselves are used, never their names, so an
object can be renamed at any time and is still found: a renamed Parameter or
model is picked, frozen and saved under its new name, a renamed Input is read
from the data by its new name, a renamed output returned under it, and a
minimizer is renamed the same way, ``model.minimizers["fit"].name = "track"``.
Only training and validation, which key their data by the names the model was
built with, ask for a new ``build()`` after an Input or an output is renamed:

.. code-block:: python

   from nnodely import Input, Modely, Output, Parameter

   u = Input("u", tw=1)
   block = Modely("block", inputs=[u], outputs=[Output("y", u * Parameter("k", value=2.0))])
   x = Input("x", tw=1)
   model = Modely("model", inputs=[x], outputs=[Output("out", block([x * Parameter("k", value=10.0)]))])

   print(model.parameters)              # index 0: the model's k, index 1: the block's
   # model.parameters["k"] raises: ... at indices [0, 1]: [0] k in 'model'; [1] k in 'block'
   outer_k = model.parameters[0]        # the model's own, 10
   model.parameters[1].freeze()         # the block's alone; freeze("k") raises too
   model.parameters[1].name = "k_block" # now each is picked by its name
   model.unfreeze("k_block")

Input and output names are the keys of the data and of the results, so they
stay one per model: two Inputs of one name are refused when the model is
built, and, renamed alike after the build, when it is run on data. They are
still two Inputs: a model saved with them loads them apart, and runs once one
is renamed, ``loaded.inputs[1].name = "x2"``. A block whose inputs share a
name is called with a list, which pairs the streams with its inputs by
position. An Input inside a block that the block does not declare reads the
data of its name. The state a layer keeps - ``Integrate``, ``Derivative``,
``Buffer`` - is an Input named after the layer and its signal, which the user
never names: named like another Input of the model, it takes a free name.

Calling the same block several times shares its weights between the calls, as
reusing a layer does:

.. code-block:: python

   left = Input("left", tw=10)
   right = Input("right", tw=10)
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
