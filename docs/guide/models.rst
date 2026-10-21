Building Models
===============

This page lists the blocks a model is made of. Their full signatures are in the
:doc:`API reference <../api/index>`.

Inputs and outputs
------------------

:class:`~nnodely.Input` declares a signal read from data, and
:class:`~nnodely.Output` names a stream the model exposes:

.. code-block:: python

   import numpy as np
   from nnodely import Input, Modely, Output

   velocity = Input("velocity")
   altitude = Input("altitude", dim=3)      # three features per sample
   gear = Input("gear")

Parameters and constants
------------------------

A :class:`~nnodely.Parameter` is a trainable tensor used directly in the graph,
and a :class:`~nnodely.Constant` is a fixed one. A ``value`` sets the shape:
every axis of the value is a ``dim`` axis and the value has no time axis - it
applies to every sample, of any stream - unless ``dim`` and ``time`` name the
axes of the value that play each part: ``Parameter("w", value=np.ones((2, 3)),
time=1)`` is a 2-feature window of 3 samples, and ``time`` can name several
axes, outermost first. A constant always takes its shape from its ``value``; a
parameter can be given ``dim`` and ``time`` as sizes instead, and start from a
random value. Like every stream they are laid out ``(batch, *time, *dim)``,
the same value for every sample:

.. code-block:: python

   from nnodely import Constant, Parameter

   drag = Parameter("drag", value=0.1)                     # trainable scalar
   weights = Parameter("weights", dim=3)                   # random initial value
   gravity = Constant("gravity", value=9.81)

   air_drag = -1.0 * drag * velocity.last() * velocity.last()

Layers
------

Every layer is created with its configuration and then called on streams. A
layer with nothing to configure - its constructor takes only ``name``, as
``Sin``, ``Exp`` or ``Sigmoid`` do - can be applied as it is created: ``Sin(x)`` is ``Sin()(x)``. To name it, configure it first:
``Sin(name="s")(x)``. A layer that reads one stream - an activation, a math or
trigonometric layer, ``Clamp``, a membership function - called on a list of
streams is applied to each and returns a list: ``Sin([a, b])`` gives
``[sin(a), sin(b)]``, and one ``PReLU`` object shares its slopes between them.
The tables group the available layers.

.. list-table:: Layers with weights
   :header-rows: 1
   :widths: 25 75

   * - Layer
     - What it does
   * - :class:`~nnodely.Linear`
     - Affine map ``W x + b`` over all the features of a sample (the dim
       axes), or over one dim axis with ``axis`` - with a matrix per position
       of the other axes, or one for all with ``shared_kernel`` - applied to
       every sample.
   * - :class:`~nnodely.Fir`
     - Finite impulse response filter: a learned weighted sum over the time
       window of every feature alone, returning one sample.
   * - :class:`~nnodely.LocalModel`
     - Local models blended by membership degrees (for example the output of
       :class:`~nnodely.Fuzzify`).
   * - :class:`~nnodely.EquationLearner`
     - Symbolic-regression block: a projection feeds a list of elementary
       functions whose results are combined.
   * - :class:`~nnodely.BatchNorm`
     - Batch normalization per feature.

.. list-table:: Layers without weights
   :header-rows: 1
   :widths: 25 75

   * - Layer
     - What it does
   * - :class:`~nnodely.Fuzzify`
     - Memberships of a scalar signal to sets placed on given centers, one
       feature per center: a membership function on each, or any layer of
       ``x - center``.
   * - Membership functions
     - :class:`~nnodely.Triangular`, :class:`~nnodely.Rectangular`,
       :class:`~nnodely.Gaussian`: layers of their own, applied to every value
       of a stream.
   * - :class:`~nnodely.Interpolation`
     - One-dimensional lookup table, linear or polynomial.
   * - Activations
     - :class:`~nnodely.ReLU`, :class:`~nnodely.LeakyReLU`, :class:`~nnodely.ELU`,
       :class:`~nnodely.PReLU`, :class:`~nnodely.Sigmoid`, :class:`~nnodely.Tanh`,
       :class:`~nnodely.Softmax`, :class:`~nnodely.Swish`, :class:`~nnodely.GELU`,
       :class:`~nnodely.Softplus`.
   * - Math
     - :class:`~nnodely.Sin`, :class:`~nnodely.Cos`, :class:`~nnodely.Tan`,
       :class:`~nnodely.Asin`, :class:`~nnodely.Acos`, :class:`~nnodely.Atan`,
       :class:`~nnodely.Atan2`,
       :class:`~nnodely.Exp`, :class:`~nnodely.Log`, :class:`~nnodely.Log10`,
       :class:`~nnodely.Sqrt`, :class:`~nnodely.Abs`, :class:`~nnodely.Sign`,
       :class:`~nnodely.Floor`, :class:`~nnodely.Ceil`, :class:`~nnodely.Deg2Rad`,
       :class:`~nnodely.Negative`, :class:`~nnodely.Clamp`, :class:`~nnodely.Sum`.
   * - Feature axes
     - :class:`~nnodely.Select` (one index), :class:`~nnodely.Range` (a slice),
       :class:`~nnodely.Concatenate`. ``axis`` counts the dim axes from 0.
   * - Time axes
     - :class:`~nnodely.TimeSelect` (one index), :class:`~nnodely.TimeRange`
       (a slice), :class:`~nnodely.TimeConcatenate`,
       :class:`~nnodely.TimeStack` (a new axis), :class:`~nnodely.TimeReverse`.
       ``axis`` counts the time axes outermost first: the default, -1, is the
       inner time, the window, and 0 the outer time - the rollout of the
       outermost Loop, or the window without one.
   * - Calculus
     - :class:`~nnodely.Differentiate`, :class:`~nnodely.Derivative`,
       :class:`~nnodely.Integrate`, :class:`~nnodely.IntegrateStep`,
       :func:`~nnodely.Ode`, :class:`~nnodely.OdeNet` (see :doc:`physics`).
   * - Recurrence
     - :class:`~nnodely.Loop`, :class:`~nnodely.Roll` (see :doc:`recurrent`).

``Select`` drops the dim axis it selects from, and ``keep_dim=True`` keeps it
with length 1; a stream always has a dim, so selecting from a vector gives a
scalar ``(1,)`` either way. ``TimeSelect`` keeps the time axis with length 1,
one sample, and ``keep_dim=False`` drops it. Dropped from the only time axis,
the sample is a stream without time, which combines with every sample of a
window: ``w - TimeSelect(-1, keep_dim=False)(w)`` subtracts the newest sample
from each. A range always keeps its axis.

Some examples:

.. code-block:: python

   from nnodely import (
       EquationLearner, Fir, Fuzzify, Interpolation, Linear, LocalModel, ReLU,
       Select, TimeSelect, Triangular,
   )

   # a gain scheduled by the gear: one FIR filter per gear, blended
   torque = Input("torque")
   gear_membership = Fuzzify(centers=[1.0, 2.0, 3.0, 4.0], function=Triangular)([gear.last()])
   engine_force = LocalModel(lambda cell: Fir(features=1))([torque.sw(10)], [gear_membership])

   # a lookup table
   rolling = Interpolation(x_points=[0.0, 10.0, 30.0], y_points=[0.0, 0.2, 0.3])([velocity.last()])

   # slope force from the three altitude features
   slope = Linear(features=1)([altitude.last()])
   first_altitude = Select(idx=0)([altitude.last()])

   # symbolic regression over velocity: sin(a), a * b, relu(c)
   learner = EquationLearner(["sin", "multiply", "relu"], linear_out=Linear(features=1))
   learned = learner([velocity.last()])

   # the oldest sample of a window
   oldest = TimeSelect(idx=0)([velocity.sw(5)])

``Fuzzify`` places a membership function on each center: a
:class:`~nnodely.Triangular`, :class:`~nnodely.Rectangular` or
:class:`~nnodely.Gaussian` (the default is ``Triangular``), sized by the
centers next to it - the triangles reach the neighboring centers and add up to
one between them. Its ``function`` can also be any other layer class, made
anew for each center, or a layer instance, shared by all of them: they read
``x - center``, so ``Fuzzify(centers, function=Sin)`` gives ``sin(x - c)``. A
list gives one function per center, such as
``[Rectangular, Triangular, Gaussian]``. A custom function is a Keras layer
whose ``call`` reads one argument, ``x - center``, evaluated on every center
like any layer and saved and exported with the model; a ``call`` that needs
more arguments is refused:

.. code-block:: python

   import keras

   class Square(keras.layers.Layer):
       def call(self, x):
           return x * x

   squares = Fuzzify([0.0, 0.5, 1.0], function=Square)([velocity.last()])
   # (velocity - c)^2 on every center c

To load such a model in another session, define the class first, decorated
with ``@keras.saving.register_keras_serializable()``.

The centers can also move with the model. A :class:`~nnodely.Parameter`
vector of shape ``(n,)`` trains them; a stream computed from the inputs, such
as a ``Fir`` with ``features=n``, gives every sample its own; a list mixes
scalar streams with fixed numbers. They are used in the order given, which
should be ascending, and the memberships are placed on them and sized by their
neighbors as they are computed, so the gradient reaches them (a
``Rectangular`` has none):

.. code-block:: python

   centers = Parameter("centers", value=[0.0, 0.5, 1.0])
   trained = Fuzzify(centers)([velocity.last()])   # where the triangles sit, learned
   middle = Fuzzify([0.0, Parameter("middle", value=[0.5]), 1.0])([velocity.last()])

The same wrapping works anywhere in a model: :class:`~nnodely.KerasFunction`
makes any Keras layer an nnodely layer, its streams passed to ``call`` as
separate arguments and its output shape read off the result:

.. code-block:: python

   from nnodely import KerasFunction

   class Norm(keras.layers.Layer):
       def call(self, x):
           return keras.ops.sqrt(keras.ops.sum(x * x, axis=-1, keepdims=True))

   class Shifted(keras.layers.Layer):
       def call(self, x, y):
           return x * x + y

   length = KerasFunction(Norm)([altitude.last()])                       # a class, made here
   shifted = KerasFunction(Shifted())([velocity.last(), torque.last()])  # an instance

Weights of such a layer are Keras weights, trained with the model, but not
:class:`~nnodely.Parameter` objects: freeze them with the layer's
``trainable``. The membership functions are layers of
their own too: ``Triangular(center=0.5, radius=0.5)([velocity.last()])``.

``LocalModel`` takes the inputs and a list of activations. Several activations
are multiplied together, so memberships over 4 gears and 5 speeds give 20 local
models, cell ``(i, j)`` being number ``i * 5 + j``. Its ``input_function`` and
``output_function`` are optional: without them the input is only weighted by
the memberships and the cells summed. A weightless elementwise layer such as
``Sin()`` or ``ReLU()`` is applied once, any other layer such as ``Fir`` is
applied to every cell with the same weights, and a plain function is a factory
that receives the cell index ``(i, j)`` and returns the layer of that cell, with
weights of its own. With ``vectorized=True`` a function or a layer such as
``Fir(features=20)`` is called once on all the cells instead, and returns them on a last
dim axis of 20 entries. Functions passed to ``EquationLearner`` can be names,
*nnodely* layer classes, or Python callables that take streams, such as
``lambda a, b: a * b``.

The model
---------

:class:`~nnodely.Modely` collects inputs and outputs into a model.
:meth:`~nnodely.Modely.build` creates the Keras model and its weights:

.. code-block:: python

   acceleration = Output("acceleration", air_drag + engine_force + slope + rolling)

   vehicle = Modely(
       "vehicle",
       inputs=[velocity, torque, gear, altitude],
       outputs=[acceleration],
   )
   vehicle.build()

Objectives (see :doc:`training`) and feedback (see :doc:`recurrent`) are
declared **before** ``build()``, because they add to the graph that gets built.

Inference
---------

A built model is called with a dictionary that maps input names to arrays of
shape ``(batch, *time, *dim)``, batch axis included. Inputs only the
minimizers read, such as targets, are ignored, so the arrays of a
:class:`~nnodely.DataLoader` - ``as_dict()`` or ``get_samples(n)`` - can be
passed as they are. The result maps output names to tensors of the active
backend.

The time axes of an input's array must cover the union of the windows the
model reads from it, which the input's ``shape`` reports once the model is
built. Here the model reads ``velocity`` through ``last()`` alone - the
``sw(5)`` of ``oldest`` above is not part of it, so it widens nothing - and
``torque`` through ``sw(10)``:

.. code-block:: python

   print(velocity.shape, torque.shape)   # Shape(T=(1,), D=(1,)) Shape(T=(10,), D=(1,))

   sample = {
       "velocity": np.full((1, 1, 1), 15.0),
       "torque": np.ones((1, 10, 1)),
       "gear": np.full((1, 1, 1), 2.0),
       "altitude": np.zeros((1, 1, 3)),
   }
   result = vehicle(sample)
   print(result["acceleration"].shape)   # (1, 1, 1)

The model has to be called with every input its outputs read, listed in
``model.inference_inputs``; inputs used only as targets can be left out.

Weights and the Keras model
---------------------------

After ``build()``, layers with weights expose them as Keras variables, and
``Modely.model`` is the underlying ``keras.Model``:

.. code-block:: python

   print(slope.kernel.shape, slope.bias.shape)
   slope.kernel.assign(np.zeros((3, 1), dtype="float32"))
   print(len(vehicle.model.trainable_variables))

Inspecting a model
------------------

.. code-block:: python

   print(vehicle)            # the nodes of the graph, in evaluation order
   vehicle.summary()         # the Keras summary of the built model

   vehicle.export_html("model_html")             # interactive graph, one page per sub-model
   vehicle.plot("figures")                       # figures/vehicle.png, one image per sub-model
   vehicle.plot("figures", "all", flatten=True)  # figures/all.png, everything in one image

:meth:`~nnodely.Modely.export_html` writes an interactive page you can open in
a browser. :meth:`~nnodely.Modely.plot` draws static images with Graphviz,
whose ``dot`` program must be installed on the system. Like ``export_html`` it
takes a folder and a file name, the model's name by default; a path with an
image suffix (``"model.svg"``) names the file and its format.

Each node shows its name, coloured by its role - inputs green, outputs red,
parameters, constants, sub-models and losses as in the legend - and each arrow
the shape it carries, a dynamic time axis written ``dyn``. Without
``flatten``, every block that wraps a model - a sub-model call, a
:class:`~nnodely.Loop`, a :class:`~nnodely.Roll` - is one sub-model node, and
its body is drawn in an image of its own named ``<filename>_<body name>``,
recursively. With ``flatten=True`` the whole model is one image, each called
sub-model inside a box. A Loop or a Roll is shown by its arrow alone: an orange
``loop`` arrow, or a teal ``roll (steps=n)`` one, back to the input it feeds.
