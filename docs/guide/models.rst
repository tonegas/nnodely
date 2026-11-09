Building Models
===============

This page lists the blocks a model is made of. Their full signatures are in the
:doc:`API reference <../api/index>`.

Inputs and outputs
------------------

:class:`~nnodely.Input` declares a signal read from data, with the window of
samples the model reads (``tw``, see :doc:`concepts`), and
:class:`~nnodely.Output` names a stream the model exposes:

.. code-block:: python

   import numpy as np
   from nnodely import Input, Modely, Output

   velocity = Input("velocity", tw=1)              # the current sample
   altitude = Input("altitude", dim=3, tw=1)       # three features per sample
   gear = Input("gear", tw=1)

Parameters and constants
------------------------

A :class:`~nnodely.Parameter` is a trainable tensor used directly in the graph,
and a :class:`~nnodely.Constant` is a fixed one. Both declare their shape as an
Input does, with ``tw`` for the window: ``dim`` the feature axes, one
feature by default, each at least 1; ``tw`` the number of samples of the
window - without it the value has none and applies to every sample, of any
stream; ``seq`` the sequence axes, outermost first, which need a window. Like
every stream they are laid out ``(batch, *seq, tw, *dim)`` - without the
window when they have none - the same value for every sample.

A constant requires a ``value``; a parameter without one starts from a draw of
its ``initializer``. A ``value`` fills the declared shape: a list of numbers
holds exactly as many as the shape, read in the order ``(*seq, tw, *dim)``;
an array of several axes has the declared sizes, in any order, and its axes
are moved into place. Anything else raises ``ValueError``. Without ``dim``,
``seq`` and ``tw`` take the axes of the value of their sizes, and the axes
they leave are its ``dim``, in their order - all of them, without a window,
when neither is given:

.. code-block:: python

   from nnodely import Constant, Parameter

   drag = Parameter("drag", value=0.1)                     # trainable scalar
   weights = Parameter("weights", dim=3)                   # random initial value
   gravity = Constant("gravity", value=9.81)
   offsets = Constant("offsets", value=[1.0, 2.0, 3.0])    # dim (3,), no window
   # A window of 2 samples of 3 features, from a list or from a matrix:
   ramp = Parameter("ramp", value=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], tw=2, dim=3)
   table = Constant("table", value=np.ones((3, 2)), tw=2, dim=3)  # moved to (2, 3)
   grid = Constant("grid", value=np.ones((2, 4, 6)), tw=4)  # the dim is (2, 6)
   # A sequence of 4 windows of 2 samples, drawn from the initializer:
   path = Parameter("path", seq=4, tw=2, dim=3, initializer="zeros")

   air_drag = -1.0 * drag * velocity * velocity

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
       functions whose results are combined. A ``threshold`` drops the small
       coefficients, and ``plot()`` draws the learned equation.
   * - :class:`~nnodely.BatchNorm`
     - Batch normalization per feature, with a learned scale and offset.
   * - :class:`~nnodely.PReLU`
     - ReLU with a learned slope below zero, one per feature.

.. list-table:: Layers without weights
   :header-rows: 1
   :widths: 25 75

   * - Layer
     - What it does
   * - :class:`~nnodely.Fuzzify`
     - Memberships of a scalar signal to sets placed on given centers, one
       feature per center: a membership function on each, or any layer of
       ``x - center``; ``normalize=True`` makes them add up to one.
   * - Membership functions
     - :class:`~nnodely.Triangular`, :class:`~nnodely.Rectangular`,
       :class:`~nnodely.Gaussian`: layers of their own, 1 at 0, applied to
       every value of a stream; their width can be a Parameter.
   * - :class:`~nnodely.Interpolation`
     - One-dimensional lookup table, linear or polynomial.
   * - Activations
     - :class:`~nnodely.ReLU`, :class:`~nnodely.LeakyReLU`, :class:`~nnodely.ELU`,
       :class:`~nnodely.Sigmoid`, :class:`~nnodely.Tanh`,
       :class:`~nnodely.Softmax`, :class:`~nnodely.Swish`, :class:`~nnodely.GELU`,
       :class:`~nnodely.Softplus`.
   * - Math
     - :class:`~nnodely.Sin`, :class:`~nnodely.Cos`, :class:`~nnodely.Tan`,
       :class:`~nnodely.Asin`, :class:`~nnodely.Acos`, :class:`~nnodely.Atan`,
       :class:`~nnodely.Atan2`, :class:`~nnodely.Sinh`, :class:`~nnodely.Cosh`,
       :class:`~nnodely.Asinh`, :class:`~nnodely.Acosh`, :class:`~nnodely.Atanh`,
       :class:`~nnodely.Exp`, :class:`~nnodely.Expm1`, :class:`~nnodely.Log`,
       :class:`~nnodely.Log1p`, :class:`~nnodely.Log10`, :class:`~nnodely.Log2`,
       :class:`~nnodely.Sqrt`, :class:`~nnodely.Square`, :class:`~nnodely.Abs`,
       :class:`~nnodely.Sign`, :class:`~nnodely.Floor`, :class:`~nnodely.Ceil`,
       :class:`~nnodely.Round`, :class:`~nnodely.Deg2Rad`,
       :class:`~nnodely.Rad2Deg`, :class:`~nnodely.Negative`,
       :class:`~nnodely.Clamp`, :class:`~nnodely.Maximum`,
       :class:`~nnodely.Minimum`.
   * - Reductions
     - :class:`~nnodely.Sum`, :class:`~nnodely.Mean`, :class:`~nnodely.Max`,
       :class:`~nnodely.Min`, :class:`~nnodely.Norm` over the dim axes, kept
       with length 1; :class:`~nnodely.Cross`, the cross product of
       3-vectors.
   * - Feature axes
     - :class:`~nnodely.Select` (one index), :class:`~nnodely.Range` (a slice),
       :class:`~nnodely.Concatenate`. ``axis`` counts the dim axes from 0.
   * - Window
     - :class:`~nnodely.TimeSelect` (one sample), :class:`~nnodely.TimeRange`
       (a slice), :class:`~nnodely.TimeConcatenate`,
       :class:`~nnodely.TimeReverse`. They act on the window, ``T``, and
       always keep it.
   * - Sequence
     - :class:`~nnodely.SeqSelect` (one step), :class:`~nnodely.SeqRange`
       (a slice), :class:`~nnodely.SeqConcatenate`,
       :class:`~nnodely.SeqStack` (a new axis), :class:`~nnodely.SeqReverse`.
       ``axis`` counts the sequence axes outermost first: the default, 0, is
       the outermost one, the rollout of the outermost Loop.
   * - Calculus
     - :class:`~nnodely.Gradient`, :class:`~nnodely.DerivativeStep`,
       :class:`~nnodely.Derivative`,
       :class:`~nnodely.Integrate`, :class:`~nnodely.IntegrateStep`,
       :func:`~nnodely.Ode` (see :doc:`physics`).
   * - Recurrence
     - :class:`~nnodely.Loop` (see :doc:`recurrent`).

``Select`` drops the dim axis it selects from, and ``keep_dim=True`` keeps it
with length 1; a stream always has a dim, so selecting from a vector gives a
scalar ``(1,)`` either way. ``TimeSelect`` keeps the window, one sample long,
which combines with every sample of a window: ``w - TimeSelect(-1)(w)``
subtracts the newest sample from each. ``SeqSelect`` removes the sequence
axis it selects from, as does a ``SeqRange`` one step long: ``SeqSelect(-1)``
takes the last step of a rollout, of any length.

Some examples:

.. code-block:: python

   from nnodely import (
       EquationLearner, Fir, Fuzzify, Interpolation, Linear, LocalModel, ReLU,
       Select, TimeSelect, Triangular,
   )

   # a gain scheduled by the gear: one FIR filter per gear, blended
   torque = Input("torque", tw=10)
   gear_membership = Fuzzify(centers=[1.0, 2.0, 3.0, 4.0], function=Triangular)([gear])
   engine_force = LocalModel(lambda cell: Fir(features=1))([torque], [gear_membership])

   # a lookup table
   rolling = Interpolation(x_points=[0.0, 10.0, 30.0], y_points=[0.0, 0.2, 0.3])([velocity])

   # slope force from the three altitude features
   slope = Linear(features=1)([altitude])
   first_altitude = Select(idx=0)([altitude])

   # symbolic regression over velocity: sin(a), a * b, relu(c)
   learner = EquationLearner(["sin", "multiply", "relu"], linear_out=Linear(features=1))
   learned = learner([velocity])

   # the oldest sample of the torque window
   oldest = TimeSelect(idx=0)([torque])

``Fuzzify`` places a membership function on each center: a
:class:`~nnodely.Triangular`, :class:`~nnodely.Rectangular` or
:class:`~nnodely.Gaussian`, given as a class or by name, sized by the centers
next to it - the triangles reach the neighboring centers and add up to one
between them, the rectangles reach halfway to them, and a Gaussian's sigma is
a sixth of the span between them. Its ``function`` can also be any other
layer class, made anew for each center, or a layer instance, shared by all of
them: they read ``x - center``, so ``Fuzzify(centers, function=Sin)`` gives
``sin(x - c)``, and ``function=Triangular(radius=0.2)`` puts a triangle of
that radius on every center. A ``Fuzzify`` is one layer, which computes all
its memberships at once. A list gives one function per center, such as
``[Rectangular, Triangular, Gaussian]``. A custom function is a Keras layer
whose ``call`` reads one argument, ``x - center``, evaluated on every center
like any layer and saved and exported with the model; a ``call`` that needs
more arguments is refused:

.. code-block:: python

   import keras

   class Square(keras.layers.Layer):
       def call(self, x):
           return x * x

   squares = Fuzzify([0.0, 0.5, 1.0], function=Square)([velocity])
   # (velocity - c)^2 on every center c

To load such a model in another session, define the class first, decorated
with ``@keras.saving.register_keras_serializable()``.

The centers can also move with the model. A :class:`~nnodely.Parameter`
vector of shape ``(n,)`` trains them; a stream computed from the inputs, such
as a ``Fir`` with ``features=n``, gives every sample its own; a list mixes
scalar streams with fixed numbers. They are used in the order given, which
should be ascending, and the memberships are placed on them and sized by their
neighbors as they are computed: the sets move with the centers, their widths
included, and the gradient reaches the centers through both. A step has none,
so centers read by rectangles alone do not train:

.. code-block:: python

   centers = Parameter("centers", value=[0.0, 0.5, 1.0])
   trained = Fuzzify(centers, function=Triangular)([velocity])   # learned
   middle = Parameter("middle", value=[0.5])
   listed = Fuzzify([0.0, middle, 1.0], function=Triangular)([velocity])

The memberships are the values of the functions. ``normalize=True`` divides
those of a sample by their sum, so that they add up to one, and a sample in no
set stays all zeros; it is meant for memberships that are not negative, unlike
a ``Sin`` or a custom function. Beyond the outer centers a sample is in no
triangle or rectangle, and only in the vanishing tail of a Gaussian.
``bounds=True`` holds the signal between the lowest and the highest center -
centers that move as they are computed - so that it is in the outer set
instead, and ``bounds=(left, right)`` holds it between two numbers:

.. code-block:: python

   from nnodely import Gaussian, Sin

   regimes = Fuzzify([0.0, 0.5, 1.0], function=Triangular, bounds=True)([velocity])
   regimes = Fuzzify([0.0, 0.5, 1.0], function=Triangular, bounds=True)([velocity])
   # velocity 3.0 reads 1.0: [0, 0, 1]; velocity -2.0 reads 0.0: [1, 0, 0]
   shares = Fuzzify([0.0, 0.5, 1.0], function=Gaussian, normalize=True)([velocity])
   waves = Fuzzify([0.0, 0.5, 1.0], function=Sin)([velocity])   # sin(velocity - c)

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

   length = KerasFunction(Norm)([altitude])                 # a class, made here
   shifted = KerasFunction(Shifted())([velocity, torque])   # an instance

Weights of such a layer are Keras weights, trained with the model, but not
:class:`~nnodely.Parameter` objects: freeze them with the layer's
``trainable``.

The membership functions are layers of their own too, 1 at 0: the signal is
shifted to center them. A radius - a number, or a ``(left, right)`` pair - or
a Gaussian's sigma can be a :class:`~nnodely.Parameter`, trained with the
model, in a ``Fuzzify`` as well:

.. code-block:: python

   radius = Parameter("radius", value=[0.5])
   bump = Triangular(radius=radius)([velocity - 0.5])   # 0 at 0 and at 1
   shaped = Fuzzify([0.0, 0.5, 1.0], function=Triangular(radius=radius))([velocity])

A width trained to zero or below is held at 1e-7: its set collapses, and no
NaN appears. A rectangle has no gradient, being a step: a radius Parameter
read by rectangles alone does not train, one read elsewhere too trains there
and the rectangles follow it.

``plot()`` draws a ``Fuzzify`` with matplotlib as it computes its memberships
now - its Parameters at their current values, so after training the trained
centers and widths - over the centers and the outer gap beyond each end, or
over ``x_range``. It draws on ``ax`` if given and returns the axes. It works
on a ``Fuzzify`` not applied yet, and on the stream it gives; centers computed
from the inputs have no value to draw, and raise ``ValueError``:

.. code-block:: python

   from nnodely import Gaussian

   Fuzzify([0.0, 0.5, 1.0], function=Gaussian).plot()
   trained.plot(x_range=(-0.5, 1.5))

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
dim axis of 20 entries. Each call of a ``LocalModel`` declares its cells in a
model of their own and calls it as a block, which ``export_html`` shows on a
page of its own. Functions passed to ``EquationLearner`` can be names,
*nnodely* layer classes, or Python callables that take streams, such as
``lambda a, b: a * b``.

An ``EquationLearner`` with a ``threshold`` computes without the coefficients
of its projections smaller than it, so that the learned equation has no terms
such as ``0.000001 * sin(x)``; they keep training, and count again if they
grow past it. The threshold can be set on a built model, which computes and
trains with it from then on. Train without it first, then set it and
fine-tune, and draw the equation the model computes with ``plot()``, which
returns it as a sympy ``Eq``:

.. code-block:: python

   model.train(data, epochs=150, lr=0.02)
   learner.threshold = 0.01
   model.train(data, epochs=50, lr=0.002)
   equation = learner.plot()   # y = ..., written in velocity

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
Building again, after a new objective, keeps the weights the model holds;
``build(keep_weights=False)`` gives every one its initial value again, those
of the models it calls and of its Loop bodies included. A build that fails
leaves the model as it was.

Inference
---------

A built model is called with a dictionary that maps each input - its name or
the Input itself - to an array of shape ``(batch, *seq, tw, *dim)``, batch
axis included. Inputs only the minimizers read, such as targets, and keys that
name no input are ignored, so the arrays of a :class:`~nnodely.DataLoader` -
``as_dict()`` or ``get_samples(n)`` - can be passed as they are. A list works
too: one array per input the model was declared with, in the order of
``inputs``, and a model with one input takes its array alone. These are the forms that make the model a block of a larger one
when it is called on streams (see :doc:`composition`). The result maps output
names to tensors of the active backend.

The array of an input has the shape the input is declared with, which its
``shape`` reports: here one sample of ``velocity`` and the ten samples of
``torque``:

.. code-block:: python

   print(velocity.shape, torque.shape)   # Shape(S=(), T=1, D=(1,)) Shape(S=(), T=10, D=(1,))

   sample = {
       "velocity": np.full((1, 1, 1), 15.0),
       "torque": np.ones((1, 10, 1)),
       "gear": np.full((1, 1, 1), 2.0),
       "altitude": np.zeros((1, 1, 3)),
   }
   result = vehicle(sample)
   print(result["acceleration"].shape)   # (1, 1, 1)

The model has to be called with every input its outputs read, listed in
``model.inference_inputs``; inputs used only as targets can be left out of a
dictionary.

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

   vehicle.export_html("model_html")   # interactive graph, one page per sub-model

:meth:`~nnodely.Modely.export_html` writes an interactive page you can open in
a browser. Every node is a card with its name and what it is, coloured by its
role - inputs green, the states a loop feeds back teal, outputs rose,
parameters amber, constants slate, sub-models indigo, losses violet - and the
legend under the graph, which highlights a kind of node when clicked, names
them all. Clicking a node shows what it is, its shape axis by axis and what it
reads and feeds, and highlights everything it depends on or feeds; a
sub-model opens on a page of its own with a double-click. The flattened page
splices every sub-model in, its body inside a shaded region. The graph is laid
out left to right, or by a force simulation (``physics``).
