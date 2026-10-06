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
and a :class:`~nnodely.Constant` is a fixed one. A ``value`` sets the shape: a
number or a vector is one time step of its ``dim``, a matrix is ``(dim, time)``
and further axes are ``seq`` axes. A constant always takes its shape from its
``value``; a parameter can be given ``dim``, ``time`` and ``seq`` instead, and
start from a random value. Like every stream they are laid out
``(batch, *dim, time, *seq)``, the same value for every sample:

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
``Sin``, ``Exp``, ``Sigmoid`` or ``TimeConcatenate`` do - can be applied as it
is created: ``Sin(x)`` is ``Sin()(x)``. To name it, configure it first:
``Sin(name="s")(x)``. The tables group the available layers.

.. list-table:: Layers with weights
   :header-rows: 1
   :widths: 25 75

   * - Layer
     - What it does
   * - :class:`~nnodely.Linear`
     - Affine map ``W x + b`` over the feature axis, applied to every sample.
   * - :class:`~nnodely.Fir`
     - Finite impulse response filter: a learned weighted sum over the time
       window, returning one sample.
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
     - Membership functions (triangular, rectangular or Gaussian) on given centers.
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
       :class:`~nnodely.Exp`, :class:`~nnodely.Log`, :class:`~nnodely.Log10`,
       :class:`~nnodely.Sqrt`, :class:`~nnodely.Abs`, :class:`~nnodely.Sign`,
       :class:`~nnodely.Floor`, :class:`~nnodely.Ceil`, :class:`~nnodely.Deg2Rad`,
       :class:`~nnodely.Negative`, :class:`~nnodely.Clamp`, :class:`~nnodely.Sum`.
   * - Feature axis
     - :class:`~nnodely.Select` (one index), :class:`~nnodely.Range` (a slice),
       :class:`~nnodely.Concatenate`.
   * - Time axis
     - :class:`~nnodely.TimeSelect`, :class:`~nnodely.TimeRange`,
       :class:`~nnodely.TimeConcatenate`.
   * - Calculus
     - :class:`~nnodely.Differentiate`, :class:`~nnodely.Derivative`,
       :class:`~nnodely.Integrate`, :class:`~nnodely.IntegrateStep`,
       :func:`~nnodely.Ode`, :class:`~nnodely.OdeNet` (see :doc:`physics`).
   * - Recurrence
     - :class:`~nnodely.Loop`, :class:`~nnodely.Roll` (see :doc:`recurrent`).

Some examples:

.. code-block:: python

   from nnodely import (
       EquationLearner, Fir, Fuzzify, Interpolation, Linear, LocalModel, ReLU,
       Select, TimeSelect,
   )

   # a gain scheduled by the gear: one FIR filter per gear, blended
   torque = Input("torque")
   gear_membership = Fuzzify(centers=[1.0, 2.0, 3.0, 4.0], function="Triangular")([gear.last()])
   engine_force = LocalModel(Fir(out_features=1))([torque.sw(10)], [gear_membership])

   # a lookup table
   rolling = Interpolation(x_points=[0.0, 10.0, 30.0], y_points=[0.0, 0.2, 0.3])([velocity.last()])

   # slope force from the three altitude features
   slope = Linear(out_features=1)([altitude.last()])
   first_altitude = Select(idx=0)([altitude.last()])

   # symbolic regression over velocity: sin(a), a * b, relu(c)
   learner = EquationLearner(["sin", "multiply", "relu"], linear_out=Linear(out_features=1))
   learned = learner([velocity.last()])

   # the oldest sample of a window
   oldest = TimeSelect(idx=0)([velocity.sw(5)])

``LocalModel`` takes the inputs and a list of activations. Several activations
are multiplied together, so memberships over 4 gears and 5 speeds give 20 local
models, cell ``(i, j)`` being number ``i * 5 + j``. Its ``input_function`` and
``output_function`` are a callable, instantiated anew for every cell, or a list
of one callable per cell; with ``pass_index=True`` a function receives the cell
index ``(i, j)`` and returns the callable of that cell. A ``Fir`` instance as
input function and a weightless elementwise layer such as ``ReLU()`` as output
function are evaluated for all the cells at once. Functions passed to ``EquationLearner`` can be names,
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
shape ``(batch, *dim, time, *seq)``, batch axis included. Inputs only the
minimizers read, such as targets, are ignored, so the arrays of a
:class:`~nnodely.DataLoader` - ``as_dict()`` or ``get_samples(n)`` - can be
passed as they are. The result maps output names to tensors of the active
backend.

The ``time`` axis of an input's array must cover the union of all its windows,
which the input's ``shape`` reports. Here ``velocity`` is read through
``sw(5)``, so it needs five samples:

.. code-block:: python

   print(velocity.shape, torque.shape)   # Shape(D=(1,), T=5, S=()) Shape(D=(1,), T=10, S=())

   sample = {
       "velocity": np.full((1, 1, 5), 15.0),
       "torque": np.ones((1, 1, 10)),
       "gear": np.full((1, 1, 1), 2.0),
       "altitude": np.zeros((1, 3, 1)),
   }
   result = vehicle(sample)
   print(result["acceleration"].shape)   # (1, 1, 1)

The model has to be called with every input of the built graph. When
objectives are declared, that includes the inputs used only as targets.

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
the shape it carries, a dynamic sequence axis written ``dyn``. Without
``flatten``, every block that wraps a model - a sub-model call, a
:class:`~nnodely.Loop`, a :class:`~nnodely.Roll` - is one sub-model node, and
its body is drawn in an image of its own named ``<filename>_<body name>``,
recursively. With ``flatten=True`` the whole model is one image, each called
sub-model inside a box. A Loop or a Roll is shown by its arrow alone: an orange
``loop`` arrow, or a teal ``roll (steps=n)`` one, back to the input it feeds.
