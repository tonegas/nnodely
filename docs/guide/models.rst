Building a Model
================

At the end of this page you will be able to declare a model from inputs,
parameters and layers, build it, run it on arrays, and inspect it.

**Prerequisites:** :doc:`concepts`.

**Tutorials:** :doc:`../tutorial/01_model_design`,
:doc:`../tutorial/06_parameter_constant`, :doc:`../tutorial/08_fir_linear`,
:doc:`../tutorial/09_fuzzify_localmodel`, :doc:`../tutorial/12_custom_layers`,
:doc:`../tutorial/13_equation_learner`.

**The example.** The longitudinal dynamics of a vehicle. Its acceleration is
the sum of an air drag, which grows with the square of the speed, an engine
force that depends on the gear, a rolling resistance read from a table, and a
slope force:

.. math::

   a = -c_d \, v^2 + F_{engine}(\tau, gear) + F_{roll}(v) + F_{slope}(h)

Step 1: inputs
--------------

:class:`~nnodely.Input` declares a signal read from data, with the window of
samples the model reads (``tw``, see :doc:`concepts`):

.. code-block:: python

   import numpy as np
   from nnodely import Input, Modely, Output

   velocity = Input("velocity", tw=1)          # the current sample
   torque = Input("torque", tw=10)             # the last 10 samples
   gear = Input("gear", tw=1)
   altitude = Input("altitude", dim=3, tw=1)   # three features per sample
   print(velocity.shape, torque.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=1, D=(1,)) Shape(S=(), T=10, D=(1,))

Step 2: parameters and constants
--------------------------------

A :class:`~nnodely.Parameter` is a value that training changes, and a
:class:`~nnodely.Constant` one it does not. Both are streams, and declare
their shape as an Input does:

- ``dim``: the features, one by default;
- ``tw``: the samples of a window. Without it the value has no window, and
  applies to every sample of the streams it is combined with;
- ``seq``: the sequence axes, outermost first, which need a window.

A Constant needs a ``value``. A Parameter without one starts from a draw of
its ``initializer``. Without ``dim``, ``tw`` and ``seq``, a value gives its
own shape: all its axes are features, without a window. The reference page
of :class:`~nnodely.Parameter` lists every way a value fills a declared
shape.

.. code-block:: python

   from nnodely import Constant, Parameter

   drag = Parameter("drag", value=0.1)                    # trainable scalar
   weights = Parameter("weights", dim=3)                  # random initial value
   gravity = Constant("gravity", value=9.81)
   offsets = Constant("offsets", value=[1.0, 2.0, 3.0])   # three features
   print(drag.shape, offsets.shape)

   air_drag = -drag * velocity * velocity

.. code-block:: text
   :class: output

   Shape(S=(), T=None, D=(1,)) Shape(S=(), T=None, D=(3,))

Step 3: layers
--------------

Every layer is created with its settings, then called on streams (see
:doc:`concepts`). The layers are grouped in the :doc:`API reference
<../api/layers>`:

- **with weights**: :class:`~nnodely.Linear`, :class:`~nnodely.Fir`,
  :class:`~nnodely.LocalModel`, :class:`~nnodely.EquationLearner`,
  :class:`~nnodely.BatchNorm`;
- **fuzzy sets and lookup tables**: :class:`~nnodely.Fuzzify`,
  :class:`~nnodely.Triangular`, :class:`~nnodely.Rectangular`,
  :class:`~nnodely.Gaussian`, :class:`~nnodely.Interpolation`;
- **custom layers**: :class:`~nnodely.KerasFunction`;
- **selecting and joining** the features, the window or the sequence, such
  as :class:`~nnodely.Select`, :class:`~nnodely.TimeSelect`,
  :class:`~nnodely.SeqSelect`;
- **reductions**, such as :class:`~nnodely.Sum` and :class:`~nnodely.Norm`;
- **calculus** and **recurrence** (see :doc:`physics` and
  :doc:`recurrent`);
- **activations**, such as :class:`~nnodely.ReLU` and
  :class:`~nnodely.PReLU`, the only one with weights;
- **math functions**, such as :class:`~nnodely.Sin` and
  :class:`~nnodely.Exp`.

The engine force is scheduled by the gear: one FIR filter of the torque per
gear, blended by the membership of the gear to each of them. The rolling
resistance is a lookup table, and the slope force a linear map of the three
altitude features:

.. code-block:: python

   from nnodely import Fir, Fuzzify, Interpolation, Linear, LocalModel, Triangular

   gear_membership = Fuzzify(centers=[1.0, 2.0, 3.0, 4.0], function=Triangular)(gear)
   engine_force = LocalModel(lambda cell: Fir())(torque, gear_membership)
   rolling = Interpolation(x_points=[0.0, 10.0, 30.0], y_points=[0.0, 0.2, 0.3])(velocity)
   slope = Linear(features=1)(altitude)
   print(gear_membership.shape, engine_force.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=1, D=(4,)) Shape(S=(), T=1, D=(1,))

:class:`~nnodely.Fuzzify` gives one membership per center: a triangle on
each center that reaches its neighbors. :class:`~nnodely.LocalModel` gives
each membership its own model, here a :class:`~nnodely.Fir` created by the
function for every cell, and sums them weighted by the memberships.

Step 4: custom layers
---------------------

:class:`~nnodely.KerasFunction` makes any Keras layer a layer of the graph.
Its streams are passed to ``call`` as separate arguments, and its output
shape is read from the result:

.. code-block:: python

   import keras
   from nnodely import KerasFunction

   class VectorLength(keras.layers.Layer):
       def call(self, x):
           return keras.ops.sqrt(keras.ops.sum(x * x, axis=-1, keepdims=True))

   class SquarePlus(keras.layers.Layer):
       def call(self, x, y):
           return x * x + y

   length = KerasFunction(VectorLength)(altitude)                 # a class
   combined = KerasFunction(SquarePlus())([velocity, gear])       # an instance
   print(length.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=1, D=(1,))

The weights of such a layer are Keras weights, trained with the model, but
they are not :class:`~nnodely.Parameter` objects: freeze them with the
``trainable`` attribute of the Keras layer. To load a model that uses such a
class in another session, define the class first, decorated with
``@keras.saving.register_keras_serializable()``.

Step 5: the model
-----------------

:class:`~nnodely.Modely` collects inputs and outputs into a model, and
:meth:`~nnodely.Modely.build` creates its Keras model and its weights:

.. code-block:: python

   acceleration = Output("acceleration", air_drag + engine_force + rolling + slope)

   vehicle = Modely(
       "vehicle",
       inputs=[velocity, torque, gear, altitude],
       outputs=[acceleration],
   )
   vehicle.build()

A model declares every Input it reads among its ``inputs``. That includes an
Input it gives a model it calls, and an Input a layer starts from, as in
``Integrate(init=x0)``. An Input read without being declared is refused when
the model is declared. Two kinds of Input may be left out:

- a fed-back Input (see :doc:`recurrent`);
- an Input read only by an objective, such as a target (see
  :doc:`training`).

Declare objectives and feedback **before** ``build()``, because they add to
the graph. Building again keeps the weights the model holds, for example
after a new objective. ``build(keep_weights=False)`` gives every weight its
initial value again. A build that fails leaves the model as it was.

Step 6: running the model
-------------------------

Call a built model with a dictionary that maps each input, by name or by the
Input itself, to an array of shape ``(batch, *seq, tw, *dim)``. The array of
an input has the shape its ``shape`` reports, **with a batch axis first**:

.. code-block:: python

   sample = {
       "velocity": np.full((1, 1, 1), 15.0),
       "torque": np.ones((1, 10, 1)),
       "gear": np.full((1, 1, 1), 2.0),
       "altitude": np.zeros((1, 1, 3)),
   }
   result = vehicle(sample)
   values = keras.ops.convert_to_numpy(result["acceleration"])
   print(values.shape)

.. code-block:: text
   :class: output

   (1, 1, 1)

The result maps every output name to a tensor of the active backend.
``keras.ops.convert_to_numpy`` makes it a NumPy array.

Other forms of the call:

- a **list**, one array per declared input, in the order of ``inputs``;
- a **single array**, for a model with one input;
- the dictionaries of a :class:`~nnodely.DataLoader`, ``as_dict()`` or
  ``get_samples(n)``, as they are. Keys that name no input, such as
  targets, are ignored.

The call needs every input the outputs read.
``vehicle.inference_inputs`` lists them.

.. warning::

   An array without the batch axis is not refused when its shape can be
   read as a batch: a window of 10 samples, ``(10, 1)``, is read as 10
   samples of one value. The sample ``loader[0]`` of a DataLoader has no
   batch axis: call the model on ``loader.get_samples(1)``, or add the axis,
   ``{k: v[None] for k, v in loader[0].items()}``.

Step 7: weights and the Keras model
-----------------------------------

After ``build()``, the stream returned by a layer with weights exposes them
as Keras variables, and ``vehicle.model`` is the Keras model:

.. code-block:: python

   print(slope.kernel.shape, slope.bias.shape)
   slope.kernel.assign(np.zeros((3, 1), dtype="float32"))
   print(len(vehicle.parameters))

.. code-block:: text
   :class: output

   (3, 1) (1,)
   11

The weights are :class:`~nnodely.Parameter` objects:
:attr:`~nnodely.Modely.parameters` lists them, and :doc:`training` shows how
to freeze them.

Step 8: inspecting a model
--------------------------

Printing a model shows its name, its inputs, its outputs and its objectives.
:attr:`~nnodely.Modely.inputs`, :attr:`~nnodely.Modely.outputs` and
:attr:`~nnodely.Modely.parameters` print as tables, with more detail:

.. code-block:: python

   print(vehicle)

.. code-block:: text
   :class: output

   ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃ model      ┃ vehicle (built)                  ┃
   ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
   │ inputs     │ velocity, torque, gear, altitude │
   │ outputs    │ acceleration                     │
   │ parameters │ 11                               │
   │ minimizers │ -                                │
   └────────────┴──────────────────────────────────┘

.. code-block:: python

   vehicle.summary()                    # the Keras summary of the built model
   vehicle.export_html("model_html")    # an interactive graph

:meth:`~nnodely.Modely.export_html` writes an interactive page to open in a
browser. Every node is a card with its name and its kind, colored by its
role: inputs green, states teal, outputs rose, parameters amber, constants
slate, blocks indigo, losses violet. Clicking a node shows its shape and
highlights what it depends on and what it feeds. A block opens on a page of
its own with a double-click.

Step 9: learning an equation
----------------------------

An :class:`~nnodely.EquationLearner` learns a formula from a list of
elementary functions. A ``threshold`` set after a first training drops the
small coefficients, so that the equation keeps only its main terms, and
``plot()`` draws the equation and returns it as a SymPy ``Eq``. Here it
recovers :math:`f = \sin z + 0.5 z^2`:

.. code-block:: python

   from nnodely import DataLoader, EquationLearner, set_seed

   set_seed(0)
   z = Input("z", tw=1)
   learner = EquationLearner(["sin", "multiply"], linear_out=Linear(features=1))
   equation_model = Modely("equation", inputs=[z], outputs=[Output("f", learner(z))])
   equation_model.minimize("fit", "f", Input("f_measured", tw=1))
   equation_model.build()

   zs = np.linspace(-2.0, 2.0, 400)
   data = DataLoader(equation_model, source={"z": zs, "f_measured": np.sin(zs) + 0.5 * zs**2})
   equation_model.train(data, epochs=100, batch_size=32, lr=0.02, printer=None)
   learner.threshold = 0.01   # drop the coefficients smaller than 0.01 in magnitude
   equation_model.train(data, epochs=30, batch_size=32, lr=0.002, printer=None)
   equation = learner.plot()

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Error
     - Cause
     - Fix
   * - ``Model 'm' reads the Inputs ['x'], which are not among its inputs``
     - An Input the outputs read is not declared.
     - Add it to ``inputs=[...]``.
   * - ``expected shape=(None, 1, 1), found shape=(1, 3)``
     - An array without the shape the input declares.
     - Give every array the shape ``(batch, *seq, tw, *dim)`` of its input.
   * - A result with a wrong batch size, and no error
     - An array without the batch axis, such as ``loader[0]``.
     - Use ``loader.get_samples(1)``, or add the axis with ``v[None]``.
   * - ``Model m is not built``
     - A DataLoader or a call before ``build()``.
     - Call ``model.build()`` first.

See also
--------

:class:`~nnodely.Modely`, :class:`~nnodely.Input`, :class:`~nnodely.Output`,
:class:`~nnodely.Parameter`, :class:`~nnodely.Constant`,
:class:`~nnodely.Fuzzify`, :class:`~nnodely.LocalModel`,
:class:`~nnodely.Interpolation`, :class:`~nnodely.KerasFunction`,
:class:`~nnodely.EquationLearner`, :doc:`../api/layers`.
