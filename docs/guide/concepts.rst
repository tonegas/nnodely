Core Concepts
=============

A symbolic graph
----------------

A *nnodely* model is a graph of **streams**. An :class:`~nnodely.Input` is a
stream read from data. Applying a layer or an arithmetic operator to streams
returns a new stream, and an :class:`~nnodely.Output` gives a stream a public
name. Nothing is computed while the graph is declared.
:meth:`Modely.build() <nnodely.Modely.build>` turns the graph into a Keras
model, and only then are the weights created.

.. code-block:: python

   from nnodely import Input, Linear, Modely, Output, ReLU

   x = Input("x")
   hidden = ReLU()([Linear(features=8)([x])])
   y = Output("y", Linear(features=1)([hidden]))

   model = Modely("regressor", inputs=[x], outputs=[y])
   model.build()

Shapes
------

Every stream has a shape made of three parts, laid out after the batch axis
as

``(batch, *seq, time, *dim)``

- **seq** (``S``): the sequence axes, outermost first, possibly none: the
  steps of a trajectory, along which a :class:`~nnodely.Loop` rolls out (see
  :doc:`recurrent`).
- **time** (``T``): the window, the consecutive samples the stream carries.
  Every stream has one: a number, a :class:`~nnodely.Parameter` declared
  without time and an input read directly are one sample, ``T=1``.
- **dim** (``D``): the feature axes. ``Input("u", dim=3)`` is a
  three-component signal, and ``dim=(3, 2)`` is a 3x2 matrix per sample. The
  default is one feature.

.. code-block:: python

   u = Input("u", dim=3)
   print(u.shape)          # Shape(S=(), T=1, D=(3,)): the current sample
   print(u.sw(10).shape)   # Shape(S=(), T=10, D=(3,)): 10 samples of 3 features
   print(u.dim, u.sw(10).time, u.sw(10).seq)

   w = Input("w", dim=3, seq=5)
   print(w.sw(10).shape)   # Shape(S=(5,), T=10, D=(3,)): 5 steps of those

Temporal operations - windows, :class:`~nnodely.Fir`, derivatives and
integrals - act on the window; a :class:`~nnodely.Loop` rolls out along the
outermost sequence axis.

Windows
-------

An input selects the samples a relation reads through windows. Windows are
counted in **samples**, not seconds: the sampling time only appears where the
physics needs it, as the ``dt`` of :doc:`derivatives and integrators <physics>`.

``x.sw(n)``
    the last ``n`` samples, up to and including the current one.
``x.sw([p, f])``
    ``p`` samples up to and including the current one, followed by the next
    ``f`` samples.
``x``
    the input read directly: the current sample, the same as ``x.sw(1)``.
``x.next()``
    the next sample, the same as ``x.sw([0, 1])``.

.. code-block:: python

   x = Input("x")
   past = x.sw(4)          # x[t-3], x[t-2], x[t-1], x[t]
   around = x.sw([2, 1])   # x[t-1], x[t], x[t+1]
   print(past.shape, around.shape)   # Shape(S=(), T=4, D=(1,)) Shape(S=(), T=3, D=(1,))

An input can be windowed several times, and read directly as well. Requesting
a window leaves the input as it is: ``build()`` finds the windows the model's
graph reads, and the data a model needs from an input is their union - a
window declared but never used widens nothing, and two models of one input
read their own windows. The :class:`~nnodely.DataLoader` builds samples that
cover it, and the input's ``shape`` reports it once the model is built.
Reading the input directly still gives its current sample, however wide the
data, in a relation declared before the build or after it. All inputs are
aligned on the current sample ``t``.

``seq=`` declares the input's sequence: with
``Input("w", seq=5)``, ``w`` is five consecutive samples and ``w.sw(3)`` five
consecutive windows of three samples; ``seq=(4, 5)`` is four sequences of five
steps (outermost first, ``-1`` for a length that follows the data). Every
window of the input keeps its sequence.

Layers
------

A layer is configured when it is created and applied by calling it on a stream
or a list of streams:

.. code-block:: python

   from nnodely import Fir

   filtered = Fir(features=1)([x.sw(10)])

Calling the **same layer object** twice applies the same weights twice:

.. code-block:: python

   shared = Linear(features=1)
   a = shared([x])
   b = shared([x * 2.0])   # same kernel and bias as a

Once the model is built, the stream returned by a layer with weights exposes
them as Keras variables (``kernel``, ``bias``), which can be read or assigned.
Here that is ``a.kernel``. The layer object ``shared`` itself only holds the
configuration.

Operators
---------

Streams support ``+``, ``-``, ``*``, ``/`` and ``**``, with other streams or
with numbers. A number becomes a :class:`~nnodely.Constant` of one sample:

.. code-block:: python

   from nnodely import Parameter

   gain = Parameter("gain", value=2.0)
   z = gain * x + 1.0

Two streams combine when their shapes agree:

- **sequence and time**: streams must have the same sequence and window;
  different ones are never blended. A single sample - every sequence axis and
  the window of length 1: a number, a Parameter, an input read directly -
  applies to every sample of a window or a rollout.
- **dim**: the dims must be the same, or one of them a scalar, which applies
  to every feature. A vector is never stretched over a matrix.
- one of the streams must already have the shape of the result: a matrix of
  one sample is not combined with the window of a scalar, which would have to
  stretch both.

.. code-block:: python

   v = Input("v", dim=3)
   v * gain                 # S=(), T=1, D=(3,): the gain applies to every feature
   v.sw(4) + x.sw(4)        # S=(), T=4, D=(3,): the scalar window, to every feature
   v.sw(4) + x              # S=(), T=4, D=(3,): the current sample, to every sample
   # v.sw(4) + x.sw(3) is refused: windows of 4 and of 3 samples

Names
-----

Every node has a name. Layers get one automatically (``Linear1``,
``Fir2``, ...) unless ``name=`` is given. Inputs and outputs are addressed by
name everywhere else: in the data dictionaries, the results of inference and
the training objectives. They must be unique within a model.

A layer's name does not decide its weights, the layer object does: two layers
created apart never share weights, even under the same name - a model and its
reloaded copy, for instance. When two such layers meet in one model, the Keras
layer of the second takes a suffix (``Fir1_1``), since Keras names the layers
of a model uniquely.
