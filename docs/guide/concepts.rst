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
   hidden = ReLU()([Linear(features=8)([x.last()])])
   y = Output("y", Linear(features=1)([hidden]))

   model = Modely("regressor", inputs=[x], outputs=[y])
   model.build()

Shapes
------

Every stream has a shape made of two parts, laid out after the batch axis as

``(batch, *time, *dim)``

- **time**: the time axes, outermost first. The last one, the **inner time**,
  is the window of consecutive samples the stream carries; the ones before it,
  the **outer time**, are the steps of a rollout over a trajectory (see
  :doc:`recurrent`). A stream may have no time axis at all: a number, a
  :class:`~nnodely.Parameter` declared without time, an input read whole.
- **dim**: the feature axes. ``Input("u", dim=3)`` is a three-component signal,
  and ``dim=(3, 2)`` is a 3x2 matrix per sample. The default is one feature.

.. code-block:: python

   u = Input("u", dim=3)
   print(u.sw(10).shape)   # Shape(T=(10,), D=(3,)): 10 samples of 3 features
   print(u.dim, u.sw(10).time)

   w = Input("w", dim=3)
   print(w.sw(10, seq=5).shape)   # Shape(T=(5, 10), D=(3,)): 5 steps of those

Temporal operations - windows, :class:`~nnodely.Fir`, derivatives and
integrals - act on the inner time; a :class:`~nnodely.Loop` rolls out along
the outer time.

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
``x.last()``
    the current sample, the same as ``x.sw(1)``.
``x.next()``
    the next sample, the same as ``x.sw([0, 1])``.

.. code-block:: python

   x = Input("x")
   past = x.sw(4)          # x[t-3], x[t-2], x[t-1], x[t]
   around = x.sw([2, 1])   # x[t-1], x[t], x[t+1]
   print(past.shape, around.shape)   # Shape(T=(4,), D=(1,)) Shape(T=(3,), D=(1,))

An input can be windowed several times. The data it needs is the union of all
its windows, and the :class:`~nnodely.DataLoader` builds samples that cover it.
All inputs are aligned on the current sample ``t``.

``seq=`` on a window declares the input's outer time: ``x.sw(3, seq=5)`` is
five consecutive windows of three samples, ``x.last(seq=(4, 5))`` four
sequences of five steps (outermost first, ``-1`` for a length that follows the
data). An input has one sequence, set by its first window: later windows may
leave ``seq`` out, but not change it.

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
   a = shared([x.last()])
   b = shared([x.last() * 2.0])   # same kernel and bias as a

Once the model is built, the stream returned by a layer with weights exposes
them as Keras variables (``kernel``, ``bias``), which can be read or assigned.
Here that is ``a.kernel``. The layer object ``shared`` itself only holds the
configuration.

Operators
---------

Streams support ``+``, ``-``, ``*``, ``/`` and ``**``, with other streams or
with numbers. A number becomes a :class:`~nnodely.Constant`, with no time axis:

.. code-block:: python

   from nnodely import Parameter

   gain = Parameter("gain", value=2.0)
   z = gain * x.last() + 1.0

Two streams combine when their shapes agree:

- **time**: streams with time axes must have the same ones; different times
  are never blended. A stream without time - a number, a Parameter - applies
  to every sample.
- **dim**: the dims must be the same, or one of them a scalar, which applies
  to every feature. A vector is never stretched over a matrix.
- one of the streams must already have the shape of the result: a matrix
  without time is not combined with the window of a scalar, which would have
  to stretch both.

.. code-block:: python

   v = Input("v", dim=3)
   v.last() * gain          # (1, 3): the gain applies to every feature
   v.sw(4) + x.sw(4)        # (4, 3): the scalar window applies to every feature
   # v.sw(4) + x.last() is refused: a window of 4 samples and one of 1

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
