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
- **time** (``T``): the window, the consecutive samples the stream carries -
  or none. An input declared with ``sw`` has one; an input declared without
  it is one value per sample, with no time axis: ``(batch, *seq, *dim)``. A
  number, and a :class:`~nnodely.Parameter` or :class:`~nnodely.Constant`
  declared without time, have none either.
- **dim** (``D``): the feature axes. ``Input("u", dim=3)`` is a
  three-component signal, and ``dim=(3, 2)`` is a 3x2 matrix per sample. The
  default is one feature.

.. code-block:: python

   u = Input("u", dim=3)
   print(u.shape)          # Shape(S=(), T=None, D=(3,)): no window
   uw = Input("uw", dim=3, sw=10)
   print(uw.shape)         # Shape(S=(), T=10, D=(3,)): 10 samples of 3 features
   print(uw.dim, uw.time, uw.seq)

   w = Input("w", dim=3, seq=5, sw=10)
   print(w.shape)          # Shape(S=(5,), T=10, D=(3,)): 5 steps of those

Temporal operations - :class:`~nnodely.Fir`, derivatives, integrals, time
selections - act on the window, and refuse a stream without one; a
:class:`~nnodely.Loop` rolls out along the outermost sequence axis.

Windows
-------

An input declares its window once, with ``sw``, and keeps it: the data a
model reads from the input has that shape. Windows are counted in
**samples**, not seconds: the sampling time only appears where the physics
needs it, as the ``dt`` of :doc:`derivatives and integrators <physics>`.

``sw=n``
    the last ``n`` samples, up to and including the current one.
``sw=(p, f)``
    ``p`` samples up to and including the current one, followed by the next
    ``f`` samples: ``T = p + f``.
no ``sw``
    the current sample, without a time axis.

The layers that read an input take what they need of its window:
:class:`~nnodely.TimeSelect` one sample, :class:`~nnodely.TimeRange` a part.

.. code-block:: python

   from nnodely import TimeRange, TimeSelect

   x = Input("x", sw=(4, 1))        # x[t-3], x[t-2], x[t-1], x[t], x[t+1]
   past = TimeRange(0, 4)(x)        # x[t-3] ... x[t]
   now = TimeSelect(-2)(x)          # x[t]
   following = TimeSelect(-1)(x)    # x[t+1]
   print(past.shape, now.shape)     # Shape(S=(), T=4, D=(1,)) Shape(S=(), T=1, D=(1,))

The :class:`~nnodely.DataLoader` builds samples of every input's window, all
aligned on the current sample ``t``.

``seq=`` declares the input's sequence: ``Input("w", seq=5, sw=3)`` is five
consecutive windows of three samples; ``seq=(4, 5)`` is four sequences of five
steps (outermost first, ``-1`` or ``"dyn"`` for a length that follows the
data). A sequence needs a window: ``sw=1`` for one sample per step.

Layers
------

A layer is configured when it is created and applied by calling it on a stream
or a list of streams:

.. code-block:: python

   from nnodely import Fir

   filtered = Fir(features=1)([past])

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
with numbers. A number becomes a :class:`~nnodely.Constant` without a window,
which applies to every sample:

.. code-block:: python

   from nnodely import Parameter

   gain = Parameter("gain", value=2.0)
   z = gain * x + 1.0

Two streams combine when their shapes agree:

- **time**: the same window, or one of them a single sample, ``T=1``, which
  applies to every sample of the other's window. A value without a window - a
  number, a Parameter, a Constant - applies to every sample of any stream; an
  input without a window does not combine with a stream that has one.
- **sequence**: the same, or a single step, which applies to every step of a
  rollout.
- **dim**: the dims must be the same, or one of them a scalar, which applies
  to every feature. A vector is never stretched over a matrix.
- one of the streams must already have the shape of the result: a matrix of
  one sample is not combined with the window of a scalar, which would have to
  stretch both.

.. code-block:: python

   v = Input("v", dim=3, sw=4)
   s = Input("s", sw=4)
   v * gain                 # S=(), T=4, D=(3,): the gain applies to every feature and sample
   v + s                    # S=(), T=4, D=(3,): the scalar window, to every feature
   v + TimeSelect(-1)(s)    # S=(), T=4, D=(3,): the newest sample, to every sample
   # v + Input("s3", sw=3) is refused: windows of 4 and of 3 samples
   # v + Input("plain", dim=3) is refused: a stream without a window

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
