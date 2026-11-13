Understanding Streams and Shapes
================================

At the end of this page you will know how a model is described as a graph of
streams, what the shape of a stream means, and how streams combine.

**Prerequisites:** :doc:`../getting_started`.

**Tutorials:** :doc:`../tutorial/01_model_design`,
:doc:`../tutorial/07_partitioning`.

**The example.** A vehicle with an accelerometer of three axes and a
speedometer. The streams of this page read those two signals.

Step 1: a model is a graph of streams
-------------------------------------

A *nnodely* model is a graph of **streams**:

- an :class:`~nnodely.Input` is a stream read from data;
- a layer or an operator applied to streams returns a new stream;
- an :class:`~nnodely.Output` gives a stream a public name.

Nothing is computed while the graph is declared.
:meth:`Modely.build() <nnodely.Modely.build>` turns the graph into a Keras
model, and only then are the weights created.

.. code-block:: python

   from nnodely import Input, Linear, Modely, Output, ReLU

   speed = Input("speed")
   hidden = ReLU()(Linear(features=8)(speed))
   drag = Output("drag", Linear(features=1)(hidden))

   model = Modely("drag_model", inputs=[speed], outputs=[drag])
   model.build()

Step 2: the shape of a stream
-----------------------------

Every stream has a shape of three parts. A tensor of the stream holds them
after its batch axis, as ``(batch, *seq, tw, *dim)``:

- **seq** (``S``): the sequence axes, outermost first, possibly none. They
  hold the steps of a trajectory, along which a :class:`~nnodely.Loop` rolls
  out (see :doc:`recurrent`).
- **tw** (``T``): the window, the consecutive samples the stream carries. A
  stream may have none: it then holds one sample, without a ``T`` axis, and
  its tensor is ``(batch, *seq, *dim)``.
- **dim** (``D``): the features of a sample. ``dim=3`` is a vector of three
  values, ``dim=(3, 2)`` a 3x2 matrix. The default is one feature.

.. code-block:: python

   acc = Input("acc", dim=3)
   print(acc.shape)

   acc_window = Input("acc_window", dim=3, tw=10)
   print(acc_window.shape)
   print(acc_window.seq, acc_window.tw, acc_window.dim)

   trajectory = Input("trajectory", dim=3, seq=5, tw=10)
   print(trajectory.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=None, D=(3,))
   Shape(S=(), T=10, D=(3,))
   () 10 (3,)
   Shape(S=(5,), T=10, D=(3,))

The first is one sample of three features, the second 10 samples of them,
the third 5 steps of those 10 samples.

Step 3: windows
---------------

An Input declares its window once, with ``tw``, and keeps it: the data the
model reads from the Input has that shape. Windows are counted in
**samples**, not seconds. The sampling time appears only where the physics
needs it, as the ``dt`` of :doc:`derivatives and integrators <physics>`.

``tw=n``
    The last ``n`` samples, up to and including the current one.
``tw=(p, f)``
    ``p`` samples up to and including the current one, then the next ``f``
    samples: ``T = p + f``.
no ``tw``
    The current sample, without a window axis.

The layers that read an Input take what they need of its window.
:class:`~nnodely.TimeSelect` takes one sample and :class:`~nnodely.TimeRange`
a part, both counting from the oldest sample, 0:

.. code-block:: python

   from nnodely import TimeRange, TimeSelect

   v = Input("v", tw=(4, 1))     # v[t-3], v[t-2], v[t-1], v[t], v[t+1]
   past = TimeRange(0, 4)(v)     # v[t-3] ... v[t]
   now = TimeSelect(-2)(v)       # v[t]
   following = TimeSelect(-1)(v) # v[t+1]
   print(past.shape, now.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=4, D=(1,)) Shape(S=(), T=1, D=(1,))

``TimeSelect`` keeps the window, one sample long, ``T=1``. With
``keep_tw=False`` it drops the window, so the sample applies to every sample
of a window it is combined with (see Step 6).

The :class:`~nnodely.DataLoader` cuts the samples of every Input around the
same current sample ``t`` (see :doc:`data`).

Step 4: sequences
-----------------

``seq`` declares the sequence of an Input:

- ``Input("w", seq=5, tw=3)`` is five consecutive windows of three samples;
- ``seq=(4, 5)`` is four sequences of five steps, outermost first;
- ``seq=-1``, or ``"dyn"`` as a shape prints it, is a length that follows
  the data.

A sequence needs a window: use ``tw=1`` for one sample per step.

.. code-block:: python

   w = Input("w", seq=-1, tw=1)
   print(w.shape)

.. code-block:: text
   :class: output

   Shape(S=(dyn,), T=1, D=(1,))

Step 5: layers
--------------

A layer is created with its settings, then called on a stream or a list of
streams. A layer that reads one stream takes it alone or in a list of one:
``Linear(features=1)(x)`` is ``Linear(features=1)([x])``.

.. code-block:: python

   from nnodely import Fir

   filtered = Fir()(past)
   print(filtered.shape)

.. code-block:: text
   :class: output

   Shape(S=(), T=1, D=(1,))

A layer without settings, whose only argument is ``name``, can be applied as
it is created: ``ReLU(x)`` is ``ReLU()(x)``. A layer that reads one stream,
called on a list, is applied to each stream and returns a list:
``ReLU([a, b])`` is ``[relu(a), relu(b)]``.

Calling the **same layer object** twice applies the same weights twice:

.. code-block:: python

   shared = Linear(features=1)
   a = shared(speed)
   b = shared(speed * 2.0)   # same kernel and bias as a

The weights of a layer are :class:`~nnodely.Parameter` objects. After
``build()``, the stream a layer returns exposes them as Keras variables,
``kernel`` and ``bias``, here ``a.kernel``.

.. _combining-streams:

Step 6: operators and combining streams
---------------------------------------

Streams support ``+``, ``-``, ``*``, ``/``, ``**`` and ``%``, with other
streams or with numbers, and ``-x`` and ``abs(x)``. A number becomes a
:class:`~nnodely.Constant` without a window, which applies to every sample:

.. code-block:: python

   from nnodely import Parameter

   gain = Parameter("gain", value=2.0)
   scaled = gain * speed + 1.0
   opposite = -speed

Two streams combine when their shapes agree. The rule is the same for every
stream: an axis that one stream lacks applies along the other's, and an axis
both have must match.

- **Window**: the same window, or none on one side. A stream without a
  window applies to every sample of the other's. One sample, ``T=1``, is a
  window like any other: it does not combine with a window of 4.
- **Sequence**: the same, or none on one side, which applies to every step.
  One step does not combine with a sequence of 4.
- **Features**: as Keras broadcasts them, lined up from the last axis. Each
  axis is the same, or 1 on one side. ``(3,)`` combines with ``(2, 3)``;
  ``(2,)`` does not.
- One of the two streams already has the shape of the result. A matrix
  without a window does not combine with the window of a scalar, which would
  stretch both.

.. code-block:: python

   a = Input("a", dim=3, tw=4)
   s = Input("s", tw=4)
   print((a * gain).shape)                              # gain on every value
   print((a + s).shape)                                 # s on every feature
   print((a + Input("plain", dim=3)).shape)             # plain on every sample
   print((a + TimeSelect(-1, keep_tw=False)(s)).shape)  # the newest s, everywhere

.. code-block:: text
   :class: output

   Shape(S=(), T=4, D=(3,))
   Shape(S=(), T=4, D=(3,))
   Shape(S=(), T=4, D=(3,))
   Shape(S=(), T=4, D=(3,))

Step 7: names
-------------

Every stream has a name. Layers get one automatically, such as ``Linear1``,
unless ``name=`` gives one. Inputs and outputs are reached by name in the
data dictionaries, in the results of a model and in the objectives, so their
names must be unique within a model.

The layer object decides the weights, not its name. Two layers created apart
never share weights, even with the same name: a model and its reloaded copy,
for instance. When two such layers meet in one model, Keras adds a suffix to
the name of the second (``Fir1_1``). :doc:`composition` explains what happens
when names clash.

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Error
     - Cause
     - Fix
   * - ``streams with windows of 3 and 4 cannot be combined``
     - Two streams with different windows.
     - Take the same part of both windows with ``TimeRange``, or one sample
       with ``TimeSelect(idx, keep_tw=False)``.
   * - ``streams with windows of 1 and 4 cannot be combined``
     - A sample taken with ``TimeSelect(idx)`` keeps a window of one.
     - Use ``TimeSelect(idx, keep_tw=False)`` to apply it to every sample.
   * - ``Fir acts on the window of its stream, and 'x' has none``
     - A temporal layer on a stream without a window.
     - Declare the Input with a window, ``Input("x", tw=n)``.
   * - ``Linear has settings, so it is configured first``
     - The layer was called before it was configured, ``Linear(x)``.
     - Configure it, then call it: ``Linear(features=1)(x)``.

See also
--------

:class:`~nnodely.Input`, :class:`~nnodely.Output`,
:class:`~nnodely.Parameter`, :class:`~nnodely.Constant`,
:class:`~nnodely.Modely`, :class:`~nnodely.TimeSelect`,
:class:`~nnodely.TimeRange`, :class:`~nnodely.Fir`, :class:`~nnodely.Linear`,
:doc:`../glossary`.
