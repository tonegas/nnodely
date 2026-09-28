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
   hidden = ReLU()([Linear(out_features=8)([x.last()])])
   y = Output("y", Linear(out_features=1)([hidden]))

   model = Modely("regressor", inputs=[x], outputs=[y])
   model.build()

Shapes
------

Every stream has a shape made of three parts, without the batch axis:

``(*dim, time, *seq)``

- **dim**: the feature axes. ``Input("u", dim=3)`` is a three-component signal,
  and ``dim=(3, 2)`` is a 3x2 matrix per sample. The default is one feature.
- **time**: the window of consecutive samples the stream carries.
- **seq**: optional sequence axes, used when a model is rolled out over a
  trajectory (see :doc:`recurrent`).

.. code-block:: python

   u = Input("u", dim=3)
   print(u.sw(10).shape)   # (3, 10): 3 features, 10 samples
   print(u.dim, u.sw(10).time)

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
   print(past.shape, around.shape)

An input can be windowed several times. The data it needs is the union of all
its windows, and the :class:`~nnodely.DataLoader` builds samples that cover it.
All inputs are aligned on the current sample ``t``.

Layers
------

A layer is configured when it is created and applied by calling it on a stream
or a list of streams:

.. code-block:: python

   from nnodely import Fir

   filtered = Fir(out_features=1)([x.sw(10)])

Calling the **same layer object** twice applies the same weights twice:

.. code-block:: python

   shared = Linear(out_features=1)
   a = shared([x.last()])
   b = shared([x.last() * 2.0])   # same kernel and bias as a

Once the model is built, the stream returned by a layer with weights exposes
them as Keras variables (``kernel``, ``bias``), which can be read or assigned.
Here that is ``a.kernel``. The layer object ``shared`` itself only holds the
configuration.

Operators
---------

Streams support ``+``, ``-``, ``*``, ``/`` and ``**``, with other streams or
with numbers. A number becomes a :class:`~nnodely.Constant`:

.. code-block:: python

   from nnodely import Parameter

   gain = Parameter("gain", value=2.0)
   z = gain * x.last() + 1.0

Names
-----

Every node has a name. Layers get one automatically (``Linear1``,
``Fir2``, ...) unless ``name=`` is given. Inputs and outputs are addressed by
name everywhere else: in the data dictionaries, the results of inference and
the training objectives. They must be unique within a model.
