Layers
======

.. currentmodule:: nnodely

A layer is created with its settings and then called on a stream, or on a list
of streams: ``Fir(features=2)(x)``. A layer that reads one stream takes it
alone or in a list of one, so ``Fir()(x)`` is ``Fir()([x])``. A layer without
settings, such as :class:`Sin`, can be applied as it is created: ``Sin(x)`` is
``Sin()(x)``. See :doc:`../guide/concepts` for the shapes and the
:doc:`../guide/models` guide for an overview.

Operators
---------

The Python operators apply to streams, and to a stream and a number. A number
becomes a :class:`Constant` without a window, which applies to every sample.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Result
   * - ``a + b``, ``a - b``, ``a * b``, ``a / b``
     - Elementwise sum, difference, product and quotient.
   * - ``a ** b``
     - Elementwise power: ``x ** 2.0`` squares, ``2.0 ** x`` raises 2.
   * - ``a % b``
     - Elementwise remainder, with the sign of ``b``, as in Python:
       ``x % 2.0`` is ``[1, 0, 1]`` for ``x = [1, 2, 3]``.
   * - ``-x``
     - The opposite, the same as :class:`Negative`.
   * - ``abs(x)``
     - The absolute value, the same as :class:`Abs`.

Two streams combine when their shapes agree (see
:ref:`the rules <combining-streams>`).

Layers with weights
-------------------

Their weights are :class:`Parameter` objects: they can be given, shared
between layers, frozen, and read after training. :class:`PReLU`, an
activation with a learned slope, is listed with the activations.

Tutorials: :doc:`../tutorial/08_fir_linear`,
:doc:`../tutorial/09_fuzzify_localmodel`,
:doc:`../tutorial/13_equation_learner`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Linear
   Fir
   LocalModel
   EquationLearner
   BatchNorm

Fuzzy sets and lookup tables
----------------------------

Tutorial: :doc:`../tutorial/09_fuzzify_localmodel`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Fuzzify
   Triangular
   Rectangular
   Gaussian
   Interpolation

Custom layers
-------------

Tutorial: :doc:`../tutorial/12_custom_layers`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   KerasFunction

Selecting and joining
---------------------

Each family acts on one kind of axis: the features (``D``), the window
(``T``) or the sequence (``S``).

Tutorial: :doc:`../tutorial/07_partitioning`.

.. rubric:: Features

.. autosummary::
   :toctree: generated
   :nosignatures:

   Select
   Range
   Concatenate

.. rubric:: Window

.. autosummary::
   :toctree: generated
   :nosignatures:

   TimeSelect
   TimeRange
   TimeConcatenate
   TimeReverse

.. rubric:: Sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   SeqSelect
   SeqRange
   SeqConcatenate
   SeqStack
   SeqReverse

Reductions
----------

They reduce the features of every sample.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Sum
   Mean
   Max
   Min
   Norm
   Cross

Calculus
--------

Derivatives with respect to an input or to time, integrators and ODE steps.

Guide: :doc:`../guide/physics`. Tutorial: :doc:`../tutorial/11_calculus`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Gradient
   DerivativeStep
   Derivative
   IntegrateStep
   Integrate
   Ode

Recurrence
----------

A :class:`Loop` rolls a model out over a sequence. :class:`ModelyLoop`, a
model declared one step at a time, is with the :doc:`models <model>`.

Guide: :doc:`../guide/recurrent`. Tutorial: :doc:`../tutorial/10_recurrent_loop`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Loop
   Buffer

Activations
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReLU
   LeakyReLU
   ELU
   PReLU
   Sigmoid
   Tanh
   Softmax
   Swish
   GELU
   Softplus

Math functions
--------------

They apply to every value of a stream, and keep its shape. Each page gives
the formula, the domain, and the gradient: :class:`Floor`, :class:`Ceil`,
:class:`Round` and :class:`Sign` have a zero gradient, so a weight that only
reaches the loss through them does not train.

.. rubric:: Trigonometric and hyperbolic

.. autosummary::
   :toctree: generated
   :nosignatures:

   Sin
   Cos
   Tan
   Asin
   Acos
   Atan
   Atan2
   Sinh
   Cosh
   Asinh
   Acosh
   Atanh

.. rubric:: Exponentials and logarithms

.. autosummary::
   :toctree: generated
   :nosignatures:

   Exp
   Expm1
   Log
   Log1p
   Log10
   Log2

.. rubric:: Powers, signs and rounding

.. autosummary::
   :toctree: generated
   :nosignatures:

   Sqrt
   Square
   Abs
   Sign
   Negative
   Floor
   Ceil
   Round

.. rubric:: Units and bounds

.. autosummary::
   :toctree: generated
   :nosignatures:

   Deg2Rad
   Rad2Deg
   Clamp
   Maximum
   Minimum
