Layers
======

.. currentmodule:: nnodely

Layers are created with their configuration and called on a stream or a
list of streams (see :doc:`../guide/concepts`). The arithmetic operators
``+ - * / ** %`` between streams, and ``-x`` and ``abs(x)``, are covered in
the same page.

Layers with weights
-------------------

.. autoclass:: Linear

.. autoclass:: Fir

.. autoclass:: LocalModel

.. autoclass:: EquationLearner

.. autoclass:: BatchNorm

Fuzzy and lookup
----------------

.. autoclass:: Fuzzify

.. autoclass:: Triangular

.. autoclass:: Rectangular

.. autoclass:: Gaussian

.. autoclass:: Interpolation

Custom functions
----------------

.. autoclass:: KerasFunction

Feature axis
------------

.. autoclass:: Select

.. autoclass:: Range

.. autoclass:: Concatenate

.. autoclass:: Sum

.. autoclass:: Mean

.. autoclass:: Max

.. autoclass:: Min

.. autoclass:: Norm

.. autoclass:: Cross

Window
------

.. autoclass:: TimeSelect

.. autoclass:: TimeRange

.. autoclass:: TimeConcatenate

.. autoclass:: TimeReverse

Sequence
--------

.. autoclass:: SeqSelect

.. autoclass:: SeqRange

.. autoclass:: SeqConcatenate

.. autoclass:: SeqStack

.. autoclass:: SeqReverse

Calculus
--------

.. autoclass:: Gradient

.. autoclass:: DerivativeStep

.. autoclass:: Derivative

.. autoclass:: Integrate

.. autoclass:: IntegrateStep

.. autofunction:: Ode

Recurrence
----------

.. autoclass:: Loop

.. autoclass:: Buffer

Activations
-----------

.. autoclass:: ReLU

.. autoclass:: LeakyReLU

.. autoclass:: ELU

.. autoclass:: PReLU

.. autoclass:: Sigmoid

.. autoclass:: Tanh

.. autoclass:: Softmax

.. autoclass:: Swish

.. autoclass:: GELU

.. autoclass:: Softplus

Trigonometric
-------------

.. autoclass:: Sin

.. autoclass:: Cos

.. autoclass:: Tan

.. autoclass:: Asin

.. autoclass:: Acos

.. autoclass:: Atan

.. autoclass:: Atan2

.. autoclass:: Sinh

.. autoclass:: Cosh

.. autoclass:: Asinh

.. autoclass:: Acosh

.. autoclass:: Atanh

Arithmetic
----------

.. autoclass:: Exp

.. autoclass:: Expm1

.. autoclass:: Log

.. autoclass:: Log1p

.. autoclass:: Log10

.. autoclass:: Log2

.. autoclass:: Sqrt

.. autoclass:: Square

.. autoclass:: Abs

.. autoclass:: Sign

.. autoclass:: Floor

.. autoclass:: Ceil

.. autoclass:: Round

.. autoclass:: Deg2Rad

.. autoclass:: Rad2Deg

.. autoclass:: Negative

.. autoclass:: Clamp

.. autoclass:: Maximum

.. autoclass:: Minimum
