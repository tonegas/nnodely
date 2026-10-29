Layers
======

.. currentmodule:: nnodely

Layers are created with their configuration and called on a stream or a
list of streams (see :doc:`../guide/concepts`). The arithmetic operators
``+ - * / **`` between streams are covered in the same page.

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

.. autoclass:: Differentiate

.. autoclass:: DerivativeStep

.. autoclass:: Derivative

.. autoclass:: Integrate

.. autoclass:: IntegrateStep

.. autofunction:: Ode

.. autoclass:: OdeNet

Recurrence
----------

.. autoclass:: Loop

.. autoclass:: Roll

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

Arithmetic
----------

.. autoclass:: Exp

.. autoclass:: Log

.. autoclass:: Log10

.. autoclass:: Sqrt

.. autoclass:: Abs

.. autoclass:: Sign

.. autoclass:: Floor

.. autoclass:: Ceil

.. autoclass:: Deg2Rad

.. autoclass:: Negative

.. autoclass:: Clamp
