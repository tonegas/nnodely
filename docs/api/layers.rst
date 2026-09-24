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

.. autoclass:: Interpolation

Feature axis
------------

.. autoclass:: Select

.. autoclass:: Range

.. autoclass:: Concatenate

.. autoclass:: Sum

Time axis
---------

.. autoclass:: TimeSelect

.. autoclass:: TimeRange

.. autoclass:: TimeConcatenate

Calculus
--------

.. autoclass:: Derivative

.. autoclass:: Integrate

.. autofunction:: Ode

.. autoclass:: OdeNet

Recurrence
----------

.. autoclass:: Loop

.. autoclass:: Roll

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
