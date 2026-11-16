API Reference
=============

Every public name of *nnodely* has a page of its own. A page has the same parts
in the same order:

- a one-line summary of what the name returns;
- **Syntax**: every calling form, each with one sentence;
- **Parameters**: every argument, with its type, default and allowed values;
- **Call arguments**, for a layer: the streams it is called on, and their
  shape;
- **Returns**, with the shape rule, and **Raises**, with the errors and their
  causes;
- **See also**, and **Notes** with the formulas;
- examples in groups, from the most common use to the edge cases: **Basic
  Examples**, **Scope** (the kinds of input it accepts), **Options** (one
  example per argument), **Applications**, **Properties & Relations** and
  **Possible Issues** (a wrong use, the error it raises, and the fix).

The examples of a page run in order, and the imports of the first one serve
them all. What they print is checked on TensorFlow, PyTorch and JAX.

Every name you create is importable from the top-level package:

.. code-block:: python
   :class: skip-test

   from nnodely import Modely, Input, Fir

The objects that *nnodely* returns and you never create, such as the result of
:meth:`Modely.validate() <nnodely.Modely.validate>`, are listed with their full
path.

The shapes are written ``Shape(S, T, D)``: the sequence axes, the window and
the features of a stream (see :doc:`../guide/concepts` and the
:doc:`../glossary`).

.. toctree::
   :maxdepth: 2

   model
   data
   streams
   layers
   utilities
