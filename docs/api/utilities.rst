Utilities
=========

.. currentmodule:: nnodely

Random seed
-----------

.. autofunction:: set_seed

.. autofunction:: get_seed

Training printers
-----------------

These Keras callbacks render the training progress. They are selected by name
with the ``printer`` argument of :meth:`Modely.train`, and can also be created
and passed directly.

.. autoclass:: TinyPrinter

.. autoclass:: LegacyPrinter

.. autoclass:: NNodelyPrinter
