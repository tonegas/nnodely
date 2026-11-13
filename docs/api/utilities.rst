Utilities
=========

.. currentmodule:: nnodely

Random seed
-----------

:func:`set_seed` makes the initial weights, the shuffling of the data and the
random draws of a script reproducible.

.. autosummary::
   :toctree: generated
   :nosignatures:

   set_seed
   get_seed

Training printers
-----------------

These Keras callbacks show the progress of a training. Select one by name with
the ``printer`` argument of :meth:`Modely.train`, or create one and pass it.

.. autosummary::
   :toctree: generated
   :nosignatures:

   TinyPrinter
   LegacyPrinter
   NNodelyPrinter
