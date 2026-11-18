Utilities
=========

.. currentmodule:: nnodely

Backend and device
------------------

:func:`get_backend` tells the Keras backend *nnodely* runs on, and
:func:`get_device` the device it computes on, both chosen at the import: see
:doc:`../getting_started`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_backend
   get_device

Random seed
-----------

:func:`set_seed` makes the initial weights, the shuffling of the data and the
random draws of a script reproducible.

.. autosummary::
   :toctree: generated
   :nosignatures:

   set_seed
   get_seed
