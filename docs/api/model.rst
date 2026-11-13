Models
======

.. currentmodule:: nnodely

A :class:`Modely` is a model: a graph of streams from its inputs to its
outputs, with the objectives that train it. A :class:`ModelyLoop` is a model
declared one step at a time, whose states are fed back over a rollout.

Guides: :doc:`../guide/models`, :doc:`../guide/training`,
:doc:`../guide/recurrent`, :doc:`../guide/composition`,
:doc:`../guide/export`. Tutorials: :doc:`../tutorial/01_model_design`,
:doc:`../tutorial/03_model_training`, :doc:`../tutorial/05_save_load_export`,
:doc:`../tutorial/10_recurrent_loop`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Modely
   ModelyLoop

Returned objects
----------------

:meth:`Modely.train` returns a :class:`~nnodely.core.results.TrainResult`,
:meth:`Modely.validate` a :class:`~nnodely.core.results.ValidationResult`,
with a :class:`~nnodely.core.validation.Signal` per objective. They are read
by their attributes, and have one form: ``losses`` under the keys Keras logs,
``loss`` and ``loss_<objective>``. You read them, and never create them, so
they are not exported by the top-level package.

Guide: :doc:`../guide/validation`. Tutorial: :doc:`../tutorial/04_model_validation`.

.. currentmodule:: nnodely.core.results

.. autosummary::
   :toctree: generated
   :nosignatures:

   TrainResult
   ValidationResult

.. currentmodule:: nnodely.core.validation

.. autosummary::
   :toctree: generated
   :nosignatures:

   Signal
