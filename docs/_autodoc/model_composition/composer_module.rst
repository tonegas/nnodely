.. _model-level_composition:

Model-level composition
========================

Model-level composition deals with independently defined sub-models and their
interactions. Sub-models are logical groups of output streams that can be added,
removed, connected, or frozen independently. Use the Composer API to manage
these operations and to prepare the whole definition for neuralization.

Key Composer operators:

- **addModel / removeModel** : register and unregister named sub-models.
  See :meth:`~nnodely.operators.composer.Composer.addModel` and
  :meth:`~nnodely.operators.composer.Composer.removeModel`.

- **addConnect / addClosedLoop / removeConnection** : connect an output stream
  of one model to an input of another (or the same) model; create closed-loop
  feedback between streams and inputs. See
  :meth:`~nnodely.operators.composer.Composer.addConnect`,
  :meth:`~nnodely.operators.composer.Composer.addClosedLoop`, and
  :meth:`~nnodely.operators.composer.Composer.removeConnection`.

- **neuralizeModel** : finalize the model definition and translate it into a
  trainable neural representation (builds time windows, slices, and internal
  tensors required for training/inference). See
  :meth:`~nnodely.operators.composer.Composer.neuralizeModel`.

.. automodule:: nnodely.operators.composer
    :undoc-members:
    :no-inherited-members:

.. autofunction:: nnodely.operators.composer.Composer.addModel
.. autofunction:: nnodely.operators.composer.Composer.removeModel
.. autofunction:: nnodely.operators.composer.Composer.addConnect
.. autofunction:: nnodely.operators.composer.Composer.addClosedLoop
.. autofunction:: nnodely.operators.composer.Composer.removeConnection
.. autofunction:: nnodely.operators.composer.Composer.neuralizeModel

For further examples please refer to the :doc:`relative tutorial <../tutorials/examples/states>`.