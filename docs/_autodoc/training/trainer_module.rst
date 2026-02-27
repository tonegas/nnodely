Training module
========================

The framework supports two complementary training modalities, depending on the
model architecture.

.. rubric:: Feed-forward Training

Follows the conventional mini-batch paradigm. Samples are shuffled and propagated through an acyclic computation graph, and model parameters are updated based on the instantaneous loss using :func:`trainModel <nnodely.operators.trainer.Trainer.trainModel>` with optimizers from the :doc:`Optimizer module <optimizer_module>`.
This modality is suitable for architectures without internal feedback, where
predictions depend only on the provided inputs.

.. rubric:: Recurrent Training
    
Targets architectures with closed-loop dependencies, such as feedback connections or coupled multi-model structures. In this modality, training is performed over a finite prediction horizon. The model is rolled out forward in time, and parameter updates are applied only after completing the full rollout using :func:`trainModel <nnodely.operators.trainer.Trainer.trainModel>`.

Early stopping strategies from the :doc:`Early Stopping module <earlystopping_module>` can be applied to control convergence in long-horizon training.

.. automodule:: nnodely.operators.trainer
    :undoc-members:
    :no-inherited-members:

.. autofunction:: nnodely.operators.trainer.Trainer.addMinimize
.. autofunction:: nnodely.operators.trainer.Trainer.removeMinimize
.. autofunction:: nnodely.operators.trainer.Trainer.getTrainingInfo
.. autofunction:: nnodely.operators.trainer.Trainer.trainModel

For more example please refer to the :doc:`training tutorial <../tutorials/examples/training>`.