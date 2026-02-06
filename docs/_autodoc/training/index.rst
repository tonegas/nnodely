.. _nnodely-training:

Training
========

Training in **nnodely** aims to identify model parameters by minimizing the
discrepancy between predicted and target outputs using user-defined loss
functions and evaluation metrics.

The training process is managed through the
:doc:`Training module <trainer_module>`, which provides utilities such as
:func:`addMinimize <nnodely.operators.trainer.Trainer.addMinimize>`, :func:`removeMinimize <nnodely.operators.trainer.Trainer.removeMinimize>`, and :func:`trainModel <nnodely.operators.trainer.Trainer.trainModel>`.
Training status and diagnostics can be accessed through
:func:`getTrainingInfo <nnodely.operators.trainer.Trainer.getTrainingInfo>`.

The framework supports standard gradient-based optimization methods and
multi-objective loss functions through optional weighting. Optimization
algorithms are provided by the :doc:`Optimizer module <optimizer_module>`,
including :class:`SGD <nnodely.basic.optimizer.SGD>` and :class:`Adam <nnodely.basic.optimizer.Adam>`.

Regularization mechanisms such as **dropout** and **early stopping** are also
available to improve generalization and prevent overfitting. Early stopping
strategies are implemented in the
:doc:`Early Stopping module <earlystopping_module>` through functions such as
:func:`early_stop_patience <nnodely.support.earlystopping.early_stop_patience>`, :func:`mean_stopping <nnodely.support.earlystopping.mean_stopping>`,
:func:`standard_early_stopping <nnodely.support.earlystopping.standard_early_stopping>`, and :func:`select_best_model <nnodely.support.earlystopping.select_best_model>`.

To handle temporal heterogeneity across samples, training uses a dedicated
batching strategy that preserves temporal coherence within each batch.
This ensures that time-dependent relationships are correctly maintained during
optimization.

**nnodely** supports two complementary training modalities, depending on the
model architecture.

Feed-forward
-----------------------------------
**Feed-forward** training follows the conventional mini-batch paradigm. Samples
are shuffled and propagated through an acyclic computation graph, and model
parameters are updated based on the instantaneous loss using
:func:`trainModel <nnodely.operators.trainer.Trainer.trainModel>` with optimizers from the :doc:`Optimizer module <optimizer_module>`.
This modality is suitable for architectures without internal feedback, where
predictions depend only on the provided inputs.

Recurrent
-----------------------------------
**Recurrent training** targets architectures with closed-loop dependencies, such
as feedback connections or coupled multi-model structures. In this modality,
training is performed over a finite prediction horizon. The model is rolled out
forward in time, and parameter updates are applied only after completing the
full rollout using :func:`trainModel <nnodely.operators.trainer.Trainer.trainModel>`.

Early stopping strategies from the
:doc:`Early Stopping module <earlystopping_module>` can be applied to control
convergence in long-horizon training.

This approach enables correct learning of long-term temporal dependencies and
dynamic behavior.

Key Features
------------

- Support for custom loss functions and evaluation metrics via
  :func:`addMinimize <nnodely.operators.trainer.Trainer.addMinimize>` and :func:`removeMinimize <nnodely.operators.trainer.Trainer.removeMinimize>`
- Gradient-based optimization with :class:`SGD <nnodely.basic.optimizer.SGD>` and :class:`Adam <nnodely.basic.optimizer.Adam>`
- Regularization through early stopping methods
- Temporally coherent batching strategies
- Dedicated support for feed-forward and recurrent architectures
- Access to training diagnostics through :func:`getTrainingInfo <nnodely.operators.trainer.Trainer.getTrainingInfo>`

Contents
---------

.. toctree::
   :maxdepth: 2

   trainer_module
   optimizer_module
   earlystopping_module
