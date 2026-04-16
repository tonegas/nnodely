.. _model_composition_training_inference:

Composition at Training and Inference Time
============================================

In addition to static composition through the Composer API, nnodely also allows model composition to be defined dynamically at execution time, during training, analysis, or inference.

In this case, connections and closed loops between signals and models can be specified directly as arguments of high-level execution methods such as train, trainAndAnalyze, and inference.

This approach is useful when:

- The same base models are reused in different configurations;

- Feedback and interconnections must change across experiments;

- Rapid prototyping and testing of architectures is required.

This dynamic composition can be performed by calling one of the following methods of the Modely class, with appropriate arguments for composition:

- :meth:`~nnodely.operators.trainer.Trainer.trainModel`;
- :meth:`~nnodely.nnodely.Modely.trainAndAnalyze`;
- :meth:`~nnodely.operators.validator.Validator.analyzeModel`;
- :meth:`~nnodely.operators.composer.Composer.__call__`.

.. code-block:: python

    msd.trainAndAnalyze(models='PID', closed_loop={'x':'x_n', 'x_m':'x_n'}, connect={'F':'F_PID'}, ...)


.. .. automethod:: nnodely.nnodely.Modely.trainAndAnalyze
..    :no-index:
