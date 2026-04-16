Glossary
========

.. rubric:: MS‑NN

A structured neural network that integrates the learning ability of neural networks with physical, control, and estimation knowledge.

.. rubric:: Modely

The nnodely framework used for creating, composing, training, and deploying MS‑NNs.

.. rubric:: Neural Model

An MS‑NN architecture defined through inputs, outputs, and building blocks (relations, layers, etc.) that describe how signals and data are transformed within the model.

.. rubric:: Input / Output / Parameter

- **Input**: variables entering the model.
- **Output**: signals predicted or calculated by the model.
- **Parameter**: quantities learned during training or fixed constants.

.. rubric:: Stream

Internal representation of a signal in the model graph, manipulable through temporal operators (delay, window, shift).

.. rubric:: Model Composition

Section dedicated to combining multiple models or sub-models using feedforward connections and closed loops to build complex architectures from modular components.

.. rubric:: Composer

Class providing API for static model composition: adding models, connections, closed loops, etc. (e.g., `addModel`, `addConnect`).

.. rubric:: Connect

Feedforward link from the output of one model to the input of another (or the same) model.

.. rubric:: Closed Loop

Feedback link where the output of a block influences one of its inputs or the input of another block.

.. rubric:: NeuralizeModel

Phase in which an MS‑NN definition is transformed into a trainable representation, ready for training, creating time windows, tensors, and internal structures.

.. rubric:: Training

Process in which model parameters are optimized based on a loss function and the provided data.

.. rubric:: Validation

Phase in which the model’s performance is measured on a dataset not used during training to assess generalization and reliability.

.. rubric:: Inference

Process of using a trained model to predict or compute outputs from new inputs.

.. rubric:: Model Export

Operations to save and/or convert a trained MS‑NN into standard formats (e.g., PyTorch, ONNX) for deployment or use outside nnodely.

.. rubric:: Dataset

Collection of data (training / test / validation) used to train and evaluate the model.
