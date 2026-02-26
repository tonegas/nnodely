.. _nnodely-export:

Inference and Model Export
===========================
Trained models in **nnodely** can be used for inference and exported for integration into external
workflows. See the :doc:`Exporter module <exporter_module>` for the available export and inference APIs.

Inference can be performed in two main modes:

- **Single forward pass**, suitable for static predictions or one-step estimation
- **Recursive inference over a temporal horizon**, which enables long-term rollout
  of system dynamics and is particularly useful for forecasting and control-oriented
  applications

The recursive mode allows the model to propagate its own predictions over time,
supporting closed-loop simulation and long-horizon analysis.

The framework provides multiple export modalities to support reproducibility,
portability, and deployment across different platforms.

.. rubric:: JSON Export

The complete model can be serialized into a framework-independent **JSON** format.
This representation encodes the hierarchical structure of the model in a
human-readable form, facilitating:

- Reproducibility
- Version control
- Model inspection
- Long-term storage

Use :func:`saveModel() <nnodely.operators.exporter.Exporter.saveModel>` and
:func:`loadModel() <nnodely.operators.exporter.Exporter.loadModel>` to save and load the JSON
model definitions.

.. rubric:: PyTorch Export

Models can be translated into optimized **PyTorch** classes using symbolic tracing. This process generates efficient, system-specific implementations that are suitable for:

- Fine-tuning
- Integration into deep learning pipelines
- High-performance inference

Relevant API: :func:`saveTorchModel() <nnodely.operators.exporter.Exporter.saveTorchModel>`,
:func:`loadTorchModel() <nnodely.operators.exporter.Exporter.loadTorchModel>`,
:func:`exportPythonModel() <nnodely.operators.exporter.Exporter.exportPythonModel>`, and
:func:`importPythonModel() <nnodely.operators.exporter.Exporter.importPythonModel>`.

.. rubric:: ONNX Export

Export to the **ONNX** format is also supported, enabling deployment across
heterogeneous hardware platforms and inference engines. This option is particularly
useful for production environments and embedded systems.

Use :func:`exportONNX() <nnodely.operators.exporter.Exporter.exportONNX>` to produce an ONNX file
and :func:`onnxInference() <nnodely.operators.exporter.Exporter.onnxInference>` to run inference
with a previously exported ONNX model.

.. rubric:: Modular Export Support

All export modalities are available both for complete models and for individual
sub-models, including recurrent components. This modular approach enables flexible reuse of specific model parts in larger systems.

.. rubric:: Benefits

By supporting multiple inference modes and export formats, **nnodely** decouples
model definition from any single execution backend. This design supports both
research workflows and real-world deployment, ensuring that models can be easily
integrated into diverse application environments.

.. rubric:: Additional tools
  
Training and validation reports (PDF) can be generated from results using
:func:`exportReport() <nnodely.operators.exporter.Exporter.exportReport>`.

.. toctree::
   :maxdepth: 1

   exporter_module
