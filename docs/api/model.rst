Model
=====

.. currentmodule:: nnodely

.. autoclass:: Modely
   :members: build, built, minimize, remove_minimizer, parameters, inputs, outputs, minimizers,
             freeze, unfreeze, train,
             validate,
             flatten, summary, export_html, save, load, export_keras,
             import_keras, export_onnx, validate_onnx

.. autoclass:: ModelyLoop
   :members: minimize

.. autoclass:: nnodely.core.validation.ValidationResult
   :members: summary
