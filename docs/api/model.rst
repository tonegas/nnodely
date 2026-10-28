Model
=====

.. currentmodule:: nnodely

.. autoclass:: Modely
   :members: build, built, minimize, remove_minimizer, parameters, all_inputs,
             freeze, unfreeze, train,
             validate, rollback,
             flatten, summary, plot, export_html, save, load, export_keras,
             import_keras, export_onnx, validate_onnx

.. autoclass:: ModelyLoop
   :members: minimize

.. autoclass:: nnodely.core.validation.ValidationResult
   :members: metrics, summary
