.. _nnodely-validation:

Validation
==========

Model validation in **nnodely** follows established practices in mechanical and
control engineering, combining predictive accuracy assessment with explicit
consideration of model complexity.

Validation is typically performed on a dedicated test set and can be integrated
into the training workflow for continuous performance monitoring using the
:doc:`Validator module <validator_module>` and the
:func:`analyzeModel <nnodely.operators.validator.Validator.analyzeModel>` function.

.. rubric:: Training Feedback and Reporting

During training and validation, **nnodely** provides both textual and graphical
feedback through configurable visualization backends.

These tools enable users to inspect:

- Convergence behavior
- Loss evolution
- Quality of model fitting

This feedback facilitates early detection of overfitting, instability, or poor
generalization.

Validation results can also be exported as a structured PDF report. The report
includes loss trends, learned fuzzy and parametric functions, and additional
diagnostic metrics generated during
:func:`analyzeModel <nnodely.operators.validator.Validator.analyzeModel>`, supporting
systematic post-analysis and long-term documentation.

.. rubric:: Domain-Specific Validation Metrics

In many applications, domain-specific performance indicators are essential for
reliable model assessment. In physical and control systems, validation may
involve:

- Frequency-domain analysis
- Stability margins
- Control-oriented performance indices

These analyses can be performed within the validation workflow using
:func:`analyzeModel <nnodely.operators.validator.Validator.analyzeModel>` with appropriate
configuration of prediction horizons, closed-loop connections, and batch
parameters.

These metrics are not commonly used in conventional machine learning but are
critical for evaluating models in engineering contexts.

By incorporating both general-purpose and domain-specific validation criteria,
**nnodely** ensures that trained models are evaluated according to the
requirements of the target application.

.. toctree::
   :maxdepth: 1

   validator_module
