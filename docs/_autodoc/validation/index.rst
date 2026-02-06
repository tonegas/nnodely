.. _nnodely-validation:

Validation
============

Model validation in nnodely follows established practices in mechanical and control engineering, combining predictive accuracy assessment with explicit consideration of model complexity.

Validation is typically performed on a dedicated test set and can be integrated into the training workflow for continuous performance monitoring.

Training Feedback and Reporting
--------------------------------
During training and validation, nnodely provides both textual and graphical feedback through configurable visualization backends. These tools enable users to inspect:

- Convergence behavior
- Loss evolution
- Quality of model fitting

This feedback facilitates early detection of overfitting, instability, or poor generalization.

Validation results can also be exported as a structured PDF report. The report includes loss trends, learned fuzzy and parametric functions, and additional diagnostic metrics, supporting systematic post-analysis and long-term documentation.

Domain-Specific Validation Metrics
----------------------------------------------------------------
In many applications, domain-specific performance indicators are essential for reliable model assessment. In physical and control systems, validation may involve:

- Frequency-domain analysis 
- Stability margins
- Control-oriented performance indices

These metrics are not commonly used in conventional machine learning but are critical for evaluating models in engineering contexts.

By incorporating both general-purpose and domain-specific validation criteria, nnodely ensures that trained models are evaluated according to the requirements of the target application.

Contents
---------
.. toctree::
   :maxdepth: 2

   validator_module