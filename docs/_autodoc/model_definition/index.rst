.. _nnodely-modely:

Model Definition
================

In **nnodely**, a neural model is defined using a small set of explicit and
modular components: :ref:`input_module`, :ref:`parameter_module`, **Relations**, and :ref:`output_module`. Together, these elements describe the model structure and data flow, corresponding to the model definition and composition phases (PH1 and PH6).

This modular decomposition makes architectural choices explicit and supports the systematic construction, extension, and reuse of knowledge-informed models.

Inputs
------

:ref:`input_module` defines the interface between the model and the data.

Each input:

- Has a unique name and defined dimensionality
- Can be associated with a temporal context window
- Can access past and/or future samples

Inputs may originate from external datasets or from internal connections to other model components. This flexibility enables complex signal routing and the construction of composite model architectures.

Parameters
----------

:ref:`parameter_module` represent the learnable quantities of the model and explicitly indicate which parts of the architecture are estimated from data.

Each parameter:

- Has a unique name
- Has a defined dimensionality
- Can be associated with temporal windows

This ensures unambiguous reference, even in large and structured models.

Parameters can be defined over time to capture dynamic effects. Their initialization is considered an integral part of the modeling process and is handled by the :ref:`initializer_module`.

In addition to standard random initialization, **nnodely** provides structured initialization methods, such as smooth interpolation or decaying profiles. These methods allow prior physical knowledge to be embedded directly into the learnable representation.

Outputs
-------

:ref:`output_module` exposes selected relations as model predictions and define the interface between the internal architecture and user-level quantities of interest.

Each output is associated with a specific relation and identified by a semantic tag.

Outputs affect training only when explicitly included in the loss function; otherwise, they serve as access points for inference and analysis

Model Structured NN building blocks
-------------------------------------

**nodely** provides reusable neural building blocks to construct MSNNs:

- :ref:`activation_module` — standard activation functions (ReLU, tanh, etc.)

- :ref:`arithmetic_module` — interpretable arithmetic operations

- :ref:`fir_module` — finite-impulse-response filters for linear dynamics

- :ref:`fuzzify_module` — fuzzy membership and rule-based transformations

- :ref:`equationlearner_module` — learn interpretable analytical expressions

- :ref:`linear_module` — linear transformations and affine blocks

- :ref:`localmodel_module` — local/expert models for piecewise dynamics

- :ref:`part_module` — partitioned components for structured inputs

- :ref:`trigonometric_module` — sin/cos features for periodic behavior

- :ref:`parametricfunction_module` — parameterized functions for initialization or prior knowledge


Contents
---------

.. toctree::
   :maxdepth: 2
   
   modely_class
   msnn_ins_out_param/index
   layers/index