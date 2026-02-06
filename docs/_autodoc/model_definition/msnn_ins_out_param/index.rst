.. _nnodely-msnn_ins_out_param:

Model structured NN Inputs Outputs and Parameters
=================================================

In **nnodely**, model definition follows a structured workflow based on three
core modules: **Input**, **Parameter**, and **Output**. These modules define how
data enters the model, how learnable quantities are represented, and how
predictions are exposed.

Input Module
^^^^^^^^^^^^

The **Input** module defines the interface between the model and external data.

Each input represents a measured or provided signal and can be associated with
a temporal context, such as past samples or sliding windows. Temporal operators
(e.g., time windows and delays) allow the model to access historical information.

During training and inference, inputs are responsible for injecting dataset
samples into the computational graph while preserving their temporal structure.

Parameter Module
^^^^^^^^^^^^^^^^

The **Parameter** module represents the learnable quantities of the model.

Parameters define which parts of the architecture are estimated from data.
They may represent coefficients of filters, weights of layers, or parameters of
parametric functions. Parameters can be static or time-dependent and may be
associated with temporal windows.

Initialization strategies allow prior knowledge to be embedded in the model
before training.

Output Module
^^^^^^^^^^^^^

The **Output** module exposes selected internal relations as model predictions.

Each output is associated with a symbolic expression built from inputs,
parameters, and computational blocks. Outputs define the interface between the
internal model structure and user-level quantities of interest.

Only outputs included in the loss function influence training. Other outputs are
used for inference, monitoring, and post-analysis.

This modular workflow enables the systematic construction of structured and
interpretable neural models.
knowledge.

.. toctree::
   :maxdepth: 2

   input_module
   parameter_module
   initializer_module
   output_module
