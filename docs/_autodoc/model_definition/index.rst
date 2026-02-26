.. _nnodely-modely:

Model Definition
================

In **nnodely**, a neural model is defined using a small set of explicit and
modular components: :ref:`input_module`, :ref:`parameter_module`, **Relations**, and :ref:`output_module`. Together, these elements describe the model structure and data flow, corresponding to the model definition and composition phases (PH1 and PH6).

This modular decomposition makes architectural choices explicit and supports the systematic construction, extension, and reuse of knowledge-informed models.

In addition to these core components, **nnodely** provides a library of reusable building blocks for constructing complex MSNN architectures. These include standard activation functions, interpretable arithmetic operations, linear transformations, local/expert models, and more. Each building block is designed to be composable and compatible with the overall model structure.

.. toctree::
   :maxdepth: 1
   
   msnn_ins_out_param/input_module
   msnn_ins_out_param/parameter_module
   msnn_ins_out_param/initializer_module
   msnn_ins_out_param/output_module
   layers/index