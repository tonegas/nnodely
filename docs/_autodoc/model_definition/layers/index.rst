.. _nnodely-modules-layers:

Building blocks
===================================

In **nnodely**, model behavior is defined through a collection of modular
**building blocks**. These blocks implement elementary operations and
structured transformations that can be combined to form complex neural models.

Building blocks operate on inputs, parameters, and intermediate signals, and
represent the computational core of the architecture.

Each block encapsulates a specific mathematical or functional behavior and can
be reused across different models.

.. toctree::
   :maxdepth: 2

   activation_module
   arithmetic_module
   fir_module
   fuzzify_module
   equationlearner_module
   linear_module
   localmodel_module
   part_module
   trigonometric_module
   parametricfunction_module
