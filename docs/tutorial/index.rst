Tutorials
=========

Each tutorial is a Jupyter notebook that teaches one part of *nnodely* with
small, complete examples: what to write, what it does, and why. The pages show
each notebook with its outputs; the notebooks themselves are in the
``docs/tutorial`` folder of the repository, to run and change. Each one runs on
its own, from top to bottom, on any Keras backend (exporting to ONNX needs
TensorFlow or PyTorch).

The first five follow the life of a model, from its design to its export: read
them in order. The others each go through one family of layers, or one way of
modelling, and can be read in any order once the first five are known.

.. toctree::
   :maxdepth: 1
   :caption: The life of a model

   01_model_design
   02_data_management
   03_model_training
   04_model_validation
   05_save_load_export

.. toctree::
   :maxdepth: 1
   :caption: Layers and modelling

   06_parameter_constant
   07_partitioning
   08_fir_linear
   09_fuzzify_localmodel
   10_recurrent_loop
   11_calculus
   12_custom_layers
   13_equation_learner
