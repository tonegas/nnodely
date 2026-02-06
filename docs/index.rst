.. nnodely documentation master file, created by
   sphinx-quickstart on Wed Oct 13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nnodely's documentation!
====================================

.. image:: https://raw.githubusercontent.com/tonegas/nnodely/main/imgs/logo_white_info.png
     :target: https://github.com/tonegas/nnodely
     :alt: Open 


nnodely is a framework designed to facilitate the creation and deployment of **Model-Structured Neural Networks** (**MSNNs**). MS-NNs combine the learning capabilities of neural networks with structural priors grounded in physics, control and estimation theory, enabling: 

- **Data Efficiency**: By embedding structural priors, MS-NNs can learn effectively from limited data, reducing the need for extensive datasets.
- **Generalization**: The incorporation of domain knowledge allows MS-NNs to generalize better to unseen scenarios.
- **Interpretability**: The structured nature of MS-NNs enhances interpretability, allowing practitioners to understand and trust the model's predictions.
- **Real-time**: MS-NNs can be designed for real-time applications, making them suitable for control and estimation tasks in dynamic environments.

The main objective of the nnodely framework is to allow fast prototyping of MS-NNs for modeling, estimation and control of physical systems by embedding structural priors knowledge into the networks' architecture.

In this documentation you will find a comprehensive guide for getting started with nnodely, illustrating the main blocks that constitute the framework.


.. raw:: html

   <p>
     <a href="https://github.com/tonegas/nnodely"
        style="display:inline-block; font-weight:900; font-size:1.4em;
               padding:0.7em 1.1em; border-radius:12px;
               border:2px solid #333; text-decoration:none;">
       For the README click here
     </a>
   </p>





.. raw:: html

   <p>
     <a href="https://github.com/tonegas/nnodely-applications"
        style="display:inline-block; font-weight:800; font-size:1.15em;
               padding:0.55em 0.9em; border-radius:10px;
               border:1px solid #333; text-decoration:none;">
       For the nnodely applications click here
     </a>
   </p>




Overview
-----------------------

.. This needs to be revised in order to explain at high level the phases of the workflow.
.. - :ref:`Modely <nnodely-modely>`: Main entry point of nnodely. It manages the composition of the MS-NNs, the connection between structural blocks and the training of the networks.
.. - :ref:`Model structured NN Inputs Outputs and Parameters <nnodely-msnn_ins_out_param>`: Description of the Input, Output and Parameter modules that can be used to build MS-NNs.
.. - :ref:`Model structured NN building blocks <nnodely-modules-layers>`: Overview of the different structural layers available in nnodely to build MS-NNs.
.. - :ref:`Training <nnodely-training>`: Explanation of the training procedures implemented in nnodely to train MS-NNs.



.. image:: https://raw.githubusercontent.com/tonegas/nnodely/docs/update/imgs/framework_p.png
   :width: 60%
   :alt: Framework

   
Overview of the *nnodely* development pipeline. It spans model design (:ref:`PH1 <nnodely-modely>`), dataset construction aligned with the network architecture (:ref:`PH2 <nnodely-dataset-creation>`), training (:ref:`PH3 <nnodely-training>`), domain-specific validation (:ref:`PH4 <nnodely-validation>`), model export (:ref:`PH5 <nnodely-export>`), and composition of complex models (:ref:`PH6 <nnodely-model-composition>`). Ellipses indicate the pipeline phases, while rectangles denote the artifacts produced at each phase.



Table of Contents
==========================
.. toctree::
   :maxdepth: 2

   _autodoc/getting_started/index
   _autodoc/model_definition/index
   _autodoc/dataset_creation/index
   _autodoc/model_composition/index
   _autodoc/training/index
   _autodoc/validation/index
   _autodoc/export/index
   _autodoc/tutorials/index
   _autodoc/citations/index
   _autodoc/q_and_a/index
   _autodoc/glossary/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

