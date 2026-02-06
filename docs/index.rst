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

   _autodoc/model_definition/index
   _autodoc/dataset_creation/index
   _autodoc/model_composition/index
   _autodoc/training/index
   _autodoc/validation/index
   _autodoc/export/index
   _autodoc/tutorials/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



Reacher example
---------------

.. image:: https://raw.githubusercontent.com/tonegas/nnodely/docs/update/imgs/Reacher2j.png
   :width: 40%
   :alt: Reacher

Here is simple two-joint planar manipulator. The inputs are the joint angles :math:`\theta_1` and :math:`\theta_2`, while the outputs are the end-effector coordinates :math:`(x_{\text{out}}, y_{\text{out}})`.
The link lengths :math:`l_1` and :math:`l_2` are unknown and are estimated from data using *nnodely* as learnable parameters.
The kinematic model is given by:

.. math::   
   x_{\text{out}} = l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2), \quad  
   y_{\text{out}} = l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2).

.. code-block:: python

   # Inputs from dataset
   theta1 = Input('theta1')
   theta2 = Input('theta2')
   x_tip  = Input('x_tip')
   y_tip  = Input('y_tip')

   l1 = Parameter('l1')  #parameters to be estimated
   l2 = Parameter('l2')  #parameters to be estimated
   
   x_out = Output('x_out', (l1 * Cos(theta1.last())) +
          (l2 * Cos(theta1.last() + theta2.last())))
   y_out = Output('y_out', (l1 * Sin(theta1.last())) +
          (l2 * Sin(theta1.last() + theta2.last())))

   # Model composition 
   model = Modely(seed=0)
   model.addModel('x_out', x_out)
   model.addModel('y_out', y_out)
   model.addMinimize('x-error', x_tip.last(), x_out, 'mse') # Objectives
   model.addMinimize('y-error', y_tip.last(), y_out, 'mse') # Objectives
   model.neuralizeModel(sample_time=0.02) 

   # Data loading 
   data_struct = ['step', 'T1','T2','theta1', 'theta2', 'x_tip', 'y_tip',
                  'thetadot1', 'thetadot2', 'thetaddot1', 'thetaddot2']

   data_folder = os.path.join(os.getcwd(), 'dataset', 'data')

   # dataset creation
   model.loadData(name='reacher_data', source=data_folder,
               format=data_struct, delimiter=';')

   # Training
   train_params = {'num_of_epochs': 200, 'train_batch_size': 128, 'lr': 0.01}
   model.trainModel(splits=[70, 20, 10], training_params=train_params)

   model.neuralizeModel()
