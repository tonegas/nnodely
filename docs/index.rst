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

Documentation Overview
-----------------------

- :ref:`Modely <nnodely-modely>`: Main entry point of nnodely. It manages the composition of the MS-NNs, the connection between structural blocks and the training of the networks.

- :ref:`Model structured NN Inputs Outputs and Parameters <nnodely-msnn_ins_out_param>`: Description of the Input, Output and Parameter modules that can be used to build MS-NNs.

- :ref:`Model structured NN building blocks <nnodely-modules-layers>`: Overview of the different structural layers available in nnodely to build MS-NNs.

- :ref:`Training <nnodely-training>`: Explanation of the training procedures implemented in nnodely to train MS-NNs.


.. image:: https://raw.githubusercontent.com/tonegas/nnodely/docs/update/imgs/framework_p.png
   :width: 60%
   :alt: Framework


Table of Contents
==========================
.. toctree::
   :maxdepth: 2

   _autodoc/modely/index
   _autodoc/msnn_ins_out_param/index
   _autodoc/layers/index
   _autodoc/training/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




Reacher example
---------------

.. code-block:: python

   # Inputs from dataset
   theta1 = Input('theta1')
   theta2 = Input('theta2')
   x_tip  = Input('x_tip')
   y_tip  = Input('y_tip')

   l1 = Parameter('l1')
   l2 = Parameter('l2')

   x_out = Output('x_out', (l1 * Cos(theta1.last())) + (l2 * Cos(theta1.last() + theta2.last())))
   y_out = Output('y_out', (l1 * Sin(theta1.last())) + (l2 * Sin(theta1.last() + theta2.last())))


   # Model container
   model = Modely(seed=0)
   model.addModel('x_out', x_out)
   model.addModel('y_out', y_out)

   # Objectives
   model.addMinimize('x-error', x_tip.last(), x_out, 'mse')
   model.addMinimize('y-error', y_tip.last(), y_out, 'mse')

   # sample_time consistent with CSV
   model.neuralizeModel(sample_time=0.02)

   # Data loading (CSV in current folder)

   data_struct = ['step', 'T1','T2','theta1', 'theta2', 'x_tip', 'y_tip',
                  'thetadot1', 'thetadot2', 'thetaddot1', 'thetaddot2']

   data_folder = os.path.join(os.getcwd(), 'dataset', 'data')

   # Load the CSV file
   model.loadData(
       name='reacher_data',
       source=data_folder,
       format=data_struct,
       delimiter=';')

   # Training
   train_params = {'num_of_epochs': 500, 'train_batch_size': 128, 'lr': 0.001}
   model.trainModel(splits=[70, 20, 10], training_params=train_params)

   model.neuralizeModel()
