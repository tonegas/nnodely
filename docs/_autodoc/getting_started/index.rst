.. _nnodely-getting-started:

Getting Started
================

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