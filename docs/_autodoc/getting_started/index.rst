.. _nnodely-getting-started:

Getting Started
================

.. raw:: html

   <p>
     <a href="https://github.com/tonegas/nnodely"
        style="display:inline-block; font-weight:800; font-size:1.15em;
               padding:0.55em 0.9em; border-radius:10px;
               border:2px solid #2306fc; text-decoration:none;">
       First start with the README. click here!
     </a>
   </p>

Installation guide
------------------
To install *nnodely*, the user can install via:

.. code-block:: bash

   pip install nnodely

Alternatively, the user can clone the repository and install from source:

.. code-block:: bash

   git clone https://github.com/tonegas/nnodely.git
   cd nnodely
   pip install -r requirements.txt
   pip install .



Reacher example
---------------

Before reading this example, start with the basic example in the README (linked above).


.. image:: https://raw.githubusercontent.com/tonegas/nnodely/docs/update/imgs/Reacher2j.png
   :width: 40%
   :alt: Reacher

Here is simple two-joint planar manipulator. The inputs are the joint angles :math:`\theta_1` and :math:`\theta_2`, while the outputs are the end-effector coordinates :math:`(x, y)`.
The link lengths :math:`l_1` and :math:`l_2` are unknown and are estimated from data using *nnodely* as learnable parameters.
The kinematic model is given by:


.. math::   
   x = l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2), \quad  
   y = l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2).

**Inputs from dataset & Parameters**

Input variables are created using the :class:`Input` class. The learnable parameters are given within the :class:`Parameter`. The :class:`Output` class defines the model output and takes two arguments: 
the name of the output and its structure.


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

**Model composition**

:class:`addModel` adds the defined output to the model, :class:`addMinimize` defines the loss function, 
and :class:`neuralizeModel` builds the discrete-time MS-NN.

.. code-block:: python

   # Model composition 
   model = Modely(seed=0)
   model.addModel('x_out', x_out)
   model.addModel('y_out', y_out)
   model.addMinimize('x-error', x_tip.last(), x_out, 'mse') # Objectives
   model.addMinimize('y-error', y_tip.last(), y_out, 'mse') # Objectives
   model.neuralizeModel(sample_time=0.02) 

**Data loading**

Nnodely requires two pieces of information: the data structure and the dataset location.

.. code-block:: python

   data_struct = ['step', 'T1','T2','theta1', 'theta2', 'x_tip', 'y_tip',
                  'thetadot1', 'thetadot2', 'thetaddot1', 'thetaddot2'] # dataset creation
   
   data_folder = os.path.join(os.getcwd(), 'dataset', 'data')
   
   model.loadData(name='reacher_data', source=data_folder,
               format=data_struct, delimiter=';')  # Data loading 

**Training**

.. code-block:: python

   # Training
   train_params = {'num_of_epochs': 200, 'train_batch_size': 128, 'lr': 0.01}
   model.trainModel(splits=[70, 20, 10], training_params=train_params)

   model.neuralizeModel()

For additional examples, please refer to the two links below.


.. raw:: html

   <p>
     <a href="https://github.com/tonegas/nnodely-applications"
        style="display:inline-block; font-weight:800; font-size:1.15em;
               padding:0.55em 0.9em; border-radius:10px;
               border:2px solid #0b93e1; text-decoration:none;">
       for nnodely applications click here
     </a>
   </p>



.. raw:: html

   <p>
     <a href="../tutorials/index.html"
        style="display:inline-block; font-weight:800; font-size:1.15em;
               padding:0.55em 0.9em; border-radius:10px;
               border:2px solid #5d8007; text-decoration:none;">
       for tutorials click here
     </a>
   </p>

