Modely
======

..  automodule:: nnodely.nnodely
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: resultAnalysis, getWorkspace

Model structured NN Inputs Outputs and Parameters
=================================================

Input module
------------

.. automodule:: nnodely.layers.input
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.input.Input
        :undoc-members:
        :inherited-members:
        :exclude-members: count, resetCount

Output module
-------------

.. automodule:: nnodely.layers.output
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.output.Output
        :undoc-members:
        :inherited-members:
        :exclude-members: count, resetCount, closedLoop, connect, tw, sw, z

Parameter module
----------------

.. automodule:: nnodely.layers.parameter
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.parameter.Constant
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.parameter.SampleTime
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.parameter.Parameter
        :undoc-members:
        :no-inherited-members:

Initializer module
^^^^^^^^^^^^^^^^^^

    .. automodule:: nnodely.support.initializer
        :undoc-members:
        :no-inherited-members:

        .. autofunction:: nnodely.support.initializer.init_constant
        .. autofunction:: nnodely.support.initializer.init_negexp
        .. autofunction:: nnodely.support.initializer.init_exp
        .. autofunction:: nnodely.support.initializer.init_lin

Relation module
---------------

.. automodule:: nnodely.basic.relation
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.basic.relation.Stream
        :members:
        :no-inherited-members:

Model structured NN building blocks
===================================

Activation module
-----------------

.. automodule:: nnodely.layers.activation
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.activation.Relu
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.activation.ELU
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.activation.Sigmoid
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.activation.Softmax
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.activation.Identity
        :undoc-members:
        :no-inherited-members:

Arithmetic module
-----------------

.. automodule:: nnodely.layers.arithmetic
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.arithmetic.Add
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.arithmetic.Sub
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.arithmetic.Mul
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.arithmetic.Div
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.arithmetic.Pow
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.arithmetic.Neg
        :undoc-members:
        :no-inherited-members:

Fir module
----------

.. automodule:: nnodely.layers.fir
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.fir.Fir
        :undoc-members:
        :no-inherited-members:


Fuzzify module
--------------

.. automodule:: nnodely.layers.fuzzify
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.fuzzify.Fuzzify
        :undoc-members:
        :no-inherited-members:

Equation Learner module
-----------------------

.. automodule:: nnodely.layers.equationlearner
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.equationlearner.EquationLearner
        :undoc-members:
        :no-inherited-members:

Linear module
-------------

.. automodule:: nnodely.layers.linear
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.linear.Linear
        :undoc-members:
        :no-inherited-members:


Localmodel module
-----------------

.. automodule:: nnodely.layers.localmodel
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.localmodel.LocalModel
        :undoc-members:
        :no-inherited-members:

Part module
-----------

.. automodule:: nnodely.layers.part
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.part.Part
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.part.Select
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.part.SamplePart
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.part.SampleSelect
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.part.TimePart
        :undoc-members:
        :no-inherited-members:

Trigonometric module
--------------------

.. automodule:: nnodely.layers.trigonometric
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.trigonometric.Sin
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.trigonometric.Cos
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.trigonometric.Tan
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.trigonometric.Tanh
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.trigonometric.Cosh
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.layers.trigonometric.Sech
        :undoc-members:
        :no-inherited-members:

Parametric Function module
--------------------------

.. automodule:: nnodely.layers.parametricfunction
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.parametricfunction.ParamFun
        :undoc-members:
        :no-inherited-members:

Training
========

optimizer module
----------------

.. automodule:: nnodely.basic.optimizer
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.basic.optimizer.SGD
        :undoc-members:
        :inherited-members:
        :exclude-members: add_option_to_params, replace_key_with_params, get_torch_optimizer

    .. autoclass:: nnodely.basic.optimizer.Adam
        :undoc-members:
        :inherited-members:
        :exclude-members: add_option_to_params, replace_key_with_params, get_torch_optimizer

earlystopping module
--------------------

.. automodule:: nnodely.support.earlystopping
    :undoc-members:
    :no-inherited-members:

    .. autofunction:: nnodely.support.earlystopping.early_stop_patience
    .. autofunction:: nnodely.support.earlystopping.select_best_model
    .. autofunction:: nnodely.support.earlystopping.mean_stopping
    .. autofunction:: nnodely.support.earlystopping.standard_early_stopping

Additional information
======================

.. include:: overview.md
   :parser: myst