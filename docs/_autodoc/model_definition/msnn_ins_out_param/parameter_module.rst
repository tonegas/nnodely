.. _parameter_module:

Parameter module
================

The **Parameter module** represents the learnable quantities of the model and explicitly indicate which parts of the architecture are estimated from data.

Each parameter:

- Has a unique name
- Has a defined dimensionality
- Can be associated with temporal windows

This ensures unambiguous reference, even in large and structured models.

Parameters can be defined over time to capture dynamic effects. Their initialization is considered an integral part of the modeling process and is handled by the :ref:`initializer_module`.

In addition to standard random initialization, **nnodely** provides structured initialization methods, such as smooth interpolation or decaying profiles. These methods allow prior physical knowledge to be embedded directly into the learnable representation.

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

For more examples of how to use the parameter module, please refer to the :doc:`relative tutorial <../../tutorials/examples/parameter>`.
