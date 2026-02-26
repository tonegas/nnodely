.. _input_module:

Input module
================

The **Input module** defines the interface between the model and the data.

Each input:

- Has a unique name and defined dimensionality
- Can be associated with a temporal context window
- Can access past and/or future samples

Inputs may originate from external datasets or from internal connections to other model components. This flexibility enables complex signal routing and the construction of composite model architectures.

.. automodule:: nnodely.layers.input
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.input.Input
        :undoc-members:
        :inherited-members:
        :exclude-members: count, resetCount
