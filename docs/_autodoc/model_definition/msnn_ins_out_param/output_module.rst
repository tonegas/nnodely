.. _output_module:

Output module
================

The **Output module** exposes selected relations as model predictions and define the interface between the internal architecture and user-level quantities of interest.

Each output is associated with a specific relation and identified by a semantic tag.

Outputs affect training only when explicitly included in the loss function; otherwise, they serve as access points for inference and analysis.

.. automodule:: nnodely.layers.output
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.layers.output.Output
        :undoc-members:
        :inherited-members:
        :exclude-members: count, resetCount, closedLoop, connect, tw, sw, z
