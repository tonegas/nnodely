Select the unitary delay considering a signal::

    T = [-3, -2, -1, 0, 1, 2]

where the time vector 0 represents the last passed instant.

.. code-block:: python

    T.z(-1)  # = 1
    T.z(0)   # = 0  # the last passed instant
    T.z(2)   # = -2
