Select a sample window considering a signal::

   T = [-3,-2,-1,0,1,2]

where time 0 is the last passed instant.

.. code-block:: python

   T.sw(2)        # [-1, 0]
   T.sw([-2,0])   # [-1, 0]
   T.sw([0,1])    # [1]
   T.sw([-4,-2])  # [-3,-2]

With offset:

.. code-block:: python

   T.sw(2, offset=-2)      # [0, 1]
   T.sw([-2,2], offset=-1)
