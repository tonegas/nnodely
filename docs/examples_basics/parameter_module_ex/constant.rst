Passing a custom scalar value -> g.dim = {'dim': 1}:

.. code-block:: python

    g = Constant('gravity', values=9.81)

Passing a custom vector value -> n.dim = {'dim': 4}:

.. code-block:: python

    n = Constant('numbers', values=[1, 2, 3, 4])

Passing a custom vector value with single sample window -> n.dim = {'dim': 4, 'sw': 1}:

.. code-block:: python

    n = Constant('numbers', values=[[1, 2, 3, 4]])

Passing a custom vector value with double sample window -> n.dim = {'dim': 4, 'sw': 2}:

.. code-block:: python

    n = Constant('numbers', values=[[2, 3, 4], [1, 2, 3]])

Passing a custom vector value with double sample window -> n.dim = {'dim': 4, 'sw': 2}.
If the value of the sw is different from the dimension of shape[0] an error will be raised.

.. code-block:: python

    n = Constant('numbers', sw=2, values=[[2, 3, 4], [1, 2, 3]])

Passing a custom vector value with time window -> n.dim = {'dim': 4, 'tw': 4}.
In this case the sampling time must be 0.5; otherwise, an error will be raised. If the Constant has a time dimension,
the input must have len(shape) == 2.

.. code-block:: python

    n = Constant('numbers', tw=4, values=[[2, 3, 4], [1, 2, 3]])
