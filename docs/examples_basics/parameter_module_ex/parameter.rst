Basic usage:

.. code-block:: python

   k = Parameter('k', dimensions=3, tw=4)

Initialize a parameter with values:

.. code-block:: python

   x = Input('x')
   gravity = Parameter('g', dimensions=(4,1), values=[[[1],[2],[3],[4]]])
   out = Output('out', Linear(W=gravity)(x.sw(3)))

Initialize a parameter with a function:

.. code-block:: python

   x = Input('x').last()
   p = Parameter('param', dimensions=1, sw=1, init=init_constant, init_params={'value':1})
   relation = Fir(parameter=p)(x)
