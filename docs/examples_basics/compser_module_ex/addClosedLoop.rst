.. code-block:: python

  model = Modely()
  x = Input('x')
  y = Input('y')
  relation = Fir(x.last())
  model.addClosedLoop(relation, y)
  