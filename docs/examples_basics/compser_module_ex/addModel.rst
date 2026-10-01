.. code-block:: python

  model = Modely()
  x = Input('x')
  out = Output('out', Fir(x.last()))
  model.addModel('example_model', [out])
