Basic usage:

.. code-block:: python

  x = Input('x')
  fuz = Fuzzify(output_dimension=5, range=[1,5])
  out = Output('out', fuz(x.last()))


Passing the centers:

.. code-block:: python

  fuz = Fuzzify(centers=[-1,0,3,5], functions='Rectangular')
  out = Output('out', fuz(x.last()))


Using a custom function:

.. code-block:: python

  def fun(x):
      import torch
      return torch.tanh(x)

  fuz = Fuzzify(output_dimension=11, range=[-5,5], functions=fun)
  out = Output('out', fuz(x.last()))
