Basic usage:

.. code-block:: python

  input = Input('in')
  relation = Fir(input.tw(0.05))

Passing a parameter:

.. code-block:: python

  input = Input('in')
  par = Parameter('par', dimensions=3, sw=2, init='init_constant')
  relation = Fir(W=par)(input.sw(2))

Parameters initialization:

.. code-block:: python

  x = Input('x')
  F = Input('F')
  fir_x = Fir(W_init='init_negexp')(x.tw(0.2))
  fir_F = Fir(W_init='init_constant', W_init_params={'value':1})(F.last())
