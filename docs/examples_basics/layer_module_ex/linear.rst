Basic usage:

.. code-block:: python

  input = Input('in').tw(0.05)
  relation = Linear(input)


Passing a weight and bias parameter:

.. code-block:: python

  input = Input('in').last()

  weight = Parameter('W', values=[[[1]]])
  bias = Parameter('b', values=[[1]])

  relation = Linear(W=weight, b=bias)(input)


Parameters initialization:

.. code-block:: python

  input = Input('in').last()

  relation = Linear(
      b=True,
      W_init=init_negexp,
      b_init=init_constant,
      b_init_params={'value': 1}
  )(input)
