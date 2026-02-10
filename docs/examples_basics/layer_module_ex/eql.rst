Basic usage:

.. code-block:: python

  x = Input('x')

  equation_learner = EquationLearner(functions=[Tan, Sin, Cos])
  out = Output('out', equation_learner(x.last()))


Passing a linear layer:

.. code-block:: python

  x = Input('x')

  linear_layer = Linear(
      output_dimension=3,
      W_init=init_constant,
      W_init_params={'value': 0}
  )

  equation_learner = EquationLearner(
      functions=[Tan, Sin, Cos],
      linear_in=linear_layer
  )

  out = Output('out', equation_learner(x.last()))


Passing a custom parametric function and multiple inputs:

.. code-block:: python

  x = Input('x')
  F = Input('F')

  def myFun(K1, p1):
      return K1 * p1

  K = Parameter('k', dimensions=1, sw=1, values=[[2.0]])
  parfun = ParamFun(myFun, parameters=[K])

  equation_learner = EquationLearner([parfun])
  out = Output('out', equation_learner((x.last(), F.last())))
