.. code-block:: python

  input1 = Input('input1')
  input2 = Input('input2')

  def my_function(x, y, param1, const1):
      return param1 * x + const1 * y

  param_fun = ParamFun(
      my_function,
      constants={'const1': 1.0},
      parameters_dimensions={'param1': 1}
  )
  result = param_fun(input1, input2)
