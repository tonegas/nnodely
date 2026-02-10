Basic usage:

.. code-block:: python

  x = Input('x')
  activation = Fuzzify(2, [0, 1], functions='Triangular')(x.last())
  loc = LocalModel(input_function=Fir())
  out = Output('out', loc(x.tw(1), activation))


Passing a custom function:

.. code-block:: python

  def myFun(in1, p1, p2):
      return p1 * in1 + p2

  x = Input('x')
  activation = Fuzzify(2, [0, 1], functions='Triangular')(x.last())
  loc = LocalModel(
      input_function=lambda: ParamFun(myFun),
      output_function=lambda: Fir
  )(x.last(), activation)
  out = Output('out', loc)


Custom function with multiple activations:

.. code-block:: python

  def myFun(in1, p1, p2):
      return p1 * in1 + p2

  x = Input('x')
  F = Input('F')
  activationA = Fuzzify(2, [0, 1], functions='Triangular')(x.tw(1))
  activationB = Fuzzify(2, [0, 1], functions='Triangular')(F.tw(1))

  loc = LocalModel(
      input_function=lambda: ParamFun(myFun),
      output_function=Fir(3)
  )(x.tw(1), (activationA, activationB))
  out = Output('out', loc)
