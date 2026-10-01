.. code-block:: python

  input1 = Input('input1').last()
  input2 = Input('input2').last()
  out = Output('output1', input1+input2)

  model = Modely()
  model.neuralizeModel()
  model.exportONNX(inputs_order=['input1', 'input2'], outputs_order=['output1'], name='example_model', model_folder='path/to/export')
