.. code-block:: python

  model = Modely()
  model.neuralizeModel()
  model.trainModel(train_dataset='train_dataset', validation_dataset='val_dataset', num_of_epochs=10)
  model.exportReport(name='example_model', model_folder='path/to/export')
