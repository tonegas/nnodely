.. code-block:: python

  model = Modely()
  model.loadData('dataset_name', 'path/to/data')

  def filter_fn(sample):
      return sample['input1'] > 0

  model.filterData(filter_fn, 'dataset_name')
