Load data from files:

.. code-block:: python

    import numpy as np
    from nnodely.basic.model import Modely
    from nnodely.layers.input import Input
    from nnodely.layers.output import Output
    from nnodely.layers.fir import Fir

    x = Input('x')
    y = Input('y')
    out = Output('out', Fir(x.tw(0.05)))

    test = Modely(visualizer=None)
    test.addModel('example_model', out)
    test.neuralizeModel(0.01)

    data_struct = ['x', '', 'y']
    test.loadData(name='example_dataset', source='path/to/data', format=data_struct)


Load data from a crafted dataset:

.. code-block:: python

    x = Input('x')
    y = Input('y')
    out = Output('out', Fir(x.tw(0.05)))

    test = Modely(visualizer=None)
    test.addModel('example_model', out)
    test.neuralizeModel(0.01)

    data_x = np.array(range(10))
    dataset = {'x': data_x, 'y': (2*data_x)}
    test.loadData(name='example_dataset', source=dataset)
