.. code-block:: python

    x = Input('x')
    F = Input('F')

    xk1 = Output('x[k+1]', Fir()(x.tw(0.2)) + Fir()(F.last()))

    mass_spring_damper = Modely(seed=0)
    mass_spring_damper.addModel('xk1', xk1)
    mass_spring_damper.neuralizeModel(sample_time=0.05)

    data_struct = ['time', 'x', 'dx', 'F']
    data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'data')
    mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

    params = {'num_of_epochs': 100, 'train_batch_size': 128, 'lr': 0.001}
    mass_spring_damper.trainModel(splits=[70, 20, 10], training_params=params)


Example - recurrent training:

.. code-block:: python

    x = Input('x')
    F = Input('F')

    xk1 = Output('x[k+1]', Fir()(x.tw(0.2)) + Fir()(F.last()))

    mass_spring_damper = Modely(seed=0)
    mass_spring_damper.addModel('xk1', xk1)
    mass_spring_damper.addClosedLoop(xk1, x)
    mass_spring_damper.neuralizeModel(sample_time=0.05)

    data_struct = ['time', 'x', 'dx', 'F']
    data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'data')
    mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

    params = {'num_of_epochs': 100, 'train_batch_size': 128, 'lr': 0.001}
    mass_spring_damper.trainModel(splits=[70, 20, 10], prediction_samples=10, training_params=params)
