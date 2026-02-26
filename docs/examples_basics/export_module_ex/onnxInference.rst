.. code-block:: python

    import numpy as np
    from nnodely.basic.model import Modely
    from nnodely.layers.input import Input

    x = Input('x')

    model_folder = "folder/"

    dummy_input = {
        'x': np.ones(shape=(3, 1, 1)).astype(np.float32)
    }

    predictions = Modely().onnxInference(dummy_input, model_folder)


Example - Recurrent:

.. code-block:: python

    import numpy as np
    from nnodely.basic.model import Modely
    from nnodely.layers.input import Input

    x = Input('x')
    y = Input('y')

    model_folder = "folder/"

    dummy_input = {
        'x': np.ones(shape=(3, 1, 1, 1)).astype(np.float32),
        'y': np.ones(shape=(1, 1, 1)).astype(np.float32)
    }

    predictions = Modely().onnxInference(dummy_input, model_folder)