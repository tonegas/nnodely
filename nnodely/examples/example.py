import numpy as np
import pandas as pd

from nnodely.core.modely import Modely
from nnodely.layers.input import Input
from nnodely.layers.output import Output
from nnodely.layers.fir import Fir
from nnodely.core.dataloader import DataLoader
from nnodely.layers.parameter import Parameter
from nnodely.layers.constant import Constant

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

def main(example=1):
    if example == 1:
        # ------- Model definition and building -------
        x = Input('x', dim=1)
        y = Input('y', dim=1)

        window_size = 10
        x_stream = x.sw(window_size)
        y_stream = y.sw(window_size)

        fir = Fir(out_features=2)
        result_fir = fir(x_stream+y_stream)

        x_out = Output('x_pred', result_fir)
        model1 = Modely('model1', outputs=x_out)
        print(f'order: {model1._order}')
        model1.build()  
        print(f"Model1 - Output stream: {x_out}, shape: {x_out.shape} \n Input streams: {model1.inputs}, shape: {list(model1._input_shapes.values())}")

        # ------- Model composition -------
        z = Input('z', dim=1)
        z_stream = z.sw(window_size)
        z_fir = Fir(out_features=1)(model1({'x': z_stream, 'y': z_stream}))
        z_out = Output('z_pred', z_fir)
        model2 = Modely('composed_model', outputs=z_out)
        print(f'order: {model2._order}')
        model2.build() 
        print(f"Model2 - Output stream: {z_out}, shape: {z_out.shape} \n Input streams: {model2.inputs}, shape: {list(model2._input_shapes.values())}")

        # ------- Model inference -------
        batch_size = 4
        dummy_input_x = np.ones((batch_size, window_size, 1), dtype=np.float32)
        dummy_input_y = np.ones((batch_size, window_size, 1), dtype=np.float32)
        dummy_input_z = np.ones((batch_size, window_size, 1), dtype=np.float32)

        result1 = model1({'x': dummy_input_x, 'y': dummy_input_y})
        print("Model1 - Input shape:", dummy_input_x.shape)
        print("Model1 - Output shape:", result1['x_pred'].shape)
        print("Model1 - Output:", result1)

        result2 = model2({'z': dummy_input_z})
        print("Model2 - Input shape:", dummy_input_z.shape)
        print("Model2 - Output shape:", result2['z_pred'].shape)
        print("Model2 - Output:", result2)

        # ------- Model visualization -------
        model1.plot(to_file='html/model1.png')
        model2.plot(to_file='html/model2.png')

        # ------- Model export to HTML -------
        model1.export_html(out_dir='html', filename='model1')
        model2.export_html(out_dir='html', filename='model2')

    if example == 2:
        # ------- Model definition and training -------
        x = Input('x', dim=1)
        y = Input('y', dim=1)
        x_fir = Fir(out_features=1)(x.sw(5))
        y_fir = Fir(out_features=1)(y.sw(5))

        x_out = Output('fir_pred', x_fir+y_fir)
        model1 = Modely('linear_fit', outputs=[x_out])

        # ------- Define loss and minimizer -------
        model1.minimize('error', source=x_out, target=Input('x_target', dim=1).sw(1), loss='mse')
        model1.build()

        # ------ Load dataset -------
        data_train = DataLoader(model1, format={'x': 'data_1', 'y': 'data_2', 'x_target': 'data_3'}, source=os.path.join('nnodely','examples','data'))

        # ------ Train the model -------
        model1.train(train_data=data_train, epochs=60, batch_size=4)

        # ------ Remove minimizer and retrain with multi loss -------
        model1.remove_minimizer('error')
        model1.minimize('error_fir_x', source=x_fir, target=Input('x_target', dim=1).sw(1), loss='mse')
        model1.minimize('error_fir_y', source=y_fir, target=None, loss='mse') ## with target=None, the loss will be minimized to zero

        model1.build()  # Rebuild the model after removing the minimizer otherwise the training loop will still try to compute the loss and update the model based on it, even if it's not used for training anymore
        model1.train(train_data=data_train, epochs=60, batch_size=4)

        # ------ Visualize the model -------
        model1.plot(to_file='html/model1_trained.png')

        # ------ Remove one minimizer and retrain with a constant value -------
        model1.remove_minimizer('error_fir_y')
        model1.minimize('error_fir_y', source=y_fir, target=3.0, loss='mse') ## this will minimize the difference between y_fir and the constant value 3.0, effectively training the model to make y_fir close to 3.0
        model1.build()
        model1.train(train_data=data_train, epochs=60, batch_size=4)

    if example == 3:
        # ------- Model definition with Parameters and Constants -------
        x = Input('x', dim=1)
        param = Parameter('param1', value=[1.0])
        const = Constant('const1', value=[1.0])
        x_param = x.sw(1) * param + const
        x_out = Output('x_out', x_param)
        model = Modely('model', outputs=x_out)
        model.minimize('error', source=x_out, target=Input('x_target', dim=1).sw(1), loss='mse')
        model.build()

        # ------ Model inference -------
        batch_size = 4
        dummy_input_x = np.ones((batch_size, 1, 1), dtype=np.float32)
        result = model({'x': dummy_input_x})
        print("Output before training:", result)

        # ------ Create a simple dataset and train the model -------

        true_param = 3.5
        dataframe = {
            'x': np.ones((100, 1, 1), dtype=np.float32),
            'x_target': np.ones((100, 1, 1), dtype=np.float32)*true_param
        }
        data_train = DataLoader(model, source=dataframe)

        model.train(train_data=data_train, epochs=50, batch_size=16)
        print(f"Learned parameter value: {param.value_numpy}, True parameter value: {true_param - const.value_numpy}")
        print(f"Constant value after training: {const.value_numpy}, True constant value: {1.0}")

        # ------ Inference after training -------
        result_after_training = model({'x': dummy_input_x})
        print("Output after training:", result_after_training)

        # ------ Visualize the model -------
        model.plot(to_file='html/model_with_param.png')
        model.export_html(out_dir='html', filename='model_with_param')

    if example == 4:
        # ------- Model Flatten and Visualization -------
        x = Input('x', dim=1)
        param = Parameter('param1', dim=1)
        x_param = x.sw(5) * param
        x_out = Output('x_out', x_param)
        model1 = Modely('model1', outputs=x_out)
        model1.build()

        y = Input('y', dim=1)
        y_fir = Fir(out_features=1)(model1({'x': y.sw(5)}))
        y_out = Output('y_out', y_fir)
        model2 = Modely('model2', outputs=y_out)
        model2.build()

        z = Input('z', dim=1)
        const_z = Constant('const_z', value=[1.0, 2.0, 3.0, 4.0, 5.0])
        z_fir = Fir(out_features=1)(model2({'y': z.sw(5) + const_z}))
        z_out = Output('z_out', z_fir)
        model3 = Modely('model3', outputs=z_out)
        model3.build()

        # ------- Visualize the Flatten model -------
        model3.export_html(out_dir='html', filename='model3')
        model3.plot(to_file='html/model3_standard.png')
        model3.plot(to_file='html/model3_flattened.png', flatten=True)

    if example == 5:
        # ------- Model with closed loop connections -------
        from nnodely.layers.loop import Loop
        x = Input(name="x", dim=1)
        y = Input(name="y", dim=1)
        r1 = x.sw(1) + y.sw(1)
        out1 = Output("out1", r1)
        model1 = Modely(name="model1", outputs=[out1])

        z = Input(name="z", dim=1, seq=5)
        const = Constant('const', value=2.0)
        r2 = z.sw(1) * const
        loop_fn = Loop(f=model1, closed_loop={out1: z})
        out = Output("out", loop_fn(z,r2))
        model = Modely(name="model", outputs=[out])

        model.build()

    if example == 6:
        # ------- Model with time window partitioning, Past and Future -------
        x = Input('x', dim=1)
        y = Input('y', dim=1)
        z = Input('z', dim=1)
        r1 = x.sw(1)+y.sw(1)
        out1 = Output('out1', r1)
        model1 = Modely('model1', outputs=out1)
        model1.build()

        r2 = Fir(out_features=1)(z.sw(3))
        out2 = Output('out2', model1({'x': r2, 'y': z.sw(1)}))
        model2 = Modely('model2', outputs=out2)
        model2.build()

        k = Input('k', dim=1)
        f = Input('f', dim=1)
        r3 = model2({'z': k.sw(3)}) + model2({'z': f.sw(3)})
        out3 = Output('out3', r3)
        model3 = Modely('model3', outputs=out3)
        model3.build()
        pass

    if example == 7:
        # ------- Model Export and Import -------
        pass

    if example == 8:
        # ------- High-level Blocks (Local Models) with Multi-inputs -------
        pass

    if example == 9:
        # ------- Custom Layers -------
        pass

if __name__ == '__main__':
    main(example=4)