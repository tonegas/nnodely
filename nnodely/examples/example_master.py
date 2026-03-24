import numpy as np
from nnodely import Input, Fir, Output, Model

# Optional: set Keras backend (torch, tensorflow, jax)
import keras
import torch
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

def main(example=1):
    if example == 1: ## Model Composition
        x = Input('x', dim=1)

        window_size = 10
        x_stream = x.sw(window_size)

        fir = Fir(out_features=2)
        x_fir = fir(x_stream)

        x_out = Output('x_pred', x_fir)
        model1 = Model('model1', inputs=[x], outputs=x_out)
        model1.build()  # Build model1 to ensure it's ready for composition

        y = Input('y', dim=1)
        y_stream = y.sw(window_size)
        y_fir = Fir(out_features=1)(model1(y_stream))
        y_out = Output('y_pred', y_fir)
        model2 = Model('composed_model', inputs=[y], outputs=y_out)

        batch_size = 4
        dummy_input = np.ones((batch_size, window_size, 1), dtype=np.float32)

        ## Model1 prediction
        result1 = model1({'x': dummy_input})
        print("Model1 - Input shape:", dummy_input.shape)
        print("Model1 - Output shape:", result1.shape)
        print("Model1 - Output:", result1)

        ## Model2 prediction
        result2 = model2({'y': dummy_input})
        print("Model2 - Input shape:", dummy_input.shape)
        print("Model2 - Output shape:", result2.shape)
        print("Model2 - Output:", result2)

    if example == 2: ## Model Training
        # Simple linear fitting example: y = a*x + b
        x = Input('x', dim=1)

        # Use a window of 1 so the model acts as a simple linear regressor
        window_size = 1
        x_stream = x.sw(window_size)

        # FIR with out_features=1 acts like a linear combination over the window
        fir = Fir(out_features=1)
        x_fir = fir(x_stream)

        x_out = Output('x_pred', x_fir)
        model1 = Model('linear_fit', inputs=[x], outputs=x_out)
        model1.minimize('error', source=x_out, target='x_target', loss='mse')

        # Create synthetic linear dataset y = a*x + b + noise
        rng = np.random.RandomState(0)
        n_samples = 256
        true_a = 2.5
        true_b = 0.7
        X_np = rng.randn(n_samples, window_size, 1).astype(np.float32)
        Y_np = (true_a * X_np[:, -1:, :] + true_b).astype(np.float32)
        Y_np += (0.05 * rng.randn(*Y_np.shape)).astype(np.float32)

        # Convert to torch tensors (CPU) for the custom torch-backed training loop
        device = torch.device('mps:0')
        X = torch.from_numpy(X_np).to(device=device)
        Y = torch.from_numpy(Y_np).to(device=device)

        # Train the model to recover the linear relation
        model1.train(train_data={'x': X, 'x_target': Y}, epochs=20, batch_size=4)

        # Evaluate and print a few predictions vs targets
        preds = model1({'x': X[:5]})
        print("\nPredictions:", preds)
        targets = Y[:5]
        print("Targets:", targets)
        model1.plot(to_file='model1.png')

        print('Training MODEL 2 - with model composition')
        y = Input('y', dim=1)
        y_stream = y.sw(1)
        y_fir = Fir(out_features=1)(y_stream)
        y_out = Output('y_pred', model1(y_fir))

        model2 = Model('model2', inputs=[y], outputs=y_out)
        model2.minimize('error', source=y_out, target='y_target', loss='mse')

        # Create synthetic linear dataset y = a*x + b + noise
        rng = np.random.RandomState(0)
        n_samples = 256
        true_a = -3.2
        true_b = 1.5
        X = rng.randn(n_samples, window_size, 1).astype(np.float32)
        Y = (true_a * X[:, -1:, :] + true_b).astype(np.float32)
        Y += (0.05 * rng.randn(*Y.shape)).astype(np.float32)
        X = torch.from_numpy(X_np).to(device=device)
        Y = torch.from_numpy(Y_np).to(device=device)

        # Train the model to recover the linear relation
        model2.train(train_data={'y': X, 'y_target': Y}, epochs=10, batch_size=4)

        # Evaluate and print a few predictions vs targets
        preds = model2({'y': X[:5]})
        print("\nPredictions:", preds)
        targets = Y[:5]
        print("Targets:", targets)
        model2.plot(to_file='model2.png') 

    if example == 3: ## Model Training with multi-loss
        # Simple linear fitting example: y = a*x + b
        x = Input('x', dim=1)
        y = Input('y', dim=1)
        # Use a window of 1 so the model acts as a simple linear regressor
        x_stream = x.sw(10)
        y_stream = y.sw(5)

        # FIR with out_features=1 acts like a linear combination over the window
        fir_x = Fir(out_features=1)(x_stream)
        fir_y = Fir(out_features=1)(y_stream)

        x_out = Output('total', fir_x+fir_y)
        fir_x = Output('x_part', fir_x)
        fir_y = Output('y_part', fir_y)
        model1 = Model('linear_fit', inputs=[x, y], outputs=[x_out, fir_x, fir_y])
        model1.minimize('error_x', source=fir_x, target='x_target', loss='mse')
        model1.minimize('error_y', source=fir_y, target='y_target', loss='mse')

        # Create synthetic linear dataset k = a*x + c + noise , z = b*y + c + noise
        rng = np.random.RandomState(0)
        n_samples = 256
        true_a = 2.5
        true_b = 0.7
        true_c = 1.0
        X_np = rng.randn(n_samples, 10, 1).astype(np.float32)
        Y_np = rng.randn(n_samples, 5, 1).astype(np.float32)
        K_np = (true_a * X_np[:, -1:, :] + true_c).astype(np.float32)
        K_np += (0.05 * rng.randn(*K_np.shape)).astype(np.float32)
        Z_np = (true_b * Y_np[:, -1:, :] + true_c).astype(np.float32)
        Z_np += (0.05 * rng.randn(*Z_np.shape)).astype(np.float32)

        # Convert to torch tensors (CPU) for the custom torch-backed training loop
        device = torch.device('mps:0')
        X = torch.from_numpy(X_np).to(device=device)
        Y = torch.from_numpy(Y_np).to(device=device)
        K = torch.from_numpy(K_np).to(device=device)
        Z = torch.from_numpy(Z_np).to(device=device)
        # Train the model to recover the linear relation
        model1.train(train_data={'x': X, 'y': Y, 'x_target': K, 'y_target': Z}, epochs=20, batch_size=4)
        model1.plot(to_file='example3.png')

        ## make predictions and plot them
        preds = model1({'x': X[:5], 'y': Y[:5]})
        print("\nPredictions:", preds)
        targets = {'x_target': K[:5], 'y_target': Z[:5]}
        print("Targets:", targets)

    if example == 4: ## Minimize to zero without target
        x = Input('x', dim=1)
        fir_x = Fir(out_features=1)(x.sw(5))
        x_out = Output('x_pred', fir_x)
        model = Model('minimize', inputs=[x], outputs=x_out)
        model.minimize('error', source=x_out, loss='mse')

        # Create synthetic linear dataset y = a*x + b + noise
        n_samples = 256
        X = rng.randn(n_samples, 5, 1).astype(np.float32)
        X = torch.from_numpy(X_np).to(device=torch.device('mps:0'))

        # Train the model to recover the linear relation
        model1.train(train_data={'x': X}, epochs=20, batch_size=4)

        # Evaluate and print a few predictions vs targets
        preds = model1({'x': X[:5]})
        print("\nPredictions:", preds)
        targets = Y[:5]
        print("Targets:", targets)
        model1.plot(to_file='example4.png')

    if example == 5: ## Load a dataset and train a model
        x = Input('x', dim=1)
        x_fir = Fir(out_features=1)(x.sw(5))
        x_out = Output('x_pred', x_fir)
        model1 = Model('linear_fit', inputs=[x], outputs=[x_out])
        model1.minimize('error', source=x_out, target=Input('x_target', dim=1).sw(1), loss='mse')

        ## load dataset
        from nnodely.dataloader import DataLoader
        data_train = DataLoader(model1, format={'x': 'data_1', 'x_target': 'data_2'}, folder='data')

        #print('data_train: ', data_train.dataset.items())
        print('first element: ', data_train[0])

        model1.train(train_data=data_train, epochs=20, batch_size=4)

    if example == 6: ## Export model graph to HTML - simple model
        x = Input('x', dim=1)
        x_fir = Fir(out_features=1)(x.sw(5))
        x_out = Output('x_pred', x_fir)
        model1 = Model('linear_fit', inputs=[x], outputs=x_out)

        #from nnodely.utils.plot import make_html
        #make_html(model1, out_dir='html')
        model1.export_html(out_dir='html', filename='model1')

    if example == 7: ## Export model graph to HTML - composed model
        x = Input('x', dim=1)
        x_target = Input('x_target', dim=1)
        x_stream = x.sw(1)
        fir = Fir(out_features=1)
        x_fir = fir(x_stream)
        x_out = Output('x_pred', x_fir)
        x_target_out = Output('x_target_out', x_target.sw(1))
        model1 = Model('linear_fit', inputs=[x, x_target], outputs=[x_out, x_target_out])
        model1.minimize('error', source=x_out, target=x_target_out, loss='mse')
        model1.plot(to_file='model1_no_minimizers.png', include_minimizers=False)
        model1.plot(to_file='model1_with_minimizers.png', include_minimizers=True)
        # from nnodely.utils.plot import make_html
        # make_html(model1, filename='model1_not_builded', out_dir='html')
        model1.build()
        # make_html(model1, filename='model1_builded', out_dir='html')

        y = Input('y', dim=1)
        y_stream = y.sw(1)
        y_fir = Fir(out_features=1)(y_stream)
        y_out = Output('y_pred', model1(y_fir))

        model2 = Model('model2', inputs=[y], outputs=y_out)
        model2.export_html(out_dir='html', filename='model2')
        model2.plot(to_file='model2_no_minimizers.png', include_minimizers=False)
        model2.plot(to_file='model2_with_minimizers.png', include_minimizers=True)
        # make_html(model2, filename='model2', out_dir='html')


if __name__ == '__main__':
    main(example=6)