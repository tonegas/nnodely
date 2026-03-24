"""
Example: nnodely_new with NetworkX graph and Keras 3.0 backend.

Minimal example: Input -> sw(window) -> Fir -> Output

Requisiti: pip install -r requirements.txt
Set KERAS_BACKEND before any import (avoids TensorFlow when not installed).
"""
# import os
# os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from nnodely import Input, Fir, Output, Model

# Optional: set Keras backend (torch, tensorflow, jax)
import keras
keras.config.set_backend("torch")

def main():
    x = Input('x', dim=1)

    window_size = 10
    x_stream = x.sw(window_size)

    fir = Fir(out_features=2)
    x_fir = fir(x_stream)

    x_out = Output('x_pred', x_fir)
    model = Model('predictor', inputs=[x], outputs=x_out)
    model.minimize('error', x_out, x.next(), loss='mse')

    batch_size = 4
    dummy_input = np.random.randn(batch_size, window_size, 1).astype(np.float32)
    result = model({'x': dummy_input})
    print("Input shape:", dummy_input.shape)
    print("Output shape:", result.shape)
    print("Output:", result)

    ## Train model
    model.train(train_data={'x': dummy_input}, epochs=5)

if __name__ == '__main__':
    main()
