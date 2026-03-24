#!/usr/bin/env python3
"""
Example: nnodely_new with NetworkX graph and Keras 3.0 backend.

Minimal example: Input -> sw(window) -> Fir -> Output

Requisiti: pip install -r requirements.txt
Set KERAS_BACKEND before any import (avoids TensorFlow when not installed).
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from nnodely import Input, Fir, Output, Model

import keras

# Optional: set Keras backend (torch, tensorflow, jax)
# import keras
# keras.config.set_backend("torch")

def main():
    # 1. Define inputs
    x = Input('x', dim=1)

    # 2. Create sample window stream (10 past samples)
    window_size = 10
    x_stream = x.sw(window_size)

    # 3. Apply FIR layer
    fir = Fir(out_features=2)
    x_fir = fir(x_stream)

    # 4. Create output
    x_out = Output('x_pred', x_fir)


    model = Model('predictor', inputs=[x], outputs=x_out)


    # 7. Test inference
    batch_size = 4
    dummy_input = np.random.randn(batch_size, window_size, 1).astype(np.float32)
    result = model({'x': dummy_input})
    print("Input shape:", dummy_input.shape)
    print("Output shape:", result.shape)
    print("Output:", result)

    x2 = Input('x2', dim=1)
    fir2 = Fir(out_features=1)
    x2_out = Output('y', fir2(x2.sw(5)))

    model2 = Model('mod2', {'x2': x2}, x2_out)
    stream_out = model2({'x2': Input('x2', dim=1).sw(5)})  # stream: composizione grafo
    print("Model2 (unbuilt) stream out:", stream_out.name)
    out2 = model2({'x2': np.random.randn(2, 5, 1).astype(np.float32)})  # tensori: build on the fly
    print("Model2 (unbuilt) tensor out (build on fly):", out2.shape)

    model2.build()
    out2b = model2({'x2': np.random.randn(2, 5, 1).astype(np.float32)})
    print("Model2 (built) tensor out (build on fly):", out2b.shape, out2b)
    stream_out2 = model2({'x2': Input('x2', dim=1).sw(5)})  # buildato + stream compatibile: ok
    print("Model2 (built) stream compatibile:", stream_out2.name, stream_out2)

    # 9. Model call syntax
    x3 = Input('u', dim=1)
    y = Output('y', Fir(out_features=1)(x3.sw(8)))

    model3 = Model('ctrl', inputs={'u': x3}, outputs=y)
    print("\nModel 3:", model3({'u': np.random.randn(1, 8, 1).astype(np.float32)}))

    # 10. Model callable with stream (graph composition) - model2 non buildato

    x2 = Input('x2', dim=1)
    model2 = Model('mod2', {'x2': x2}, Output('y', Fir(1)(x2.sw(5))))
    x4 = Input('x4', dim=1)
    y4 = Output('y4', model2({'x2': x4.sw(5)}))  # model2 con stream, dim dinamiche
    model4 = Model('mod4', {'x4': x4}, y4).build()
    out4 = model4({'x4': np.random.randn(2, 5, 1).astype(np.float32)})
    print("\nModel 4 output:", out4.shape)

if __name__ == '__main__':
    main()
