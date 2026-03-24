#!/usr/bin/env python3
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
from nnodely import Input, Output, Model, Loop, Fir
from nnodely.dag import next_name, get_seq_time_dim, to_tuple
from nnodely.layer import Layer, LayerBase, _is_stream
from nnodely.stream import Stream
from tensorflow.keras.utils import plot_model

# =============================================================================
# Esempio d'uso
# =============================================================================
class Half(Layer):
    def build_layer(self):
        pass

    def call(self, x):
        return x * 0.5

def main():
    x = Input('x', dim=1)
    y = Input('y', dim=1)
    out = Output('out', Fir(x.sw(1)*y.sw(1)))
    m1 = Model('m1', [x, y], out).build()

    x_seq = Input('x_seq', seq=None, dim=1)
    y_seq = Input('y_seq', seq=None, dim=1)
    loop = Loop(m1)
    out2 = Output('out2', loop([x_seq.sw(1), y_seq.sw(1)]))  # Loop con cell_model m1, iterato su sequenza
    m2 = Model('m2', [x_seq, y_seq], out2).build()

    #plot_model(m2._model, to_file='m2.png', show_shapes=True, expand_nested=True, show_layer_names=True)

    # Test
    X = np.ones((1, 3, 1, 1)) * 32  # (B, S, T, D)
    Y = np.ones((1, 3, 1, 1)) * 0.5  # (B, S, T, D)
    print(m2({'x_seq': X, 'y_seq': Y}))
    print("loop out:", loop([X, Y]))
    print(m2([X, Y]))

if __name__ == "__main__":
    main()