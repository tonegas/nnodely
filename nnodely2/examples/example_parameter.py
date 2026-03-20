#!/usr/bin/env python3
"""
Example: Parameter trainable - operazioni aritmetiche e inizializzazione Fir.

- Parameter(sw, dim): sw=dimensione temporale, dim=features -> shape=(sw,)+dim
- stream + param, param * stream, ecc.
- Fir(..., kernel_initializer=param) per inizializzare i pesi
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from nnodely import Input, Fir, Output, Model, Parameter


def main():
    print("=" * 60)
    print("Example: Parameter trainable")
    print("=" * 60)

    # 1. stream + param (addizione con parametro trainable)
    print("\n1. stream + param (offset trainable)")
    x = Input('x', dim=1)
    offset = Parameter('offset', sw=1, dim=1, initial_value=0.5)
    s = x.sw(5) + offset
    out = Output('y', Fir(out_features=1)(s))
    m1 = Model('m1', x, out).build()
    d = np.random.randn(2, 5, 1).astype(np.float32)
    r1 = m1(d)
    print(f"   Input shape: {d.shape} -> Output shape: {r1.shape}")

    # 2. param * stream (scaling trainable)
    print("\n2. param * stream (scaling trainable)")
    x2 = Input('x2', dim=1)
    scale = Parameter('scale', sw=1, dim=1, initial_value=1.0)
    s2 = scale * x2.sw(4)
    out2 = Output('y2', Fir(out_features=1)(s2))
    m2 = Model('m2', x2, out2).build()
    r2 = m2(np.random.randn(2, 4, 1).astype(np.float32))
    print(f"   Output shape: {r2.shape}")

    # 3. Fir con kernel_initializer da Parameter
    print("\n3. Fir con kernel_initializer da Parameter")
    # Per Fir con time=5, dim=1, out_features=1: Dense ha kernel (5, 1) = (sw, dim)
    T, D = 5, 12
    w_init = Parameter('w_init', sw=T, dim=D, initial_value=np.eye(T, 1, dtype=np.float32) * 0.1)
    x3 = Input('x3', dim=1)
    out3 = Output('y3', Fir(out_features=12, kernel_initializer=w_init)(x3.sw(T)))
    m3 = Model('m3', x3, out3).build()
    r3 = m3(np.random.randn(2, T, 1).astype(np.float32))
    print(f"   Output shape: {r3.shape}")

    # 4. Combinazione: (stream + param) * param2 -> Fir
    print("\n4. (stream + offset) * scale -> Fir")
    x4 = Input('x4', dim=1)
    off = Parameter('off', sw=1, dim=1, initial_value=0.0)
    sc = Parameter('sc', sw=1, dim=1, initial_value=1.0)
    s4 = (x4.sw(3) + off) * sc
    out4 = Output('y4', Fir(out_features=1)(s4))
    m4 = Model('m4', x4, out4).build()
    r4 = m4(np.random.randn(2, 3, 1).astype(np.float32))
    print(f"   Output shape: {r4.shape}")

    # 5. Parameter con sw e dim espliciti (shape = (sw,) + dim)
    print("\n5. Parameter(sw=3, dim=(2,)) -> shape (3, 2)")
    p = Parameter('p', sw=3, dim=(2,))
    print(f"   p.shape = {p.shape}, p.sw = {p.sw}, p.dim = {p.dim}")

    # 6. Verifica che i parametri siano trainable
    print("\n6. Parametri trainable nel modello")
    trainable = [w.name for w in m1._model.trainable_weights]
    print(f"   m1 trainable weights: {len(trainable)} parametri")

    print("\n" + "=" * 60)
    print("OK - example_parameter completato")
    print("=" * 60)


if __name__ == '__main__':
    main()
