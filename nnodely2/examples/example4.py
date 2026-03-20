#!/usr/bin/env python3
"""
Example4: Input con dimensioni definite da tupla.

Testa dim come int o tuple:
- dim=1  -> tensore (B, T, 1)
- dim=(5,) -> tensore (B, T, 5)
- dim=(3, 4) -> tensore (B, T, 3, 4)
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from nnodely import Input, Fir, Output, Model


def main():
    print("=" * 60)
    print("Example4: Input con varie dimensioni (dim come int o tuple)")
    print("=" * 60)

    # 1. dim=1 (scalare)
    print("\n1. dim=1 -> input shape (B, T, 1)")

    x1 = Input('x1', dim=1)
    out1 = Output('y1', Fir(out_features=2)(x1.sw(5)))
    m1 = Model('m1', x1, out1).build()
    d1 = np.random.randn(2, 5, 1).astype(np.float32)
    r1 = m1(d1)
    print(f"   Input shape: {d1.shape} -> Output shape: {r1.shape}")

    # 2. dim=(5,) - 1D features
    print("\n2. dim=(5,) -> input shape (B, T, 5)")

    x2 = Input('x2', dim=(5,))
    out2 = Output('y2', Fir(out_features=1)(x2.sw(4)))
    m2 = Model('m2', x2, out2).build()
    d2 = np.random.randn(3, 4, 5).astype(np.float32)
    r2 = m2(d2)
    print(f"   Input shape: {d2.shape} -> Output shape: {r2.shape}")

    # 3. dim=(3, 4) - 2D features
    print("\n3. dim=(3, 4) -> input shape (B, T, 3, 4)")

    x3 = Input('x3', dim=(3, 4))
    out3 = Output('y3', Fir(out_features=1)(x3.sw(6)))
    m3 = Model('m3', x3, out3).build()
    d3 = np.random.randn(2, 6, 3, 4).astype(np.float32)
    r3 = m3(d3)
    print(f"   Input shape: {d3.shape} -> Output shape: {r3.shape}")

    # 4. dim=(2,) con Fir(out_features=3) -> out_dim = (2, 3)
    print("\n4. dim=(2,) + Fir(out_features=3) -> output dim (2, 3)")

    x4 = Input('x4', dim=(2,))
    out4 = Output('y4', Fir(out_features=3)(x4.sw(5)))
    m4 = Model('m4', x4, out4).build()
    d4 = np.random.randn(2, 5, 2).astype(np.float32)
    r4 = m4(d4)
    print(f"   Input shape: {d4.shape} -> Output shape: {r4.shape}")

    # 5. Modello con due input a dimensioni diverse
    print("\n5. Due input con dim diverse: x_a dim=1, x_b dim=(2,)")

    x_a = Input('x_a', dim=1)
    x_b = Input('x_b', dim=(2,3,3))
    out_a = Output('out_a', Fir(out_features=1)(x_a.sw(3)))
    out_b = Output('out_b', Fir(out_features=2)(x_b.sw(3)))
    m5 = Model('m5', [x_a, x_b], [out_a, out_b]).build()
    da = np.random.randn(2, 3, 1).astype(np.float32)
    db = np.random.randn(2, 3, 2, 3, 3).astype(np.float32)
    r5 = m5({'x_a': da, 'x_b': db})
    print(f"   x_a {da.shape} -> out_a {r5['out_a'].shape}")
    print(f"   x_b {db.shape} -> out_b {r5['out_b'].shape}")

    print("\n" + "=" * 60)
    print("OK - example4 completato")
    print("=" * 60)


if __name__ == '__main__':
    main()
