#!/usr/bin/env python3
"""
Example3: Input/output flessibili - singolo o lista.
- Input: singolo Input, lista di Input, o dict
- Output: singolo -> valore diretto; multipli -> dict (out['out1'], out['out2'])
- Chiamata: model(data) con singolo input, model([d1,d2]) o model({n:v})
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from nnodely import Input, Fir, Output, Model


def main():
    print("=" * 60)
    print("1. SINGOLO INPUT (non dict) + SINGOLO OUTPUT")
    print("=" * 60)

    x = Input('x', dim=1)
    out = Output('y', Fir(out_features=2)(x.sw(5)))
    m = Model('m1', x, out)  # input singolo, non dict

    # Chiamata con singolo valore (non dict)
    result = m(np.random.randn(3, 5, 1).astype(np.float32))
    print(f"   m(data) -> valore diretto, shape: {result.shape}")

    # Anche con dict funziona
    result2 = m({'x': np.random.randn(2, 5, 1).astype(np.float32)})
    print(f"   m({{'x': data}}) -> shape: {result2.shape}")

    print("\n" + "=" * 60)
    print("2. SINGOLO OUTPUT - valore diretto, no out['y']")
    print("=" * 60)
    print(f"   result è il tensore: shape {result.shape}")

    print("\n" + "=" * 60)
    print("3. MULTIPLI OUTPUT - accesso out['pred'], out['aux']")
    print("=" * 60)

    x = Input('x', dim=1)
    s = x.sw(5)
    pred = Output('pred', Fir(out_features=2)(s))
    aux = Output('aux', Fir(out_features=1)(s))
    m_multi = Model('m_multi', x, [pred, aux])

    result_multi = m_multi(np.random.randn(2, 5, 1).astype(np.float32))
    print(f"   Keys: {list(result_multi.keys())}")
    print(f"   out['pred'] -> shape {result_multi['pred'].shape}")
    print(f"   out['aux'] -> shape {result_multi['aux'].shape}")

    print("\n" + "=" * 60)
    print("4. LISTA DI INPUT (entrambi connessi agli output)")
    print("=" * 60)
    x1 = Input('x1', dim=1)
    x2 = Input('x2', dim=1)
    # Due input -> due output (ogni input connesso al proprio output)
    out1 = Output('z1', Fir(out_features=1)(x1.sw(3)))
    out2 = Output('z2', Fir(out_features=1)(x2.sw(3)))
    m_two = Model('m_two', [x1, x2], [out1, out2])

    # Chiamata con lista [d1, d2] nell'ordine degli input
    result4 = m_two([
        np.random.randn(2, 3, 1).astype(np.float32),
        np.random.randn(2, 3, 1).astype(np.float32),
    ])
    print(f"   m([d1, d2]) -> out['z1']: {result4['z1'].shape}, out['z2']: {result4['z2'].shape}")

    # Oppure con dict
    result4b = m_two({'x1': np.random.randn(2, 3, 1).astype(np.float32),
                      'x2': np.random.randn(2, 3, 1).astype(np.float32)})
    print(f"   m({{'x1': d1, 'x2': d2}}) -> out['z1']: {result4b['z1'].shape}")

    print("\n" + "=" * 60)
    print("OK - esempio 3 completato")
    print("=" * 60)


if __name__ == '__main__':
    main()
