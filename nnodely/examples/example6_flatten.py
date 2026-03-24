#!/usr/bin/env python3
"""
Example6: Layer custom Flatten - appiattisce dim ma non time.

(B, T, D1, D2, D3) -> (B, T, D1*D2*D3)

L'utente ridefinisce output_shape per modificare la forma di uscita.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
import functools
from nnodely import Input, Fir, Output, Model
from nnodely.dag import to_tuple
from nnodely.layer import Layer


# =============================================================================
# Layer custom: Flatten (definito dall'utente)
# =============================================================================

class Flatten(Layer):
    """Appiattisce tutte le dim tranne time: (B, T, D1, D2, ...) -> (B, T, D1*D2*...)."""

    def output_shape(self, seq, time, dim):
        """Propaga seq e time; out_dim = (D1*D2*...,)."""
        dim = to_tuple(dim, (1,))
        D_prod = functools.reduce(lambda a, b: a * b, dim, 1)
        return (seq, time, (D_prod,))

    def build_layer(self, **kwargs):
        T = self.time
        D_prod = functools.reduce(lambda a, b: a * b, self.dim, 1)
        self._layer = keras.layers.Reshape((T, D_prod), name=self.name)
        return self._layer

    def call(self, x):
        return self._layer(x)


# =============================================================================
# Esempio d'uso
# =============================================================================

def main():
    print("=" * 60)
    print("Example6: Layer custom Flatten")
    print("=" * 60)

    # 1. Input (B, T, 2, 3) -> Flatten -> (B, T, 6)
    print("\n1. Input dim=(2, 3) -> Flatten -> dim=(6,)")
    x = Input('x', dim=(2, 3))
    s = Flatten(x.sw(4))
    out = Output('y', Fir(out_features=1)(s))
    m1 = Model('m1', x, out).build()
    d1 = np.random.randn(2, 4, 2, 3).astype(np.float32)
    r1 = m1(d1)
    print(f"   Input shape: {d1.shape} -> Output shape: {r1.shape}")

    # 2. Input (B, T, 2, 3, 4) -> Flatten -> (B, T, 24)
    print("\n2. Input dim=(2, 3, 4) -> Flatten -> dim=(24,)")
    x2 = Input('x2', dim=(2, 3, 4))
    s2 = Flatten(x2.sw(3))
    out2 = Output('y2', Fir(out_features=1)(s2))
    m2 = Model('m2', x2, out2).build()
    d2 = np.random.randn(3, 3, 2, 3, 4).astype(np.float32)
    r2 = m2(d2)
    print(f"   Input shape: {d2.shape} -> Output shape: {r2.shape}")

    # 3. Forma veloce: Flatten(stream)
    print("\n3. Forma veloce Flatten(stream)")
    x3 = Input('x3', dim=(5,))
    s3 = Flatten(x3.sw(2))  # dim (5,) -> (5,) invariato
    out3 = Output('y3', s3)
    m3 = Model('m3', x3, out3).build()
    d3 = np.random.randn(2, 2, 5).astype(np.float32)
    r3 = m3(d3)
    print(f"   Input shape: {d3.shape} -> Output shape: {r3.shape}")

    print("\n" + "=" * 60)
    print("OK - example6_flatten completato")
    print("=" * 60)


if __name__ == '__main__':
    main()
