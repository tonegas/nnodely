#!/usr/bin/env python3
"""
Example5: Layer custom creato dall'utente - conversione gradi/radianti.

L'utente definisce le proprie classi Layer nello stesso file dell'esempio,
estendendo Layer e implementando build_layer e call.
Le dimensioni (seq, time, dim) si propagano automaticamente se output_shape
non viene ridefinito.

Se __init__ ha tutti argomenti opzionali, forma veloce (solo stream, parametri default):
  s = DegToRad(x.sw(5))   invece di  deg = DegToRad(); s = deg(x.sw(5))
  Per parametri non default: Fir(out_features=2)(x.sw(5))
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
from nnodely import Input, Fir, Output, Model
from nnodely.layer import Layer


# =============================================================================
# Layer custom: conversione gradi <-> radianti (definiti dall'utente)
# =============================================================================

class DegToRad(Layer):
    """Converte gradi in radianti: x * (pi/180)."""

    def build_layer(self):
        factor = np.pi / 180.0
        self._layer = keras.layers.Lambda(
            lambda x, f=factor: x * f,
            name=self.name
        )
        return self._layer

    def call(self, x):
        return self._layer(x)


class RadToDeg(Layer):
    """Converte radianti in gradi: x * (180/pi)."""

    def build_layer(self):
        factor = 180.0 / np.pi
        self._layer = keras.layers.Lambda(
            lambda x, f=factor: x * f,
            name=self.name
        )
        return self._layer

    def call(self, x):
        return self._layer(x)


# =============================================================================
# Esempio d'uso
# =============================================================================

def main():
    print("=" * 60)
    print("Example5: Layer custom - DegToRad e RadToDeg")
    print("=" * 60)

    # 1. Gradi -> radianti -> Fir
    print("\n1. Input (gradi) -> DegToRad -> Fir -> Output")
    x = Input('x', dim=1)
    deg_rad = DegToRad()
    s = deg_rad(x.sw(5))
    out = Output('y', Fir(out_features=1)(s))
    m1 = Model('m1', x, out).build()

    # Input in gradi (es. 90, 180, 270)
    deg_data = np.array([[[90.0], [180.0], [270.0], [0.0], [45.0]]], dtype=np.float32)
    deg_data = np.broadcast_to(deg_data, (2, 5, 1)).copy()
    r1 = m1(deg_data)
    print(f"   Input (gradi): {deg_data[0, :, 0]} -> Output shape: {r1.shape}")

    # 2. Round-trip: gradi -> radianti -> gradi
    print("\n2. Round-trip: gradi -> DegToRad -> RadToDeg (dovrebbe tornare ~gradi)")
    x2 = Input('x2', dim=1)
    s2 = RadToDeg()(DegToRad()(x2.sw(4)))
    out2 = Output('y2', Fir(out_features=1)(s2))
    m2 = Model('m2', x2, out2).build()
    deg_in = np.array([[[30.0], [60.0], [90.0], [120.0]]], dtype=np.float32)
    deg_in = np.broadcast_to(deg_in, (2, 4, 1)).copy()
    r2 = m2(deg_in)
    print(f"   Input shape: {deg_in.shape} -> Output shape: {r2.shape}")

    # 3. Verifica numerica: 90° = pi/2 rad (forma diretta DegToRad(stream))
    print("\n3. Verifica: 90° -> DegToRad -> ~1.5708 rad (forma diretta)")
    x3 = Input('x3', dim=1)
    s3 = DegToRad(x3.sw(1))  # forma diretta: crea e chiama in un colpo
    out3 = Output('y3', s3)
    m3 = Model('m3', x3, out3).build()
    ninety = np.array([[[90.0]]], dtype=np.float32)
    r3 = m3(ninety)
    expected = 90.0 * np.pi / 180.0
    print(f"   90° -> {r3[0, 0, 0]:.4f} rad (atteso pi/2 = {expected:.4f})")

    print("\n" + "=" * 60)
    print("OK - example5_custom_layer completato")
    print("=" * 60)


if __name__ == '__main__':
    main()
