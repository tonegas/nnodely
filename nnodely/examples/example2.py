#!/usr/bin/env python3
"""
Example2: Le 4 modalità di funzionamento del Model (con/senza build, stream/tensori).
- Modello 1: standalone
- Modello 2: usa m1 buildato
- Modello 3: usa m1 non buildato (dopo unbuild())
Requisiti: pip install -r requirements.txt
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from nnodely import Input, Fir, Output, Model
from tensorflow.keras.utils import plot_model


def main():
    # --- Modello 1: x -> Fir(2) -> y ---
    x = Input('x', dim=1)
    out = Output('y', Fir(out_features=2)(x.sw(5)))
    m1 = Model('m1', {'x': x}, out)

    print("=" * 55)
    print("MODELLO 1: x -> Fir(2) -> y")
    print("=" * 55)
    print("1. Modello NON buildato + STREAM")
    s = m1({'x': Input('x', dim=1).sw(5)})
    print(f"   Output stream: {s.name}, built={m1.built}")

    print("\n2. Modello NON buildato + TENSORI")
    t = m1({'x': np.random.randn(3, 5, 1).astype(np.float32)})
    print(f"   Output shape: {t.shape}, built={m1.built}")

    print("\n3. Modello BUILDATO + TENSORI")
    m1.build()
    t2 = m1({'x': np.random.randn(2, 5, 1).astype(np.float32)})
    print(f"   Output shape: {t2.shape}, built={m1.built}")

    print("\n4. Modello BUILDATO + STREAM (compatibile)")
    s2 = m1({'x': Input('x', dim=1).sw(5)})
    print(f"   Output stream: {s2.name}")

    print("\n5. Modello BUILDATO + STREAM (non compatibile)")
    try:
        m1({'x': Input('x', dim=1).sw(12)})
    except ValueError as e:
        print(f"   Error: {e}")
    #m1.unbuild()
    # --- Modello 2: x2 -> m1(x2.sw(5)) -> z (usa m1 buildato) ---

    x = Input('x', dim=1)
    m = Model('m', {'x': x}, [
        Output('y5', Fir(out_features=1)(x.sw(5))),
        Output('y10', Fir(out_features=2)(x.sw(10))),
        Output('y3', Fir(out_features=2)(x.sw(3))),
        Output('y12', Fir(out_features=2)(x.sw(12))),
    ])
    m.build()
    plot_model(m._model, to_file='m.png', show_shapes=True, expand_nested=True, show_layer_names=True)

    k = Input('k', dim=2)  # dim=2 = output di m1 (Fir dim=1, out_features=2 → dim=(2,))
    kk = Output('kk', Fir(out_features=1)(k.sw(1)))
    m5 = Model('m5', {'k': k}, kk).build()
    plot_model(m5._model, to_file='m5.png', show_shapes=True, expand_nested=True, show_layer_names=True)

    #m1.unbuild()
    #m5.unbuild()
    x2 = Input('x2', dim=1)
    out2 = Output('z', m5(m1({'x': x2.sw(5)})))
    m2 = Model('m2', {'x2': x2}, out2).build()
    plot_model(m2._model, to_file='m2.png', show_shapes=True, expand_nested=True, show_layer_names=True)

    print("\n" + "=" * 55)
    print("MODELLO 2: x2 -> m1(x2.sw(5)) -> z  (usa m1 internamente)")
    print("=" * 55)
    print("1. Modello NON buildato + STREAM")
    s3 = m2({'x2': Input('x2', dim=1).sw(5)})
    print(f"   Output stream: {s3.name}, built={m2.built}")

    print("\n2. Modello NON buildato + TENSORI")
    t3 = m2({'x2': np.random.randn(3, 5, 1).astype(np.float32)})
    print(f"   Output shape: {t3.shape}, built={m2.built}")

    print("\n3. Modello BUILDATO + TENSORI")
    m2.build()
    t4 = m2({'x2': np.random.randn(2, 5, 1).astype(np.float32)})
    print(f"   Output shape: {t4.shape}, built={m2.built}")

    print("\n4. Modello BUILDATO + STREAM (compatibile)")
    s4 = m2({'x2': Input('x2', dim=1).sw(5)})
    print(f"   Output stream: {s4.name}")

    # --- Modello 3: x3 -> m1(x3.sw(5)) -> w (usa m1 NON buildato) ---
    m1.unbuild()
    x3 = Input('x3', dim=1)
    out3 = Output('w', m1({'x': x3.sw(5)}))
    m3 = Model('m3', {'x3': x3}, out3)

    print("\n" + "=" * 55)
    print("MODELLO 3: x3 -> m1(x3.sw(5)) -> w  (usa m1 NON buildato)")
    print("=" * 55)
    print("1. Modello NON buildato + STREAM")
    s5 = m3({'x3': Input('x3', dim=1).sw(5)})
    print(f"   Output stream: {s5.name}, built={m3.built}")

    print("\n2. Modello NON buildato + TENSORI")
    t5 = m3({'x3': np.random.randn(3, 5, 1).astype(np.float32)})
    print(f"   Output shape: {t5.shape}, built={m3.built}")

    print("\n3. Modello BUILDATO + TENSORI")
    m3.build()
    t6 = m3({'x3': np.random.randn(2, 5, 1).astype(np.float32)})
    print(f"   Output shape: {t6.shape}, built={m3.built}")

    print("\n4. Modello BUILDATO + STREAM (compatibile)")
    s6 = m3({'x3': Input('x3', dim=1).sw(5)})
    print(f"   Output stream: {s6.name}")

    print("\n" + "=" * 55)
    print("OK - tutte e 4 le modalità verificate per tutti e 3 i modelli")


if __name__ == '__main__':
    main()
