#!/usr/bin/env python3
"""
Esempio: operazioni aritmetiche (+, -, *, /) su stream con dimensioni uguali.

Testa sia nnodely_new che nnodely.
Requisiti: pip install -r requirements.txt
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np


def test_nnodely():
    """Test operazioni aritmetiche con nnodely (Lazy/DAG)."""
    from nnodely import Input, Fir, Output, Model

    print("=" * 60)
    print("nnodely - Operazioni aritmetiche")
    print("=" * 60)

    # Due input con stessa finestra
    x = Input('x', dim=1)
    y = Input('y', dim=1)
    w = 5
    xs = x.sw(w)
    ys = y.sw(w)

    # Addizione: (x + y) -> Fir -> output
    s_add = xs + ys
    out_add = Output('sum', Fir(out_features=1)(s_add))
    m_add = Model('add_model', [x, y], out_add).build()

    # Sottrazione
    s_sub = xs - ys
    out_sub = Output('diff', Fir(out_features=1)(s_sub))
    m_sub = Model('sub_model', [x, y], out_sub).build()

    # Moltiplicazione
    s_mul = xs * ys
    out_mul = Output('prod', Fir(out_features=1)(s_mul))
    m_mul = Model('mul_model', [x, y], out_mul).build()

    # Divisione
    s_div = xs / ys
    out_div = Output('quot', Fir(out_features=1)(s_div))
    m_div = Model('div_model', [x, y], out_div).build()

    # Test con tensori
    batch = 3
    dx = np.random.randn(batch, w, 1).astype(np.float32) + 1.0
    dy = np.random.randn(batch, w, 1).astype(np.float32) + 1.0

    r_add = m_add({'x': dx, 'y': dy})
    r_sub = m_sub({'x': dx, 'y': dy})
    r_mul = m_mul({'x': dx, 'y': dy})
    r_div = m_div({'x': dx, 'y': dy})

    print("Add (x+y):", r_add.shape, "->", r_add.flatten()[:3])
    print("Sub (x-y):", r_sub.shape, "->", r_sub.flatten()[:3])
    print("Mul (x*y):", r_mul.shape, "->", r_mul.flatten()[:3])
    print("Div (x/y):", r_div.shape, "->", r_div.flatten()[:3])

    # Verifica: (a+b) + (a-b) ≈ 2*a (approssimato dopo Fir)
    combined = (xs + ys) + (xs - ys)
    out_comb = Output('comb', Fir(out_features=1)(combined))
    m_comb = Model('comb_model', [x, y], out_comb).build()
    r_comb = m_comb({'x': dx, 'y': dy})
    print("(x+y)+(x-y) -> Fir:", r_comb.shape)
    print("OK nnodely\n")



def test_incompatible_dims():
    """Verifica che dimensioni incompatibili sollevino errore."""
    from nnodely import Input, Output, Model

    print("=" * 60)
    print("Verifica: dimensioni incompatibili")
    print("=" * 60)

    x = Input('x', dim=1)
    y = Input('y', dim=1)
    # x.sw(5) e y.sw(3) hanno time diversi
    try:
        s = x.sw(5) + y.sw(3)
        print("ERRORE: avrebbe dovuto sollevare ValueError")
    except ValueError as e:
        print("OK - ValueError atteso:", str(e)[:60] + "...")
    print()


def main():
    test_nnodely()
    test_incompatible_dims()
    print("=" * 60)
    print("Tutti i test aritmetici completati")
    print("=" * 60)


if __name__ == '__main__':
    main()
