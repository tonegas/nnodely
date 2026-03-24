"""
Parameter - parametro trainable utilizzabile in operazioni aritmetiche e per inizializzare Fir.

Shape composta da sw (dimensione temporale) e dim (dimensioni features), come per Input/Stream.
"""

import numpy as np
from nnodely.dag import next_name, to_tuple


def _is_parameter(x):
    """True se x è un Parameter."""
    return isinstance(x, Parameter)


class Parameter:
    """
    Parametro trainabile. Può essere usato in:
    - Operazioni aritmetiche: stream + param, param * stream, ecc.
    - Fir: kernel_initializer per inizializzare i pesi del Dense

    sw: int - dimensione temporale (come sample window)
    dim: int o tuple - dimensioni features, es. 1, (5,), (3,4)
    shape = (sw,) + dim

    initial_value: valore iniziale (float, np.ndarray). Se None, usa zeros.
    """

    def __init__(self, name: str = None, sw: int = 1, dim: int | tuple = 1, initial_value: float | np.ndarray = None):
        self.name = name or next_name('Param')
        self.sw = int(sw)
        self.dim = to_tuple(dim, (1,))
        self.shape = (self.sw,) + self.dim
        if initial_value is None:
            self.initial_value = np.zeros(self.shape, dtype=np.float32)
        elif isinstance(initial_value, (int, float)):
            self.initial_value = np.full(self.shape, float(initial_value), dtype=np.float32)
        else:
            self.initial_value = np.asarray(initial_value, dtype=np.float32)
            if self.initial_value.shape != self.shape:
                self.shape = self.initial_value.shape
                self.sw = self.shape[0]
                self.dim = self.shape[1:]
        self.predecessors = []  # per compatibilità con DAG (param non ha predecessori)
