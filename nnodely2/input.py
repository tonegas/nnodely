"""
Input - nodo radice del DAG. Come Layer ha seq, time, dim.
SampleWindow - Layer che applica finestra temporale (slice se necessario).
"""

from nnodely.dag import to_tuple
from nnodely.layer import Layer

import keras


class SampleWindow(Layer):
    """
    Layer che estrae finestra temporale. Se window_size < input.time, applica slice.
    Simmetrico agli altri layer: usa build_layer e call.
    """

    node_type = 'SampleWindow'
    output_stream_prefix = 'SampleWindow'

    def __init__(self, window_size: int, **kwargs):
        super().__init__(**kwargs)
        self._window_size = window_size

    def output_shape(self, seq, time, dim):
        """Output ha time=window_size."""
        return seq[0], self._window_size, dim[0]

    def build_layer(self, **kwargs):
        """Crea Lambda per slice se window_size < input.time, altrimenti Identity."""
        seq_t = tuple(self.seq or ())
        dim_t = to_tuple(self.dim, (1,))
        n = self._window_size
        if n >= self.time:
            self._layer = keras.layers.Identity(name=self.name)
        else:
            slices = [slice(None)] * (1 + len(seq_t)) + [slice(-n, None)] + [slice(None)] * len(dim_t)
            self._layer = keras.layers.Lambda(
                lambda x: x[tuple(slices)],
                name=self.name,
            )
        return self._layer

    def call(self, x):
        return self._layer(x)


class Input:
    """
    Input di rete. Come Layer ha seq, time, dim.
    .sw(n) crea SampleWindow(n) - Layer con build_layer simmetrico agli altri.
    """

    def __init__(self, name: str, *, dim: int | tuple = 1, seq: int | tuple | None = ()):
        self.name = name
        self.seq = to_tuple(seq, ()) if seq is not None else (None,)
        self.time = 1
        self.dim = to_tuple(dim)
        self.predecessors = []
        self.node_type = 'input'

    def sw(self, window_size: int):
        """Crea SampleWindow (Layer) con finestra temporale. Aggiorna self.time (max finestra)."""
        self.time = max(self.time, window_size)
        return SampleWindow(window_size)(self)

    def call(self, x):
        """Pass-through per esecuzione Keras."""
        return x
