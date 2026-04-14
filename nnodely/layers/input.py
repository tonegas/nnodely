"""
Input - nodo radice del DAG. Come Layer ha seq, time, dim.
SampleWindow - Layer che applica finestra temporale (slice se necessario).
"""
from nnodely.layers.time_ops import SampleWindow
from nnodely.core.stream import Stream

import keras

class Input(Stream):
    """
    Input di rete. Come Layer ha seq, time, dim.
    """
    def __init__(self, name: str, *, dim: int | tuple = 1, seq: int | tuple | None = ()):
        super().__init__(name=name, node_type='Input', seq=seq, time=1, dim=dim, predecessors=[])
        self.input = keras.Input(shape=self.shape, name=self.name)

    def sw(self, window_size: int):
        """Crea SampleWindow (Layer) con finestra temporale. Aggiorna self.time (max finestra)."""
        self.time = max(self.time, window_size)
        self.input = keras.Input(shape=self.shape, name=self.name)
        return SampleWindow(window_size)(self)
