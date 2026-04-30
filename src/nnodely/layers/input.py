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

    def __init__(
        self,
        name: str,
        *,
        dim: int | tuple | None = None,
        seq: int | tuple[int] | None = None,
    ):
        super().__init__(name=name, seq=seq, time=None, dim=dim, predecessors=None)
        self.input = keras.Input(shape=self.shape, name=self.name)

    def sw(self, window_size: int):
        """Crea SampleWindow (Layer) con finestra temporale. Aggiorna self.time (max finestra)."""
        self.time = max(self.time, window_size)
        self.input = keras.Input(shape=self.shape, name=self.name)
        return SampleWindow(window_size)(self)
