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
        seq: int | tuple[int | None, ...] | None = None,
    ):
        super().__init__(name=name, seq=seq, time=None, dim=dim, preds=None)
        self.input = keras.Input(shape=self.shape, name=self.name)
        self.past, self.future = (
            1,
            0,
        )  # default window sizes, can be updated by sw() or by build() with sampling

    def sw(self, window_size: int | list[int]):
        """Crea SampleWindow (Layer) con finestra temporale. Aggiorna self.time (max finestra)."""
        if isinstance(window_size, list):
            if len(window_size) != 2:
                raise ValueError(
                    f"{self.name}: window_size list must have length 2, got {len(window_size)}."
                )
            local_past, local_future = window_size[0], window_size[1]
        else:
            local_past, local_future = window_size, 0

        self.past, self.future = (
            max(self.past, local_past),
            max(self.future, local_future),
        )
        self.time = self.past + self.future
        self.input = keras.Input(shape=self.shape, name=self.name)
        return SampleWindow(past=local_past, future=local_future)([self])

    # def tw(self, window_size: float | list[float]):
    #     """Crea SampleWindow (Layer) con finestra temporale basata su tempo reale. Aggiorna self.time (max finestra)."""
    #     if self.sampling is None:
    #         raise ValueError(f"{self.name}: cannot use tw() without sampling defined.")
    #     if isinstance(window_size, list):
    #         if len(window_size) != 2:
    #             raise ValueError(f"{self.name}: window_size list must have length 2, got {len(window_size)}.")
    #         self.past, self.future = int(window_size[0] / self.sampling), int(window_size[1] / self.sampling)
    #         window_size = self.past + self.future
    #         self.time = max(self.time, window_size)
    #     else:
    #         window_size = int(window_size / self.sampling)
    #         self.time = max(self.time, window_size)
    #         self.past, self.future = window_size, 0

    #     self.input = keras.Input(shape=self.shape, name=self.name)
    #     return SampleWindow(window_size)([self])

    def last(self):
        """Shortcut per sw(1)."""
        return self.sw(1)

    def next(self):
        """Shortcut per sw([0, 1])."""
        return self.sw([0, 1])
