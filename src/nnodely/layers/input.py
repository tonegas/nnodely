"""
Input - nodo radice del DAG. Come Layer ha seq, time, dim.
SampleWindow - Layer che applica finestra temporale (slice se necessario).
"""

from nnodely.layers.time_ops import SampleWindow
from nnodely.core.stream import Stream

import keras


class Input(Stream):
    """
    A signal the model reads from data.

    ``dim`` sets the feature axes (one feature by default) and ``seq`` the
    optional sequence axes, used by rollouts; ``seq=(None,)`` leaves the
    sequence length dynamic. The time window is not declared here: it is the
    union of the windows requested with :meth:`sw`, :meth:`last` and
    :meth:`next`.
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
            0,
            0,
        )

    def sw(self, window_size: int | list[int]):
        """Return a window of samples of this input.

        ``sw(n)`` is the last ``n`` samples, up to and including the current
        one. ``sw([p, f])`` is ``p`` samples up to and including the current
        one followed by the next ``f`` samples. The input's own window grows to
        cover every window requested from it.
        """
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
        self.shape.time = self.past + self.future
        self.input = keras.Input(shape=self.shape, name=self.name)
        return SampleWindow(past=local_past, future=local_future)([self])

    def last(self):
        """The current sample, the same as ``sw(1)``."""
        return self.sw(1)

    def next(self):
        """The next sample, the same as ``sw([0, 1])``."""
        return self.sw([0, 1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "past": self.past,
                "future": self.future,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        node = cls(
            name=config["name"],
            dim=config["dim"],
            seq=config["seq"],
        )

        node.past = config["past"]
        node.future = config["future"]
        node.shape.time = node.past + node.future or 1
        node.input = keras.Input(shape=node.shape, name=node.name)
        return node
