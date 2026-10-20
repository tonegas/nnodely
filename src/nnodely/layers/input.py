"""Input: a signal read from data, a root of the graph. Its windows are
``SampleWindow`` nodes."""

from nnodely.layers.partitions import SampleWindow
from nnodely.core.shape import Shape
from nnodely.core.stream import Stream

import keras


def _outer_time(name: str, seq) -> tuple | None:
    """The outer time axes a window's ``seq`` declares, outermost first, a
    dynamic one as None; None if not given."""
    if seq is None:
        return None
    seq = (seq,) if isinstance(seq, int) else tuple(seq)
    if None in seq:
        raise ValueError(
            f"{name}: a dynamic sequence length is declared with -1, not None."
        )
    # Keras marks a dynamic axis with None.
    return tuple(None if length == -1 else length for length in seq)


class Input(Stream):
    """
    A signal the model reads from data.

    ``dim`` sets the feature axes (one feature by default). Read whole, an
    Input has no time axis. Its windows, requested with :meth:`sw`,
    :meth:`last` and :meth:`next`, have the window as their inner time, and
    the Input's own is the union of them. A window's ``seq`` sets the
    sequence: the outer time axes, outermost first, used by rollouts; ``-1``
    leaves a length dynamic. An Input has one sequence, set by its first
    window, and every window has it: a later one may leave it out, but not
    change it.
    """

    def __init__(self, name: str, *, dim: int | tuple | None = None):
        super().__init__(name=name, dim=dim, preds=None)
        self.past, self.future = (0, 0)
        # The time axes outside the window, fixed by the first window.
        self._outer_time: tuple | None = None
        self._update_shape()

    def _update_shape(self):
        """The time axes of the union of the windows: the outer ones, then
        the window; none before the first window."""
        time = (
            ()
            if self._outer_time is None
            else (*self._outer_time, self.past + self.future)
        )
        self.shape = Shape(dim=self.shape.dim, time=time)
        self.input = keras.Input(shape=self.shape, name=self.name)

    def _check_sequence(self, seq):
        """Fix the Input's outer time with its first window; ``seq`` may not
        change it afterwards."""
        outer_time = _outer_time(self.name, seq)
        if self._outer_time is None:
            self._outer_time = () if outer_time is None else outer_time
        elif outer_time is not None and outer_time != self._outer_time:
            raise ValueError(
                f"{self.name}: every window of an Input has the Input's sequence "
                f"{self._outer_time}, so a window cannot have seq={outer_time}. "
                "Windows may differ only in their inner time; for another "
                "sequence, declare another Input."
            )

    def sw(
        self,
        window_size: int | list[int] | tuple[int, int],
        seq: int | tuple[int, ...] | None = None,
    ):
        """Return a window of samples of this input.

        ``sw(n)`` is the last ``n`` samples, up to and including the current
        one. ``sw([p, f])`` is ``p`` samples up to and including the current
        one followed by the next ``f`` samples. The input's own window grows to
        cover every window requested from it. ``seq`` is the sequence of the
        Input, its outer time axes, outermost first (see :class:`Input`).
        """
        if isinstance(window_size, (list, tuple)):
            if len(window_size) != 2:
                raise ValueError(
                    f"{self.name}: window_size list must have length 2, got {len(window_size)}."
                )
            local_past, local_future = window_size[0], window_size[1]
        else:
            local_past, local_future = window_size, 0

        self._check_sequence(seq)
        self.past, self.future = (
            max(self.past, local_past),
            max(self.future, local_future),
        )
        self._update_shape()
        return SampleWindow(past=local_past, future=local_future)([self])

    def last(self, seq: int | tuple[int, ...] | None = None):
        """The current sample, the same as ``sw(1)``."""
        return self.sw(1, seq=seq)

    def next(self, seq: int | tuple[int, ...] | None = None):
        """The next sample, the same as ``sw([0, 1])``."""
        return self.sw([0, 1], seq=seq)

    def get_config(self):
        return {
            "name": self.name,
            "dim": tuple(self.dim),
            "seq": None
            if self._outer_time is None
            else tuple(-1 if length is None else length for length in self._outer_time),
            "past": self.past,
            "future": self.future,
        }

    @classmethod
    def from_config(cls, config: dict, preds=None):
        node = cls(name=config["name"], dim=config["dim"])
        node._outer_time = _outer_time(node.name, config["seq"])
        node.past = config["past"]
        node.future = config["future"]
        node._update_shape()
        return node
