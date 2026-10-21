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
    :meth:`last` and :meth:`next`, have the window as their inner time.
    Requesting a window leaves the Input as it is: a model reads from it the
    union of the windows its own graph takes, found by ``build()``, which
    then gives the Input that window - a window declared but never used
    widens nothing.

    A window's ``seq`` sets the sequence: the outer time axes, outermost
    first, used by rollouts; ``-1`` leaves a length dynamic. An Input has one
    sequence, set once, by its first window: every later window has it and
    leaves ``seq`` out.
    """

    def __init__(self, name: str, *, dim: int | tuple | None = None):
        super().__init__(name=name, dim=dim, preds=None)
        # The window the graph of the last model built reads, found by build().
        self.past, self.future = (0, 0)
        # The time axes outside the window, set by the first window.
        self._outer_time: tuple | None = None
        # The window a saved model read, kept by the model loaded from it.
        self._saved_window = (0, 0)
        self._update_shape()

    @property
    def sequence(self) -> tuple:
        """The outer time axes of the Input's windows, outermost first, a
        dynamic one as None: ``()`` until a window sets them, or when it has
        no sequence."""
        return self._outer_time or ()

    def _update_shape(self):
        """The time axes of the window the Input is read through: its outer
        ones, then the window; none before build() has found a window."""
        window = self.past + self.future
        time = () if window == 0 else (*self.sequence, window)
        self.shape = Shape(dim=self.shape.dim, time=time)
        self.input = keras.Input(shape=self.shape, name=self.name)

    def _set_window(self, past: int, future: int) -> None:
        """Read through ``past`` samples up to the current one and ``future``
        after it: the union of the windows of the graph being built."""
        self.past, self.future = int(past), int(future)
        self._update_shape()

    def _declare_sequence(self, seq):
        """Set the Input's outer time with its first window; a later window
        may not set it again."""
        outer_time = _outer_time(self.name, seq)
        if self._outer_time is None:
            self._outer_time = () if outer_time is None else outer_time
        elif outer_time is not None:
            raise ValueError(
                f"{self.name}: the sequence of an Input is set only once, by its "
                f"first window, and it is {self._outer_time or 'none'}: a later "
                "window has it already, so leave seq out. For another sequence, "
                "declare another Input."
            )

    def sw(
        self,
        window_size: int | list[int] | tuple[int, int],
        seq: int | tuple[int, ...] | None = None,
    ):
        """Return a window of samples of this input.

        ``sw(n)`` is the last ``n`` samples, up to and including the current
        one. ``sw([p, f])`` is ``p`` samples up to and including the current
        one followed by the next ``f`` samples. The Input itself is left as it
        is: ``build()`` finds the windows a model reads. ``seq`` is the
        sequence of the Input, its outer time axes, outermost first, set only
        by its first window (see :class:`Input`).
        """
        if isinstance(window_size, (list, tuple)):
            if len(window_size) != 2:
                raise ValueError(
                    f"{self.name}: window_size list must have length 2, got {len(window_size)}."
                )
            local_past, local_future = window_size[0], window_size[1]
        else:
            local_past, local_future = window_size, 0

        self._declare_sequence(seq)
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
        node._saved_window = (config["past"], config["future"])
        node._set_window(*node._saved_window)
        return node


def resolve_windows(order) -> dict[Input, tuple[int, int]]:
    """Give every Input of a graph the union of the windows the graph reads
    from it, and return them: ``{input: (past, future)}``.

    ``order`` is the graph's nodes. A window declared on an Input but not in
    the graph widens nothing; an Input the graph reads through no window is
    left as it is. An Input loaded from a file keeps at least the window the
    saved model read, so it reads the same data.
    """
    windows: dict[Input, tuple[int, int]] = {}
    for node in order:
        if isinstance(node, SampleWindow) and isinstance(node.preds[0], Input):
            source = node.preds[0]
            past, future = windows.get(source, source._saved_window)
            windows[source] = (max(past, node.past), max(future, node.future))
    # Read whole, an Input has no time axis; read through a window, the
    # data holds one: one Input cannot be both.
    for node in order:
        if isinstance(node, SampleWindow):
            continue
        for pred in node.preds:
            if isinstance(pred, Input) and pred in windows:
                raise ValueError(
                    f"{node.name} reads the Input {pred.name!r} whole, and the "
                    "graph also reads it through windows, so its data has a "
                    "time axis the whole read does not have. Read it through a "
                    f"window there too, such as {pred.name}.last()."
                )
    for source, (past, future) in windows.items():
        source._set_window(past, future)
    return windows
