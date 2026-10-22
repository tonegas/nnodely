"""Input: a signal read from data, a root of the graph. Its windows are
``SampleWindow`` nodes."""

from nnodely.layers.partitions import SampleWindow
from nnodely.core.shape import Shape
from nnodely.core.stream import Stream

import keras


def _outer_time(name: str, seq) -> tuple:
    """The outer time axes ``seq`` declares, outermost first, a dynamic one
    as None; ``()`` if not given."""
    if seq is None:
        return ()
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

    ``dim`` sets the feature axes (one feature by default). ``seq`` sets the
    sequence: the time axes outside the window, outermost first, used by
    rollouts; ``-1`` leaves a length dynamic.

    Read directly, an Input is its current sample, time ``(*seq, 1)``. Its
    windows, requested with :meth:`sw` and :meth:`next`, have the window as
    their inner time and keep the sequence. Requesting a window leaves the
    Input as it is: a model reads from it the union of the windows its own
    graph takes, the current sample included when the graph reads the Input
    directly, found by ``build()`` - a window declared but never used widens
    nothing.
    """

    def __init__(
        self,
        name: str,
        *,
        dim: int | tuple | None = None,
        seq: int | tuple[int, ...] | None = None,
    ):
        sequence = _outer_time(name, seq)
        super().__init__(name=name, dim=dim, time=(*sequence, 1), preds=None)
        self._sequence = sequence
        # The window the graph of the last model built reads, found by build():
        # the data holds it, the Input itself stays its current sample.
        self.past, self.future = (1, 0)
        # The window a saved model read, kept by the model loaded from it.
        self._saved_window = (0, 0)
        self._update_input()

    @property
    def sequence(self) -> tuple:
        """The outer time axes of the Input, outermost first, a dynamic one
        as None: ``()`` without a sequence."""
        return self._sequence

    @property
    def data_shape(self) -> Shape:
        """The shape of one sample of the data the Input is read from: its
        sequence, then the union of the windows the last model built reads.
        Before a build, the current sample: the Input's own shape."""
        return Shape(dim=self.dim, time=(*self.sequence, self.past + self.future))

    def _update_input(self):
        """The Keras input the data feeds."""
        self.input = keras.Input(shape=self.data_shape, name=self.name)

    def _set_window(self, past: int, future: int) -> None:
        """Read through ``past`` samples up to the current one and ``future``
        after it: the union of the windows of the graph being built."""
        self.past, self.future = int(past), int(future)
        self._update_input()

    def sw(self, window_size: int | list[int] | tuple[int, int]):
        """Return a window of samples of this input.

        ``sw(n)`` is the last ``n`` samples, up to and including the current
        one. ``sw([p, f])`` is ``p`` samples up to and including the current
        one followed by the next ``f`` samples. The window replaces the inner
        time and keeps the sequence. The Input itself is left as it is:
        ``build()`` finds the windows a model reads.
        """
        if isinstance(window_size, (list, tuple)):
            if len(window_size) != 2:
                raise ValueError(
                    f"{self.name}: window_size list must have length 2, got {len(window_size)}."
                )
            local_past, local_future = window_size[0], window_size[1]
        else:
            local_past, local_future = window_size, 0

        return SampleWindow(past=local_past, future=local_future)([self])

    def next(self):
        """The next sample, the same as ``sw([0, 1])``."""
        return self.sw([0, 1])

    def get_config(self):
        return {
            "name": self.name,
            "dim": tuple(self.dim),
            "seq": tuple(-1 if length is None else length for length in self.sequence),
            "past": self.past,
            "future": self.future,
        }

    @classmethod
    def from_config(cls, config: dict, preds=None):
        node = cls(name=config["name"], dim=config["dim"], seq=config["seq"])
        node._saved_window = (config["past"], config["future"])
        node._set_window(*node._saved_window)
        return node


def windows_read(order) -> dict[Input, tuple[int, int]]:
    """The window a graph reads from each of its Inputs: the union of its
    windows, the current sample for a direct read, ``{input: (past, future)}``.

    ``order`` is the graph's nodes. A window declared on an Input but not in
    the graph widens nothing. An Input loaded from a file keeps at least the
    window the saved model read, so it reads the same data.
    """
    windows: dict[Input, tuple[int, int]] = {}
    for node in order:
        # A model call reads its Inputs through the windows of its own graph.
        if not isinstance(node, Stream):
            continue
        for pred in node.preds:
            if not isinstance(pred, Input):
                continue
            if isinstance(node, SampleWindow):
                read = (node.past, node.future)
            elif passes_on(node, pred):
                read = (pred.past, pred.future)
            else:
                read = (1, 0)
            past, future = windows.get(pred, pred._saved_window)
            windows[pred] = (max(past, read[0]), max(future, read[1]))
    return windows


def passes_on(node, pred) -> bool:
    """Whether ``node`` - a Loop, a Roll, an OdeNet - hands ``pred``, an Input
    of the model it wraps, on to that model as its data holds it, rather than
    reading it. Inputs are fed by name, and a flattened graph reads copies."""
    inputs = getattr(getattr(node, "f", None), "inputs", ())
    return any(pred.name == body_input.name for body_input in inputs)


def resolve_windows(order) -> dict[Input, tuple[int, int]]:
    """Give every Input of a graph the window the graph reads from it
    (:func:`windows_read`), and return them: ``{input: (past, future)}``.
    """
    windows = windows_read(order)
    for source, (past, future) in windows.items():
        source._set_window(past, future)
    return windows
