"""Input: a signal read from data, a root of the graph. Its windows are
``SampleWindow`` nodes."""

from nnodely.layers.partitions import SampleWindow
from nnodely.core.stream import Stream

import keras


def _sequence(name: str, seq) -> tuple:
    """The sequence axes ``seq`` declares, outermost first, a dynamic one as
    None; ``()`` if not given."""
    if seq is None:
        return ()
    seq = (seq,) if isinstance(seq, (int, str)) else tuple(seq)
    if None in seq:
        raise ValueError(
            f"{name}: a dynamic sequence length is declared with -1 or 'dyn', not None."
        )
    # "dyn" is -1 as a shape prints it; Keras marks a dynamic axis with None.
    return tuple(None if length in (-1, "dyn") else length for length in seq)


class Input(Stream):
    """
    A signal the model reads from data.

    ``dim`` sets the feature axes (one feature by default). ``seq`` sets the
    sequence: the axes outside the window, outermost first, used by
    rollouts; ``-1`` or ``"dyn"`` leaves a length dynamic.

    Read directly, an Input is its current sample, ``S=seq, T=1``. Its
    windows, requested with :meth:`sw` and :meth:`next`, set the window,
    ``T``, and keep the sequence. Requesting a window leaves the Input as it
    is: ``build()`` gives it the union of the windows the model's graph
    reads, the current sample included when the graph reads the Input
    directly - a window declared but never used widens nothing - and its
    ``shape`` then reports the data it is read from.
    """

    def __init__(
        self,
        name: str,
        *,
        dim: int | tuple | None = None,
        seq: int | str | tuple[int | str, ...] | None = None,
    ):
        super().__init__(name=name, dim=dim, seq=_sequence(name, seq), preds=None)
        # The window the graph of the last model built reads, found by build().
        self.past, self.future = (1, 0)
        # The window a saved model read, kept by the model loaded from it.
        self._saved_window = (0, 0)
        # The current sample, the node every direct read of the Input is.
        self._current_sample = None
        self._set_window(1, 0)

    @property
    def _current(self):
        """The current sample of the Input, as a relation reads it directly:
        one window, shared by every such read."""
        if self._current_sample is None:
            self._current_sample = SampleWindow(
                past=1, future=0, name=f"{self.name}_current"
            )([self])
        return self._current_sample

    def _set_window(self, past: int, future: int) -> None:
        """Read through ``past`` samples up to the current one and ``future``
        after it: the union of the windows of the graph being built. The
        Input's shape and the Keras input the data feeds follow it."""
        self.past, self.future = int(past), int(future)
        self.shape = self.shape.replace(time=self.past + self.future)
        self.input = keras.Input(shape=self.shape, name=self.name)

    def sw(self, window_size: int | list[int] | tuple[int, int]):
        """Return a window of samples of this input.

        ``sw(n)`` is the last ``n`` samples, up to and including the current
        one. ``sw([p, f])`` is ``p`` samples up to and including the current
        one followed by the next ``f`` samples. The window sets ``T`` and
        keeps the sequence. The Input itself is left as it is: ``build()``
        finds the windows a model reads.
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
            "seq": tuple(-1 if length is None else length for length in self.seq),
            "past": self.past,
            "future": self.future,
        }

    @classmethod
    def from_config(cls, config: dict, preds=None):
        node = cls(name=config["name"], dim=config["dim"], seq=config["seq"])
        node._saved_window = (config["past"], config["future"])
        node._set_window(*node._saved_window)
        return node


def read(stream):
    """``stream`` as a relation reads it: an Input read directly is its
    current sample, whatever window of it the data holds."""
    return stream._current if isinstance(stream, Input) else stream


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
