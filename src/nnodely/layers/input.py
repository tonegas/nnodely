"""Input: a signal read from data, a root of the graph."""

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


def _window(name: str, sw) -> tuple[int, int] | None:
    """The samples ``sw`` declares, ``(past, future)``; None without a window."""
    if sw is None:
        return None
    window = tuple(sw) if isinstance(sw, (list, tuple)) else (sw, 0)
    if (
        len(window) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) for n in window)
        or min(window) < 0
        or sum(window) < 1
    ):
        raise ValueError(
            f"{name}: sw is the window, n samples up to and including the "
            "current one, or (past, future) samples, at least one in all; got "
            f"{sw!r}."
        )
    return window


class Input(Stream):
    """
    A signal the model reads from data, declared once with the shape of that
    data.

    ``sw`` sets the window, ``T``: ``sw=n`` is the last ``n`` samples, up to
    and including the current one, and ``sw=(past, future)`` is ``past``
    samples up to the current one followed by the next ``future``, so
    ``T = past + future``. Without ``sw`` the Input has no window: one value
    per sample, which does not combine with a stream that has one. ``dim``
    sets the feature axes, one feature by default. ``seq`` sets the
    sequence: the axes outside the window, outermost first, used by
    rollouts; ``-1`` or ``"dyn"`` leaves a length dynamic. A sequence needs
    a window.

    An Input does not change: the layers that read it - ``TimeSelect``,
    ``TimeRange`` and the others - take what they need of its window.
    """

    def __init__(
        self,
        name: str,
        *,
        dim: int | tuple | None = None,
        seq: int | str | tuple[int | str, ...] | None = None,
        sw: int | tuple[int, int] | None = None,
    ):
        sequence = _sequence(name, seq)
        window = _window(name, sw)
        if sequence and window is None:
            raise ValueError(
                f"{name}: a sequence needs a window: declare sw as well, sw=1 "
                "for one sample per step."
            )
        super().__init__(
            name=name,
            dim=dim,
            time=None if window is None else sum(window),
            seq=sequence,
            preds=None,
        )
        # The samples read up to and including the current one, and after it:
        # without a window, the current one alone.
        self.past, self.future = (1, 0) if window is None else window
        self.input = keras.Input(shape=self.shape.tuple, name=self.name)

    def get_config(self):
        return {
            "name": self.name,
            "dim": tuple(self.dim),
            "seq": tuple(-1 if length is None else length for length in self.seq),
            "sw": None if self.time is None else (self.past, self.future),
        }

    @classmethod
    def from_config(cls, config: dict, preds=None):
        sw = config["sw"]
        return cls(
            name=config["name"],
            dim=config["dim"],
            seq=config["seq"] or None,
            sw=None if sw is None else tuple(sw),
        )
