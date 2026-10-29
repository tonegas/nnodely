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


def _validate_init(owner: str, init):
    """The value before the first step, as configured - a Stream, a number or None."""
    if init is None or isinstance(init, Stream):
        return init
    if isinstance(init, bool) or not isinstance(init, (int, float)):
        raise TypeError(
            f"{owner}: init must be a Stream, a number or None, got "
            f"{type(init).__name__}."
        )
    return float(init)


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

    An Input can also be fed back, the value at the end of the previous step,
    with :meth:`close`::

        x = Input("x", sw=1)                          # x[k-1]
        x_next = x.close(IntegrateStep(dt=dt)(x, v))  # x[k]

    It is declared before the stream it is closed with, so the streams of a
    step can read each other's previous values, and coupled dynamics are
    written as they are, one Input per state variable.
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
        # Set by close(): the stream fed back and the value before the first
        # step. None while the Input is open, read from data.
        self.update: Stream | None = None
        self.init: Stream | float | None = None
        # Set by a layer that keeps its state in the Input - Integrate,
        # Derivative, Buffer - so the user has no handle on it: a Modely that
        # does not declare it reads its init in its place, not data.
        self._hidden = False

    def close(self, stream: Stream, init: Stream | float | None = None) -> Stream:
        """Feed ``stream`` back into this Input, and return it: the value of
        the next step, one sample, shifted into the window after the step.

        Nothing rolls out until a :class:`~nnodely.ModelyLoop` closes the
        model: every closed Input it reads, inside the models it calls too,
        becomes one feedback of a single loop. Built as a
        :class:`~nnodely.Modely` instead, the Input is read from data - the
        one-step model of the same graph - and the model warns unless it is
        among the declared inputs.

        ``init`` is the value before the first step: a number, a Parameter or
        a Constant, or an Input, of which the ModelyLoop reads the first step.
        With None, the Input starts from the first window of its own data when
        it is among the inputs of the ModelyLoop, from zero when the data
        leaves it out or the ModelyLoop does not list it.

        A model that declares the Input among its inputs, called as a block,
        reads the stream given for it as it would read its data: at every step
        inside a Modely; inside a ModelyLoop the Input keeps its feedback, and
        the stream is its initial value, in place of ``init``.
        """
        if self.future:
            raise ValueError(
                f"Input {self.name!r} reads future samples, sw=({self.past}, "
                f"{self.future}): closed, it holds values already computed, so "
                "its window reads no future samples."
            )
        if not isinstance(stream, Stream):
            raise TypeError(
                f"Input {self.name!r} is closed with a Stream, got "
                f"{type(stream).__name__}."
            )
        if self.update is not None:
            raise ValueError(
                f"Input {self.name!r} is already closed with {self.update.name!r}."
            )
        # One sample of the Input, or no window for an Input without one.
        sample = self.shape.replace(time=None if self.time is None else 1)
        if (
            stream.time != sample.time
            or tuple(stream.dim) != tuple(sample.dim)
            or tuple(stream.seq) != tuple(sample.seq)
        ):
            raise ValueError(
                f"Input {self.name!r} is closed with the value of one step, "
                f"{sample}; {stream.name!r} is {stream.shape}."
            )
        self.init = _validate_init(self.name, init)
        self.update = stream
        return stream

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


def _seq_cell(seq) -> str:
    """Sequence axes as a shape prints them, ``dyn`` for a dynamic one."""
    axes = ["dyn" if length is None else str(length) for length in seq]
    return f"({', '.join(axes)}{',' if len(axes) == 1 else ''})"


def _sw_cell(node: Input) -> str:
    """The window as ``sw`` declares it: n samples, or (past, future)."""
    if node.time is None:
        return "-"
    return str(node.past) if not node.future else f"({node.past}, {node.future})"


class Inputs(list):
    """The Inputs of a model, as :attr:`Modely.all_inputs` lists them: a list
    of the Input objects, printed as a table of their shapes and of how each
    is fed - whether it is declared, the stream it is closed with, what the
    data must hold of it and, fed back in a ModelyLoop, what it starts from."""

    _HEADER = (
        "name",
        "seq",
        "sw",
        "dim",
        "declared",
        "closed with",
        "data",
        "starts from",
    )

    def __init__(self, rows: list[dict], title: str = ""):
        super().__init__(row["input"] for row in rows)
        self.rows = rows
        self.title = title

    def __str__(self):
        cells = [
            (
                row["input"].name,
                _seq_cell(row["seq"]),
                _sw_cell(row["input"]),
                str(tuple(row["input"].dim)),
                "yes" if row["declared"] else "no",
                row["closed with"] or "-",
                row["data"] or "-",
                row["starts from"] or "-",
            )
            for row in self.rows
        ]
        widths = [
            max(len(line[column]) for line in (self._HEADER, *cells))
            for column in range(len(self._HEADER))
        ]

        def line(row):
            return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

        closed = sum(row["closed with"] is not None for row in self.rows)
        return "\n".join(
            [
                *([self.title] if self.title else []),
                line(self._HEADER).rstrip(),
                line(tuple("-" * width for width in widths)).rstrip(),
                *(line(row).rstrip() for row in cells),
                f"{len(self)} inputs, {closed} closed",
                "closed with: the stream a ModelyLoop feeds back into the Input at "
                "the next step; a Modely reads the Input from data instead",
                "data: required, optional (zeros where left out), target (to train "
                "and validate) or - (fed back only)",
            ]
        )

    __repr__ = __str__
