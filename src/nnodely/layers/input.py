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


def _validate_init(owner: str, init, name: str = "init"):
    """The value before the first step, as configured - a Stream, a number or None."""
    if init is None or isinstance(init, Stream):
        return init
    if isinstance(init, bool) or not isinstance(init, (int, float)):
        raise TypeError(
            f"{owner}: {name} must be a Stream, a number or None, got "
            f"{type(init).__name__}."
        )
    return float(init)


def _check_history(owner: str, init, signal, past: int, name: str = "init") -> None:
    """Refuse ``init``, the samples before the first one ``owner`` reads of
    ``signal``, unless it holds one sample - repeated when ``past`` are read -
    or ``past`` of them, oldest first, along its window: a value without a
    window - a number, a Parameter, a Constant - is one sample. Its dim is a
    scalar or the signal's, its sequence none or the signal's.

    The rule of every layer that starts from the past of a signal:
    ``IntegrateStep`` and ``DerivativeStep`` before a window, ``Integrate``,
    ``Derivative`` and ``Buffer`` before the first step of a loop."""
    samples, seq = init.shape.time, tuple(init.seq)
    if samples is None and not init._is_value:
        raise ValueError(
            f"{owner}: {name} has no window, {init.shape}, and is read from "
            "data: declare the Input with sw=1, one sample."
        )
    if samples not in (None, 1, past):
        held = (
            "a single sample"
            if past == 1
            else f"1 sample, repeated, or the {past} read before the first"
        )
        raise ValueError(
            f"{owner}: {name} holds the samples before the first, {held}; "
            f"got {samples}."
        )
    if tuple(init.dim) not in ((1,), tuple(signal.dim)):
        raise ValueError(
            f"{owner}: {name} has dim {tuple(init.dim)}, which is neither a "
            f"scalar nor the dim of {signal.name!r}, {tuple(signal.dim)}."
        )
    if seq not in ((), tuple(signal.seq)) or (samples is None and seq):
        raise ValueError(
            f"{owner}: {name} has the sequence {seq}, which is neither empty "
            f"nor the sequence of {signal.name!r}, {tuple(signal.seq)}."
        )


def _declared_seq(stream):
    """The ``seq`` that declares an Input with the sequence of ``stream``:
    a dynamic axis as -1, None without one."""
    return tuple(-1 if length is None else length for length in stream.seq) or None


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
    with :meth:`feedback`::

        x = Input("x", sw=1)                             # x[k-1]
        x_next = x.feedback(IntegrateStep(dt=dt)(x, v))  # x[k]

    It is declared before the stream it is fed back with, so the streams of a
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
        self._input = keras.Input(shape=self.shape.tuple, name=self.name)
        # Set by feedback(): the stream fed back. None while the Input is read
        # from data.
        self.update: Stream | None = None
        # The value before the first step, given with the feedback: only a
        # layer that keeps its state in the Input has one of its own.
        self.init: Stream | float | None = None
        # Set by a layer that keeps its state in the Input - Integrate,
        # Derivative, Buffer - so the user has no handle on it: a Modely that
        # does not declare it reads its init in its place, not data.
        self._hidden = False

    def feedback(self, stream: Stream) -> Stream:
        """Feed ``stream`` back into this Input, and return it: the value of
        the next step, one sample, shifted into the window after the step.

        Nothing rolls out until a :class:`~nnodely.Loop` or a
        :class:`~nnodely.ModelyLoop` applies the feedback: every fed-back
        Input the model reads, inside the models it calls too, becomes one
        feedback of a single loop. Built as a :class:`~nnodely.Modely`
        instead, the Input is read from data - the one-step model of the same
        graph - and the model warns unless it is among the declared inputs.

        The value before the first step is given where the feedback is
        applied: ``Loop(init={x: x0})``, or, in a ModelyLoop, the first window
        of the data of the Input when it is among the inputs. Without one, the
        Input starts from zero.

        A model that declares the Input among its inputs, called as a block,
        reads the stream given for it as it would read its data: at every step
        inside a Modely; inside a loop the Input keeps its feedback, and the
        stream is its initial value.
        """
        return self._feed_back(stream)

    def _feed_back(self, stream: Stream, init: Stream | float | None = None) -> Stream:
        """:meth:`feedback`, with the value before the first step: how a layer
        that keeps its state in an Input of its own starts it."""
        if self.future:
            raise ValueError(
                f"Input {self.name!r} reads future samples, sw=({self.past}, "
                f"{self.future}): fed back, it holds values already computed, so "
                "its window reads no future samples."
            )
        if not isinstance(stream, Stream):
            raise TypeError(
                f"Input {self.name!r} is fed back with a Stream, got "
                f"{type(stream).__name__}."
            )
        if self.update is not None:
            raise ValueError(
                f"Input {self.name!r} is already fed back with {self.update.name!r}."
            )
        # One sample of the Input, or no window for an Input without one.
        sample = self.shape.replace(time=None if self.time is None else 1)
        if (
            stream.time != sample.time
            or tuple(stream.dim) != tuple(sample.dim)
            or tuple(stream.seq) != tuple(sample.seq)
        ):
            raise ValueError(
                f"Input {self.name!r} is fed back with the value of one step, "
                f"{sample}; {stream.name!r} is {stream.shape}."
            )
        self.init = _validate_init(self.name, init)
        self.update = stream
        return stream

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": tuple(self.dim),
                "seq": tuple(-1 if length is None else length for length in self.seq),
                "sw": None if self.time is None else (self.past, self.future),
            }
        )
        return config


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
    """The Inputs of a model, as :attr:`Modely.inputs` lists them: a list
    of the Input objects, printed as a table of their shapes and of how each
    is fed - whether it is declared, the stream it is fed back with, what the
    data must hold of it and, fed back by a Loop, what it starts from."""

    _HEADER = (
        "name",
        "seq",
        "sw",
        "dim",
        "declared",
        "fed back with",
        "data",
        "starts from",
    )

    def __init__(self, rows: list[dict], title: str = ""):
        super().__init__(row["input"] for row in rows)
        self._rows = rows
        self._title = title

    def __str__(self):
        cells = [
            (
                row["input"].name,
                _seq_cell(row["seq"]),
                _sw_cell(row["input"]),
                str(tuple(row["input"].dim)),
                "yes" if row["declared"] else "no",
                row["fed back with"] or "-",
                row["data"] or "-",
                row["starts from"] or "-",
            )
            for row in self._rows
        ]
        widths = [
            max(len(line[column]) for line in (self._HEADER, *cells))
            for column in range(len(self._HEADER))
        ]

        def line(row):
            return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

        fed_back = sum(row["fed back with"] is not None for row in self._rows)
        return "\n".join(
            [
                *([self._title] if self._title else []),
                line(self._HEADER).rstrip(),
                line(tuple("-" * width for width in widths)).rstrip(),
                *(line(row).rstrip() for row in cells),
                f"{len(self)} inputs, {fed_back} fed back",
                "fed back with: the stream a loop feeds back into the Input at "
                "the next step; a Modely reads the Input from data instead",
                "data: required, optional (zeros where left out), target (to train "
                "and validate) or - (fed back only)",
            ]
        )

    __repr__ = __str__
