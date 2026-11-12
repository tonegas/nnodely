"""Input: a signal read from data, a root of the graph."""

from numbers import Integral
from typing import Any

from nnodely.core.dag import Table
from nnodely.core.shape import _same_steps
from nnodely.core.stream import Stream
from nnodely.utils.printers import _table

import keras


def _positive_sizes(sizes) -> tuple[int, ...] | None:
    """``sizes`` - an int, or a list or tuple of them - as a tuple of ints;
    None unless each is at least 1."""
    items = (
        (sizes,)
        if isinstance(sizes, Integral)
        else tuple(sizes)
        if isinstance(sizes, (list, tuple))
        else None
    )
    if items is None or not all(
        isinstance(n, Integral) and not isinstance(n, bool) and int(n) >= 1
        for n in items
    ):
        return None
    return tuple(int(n) for n in items)


def _dimension(name: str, dim) -> tuple[int, ...]:
    """The feature axes ``dim`` declares, one feature by default: one axis
    at least, each of at least one feature. The rule of an Input, a Parameter
    and a Constant."""
    if dim is None:
        return (1,)
    sizes = _positive_sizes(dim)
    if not sizes:
        raise ValueError(
            f"{name}: dim is the size of each feature axis, an int or a tuple of "
            f"them, each at least 1; got {dim!r}."
        )
    return sizes


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


def _window(name: str, tw) -> tuple[int, int] | None:
    """The samples ``tw`` declares, ``(past, future)``; None without a window."""
    if tw is None:
        return None
    window = tuple(tw) if isinstance(tw, (list, tuple)) else (tw, 0)
    if (
        len(window) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) for n in window)
        or min(window) < 0
        or sum(window) < 1
    ):
        raise ValueError(
            f"{name}: tw is the window, n samples up to and including the "
            "current one, or (past, future) samples, at least one in all; got "
            f"{tw!r}."
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
    scalar or the signal's, its sequence none or the signal's, a dynamic axis
    matching any length.

    The rule of every layer that starts from the past of a signal:
    ``IntegrateStep`` and ``DerivativeStep`` before a window, ``Integrate``,
    ``Derivative`` and ``Buffer`` before the first step of a loop."""
    samples, seq = init.shape.tw, tuple(init.seq)
    if samples is None and not init._is_value:
        raise ValueError(
            f"{owner}: {name} has no window, {init.shape}, and is read from "
            "data: declare the Input with tw=1, one sample."
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
    if (seq and not _same_steps(seq, signal.seq)) or (samples is None and seq):
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

    ``tw`` sets the window, ``T``: ``tw=n`` is the last ``n`` samples, up to
    and including the current one, and ``tw=(past, future)`` is ``past``
    samples up to the current one followed by the next ``future``, so
    ``T = past + future``. Without ``tw`` the Input has no window: one value
    per sample, which does not combine with a stream that has one. ``dim``
    sets the feature axes, one feature by default, each of at least one
    feature. ``seq`` sets the
    sequence: the axes outside the window, outermost first, used by
    rollouts; ``-1`` or ``"dyn"`` leaves a length dynamic. A sequence needs
    a window.

    An Input does not change: the layers that read it - ``TimeSelect``,
    ``TimeRange`` and the others - take what they need of its window.

    An Input can also be fed back, the value at the end of the previous step,
    with :meth:`feedback`::

        x = Input("x", tw=1)                             # x[k-1]
        x_next = x.feedback(IntegrateStep(dt=dt)(x, v))  # x[k]

    It is declared before the stream it is fed back with, so the streams of a
    step can read each other's previous values, and coupled dynamics are
    written as they are, one Input per state variable.
    """

    # The Keras input the data is fed to, named as the Input when built.
    _input: Any

    def __init__(
        self,
        name: str,
        *,
        dim: int | tuple | None = None,
        seq: int | str | tuple[int | str, ...] | None = None,
        tw: int | tuple[int, int] | None = None,
    ):
        sequence = _sequence(name, seq)
        window = _window(name, tw)
        if sequence and window is None:
            raise ValueError(
                f"{name}: a sequence needs a window: declare tw as well, tw=1 "
                "for one sample per step."
            )
        super().__init__(
            name=name,
            dim=_dimension(name, dim),
            tw=None if window is None else sum(window),
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
                f"Input {self.name!r} reads future samples, tw=({self.past}, "
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
        sample = self.shape.replace(tw=None if self.tw is None else 1)
        if (
            stream.tw != sample.tw
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
                "tw": None if self.tw is None else (self.past, self.future),
            }
        )
        return config


def _seq_cell(seq) -> str:
    """Sequence axes as a shape prints them, ``dyn`` for a dynamic one."""
    axes = ["dyn" if length is None else str(length) for length in seq]
    return f"({', '.join(axes)}{',' if len(axes) == 1 else ''})"


def _tw_cell(node: Input) -> str:
    """The window as ``tw`` declares it: n samples, or (past, future)."""
    if node.tw is None:
        return "-"
    return str(node.past) if not node.future else f"({node.past}, {node.future})"


class Inputs(Table):
    """The Inputs of a model, as :attr:`Modely.inputs` lists them: a list
    of the Input objects, printed as a table of their shapes and of how each
    is fed - whether it is declared, the stream it is fed back with, what the
    data must hold of it and, fed back by a Loop, what it starts from."""

    _HEADER = (
        "index",
        "input",
        "seq",
        "tw",
        "dim",
        "declared",
        "fed back with",
        "data",
        "starts from",
    )

    _kind, _attr = "Input", "inputs"

    def __init__(self, rows: list[dict]):
        super().__init__(row["input"] for row in rows)
        self._rows = rows

    def _where(self, index, item) -> str:
        return "(declared)" if self._rows[index]["declared"] else "(not declared)"

    def __str__(self):
        cells = [
            (
                str(index),
                row["input"].name,
                _seq_cell(row["seq"]),
                _tw_cell(row["input"]),
                str(tuple(row["input"].dim)),
                "yes" if row["declared"] else "no",
                row["fed back with"] or "-",
                row["data"] or "-",
                row["starts from"] or "-",
            )
            for index, row in enumerate(self._rows)
        ]
        return _table(self._HEADER, [("", cells)])

    __repr__ = __str__
