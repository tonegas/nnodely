"""Input: a signal read from data, a root of the graph."""

from numbers import Integral
from typing import Any

from nnodely.core.dag import Table
from nnodely.core.layer import Layer
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
    r"""A signal the model reads from data, declared with the shape of one
    sample.

    Syntax
    ------
    ``Input(name)``
        One value per sample, without a window.
    ``Input(name, tw=n)``
        A window of the last ``n`` samples, up to and including the current
        one.
    ``Input(name, tw=(p, f))``
        A window of ``p`` samples up to and including the current one,
        followed by the next ``f``.
    ``Input(name, dim=d, seq=s, tw=n)``
        A sequence of ``s`` windows of ``n`` samples of ``d`` features.
    ``x.feedback(stream)``
        Feeds ``stream`` back into the Input ``x`` at the next step of a
        rollout.
    ``x.feedback(stream, init=x0, hidden=True)``
        The same, ``x`` a hidden state: the model's own memory, which starts
        from ``x0`` and is none of its inputs.

    An Input is a stream. The data feeds it, and the layers that read it take
    what they need of its window. The data of an Input is keyed by its name:
    in the dict given to a model call, and in a :class:`DataLoader`.

    ``tw`` sets the window, ``T``, counted in samples. ``dim`` sets the
    feature axes, ``D``. ``seq`` sets the sequence axes, ``S``, outermost
    first: a :class:`Loop` rolls out along the outermost one. ``x.shape``
    prints the three, ``Shape(S=(...), T=..., D=(...))``, with ``T=None``
    without a window and ``dyn`` for a dynamic length.

    Parameters
    ----------
    name : str
        The name of the Input, the key of its data. Two Inputs of one model
        cannot share it.
    dim : int or tuple of int, optional
        The size of each feature axis, each at least 1. ``dim=3`` is three
        features per sample, ``dim=(3, 2)`` a 3x2 matrix per sample. The
        default is one feature, ``(1,)``.
    seq : int, str or tuple, optional
        The length of each sequence axis, outermost first. Each is:

        - an int of at least 1, a fixed length;
        - ``-1`` or ``"dyn"``, a dynamic length, which the data sets.

        A sequence needs ``tw``. The default is no sequence, ``S=()``.
    tw : int or tuple of int, optional
        The window. It is one of:

        - ``n``, the last ``n`` samples, up to and including the current
          one, so ``T = n``;
        - ``(p, f)``, ``p`` samples up to and including the current one,
          then the next ``f``, so ``T = p + f``;
        - not given: no window, one value per sample.

        Every number is at least 0, and ``T`` at least 1.

    Attributes
    ----------
    name : str
        The name, the key of the data. It can be changed, ``x.name = "y"``.
    tw : int or None
        The length of the window, ``T``, or ``None`` without one.
    dim : tuple of int
        The feature axes, ``D``.
    seq : tuple
        The sequence axes, ``S``, outermost first. A dynamic length is
        ``None``.
    past : int
        The samples of the window up to and including the current one: ``p``
        of ``tw=(p, f)``, ``n`` of ``tw=n``, and 1 without a window.
    future : int
        The samples of the window after the current one: ``f`` of
        ``tw=(p, f)``, and 0 otherwise.

    Returns
    -------
    Input
        A stream of ``Shape(S, T, D)``: ``Input("u", dim=3, seq=5, tw=10)``
        is ``Shape(S=(5,), T=10, D=(3,))``. The data of one sample is an
        array of shape ``(*S, T, *D)``, ``(*S, *D)`` without a window.

    Raises
    ------
    ValueError
        If ``tw`` is not an allowed window ("tw is the window"), if a size
        of ``dim`` is below 1 ("dim is the size of each feature axis"), if
        ``seq`` is given without ``tw`` ("a sequence needs a window"), or if
        a length of ``seq`` is ``None`` ("declared with -1 or 'dyn'").

    See Also
    --------
    Output : Names a stream as an output of the model.
    Parameter : A trainable value.
    DataLoader : Cuts recorded signals into samples of the Inputs.
    TimeSelect : One sample of a window.
    ModelyLoop : A model declared one step at a time, with fed-back Inputs.

    Notes
    -----
    A model call takes, for each Input, an array of shape
    ``(batch, *S, T, *D)``, or ``(batch, *S, *D)`` without a window. The
    batch axis is always there, also for a single sample.

    The window of ``tw=(p, f)`` at the current sample :math:`t` holds

    .. math:: x[t-p+1], \dots, x[t], x[t+1], \dots, x[t+f]

    An Input without a window combines with a stream that has one: its value
    applies to every sample of that window.

    Basic Examples
    --------------
    A window of the last three samples:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, ModelyLoop, Output, TimeSelect
    >>> x = Input("x", tw=3)
    >>> x.shape
    Shape(S=(), T=3, D=(1,))
    >>> model = Modely("newest", inputs=[x], outputs=[Output("y", TimeSelect(-1)(x))]).build()
    >>> result = model({"x": np.array([[[1.0], [2.0], [3.0]]])})
    >>> keras.ops.convert_to_numpy(result["y"])
    array([[[3.]]], dtype=float32)

    Scope
    -----
    Without ``tw`` there is no window:

    >>> Input("u").shape
    Shape(S=(), T=None, D=(1,))

    Several features, and a matrix per sample:

    >>> Input("v", dim=3, tw=10).shape
    Shape(S=(), T=10, D=(3,))
    >>> Input("m", dim=(3, 2), tw=1).shape
    Shape(S=(), T=1, D=(3, 2))

    A sequence of windows, of a fixed or a dynamic length:

    >>> Input("w", dim=3, seq=5, tw=10).shape
    Shape(S=(5,), T=10, D=(3,))
    >>> Input("d", seq=(4, -1), tw=1).shape
    Shape(S=(4, dyn), T=1, D=(1,))

    Options
    -------
    ``name`` is the key of the data, and ``dim`` the features:

    >>> u = Input("u", dim=2)
    >>> u.name, u.dim
    ('u', (2,))

    ``tw=(p, f)`` reads ``f`` samples after the current one. ``past`` and
    ``future`` give the two parts:

    >>> target = Input("target", tw=(0, 1))
    >>> target.tw, target.past, target.future
    (1, 0, 1)

    ``seq="dyn"`` is ``seq=-1``, as a shape prints it:

    >>> Input("s", seq="dyn", tw=1).seq
    (None,)

    Applications
    ------------
    A mass-spring-damper predicts the next position from the last positions
    and the current force. The target is the position one sample ahead:

    >>> position = Input("position", tw=4)
    >>> force = Input("force", tw=1)
    >>> next_position = Input("next_position", tw=(0, 1))

    A state fed back at every step of a rollout counts up from its data:

    >>> count = Input("count", tw=1)
    >>> count_next = count.feedback(count + 1.0)
    >>> counter = ModelyLoop("counter", inputs=[count], outputs=[Output("c", count_next)], seq=3)
    >>> result = counter.build()({"count": np.full((1, 3, 1, 1), 10.0)})
    >>> keras.ops.convert_to_numpy(result["c"])[0, :, 0, 0]
    array([11., 12., 13.], dtype=float32)

    Properties & Relations
    ----------------------
    ``tw=n`` is ``tw=(n, 0)``, and ``T`` is ``past + future``:

    >>> a, b = Input("a", tw=3), Input("b", tw=(3, 0))
    >>> (a.past, a.future) == (b.past, b.future)
    True
    >>> target.tw == target.past + target.future
    True

    Possible Issues
    ---------------
    A sequence needs a window. Declare ``tw=1`` for one sample per step:

    >>> Input("bad", seq=5)
    Traceback (most recent call last):
        ...
    ValueError: bad: a sequence needs a window: declare tw as well, tw=1 for one sample per step.
    >>> Input("good", seq=5, tw=1).shape
    Shape(S=(5,), T=1, D=(1,))

    A window holds one sample at least:

    >>> Input("empty", tw=0)
    Traceback (most recent call last):
        ...
    ValueError: empty: tw is the window, ...at least one in all; got 0.

    The data of a model call has a batch axis. An array of one sample without
    it is read wrongly, or raises: add the axis with ``v[None]``.

    .. versionadded:: 2.0
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
        # A hidden state: the model's own memory, never one of its inputs. Set
        # by feedback(hidden=True), or by a layer that keeps its state in the
        # Input - Integrate, Derivative, Buffer. A Modely that does not declare
        # it - only the body of a Loop does - reads its init in its place, not
        # data.
        self._hidden = False
        # Hidden by the user, feedback(hidden=True): it keeps the name the user
        # gave it, where the state of a layer yields its own.
        self._hidden_by_user = False

    def feedback(
        self,
        stream: Stream,
        *,
        init: Stream | float | None = None,
        hidden: bool = False,
    ) -> Stream:
        """Feed ``stream`` back into this Input at the next step of a rollout,
        and return ``stream``.

        ``stream`` is the value of the Input at the end of this step, one
        sample. At the next step the Input reads it, shifted into its window
        as the newest sample. Inside a step, the Input holds the value at the
        end of the previous step.

        Nothing rolls out until a :class:`Loop` or a :class:`ModelyLoop`
        applies the feedback. Every fed-back Input the model reads, also
        inside the models it calls, is fed back by the same rollout. Built as
        a :class:`Modely` instead, the graph is the one-step model.

        A fed-back Input is a state, of one of two kinds:

        - declared among the inputs of the model that reads it: an input,
          given by the data or by the model's caller. A Modely reads it from
          data at every step. A Loop starts it from its ``init``,
          ``Loop(init={x: x0})``; a ModelyLoop from the first window of its
          data, zero where the data leaves it out; a block from the stream its
          caller gives it.
        - hidden, ``hidden=True``: the model's own memory, as the state of an
          :class:`Integrate`. It is never an input, and cannot be declared. It
          starts from ``init``, zero without one, in every rollout - once per
          call of the model that holds it - and a Modely reads ``init`` in its
          place at every step.

        Parameters
        ----------
        stream : Stream
            The value of one step: the dim and the sequence of the Input,
            ``T=1``, or no window for an Input without one.
        init : Stream or float, optional
            The value of a hidden state before the first step: a number, a
            :class:`Parameter` - a trainable initial state - a
            :class:`Constant`, or a declared Input, whose first step is read.
            It holds one sample, repeated over the window, or the whole
            window, oldest first; its dim is a scalar or the Input's. Only a
            hidden state takes one. The default is zero.
        hidden : bool, optional
            Whether the Input is a hidden state, the model's own memory. The
            default is False: the model that reads it declares it.

        Returns
        -------
        Stream
            ``stream`` itself, so the call can be written inline:
            ``x_next = x.feedback(x + v)``.

        Raises
        ------
        ValueError
            If ``stream`` is not one sample of the Input ("fed back with the
            value of one step"), if the Input reads future samples ("reads
            future samples"), if it is already fed back ("is already fed
            back"), if ``init`` is given to a state that is not hidden ("only
            a hidden state"), or if ``init`` holds another number of samples,
            features or sequence ("init holds the samples before the first").
        TypeError
            If ``stream`` is not a stream ("is fed back with a Stream"), or if
            ``init`` is not a stream, a number or None.

        See Also
        --------
        ModelyLoop : A model declared one step at a time.
        Loop : Rolls a model out along the outermost sequence axis.
        IntegrateStep : One step of an integrator, often fed back.

        Examples
        --------
        A counter that adds one at every step, starting from its data:

        >>> import keras
        >>> import numpy as np
        >>> from nnodely import Input, ModelyLoop, Output, TimeSelect
        >>> count = Input("count", tw=1)
        >>> count_next = count.feedback(count + 1.0)
        >>> counter = ModelyLoop("counter", inputs=[count], outputs=[Output("c", count_next)], seq=3)
        >>> result = counter.build()({"count": np.full((1, 3, 1, 1), 10.0)})
        >>> keras.ops.convert_to_numpy(result["c"])[0, :, 0, 0]
        array([11., 12., 13.], dtype=float32)

        A window of several samples is fed back with one sample:

        >>> x = Input("x", tw=2)
        >>> x.feedback(x + 1.0)
        Traceback (most recent call last):
            ...
        ValueError: Input 'x' is fed back with the value of one step, Shape(S=(), T=1, D=(1,)); ...
        >>> x.feedback(TimeSelect(-1)(x) + 1.0).shape
        Shape(S=(), T=1, D=(1,))

        A hidden state is the model's own: it starts from ``init``, and the
        model does not declare it:

        >>> total, u = Input("total", tw=1), Input("u", tw=1)
        >>> total_next = total.feedback(total + u, init=10.0, hidden=True)
        >>> summer = ModelyLoop("summer", inputs=[u], outputs=[Output("s", total_next)], seq=3)
        >>> result = summer.build()({"u": np.ones((1, 3, 1, 1))})
        >>> keras.ops.convert_to_numpy(result["s"])[0, :, 0, 0]
        array([11., 12., 13.], dtype=float32)
        """
        if init is not None and not hidden:
            raise ValueError(
                f"Input {self.name!r}: only a hidden state takes an init, "
                "x.feedback(stream, init=..., hidden=True). A declared state "
                "starts from its data, or from the stream the caller of its "
                "model gives it."
            )
        if isinstance(init, Stream):
            _check_history(f"Input {self.name!r}", init, self, self.past)
        self._feed_back(stream, init=init)
        if hidden:
            self._hidden = self._hidden_by_user = True
        return stream

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


@keras.saving.register_keras_serializable(package="nnodely")
class _HiddenInitImpl(keras.layers.Layer):
    """The init of a hidden state, ``(batch, ...)``, broadcast to the shape of
    the state, ``shape``, as a loop starts the state from it."""

    def __init__(self, shape, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.state_shape = tuple(shape)

    def call(self, init):
        from nnodely.layers.loop import LoopImpl

        return LoopImpl._align_to_shape(init, self.state_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"shape": self.state_shape})
        return config


class _HiddenInit(Layer):
    """The init of the hidden state ``state``, which a Modely reads in its
    place, with the shape of the state: the layers that read the state were
    declared on that shape, and a number, a Constant or one sample of an Input
    has fewer axes."""

    def output_shape(self, *inputs):
        return self._properties["state"]

    def build_layer(self):
        return _HiddenInitImpl(self._properties["state"].tuple, name=self.name)


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
    """The Inputs of a model, as :attr:`Modely.inputs` lists them.

    It is a list of the Input objects. Pick one by its name,
    ``inputs["x"]``, or by its index, ``inputs[0]``: the ``idx`` column of
    the table. Printed, it is a table of their shapes and of how each is
    fed: whether it is declared, the stream it is fed back with, what the
    data must hold of it and, fed back by a Loop, what it starts from."""

    _HEADER = (
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
                row["input"].name,
                _seq_cell(row["seq"]),
                _tw_cell(row["input"]),
                str(tuple(row["input"].dim)),
                "yes" if row["declared"] else "no",
                row["fed back with"] or "-",
                row["data"] or "-",
                row["starts from"] or "-",
            )
            for row in self._rows
        ]
        return _table(self._HEADER, [("", cells)], indexed=True)

    __repr__ = __str__
