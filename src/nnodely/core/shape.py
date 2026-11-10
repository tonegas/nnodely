"""The shape of a stream, and where its axes sit in a tensor.

A stream has three kinds of axes:

- ``seq``, its sequence axes, outermost first, possibly none: the steps of a
  trajectory. A Loop rolls out along the outermost one, and adds its rollout
  as a new outermost sequence axis. A sequence axis may be dynamic
  (``None``), its length known only at run time.
- ``tw``, its time window: the consecutive samples along which the temporal
  operations (filters, derivatives, integrals, time selections) act, or
  ``None`` for a stream without one - an Input declared without ``tw``, a
  number, a Parameter or Constant declared without ``tw``.
- ``dim``, its feature axes, one or more.

A tensor holds them after its batch axis, the window only when there is one::

    (batch, *seq, tw, *dim)        (batch, *seq, *dim)

This module is the only one that knows that layout. A layer asks the Shape
of its stream where an axis is (:attr:`Shape.seq_axes`,
:attr:`Shape.tw_axis`, :attr:`Shape.dim_axes`) and hands its Keras layer
the tensor axis to act on; code that sees only tensors uses the helpers at
the end of this module.
"""

from __future__ import annotations

from typing import Callable

import keras

#: The tensor axis of the batch.
BATCH_AXIS = 0

#: The first tensor axis after the batch: the outermost sequence axis - the
#: rollout of the outermost Loop - or the window of a stream without a
#: sequence, or its first dim axis. A rollout mask and sequence weights run
#: along it.
OUTER_AXIS = 1

#: Given for an argument left as it is.
_KEEP = object()


def _axes(value, default) -> tuple:
    """``value`` as a tuple of axis sizes: None is ``default``, an int one axis."""
    if value is None:
        return default
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def resolve_axis(axis: int, rank: int, owner: str, kind: str) -> int:
    """``axis`` among ``rank`` axes of a kind, a negative one counted from
    the last; one out of bounds is an error of ``owner``."""
    resolved = axis + rank if axis < 0 else axis
    if not 0 <= resolved < rank:
        raise ValueError(f"{owner}: axis {axis} out of bounds for {kind} rank {rank}.")
    return resolved


class Shape:
    """The sizes of the axes of a stream, batch aside.

    ``seq`` is the tuple of its sequence axes, outermost first, ``()`` for
    none; an int is one axis. ``tw`` is the length of its window, one
    sample by default, or ``None`` for no window. ``dim`` is one feature by
    default.
    """

    dim: tuple[int, ...]
    tw: int | None
    seq: tuple[int | None, ...]

    def __init__(self, dim=None, tw: int | None = 1, seq=None):
        # Always tuples: a shape read back from JSON arrives as lists, and two
        # shapes that describe the same axes have to compare equal.
        self.dim = _axes(dim, (1,))
        self.seq = _axes(seq, ())
        if tw is not None and (
            isinstance(tw, bool) or not isinstance(tw, int) or tw < 1
        ):
            raise TypeError(
                f"tw is the length of the window, a positive int or None, got "
                f"{tw!r}; the axes outside the window are seq."
            )
        self.tw = None if tw is None else int(tw)

    @classmethod
    def from_tensor_shape(
        cls,
        shape,
        *,
        seq_rank: int | None = None,
        dim_rank: int | None = None,
        tw: bool = True,
    ) -> "Shape":
        """The Shape of a tensor of ``shape``, batch axis excluded, told apart
        by how many sequence axes, or dim axes, it has, and whether it has a
        window."""
        shape = tuple(shape)
        window = 1 if tw else 0
        if seq_rank is None:
            seq_rank = len(shape) - window - int(dim_rank)  # type: ignore[arg-type]
        return cls(
            dim=shape[seq_rank + window :],
            tw=shape[seq_rank] if tw else None,
            seq=shape[:seq_rank],
        )

    def replace(self, dim=_KEEP, tw=_KEEP, seq=_KEEP) -> "Shape":
        """This shape with its dim axes, its window or its sequence changed."""
        return Shape(
            dim=self.dim if dim is _KEEP else dim,
            tw=self.tw if tw is _KEEP else tw,  # type: ignore[arg-type]
            seq=self.seq if seq is _KEEP else seq,
        )

    @staticmethod
    def broadcast(shapes, owner: str) -> "Shape":
        """The shape of an elementwise operation between streams of ``shapes``.

        Every stream follows the same rule, read from data or not - an Input,
        a number, a Parameter, a Constant: an axis it does not have broadcasts
        along the others', an axis it has must match theirs.

        The window: the same, or none, which applies to every sample of the
        others'. A single sample, ``T=1``, is a window like any other.

        The sequence: the same - a dynamic axis matches any length and stays
        dynamic - or none, which applies at every step of the others'. A
        single step is a sequence like any other.

        The dim axes: lined up from the last, as Keras broadcasts them - a dim
        of fewer axes gets leading axes of 1 - each axis the same, or 1 in one
        of them, which stretches along the others'.

        And one stream must already have the shape of the result, the others
        broadcasting into it: two streams that would both be broadcast, one
        along the window of the other and the other along its dim - a matrix
        without a window and the window of a scalar - are not combined.
        Streams of the same sequence and window are the exception: their dims
        broadcast both ways, as Keras's do.
        """
        windows = sorted({shape.tw for shape in shapes if shape.tw is not None})
        if len(windows) > 1:
            hint = (
                " A sample taken with TimeSelect(idx, keep_tw=False) has none."
                if windows[0] == 1
                else ""
            )
            raise ValueError(
                f"{owner}: streams with windows of "
                f"{' and '.join(map(str, windows))} cannot be combined: the "
                "windows must be the same, or a stream have none, which applies "
                f"to every sample of the other's.{hint}"
            )
        tw = windows[0] if windows else None

        seq: tuple = ()
        for shape in shapes:
            if not shape.seq:
                continue
            if seq and not _same_steps(shape.seq, seq):
                raise ValueError(
                    f"{owner}: streams with a different sequence cannot be "
                    f"combined, got {_sizes(seq)} and {shape!r}: the sequences "
                    "must be the same, or a stream have none, which applies at "
                    "every step of the other's."
                )
            seq = (
                tuple(None if None in pair else pair[0] for pair in zip(shape.seq, seq))
                if seq
                else shape.seq
            )

        rank = max(shape.dim_rank for shape in shapes)
        dim = [1] * rank
        for shape in shapes:
            for axis, size in enumerate(_padded(shape.dim, rank)):
                if size != 1 and dim[axis] not in (1, size):
                    raise ValueError(
                        f"{owner}: dims "
                        f"{' and '.join(str(s.dim) for s in shapes)} cannot be "
                        "combined: lined up from the last axis, as Keras "
                        "broadcasts them, each axis must be the same, or 1 in "
                        "one of them."
                    )
                dim[axis] = max(dim[axis], size)

        result = Shape(dim=tuple(dim), tw=tw, seq=seq)
        same_steps = all(
            shape.tw == tw and _same_steps(shape.seq, seq) for shape in shapes
        )
        if not same_steps and not any(shape._fills(result) for shape in shapes):
            raise ValueError(
                f"{owner}: streams of {' and '.join(map(repr, shapes))} cannot be "
                f"combined: none has the shape of the result, {result!r}, so "
                "each would be broadcast, one along the window or the sequence "
                "of another and that one along its dim."
            )
        return result

    @property
    def _steps(self) -> tuple:
        """Every sequence axis, then the window if there is one: the axes
        before the dim."""
        return (*self.seq, *self._window)

    @property
    def _window(self) -> tuple:
        """The window as axes: one, or none."""
        return () if self.tw is None else (self.tw,)

    def _fills(self, result: "Shape") -> bool:
        """Whether this shape is already ``result``'s, but for leading dim
        axes of 1: a stream of it would not be broadcast to ``result``. A
        stream without a window or a sequence is broadcast along
        ``result``'s, even of a single sample or a single step."""
        return (
            self.tw == result.tw
            and _same_steps(self.seq, result.seq)
            and _padded(self.dim, len(result.dim)) == result.dim
        )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    @property
    def dimensions(self) -> tuple:
        """The sequence, the window and the dim, in the order of the tensor."""
        return self.seq, self.tw, self.dim

    @property
    def rank(self) -> int:
        """Number of axes, batch aside."""
        return len(self.tuple)

    @property
    def dim_rank(self) -> int:
        return len(self.dim)

    @property
    def seq_rank(self) -> int:
        return len(self.seq)

    def resolve_dim(self, axis: int, owner: str) -> int:
        """Dim axis ``axis``, a negative one counted from the last, checked."""
        return resolve_axis(axis, self.dim_rank, owner, "dim")

    def resolve_seq(self, axis: int, owner: str) -> int:
        """Sequence axis ``axis``, counted from the outermost, a negative one
        from the innermost, checked."""
        return resolve_axis(axis, self.seq_rank, owner, "seq")

    @property
    def outer(self) -> int | None:
        """Length of the outer axis (:data:`OUTER_AXIS`): the outermost
        sequence axis, or the window without a sequence, or the first dim
        axis without either."""
        return self.tuple[0]

    def with_seq(self, length: int | None) -> "Shape":
        """This shape rolled out by one more Loop, of ``length`` steps: a new
        outermost sequence axis."""
        return self.replace(seq=(length, *self.seq))

    def without_outer_seq(self) -> "Shape":
        """The shape of one step of the outermost sequence axis, as the body
        of a Loop rolling along it sees it."""
        return self.replace(seq=self.seq[1:])

    # ------------------------------------------------------------------
    # Layout: the tensor axes, counted with the batch axis as 0
    # ------------------------------------------------------------------
    @property
    def tuple(self) -> tuple:
        """The sizes of the axes of one sample, in the order of the tensor."""
        return (*self.seq, *self._window, *self.dim)

    @property
    def seq_axes(self) -> tuple[int, ...]:
        """The sequence axes, outermost first."""
        return tuple(range(1, 1 + self.seq_rank))

    @property
    def tw_axis(self) -> int | None:
        """The axis of the window, None without one."""
        return None if self.tw is None else 1 + self.seq_rank

    @property
    def dim_axes(self) -> tuple[int, ...]:
        first = 1 + self.seq_rank + len(self._window)
        return tuple(range(first, first + self.dim_rank))

    # ------------------------------------------------------------------
    # Serialization and display
    # ------------------------------------------------------------------
    def get_config(self):
        return {"dim": tuple(self.dim), "tw": self.tw, "seq": tuple(self.seq)}

    @classmethod
    def from_config(cls, config: dict) -> "Shape":
        return cls(dim=config.get("dim"), tw=config.get("tw"), seq=config.get("seq"))

    # A Shape reads as the sizes of one sample, as keras.Input takes them.
    def __iter__(self):
        return iter(self.tuple)

    def __len__(self):
        return len(self.tuple)

    def __getitem__(self, item):
        return self.tuple[item]

    def __repr__(self):
        # Named axes: a bare (3, 10) reads as well as three features of ten
        # samples as ten features of three.
        return f"Shape(S={_sizes(self.seq)}, T={self.tw}, D={_sizes(self.dim)})"


def _is_scalar(dim) -> bool:
    """Whether a dim holds one value: every axis of size 1."""
    return all(size == 1 for size in dim)


def _padded(dim, rank: int) -> tuple:
    """``dim`` given leading axes of 1 up to ``rank``."""
    return (1,) * (rank - len(dim)) + tuple(dim)


def _same_steps(steps, other) -> bool:
    """Whether two sequences and windows are the same, a dynamic axis
    matching any length."""
    return len(steps) == len(other) and all(
        None in (size, length) or size == length for size, length in zip(steps, other)
    )


def _sizes(values) -> str:
    """Axis sizes as a tuple, a dynamic one as "dyn"."""
    items = ["dyn" if value is None else str(value) for value in values]
    return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def slice_axis(x, axis: int, index):
    """``x`` indexed along ``axis`` alone, by a slice, or by an integer,
    which drops the axis."""
    items: list = [slice(None)] * len(x.shape)
    items[axis] = index
    return x[tuple(items)]


def transpose(x, permutation):
    """``x`` with its axes permuted; one that keeps them in place is no op."""
    permutation = list(permutation)
    if permutation == list(range(len(permutation))):
        return x
    return keras.ops.transpose(x, permutation)


def align(
    x,
    seq_rank: int,
    dim_rank: int,
    to_seq_rank: int,
    to_dim_rank: int,
    expand_dims: Callable = keras.ops.expand_dims,
    *,
    window: bool = True,
    to_window: bool = True,
):
    """``x``, of ``seq_rank`` sequence and ``dim_rank`` dim axes and a window
    or not, given axes of size 1 to broadcast against a tensor of
    ``to_seq_rank`` and ``to_dim_rank`` and a window or not, as
    :meth:`Shape.broadcast` lines them up.

    A stream of fewer sequence axes gets its missing outer ones right after
    the batch, a stream without a window gets one, and a dim of fewer axes
    its missing leading ones, as Keras lines dims up. ``expand_dims`` is
    ``np.expand_dims`` for numpy.
    """
    for _ in range(to_seq_rank - seq_rank):
        x = expand_dims(x, 1)
    if to_window and not window:
        x = expand_dims(x, 1 + to_seq_rank)
    for _ in range(to_dim_rank - dim_rank):
        x = expand_dims(x, 1 + to_seq_rank + int(to_window))
    return x


def expand_to_rank(x, rank: int, expand_dims: Callable = keras.ops.expand_dims):
    """``x`` given the axes it lacks to have ``rank``, of size 1, so that it
    broadcasts along them.

    An operand may lack the outer sequence axes: a number or a Constant
    carries no rollout. ``expand_dims`` is ``np.expand_dims`` for numpy.
    """
    while len(x.shape) < rank:
        x = expand_dims(x, 1)
    return x


def outer_view(x, rank: int, expand_dims: Callable = keras.ops.expand_dims):
    """``x``, laid out ``(batch, steps)``, given axes of size 1 so that it
    broadcasts against a stream of ``rank`` axes with its steps along
    :data:`OUTER_AXIS`. ``expand_dims`` is ``np.expand_dims`` for numpy.
    """
    for _ in range(rank - 2):
        x = expand_dims(x, -1)
    return x
