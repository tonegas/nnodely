"""The shape of a stream, and where its axes sit in a tensor.

A stream has three kinds of axes:

- ``seq``, its sequence axes, outermost first, possibly none: the steps of a
  trajectory. A Loop rolls out along the outermost one, and adds its rollout
  as a new outermost sequence axis. A sequence axis may be dynamic
  (``None``), its length known only at run time.
- ``time``, its window: the consecutive samples along which the temporal
  operations (filters, derivatives, integrals, time selections) act, or
  ``None`` for a stream without one - an Input declared without ``sw``, a
  number, a Parameter or Constant declared without ``time``.
- ``dim``, its feature axes, one or more.

A tensor holds them after its batch axis, the window only when there is one::

    (batch, *seq, time, *dim)        (batch, *seq, *dim)

This module is the only one that knows that layout. A layer asks the Shape
of its stream where an axis is (:attr:`Shape.seq_axes`,
:attr:`Shape.time_axis`, :attr:`Shape.dim_axes`) and hands its Keras layer
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
    none; an int is one axis. ``time`` is the length of its window, one
    sample by default, or ``None`` for no window. ``dim`` is one feature by
    default.
    """

    dim: tuple[int, ...]
    time: int | None
    seq: tuple[int | None, ...]

    def __init__(self, dim=None, time: int | None = 1, seq=None):
        # Always tuples: a shape read back from JSON arrives as lists, and two
        # shapes that describe the same axes have to compare equal.
        self.dim = _axes(dim, (1,))
        self.seq = _axes(seq, ())
        if time is not None and (
            isinstance(time, bool) or not isinstance(time, int) or time < 1
        ):
            raise TypeError(
                f"time is the length of the window, a positive int or None, got "
                f"{time!r}; the axes outside the window are seq."
            )
        self.time = None if time is None else int(time)

    @classmethod
    def from_tensor_shape(
        cls,
        shape,
        *,
        seq_rank: int | None = None,
        dim_rank: int | None = None,
        time: bool = True,
    ) -> "Shape":
        """The Shape of a tensor of ``shape``, batch axis excluded, told apart
        by how many sequence axes, or dim axes, it has, and whether it has a
        window."""
        shape = tuple(shape)
        window = 1 if time else 0
        if seq_rank is None:
            seq_rank = len(shape) - window - int(dim_rank)  # type: ignore[arg-type]
        return cls(
            dim=shape[seq_rank + window :],
            time=shape[seq_rank] if time else None,
            seq=shape[:seq_rank],
        )

    def replace(self, dim=_KEEP, time=_KEEP, seq=_KEEP) -> "Shape":
        """This shape with its dim axes, its window or its sequence changed."""
        return Shape(
            dim=self.dim if dim is _KEEP else dim,
            time=self.time if time is _KEEP else time,  # type: ignore[arg-type]
            seq=self.seq if seq is _KEEP else seq,
        )

    @staticmethod
    def broadcast(shapes, owner: str, values=None) -> "Shape":
        """The shape of an elementwise operation between streams of ``shapes``.

        ``values`` tells, for each shape, whether its stream is a value - a
        number, a Parameter or a Constant, or a relation of those alone -
        rather than read from data.

        The window: equal, or one of them a single sample, ``T=1``, which
        broadcasts along the other's. A value without one applies to every
        sample of any stream; a stream from data without one does not combine
        with a stream that has one.

        The sequence: the same - a dynamic axis matches any length and stays
        dynamic - unless one stream's is a single step, every sequence axis of
        size 1, which broadcasts along the other's, as numpy aligns them, from
        the innermost. The dim axes must be the same, unless one stream's are
        a scalar - every axis of size 1 - which broadcasts to the others' as
        numpy aligns them, from the last.

        And one stream must already have the shape of the result, the others
        broadcasting into it: two streams that would both be broadcast, one
        along the time of the other and the other along its dim - a matrix of
        one sample and the window of a scalar - are not combined.
        """
        values = [False] * len(shapes) if values is None else list(values)
        windows = [shape.time for shape in shapes if shape.time is not None]
        time = max(windows) if windows else None
        for shape, value in zip(shapes, values):
            if shape.time is None and time is not None and not value:
                raise ValueError(
                    f"{owner}: a stream with no time, {shape!r}, cannot be "
                    f"combined with a stream with a window of {time}. Declare "
                    "the Input with a window, sw=1 for one sample."
                )
            if shape.time not in (None, 1, time):
                raise ValueError(
                    f"{owner}: streams with windows of {shape.time} and {time} "
                    "cannot be combined: only a single sample, T=1, broadcasts "
                    "along the window of another."
                )

        seq: tuple | None = None
        for shape in shapes:
            if seq is None or _is_scalar(seq):
                seq = _broadcast_ones(seq, shape.seq)
            elif _is_scalar(shape.seq):
                seq = _broadcast_ones(shape.seq, seq)
            elif _same_steps(shape.seq, seq):
                seq = tuple(
                    None if None in (size, other) else size
                    for size, other in zip(shape.seq, seq)
                )
            else:
                raise ValueError(
                    f"{owner}: streams with a different sequence cannot be "
                    f"combined, got {_sizes(seq)} and {shape!r}. Only a single "
                    "step broadcasts along the sequence of another."
                )
        assert seq is not None

        dim: tuple[int, ...] | None = None
        for shape in shapes:
            if dim is None or shape.dim == dim:
                dim = shape.dim
            elif _is_scalar(shape.dim) or _is_scalar(dim):
                # The scalar, padded with leading axes of 1, takes the other's.
                rank = max(len(dim), len(shape.dim))
                dim = tuple(
                    max(a, b)
                    for a, b in zip(_padded(dim, rank), _padded(shape.dim, rank))
                )
            else:
                raise ValueError(
                    f"{owner}: dim {dim} and dim {shape.dim} cannot be combined: "
                    "the dims must be the same, or one of them a scalar."
                )

        result = Shape(dim=dim, time=time, seq=seq)
        if not any(shape._fills(result, value) for shape, value in zip(shapes, values)):
            raise ValueError(
                f"{owner}: streams of {' and '.join(map(repr, shapes))} cannot be "
                "combined: each would be broadcast, one along the time of another "
                "and that one along its dim. A single sample combines with a "
                "window only if its dim is a scalar or the same."
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
        return () if self.time is None else (self.time,)

    def _fills(self, result: "Shape", value: bool = False) -> bool:
        """Whether this shape is already ``result``'s, but for axes of 1: a
        stream of it would not be broadcast to ``result``. A value without a
        window counts as one sample of ``result``'s."""
        time = (
            1 if value and self.time is None and result.time is not None else self.time
        )
        return (
            time == result.time
            and _same_steps(_padded(self.seq, len(result.seq)), result.seq)
            and _padded(self.dim, len(result.dim)) == result.dim
        )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    @property
    def dimensions(self) -> tuple:
        """The sequence, the window and the dim, in the order of the tensor."""
        return self.seq, self.time, self.dim

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
    def time_axis(self) -> int | None:
        """The axis of the window, None without one."""
        return None if self.time is None else 1 + self.seq_rank

    @property
    def dim_axes(self) -> tuple[int, ...]:
        first = 1 + self.seq_rank + len(self._window)
        return tuple(range(first, first + self.dim_rank))

    # ------------------------------------------------------------------
    # Serialization and display
    # ------------------------------------------------------------------
    def get_config(self):
        return {"dim": tuple(self.dim), "time": self.time, "seq": tuple(self.seq)}

    @classmethod
    def from_config(cls, config: dict) -> "Shape":
        return cls(
            dim=config.get("dim"), time=config.get("time"), seq=config.get("seq")
        )

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
        return f"Shape(S={_sizes(self.seq)}, T={self.time}, D={_sizes(self.dim)})"


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


def _broadcast_ones(ones, steps) -> tuple:
    """``steps`` broadcast with ``ones``, a single sample - every axis of
    size 1 - or None: ``steps`` given the leading axes ``ones`` has more."""
    return tuple(steps) if ones is None else _padded(steps, len(ones))


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

    A single sample of fewer sequence axes gets its missing outer ones right
    after the batch, a value without a window gets one, and a scalar dim of
    fewer axes its missing leading ones. ``expand_dims`` is
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
