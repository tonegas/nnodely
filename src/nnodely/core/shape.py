"""The shape of a stream, and where its axes sit in a tensor.

A stream has three kinds of axes:

- ``seq``, its sequence axes, outermost first, possibly none: the steps of a
  trajectory. A Loop rolls out along the outermost one, and adds its rollout
  as a new outermost sequence axis. A sequence axis may be dynamic
  (``None``), its length known only at run time.
- ``time``, its window: the consecutive samples along which the temporal
  operations (windows, filters, derivatives, integrals) act. Every stream
  has one: a number, a Parameter or an Input read directly are one sample,
  ``time=1``.
- ``dim``, its feature axes, one or more.

A tensor holds them after its batch axis::

    (batch, *seq, time, *dim)

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
#: sequence. A rollout mask and sequence weights run along it.
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
    sample by default. ``dim`` is one feature by default.
    """

    dim: tuple[int, ...]
    time: int
    seq: tuple[int | None, ...]

    def __init__(self, dim=None, time=None, seq=None):
        # Always tuples: a shape read back from JSON arrives as lists, and two
        # shapes that describe the same axes have to compare equal.
        self.dim = _axes(dim, (1,))
        self.seq = _axes(seq, ())
        if time is None:
            time = 1
        if isinstance(time, bool) or not isinstance(time, int) or time < 1:
            raise TypeError(
                f"time is the length of the window, a positive int, got {time!r}; "
                "the axes outside the window are seq."
            )
        self.time = int(time)

    @classmethod
    def from_tensor_shape(
        cls, shape, *, seq_rank: int | None = None, dim_rank: int | None = None
    ) -> "Shape":
        """The Shape of a tensor of ``shape``, batch axis excluded, told apart
        by how many sequence axes, or dim axes, it has."""
        shape = tuple(shape)
        if seq_rank is None:
            seq_rank = len(shape) - 1 - int(dim_rank)  # type: ignore[arg-type]
        return cls(
            dim=shape[seq_rank + 1 :], time=shape[seq_rank], seq=shape[:seq_rank]
        )

    def replace(self, dim=_KEEP, time=_KEEP, seq=_KEEP) -> "Shape":
        """This shape with its dim axes, its window or its sequence changed."""
        return Shape(
            dim=self.dim if dim is _KEEP else dim,
            time=self.time if time is _KEEP else time,
            seq=self.seq if seq is _KEEP else seq,
        )

    @staticmethod
    def broadcast(shapes, owner: str) -> "Shape":
        """The shape of an elementwise operation between streams of ``shapes``.

        Sequence and time are never blended: the streams must have the same
        ones - a dynamic axis matches any length and stays dynamic - unless
        one stream is a single sample - every sequence axis and its window of
        size 1 - which broadcasts along the others', as numpy aligns them,
        from the window. The dim axes must be the same, unless one stream's
        are a scalar - every axis of size 1 - which broadcasts to the others'
        as numpy aligns them, from the last.

        And one stream must already have the shape of the result, the others
        broadcasting into it: two streams that would both be broadcast, one
        along the time of the other and the other along its dim - a matrix of
        one sample and the window of a scalar - are not combined.
        """
        steps: tuple | None = None  # every sequence axis, then the window
        for shape in shapes:
            if steps is None or _is_scalar(steps):
                steps = _broadcast_ones(steps, shape._steps)
            elif _is_scalar(shape._steps):
                steps = _broadcast_ones(shape._steps, steps)
            elif _same_steps(shape._steps, steps):
                steps = tuple(
                    None if None in (size, other) else size
                    for size, other in zip(shape._steps, steps)
                )
            else:
                raise ValueError(
                    f"{owner}: streams with a different sequence or time cannot "
                    f"be combined, got {Shape(time=steps[-1], seq=steps[:-1])!r} "
                    f"and {shape!r}. Only a single sample broadcasts along the "
                    "sequence and the time of another."
                )
        assert steps is not None

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

        result = Shape(dim=dim, time=steps[-1], seq=steps[:-1])
        if not any(shape._fills(result) for shape in shapes):
            raise ValueError(
                f"{owner}: streams of {' and '.join(map(repr, shapes))} cannot be "
                "combined: each would be broadcast, one along the time of another "
                "and that one along its dim. A single sample combines with a "
                "window only if its dim is a scalar or the same."
            )
        return result

    @property
    def _steps(self) -> tuple:
        """Every sequence axis, then the window: the axes before the dim."""
        return (*self.seq, self.time)

    def _fills(self, result: "Shape") -> bool:
        """Whether this shape is already ``result``'s, but for axes of 1: a
        stream of it would not be broadcast to ``result``."""
        return (
            _same_steps(_padded(self._steps, len(result._steps)), result._steps)
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
        sequence axis, or the window without a sequence."""
        return self.seq[0] if self.seq else self.time

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
        return (*self.seq, self.time, *self.dim)

    @property
    def seq_axes(self) -> tuple[int, ...]:
        """The sequence axes, outermost first."""
        return tuple(range(1, 1 + self.seq_rank))

    @property
    def time_axis(self) -> int:
        """The axis of the window."""
        return 1 + self.seq_rank

    @property
    def dim_axes(self) -> tuple[int, ...]:
        first = 2 + self.seq_rank
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
):
    """``x``, of ``seq_rank`` sequence and ``dim_rank`` dim axes, given axes
    of size 1 to broadcast against a tensor of ``to_seq_rank`` and
    ``to_dim_rank``, as :meth:`Shape.broadcast` lines them up.

    A single sample of fewer sequence axes gets its missing outer ones right
    after the batch, and a scalar dim of fewer axes its missing leading ones.
    ``expand_dims`` is ``np.expand_dims`` for numpy.
    """
    for _ in range(to_seq_rank - seq_rank):
        x = expand_dims(x, 1)
    for _ in range(to_dim_rank - dim_rank):
        x = expand_dims(x, 2 + to_seq_rank)
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
