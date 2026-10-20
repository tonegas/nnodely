"""The shape of a stream, and where its axes sit in a tensor.

A stream has two kinds of axes:

- ``time``, its time axes, outermost first: ``(T_k, ..., T_2, T_1)``. The
  inner time ``T_1`` is the window of samples, along which the temporal
  operations (windows, filters, derivatives) act; a Loop rolls out along
  the outer time ``T_k``, and adds its rollout as a new outer time. A time
  axis may be dynamic (``None``), its length known only at run time. A
  stream may have no time axis at all - a number, a Parameter, an Input
  read whole.
- ``dim``, its feature axes, one or more.

A tensor holds them after its batch axis::

    (batch, *time, *dim)

This module is the only one that knows that layout. A layer asks the Shape
of its stream where an axis is (:attr:`Shape.inner_time_axis`,
:attr:`Shape.time_axes`, :attr:`Shape.dim_axes`) and hands its Keras layer
the tensor axis to act on; code that sees only tensors uses the helpers at
the end of this module.
"""

from __future__ import annotations

from typing import Callable

import keras

#: The tensor axis of the batch.
BATCH_AXIS = 0

#: The tensor axis of the outer time of any stream with time: the rollout of
#: its outermost Loop, or its window when it has none. A rollout mask and
#: sequence weights run along it, and a new rollout is stacked on it.
OUTER_TIME_AXIS = 1


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

    ``time`` is the tuple of its time axes, outermost first, ``()`` for a
    stream without time; an int is one axis. ``dim`` is one feature by
    default.
    """

    dim: tuple[int, ...]
    time: tuple[int | None, ...]

    def __init__(self, dim=None, time=None):
        # Always tuples: a shape read back from JSON arrives as lists, and two
        # shapes that describe the same axes have to compare equal.
        self.dim = _axes(dim, (1,))
        self.time = _axes(time, ())

    @classmethod
    def from_tensor_shape(
        cls, shape, *, time_rank: int | None = None, dim_rank: int | None = None
    ) -> "Shape":
        """The Shape of a tensor of ``shape``, batch axis excluded, told apart
        by how many time axes, or dim axes, it has."""
        shape = tuple(shape)
        if time_rank is None:
            time_rank = len(shape) - int(dim_rank)  # type: ignore[arg-type]
        return cls(dim=shape[time_rank:], time=shape[:time_rank])

    def replace(self, dim=None, time=None) -> "Shape":
        """This shape with its dim axes, or all its time axes, changed."""
        return Shape(
            dim=self.dim if dim is None else dim,
            time=self.time if time is None else time,
        )

    @staticmethod
    def broadcast(shapes, owner: str) -> "Shape":
        """The shape of an elementwise operation between streams of ``shapes``.

        Time is never blended: the streams with time axes must have the same
        ones - a dynamic axis matches any length and stays dynamic - and a
        stream without time broadcasts along them. The dim axes must be the
        same, unless one stream's are a scalar - every axis of size 1 - which
        broadcasts to the others' as numpy aligns them, from the last.

        And one stream must already have the shape of the result, the others
        broadcasting into it: two streams that would both be broadcast, one
        along the time of the other and the other along its dim - a matrix
        without time and the window of a scalar - are not combined.
        """
        time: tuple | None = None
        for shape in shapes:
            if not shape.time:
                continue
            if time is None:
                time = shape.time
                continue
            if len(shape.time) != len(time) or any(
                None not in (size, other) and size != other
                for size, other in zip(shape.time, time)
            ):
                raise ValueError(
                    f"{owner}: streams with different time axes cannot be combined, "
                    f"got {_sizes(time)} and {_sizes(shape.time)}. Only a stream "
                    "without time broadcasts along the time of another."
                )
            time = tuple(
                None if None in (size, other) else size
                for size, other in zip(shape.time, time)
            )

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

        result = Shape(dim=dim, time=time)
        if not any(shape._fills(result) for shape in shapes):
            raise ValueError(
                f"{owner}: streams of {' and '.join(map(repr, shapes))} cannot be "
                "combined: each would be broadcast, one along the time of another "
                "and that one along its dim. A stream without time combines with "
                "a timed one only if its dim is a scalar or the same."
            )
        return result

    def _fills(self, result: "Shape") -> bool:
        """Whether this shape is already ``result``'s, but for axes of 1: a
        stream of it would not be broadcast to ``result``."""
        if not self.time and any(size != 1 for size in result.time):
            return False
        return _padded(self.dim, len(result.dim)) == result.dim

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    @property
    def dimensions(self) -> tuple:
        """The time axes and the dim axes, in the order of the tensor."""
        return self.time, self.dim

    @property
    def rank(self) -> int:
        """Number of axes, batch aside."""
        return len(self.tuple)

    @property
    def dim_rank(self) -> int:
        return len(self.dim)

    @property
    def time_rank(self) -> int:
        return len(self.time)

    def resolve_dim(self, axis: int, owner: str) -> int:
        """Dim axis ``axis``, a negative one counted from the last, checked."""
        return resolve_axis(axis, self.dim_rank, owner, "dim")

    def resolve_time(self, axis: int, owner: str) -> int:
        """Time axis ``axis``, counted from the outermost, a negative one from
        the inner time, checked."""
        return resolve_axis(axis, self.time_rank, owner, "time")

    # ------------------------------------------------------------------
    # The inner time, where temporal operations act, and the outer time,
    # where a Loop rolls out
    # ------------------------------------------------------------------
    @property
    def inner_time(self) -> int | None:
        """Samples of the inner time ``T_1``, the window."""
        return self._timed()[-1]

    @property
    def outer_time(self) -> int | None:
        """Steps of the outer time ``T_k``: the outermost rollout, or the
        window without one."""
        return self._timed()[0]

    def with_outer_time(self, length: int | None) -> "Shape":
        """This shape rolled out by one more Loop, of ``length`` steps: a new
        outer time."""
        return self.replace(time=(length, *self.time))

    def without_outer_time(self) -> "Shape":
        """The shape of one step of the outer time, as a Loop's body sees it."""
        return self.replace(time=self.time[1:])

    def _timed(self) -> tuple:
        if not self.time:
            raise ValueError(f"{self} has no time axis.")
        return tuple(self.time)

    # ------------------------------------------------------------------
    # Layout: the tensor axes, counted with the batch axis as 0
    # ------------------------------------------------------------------
    @property
    def tuple(self) -> tuple:
        """The sizes of the axes of one sample, in the order of the tensor."""
        return (*self.time, *self.dim)

    @property
    def time_axes(self) -> tuple[int, ...]:
        """The time axes, outermost first."""
        return tuple(range(1, 1 + self.time_rank))

    @property
    def dim_axes(self) -> tuple[int, ...]:
        first = 1 + self.time_rank
        return tuple(range(first, first + self.dim_rank))

    @property
    def inner_time_axis(self) -> int:
        """The axis of the inner time ``T_1``."""
        return len(self._timed())

    @property
    def outer_time_axis(self) -> int:
        """The axis of the outer time ``T_k``."""
        self._timed()
        return OUTER_TIME_AXIS

    # ------------------------------------------------------------------
    # Serialization and display
    # ------------------------------------------------------------------
    def get_config(self):
        return {"dim": tuple(self.dim), "time": tuple(self.time)}

    @classmethod
    def from_config(cls, config: dict) -> "Shape":
        return cls(dim=config.get("dim"), time=config.get("time"))

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
        return f"Shape(T={_sizes(self.time)}, D={_sizes(self.dim)})"


def _is_scalar(dim) -> bool:
    """Whether a dim holds one value: every axis of size 1."""
    return all(size == 1 for size in dim)


def _padded(dim, rank: int) -> tuple:
    """``dim`` given leading axes of 1 up to ``rank``."""
    return (1,) * (rank - len(dim)) + tuple(dim)


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
    time_rank: int,
    dim_rank: int,
    to_time_rank: int,
    to_dim_rank: int,
    expand_dims: Callable = keras.ops.expand_dims,
):
    """``x``, of ``time_rank`` time and ``dim_rank`` dim axes, given axes of
    size 1 to broadcast against a tensor of ``to_time_rank`` and
    ``to_dim_rank``, as :meth:`Shape.broadcast` lines them up.

    A stream without time gets the time axes right after the batch, and a
    scalar dim of fewer axes its missing leading ones. ``expand_dims`` is
    ``np.expand_dims`` for numpy.
    """
    for _ in range(to_time_rank - time_rank):
        x = expand_dims(x, 1)
    for _ in range(to_dim_rank - dim_rank):
        x = expand_dims(x, 1 + to_time_rank)
    return x


def expand_to_rank(x, rank: int, expand_dims: Callable = keras.ops.expand_dims):
    """``x`` given the axes it lacks to have ``rank``, of size 1, so that it
    broadcasts along them.

    An operand may lack the outer time axes: a number or a Constant carries
    no rollout. ``expand_dims`` is ``np.expand_dims`` for numpy.
    """
    while len(x.shape) < rank:
        x = expand_dims(x, 1)
    return x


def outer_time_view(x, rank: int, expand_dims: Callable = keras.ops.expand_dims):
    """``x``, laid out ``(batch, steps)``, given axes of size 1 so that it
    broadcasts against a stream of ``rank`` axes with its steps along
    :data:`OUTER_TIME_AXIS`. ``expand_dims`` is ``np.expand_dims`` for numpy.
    """
    for _ in range(rank - 2):
        x = expand_dims(x, -1)
    return x
