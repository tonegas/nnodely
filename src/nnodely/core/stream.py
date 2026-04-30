"""
Stream - Symbolic DAG node representing a data stream.
"""

from __future__ import annotations

from nnodely.utils.type_defs import StreamOperand


class Stream:
    """

    Symbolic DAG node representing a data stream.

    Shape convention (without batch):
        seq + (time,) + dim

    Examples:
        seq=(), time=10, dim=(3,)      -> shape = (10, 3)
        seq=(4,), time=10, dim=(3,)    -> shape = (4, 10, 3)

    Attributes
    ----------
    name : str|None
        Unique node name in the graph.
    seq : int | tuple[int, ...] | None
        Optional sequence axes before time.
    time : int | None
        Time/window axis.
    dim : int | tuple[int, ...] | None
        Feature dimensions after time.
    predecessors : list[Stream]
        Parent nodes in the DAG.
    """

    _literal_constant_cache = {}

    def __init__(
        self,
        name: str | None = None,
        seq: int | tuple[int, ...] | None = None,
        time: int | None = None,
        dim: int | tuple[int, ...] | None = None,
        predecessors: list["Stream"] | None = None,
    ):
        from nnodely.core.dag import next_name
        self.name = (
            str(name) if name is not None else next_name(self.__class__.__name__)
        )
        self.seq = () if seq is None else (seq,) if isinstance(seq, int) else seq
        self.time = 1 if time is None else time
        self.dim = (1,) if dim is None else (dim,) if isinstance(dim, int) else dim
        self.predecessors = list(predecessors) if predecessors is not None else []

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape without batch dimension.
        """
        return self.seq + (self.time,) + self.dim

    @property
    def dimensions(self) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
        """
        Return seq, time, dim separately
        """
        return self.seq, self.time, self.dim

    @property
    def rank(self) -> int:
        """Rank without batch dimension."""
        return len(self.shape)

    def __add__(self, other: StreamOperand) -> "Stream":
        from nnodely.core.layer import Add

        return Add()(self, self._coerce_operand(other))

    def __radd__(self, other: StreamOperand) -> "Stream":
        from nnodely.core.layer import Add

        return Add()(self._coerce_operand(other), self)

    def __sub__(self, other: StreamOperand) -> "Stream":
        from nnodely.core.layer import Subtract

        return Subtract()(self, self._coerce_operand(other))

    def __rsub__(self, other: StreamOperand) -> "Stream":
        from nnodely.core.layer import Subtract

        return Subtract()(self._coerce_operand(other), self)

    def __mul__(self, other: StreamOperand) -> "Stream":
        from nnodely.core.layer import Multiply

        return Multiply()(self, self._coerce_operand(other))

    def __rmul__(self, other: StreamOperand) -> "Stream":
        from nnodely.core.layer import Multiply

        return Multiply()(self._coerce_operand(other), self)

    def __truediv__(self, other: StreamOperand) -> "Stream":
        from nnodely.core.layer import Divide

        return Divide()(self, self._coerce_operand(other))

    def __rtruediv__(self, other: StreamOperand) -> "Stream":
        from nnodely.core.layer import Divide

        return Divide()(self._coerce_operand(other), self)

    @staticmethod
    def _coerce_operand(value: StreamOperand) -> "Stream":
        if isinstance(value, Stream):
            return value

        from nnodely.layers.constant import Constant

        key = Stream._literal_constant_key(value)
        if key is not None:
            cached = Stream._literal_constant_cache.get(key)
            if cached is None:
                cached = Constant(name=None, value=key[1])
                Stream._literal_constant_cache[key] = cached
            return cached

        return Constant(name=None, value=value)

    @staticmethod
    def _literal_constant_key(
        value: list | int | float,
    ) -> tuple[str, list[int | float | list] | float]:
        if isinstance(value, (int, float)):
            return ("scalar", float(value))

        return ("array", value)

    def __repr__(self) -> str:
        pred_names = [pred.name for pred in self.predecessors]
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"type={self.__class__.__name__!r}, "
            f"shape={self.shape}, "
            f"predecessors={pred_names}"
            f")"
        )
