"""
Stream - nodo del DAG con predecessors.
"""

from __future__ import annotations

from copy import copy
from typing import Sequence


class Node:
    name: str
    preds: list[Node]

    def __init__(self, name: str, preds: list[Node] | None = None) -> None:
        self.name = name
        self.preds = preds or []

    def __call__(self, inputs: Sequence[Node]) -> Node:
        node = copy(self)
        node.preds = list(inputs)

        return node

    def __repr__(self) -> str:
        return str(
            f'{self.__class__.__name__}("{self.name}: {[pred.name for pred in self.preds]}")'
        )

    def __hash__(self) -> int:
        return id(self)


class Stream(Node):
    _literal_constant_cache = {}
    seq: tuple[int | None, ...]
    time: int | tuple[()]
    dim: tuple[int, ...]

    def __init__(
        self,
        name: str | None = None,
        dim: int | tuple[int, ...] | None = None,
        time: int | tuple[()] | None = None,
        seq: int | tuple[int | None, ...] | None = None,
        preds: list[Node] | None = None,
    ):
        super().__init__(
            name if name is not None else f"{self.__class__.__name__}_{id(self)}", preds
        )
        self.dim = (1,) if dim is None else (dim,) if isinstance(dim, int) else dim
        self.time = () if time is None else time
        self.seq = () if seq is None else (seq,) if isinstance(seq, int) else seq

    @property
    def shape(self) -> tuple:
        """
        Shape without batch dimension.
        """
        time = (self.time,) if isinstance(self.time, int) else self.time
        return self.dim + time + self.seq

    @property
    def dimensions(self):
        """
        Return seq, time, dim separately
        """
        return self.dim, self.time, self.seq
    
    @property
    def shape_len(self):
        """
        Return seq, time, dim separately
        """
        time = (self.time,) if isinstance(self.time, int) else self.time
        return len(self.dim), len(time), len(self.seq)

    @property
    def rank(self) -> int:
        """
        Rank without batch dimension.
        """
        return len(self.shape)
    
    @property
    def time_index(self) -> int|None:
        """
        Return the index of the time dimension in the shape, or None if no time dimension.
        """
        return len(self.dim)+1 if isinstance(self.time, int) else None

    def __add__(self, other):
        from nnodely.core.layer import Add

        return Add()([self, self._coerce_operand(other)])

    def __radd__(self, other):
        from nnodely.core.layer import Add

        return Add()([self._coerce_operand(other), self])

    def __sub__(self, other):
        from nnodely.core.layer import Subtract

        return Subtract()([self, self._coerce_operand(other)])

    def __rsub__(self, other):
        from nnodely.core.layer import Subtract

        return Subtract()([self._coerce_operand(other), self])

    def __mul__(self, other):
        from nnodely.core.layer import Multiply

        return Multiply()([self, self._coerce_operand(other)])

    def __rmul__(self, other):
        from nnodely.core.layer import Multiply

        return Multiply()([self._coerce_operand(other), self])

    def __truediv__(self, other):
        from nnodely.core.layer import Divide

        return Divide()([self, self._coerce_operand(other)])

    def __rtruediv__(self, other):
        from nnodely.core.layer import Divide

        return Divide()([self._coerce_operand(other), self])

    @staticmethod
    def _coerce_operand(value):
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
    def _literal_constant_key(value):
        if isinstance(value, bool):
            return ("scalar_bool", bool(value))

        if isinstance(value, (int, float)):
            return ("scalar_num", float(value))

        if hasattr(value, "item"):
            try:
                return Stream._literal_constant_key(value.item())
            except Exception:
                return None

        return None

    def __repr__(self):
        pred_names = [pred.name for pred in self.preds]
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"type={self.__class__.__name__!r}, "
            f"shape={self.shape}, "
            f"predecessors={pred_names}"
            f")"
        )
