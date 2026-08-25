"""
Stream - nodo del DAG con predecessors.
"""

from __future__ import annotations
import inspect

from copy import copy
from typing import Any, Sequence

NODE_REGISTRY: dict[str, type["Node"]] = {}


class Node:
    name: str
    preds: list[Node]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            NODE_REGISTRY[cls.__name__] = cls

    def __init__(self, name: str, preds: list[Node] | None = None) -> None:
        self.name = name
        self.preds = preds or []

    def __call__(self, inputs: Sequence[Node]) -> Any:
        node = copy(self)
        node.preds = list(inputs)

        return node

    def __repr__(self) -> str:
        return str(
            f'{self.__class__.__name__}("{self.name}: {[pred.name for pred in self.preds]}")'
        )

    def __hash__(self) -> int:
        return id(self)

    def get_config(self) -> dict:
        return {
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config, preds=None):
        return cls(**config)


class Shape:
    dim: tuple[int]
    time: int
    seq: tuple[int] | tuple[()]

    def __init__(self, dim=None, time=None, seq=None):
        self.dim = (1,) if dim is None else (dim,) if isinstance(dim, int) else dim
        self.time = 1 if time is None else time
        self.seq = () if seq is None else (seq,) if isinstance(seq, int) else seq

    @property
    def tuple(self) -> tuple:
        return tuple(self.dim) + (self.time,) + tuple(self.seq)

    @property
    def dimensions(self) -> tuple:
        return self.dim, self.time, self.seq

    @property
    def rank(self) -> int:
        return len(self.tuple)

    @property
    def dim_rank(self) -> int:
        return len(self.dim)

    @property
    def seq_rank(self) -> int:
        return len(self.seq)

    def with_dim(self, dim) -> "Shape":
        return Shape(dim=dim, time=self.time, seq=self.seq)

    def with_time(self, time) -> "Shape":
        return Shape(dim=self.dim, time=time, seq=self.seq)

    def with_seq(self, seq) -> "Shape":
        return Shape(dim=self.dim, time=self.time, seq=seq)

    def get_config(self):
        return {
            "dim": tuple(self.dim),
            "time": self.time,
            "seq": tuple(self.seq),
        }

    @classmethod
    def from_config(cls, config: dict) -> "Shape":
        return cls(
            dim=config.get("dim"),
            time=config.get("time"),
            seq=config.get("seq"),
        )

    def __iter__(self):
        return iter(self.tuple)

    def __len__(self):
        return len(self.tuple)

    def __getitem__(self, item):
        return self.tuple[item]

    def __repr__(self):
        return f"Shape(dim={self.dim}, time={self.time}, seq={self.seq})"


class Stream(Node):
    _literal_constant_cache = {}
    shape: Shape

    def __init__(
        self,
        name: str,
        dim=None,
        time=None,
        seq=None,
        preds: list[Node] | None = None,
    ):
        super().__init__(name, preds)
        self.shape = Shape(dim=dim, time=time, seq=seq)

    @property
    def dim(self) -> tuple[int]:
        return self.shape.dim

    @property
    def time(self) -> int:
        return self.shape.time

    @property
    def seq(self) -> tuple[int] | tuple[()]:
        return self.shape.seq

    @property
    def dimensions(self):
        """
        Return seq, time, dim separately
        """
        return self.shape.dimensions

    @property
    def rank(self) -> int:
        """
        Rank without batch dimension.
        """
        return self.shape.rank

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

    def __pow__(self, other):
        from nnodely.core.layer import Power

        return Power()([self, self._coerce_operand(other)])

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

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self.shape.get_config())
        return config
