"""Stream: a node of the graph, with its shape and its predecessors."""

from __future__ import annotations
import inspect

from copy import copy
from typing import Any, Sequence

from nnodely.core.shape import Shape

NODE_REGISTRY: dict[str, type["Node"]] = {}

#: Every node name this process has used - given, generated or loaded - with
#: the "_"-separated prefixes it is built from. Layers are told apart by the
#: object they come from, but names are how a graph is read and searched, so a
#: generated name avoids all of these.
NODE_NAMES: set[str] = set()


def reserve_name(name: str) -> None:
    parts = name.split("_")
    NODE_NAMES.update("_".join(parts[:index]) for index in range(1, len(parts) + 1))


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
        reserve_name(name)

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


class Stream(Node):
    _literal_constant_cache = {}
    shape: Shape
    #: Whether the stream is a value - a number, a Parameter or a Constant, or
    #: a relation of those alone - rather than read from data. A value without
    #: a window applies to every sample of any stream.
    _is_value = False

    def __init__(
        self,
        name: str,
        dim=None,
        time=None,
        preds: list[Node] | None = None,
        seq=None,
    ):
        super().__init__(name, preds)
        self.shape = Shape(dim=dim, time=time, seq=seq)

    @property
    def dim(self) -> tuple[int, ...]:
        return self.shape.dim

    @property
    def time(self) -> int | None:
        """The length of the window, None without one."""
        return self.shape.time

    @property
    def seq(self) -> tuple[int | None, ...]:
        """The sequence axes, outermost first."""
        return self.shape.seq

    @property
    def dimensions(self):
        """
        Return the sequence, the window and the dim axes separately
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

        from nnodely.layers.parameter import Constant

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
