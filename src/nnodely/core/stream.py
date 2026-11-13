"""Stream: a node of the graph, with its shape and its predecessors."""

from __future__ import annotations
import inspect

from copy import copy
from typing import Any, Sequence

from nnodely.core.shape import Shape

#: Every node class, by its module and name (:func:`class_path`): a model
#: saves the class of each node so, and is loaded with that class, not with
#: another of the same name - a layer class of the user's named ``Sin``
#: leaves nnodely's own in place.
NODE_REGISTRY: dict[str, type["Node"]] = {}


def class_path(cls: type) -> str:
    """``cls`` by its module and its name, as a model saves it."""
    return f"{cls.__module__}.{cls.__qualname__}"


def registered_class(saved: str) -> type["Node"] | None:
    """The node class a model was saved with: by its module and name; else
    by its name alone, as a model saved before classes were saved with their
    module, or one whose class has moved to another module - when exactly one
    class has that name. None if no class has it."""
    if saved in NODE_REGISTRY:
        return NODE_REGISTRY[saved]
    name = saved.rsplit(".", 1)[-1]
    alike = [cls for cls in NODE_REGISTRY.values() if cls.__qualname__ == name]
    if len(alike) > 1:
        raise ValueError(
            f"The saved model holds a {saved!r} node, and {len(alike)} classes "
            f"are named {name!r}: {sorted(class_path(cls) for cls in alike)}. "
            "Import the module that defines the one it was saved with, from "
            "where it was saved."
        )
    return alike[0] if alike else None


#: Every node name this process has used - given, generated or loaded - with
#: the "_"-separated prefixes it is built from. Layers are told apart by the
#: object they come from, but names are how a graph is read and searched, so a
#: generated name avoids all of these.
NODE_NAMES: set[str] = set()


def reserve_name(name: str) -> None:
    parts = name.split("_")
    NODE_NAMES.update("_".join(parts[:index]) for index in range(1, len(parts) + 1))


def object_repr(obj, **fields) -> str:
    """How every object of nnodely prints: ``Class(field=value, ...)`` on
    one line. An object it refers to - a stream, a layer, a model - prints
    by its name; reading the attribute gives the object."""
    text = ", ".join(
        f"{key}={getattr(value, 'name', value)!r}" for key, value in fields.items()
    )
    return f"{type(obj).__name__}({text})"


class Node:
    name: str
    _preds: list[Node]
    #: The model the node holds - the one it calls, or rolls out - or None.
    _model: Any = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            NODE_REGISTRY[class_path(cls)] = cls

    def __init__(self, name: str, preds: list[Node] | None = None) -> None:
        self.name = name
        self._preds = preds or []
        reserve_name(name)

    def __call__(self, inputs: Sequence[Node]) -> Any:
        node = copy(self)
        node._preds = list(inputs)

        return node

    def __repr__(self) -> str:
        return str(
            f'{self.__class__.__name__}("{self.name}: {[pred.name for pred in self._preds]}")'
        )

    def __hash__(self) -> int:
        return id(self)

    def get_config(self) -> dict:
        """The arguments of the constructor that rebuild the node: every class
        adds those its own constructor takes to its parent's."""
        return {
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config, preds=None):
        """The node ``config`` describes, made by the constructor of ``cls``.
        A class that reads predecessors takes ``preds`` from here."""
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
        tw=None,
        preds: list[Node] | None = None,
        seq=None,
    ):
        super().__init__(name, preds)
        self.shape = Shape(dim=dim, tw=tw, seq=seq)

    @property
    def dim(self) -> tuple[int, ...]:
        return self.shape.dim

    @property
    def tw(self) -> int | None:
        """The length of the window, None without one."""
        return self.shape.tw

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

    def __rpow__(self, other):
        from nnodely.core.layer import Power

        return Power()([self._coerce_operand(other), self])

    def __mod__(self, other):
        from nnodely.core.layer import Mod

        return Mod()([self, self._coerce_operand(other)])

    def __rmod__(self, other):
        from nnodely.core.layer import Mod

        return Mod()([self._coerce_operand(other), self])

    def __neg__(self):
        from nnodely.layers.arithmetic import Negative

        return Negative()([self])

    def __pos__(self):
        return self

    def __abs__(self):
        from nnodely.layers.arithmetic import Abs

        return Abs()([self])

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
        """An Input or an Output prints as its class. A stream a call
        returns - of a layer, an operator, a model or a loop - prints as a
        Stream, whatever made it."""
        from nnodely.layers.input import Input
        from nnodely.layers.output import Output

        if type(self) in (Input, Output):
            return object_repr(self, name=self.name, shape=self.shape)
        return f"Stream(name={self.name!r}, shape={self.shape})"
