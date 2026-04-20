"""
Stream - Symbolic DAG node representing a data stream.
"""

from __future__ import annotations

from nnodely.core.dag import next_name
from nnodely.core.types import Shape


class Stream:
    """Symbolic DAG node representing a data stream.

    :param name: Unique node name in the graph.
    :param node_type: Semantic node type, e.g. 'input', 'output', 'Fir', 'SampleWindow'.
    :param seq: Optional sequence axes before time.
    :param time: Time/window axis.
    :param dim: Feature dimensions after time.
    :param predecessors: Parent nodes in the DAG.


    **Examples**

    Shape convention (without batch): seq + (time,) + dim.

    Basic usage:

    >>> Stream(seq=(), time=10, dim=(3,)) # shape = (10, 3)
    >>> Stream(seq=(4,), time=10, dim=(3,)) # shape = (4, 10, 3)
    """

    name: str
    node_type: str
    seq: Shape
    time: int
    dim: Shape
    predecessors: list[Stream]

    def __init__(
        self,
        name: str | None = None,
        node_type: str = "Stream",
        seq: int | Shape | None = None,
        time: int = 1,
        dim: Shape = (1,),
        predecessors: list[Stream] | None = None,
    ) -> None:
        self.name = str(name) if name is not None else next_name(node_type)
        self.node_type = str(node_type)
        self.seq = () if seq is None else (seq,) if isinstance(seq, int) else seq
        self.time = time
        self.dim = (dim,) if isinstance(dim, int) else dim
        self.predecessors = list(predecessors) if predecessors is not None else []

    @property
    def shape(self) -> Shape:
        """Shape without batch dimension."""
        return self.seq + (self.time,) + self.dim

    @property
    def dimensions(self):
        """
        Return seq, time, dim separately
        """
        return self.seq, self.time, self.dim

    @property
    def rank(self) -> int:
        """Rank without batch dimension."""
        return len(self.shape)

    def __add__(self, other):
        from nnodely.core.layer import Add

        return Add()(self, other)

    def __sub__(self, other):
        from nnodely.core.layer import Subtract

        return Subtract()(self, other)

    def __mul__(self, other):
        from nnodely.core.layer import Multiply

        return Multiply()(self, other)

    def __truediv__(self, other):
        from nnodely.core.layer import Divide

        return Divide()(self, other)

    def __repr__(self):
        pred_names = [pred.name for pred in self.predecessors]
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"type={self.node_type!r}, "
            f"shape={self.shape}, "
            f"predecessors={pred_names}"
            f")"
        )
