"""
Stream - nodo del DAG con predecessors.
"""

from nnodely.dag import to_tuple
from nnodely.mixins import _ArithmeticMixin


class Stream(_ArithmeticMixin):
    """
    Data stream con shape (S1,...,B,T,D1,D2,...).
    seq, time, dim. predecessors = nodi che alimentano questo stream.
    """

    def __init__(
        self,
        name: str,
        node_type: str,
        seq: tuple,
        time: int,
        dim: tuple,
        predecessors: list,
        **attrs
    ):
        self.name = name
        self.node_type = node_type
        self.seq = to_tuple(seq, ())
        self.time = time
        self.dim = to_tuple(dim, (1,))
        self.predecessors = list(predecessors)
        for k, v in attrs.items():
            setattr(self, k, v)

    @property
    def shape(self) -> tuple:
        """Shape senza batch."""
        if self.seq:
            return self.seq + (self.time,) + self.dim
        return (self.time,) + self.dim

    def call(self, x):
        """Pass-through per Keras."""
        return x
