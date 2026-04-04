"""
Stream - nodo del DAG con predecessors.
"""

from nnodely.core.dag import to_tuple, next_name


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
    name : str
        Unique node name in the graph.
    node_type : str
        Semantic node type, e.g. 'input', 'output', 'Fir', 'SampleWindow'.
    seq : tuple
        Optional sequence axes before time.
    time : int
        Time/window axis.
    dim : tuple
        Feature dimensions after time.
    predecessors : list[Stream]
        Parent nodes in the DAG.
    """

    def __init__(
        self,
        name: str|None = None,
        node_type: str = 'Stream',
        seq=(),
        time: int = 1,
        dim=(1,),
        predecessors=None,
    ):
        self.name = str(name) if name is not None else next_name(node_type)
        self.node_type = str(node_type)
        self.seq = to_tuple(seq, ())
        self.time = int(time)
        self.dim = to_tuple(dim, (1,))
        self.predecessors = list(predecessors) if predecessors is not None else []

    @property
    def shape(self) -> tuple:
        """
        Shape without batch dimension.
        """
        if isinstance(self.seq, list):
            self.seq = tuple(self.seq)
        if isinstance(self.dim, list):
            self.dim = tuple(self.dim)
        return self.seq + (self.time,) + self.dim

    @property
    def rank(self) -> int:
        """
        Rank without batch dimension.
        """
        return len(self.shape)

    def with_predecessors(self, predecessors):
        """
        Utility for symbolic graph construction.
        """
        self.predecessors = list(predecessors)
        return self

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
