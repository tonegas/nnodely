"""
Stream - nodo del DAG con predecessors.
"""

from nnodely.core.dag import next_name


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
        name: str | None = None,
        # node_type: str = "Stream",
        seq: int | tuple[int] | None = None,
        time: int = 1,
        dim: int | tuple = (1,),
        predecessors=None,
    ):
        self.name = (
            str(name) if name is not None else next_name(self.__class__.__name__)
        )
        # self.node_type = str(node_type)
        self.seq = () if seq is None else (seq,) if isinstance(seq, int) else seq
        self.time = time
        self.dim = (dim,) if isinstance(dim, int) else dim
        self.predecessors = list(predecessors) if predecessors is not None else []

    @property
    def shape(self) -> tuple:
        """
        Shape without batch dimension.
        """
        return self.seq + (self.time,) + self.dim

    @property
    def dimensions(self):
        """
        Return seq, time, dim separately
        """
        return self.seq, self.time, self.dim

    @property
    def rank(self) -> int:
        """
        Rank without batch dimension.
        """
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
            f"type={self.__class__.__name__!r}, "
            f"shape={self.shape}, "
            f"predecessors={pred_names}"
            f")"
        )
