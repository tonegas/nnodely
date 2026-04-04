"""
Output - nomina uno stream come output del modello.
"""
from nnodely.core.stream import Stream

class Output(Stream):
    """Nomina uno stream come output. È un nodo DAG con predecessors=[stream].
    Propaga seq, time, dim dallo stream per compatibilità con get_seq_time_dim.
    """
    def __init__(self, name: str, stream):
        self.stream = stream
        super().__init__(
            name=name,
            node_type="output",
            seq = stream.seq,
            time = stream.time,
            dim = stream.dim,
            predecessors=[stream]
        )
