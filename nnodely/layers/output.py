"""
Output - nomina uno stream come output del modello.
"""
from nnodely.core.stream import Stream

class Output(Stream):
    """Nomina uno stream come output. È un nodo DAG con predecessors=[stream].
    Propaga seq, time, dim dallo stream per compatibilità con Keras.
    """
    def __init__(self, name: str, stream):
        self.stream = stream
        super().__init__(
            name=name,
            node_type="Output",
            seq = stream.seq,
            time = stream.time,
            dim = stream.dim,
            predecessors=[stream]
        )
