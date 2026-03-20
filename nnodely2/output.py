"""
Output - nomina uno stream come output del modello.
"""

from nnodely.dag import to_tuple


class Output:
    """Nomina uno stream come output. È un nodo DAG con predecessors=[stream].
    Propaga seq, time, dim dallo stream per compatibilità con get_seq_time_dim.
    """

    def __init__(self, name: str, stream):
        # if isinstance(stream, dict) and len(stream) == 1:
        #     stream = next(iter(stream.values()))
        self.name = name
        self.stream = stream
        self.predecessors = [stream]

    @property
    def seq(self):
        return tuple(getattr(self.stream, 'seq', ()) or ())

    @property
    def time(self):
        return getattr(self.stream, 'time', None) or 1

    @property
    def dim(self):
        return to_tuple(getattr(self.stream, 'dim', (1,)), (1,))
