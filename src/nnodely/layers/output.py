"""
Output - nomina uno stream come output del modello.
"""
from nnodely.core.stream import Stream

<<<<<<< HEAD:nnodely/layers/output.py
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
=======
from nnodely.basic.relation import Stream, NeuObj
from nnodely.support.utils import enforce_types
from nnodely.support.jsonutils import stream_to_str

from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.INFO)


class Output(NeuObj):
    """
    Represents an output in the neural network model. This relation is what the network will give as output during inference.

    Parameters
    ----------
    name : str
        The name of the output.
    relation : Stream
        The relation to be used for the output.

    Attributes
    ----------
    name : str
        The name of the output.
    json : dict
        A dictionary containing the configuration of the output.
    dim : dict
        A dictionary containing the dimensions of the output.
    """

    @enforce_types
    def __init__(self, name: str, relation: Stream):
        """
        Initializes the Output object.

        Parameters
        ----------
        name : str
            The name of the output.
        relation : Stream
            The relation to be used for the output.
        """
        super().__init__(name, relation.json, relation.dim)
        log.debug(f"Output {name}")
        self.json["Outputs"][name] = {}
        self.json["Outputs"][name] = relation.name
        log.debug("\n" + pformat(self.json))

    def __str__(self):
        return stream_to_str(self, "Output")

    def __repr__(self):
        return self.__str__()
>>>>>>> develop:src/nnodely/layers/output.py
