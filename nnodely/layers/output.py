from pprint import pformat

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
    def __init__(self, name:str, relation:Stream):
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
        self.json['Outputs'][name] = {}
        self.json['Outputs'][name] = relation.name
        log.debug("\n"+pformat(self.json))

    def __str__(self):
        return stream_to_str(self, 'Output')

    def __repr__(self):
        return self.__str__()