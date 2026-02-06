from nnodely.basic.relation import Relation
from nnodely.support.utils import enforce_types

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

minimize_relation_name = 'Minimize'

class Minimize(Relation):
    """
    Minimization loss function inside the neural network model.
    """
    @enforce_types
    def __init__(self, input:Relation, target:Relation, loss: str = 'mse', name : str | None = None) -> Relation:
        name = name if name is not None else minimize_relation_name
        attrs = {'loss': loss, 'input': input.name, 'target': target.name}
        super().__init__(name=name, edges=[input.name, target.name], **attrs)