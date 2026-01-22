import torch.nn as nn
import torch

from nnodely.basic.relation import Stream, Relation
from nnodely.support.utils import enforce_types, check
from nnodely.support.jsonutils import merge, subjson_from_relation
from nnodely.basic.model import Model
from nnodely.support.fixstepsolver import Euler, Trapezoidal

SOLVERS = {
    'euler': Euler,
    'trapezoidal': Trapezoidal
}

# Binary operators
int_relation_name = 'Integrate'
der_relation_name = 'Derivate'

class Integrate(Relation):
    """
    This operation Integrate a Stream

    Parameters
    ----------
    method : is the integration method
    """
    @enforce_types
    def __init__(self, output:Stream, *,
                 int_name:str|None = None, der_name:str|None = None, method:str = 'euler') -> Stream:
        if int_name is None:
            int_name = output.name + "_int"
        if der_name is None:
            der_name = output.name + "_der"
        check(method in SOLVERS, ValueError, f"The method '{method}' is not supported yet")
        solver = SOLVERS[method](int_name,der_name)
        output_int = solver.integrate(output)
        super().__init__(output_int.name, output_int.json, output_int.dim)

class Derivate(Relation):
    """
    This operation Derivate a Stream with respect to time or another Stream

    Parameters
    ----------
    method : is the derivative method
    """
    @enforce_types
    def __init__(self, output:Stream, input:Stream = None, *,
                 int_name:str|None = None, der_name:str|None = None, method:str = 'euler') -> Stream:
        if input is None:
            if int_name is None:
                int_name = output.name + "_int"
            if der_name is None:
                der_name = output.name + "_der"
            check(method in SOLVERS, ValueError, f"The method '{method}' is not supported yet")
            solver = SOLVERS[method](int_name,der_name)
            output_der = solver.derivate(output)
            super().__init__(output_der.name, output_der.json, output_der.dim)
        else:
            super().__init__(der_relation_name + str(Stream.count), merge(output.json,input.json), input.dim)
            self.json['Relations'][self.name] = [der_relation_name, [output.name, input.name]]
            subjson = subjson_from_relation(self.json, input.name)
            grad_inputs = subjson['Inputs'].keys()
            for i in grad_inputs:
                self.json['Inputs'][i]['type'] = 'derivate'


class Derivate_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Derivate_Layer, self).__init__()

    def forward(self, *inputs):
        return torch.autograd.grad(inputs[0], inputs[1], grad_outputs=torch.ones_like(inputs[0]), create_graph=True, retain_graph=True, allow_unused=False)[0]

def createAdd(name, *inputs):
    #: :noindex:
    return Derivate_Layer()

setattr(Model, der_relation_name, createAdd)
