import copy, inspect, textwrap, torch

import torch.nn as nn

from collections.abc import Callable

from nnodely.relation import NeuObj, Stream, AutoToStream
from nnodely.model import Model
from nnodely.parameter import Parameter
from nnodely.utils import check, merge, enforce_types

from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

linear_relation_name = 'Linear'

class Linear(NeuObj, AutoToStream):
    """
    Represents a Linear relation in the neural network model.

    Notes
    -----
    .. note::
        The Linear relation works along the input dimension (third dimension) of the input tensor.
        You can find some initialization functions inside the initializer module.

    Parameters
    ----------
    output_dimension : int, optional
        The output dimension of the Linear relation.
    W_init : Callable, optional
        A callable for initializing the weights.
    W_init_params : dict, optional
        A dictionary of parameters for the weight initializer.
    b_init : Callable, optional
        A callable for initializing the bias.
    b_init_params : dict, optional
        A dictionary of parameters for the bias initializer.
    W : Parameter or str, optional
        The weight parameter object or name. If not given a new parameter will be auto-generated.
    b : bool, str, or Parameter, optional
        The bias parameter object, name, or a boolean indicating whether to use bias. If set to 'True' a new parameter will be auto-generated.
    dropout : int or float, optional
        The dropout rate. Default is 0.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    W_init : Callable
        The weight initializer.
    W_init_params : dict
        The parameters for the weight initializer.
    b_init : Callable
        The bias initializer.
    b_init_params : dict
        The parameters for the bias initializer.
    W : Parameter or str
        The weight parameter object or name.
    b : bool, str, or Parameter
        The bias parameter object, name, or a boolean indicating whether to use bias.
    Wname : str
        The name of the weight parameter.
    bname : str
        The name of the bias parameter.
    dropout : int or float
        The dropout rate.
    output_dimension : int
        The output dimension of the Linear relation.

    Examples
    --------

    Example - basic usage:
        >>> input = Input('in').tw(0.05)
        >>> relation = Linear(input)

    Example - passing a weight and bias parameter:
        >>> input = Input('in').last()
        >>> weight = Parameter('W', values=[[[1]]])
        >>> bias = Parameter('b', values=[[1]])
        >>> relation = Linear(W=weight, b=bias)(input)

    Example - parameters initialization:
        >>> input = Input('in').last()
        >>> relation = Linear(b=True, W_init=init_negexp, b_init=init_constant, b_init_params={'value':1})(input)
    """

    @enforce_types
    def __init__(self, output_dimension:int|None = None,
                 W_init:Callable|None = None,
                 W_init_params:dict|None = None,
                 b_init:Callable|None = None,
                 b_init_params:dict|None = None,
                 W:Parameter|str|None = None,
                 b:bool|str|Parameter|None = None,
                 dropout:int|float = 0):

        self.relation_name = linear_relation_name
        self.W_init = W_init
        self.W_init_params = W_init_params
        self.b_init = b_init
        self.b_init_params = b_init_params
        self.W = W
        self.b = b
        self.bname = None
        self.Wname = None
        self.dropout = dropout
        super().__init__('P' + linear_relation_name + str(NeuObj.count))

        if W is None:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            self.Wname = self.name + 'W'
            self.json['Parameters'][self.Wname] = {}
        elif type(W) is str:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            self.Wname = W
            self.json['Parameters'][self.Wname] = {}
        else:
            check(type(W) is Parameter or type(W) is str, TypeError, 'The "W" must be of type Parameter or str.')
            check('sw' in W.dim and W.dim['sw'] == 1, ValueError, f'The "W" must have window dimension sw=1 but was {W.dim}.')
            check(len(W.dim['dim']) == 2, ValueError,'The "W" dimensions must be a list of 2.')
            self.output_dimension = W.dim['dim'][1]
            if output_dimension is not None:
                check(W.dim['dim'][1] == output_dimension, ValueError, 'output_dimension must be equal to the second dim of "W".')
            self.Wname = W.name
            self.json['Parameters'][W.name] = copy.deepcopy(W.json['Parameters'][W.name])

        if b is not None:
            check(type(b) is Parameter or type(b) is bool or type(b) is str, TypeError, 'The "b" must be of type Parameter, bool or str.')
            if type(b) is Parameter:
                check(type(b.dim['dim']) is int, ValueError, 'The "b" dimensions must be an integer.')
                if output_dimension is not None:
                    check(b.dim['dim'] == output_dimension, ValueError,
                          'output_dimension must be equal to the dim of the "b".')
                self.bname = b.name
                self.json['Parameters'][b.name] = copy.deepcopy(b.json['Parameters'][b.name])
            elif type(b) is str:
                self.bname = b
                self.json['Parameters'][self.bname] = { 'dim': self.output_dimension, 'sw': 1}
            else:
                self.bname = self.name + 'b'
                self.json['Parameters'][self.bname] = { 'dim': self.output_dimension, 'sw': 1}

        if self.W_init is not None:
            check('values' not in self.json['Parameters'][self.Wname], ValueError, f"The parameter {self.Wname} is already initialized.")
            check(inspect.isfunction(self.W_init), ValueError,
                  f"The W_init parameter must be a function.")
            code = textwrap.dedent(inspect.getsource(self.W_init)).replace('\"', '\'')
            self.json['Parameters'][self.Wname]['init_fun'] = { 'code' : code, 'name' : self.W_init.__name__}
            if self.W_init_params is not None:
                self.json['Parameters'][self.Wname]['init_fun']['params'] = self.W_init_params

        if self.b_init is not None:
            check(self.bname is not None, ValueError,f"The bias is missing.")
            check('values' not in self.json['Parameters'][self.bname], ValueError, f"The parameter {self.bname} is already initialized.")
            check(inspect.isfunction(self.b_init), ValueError,
                  f"The b_init parameter must be a function.")
            code = textwrap.dedent(inspect.getsource(self.b_init)).replace('\"', '\'')
            self.json['Parameters'][self.bname]['init_fun'] = { 'code' : code, 'name' : self.b_init.__name__ }
            if self.b_init_params is not None:
                self.json['Parameters'][self.bname]['init_fun']['params'] = self.b_init_params

        self.json_stream = {}

    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        stream_name = linear_relation_name + str(Stream.count)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Linear operation.")
        window = 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)
        assert(window is not None), f"Parameters {obj.name} with no window dimension"

        json_stream_name = obj.dim['dim']
        if obj.dim['dim'] not in self.json_stream:
            if len(self.json_stream) > 0:
                log.warning(
                    f"The Linear {self.name} was called with inputs with different dimensions. If both Linear enter in the model an error will be raised.")
            self.json_stream[json_stream_name] = copy.deepcopy(self.json)

            if  type(self.W) is not Parameter:
                self.json_stream[json_stream_name]['Parameters'][self.Wname]['dim'] = [obj.dim['dim'],self.output_dimension,]
                self.json_stream[json_stream_name]['Parameters'][self.Wname]['sw'] = 1

        if type(self.W) is Parameter:
            check(self.json['Parameters'][self.Wname]['dim'][0] == obj.dim['dim'], ValueError,
                  'the input dimension must be equal to the first dim of the parameter')

        stream_json = merge(self.json_stream[json_stream_name],obj.json)
        stream_json['Relations'][stream_name] = [linear_relation_name, [obj.name], self.Wname, self.bname, self.dropout]
        return Stream(stream_name, stream_json,{'dim': self.output_dimension, window:obj.dim[window]})


class Linear_Layer(nn.Module):
    def __init__(self, weights, bias=None, dropout=0):
        super(Linear_Layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        # x is expected to be of shape [batch, window, input_dimension]
        # Using torch.einsum for batch matrix multiplication
        y = torch.einsum('bwi,io->bwo', x, self.weights[0])  # y will have shape [batch, window, output_features]
        if self.bias is not None:
            y += self.bias  # Add bias
        # Add dropout if necessary
        if self.dropout is not None:
            y = self.dropout(y)
        return y

def createLinear(self, *inputs):
    return Linear_Layer(weights=inputs[0], bias=inputs[1], dropout=inputs[2])

setattr(Model, linear_relation_name, createLinear)
