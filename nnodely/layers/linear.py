import copy

import torch.nn as nn
import torch

from collections.abc import Callable

from nnodely.basic.relation import NeuObj, Stream, AutoToStream
from nnodely.basic.model import Model
from nnodely.layers.parameter import Parameter
from nnodely.support.utils import check, enforce_types
from nnodely.support.jsonutils import merge

from nnodely.support.logger import logging, nnLogger
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
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/linear.ipynb
        :alt: Open in Colab

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
    def __init__(self, output_dimension:int|None = None, *,
                 W_init:Callable|str|None = None,
                 W_init_params:dict|None = None,
                 b_init:Callable|str|None = None,
                 b_init_params:dict|None = None,
                 W:Parameter|str|None = None,
                 b:bool|str|Parameter|None = None,
                 dropout:int|float = 0):

        self.W = W
        self.b = b
        self.bname = None
        self.Wname = None
        self.dropout = dropout

        super().__init__('P' + linear_relation_name + str(NeuObj.count))

        if type(self.W) is Parameter:
            check('tw' not in self.W.dim.keys() and 'sw' not in self.W.dim.keys(), TypeError, f'The "W" must no have time dimension but was {W.dim}.')
            check(len(self.W.dim['dim']) == 2, ValueError,'The "W" dimensions must be a list of 2.')
            self.output_dimension = self.W.dim['dim'][1]
            if output_dimension is not None:
                check(self.W.dim['dim'][1] == output_dimension, ValueError, 'output_dimension must be equal to the second dim of "W".')
            self.Wname = self.W.name
            W_json = W.json
        else:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            self.Wname = W if type(W) is str else self.name + 'W'
            W_json = Parameter(name=self.Wname, dimensions=self.output_dimension, init=W_init, init_params=W_init_params).json
        self.json = merge(self.json,W_json)

        if self.b is not None and self.b is not False:
            if type(self.b) is Parameter:
                check('tw' not in self.b.dim and 'sw' not in self.b.dim, TypeError, f'The "bias" must no have a time dimensions but got {self.b.dim}.')
                check(type(self.b.dim['dim']) is int, TypeError, 'The "b" dimensions must be an integer.')
                check(self.b.dim['dim'] == self.output_dimension, ValueError,'output_dimension must be equal to the dim of the "b".')
                self.bname = self.b.name
                b_json = self.b.json
            else:
                self.bname = b if type(self.b) is str else self.name + 'b'
                b_json = Parameter(name=self.bname, dimensions=self.output_dimension, init=b_init, init_params=b_init_params).json
            self.json = merge(self.json,b_json)

        self.json_stream = {}

    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        stream_name = linear_relation_name + str(Stream.count)
        check(type(obj) is Stream, TypeError,f"The type of {obj} is {type(obj)} and is not supported for Linear operation.")
        window = 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)

        json_stream_name = obj.dim['dim']
        if obj.dim['dim'] not in self.json_stream:
            if len(self.json_stream) > 0:
                log.warning(f"The Linear {self.name} was called with inputs with different dimensions. If both Linear enter in the model an error will be raised.")
            self.json_stream[json_stream_name] = copy.deepcopy(self.json)

            self.json_stream[json_stream_name]['Parameters'][self.Wname]['dim'] = [obj.dim['dim'],self.output_dimension,]

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
        y = torch.einsum('bwi,io->bwo', x, self.weights)  # y will have shape [batch, window, output_features]
        if self.bias is not None:
            y += self.bias  
        # Add dropout if necessary
        if self.dropout is not None:
            y = self.dropout(y)
        return y

def createLinear(self, *inputs):
    return Linear_Layer(weights=inputs[0], bias=inputs[1], dropout=inputs[2])

setattr(Model, linear_relation_name, createLinear)
