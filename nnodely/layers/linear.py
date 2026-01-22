import copy

import torch.nn as nn
import torch

from collections.abc import Callable

from nnodely.basic.relation import Stream, Relation
from nnodely.basic.model import Model
from nnodely.layers.parameter import Parameter
from nnodely.support.utils import check, enforce_types
from nnodely.support.jsonutils import merge

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

linear_relation_name = 'Linear'

class Linear(Relation):
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
                 dropout:int|float = 0,
                 name: str | None = None):

        name = name if name is not None else linear_relation_name
        attrs = {'W': None, 'b': None, 'dropout': dropout}

        if type(W) is Parameter:
            self.output_dimension = W.attrs['dim']
            if output_dimension is not None:
                check(self.output_dimension == output_dimension, ValueError, 'the output dimension must be equal to the dimension of "W".')
            self.W = W
        else:
            self.output_dimension= 1 if output_dimension is None else output_dimension
            Wname = W if type(W) is str else name + 'W'
            self.W = Parameter(name=Wname, dimensions=self.output_dimension, init=W_init, init_params=W_init_params)
        attrs['W'] = self.W.name

        if b is not None and b is not False:
            if type(b) is Parameter:
                check(b.attrs['dim'] == self.output_dimension, ValueError,'output_dimension must be equal to the dim of the "b".')
                self.b = b
            else:
                bname = b if type(b) is str else name + 'b'
                self.b = Parameter(name=bname, dimensions=self.output_dimension, init=b_init, init_params=b_init_params)
            attrs['b'] = self.b.name

        super().__init__(name, [attrs['W'], attrs['b']], **attrs)

    @enforce_types
    def __call__(self, obj:Stream):
        super().__call__(edges=obj.name)


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

def createLinear(*inputs):
    return Linear_Layer(weights=inputs[0], bias=inputs[1], dropout=inputs[2])

setattr(Model, linear_relation_name, createLinear)
