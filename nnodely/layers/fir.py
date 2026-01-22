import copy, torch

import torch.nn as nn

from collections.abc import Callable

from nnodely.basic.relation import Stream, Relation
from nnodely.basic.model import Model
from nnodely.layers.parameter import Parameter
from nnodely.support.utils import check, enforce_types, TORCH_DTYPE
from nnodely.support.jsonutils import merge

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

fir_relation_name = 'Fir'

class Fir(Relation):
    """
    Represents a Finite Impulse Response (FIR) relation in the neural network model.

    Notes
    -----
    .. note::
        The FIR relation works along the time dimension (second dimension) of the input tensor.
        You can find some initialization functions inside the initializer module.

    Parameters
    ----------
    output_dimension : int, optional
        The output dimension of the FIR relation.
    W_init : Callable, str, optional
        A callable for initializing the parameters.
    W_init_params : dict, optional
        A dictionary of parameters for the parameter initializer.
    b_init : Callable, str, optional
        A callable for initializing the bias.
    b_init_params : dict, optional
        A dictionary of parameters for the bias initializer.
    W : Parameter or str, optional
        The parameter object or tag. The parameter can be defined using the relative class 'Parameter'.
        If not given a new parameter will be auto-generated.
    b : bool, str, or Parameter, optional
        The bias parameter object, tag, or a boolean indicating whether to use bias.
        If set to 'True' a new parameter will be auto-generated.
    dropout : int or float, optional
        The dropout rate. Default is 0.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    W_init : Callable
        The parameter initializer.
    W_init_params : dict
        The parameters for the parameter initializer.
    W : Parameter or str
        The parameter object or name.
    b_init : Callable
        The bias initializer.
    b_init_params : dict
        The parameters for the bias initializer.
    b : bool, str, or Parameter
        The bias object, name, or a boolean indicating whether to use bias.
    pname : str
        The name of the parameter.
    bname : str
        The name of the bias.
    dropout : int or float
        The dropout rate.
    output_dimension : int
        The output dimension of the FIR relation.

    Examples 
    --------
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/fir.ipynb
        :alt: Open in Colab

    Example - basic usage:
        >>> input = Input('in')
        >>> relation = Fir(input.tw(0.05))

    Example - passing a parameter:
        >>> input = Input('in')
        >>> par = Parameter('par', dimensions=3, sw=2, init='init_constant')
        >>> relation = Fir(W=par)(input.sw(2))

    Example - parameters initialization:
        >>> x = Input('x')
        >>> F = Input('F')
        >>> fir_x = Fir(W_init='init_negexp')(x.tw(0.2))
        >>> fir_F = Fir(W_init='init_constant', W_init_params={'value':1})(F.last())

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

        name = name if name is not None else fir_relation_name
        attrs = {'W': None, 'b': None, 'dropout': dropout}

        if type(W) is Parameter:
            self.output_dimension = W.attrs['dim'] if output_dimension is None else output_dimension
            check(W.attrs['dim'] == self.output_dimension ,ValueError, 'the output dimension must be equal to the dimension of "W".')
            self.W = W
        else:  ## Create a new default parameter
            self.output_dimension = 1 if output_dimension is None else output_dimension
            Wname = W if type(W) is str else self.name + 'W'
            self.W = Parameter(name=Wname, dimensions=self.output_dimension, init=W_init, init_params=W_init_params)
        attrs['W'] = self.W.name

        if b is not None and b is not False:
            if type(b) is Parameter:
                check(b.attrs['dim'] == self.output_dimension, ValueError, 'the output dimension must be equal to the dimension of the "bias".')
                self.b = b
            else:
                bname = b if type(b) is str else self.name + 'b'
                self.b = Parameter(name=bname, dimensions=self.output_dimension, init=b_init, init_params=b_init_params)
            attrs['b'] = self.b.name
        super().__init__(name, [attrs['W'], attrs['b']], **attrs)

    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        super().__call__(edges=obj.name)


class Fir_Layer(nn.Module):
    def __init__(self, weights, bias=None, dropout=0):
        super(Fir_Layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        # x is expected to be of shape [batch, window, 1]
        batch_size = x.size(0)
        output_features = self.weights.size(1)
        # Remove the last dimension (1) to make x shape [batch, window]
        x = x.squeeze(-1)
        # Perform the linear transformation: y = xW^T
        x = torch.matmul(x, self.weights).to(dtype=TORCH_DTYPE)
        # Reshape y to be [batch, 1, output_features]
        x = x.view(batch_size, 1, output_features)
        # Add bias if necessary
        if self.bias is not None:
            x += self.bias  # Add bias
        # Add dropout if necessary
        if self.dropout is not None:
            x = self.dropout(x)
        return x

def createFir(self, *inputs):
    return Fir_Layer(weights=inputs[0], bias=inputs[1], dropout=inputs[2])

setattr(Model, fir_relation_name, createFir)
