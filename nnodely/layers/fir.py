import copy, torch

import torch.nn as nn

from collections.abc import Callable

from nnodely.basic.relation import NeuObj, Stream, AutoToStream
from nnodely.basic.model import Model
from nnodely.layers.parameter import Parameter
from nnodely.support.utils import check, enforce_types, TORCH_DTYPE
from nnodely.support.jsonutils import merge

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

fir_relation_name = 'Fir'

class Fir(NeuObj, AutoToStream):
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
                 dropout:int|float = 0):

        self.W = W
        self.b = b
        self.Wname = None
        self.bname = None
        self.dropout = dropout

        super().__init__('P'+fir_relation_name + str(NeuObj.count))

        if type(self.W) is Parameter:
            check(len(self.W.dim) == 2,ValueError,f"The values of the parameters must have two dimensions (tw/sample_rate or sw,output_dimension).")
            if output_dimension is None:
                check(type(self.W.dim['dim']) is int, TypeError, 'Dimension of the parameter must be an integer for the Fir')
                self.output_dimension = self.W.dim['dim']
            else:
                self.output_dimension = output_dimension
                check(self.W.dim['dim'] == self.output_dimension,
                      ValueError,
                      'output_dimension must be equal to dim of the Parameter')
            self.Wname = self.W.name
            W_json = self.W.json
        else:  ## Create a new default parameter
            self.output_dimension = 1 if output_dimension is None else output_dimension
            self.Wname = W if type(W) is str else self.name + 'W'
            W_json = Parameter(name=self.Wname, dimensions=self.output_dimension, init=W_init, init_params=W_init_params).json
        self.json = merge(self.json,W_json)

        if self.b is not None and self.b is not False:
            if type(self.b) is Parameter:
                check('tw' not in self.b.dim and 'sw' not in self.b.dim, TypeError, f'The "bias" must no have a time dimensions but got {self.b.dim}.')
                check(type(self.b.dim['dim']) is int, ValueError, 'The "bias" dimensions must be an integer.')
                check(self.b.dim['dim'] == self.output_dimension, ValueError, 'output_dimension must be equal to the dim of the "bias".')
                self.bname = self.b.name
                b_json = self.b.json
            else:
                self.bname = b if type(self.b) is str else self.name + 'b'
                b_json = Parameter(name=self.bname, dimensions=self.output_dimension, init=b_init, init_params=b_init_params).json
            self.json = merge(self.json,b_json)
        self.json_stream = {}

    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        stream_name = fir_relation_name + str(Stream.count)
        check('dim' in obj.dim and obj.dim['dim'] == 1,
              ValueError,
              f"Input dimension is {obj.dim['dim']} and not scalar")
        window = 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)

        json_stream_name = window + str(obj.dim[window])
        if json_stream_name not in self.json_stream:
            if len(self.json_stream) > 0:
                log.warning(f"The Fir {self.name} was called with inputs with different dimensions. If both Fir enter in the model an error will be raised.")
            self.json_stream[json_stream_name] = copy.deepcopy(self.json)
        self.json_stream[json_stream_name]['Parameters'][self.Wname][window] = obj.dim[window]

        if window:
            if type(self.W) is Parameter:
                check(window in self.json['Parameters'][self.Wname],
                      TypeError,
                      f"The window \'{window}\' of the input is not in the W")
                check(self.json['Parameters'][self.Wname][window] == obj.dim[window],
                      ValueError,
                      f"The window \'{window}\' of the input must be the same of the W")
        else:
            if type(self.W) is Parameter:
                cond = 'sw' not in self.json_stream[json_stream_name]['Parameters'][self.Wname] and 'tw' not in \
                       self.json_stream[json_stream_name]['Parameters'][self.Wname]
                check(cond, KeyError, 'The W have a time window and the input no')

        stream_json = merge(self.json_stream[json_stream_name],obj.json)
        stream_json['Relations'][stream_name] = [fir_relation_name, [obj.name], self.Wname, self.bname, self.dropout]
        return Stream(stream_name, stream_json,{'dim':self.output_dimension, 'sw': 1})


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
