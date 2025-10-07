import inspect, copy, textwrap, torch

import numpy as np
import torch.nn as nn

from collections.abc import Callable

from nnodely.basic.relation import NeuObj, Stream
from nnodely.basic.model import Model
from nnodely.support.utils import check, enforce_types
from nnodely.support.jsonutils import merge

fuzzify_relation_name = 'Fuzzify'

class Fuzzify(NeuObj):
    """
    Represents a Fuzzify relation in the neural network model.

    Parameters
    ----------
    output_dimension : int, optional
        The output dimension of the Fuzzify relation. If provided, `range` must also be provided and `centers` must be None.
    range : list, optional
        A list containing the start and end values for the range. Required if `output_dimension` is provided.
    centers : list, optional
        A list of center values for the fuzzy functions. Required if `output_dimension` is None.
        The `output_dimension` will be inferred from the number of centers provided.
    functions : str, list, or Callable, optional
        The fuzzy functions to use. Can be a string specifying a predefined function type, a custom function, or a list of callable functions. Default is 'Triangular'.
    
    Notes
    -----
    .. note::
        The predefined function types are 'Triangular' and 'Rectangular'.
        It is also possible to pass a list of custom functions. In this case, each center will be associated with the respective function in the list.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    output_dimension : dict
        The output dimension of the Fuzzify relation.
    json : dict
        A dictionary containing the configuration of the Fuzzify relation.

    Examples
    --------
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/fuzzify.ipynb
        :alt: Open in Colab
        
    Example - basic usage:
        >>> x = Input('x')
        >>> fuz = Fuzzify(output_dimension=5, range=[1,5])
        >>> out = Output('out', fuz(x.last()))

    Example - passing the centers:
        >>> fuz = Fuzzify(centers=[-1,0,3,5], functions='Rectangular')
        >>> out = Output('out', fuz(x.last()))

    Example - using a custom function:
        >>> def fun(x):
        >>>     import torch
        >>>     return torch.tanh(x)
        >>> fuz = Fuzzify(output_dimension=11, range=[-5,5], functions=fun)
        >>> out = Output('out', fuz(x.last()))
    """
    @enforce_types
    def __init__(self, output_dimension: int | None = None,
                 range: list | None = None, *,
                 centers: list | None = None,
                 functions: str | list | Callable = 'Triangular'):

        self.relation_name = fuzzify_relation_name
        super().__init__('F' + fuzzify_relation_name + str(NeuObj.count))
        self.json['Functions'][self.name] = {}
        if output_dimension is not None:
            check(range is not None, ValueError, 'if "output_dimension" is not None, "range" must be not setted')
            check(centers is None, ValueError,
                  'if "output_dimension" and "range" are not None, then "centers" must be None')
            self.output_dimension = {'dim': output_dimension}
            interval = ((range[1] - range[0]) / (output_dimension - 1))
            self.json['Functions'][self.name]['centers'] = np.arange(range[0], range[1] + interval, interval).tolist()
        else:
            check(centers is not None, ValueError, 'if "output_dimension" is None and "centers" must be setted')
            self.output_dimension = {'dim': len(centers)}
            self.json['Functions'][self.name]['centers'] = np.array(centers).tolist()
        self.json['Functions'][self.name]['dim_out'] = copy.deepcopy(self.output_dimension)

        if type(functions) is str:
            self.json['Functions'][self.name]['functions'] = functions
            self.json['Functions'][self.name]['names'] = functions
        elif type(functions) is list:
            self.json['Functions'][self.name]['functions'] = []
            self.json['Functions'][self.name]['names'] = []
            for func in functions:
                code = textwrap.dedent(inspect.getsource(func)).replace('\"', '\'')
                self.json['Functions'][self.name]['functions'].append(code)
                self.json['Functions'][self.name]['names'].append(func.__name__)
        else:
            code = textwrap.dedent(inspect.getsource(functions)).replace('\"', '\'')
            self.json['Functions'][self.name]['functions'] = code
            self.json['Functions'][self.name]['names'] = functions.__name__

    @enforce_types
    def __call__(self, obj: Stream) -> Stream:
        stream_name = fuzzify_relation_name + str(Stream.count)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Fuzzify operation.")
        check('dim' in obj.dim and obj.dim['dim'] == 1, ValueError, 'Input dimension must be scalar')
        output_dimension = copy.deepcopy(obj.dim)
        output_dimension.update(self.output_dimension)
        stream_json = merge(self.json, obj.json)
        stream_json['Relations'][stream_name] = [fuzzify_relation_name, [obj.name], self.name]
        return Stream(stream_name, stream_json, output_dimension)

def return_fuzzify(json, xlim=None, num_points=1000):
    if xlim is not None:
        x = torch.from_numpy(np.linspace(xlim[0], xlim[1], num=num_points))
    else:
        x = torch.from_numpy(np.linspace(json['centers'][0] - 2, json['centers'][-1] + 2, num=num_points))
    chan_centers = np.array(json['centers'])
    activ_fun = {}
    if isinstance(json['names'], list):
        n_func = len(json['names'])
    else:
        n_func = 1
    for i in range(len(chan_centers)):
        if json['functions'] == 'Triangular':
            activ_fun[i] = triangular(x, i, chan_centers).tolist()
        elif json['functions'] == 'Rectangular':
            activ_fun[i] = rectangular(x, i, chan_centers).tolist()
        else:
            if isinstance(json['names'], list):
                if i >= n_func:
                    func_idx = i - round(n_func * (i // n_func))
                else:
                    func_idx = i
                exec(json['functions'][func_idx], globals())
                function_to_call = globals()[json['names'][func_idx]]
            else:
                exec(json['functions'], globals())
                function_to_call = globals()[json['names']]
            activ_fun[i] = custom_function(function_to_call, x, i, chan_centers).tolist()
    return x.tolist(), activ_fun

def triangular(x, idx_channel, chan_centers):
    # Compute the number of channels
    num_channels = len(chan_centers)
    # First dimension of activation
    if idx_channel == 0:
        if num_channels != 1:
            ampl = chan_centers[1] - chan_centers[0]
            act_fcn = torch.minimum(torch.maximum(-(x - chan_centers[0]) / ampl + 1, torch.tensor(0.0)), torch.tensor(1.0))
        else:
            # In case the user only wants one channel
            act_fcn = 1
    elif idx_channel != 0 and idx_channel == (num_channels - 1):
        ampl = chan_centers[-1] - chan_centers[-2]
        act_fcn = torch.minimum(torch.maximum((x - chan_centers[-2]) / ampl, torch.tensor(0.0)), torch.tensor(1.0))
    else:
        ampl_1 = chan_centers[idx_channel] - chan_centers[idx_channel - 1]
        ampl_2 = chan_centers[idx_channel + 1] - chan_centers[idx_channel]
        act_fcn = torch.minimum(torch.maximum((x - chan_centers[idx_channel - 1]) / ampl_1, torch.tensor(0.0)), torch.maximum(-(x - chan_centers[idx_channel]) / ampl_2 + 1, torch.tensor(0.0)))
    return act_fcn

def rectangular(x, idx_channel, chan_centers):
    ## compute number of channels
    num_channels = len(chan_centers)
    ## First dimension of activation
    if idx_channel == 0:
        if num_channels != 1:
            width = abs(chan_centers[idx_channel + 1] - chan_centers[idx_channel]) / 2
            act_fcn = torch.where(x < (chan_centers[idx_channel] + width), 1.0, 0.0)
        else:
            # In case the user only wants one channel
            act_fcn = 1.0
    elif idx_channel != 0 and idx_channel == (num_channels - 1):
        width = abs(chan_centers[idx_channel] - chan_centers[idx_channel - 1]) / 2
        act_fcn = torch.where(x >= chan_centers[idx_channel] - width, 1.0, 0.0)
    else:
        width_forward = abs(chan_centers[idx_channel + 1] - chan_centers[idx_channel]) / 2
        width_backward = abs(chan_centers[idx_channel] - chan_centers[idx_channel - 1]) / 2
        act_fcn = torch.where((x >= chan_centers[idx_channel] - width_backward) & (x < chan_centers[idx_channel] + width_forward), 1.0, 0.0)
    return act_fcn


def custom_function(func, x, idx_channel, chan_centers):
    act_fcn = func(x - chan_centers[idx_channel])
    return act_fcn

class Fuzzify_Layer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.centers = params['centers']
        self.function = params['functions']
        self.dimension = params['dim_out']['dim']
        self.name = params['names']

        if type(self.name) is list:
            self.n_func = len(self.name)
            for func, name in zip(self.function, self.name):
                ## Add the function to the globals
                try:
                    code = 'import torch\n@torch.fx.wrap\n' + func
                    exec(code, globals())
                except Exception as e:
                    check(False, RuntimeError, f"An error occurred when running the function '{name}':\n {e}")
        else:
            self.n_func = 1
            if self.name not in ['Triangular', 'Rectangular']:  ## custom function
                ## Add the function to the globals
                try:
                    code = 'import torch\n@torch.fx.wrap\n' + self.function
                    exec(code, globals())
                except Exception as e:
                    check(False, RuntimeError, f"An error occurred when running the function '{self.name}':\n {e}")

    def forward(self, x):
        res = torch.zeros_like(x).repeat(1, 1, self.dimension)
        if self.function == 'Triangular':
            for i in range(len(self.centers)):
                slicing(res, torch.tensor(i), triangular(x, i, self.centers))
        elif self.function == 'Rectangular':
            for i in range(len(self.centers)):
                slicing(res, torch.tensor(i), rectangular(x, i, self.centers))
        else:  ## Custom_function
            if self.n_func == 1:
                # Retrieve the function object from the globals dictionary
                function_to_call = globals()[self.name]
                for i in range(len(self.centers)):
                    slicing(res, torch.tensor(i), custom_function(function_to_call, x, i, self.centers))
            else:  ## we have multiple functions
                for i in range(len(self.centers)):
                    if i >= self.n_func:
                        func_idx = i - round(self.n_func * (i // self.n_func))
                    else:
                        func_idx = i
                    function_to_call = globals()[self.name[func_idx]]
                    slicing(res, torch.tensor(i), custom_function(function_to_call, x, i, self.centers))
        return res

@torch.fx.wrap
def slicing(res, i, x):
    res[:, :, i:i + 1] = x

def createFuzzify(self, *params):
    return Fuzzify_Layer(params[0])

setattr(Model, fuzzify_relation_name, createFuzzify)
