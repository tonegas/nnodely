import copy, inspect, textwrap
import numpy as np

from collections.abc import Callable

from nnodely.basic.relation import Relation, Stream
from nnodely.support.utils import check, enforce_types, NP_DTYPE


def is_numpy_float(var):
    return isinstance(var, (np.float16, np.float32, np.float64))

class Constant(Stream):
    """
    Represents a constant value in the neural network model.

    Parameters
    ----------
    name : str
        The name of the constant.
    values : list, float, int, or np.ndarray
        The values of the constant.
    tw : float or int, optional
        The time window for the constant. Default is None.
    sw : int, optional
        The sample window for the constant. Default is None.

    Attributes
    ----------
    name : str
        The name of the constant.
    dim : dict
        A dictionary containing the dimensions of the constant.
    json : dict
        A dictionary containing the configuration of the constant.

    Examples
    --------
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/parameter.ipynb
        :alt: Open in Colab

    Example - passing a custom scalar value -> g.dim = {'dim': 1}:

        >>> g = Constant('gravity',values=9.81)

    Example - passing a custom vector value -> n.dim = {'dim': 4}:

        >>> n = Constant('numbers', values=[1,2,3,4])

    Example - passing a custom vector value with single sample window -> n.dim = {'dim': 4, 'sw': 1}:

        >>> n = Constant('numbers', values=[[1,2,3,4]])

    Example - passing a custom vector value with double sample window -> n.dim = {'dim': 4, 'sw': 2}:

        >>> n = Constant('numbers', values=[[2,3,4],[1,2,3]])

    Example - passing a custom vector value with double sample window -> n.dim = {'dim': 4, 'sw': 2}.
    If the value of the sw is differnt from the dimension of shape[0] an error will be raised.

        >>> n = Constant('numbers', sw = 2, values=[[2,3,4],[1,2,3]])

    Example - passing a custom vector value with time window -> n.dim = {'dim': 4, 'tw': 4}.
    In this case the samplingtime must be 0.5 otherwise an error will be raised. If the Constant have a time dimension,
    the input must have a len(shape) == 2.

        >>> n = Constant('numbers', tw = 4, values=[[2,3,4],[1,2,3]])
    """
    @enforce_types
    def __init__(self, name:str,
                 values:list|float|int|np.ndarray, *,
                 tw:float|int|None = None,
                 sw:int|None = None):

        attrs = {}
        values = np.array(values, dtype=NP_DTYPE)
        attrs['values'] = values

        shape = values.shape
        values = values.tolist()
        if tw is not None:
            check(len(shape) >= 2, ValueError, "The dimension must be at least 2 if tw is set.")
            check(sw is None, ValueError, "If tw is set sw must be None")
            dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
            attrs['tw'] = tw
        elif sw is not None:
            check(len(shape) >= 2, ValueError, "The dimension must be at least 2 if sw is set.")
            attrs['sw'] = sw
            check(shape[0] == attrs['sw'],ValueError, f"The sw = {sw} is different from sw = {shape[0]} of the values.")
            dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
        else:
            dimensions = 1 if len(shape[0:]) == 0 else shape[0] if len(shape[0:]) == 1 else list(shape[0:])
        attrs['dim'] = dimensions
        super().__init__(name, **attrs)

class Parameter(Stream):
    """
    Represents a trainable parameter in the neural network model.

    Notes
    -----
    .. note::
        You can find some initialization functions for the 'init' and 'init_params' parameters inside the initializer module.

    Parameters
    ----------
    name : str
        The name of the parameter.
    dimensions : int, list, tuple, or None, optional
        The dimensions of the parameter. Default is None.
    tw : float or int, optional
        The time window for the parameter. Default is None.
    sw : int, optional
        The sample window for the parameter. Default is None.
    values : list, float, int, np.ndarray, or None, optional
        The values by which initialize the parameter. Default is None.
    init : Callable, optional
        A callable for initializing the parameter values. Default is None.
    init_params : dict, optional
        A dictionary of parameters for the initializer. Default is None.

    Attributes
    ----------
    name : str
        The name of the parameter.
    dim : dict
        A dictionary containing the dimensions of the parameter.
    json : dict
        A dictionary containing the configuration of the parameter.

    Examples
    --------
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/parameter.ipynb
        :alt: Open in Colab

    Example - basic usage:
        >>> k = Parameter('k', dimensions=3, tw=4)

    Example - initialize a parameter with values:
        >>> x = Input('x')
        >>> gravity = Parameter('g', dimensions=(4,1),values=[[[1],[2],[3],[4]]])
        >>> out = Output('out', Linear(W=gravity)(x.sw(3)))

    Example - initialize a parameter with a function:
        >>> x = Input('x').last()
        >>> p = Parameter('param', dimensions=1, sw=1, init=init_constant, init_params={'value':1})
        >>> relation = Fir(parameter=param)(x)
    """
    @enforce_types
    def __init__(self, name:str,
                 dimensions:int|list|tuple|None = None, *,
                 tw:float|int|None = None,
                 sw:int|None = None,
                 values:list|float|int|np.ndarray|None = None,
                 init:Callable|str|None = None,
                 init_params:dict|None = None):
        
        attrs = {}
        dimensions = list(dimensions) if type(dimensions) is tuple else dimensions
        if values is None:
            if dimensions is None:
                dimensions = 1
            attrs['dim'] = dimensions
            if tw is not None:
                attrs['tw'] = tw
            elif sw is not None:
                attrs['sw'] = sw
        else:
            values = np.array(values, dtype=NP_DTYPE)
            shape = values.shape
            values = values.tolist()
            attrs['values'] = values
            if tw is not None:
                check(len(shape) >= 2, ValueError, "The dimension must be at least 2.")
                dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
                attrs['tw'] = tw
            elif sw is not None:
                check(len(shape) >= 2, ValueError, "The dimension must be at least 2.")
                attrs['sw'] = sw
                check(shape[0] == attrs['sw'], ValueError, f"The sw = {sw} is different from sw = {shape[0]} of the values.")
                dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
            else:
                dimensions = 1 if len(shape[0:]) == 0 else shape[0] if len(shape[0:]) == 1 else list(shape[0:])
            attrs['dim'] = dimensions
        super().__init__(name, **attrs)
        # if init is not None:
        #     check('values' not in self.json['Parameters'][self.name], ValueError, f"The parameter {self.name} is already initialized.")
        #     if inspect.isfunction(init):
        #         code = textwrap.dedent(inspect.getsource(init)).replace('\"', '\'')
        #         self.json['Parameters'][self.name]['init_fun'] = { 'code' : code, 'name' : init.__name__}
        #     elif type(init) is str:
        #         self.json['Parameters'][self.name]['init_fun'] = { 'name' : init }
        #     if init_params is not None:
        #         self.json['Parameters'][self.name]['init_fun']['params'] = init_params
    
class SampleTime():
    """
    Represents a constant value that is equal to the sample time.

    Attributes
    ----------
    name : str
        The name of the constant.
    dim : dict
        A dictionary containing the dimensions of the constant.
    json : dict
        A dictionary containing the configuration of the constant.

    Example
    -------
        >>> dt = SampleTime()
    """
    def __init__(self):
        return Constant('SampleTime', values=0)
        
    # def __new__(cls):
    #     SampleTime.g.dim = {'dim': 1}
    #     SampleTime.g.json['Constants'][SampleTime.name] = copy.deepcopy(SampleTime.g.dim)
    #     SampleTime.g.json['Constants'][SampleTime.name]['values'] = SampleTime.name
    #     return SampleTime.g
