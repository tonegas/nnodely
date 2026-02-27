import inspect, copy, textwrap, torch, math

import torch.nn as nn

from nnodely.basic.relation import NeuObj, Stream, toStream
from nnodely.basic.model import Model
from nnodely.layers.parametricfunction import ParamFun
from nnodely.support.utils import check, enforce_types
from nnodely.support.jsonutils import merge

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

ode_relation_name = 'NeuralODE'

class NeuralODE(NeuObj):
    """Neural ODE layer implementing continuous-depth models.

    Args:
        func (Callable): The function defining the ODE dynamics.
        rtol (float, optional): Relative tolerance for the ODE solver. Default is 1e-7.
        atol (float, optional): Absolute tolerance for the ODE solver. Default is 1e-9.
        method (str, optional): The ODE solver method to use. Default is 'dopri5'.
    """

    @enforce_types
    def __init__(self,
                 func: ParamFun,
                 dt: float,
                 rtol: float = 1e-7,
                 atol: float = 1e-9,
                 method: str = 'dopri5') -> Stream:
        super().__init__('F'+ode_relation_name + str(NeuObj.count))

        self.func = func
        self.dt = dt
        self.rtol = rtol
        self.atol = atol
        self.method = method

        code = textwrap.dedent(inspect.getsource(func.param_fun)).replace('\"', '\'')
        code = 'def ' + f'{self.name}' + '(state, *weights):\n' + \
                '    from nnodely.support.odeint.adjoint import odeint_adjoint as odeint\n    ' + \
                code.replace('\n', '\n    ') + \
                '\n' + \
                f'    ans = odeint(lambda t, y: {func.param_fun.__name__}(t, y, *weights), state, t=torch.tensor([0.0, {self.dt}]), rtol={self.rtol}, atol={self.atol}, method=\'{self.method}\', adjoint_params=list(weights))' + \
                f'\n    return ans[-1]\n'
        
        self.json['Functions'][self.name] = {
            'code' : code,
            'name' : f'{self.name}',
            'rtol' : rtol,
            'atol' : atol,
            'method' : method,
            'dt' : dt
        }

        self.json_stream = {}

    def __call__(self, *obj: Stream) -> Stream:
        stream_name = ode_relation_name + str(Stream.count)
        stream_json = copy.deepcopy(self.json)
        input_names = []
        for ind, o in enumerate(obj):
            o = toStream(o)
            check(type(o) is Stream, TypeError,
                  f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
            stream_json = merge(stream_json, o.json)
            input_names.append(o.name)

        stream_json['Relations'][stream_name] = [ode_relation_name, input_names, self.name]
        return Stream(stream_name, stream_json, obj[0].dim)


class ODE_Layer(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.name = func['name']
        self.dt = func['dt']
        self.rtol = func['rtol']
        self.atol = func['atol']
        self.method = func['method']
        ## Add the function to the globals
        try:
            code = 'import torch\n@torch.fx.wrap\n' + func['code']
            #print(f"Defining ODE function:\n{code}")
            exec(code, globals())
        except Exception as e:
            print(f"An error occurred: {e}")

    def forward(self, *inputs):
        # Retrieve the function object from the globals dictionary
        function_to_call = globals()[self.name]
        weights = list(inputs)[1:]
        return function_to_call(list(inputs)[0], *weights)  # Return the last state

def createODE(self, *func_params):
    # for key, value in func_params[0].items():
    #     print(f"{key}: {value}")
    
    return ODE_Layer(func_params[0])

setattr(Model, ode_relation_name, createODE)