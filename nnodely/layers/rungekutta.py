import torch.nn as nn
import torch

from nnodely.layers.parametricfunction import ParamFun
from nnodely.layers.parameter import SampleTime

from nnodely.basic.relation import Stream, NeuObj, ToStream
from nnodely.support.utils import enforce_types, check
from nnodely.support.jsonutils import merge, subjson_from_relation
from nnodely.basic.model import Model
import textwrap, inspect
from collections.abc import Callable

fe_relation_name = 'ForwardEuler'
rk2_relation_name = 'RK2'
rk4_relation_name = 'RK4'

# class ForwardEuler(NeuObj):
#     """
#     This operation perform Forward Euler Integration on a Stream
#     """
#     @enforce_types
#     def __init__(self, f:Callable) -> Stream:
#         super().__init__('F' + fe_relation_name + str(NeuObj.count))
#         self.f = f

#         code = textwrap.dedent(inspect.getsource(f)).replace('\"', '\'')
#         self.json['Functions'][self.name] = {
#             'code' : code,
#             'name' : f.__name__,
#         }

#     @enforce_types
#     def __call__(self, obj:Stream) -> Stream:
#         stream_name = fe_relation_name + str(Stream.count)
#         #funinfo = inspect.getfullargspec(self.f)
#         window = 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)
#         stream_json = merge(self.json, obj.json)
#         stream_json['Relations'][stream_name] = [fe_relation_name, [obj.name], self.name]
#         return Stream(stream_name, stream_json, {'dim': obj.dim['dim'], window: obj.dim[window]})
class ForwardEuler(NeuObj):
    """
    This operation perform Forward Euler Integration on a Stream
    """
    @enforce_types
    def __init__(self, f:Callable|ParamFun) -> Stream:
        super().__init__('F' + fe_relation_name + str(NeuObj.count))
        self.f = f if isinstance(f, ParamFun) else ParamFun(f)
        self.dt = SampleTime()
    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        return obj + self.dt * self.f(obj)

class RK2(NeuObj):
    """
    This operation perform RK2 Integration on a Stream
    """
    @enforce_types
    def __init__(self, f:Callable|ParamFun) -> Stream:
        super().__init__(rk2_relation_name + str(NeuObj.count))
        self.f = f if isinstance(f, ParamFun) else ParamFun(f)
        #self.fe = ForwardEuler(self.f)
        self.dt = SampleTime()

    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        f1 = self.f(obj)
        f2 = self.f(obj + (self.dt/2) * f1)
        return obj + self.dt * f2

class RK4(NeuObj):
    """
    This operation perform RK4 Integration on a Stream
    """
    @enforce_types
    def __init__(self, f:Callable|ParamFun) -> Stream:
        super().__init__(rk4_relation_name + str(NeuObj.count))
        self.f = f if isinstance(f, ParamFun) else ParamFun(f)
        self.dt = SampleTime()

    @enforce_types
    def __call__(self, obj:Stream, t:Stream|None = None) -> Stream:
        if t: ## Partial differential equation
            f1 = self.f(obj, t)
            f2 = self.f(obj + (self.dt/2) * f1, t + (self.dt/2))
            f3 = self.f(obj + (self.dt/2) * f2, t + (self.dt/2))
            f4 = self.f(obj + self.dt * f3, t + self.dt)
        else: ## Ordinary differential equation
            f1 = self.f(obj)
            f2 = self.f(obj + (self.dt/2) * f1)
            f3 = self.f(obj + (self.dt/2) * f2)
            f4 = self.f(obj + self.dt * f3)
        return obj + (self.dt/6) * (f1 + 2*f2 + 2*f3 + f4)

# class ForwardEuler_Layer(nn.Module):
#     #: :noindex:
#     def __init__(self, f):
#         super(ForwardEuler_Layer, self).__init__()
#         self.dt = SampleTime()
#         self.name = f['name']
#         try:
#             code = 'import torch\n@torch.fx.wrap\n' + f['code']
#             exec(code, globals())
#         except Exception as e:
#             print(f"An error occurred: {e}")

#     def forward(self, x):
#         func = globals()[self.name]
#         print(f'ForwardEuler executing function name {self.name} which is {func}')
#         return x + self.dt * func(x)
    
# def createForwardEuler(name, *inputs):
#     #: :noindex:
#     return ForwardEuler_Layer(inputs[0])

# class RK2_Layer(nn.Module):
#     #: :noindex:
#     def __init__(self):
#         super(RK2_Layer, self).__init__()

#     def forward(self, *inputs):
#         return None  # Placeholder for RK2 implementation

# def createRK2(name, *inputs):
#     #: :noindex:
#     return RK2_Layer()


# class RK4_Layer(nn.Module):
#     #: :noindex:
#     def __init__(self):
#         super(RK4_Layer, self).__init__()

#     def forward(self, *inputs):
#         return None  # Placeholder for RK4 implementation

# def createRK4(name, *inputs):
#     #: :noindex:
#     return RK4_Layer()

# #setattr(Model, fe_relation_name, createForwardEuler)
# setattr(Model, rk2_relation_name, createRK2)
# setattr(Model, rk4_relation_name, createRK4)
