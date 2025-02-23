import copy, torch, inspect
import numpy as np

from pprint import pformat
from functools import wraps
from typing import get_type_hints

from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

TORCH_DTYPE = torch.float32
NP_DTYPE = np.float32

def enforce_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        all_args = kwargs.copy()
        all_args.update(dict(zip(inspect.signature(func).parameters, args)))

        for arg, arg_type in hints.items():
            if arg in all_args and not isinstance(all_args[arg], arg_type):
                raise TypeError(
                    f"Expected argument '{arg}' to be of type {arg_type}, but got {type(all_args[arg]).__name__}")

        return func(*args, **kwargs)

    return wrapper


# Linear interpolation function, operating on batches of input data and returning batches of output data
def linear_interp(x,x_data,y_data):
    # Inputs: 
    # x: query point, a tensor of shape torch.Size([N, 1, 1])
    # x_data: map of x values, sorted in ascending order, a tensor of shape torch.Size([Q, 1])
    # y_data: map of y values, a tensor of shape torch.Size([Q, 1])
    # Output:
    # y: interpolated value at x, a tensor of shape torch.Size([N, 1, 1])

    # Saturate x to the range of x_data
    x = torch.min(torch.max(x,x_data[0]),x_data[-1])

    # Find the index of the closest value in x_data
    idx = torch.argmin(torch.abs(x_data[:-1] - x),dim=1)
    
    # Linear interpolation
    y = y_data[idx] + (y_data[idx+1] - y_data[idx])/(x_data[idx+1] - x_data[idx])*(x - x_data[idx])
    return y

def tensor_to_list(data):
    if isinstance(data, torch.Tensor):
        # Converte il tensore in una lista
        return data.tolist()
    elif isinstance(data, dict):
        # Ricorsione per i dizionari
        return {key: tensor_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Ricorsione per le liste
        return [tensor_to_list(item) for item in data]
    elif isinstance(data, tuple):
        # Ricorsione per tuple
        return tuple(tensor_to_list(item) for item in data)
    elif isinstance(data, torch.nn.modules.container.ParameterDict):
        # Ricorsione per parameter dict
        return {key: tensor_to_list(value) for key, value in data.items()}
    else:
        # Altri tipi di dati rimangono invariati
        return data

# Codice per comprimere le relazioni
        #print(self.json['Relations'])
        # used_rel = {string for values in self.json['Relations'].values() for string in values[1]}
        # if obj1.name not in used_rel and obj1.name in self.json['Relations'].keys() and self.json['Relations'][obj1.name][0] == add_relation_name:
        #     self.json['Relations'][self.name] = [add_relation_name, self.json['Relations'][obj1.name][1]+[obj2.name]]
        #     del self.json['Relations'][obj1.name]
        # else:
        # Devo aggiungere un operazione che rimuove un operazione di Add,Sub,Mul,Div se può essere unita ad un'altra operazione dello stesso tipo
        #
def merge(source, destination, main = True):
    if main:
        for key, value in destination["Functions"].items():
            if key in source["Functions"].keys() and 'n_input' in value.keys() and 'n_input' in source["Functions"][key].keys():
                check(value == {} or source["Functions"][key] == {} or value['n_input'] == source["Functions"][key]['n_input'],
                      TypeError,
                      f"The ParamFun {key} is present multiple times, with different number of inputs. "
                      f"The ParamFun {key} is called with {value['n_input']} parameters and with {source['Functions'][key]['n_input']} parameters.")
        log.debug("Merge Source")
        log.debug("\n"+pformat(source))
        log.debug("Merge Destination")
        log.debug("\n"+pformat(destination))
        result = copy.deepcopy(destination)
    else:
        result = destination
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = result.setdefault(key, {})
            merge(value, node, False)
        else:
            if key in result and type(result[key]) is list:
                if key == 'tw' or key == 'sw':
                    if result[key][0] > value[0]:
                        result[key][0] = value[0]
                    if result[key][1] < value[1]:
                        result[key][1] = value[1]
            else:
                result[key] = value
    if main == True:
        log.debug("Merge Result")
        log.debug("\n" + pformat(result))
    return result

def check(condition, exception, string):
    if not condition:
        raise exception(string)

def argmax_max(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])

def argmin_min(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])

def argmax_dict(iterable: dict):
    return max(iterable.items(), key=lambda x: x[1])

def argmin_dict(iterable: dict):
    return min(iterable.items(), key=lambda x: x[1])