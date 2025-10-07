import torch, inspect
import types

from collections import OrderedDict

import numpy as np
from functools import wraps
from typing import get_type_hints
import keyword

TORCH_DTYPE = torch.float32
NP_DTYPE = np.float32

ForbiddenTags = keyword.kwlist

class ReadOnlyDict:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        value = self._data[key]
        if isinstance(value, dict):
            return dict(ReadOnlyDict(value))
        return value

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def __repr__(self):
        from pprint import pformat
        return pformat(self._data)

    def __or__(self, other):
        if not isinstance(other, ReadOnlyDict):
            return NotImplemented
        combined_data = {**self._data, **other._data}
        return ReadOnlyDict(combined_data)

    def __str__(self):
        from nnodely.visualizer.emptyvisualizer import color, GREEN
        from pprint import pformat
        return color(pformat(self._data), GREEN)

    def __eq__(self, other):
        if not isinstance(other, ReadOnlyDict):
            return self._data == other
        return self._data == other._data

class ParamDict(ReadOnlyDict):
    def __init__(self, data, internal_data = None):
        super().__init__(data)
        self._internal_data = internal_data if internal_data is not None else {}

    def __setitem__(self, key, value):
        self._data[key]['values'] = value
        self._internal_data[key] = self._internal_data[key].new_tensor(value)

    def __getitem__(self, key):
        value = self._data[key]['values'] if 'values' in self._data[key] else None
        return value

def enforce_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        all_args = kwargs.copy()

        sig = OrderedDict(inspect.signature(func).parameters)
        if len(sig) != len(args):
            var_type = None
            for ind, arg in enumerate(args):
                if ind < len(list(sig.values())) and list(sig.values())[ind].kind == inspect.Parameter.VAR_POSITIONAL:
                    var_name = list(sig.keys())[ind]
                    var_type = sig.pop(var_name)
                if var_type:
                    sig[var_name+str(ind)] = var_type

        all_args.update(dict(zip(sig, args)))
        if 'self' in sig.keys():
            sig.pop('self')

        for arg_name, arg in all_args.items():
            if (arg_name in hints.keys() or arg_name in sig.keys()) and not isinstance(arg,sig[arg_name].annotation):
                class_name = func.__qualname__.split('.')[0]
                if isinstance(sig[arg_name].annotation, types.UnionType):
                    type_list = [val.__name__ for val in sig[arg_name].annotation.__args__]
                else:
                    type_list = sig[arg_name].annotation.__name__
                raise TypeError(
                    f"In Function or Class {class_name} the argument '{arg_name}' to be of type {type_list}, but got {type(arg).__name__}")

        # for arg, arg_type in hints.items():
        #     if arg in all_args and not isinstance(all_args[arg], arg_type):
        #         raise TypeError(
        #             f"In Function or Class {func} Expected argument '{arg}' to be of type {arg_type}, but got {type(all_args[arg]).__name__}")

        return func(*args, **kwargs)

    return wrapper

def is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True  # È un notebook
    except Exception:
        pass
    return False  # È uno script

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

def get_batch_size(n_samples, batch_size = None, predicion_samples = 0):
    batch_size = batch_size if batch_size is not None else n_samples
    predicion_samples = 0 if predicion_samples == -1 else predicion_samples #This value is used to disconnect the connect
    return batch_size if batch_size <= n_samples - predicion_samples else max(0, n_samples - predicion_samples)

def check_and_get_list(name_list, available_names, error_fun):
    if type(name_list) is str:
        name_list = [name_list]
    if type(name_list) is list:
        for name in name_list:
            check(name in available_names, IndexError,  error_fun(name))
    return name_list

def check(condition, exception, string):
    if not condition:
        raise exception(string)

# Function used to verified the number of gradient operations in the graph
# def count_gradient_operations(grad_fn):
#     count = 0
#     if grad_fn is None:
#         return count
#     nodes = [grad_fn]
#     while nodes:
#         node = nodes.pop()
#         count += 1
#         nodes.extend(next_fn[0] for next_fn in node.next_functions if next_fn[0] is not None)
#     return count

# def check_gradient_operations(X:dict):
#     count = 0
#     for key in X.keys():
#         count += count_gradient_operations(X[key].grad_fn)
#     return count