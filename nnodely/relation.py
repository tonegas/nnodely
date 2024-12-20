import copy

import numpy as np

from nnodely.utils import check, merge

from nnodely.logger import logging, nnLogger
log = nnLogger(__name__, logging.CRITICAL)

MAIN_JSON = {
                'Info' : {},
                'Inputs' : {},
                'States' : {},
                'Constants': {},
                'Parameters' : {},
                'Functions' : {},
                'Relations': {},
                'Outputs': {}
            }

CHECK_NAMES = True
NeuObj_names = []

def toStream(obj):
    from nnodely.parameter import Parameter, Constant
    if type(obj) in (int,float,list,np.ndarray):
        obj = Constant('Constant'+str(NeuObj.count), obj)
        #obj = Stream(obj, MAIN_JSON, {'dim': 1}) if type(obj) in (int, float) else obj
    if type(obj) is Parameter or type(obj) is Constant:
        obj = Stream(obj.name, obj.json, obj.dim)
    return obj


class NeuObj():
    count = 0
    @classmethod
    def reset_count(self):
        NeuObj.count = 0
    def __init__(self, name='', json={}, dim=0):
        NeuObj.count += 1
        if CHECK_NAMES == True:
            check(name not in NeuObj_names, NameError, f"The name {name} is already used change the name of NeuObj.")
            NeuObj_names.append(name)
        self.name = name
        self.dim = dim
        if json:
            self.json = copy.deepcopy(json)
        else:
            self.json = copy.deepcopy(MAIN_JSON)

class Relation():
    def __add__(self, obj):
        from nnodely.arithmetic import Add
        return Add(self, obj)

    def __sub__(self, obj):
        from nnodely.arithmetic import Sub
        return Sub(self, obj)

    def __truediv__(self, obj):
        from nnodely.arithmetic import Div
        return Div(self, obj)

    def __mul__(self, obj):
        from nnodely.arithmetic import Mul
        return Mul(self, obj)

    def __pow__(self, obj):
        from nnodely.arithmetic import Pow
        return Pow(self, obj)

    def __neg__(self):
        from nnodely.arithmetic import Neg
        return Neg(self)

class Stream(Relation):
    count = 0
    @classmethod
    def reset_count(self):
        Stream.count = 0

    def __init__(self, name, json, dim, count = 1):
        Stream.count += count
        self.name = name
        self.json = copy.deepcopy(json)
        self.dim = dim

    def tw(self, tw, offset = None):
        from nnodely.input import State, Connect
        from nnodely.utils import merge
        s = State(self.name+"_state",dimensions=self.dim['dim'])
        if type(tw) == int:
            out_connect = Connect(self, s)
            win_state = s.tw(tw, offset)
            return Stream(win_state.name, merge(win_state.json, out_connect.json), win_state.dim,0 )

    def sw(self, sw, offset = None):
        from nnodely.input import State, Connect
        from nnodely.utils import merge
        s = State(self.name+"_state",dimensions=self.dim['dim'])
        if type(sw) == int:
            out_connect = Connect(self, s)
            win_state = s.sw(sw, offset)
            return Stream(win_state.name, merge(win_state.json, out_connect.json), win_state.dim,0 )

    def z(self, delay):
        from nnodely.input import State, Connect
        from nnodely.utils import merge
        s = State(self.name + "_state",dimensions=self.dim['dim'])
        if type(delay) == int and delay > 0:
            out_connect = Connect(self, s)
            win_state = s.z(delay)
            return Stream(win_state.name, merge(win_state.json, out_connect.json), win_state.dim,0 )

    def connect(self, obj):
        """
        Connects the current stream to a given state object.

        Parameters
        ----------
        obj : State
            The state object to connect to.

        Returns
        -------
        Stream
            A new Stream object representing the connected state.

        Raises
        ------
        TypeError
            If the provided object is not of type State.
        KeyError
            If the state variable is already connected.
        """
        from nnodely.input import State
        check(type(obj) is State, TypeError,
              f"The {obj} must be a State and not a {type(obj)}.")
        self.json = merge(self.json, obj.json)
        check('closedLoop' not in self.json['States'][obj.name] or 'connect' not in self.json['States'][obj.name], KeyError,
              f"The state variable {obj.name} is already connected.")
        self.json['States'][obj.name]['connect'] = self.name
        return Stream(self.name, self.json, self.dim,0 )

    def closedLoop(self, obj):
        """
        Creates a closed loop connection with a given state object.

        Parameters
        ----------
        obj : State
            The state object to create a closed loop with.

        Returns
        -------
        Stream
            A new Stream object representing the closed loop state.

        Raises
        ------
        TypeError
            If the provided object is not of type State.
        KeyError
            If the state variable is already connected.
        """
        from nnodely.input import State
        check(type(obj) is State, TypeError,
              f"The {obj} must be a State and not a {type(obj)}.")
        self.json = merge(self.json, obj.json)
        check('closedLoop' not in self.json['States'][obj.name] or 'connect' not in self.json['States'][obj.name],
              KeyError,
              f"The state variable {obj.name} is already connected.")
        self.json['States'][obj.name]['closedLoop'] = self.name
        return Stream(self.name, self.json, self.dim,0 )

class ToStream():
    def __new__(cls, *args, **kwargs):
        out = super(ToStream,cls).__new__(cls)
        out.__init__(*args, **kwargs)
        return Stream(out.name,out.json,out.dim,0)

class AutoToStream():
    def __new__(cls, *args,  **kwargs):
        if len(args) > 0 and (issubclass(type(args[0]),NeuObj) or type(args[0]) is Stream):
            instance = super().__new__(cls)
            instance.__init__()
            return instance(args[0])
        instance = super().__new__(cls)
        return instance
