import copy

from nnodely.relation import NeuObj, Stream, ToStream
from nnodely.utils import check, merge
from nnodely.part import SamplePart, TimePart

class InputState(NeuObj, Stream):
    """
    Represents an input state in the neural network model.

    Parameters
    ----------
    json_name : str
        The name of the JSON field to store the input state configuration.
    name : str
        The name of the input state.
    dimensions : int, optional
        The number of dimensions for the input state. Default is 1.

    Attributes
    ----------
    json_name : str
        The name of the JSON field to store the input state configuration.
    name : str
        The name of the input state.
    dim : dict
        A dictionary containing the dimensions of the input state.
    json : dict
        A dictionary containing the configuration of the input state.
    """
    def __init__(self, json_name, name, dimensions:int = 1):
        """
        Initializes the InputState object.

        Parameters
        ----------
        json_name : str
            The name of the JSON field to store the input state configuration.
        name : str
            The name of the input state.
        dimensions : int, optional
            The number of dimensions for the input state. Default is 1.
        """
        NeuObj.__init__(self, name)
        check(type(dimensions) == int, TypeError,"The dimensions must be a integer")
        self.json_name = json_name
        self.json[self.json_name][self.name] = {'dim': dimensions, 'tw': [0, 0], 'sw': [0,0] }
        self.dim = {'dim': dimensions}
        Stream.__init__(self, name, self.json, self.dim)

    def tw(self, tw, offset = None):
        """
        Selects a time window for the input state.

        Parameters
        ----------
        tw : list or int
            The time window. If a list, it should contain the start and end values. If an int, it represents the time window size.
        offset : int, optional
            The offset for the time window. Default is None.

        Returns
        -------
        TimePart
            A TimePart object representing the selected time window.

        Raises
        ------
        ValueError
            If the time window is not positive.
        IndexError
            If the offset is not within the time window.
        """
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        if type(tw) is list:
            json[self.json_name][self.name]['tw'] = tw
            tw = tw[1] - tw[0]
        else:
            json[self.json_name][self.name]['tw'][0] = -tw
        check(tw > 0, ValueError, "The time window must be positive")
        dim['tw'] = tw
        if offset is not None:
            check(json[self.json_name][self.name]['tw'][0] <= offset < json[self.json_name][self.name]['tw'][1],
                  IndexError,
                  "The offset must be inside the time window")
        return TimePart(Stream(self.name, json, dim), json[self.json_name][self.name]['tw'][0], json[self.json_name][self.name]['tw'][1], offset)

    # Select a sample window
    # Example T = [-3,-2,-1,0,1,2]       # time vector 0 represent the last passed instant
    # If sw is an integer #1 represent the number of step in the past
    # T.s(2)                = [-1, 0]    # represents two time step in the past
    # If sw is a list [#1,#2] the numbers represent the time index in the vector second element excluded
    # T.s([-2,0])           = [-1, 0]    # represents two time step in the past zero in the future
    # T.s([0,1])            = [1]        # the first time in the future
    # T.s([-4,-2])          = [-3,-2]
    # The total number of samples can be computed #2-#1
    # The offset represent the index of the vector that need to be used to offset the window
    # T.s(2,offset=-2)      = [0, 1]      # the value of the window is [-1,0]
    # T.s([-2,2],offset=-1)  = [-1,0,1,2]  # the value of the window is [-1,0,1,2]
    def sw(self, sw, offset = None):
        """
        Selects a sample window for the input state.

        Parameters
        ----------
        sw : list or int
            The sample window. If a list, it should contain the start and end indices. If an int, it represents the number of steps in the past.
        offset : int, optional
            The offset for the sample window. Default is None.

        Returns
        -------
        SamplePart
            A SamplePart object representing the selected sample window.

        Raises
        ------
        TypeError
            If the sample window is not an integer or a list of integers.
        """
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        if type(sw) is list:
            check(type(sw[0]) == int and type(sw[1]) == int, TypeError, "The sample window must be integer")
            json[self.json_name][self.name]['sw'] = sw
            sw = sw[1] - sw[0]
        else:
            check(type(sw) == int, TypeError, "The sample window must be integer")
            json[self.json_name][self.name]['sw'][0] = -sw
        check(sw > 0, ValueError, "The sample window must be positive")
        dim['sw'] = sw
        if offset is not None:
            check(json[self.json_name][self.name]['sw'][0] <= offset < json[self.json_name][self.name]['sw'][1],
                  IndexError,
                  "The offset must be inside the sample window")
        return SamplePart(Stream(self.name, json, dim), json[self.json_name][self.name]['sw'][0], json[self.json_name][self.name]['sw'][1], offset)

    # Select the unitary delay
    # Example T = [-3,-2,-1,0,1,2] # time vector 0 represent the last passed instant
    # T.z(-1) = 1
    # T.z(0)  = 0 #the last passed instant
    # T.z(2)  = -2
    def z(self, delay):
        """
        Selects a unitary delay for the input state.

        Parameters
        ----------
        delay : int
            The delay value.

        Returns
        -------
        SamplePart
            A SamplePart object representing the selected delay.
        """
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        sw = [(-delay) - 1, (-delay)]
        json[self.json_name][self.name]['sw'] = sw
        dim['sw'] = sw[1] - sw[0]
        return SamplePart(Stream(self.name, json, dim), json[self.json_name][self.name]['sw'][0], json[self.json_name][self.name]['sw'][1], None)

    def last(self):
        """
        Selects the last passed instant for the input state.

        Returns
        -------
        SamplePart
            A SamplePart object representing the last passed instant.
        """
        return self.z(0)

    def next(self):
        """
        Selects the next instant for the input state.

        Returns
        -------
        SamplePart
            A SamplePart object representing the next instant.
        """
        return self.z(-1)

    # def s(self, derivate):
    #     return Stream((self.name, {'s':derivate}), self.json, self.dim)


class Input(InputState):
    def __init__(self, name, dimensions:int = 1):
        InputState.__init__(self, 'Inputs', name, dimensions)

class State(InputState):
    def __init__(self, name, dimensions:int = 1):
        InputState.__init__(self, 'States', name, dimensions)


# connect operation
connect_name = 'connect'
closedloop_name = 'closedLoop'


# class Connect(Stream, ToStream):
#     def __init__(self, obj1: Stream, obj2: State) -> Stream:
#         check(type(obj1) is Stream, TypeError,
#               f"The {obj1} must be a Stream or Output and not a {type(obj1)}.")
#         obj1.connect(obj2)
#
# class ClosedLoop(Stream, ToStream):
#     def __init__(self, obj1: Stream, obj2: State) -> Stream:
#         check(type(obj1) is Stream, TypeError,
#               f"The {obj1} must be a Stream or Output and not a {type(obj1)}.")
#         obj1.closedloop(obj2)

class Connect(Stream, ToStream):
    def __init__(self, obj1:Stream, obj2:State) -> Stream:
        check(type(obj1) is Stream, TypeError,
              f"The {obj1} must be a Stream and not a {type(obj1)}.")
        check(type(obj2) is State, TypeError,
              f"The {obj2} must be a State and not a {type(obj2)}.")
        super().__init__(obj1.name,merge(obj1.json, obj2.json),obj1.dim)
        check(closedloop_name not in self.json['States'][obj2.name] or connect_name not in self.json['States'][obj2.name],
              KeyError,f"The state variable {obj2.name} is already connected.")
        self.json['States'][obj2.name][connect_name] = obj1.name

class ClosedLoop(Stream, ToStream):
    def __init__(self, obj1:Stream, obj2: State) -> Stream:
        check(type(obj1) is Stream, TypeError,
              f"The {obj1} must be a Stream and not a {type(obj1)}.")
        check(type(obj2) is State, TypeError,
              f"The {obj2} must be a State and not a {type(obj2)}.")
        super().__init__(obj1.name, merge(obj1.json, obj2.json), obj1.dim)
        check(closedloop_name not in self.json['States'][obj2.name] or connect_name not in self.json['States'][obj2.name],
              KeyError, f"The state variable {obj2.name} is already connected.")
        self.json['States'][obj2.name][closedloop_name] = obj1.name