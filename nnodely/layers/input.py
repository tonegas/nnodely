import copy

from nnodely.basic.relation import NeuObj, Stream, ToStream
from nnodely.support.utils import check, enforce_types
from nnodely.support.jsonutils import merge, stream_to_str
from nnodely.layers.part import SamplePart, TimePart
from nnodely.layers.timeoperation import Derivate, Integrate

class Input(NeuObj):
    """
    Represents an Input in the neural network model.

    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/states.ipynb
        :alt: Open in Colab

    Parameters
    ----------
    json_name : str
        The name of the JSON field to store the Input configuration.
    name : str
        The name of the Input.
    dimensions : int, optional
        The number of dimensions for the input. Default is 1.

    Attributes
    ----------
    json_name : str
        The name of the JSON field to store the input configuration.
    name : str
        The name of the Input.
    dim : dict
        A dictionary containing the dimensions of the Input.
    json : dict
        A dictionary containing the configuration of the Input.
    """
    def __init__(self, name:str, *, dimensions:int = 1):
        """
        Initializes the Input object.

        Parameters
        ----------
        json_name : str
            The name of the JSON field to store the Input configuration.
        name : str
            The name of the Input.
        dimensions : int, optional
            The number of dimensions for the Input. Default is 1.
        """
        NeuObj.__init__(self, name)
        check(type(dimensions) == int, TypeError,"The dimensions must be a integer")
        self.json['Inputs'][self.name] = {'dim': dimensions }
        self.dim = {'dim': dimensions}

    @enforce_types
    def tw(self, tw:int|float|list, offset:int|float|None = None) -> Stream:
        """
        Selects a time window for the Input.

        Parameters
        ----------
        tw : list or float
            The time window. If a list, it should contain the start and end values. If a float, it represents the time window size.
        offset : float, optional
            The offset for the time window. Default is None.

        Returns
        -------
        Stream
            A Stream representing the TimePart object with the selected time window.

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
            check(len(tw) == 2, TypeError, "The time window must be a list of two elements.")
            check(tw[1] > tw[0], ValueError, "The dimension of the sample window must be positive")
            json['Inputs'][self.name]['tw'] = tw
            tw = tw[1] - tw[0]
        else:
            json['Inputs'][self.name]['tw'] = [-tw, 0]
        check(tw > 0, ValueError, "The time window must be positive")
        dim['tw'] = tw
        if offset is not None:
            check(json['Inputs'][self.name]['tw'][0] <= offset < json['Inputs'][self.name]['tw'][1],
                  IndexError,
                  "The offset must be inside the time window")
        return TimePart(Stream(self.name, json, dim), json['Inputs'][self.name]['tw'][0], json['Inputs'][self.name]['tw'][1], offset)


    @enforce_types
    def sw(self, sw:int|list, offset:int|None = None) -> Stream:
        """
        Selects a sample window for the Input.

        Parameters
        ----------
        sw : list, int
            The sample window. If a list, it should contain the start and end indices. If an int, it represents the number of steps in the past.
        offset : int, optional
            The offset for the sample window. Default is None.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the selected samples.

        Raises
        ------
        TypeError
            If the sample window is not an integer or a list of integers.

        Examples
        --------
        Select a sample window considering a signal T = [-3,-2,-1,0,1,2] where the time vector 0 represent the last passed instant. If sw is an integer #1 represent the number of step in the past
            >>> T.sw(2) #= [-1, 0] represents two sample step in the past

        If sw is a list [#1,#2] the numbers represent the sample indexes in the vector with the second element excluded
            >>> T.sw([-2,0])  #= [-1, 0] represents two time step in the past zero in the future
            >>> T.sw([0,1])   #= [1]     the first time in the future
            >>> T.sw([-4,-2]) #= [-3,-2]

        The total number of samples can be computed #2-#1. The offset represent the index of the vector that need to be used to offset the window
            >>> T.sw(2,offset=-2)       #= [0, 1]      the value of the window is [-1,0]
            >>> T.sw([-2,2],offset=-1)  #= [-1,0,1,2]  the value of the window is [-1,0,1,2]
        """
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        if type(sw) is list:
            check(len(sw) == 2, TypeError, "The sample window must be a list of two elements.")
            check(type(sw[0]) == int and type(sw[1]) == int, TypeError, "The sample window must be integer")
            check(sw[1] > sw[0], ValueError, "The dimension of the sample window must be positive")
            json['Inputs'][self.name]['sw'] = sw
            sw = sw[1] - sw[0]
        else:
            check(type(sw) == int, TypeError, "The sample window must be integer")
            json['Inputs'][self.name]['sw'] = [-sw, 0]
        check(sw > 0, ValueError, "The sample window must be positive")
        dim['sw'] = sw
        if offset is not None:
            check(json['Inputs'][self.name]['sw'][0] <= offset < json['Inputs'][self.name]['sw'][1],
                  IndexError,
                  "The offset must be inside the sample window")
        return SamplePart(Stream(self.name, json, dim), json['Inputs'][self.name]['sw'][0], json['Inputs'][self.name]['sw'][1], offset)

    @enforce_types
    def z(self, delay:int) -> Stream:
        """
        Considering the Zeta transform notation. The function is used to selects a unitary delay from the Input.

        Parameters
        ----------
        delay : int
            The delay value.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the selected delay.

        Examples
        --------
        Select the unitary delay considering a signal T = [-3,-2,-1,0,1,2], where the time vector 0 represent the last passed instant
            T.z(-1) = 1
            T.z(0)  = 0 # the last passed instant
            T.z(2)  = -2
        """
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        sw = [(-delay) - 1, (-delay)]
        json['Inputs'][self.name]['sw'] = sw
        dim['sw'] = sw[1] - sw[0]
        return SamplePart(Stream(self.name, json, dim), json['Inputs'][self.name]['sw'][0], json['Inputs'][self.name]['sw'][1], None)

    @enforce_types
    def last(self) -> Stream:
        """
        Selects the last passed instant for the input.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the last passed instant.
        """
        return self.z(0)

    @enforce_types
    def next(self) -> Stream:
        """
        Selects the next instant for the input.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the next instant.
        """
        return self.z(-1)

    @enforce_types
    def s(self, order:int, *, der_name:str|None = None, int_name:str|None = None, method:str = 'euler') -> Stream:
        """
        Considering the Laplace transform notation. The function is used to operate an integral or derivate operation on the input.
        The order of the integral or the derivative operation is indicated by the order parameter.

        Parameters
        ----------
        order : int
            Order of the Laplace transform
        method : str, optional
            Integration or derivation method

        Returns
        -------
        Stream
            A Stream of the signal represents the integral or derivation operation.
        """
        check(order != 0, ValueError, "The order must be a positive or negative integer not a zero")
        if order > 0:
            o = self.last()
            for i in range(order):
                o = Derivate(o, der_name = der_name, int_name = int_name, method = method)
        elif order < 0:
            o = self.last()
            for i in range(-order):
                o = Integrate(o, der_name = der_name, int_name = int_name, method = method)
        return o

    @enforce_types
    def connect(self, obj:Stream) -> "Input":
        """
        Update and return the current Input with a given Stream object.

        Parameters
        ----------
        obj : Stream
            The Stream object for update the Input.

        Returns
        -------
        Input
            A Input with the connection to the obj Stream

        Raises
        ------
        TypeError
            If the provided object is not of type Input.
        KeyError
            If the Input variable is already connected.
        """
        check(type(obj) is Stream, TypeError,
              f"The {obj} must be a Stream and not a {type(obj)}.")
        self.json = merge(self.json, obj.json)
        check('closedLoop' not in self.json['Inputs'][self.name] or 'connect' not in self.json['Inputs'][self.name], KeyError,
              f"The Input variable {self.name} is already connected.")
        self.json['Inputs'][self.name]['connect'] = obj.name
        self.json['Inputs'][self.name]['local'] = 1
        return self

    @enforce_types
    def closedLoop(self, obj:Stream) -> "Input":
        """
        Update and return the current Input in a closed loop with a given Stream object.

        Parameters
        ----------
        obj : Stream
            The Stream object for update the Input.

        Returns
        -------
        Input
            A Input with the connection to the obj Stream

        Raises
        ------
        TypeError
            If the provided object is not of type Input.
        KeyError
            If the Input variable is already connected.
        """
        from nnodely.layers.input import Input
        check(type(obj) is Stream, TypeError,
              f"The {obj} must be a Stream and not a {type(obj)}.")
        self.json = merge(self.json, obj.json)
        check('closedLoop' not in self.json['Inputs'][self.name] or 'connect' not in self.json['Inputs'][self.name],
              KeyError,
              f"The Input variable {self.name} is already connected.")
        self.json['Inputs'][self.name]['closedLoop'] = self.name
        self.json['Inputs'][self.name]['local'] = 1
        return self

    def __str__(self):
        return stream_to_str(self, 'Input')

    def __repr__(self):
        return self.__str__()

# connect operation
connect_name = 'connect'
closedloop_name = 'closedLoop'

class Connect(Stream, ToStream):
    @enforce_types
    def __init__(self, obj1:Stream, obj2:Input, *, local:bool=False) -> Stream:
        super().__init__(obj1.name,merge(obj1.json, obj2.json),obj1.dim)
        check(closedloop_name not in self.json['Inputs'][obj2.name] or connect_name not in self.json['Inputs'][obj2.name],
              KeyError,f"The input variable {obj2.name} is already connected.")
        self.json['Inputs'][obj2.name][connect_name] = obj1.name
        self.json['Inputs'][obj2.name]['local'] = int(local)

class ClosedLoop(Stream, ToStream):
    @enforce_types
    def __init__(self, obj1:Stream, obj2:Input, *, local:bool=False) -> Stream:
        super().__init__(obj1.name, merge(obj1.json, obj2.json), obj1.dim)
        check(closedloop_name not in self.json['Inputs'][obj2.name] or connect_name not in self.json['Inputs'][obj2.name],
              KeyError, f"The input variable {obj2.name} is already connected.")
        self.json['Inputs'][obj2.name][closedloop_name] = obj1.name
        self.json['Inputs'][obj2.name]['local'] = int(local)


class Input(Stream):
    def __init__(self, name:str, *, dimensions:int = 1):
        self.name = name
        self.attrs = {'dim': dimensions}
        check(type(dimensions) == int, TypeError,"The dimensions must be a integer")
        super().__init__(self, name, self.attrs)

    @enforce_types
    def tw(self, tw:int|float|list, offset:int|float|None = None) -> Stream:
        """
        Selects a time window for the Input.

        Parameters
        ----------
        tw : list or float
            The time window. If a list, it should contain the start and end values. If a float, it represents the time window size.
        offset : float, optional
            The offset for the time window. Default is None.

        Returns
        -------
        Stream
            A Stream representing the TimePart object with the selected time window.

        Raises
        ------
        ValueError
            If the time window is not positive.
        IndexError
            If the offset is not within the time window.
        """
        if isinstance(tw, list):
            check(len(tw) == 2, TypeError, "The time window must be a list of two elements.")
            check(tw[1] > tw[0], ValueError, "The second value of the time window must be greater than the first one.")
            tw = tw[1] - tw[0]
        check(tw > 0, ValueError, "The time window must be positive")
        self.attrs['tw'] = [-tw, 0]
        if offset is not None:
            check(self.attrs['tw'][0] <= offset < self.attrs['tw'][1], IndexError, "The offset must be inside the time window")
        return TimePart(self, self.attrs['tw'][0], self.attrs['tw'][1], offset)


    @enforce_types
    def sw(self, sw:int|list, offset:int|None = None) -> Stream:
        """
        Selects a sample window for the Input.

        Parameters
        ----------
        sw : list, int
            The sample window. If a list, it should contain the start and end indices. If an int, it represents the number of steps in the past.
        offset : int, optional
            The offset for the sample window. Default is None.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the selected samples.

        Raises
        ------
        TypeError
            If the sample window is not an integer or a list of integers.

        Examples
        --------
        Select a sample window considering a signal T = [-3,-2,-1,0,1,2] where the time vector 0 represent the last passed instant. If sw is an integer #1 represent the number of step in the past
            >>> T.sw(2) #= [-1, 0] represents two sample step in the past

        If sw is a list [#1,#2] the numbers represent the sample indexes in the vector with the second element excluded
            >>> T.sw([-2,0])  #= [-1, 0] represents two time step in the past zero in the future
            >>> T.sw([0,1])   #= [1]     the first time in the future
            >>> T.sw([-4,-2]) #= [-3,-2]

        The total number of samples can be computed #2-#1. The offset represent the index of the vector that need to be used to offset the window
            >>> T.sw(2,offset=-2)       #= [0, 1]      the value of the window is [-1,0]
            >>> T.sw([-2,2],offset=-1)  #= [-1,0,1,2]  the value of the window is [-1,0,1,2]
        """
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        if type(sw) is list:
            check(len(sw) == 2, TypeError, "The sample window must be a list of two elements.")
            check(type(sw[0]) == int and type(sw[1]) == int, TypeError, "The sample window must be integer")
            check(sw[1] > sw[0], ValueError, "The dimension of the sample window must be positive")
            json['Inputs'][self.name]['sw'] = sw
            sw = sw[1] - sw[0]
        else:
            check(type(sw) == int, TypeError, "The sample window must be integer")
            json['Inputs'][self.name]['sw'] = [-sw, 0]
        check(sw > 0, ValueError, "The sample window must be positive")
        dim['sw'] = sw
        if offset is not None:
            check(json['Inputs'][self.name]['sw'][0] <= offset < json['Inputs'][self.name]['sw'][1],
                  IndexError,
                  "The offset must be inside the sample window")
        return SamplePart(Stream(self.name, json, dim), json['Inputs'][self.name]['sw'][0], json['Inputs'][self.name]['sw'][1], offset)

    @enforce_types
    def z(self, delay:int) -> Stream:
        """
        Considering the Zeta transform notation. The function is used to selects a unitary delay from the Input.

        Parameters
        ----------
        delay : int
            The delay value.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the selected delay.

        Examples
        --------
        Select the unitary delay considering a signal T = [-3,-2,-1,0,1,2], where the time vector 0 represent the last passed instant
            T.z(-1) = 1
            T.z(0)  = 0 # the last passed instant
            T.z(2)  = -2
        """
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        sw = [(-delay) - 1, (-delay)]
        json['Inputs'][self.name]['sw'] = sw
        dim['sw'] = sw[1] - sw[0]
        return SamplePart(Stream(self.name, json, dim), json['Inputs'][self.name]['sw'][0], json['Inputs'][self.name]['sw'][1], None)

    @enforce_types
    def last(self) -> Stream:
        """
        Selects the last passed instant for the input.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the last passed instant.
        """
        return self.z(0)

    @enforce_types
    def next(self) -> Stream:
        """
        Selects the next instant for the input.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the next instant.
        """
        return self.z(-1)

    @enforce_types
    def s(self, order:int, *, der_name:str|None = None, int_name:str|None = None, method:str = 'euler') -> Stream:
        """
        Considering the Laplace transform notation. The function is used to operate an integral or derivate operation on the input.
        The order of the integral or the derivative operation is indicated by the order parameter.

        Parameters
        ----------
        order : int
            Order of the Laplace transform
        method : str, optional
            Integration or derivation method

        Returns
        -------
        Stream
            A Stream of the signal represents the integral or derivation operation.
        """
        check(order != 0, ValueError, "The order must be a positive or negative integer not a zero")
        if order > 0:
            o = self.last()
            for i in range(order):
                o = Derivate(o, der_name = der_name, int_name = int_name, method = method)
        elif order < 0:
            o = self.last()
            for i in range(-order):
                o = Integrate(o, der_name = der_name, int_name = int_name, method = method)
        return o

    @enforce_types
    def connect(self, obj:Stream) -> "Input":
        """
        Update and return the current Input with a given Stream object.

        Parameters
        ----------
        obj : Stream
            The Stream object for update the Input.

        Returns
        -------
        Input
            A Input with the connection to the obj Stream

        Raises
        ------
        TypeError
            If the provided object is not of type Input.
        KeyError
            If the Input variable is already connected.
        """
        check(type(obj) is Stream, TypeError,
              f"The {obj} must be a Stream and not a {type(obj)}.")
        self.json = merge(self.json, obj.json)
        check('closedLoop' not in self.json['Inputs'][self.name] or 'connect' not in self.json['Inputs'][self.name], KeyError,
              f"The Input variable {self.name} is already connected.")
        self.json['Inputs'][self.name]['connect'] = obj.name
        self.json['Inputs'][self.name]['local'] = 1
        return self

    @enforce_types
    def closedLoop(self, obj:Stream) -> "Input":
        """
        Update and return the current Input in a closed loop with a given Stream object.

        Parameters
        ----------
        obj : Stream
            The Stream object for update the Input.

        Returns
        -------
        Input
            A Input with the connection to the obj Stream

        Raises
        ------
        TypeError
            If the provided object is not of type Input.
        KeyError
            If the Input variable is already connected.
        """
        from nnodely.layers.input import Input
        check(type(obj) is Stream, TypeError,
              f"The {obj} must be a Stream and not a {type(obj)}.")
        self.json = merge(self.json, obj.json)
        check('closedLoop' not in self.json['Inputs'][self.name] or 'connect' not in self.json['Inputs'][self.name],
              KeyError,
              f"The Input variable {self.name} is already connected.")
        self.json['Inputs'][self.name]['closedLoop'] = self.name
        self.json['Inputs'][self.name]['local'] = 1
        return self

    def __str__(self):
        return stream_to_str(self, 'Input')

    def __repr__(self):
        return self.__str__()