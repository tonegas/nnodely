from nnodely.basic.relation import Stream
from nnodely.support.utils import check, enforce_types
from nnodely.support.jsonutils import stream_to_str
from nnodely.layers.part import TimePart, SamplePart

class Input(Stream):
    def __init__(self, name:str, *, dimensions:int = 1):
        check(type(dimensions) == int, TypeError,"The dimensions must be a integer")
        attrs = {'dim': dimensions}
        super().__init__(name, **attrs)

    @enforce_types
    def tw(self, tw:int|float|list, offset:int|float|None = None):
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
        return TimePart(self, self.attrs['tw'][0], self.attrs['tw'][1], name="TimePart_"+self.name, offset=offset)


    @enforce_types
    def sw(self, sw:int|list, offset:int|None = None):
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
        if isinstance(sw, list):
            check(len(sw) == 2, TypeError, "The sample window must be a list of two elements.")
            check(sw[1] > sw[0], ValueError, "The second value of the sample window must be greater than the first one.")
            sw = sw[1] - sw[0]
        check(sw > 0, ValueError, "The sample window must be positive")
        self.attrs['sw'] = [-sw, 0]
        if offset is not None:
            check(self.attrs['sw'][0] <= offset < self.attrs['sw'][1], IndexError, "The offset must be inside the sample window")
        return SamplePart(self, self.attrs['sw'][0], self.attrs['sw'][1], name="SamplePart_"+self.name, offset=offset)

    # @enforce_types
    # def z(self, delay:int) -> Stream:
    #     """
    #     Considering the Zeta transform notation. The function is used to selects a unitary delay from the Input.

    #     Parameters
    #     ----------
    #     delay : int
    #         The delay value.

    #     Returns
    #     -------
    #     Stream
    #         A Stream representing the SamplePart object with the selected delay.

    #     Examples
    #     --------
    #     Select the unitary delay considering a signal T = [-3,-2,-1,0,1,2], where the time vector 0 represent the last passed instant
    #         T.z(-1) = 1
    #         T.z(0)  = 0 # the last passed instant
    #         T.z(2)  = -2
    #     """
    #     dim = copy.deepcopy(self.dim)
    #     json = copy.deepcopy(self.json)
    #     sw = [(-delay) - 1, (-delay)]
    #     json['Inputs'][self.name]['sw'] = sw
    #     dim['sw'] = sw[1] - sw[0]
    #     return SamplePart(Stream(self.name, json, dim), json['Inputs'][self.name]['sw'][0], json['Inputs'][self.name]['sw'][1], None)

    @enforce_types
    def last(self) -> Stream:
        """
        Selects the last passed instant for the input.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the last passed instant.
        """
        self.attrs['sw'] = [-1, 0]
        return SamplePart(self, self.attrs['sw'][0], self.attrs['sw'][1], name="SamplePart_"+self.name)

    @enforce_types
    def next(self) -> Stream:
        """
        Selects the next instant for the input.

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the next instant.
        """
        self.attrs['sw'] = [0, 1]
        return SamplePart(self, self.attrs['sw'][0], self.attrs['sw'][1], name="SamplePart_"+self.name)

    # @enforce_types
    # def connect(self, obj:Stream) -> "Input":
    #     """
    #     Update and return the current Input with a given Stream object.

    #     Parameters
    #     ----------
    #     obj : Stream
    #         The Stream object for update the Input.

    #     Returns
    #     -------
    #     Input
    #         A Input with the connection to the obj Stream

    #     Raises
    #     ------
    #     TypeError
    #         If the provided object is not of type Input.
    #     KeyError
    #         If the Input variable is already connected.
    #     """
    #     check(type(obj) is Stream, TypeError,
    #           f"The {obj} must be a Stream and not a {type(obj)}.")
    #     self.json = merge(self.json, obj.json)
    #     check('closedLoop' not in self.json['Inputs'][self.name] or 'connect' not in self.json['Inputs'][self.name], KeyError,
    #           f"The Input variable {self.name} is already connected.")
    #     self.json['Inputs'][self.name]['connect'] = obj.name
    #     self.json['Inputs'][self.name]['local'] = 1
    #     return self

    # @enforce_types
    # def closedLoop(self, obj:Stream) -> "Input":
    #     """
    #     Update and return the current Input in a closed loop with a given Stream object.

    #     Parameters
    #     ----------
    #     obj : Stream
    #         The Stream object for update the Input.

    #     Returns
    #     -------
    #     Input
    #         A Input with the connection to the obj Stream

    #     Raises
    #     ------
    #     TypeError
    #         If the provided object is not of type Input.
    #     KeyError
    #         If the Input variable is already connected.
    #     """
    #     from nnodely.layers.input import Input
    #     check(type(obj) is Stream, TypeError,
    #           f"The {obj} must be a Stream and not a {type(obj)}.")
    #     self.json = merge(self.json, obj.json)
    #     check('closedLoop' not in self.json['Inputs'][self.name] or 'connect' not in self.json['Inputs'][self.name],
    #           KeyError,
    #           f"The Input variable {self.name} is already connected.")
    #     self.json['Inputs'][self.name]['closedLoop'] = self.name
    #     self.json['Inputs'][self.name]['local'] = 1
    #     return self

    # def __str__(self):
    #     return stream_to_str(self, 'Input')

    # def __repr__(self):
    #     return self.__str__()