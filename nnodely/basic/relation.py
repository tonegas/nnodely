from nnodely.support.utils import check, enforce_types
from nnodely.basic.modeldef import ModelGraph
from nnodely.nnodely import get_manager

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)
    
# class Stream():
#     """
#     Represents a stream of data inside the neural network. 
#     A Stream is automatically create when you operate over a Input, Parameter, or Constant object.
#     """
#     def __init__(self, name : str, **attrs):
#         #mg = graph if graph is not None else get_current_model_graph()
#         is_nested = getattr(self, 'isnested', False)
#         if is_nested:
#             set_current_model_graph(self.parent)
#         else:
#             mg = get_current_model_graph()
#             self.attrs = attrs
#             self.attrs['type'] = self.__class__.__name__
#             name = mg.set_node(name=name, **self.attrs)
#         self.name = name

#     def __build__(self, name, from_node):
#         ## add a logic node to the parent graph containing the child graph
#         self.isnested = True
#         self.parent = get_current_model_graph()
#         mg = ModelGraph(name=name)
#         self.attrs = {'graph': mg, 'from': from_node} ## add the internal graph as attribute
#         self.parent.set_node(name=name, **self.attrs) ## add the logic node to the parent graph
#         set_current_model_graph(mg) ## set the current graph to the internal graph
#         get_manager().add_model(name, mg) ## add the internal graph to the model manager


#     @enforce_types
#     def tw(self, tw:float|int|list, offset:float|int|None = None, *, name:str|None = None) -> "Stream":
#         """
#         Selects a time window on Stream. It is possible to create a smaller or bigger time window on the stream.
#         The Time Window must be in the past not in the future.

#         Parameters
#         ----------
#         tw : float, int, list
#             The time window represents the time in the past. If a list, it should contain the start and end times, both indexes must be in the past.
#         offset : float, int, optional
#             The offset for the sample window. Default is None.
#         name : str, None
#             The name of the internal variable

#         Returns
#         -------
#         Stream
#             A Stream representing the TimePart object with the selected time window.

#         """
#         from nnodely.layers.part import TimePart
#         if isinstance(tw, list):
#             check(0 >= tw[1] > tw[0] and tw[0] < 0, ValueError, "The dimension of the time window must be in the past.")
#             return TimePart(self,tw[0],tw[1], name=name, offset=offset)
#         return TimePart(self,-abs(tw),0, name=name, offset=offset)

#     @enforce_types
#     def sw(self, sw:int|list, offset:int|None = None, *, name:str|None = None) -> "Stream":
#         """
#         Selects a sample window on Stream. It is possible to create a smaller or bigger window on the stream.
#         The Sample Window must be in the past not in the future.

#         Parameters
#         ----------
#         sw : int, list
#             The sample window represents the number of steps in the past. If a list, it should contain the start and end indices, both indexes must be in the past.
#         offset : int, optional
#             The offset for the sample window. Default is None.
#         name : str, None
#             The name of the internal variable

#         Returns
#         -------
#         Stream
#             A Stream representing the SamplePart object with the selected samples.

#         """
#         from nnodely.layers.part import SamplePart
#         if isinstance(sw, list):
#             check(0 >= sw[1] > sw[0] and sw[0] < 0, ValueError, "The dimension of the sample window must be in the past.")
#             return SamplePart(self,sw[0],sw[1], name=name, offset=offset)
#         return SamplePart(self,-abs(sw),0, name=name, offset=offset)

    
## Relation must have a name , a type, and attributes
class Relation():
    def __init__(self, name, edges : str | list | None  = None, **attrs):
        is_nested = getattr(self, 'isnested', False)
        if is_nested:
            get_manager().set_current_model(self.parent.name)
        else:
            model = get_manager().get_current_model()
            self.attrs = attrs
            self.attrs['type'] = self.__class__.__name__
            name = model.set_node(name=name, **self.attrs)
        self.name = name
        if edges is not None:
            self.__set_edges__(edges)

    def __build__(self, name, from_node):
        ## add a logic node to the parent graph containing the child graph
        self.isnested = True
        self.parent = get_manager().get_current_model()
        mg = ModelGraph(name=name)
        self.attrs = {'graph': mg, 'from': from_node} ## add the internal graph as attribute
        self.attrs['type'] = self.__class__.__name__
        self.parent.set_node(name=name, **self.attrs) ## add the logic node to the parent graph
        name = get_manager().add_model(name, mg) ## add the internal graph to the model manager
        get_manager().set_current_model(mg.name) ## set the current graph to the internal graph

    def __call__(self, edges:list|str):
        self.__set_edges__(edges)
    
    def __set_edges__(self, edges:list|str):
        edges = edges if type(edges) is list else [edges]
        model = get_manager().get_current_model()
        for edge in edges:
            edge_attrs = {}
            model.set_edge(edge, self.name, **edge_attrs)

    @enforce_types
    def tw(self, tw:float|int|list, offset:float|int|None = None, *, name:str|None = None) -> "Relation":
        """
        Selects a time window on Stream. It is possible to create a smaller or bigger time window on the stream.
        The Time Window must be in the past not in the future.

        Parameters
        ----------
        tw : float, int, list
            The time window represents the time in the past. If a list, it should contain the start and end times, both indexes must be in the past.
        offset : float, int, optional
            The offset for the sample window. Default is None.
        name : str, None
            The name of the internal variable

        Returns
        -------
        Stream
            A Stream representing the TimePart object with the selected time window.

        """
        from nnodely.layers.part import TimePart
        if isinstance(tw, list):
            check(0 >= tw[1] > tw[0] and tw[0] < 0, ValueError, "The dimension of the time window must be in the past.")
            return TimePart(self,tw[0],tw[1], name=name, offset=offset)
        return TimePart(self,-abs(tw),0, name=name, offset=offset)

    @enforce_types
    def sw(self, sw:int|list, offset:int|None = None, *, name:str|None = None) -> "Relation":
        """
        Selects a sample window on Stream. It is possible to create a smaller or bigger window on the stream.
        The Sample Window must be in the past not in the future.

        Parameters
        ----------
        sw : int, list
            The sample window represents the number of steps in the past. If a list, it should contain the start and end indices, both indexes must be in the past.
        offset : int, optional
            The offset for the sample window. Default is None.
        name : str, None
            The name of the internal variable

        Returns
        -------
        Stream
            A Stream representing the SamplePart object with the selected samples.

        """
        from nnodely.layers.part import SamplePart
        if isinstance(sw, list):
            check(0 >= sw[1] > sw[0] and sw[0] < 0, ValueError, "The dimension of the sample window must be in the past.")
            return SamplePart(self,sw[0],sw[1], name=name, offset=offset)
        return SamplePart(self,-abs(sw),0, name=name, offset=offset)

    def connect(self, obj) -> "Relation":
        """
        Update the Stream adding a connects with a given input object.

        Parameters
        ----------
        obj : Input
            The Input object to connect to.

        Returns
        -------
        Stream
            A Stream of the signal that updates the Inputs with the connection.

        Raises
        ------
        TypeError
            If the provided object is not of type Input.
        KeyError
            If the input variable is already connected.
        """
        from nnodely.layers.input import Input
        if not isinstance(obj, Input):
            raise TypeError("The object to connect must be of type Input.")
        model = get_manager().get_current_model()
        model.set_edge(from_node=self.name, to_node=obj.name, type='connect')
        model.set_node_attr(obj.name, state='connect')


    def loop(self, obj) -> "Relation":
        """
        Update the Stream adding a closed loop connection with a given input object.

        Parameters
        ----------
        obj : Input
            The Input object to create a closed loop with.

        Returns
        -------
        Stream
            A Stream of the signal that updates the Inputs with the connection.

        Raises
        ------
        TypeError
            If the provided object is not of type Input.
        KeyError
            If the input variable is already connected.
        """
        from nnodely.layers.input import Input
        if not isinstance(obj, Input):
            raise TypeError("The object to close in loop must be of type Input.")
        model = get_manager().get_current_model()
        model.set_edge(from_node=self.name, to_node=obj.name, type='loop')
        model.set_node_attr(obj.name, state='loop')

    def __add__(self, obj):
        from nnodely.layers.arithmetic import Add
        return Add(self, obj)

    def __radd__(self, obj):
        from nnodely.layers.arithmetic import Add
        return Add(self, obj)

    def __sub__(self, obj):
        from nnodely.layers.arithmetic import Sub
        return Sub(self, obj)

    def __rsub__(self, obj):
        from nnodely.layers.arithmetic import Sub
        return Sub(obj, self)

    def __truediv__(self, obj):
        from nnodely.layers.arithmetic import Div
        return Div(self, obj)

    def __rtruediv__(self, obj):
        from nnodely.layers.arithmetic import Div
        return Div(obj, self)

    def __mul__(self, obj):
        from nnodely.layers.arithmetic import Mul
        return Mul(self, obj)

    def __rmul__(self, obj):
        from nnodely.layers.arithmetic import Mul
        return Mul(obj, self)

    def __pow__(self, obj):
        from nnodely.layers.arithmetic import Pow
        return Pow(self, obj)

    def __rpow__(self, obj):
        from nnodely.layers.arithmetic import Pow
        return Pow(obj, self)

    def __neg__(self):
        from nnodely.layers.arithmetic import Neg
        return Neg(self)