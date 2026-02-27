import inspect

from collections.abc import Callable

from nnodely.basic.relation import NeuObj, Stream
from nnodely.layers.part import Select
from nnodely.support.utils import check, enforce_types

localmodel_relation_name = 'LocalModel'

class LocalModel(NeuObj):
    """
    Represents a Local Model relation in the neural network model.

    Parameters
    ----------
    input_function : Callable, optional
        A callable function to process the inputs. 
    output_function : Callable, optional
        A callable function to process the outputs. 
    pass_indexes : bool, optional
        A boolean indicating whether to pass indexes to the functions. Default is False.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    pass_indexes : bool
        A boolean indicating whether to pass indexes to the functions.
    input_function : Callable
        The function to process the inputs.
    output_function : Callable
        The function to process the outputs.

    Examples
    --------
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/localmodel.ipynb
        :alt: Open in Colab

    Example - basic usage:
        >>> x = Input('x')
        >>> activation = Fuzzify(2,[0,1],functions='Triangular')(x.last())
        >>> loc = LocalModel(input_function=Fir())
        >>> out = Output('out', loc(x.tw(1), activation))

    Example - passing a custom function:
        >>> def myFun(in1,p1,p2):
        >>>     return p1*in1+p2

        >>> x = Input('x')
        >>> activation = Fuzzify(2,[0,1],functions='Triangular')(x.last())
        >>> loc = LocalModel(input_function = lambda:ParamFun(myFun), output_function = lambda:Fir)(x.last(), activation)
        >>> out = Output('out', loc)

    Example - custom function with multiple activations:
        >>> x = Input('x')
        >>> F = Input('F')
        >>> activationA = Fuzzify(2,[0,1],functions='Triangular')(x.tw(1))
        >>> activationB = Fuzzify(2,[0,1],functions='Triangular')(F.tw(1))

        >>> def myFun(in1,p1,p2):
        >>>     return p1*in1+p2

        >>> loc = LocalModel(input_function = lambda:ParamFun(myFun), output_function = Fir(3))(x.tw(1),(activationA,activationB))
        >>> out = Output('out', loc)
    """
    @enforce_types
    def __init__(self, input_function:Callable|None = None,
                 output_function:Callable|None = None, *,
                 pass_indexes:bool = False):

        self.relation_name = localmodel_relation_name
        self.pass_indexes = pass_indexes
        super().__init__(localmodel_relation_name + str(NeuObj.count))
        self.json['Functions'][self.name] = {}
        if input_function is not None:
            check(callable(input_function), TypeError, 'The input_function must be callable')
        self.input_function = input_function
        if output_function is not None:
            check(callable(output_function), TypeError, 'The output_function must be callable')
        self.output_function = output_function

    @enforce_types
    def __call__(self, inputs:Stream|tuple, activations:Stream|tuple= None):
        out_sum = []
        if type(activations) is not tuple:
            activations = (activations,)
        self.___activations_matrix(activations,inputs,out_sum)

        out = out_sum[0]
        for ind in range(1,len(out_sum)):
            out = out + out_sum[ind]
        return out

    # Definisci una funzione ricorsiva per annidare i cicli for
    def ___activations_matrix(self, activations, inputs, out, idx=0, idx_list=[]):
        if idx != len(activations):
            for i in range(activations[idx].dim['dim']):
                self.___activations_matrix(activations, inputs, out, idx+1, idx_list+[i])
        else:
            if self.input_function is not None:
                if len(inspect.signature(self.input_function).parameters) == 0:
                    if type(inputs) is tuple:
                        out_in = self.input_function()(*inputs)
                    else:
                        out_in = self.input_function()(inputs)
                else:
                    if self.pass_indexes:
                        if type(inputs) is tuple:
                            out_in = self.input_function(idx_list)(*inputs)
                        else:
                            out_in = self.input_function(idx_list)(inputs)
                    else:
                        if type(inputs) is tuple:
                            out_in = self.input_function(*inputs)
                        else:
                            out_in = self.input_function(inputs)
            else:
                check(type(inputs) is not tuple, TypeError, 'The input cannot be a tuple without input_function')
                out_in = inputs

            act = Select(activations[0], idx_list[0])
            for ind, i  in enumerate(idx_list[1:]):
                act = act * Select(activations[ind+1], i)

            prod = out_in * act

            if self.output_function is not None:
                if len(inspect.signature(self.output_function).parameters) == 0:
                    out.append(self.output_function()(prod))
                else:
                    if self.pass_indexes:
                        out.append(self.output_function(idx_list)(prod))
                    else:
                        out.append(self.output_function(prod))
            else:
                out.append(prod)
