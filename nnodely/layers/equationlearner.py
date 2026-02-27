import inspect

from nnodely.basic.relation import NeuObj, Stream
from nnodely.support.utils import check, enforce_types

from nnodely.layers.linear import Linear
from nnodely.layers.part import Select, Concatenate
from nnodely.layers.fuzzify import Fuzzify
from nnodely.layers.parametricfunction import ParamFun
from nnodely.layers.activation import Relu, ELU, Identity, Sigmoid
from nnodely.layers.trigonometric import Sin, Cos, Tan, Tanh, Cosh, Sech
from nnodely.layers.arithmetic import Add, Mul, Sub, Neg, Pow, Sum

equationlearner_relation_name = 'EquationLearner'
Available_functions = [Sin, Cos, Tan, Cosh, Tanh, Sech, Add, Mul, Sub, Neg, Pow, Sum, Concatenate, Relu, ELU, Identity, Sigmoid]
Initialized_functions = [ParamFun, Fuzzify]

class EquationLearner(NeuObj):
    """
    Represents a nnodely implementation of the Task-Parametrized Equation Learner block.

    See also:
        Task-Parametrized Equation Learner official paper: 
        `Equation Learner <https://www.sciencedirect.com/science/article/pii/S0921889022001981>`_

    Parameters
    ----------
    functions : list
        A list of callable functions to be used as activation functions.
    linear_in : Linear, optional
        A Linear layer to process the input before applying the activation functions. If not provided a random initialized linear layer will be used instead.
    linear_out : Linear, optional
        A Linear layer to process the output after applying the activation functions. Can be omitted.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    linear_in : Linear or None
        The Linear layer to process the input.
    linear_out : Linear or None
        The Linear layer to process the output.
    functions : list
        The list of activation functions.
    func_parameters : dict
        A dictionary mapping function indices to the number of parameters they require.
    n_activations : int
        The total number of activation functions.

    Examples
    --------
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/equation_learner.ipynb
        :alt: Open in Colab

    Example - basic usage:
        >>> x = Input('x')

        >>> equation_learner = EquationLearner(functions=[Tan, Sin, Cos])
        >>> out = Output('out',equation_learner(x.last()))

    Example - passing a linear layer:
        >>> x = Input('x')

        >>> linear_layer = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0})
        >>> equation_learner = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer)

        >>> out = Output('out',equation_learner(x.last()))

    Example - passing a custom parametric function and multiple inputs:
        >>> x = Input('x')
        >>> F = Input('F')

        >>> def myFun(K1,p1):
                return K1*p1

        >>> K = Parameter('k', dimensions =  1, sw = 1,values=[[2.0]])
        >>> parfun = ParamFun(myFun, parameters = [K] )

        >>> equation_learner = EquationLearner([parfun])
        >>> out = Output('out',equation_learner((x.last(),F.last())))
    """
    @enforce_types
    def __init__(self, functions:list, *, linear_in:Linear|None = None, linear_out:Linear|None = None) -> Stream:
        self.relation_name = equationlearner_relation_name
        self.linear_in = linear_in
        self.linear_out = linear_out

        # input parameters
        self.functions = functions
        super().__init__(equationlearner_relation_name + str(NeuObj.count))

        self.func_parameters = {}
        for func_idx, func in enumerate(self.functions):
            check(callable(func), TypeError, 'The activation functions must be callable')
            if type(func) in Initialized_functions:
                if type(func) == ParamFun:
                    funinfo = inspect.getfullargspec(func.param_fun)
                    num_args = len(funinfo.args) - len(func.parameters_and_constants) if func.parameters_and_constants else len(funinfo.args)
                elif type(func) == Fuzzify:
                    init_signature = inspect.signature(func.__call__)  
                    parameters = list(init_signature.parameters.values())
                    num_args = len([param for param in parameters if param.name != "self"])
            else:
                check(func in Available_functions, ValueError, f'The function {func} is not available for the EquationLearner operation')
                init_signature = inspect.signature(func.__init__)  
                parameters = list(init_signature.parameters.values())
                num_args = len([param for param in parameters if param.name != "self"])
            self.func_parameters[func_idx] = num_args

        self.n_activations = sum(self.func_parameters.values())
        check(self.n_activations > 0, ValueError, 'At least one activation function must be provided')

    def __call__(self, inputs):
        if type(inputs) is not tuple:
            inputs = (inputs,)
        check(len(set([x.dim['sw'] if 'sw' in x.dim.keys() else x.dim['tw'] for x in inputs])) == 1, ValueError, 'All inputs must have the same time dimension')
        concatenated_input = inputs[0]
        for inp in inputs[1:]:
            concatenated_input = Concatenate(concatenated_input, inp)
        linear_layer = self.linear_in(concatenated_input) if self.linear_in else Linear(output_dimension=self.n_activations, b=True)(concatenated_input)
        idx, out = 0, None
        for func_idx, func in enumerate(self.functions):
            arguments = [Select(linear_layer,idx+arg_idx) for arg_idx in range(self.func_parameters[func_idx])]
            idx += self.func_parameters[func_idx]
            out = func(*arguments) if func_idx == 0 else Concatenate(out, func(*arguments))
        if self.linear_out:
            out = self.linear_out(out)
        return out
