import torch
import torch.nn as nn

from nnodely import Parameter, Constant
from nnodely.basic.relation import Stream, ToStream, toStream
from nnodely.basic.model import Model
from nnodely.support.utils import check, enforce_types


relu_relation_name = 'Relu'
elu_relation_name = 'ELU'
sigmoid_relation_name = 'Sigmoid'
identity_relation_name = 'Identity'
softmax_relation_name = 'Softmax'

class Relu(Stream, ToStream):
    """
        Implement the Rectified-Linear Unit (ReLU) relation function.

        See also:
            Official PyTorch ReLU documentation: 
            `torch.nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_

        :param obj: The relation stream.
        :type obj: Stream 

        Example:
            >>> x = Relu(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|float|int) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Relu operation.")
        super().__init__(relu_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [relu_relation_name,[obj.name]]

class ELU(Stream, ToStream):
    """
        Implement the Exponential-Linear Unit (ELU) relation function.

        See also:
            Official PyTorch ReLU documentation: 
            `torch.nn.ELU <https://pytorch.org/docs/stable/generated/torch.nn.ELU.html>`_

        :param obj: The relation stream.
        :type obj: Stream 

        Example:
            >>> x = ELU(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|float|int) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream,TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Tanh operation.")
        super().__init__(elu_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [elu_relation_name,[obj.name]]

class Identity(Stream, ToStream):
    """
    Implement the Identity relation function that simply returns the input vector x.

    See also:
        Official PyTorch Identity documentation: 
        `torch.nn.Identity <https://pytorch.org/docs/stable/generated/torch.nn.Identity.html>`_

    :param obj: The relation stream.
    :type obj: Stream 

    Example:
        >>> x = Identity(x)
    """
    @enforce_types
    def __init__(self, obj: Stream|Parameter|Constant|float|int) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Identity operation.")
        super().__init__(identity_relation_name + str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [identity_relation_name, [obj.name]]


class Softmax(Stream, ToStream):
    """
    Implement the Softmax relation function.

    See also:
        Official PyTorch Softmax documentation:
        `torch.nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_

    :param obj: The relation stream.
    :type obj: Stream

    Example:
        >>> x = Softmax(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|float|int) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Softmax operation.")
        super().__init__(softmax_relation_name + str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [softmax_relation_name, [obj.name]]

class Sigmoid(Stream, ToStream):
    r"""
    Implement the Sigmoid relation function.
    The Sigmoid function is defined as:

    see also:
        Official PyTorch Softmax documentation:
        `Sigmoid function <https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid>`_

    .. math::
        \sigma(x) = \frac{1}{1 + e^{-x}}

    :param obj: The relation stream.
    :type obj: Stream

    Example:
        >>> x = Sigmoid(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant|float|int) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for {sigmoid_relation_name} operation.")
        super().__init__(sigmoid_relation_name + str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [sigmoid_relation_name, [obj.name]]

class Relu_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self,):
        super(Relu_Layer, self).__init__()
    def forward(self, x):
        return torch.relu(x)
    
def createRelu(self, *input):
    """
     :noindex:
    """
    return Relu_Layer()
    

def createELU(self, *input):
    """
     :noindex:
    """
    return nn.ELU()

class Identity_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self, *args):
        super(Identity_Layer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
def createIdentity(self, *input):
    """
     :noindex:
    """
    return Identity_Layer()


class Sigmoid_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self,):
        super(Sigmoid_Layer, self).__init__()
    def forward(self, x):
        return 1/(1+torch.exp(-x))
    
def createSigmoid(self, *input):
    """
     :noindex:
    """
    return Sigmoid_Layer()

def createSoftmax(self, *input):
    """
     :noindex:
    """
    return nn.Softmax(dim=-1)

setattr(Model, relu_relation_name, createRelu)
setattr(Model, elu_relation_name, createELU)
setattr(Model, sigmoid_relation_name, createSigmoid)
setattr(Model, identity_relation_name, createIdentity)
setattr(Model, softmax_relation_name, createSoftmax)
