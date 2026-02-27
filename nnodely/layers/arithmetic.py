import torch.nn as nn
import torch

from nnodely.basic.relation import ToStream, Stream, toStream
from nnodely.basic.model import Model
from nnodely.support.utils import check, enforce_types
from nnodely.layers.parameter import Parameter, Constant
from nnodely.support.jsonutils import merge, binary_cheks


# Binary operators
add_relation_name = 'Add'
sub_relation_name = 'Sub'
mul_relation_name = 'Mul'
div_relation_name = 'Div'
pow_relation_name = 'Pow'

# Unary operators
neg_relation_name = 'Neg'
sign_relation_name = 'Sign'

# Merge operator
sum_relation_name = 'Sum'

class Add(Stream, ToStream):
    """
        Implement the addition function between two tensors. 
        (it is also possible to use the classical math operator '+')

        See also:
            Official PyTorch Add documentation: 
            `torch.add <https://pytorch.org/docs/stable/generated/torch.add.html>`_

        :param input1: the first element of the addition
        :type obj: Tensor
        :param input2: the second element of the addition
        :type obj: Tensor

        Example:
            >>> add = Add(relation1, relation2)
            or
            >>> add = relation1 + relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = binary_cheks(self, obj1, obj2, 'addition operators (+)')
        super().__init__(add_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [add_relation_name,[obj1.name,obj2.name]]

## TODO: check the scalar dimension, helpful for the offset
class Sub(Stream, ToStream):
    """
        Implement the subtraction function between two tensors. 
        (it is also possible to use the classical math operator '-')

        :param input1: the first element of the subtraction
        :type obj: Tensor
        :param input2: the second element of the subtraction
        :type obj: Tensor

        Example:
            >>> sub = Sub(relation1, relation2)
            or
            >>> sub = relation1 - relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = binary_cheks(self, obj1, obj2, 'subtraction operators (-)')
        super().__init__(sub_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [sub_relation_name,[obj1.name,obj2.name]]

class Mul(Stream, ToStream):
    """
        Implement the multiplication function between two tensors. 
        (it is also possible to use the classical math operator '*')

        :param input1: the first element of the multiplication
        :type obj: Tensor
        :param input2: the second element of the multiplication
        :type obj: Tensor

        Example:
            >>> mul = Mul(relation1, relation2)
            or
            >>> mul = relation1 * relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = binary_cheks(self, obj1, obj2, 'multiplication operators (*)')
        super().__init__(mul_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [mul_relation_name,[obj1.name,obj2.name]]

class Div(Stream, ToStream):
    """
        Implement the division function between two tensors. 
        (it is also possible to use the classical math operator '/')

        :param input1: the numerator of the division
        :type obj: Tensor
        :param input2: the denominator of the division
        :type obj: Tensor

        Example:
            >>> div = Div(relation1, relation2)
            or
            >>> div = relation1 / relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = binary_cheks(self, obj1, obj2, 'division operators (/) ')
        super().__init__(div_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [div_relation_name,[obj1.name,obj2.name]]

class Pow(Stream, ToStream):
    """
        Implement the power function given an input and an exponent. 
        (it is also possible to use the classical math operator '**')

        See also:
            Official PyTorch pow documentation:
            `torch.pow <https://pytorch.org/docs/stable/generated/torch.pow.html>`_

        :param input: the base of the power function
        :type obj: Tensor
        :param exp: the exponent of the power function
        :type obj: float or Tensor

        Example:
            >>> pow = Pow(relation, exp)
            or
            >>> pow = relation1 ** relation2
    """
    @enforce_types
    def __init__(self, obj1:Stream|Parameter|Constant|int|float, obj2:Stream|Parameter|Constant|int|float) -> Stream:
        obj1, obj2, dim = binary_cheks(self, obj1, obj2, 'pow operators (**)')
        super().__init__(pow_relation_name + str(Stream.count),merge(obj1.json,obj2.json),dim)
        self.json['Relations'][self.name] = [pow_relation_name,[obj1.name,obj2.name]]

class Neg(Stream, ToStream):
    """
        Implement the negate function given an input. 

        :param input: the input to negate
        :type obj: Tensor

        Example:
            >>> x = Neg(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for neg operation.")
        super().__init__(neg_relation_name+str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [neg_relation_name,[obj.name]]

class Sign(Stream, ToStream):
    """
        Implement the sign function given an input. 

        :param input: the input for the sign function
        :type obj: Tensor

        Example:
            >>> x = Sign(x)
    """
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for sign operation.")
        super().__init__(sign_relation_name+str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [sign_relation_name,[obj.name]]

class Sum(Stream, ToStream):
    @enforce_types
    def __init__(self, obj:Stream|Parameter|Constant) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for sum operation.")
        obj.dim['dim'] = 1
        super().__init__(sum_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [sum_relation_name,[obj.name]]

class Add_Layer(nn.Module):
    #: :noindex:
    def __init__(self):
        super(Add_Layer, self).__init__()

    def forward(self, *inputs):
        results = inputs[0]
        for input in inputs[1:]:
            results = results + input
        return results

def createAdd(name, *inputs):
    """
     :noindex:
    """
    return Add_Layer()

class Sub_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self):
        super(Sub_Layer, self).__init__()

    def forward(self, *inputs):
        # Perform element-wise subtraction
        results = inputs[0]
        for input in inputs[1:]:
            results = results - input
        return results

def createSub(self, *inputs):
    """
     :noindex:
    """
    return Sub_Layer()


class Mul_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self):
        super(Mul_Layer, self).__init__()

    def forward(self, *inputs):
        results = inputs[0]
        for input in inputs[1:]:
            results = results * input
        return results

def createMul(name, *inputs):
    """
     :noindex:
    """
    return Mul_Layer()

class Div_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self):
        super(Div_Layer, self).__init__()

    def forward(self, *inputs):
        results = inputs[0]
        for input in inputs[1:]:
            results = results / input
        return results

def createDiv(name, *inputs):
    """
     :noindex:
    """
    return Div_Layer()

class Pow_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self):
        super(Pow_Layer, self).__init__()

    def forward(self, *inputs):
        return torch.pow(inputs[0], inputs[1])

def createPow(name, *inputs):
    """
     :noindex:
    """
    return Pow_Layer()

class Neg_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self):
        super(Neg_Layer, self).__init__()

    def forward(self, x):
        return -x

def createNeg(self, *inputs):
    """
     :noindex:
    """
    return Neg_Layer()

class Sign_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self):
        super(Sign_Layer, self).__init__()

    def forward(self, x):
        return torch.sign(x)

def createSign(self, *inputs):
    """
     :noindex:
    """
    return Sign_Layer()

class Sum_Layer(nn.Module):
    """
     :noindex:
    """
    def __init__(self):
        super(Sum_Layer, self).__init__()

    def forward(self, inputs):
        return torch.sum(inputs, dim = 2, keepdim = True)

def createSum(name, *inputs):
    """
     :noindex:
    """
    return Sum_Layer()

setattr(Model, add_relation_name, createAdd)
setattr(Model, sub_relation_name, createSub)
setattr(Model, mul_relation_name, createMul)
setattr(Model, div_relation_name, createDiv)
setattr(Model, pow_relation_name, createPow)

setattr(Model, neg_relation_name, createNeg)
setattr(Model, sign_relation_name, createSign)

setattr(Model, sum_relation_name, createSum)


