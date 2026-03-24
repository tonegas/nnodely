"""
Mixin per operazioni aritmetiche su Input e Stream.
Import lazy per evitare circular imports.
"""


def _binary_op(self, other, op_cls, param_cls, swapped=False):
    """Esegue op tra self e other. swapped=True per radd, rsub, rtruediv."""
    from nnodely.parameter import _is_parameter
    if _is_parameter(other):
        return param_cls()(other, self) if swapped else param_cls()(self, other)
    return op_cls()(self, other)


class _ArithmeticMixin:
    """Mixin per __add__, __sub__, __mul__, __truediv__ e varianti r."""

    def __add__(self, other):
        from nnodely.arithmetic import Add
        from nnodely.param_arithmetic import ParamAdd
        return _binary_op(self, other, Add, ParamAdd)

    def __radd__(self, other):
        from nnodely.arithmetic import Add
        from nnodely.param_arithmetic import ParamAdd
        return _binary_op(self, other, Add, ParamAdd)

    def __sub__(self, other):
        from nnodely.arithmetic import Subtract
        from nnodely.param_arithmetic import ParamSubtract
        return _binary_op(self, other, Subtract, ParamSubtract)

    def __rsub__(self, other):
        from nnodely.arithmetic import Subtract
        from nnodely.param_arithmetic import ParamSubtract
        return _binary_op(self, other, Subtract, ParamSubtract, swapped=True)

    def __mul__(self, other):
        from nnodely.arithmetic import Multiply
        from nnodely.param_arithmetic import ParamMultiply
        return _binary_op(self, other, Multiply, ParamMultiply)

    def __rmul__(self, other):
        from nnodely.arithmetic import Multiply
        from nnodely.param_arithmetic import ParamMultiply
        return _binary_op(self, other, Multiply, ParamMultiply)

    def __truediv__(self, other):
        from nnodely.arithmetic import Divide
        from nnodely.param_arithmetic import ParamDivide
        return _binary_op(self, other, Divide, ParamDivide)

    def __rtruediv__(self, other):
        from nnodely.arithmetic import Divide
        from nnodely.param_arithmetic import ParamDivide
        return _binary_op(self, other, Divide, ParamDivide, swapped=True)
