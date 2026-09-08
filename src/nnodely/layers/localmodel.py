from __future__ import annotations


from nnodely.layers.time_ops import Select
from nnodely.core.layer import Add
from nnodely.core.modely import Modely
from nnodely.layers.input import Input
from nnodely.layers.output import Output


class LocalModel:
    """
    High-level abstraction for a local model built using only nnodely blocks
    """

    def __init__(
        self,
        input_function,
        output_function=None,
        name: str | None = None,
    ):
        self.input_function = input_function
        self.output_function = output_function
        self.name = name

    def __call__(self, activation):
        ret = []
        local = Input("local_input")
        for i in range(activation.dim[0]):
            x = self.input_function([local]) * Select(idx=i, axis=0)([activation])
            if self.output_function is not None:
                x = self.output_function([x])
            ret.append(x)
        ret = Add()(ret)
        out = Output("local_output", ret)
        return Modely(name=f"{self.name}", inputs=[local], outputs=[out])
