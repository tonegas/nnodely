"""
Operazioni aritmetiche Stream con Parameter (trainable).
stream + param, param * stream, ecc.
"""

import keras
import numpy as np
from nnodely.dag import next_name, to_tuple, get_seq_time_dim
from nnodely.stream import Stream
from nnodely.layer import LayerBase, _is_stream
from nnodely.parameter import Parameter, _is_parameter


def _param_stream_pair(a, b):
    """Restituisce (stream, param) se uno è stream e uno è param, altrimenti (None, None)."""
    if _is_stream(a) and _is_parameter(b):
        return a, b
    if _is_parameter(a) and _is_stream(b):
        return b, a
    return None, None


class _ParamOpKeras(keras.Layer):
    """Layer Keras che applica op tra input e parametro trainable."""

    def __init__(self, param: Parameter, op: str, swapped: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.param_ref = param
        self.op = op
        self._swapped = swapped

    def build(self, input_shape):
        # Param broadcast: per scalare (1,) va bene; per dim maggiori deve matchare
        self.w = self.add_weight(
            shape=self.param_ref.shape,
            initializer=keras.initializers.Constant(self.param_ref.initial_value),
            trainable=True,
            name=self.param_ref.name + "_w"
        )
        super().build(input_shape)

    def call(self, x):
        eps = 1e-8
        if self.op == 'add':
            return x + self.w
        if self.op == 'sub':
            return x - self.w if not self._swapped else self.w - x
        if self.op == 'mul':
            return x * self.w
        if self.op == 'div':
            return x / (self.w + eps) if not self._swapped else (self.w + eps) / x
        return x


class _ParamOp(LayerBase):
    """Base per op stream-op-param. Crea Stream con predecessors=[stream], param=param."""

    node_type = 'Stream'
    output_prefix = 'ParamOp'

    def __init__(self, op_name: str, op: str):
        self.op_name = op_name
        self.op = op
        self._layer = None
        self.name = next_name(op_name)
        self.predecessors = []
        self.param = None

    def __call__(self, a, b):
        stream, param = _param_stream_pair(a, b)
        if stream is None or param is None:
            raise TypeError(
                f"{self.op_name} richiede (Stream, Parameter) o (Parameter, Stream), non {type(a)}, {type(b)}"
            )
        self.param = param
        self._swapped = _is_parameter(a)  # param - stream vs stream - param
        seq, time, dim = get_seq_time_dim(stream)
        out_name = next_name(self.output_prefix)
        out_stream = Stream(
            out_name, self.node_type,
            seq=tuple(seq), time=time, dim=dim,
            predecessors=[stream],
            layer=self,
            param=param
        )
        return out_stream

    def build_layer(self):
        self._layer = _ParamOpKeras(
            self.param, self.op, swapped=getattr(self, '_swapped', False),
            name=self.name + "_layer"
        )
        return self._layer

    def call(self, x):
        if self._layer is None:
            self.build_layer()
        return self._layer(x)


class ParamAdd(_ParamOp):
    """stream + param (param trainable)."""

    def __init__(self):
        super().__init__('ParamAdd', 'add')


class ParamSubtract(_ParamOp):
    """stream - param o param - stream."""

    def __init__(self):
        super().__init__('ParamSub', 'sub')


class ParamMultiply(_ParamOp):
    """stream * param (param trainable)."""

    def __init__(self):
        super().__init__('ParamMul', 'mul')


class ParamDivide(_ParamOp):
    """stream / param (param trainable)."""

    def __init__(self):
        super().__init__('ParamDiv', 'div')
