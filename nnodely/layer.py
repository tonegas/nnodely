"""
Layer - operazioni stream-to-stream. Crea Stream con predecessor=[input_stream].
"""

from nnodely.dag import next_name, to_tuple
from nnodely.stream import Stream
import keras

def _is_stream(x):
    """True se x è Stream o Input (nodo grafo)."""
    return hasattr(x, 'node_type') and hasattr(x, 'predecessors')


class LayerBase:
    """Base per oggetti usati come layer nel Model (Layer, _BinaryOp, _ParamOp)."""


class Layer(LayerBase):
    """
    Base per layer. __call__ accetta Stream o tensor.
    Crea output Stream con predecessor=[input_stream].

    Se il layer ha __init__ con tutti argomenti opzionali, può essere usato in due modi:
    - deg_rad = DegToRad(); s = deg_rad(x.sw(5))
    - s = DegToRad(x.sw(5))   # forma veloce: solo stream, parametri tutti default
    Per parametri non default: Fir(out_features=2)(x.sw(5))
    """

    input_stream_type = 'Stream'

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and not kwargs and _is_stream(args[0]):
            stream = args[0]
            instance = super().__new__(cls)
            instance.__init__()
            return instance(stream)
        return super().__new__(cls)

    def __init__(self, name=None, node_type=None, **kwargs):
        self._properties = {}
        self._layer = None
        self.node_type = node_type or getattr(self.__class__, 'node_type', None) or self.__class__.__name__
        self.name = name or next_name(self.node_type)
        self.predecessors = []
        for k, v in kwargs.items():
            self._properties[k] = v
            setattr(self, k, v)
        self.seq = getattr(self, 'seq', None)
        self.time = getattr(self, 'time', None)
        self.dim = getattr(self, 'dim', None)

    def __call__(self, x):
        if _is_stream(x) or (isinstance(x, list) and all(_is_stream(s) for s in x)):
            return self._call_stream(x)
        return self.call(x)

    def _call_stream(self, stream: Stream | list[Stream]) -> Stream:
        if not isinstance(stream, list):
            stream = [stream]
        if any(s.node_type != self.input_stream_type and s.node_type != 'input' for s in stream):
            raise ValueError(
                f"{self.__class__.__name__} expects {self.input_stream_type}, got {[s.node_type for s in stream]}"
            )
        self.seq = [s.seq for s in stream]
        self.time = [s.time for s in stream]
        self.dim = [to_tuple(s.dim, (1,)) for s in stream]
        out_seq, out_time, out_dim = self.output_shape(self.seq, self.time, self.dim)
        prefix = getattr(self.__class__, 'output_stream_prefix', None) or self.__class__.__name__
        # print(f"{prefix} | seq: {self.seq}, time: {self.time}, dim: {self.dim} → out_seq: {out_seq}, out_time: {out_time}, out_dim: {out_dim}")
        out_name = next_name(prefix)
        props = {**self._properties, 'layer': self}
        out_stream = Stream(
            out_name, self.input_stream_type,
            seq=out_seq, time=out_time, dim=out_dim,
            predecessors=stream,
            **props
        )
        return out_stream
    
    def output_shape(self, seq, time, dim):
        """Ritorna (out_seq, out_time, out_dim). Override per modificare. Default: propaga invariato."""
        return seq[0], time[0], dim[0]
    
    def build_layer(self):
        raise NotImplementedError("build_layer must be implemented in subclass")
    
    def call(self, x):
        raise NotImplementedError("call must be implemented in subclass")
