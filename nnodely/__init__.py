"""
nnodely - Lazy/DAG implementation. No global graph.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

from nnodely.input import Input
from nnodely.stream import Stream
from nnodely.layer import Layer, LayerBase
from nnodely.fir import Fir
from nnodely.loop import Loop
#from nnodely.arithmetic import Add, Subtract, Multiply, Divide
from nnodely.parameter import Parameter
from nnodely.output import Output
from nnodely.model import Model

__all__ = [
    'Layer', 'LayerBase', 'Input', 'Stream', 'Fir', 'Parameter', 'Loop',
    'Output', 'Model'
]
