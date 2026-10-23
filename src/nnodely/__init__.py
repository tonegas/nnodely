from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("nnodely")
except _PackageNotFoundError:  # a source tree used without being installed
    __version__ = "unknown"

# Keras imports its backend on import, and names only the missing module when
# that backend is not installed.
try:
    import keras  # noqa: F401
except ModuleNotFoundError as error:
    missing = (error.name or "").split(".")[0]
    if missing not in ("tensorflow", "torch", "jax", "jaxlib"):
        raise
    raise ImportError(
        f"nnodely runs on Keras, whose backend needs {missing!r}, which is "
        "not installed. Install nnodely with a backend - "
        'pip install "nnodely[tensorflow]", "nnodely[torch]" or "nnodely[jax]" '
        "- and choose it with the KERAS_BACKEND environment variable, "
        "tensorflow when it is not set."
    ) from error

# Randomness is configured before importing layers so an environment seed is
# applied before any model objects or initializers are created.
from nnodely.utils.random import get_seed, set_seed  # noqa: E402

# Core
from nnodely.core.modely import Modely
from nnodely.core.dataloader import DataLoader
from nnodely.utils.printers import TinyPrinter, LegacyPrinter, NNodelyPrinter

# Layers
from nnodely.layers.input import Input
from nnodely.layers.output import Output
from nnodely.layers.fir import Fir
from nnodely.layers.linear import Linear
from nnodely.layers.parameter import Constant, Parameter
from nnodely.layers.roll import Roll
from nnodely.layers.loop import Loop
from nnodely.layers.localmodel import LocalModel
from nnodely.layers.function import KerasFunction
from nnodely.layers.fuzzify import Fuzzify, Gaussian, Rectangular, Triangular
from nnodely.layers.interpolation import Interpolation
from nnodely.layers.equationlearner import EquationLearner
from nnodely.layers.derivative import Derivative, DerivativeStep, Differentiate
from nnodely.layers.integrate import Integrate, IntegrateStep
from nnodely.layers.partitions import (
    Concatenate,
    Range,
    Select,
    SeqConcatenate,
    SeqRange,
    SeqReverse,
    SeqSelect,
    SeqStack,
    TimeConcatenate,
    TimeRange,
    TimeReverse,
    TimeSelect,
)
from nnodely.layers.batchnorm import BatchNorm
from nnodely.layers.ode import Ode, OdeNet
from nnodely.layers.activations import (
    ELU,
    GELU,
    PReLU,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Softmax,
    Softplus,
    Swish,
    Tanh,
)
from nnodely.layers.trigonometric import Acos, Asin, Atan, Atan2, Cos, Sin, Tan
from nnodely.layers.arithmetic import (
    Abs,
    Ceil,
    Clamp,
    Deg2Rad,
    Exp,
    Floor,
    Log,
    Log10,
    Sign,
    Sqrt,
    Sum,
    Negative,
)

# Public API
__all__ = [
    "set_seed",
    "get_seed",
    "Modely",
    "DataLoader",
    "TinyPrinter",
    "NNodelyPrinter",
    "LegacyPrinter",
    "Input",
    "Output",
    "Fir",
    "Linear",
    "Parameter",
    "Constant",
    "Roll",
    "Loop",
    "LocalModel",
    "Fuzzify",
    "KerasFunction",
    "Triangular",
    "Rectangular",
    "Gaussian",
    "Interpolation",
    "EquationLearner",
    "Derivative",
    "DerivativeStep",
    "Differentiate",
    "Integrate",
    "IntegrateStep",
    "Concatenate",
    "TimeConcatenate",
    "TimeSelect",
    "Select",
    "TimeRange",
    "TimeReverse",
    "SeqSelect",
    "SeqRange",
    "SeqConcatenate",
    "SeqStack",
    "SeqReverse",
    "Range",
    "BatchNorm",
    "Ode",
    "OdeNet",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "PReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Swish",
    "GELU",
    "Softplus",
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Atan2",
    "Exp",
    "Log",
    "Log10",
    "Floor",
    "Deg2Rad",
    "Negative",
    "Abs",
    "Ceil",
    "Clamp",
    "Sign",
    "Sqrt",
    "Sum",
]
