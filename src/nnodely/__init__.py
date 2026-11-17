from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("nnodely")
except _PackageNotFoundError:  # a source tree used without being installed
    __version__ = "unknown"

import importlib.util as _importlib_util
import json as _json
import os as _os
import sys as _sys

#: The Keras backends nnodely runs on, in the order one is taken when the
#: backend Keras would use is not installed.
_BACKENDS = ("tensorflow", "torch", "jax")


def _configured_backend() -> tuple[str, str]:
    """The backend Keras imports, and what chose it: KERAS_BACKEND, else
    the configuration file of Keras, else tensorflow, its default."""
    if _os.environ.get("KERAS_BACKEND"):
        return _os.environ["KERAS_BACKEND"], "set by KERAS_BACKEND"
    folder = _os.environ.get("KERAS_HOME") or _os.path.join(
        _os.path.expanduser("~"), ".keras"
    )
    path = _os.path.join(folder, "keras.json")
    try:
        with open(path, encoding="utf-8") as file:
            return str(_json.load(file)["backend"]), f"set in {path}"
    except (OSError, ValueError, KeyError, TypeError):
        return "tensorflow", "the default"


def _installed(backend: str) -> bool:
    return _importlib_util.find_spec(backend) is not None


# Keras reads its backend once, when it is imported. Unless the user chose
# one with KERAS_BACKEND, one that is not installed gives way to the first
# installed: nnodely[jax] alone runs on jax.
if "keras" not in _sys.modules and not _os.environ.get("KERAS_BACKEND"):
    if not _installed(_configured_backend()[0]):
        _available = [name for name in _BACKENDS if _installed(name)]
        if _available:
            _os.environ["KERAS_BACKEND"] = _available[0]

# What chose the backend, read before Keras writes its configuration file.
_chosen, _origin = _configured_backend()

# Keras imports its backend on import, and names only the missing module when
# that backend is not installed.
try:
    import keras  # noqa: F401
except ModuleNotFoundError as error:
    missing = (error.name or "").split(".")[0]
    if missing not in ("tensorflow", "torch", "jax", "jaxlib"):
        raise
    others = [name for name in _BACKENDS if name != _chosen and _installed(name)]
    raise ImportError(
        f"nnodely runs on Keras, whose {_chosen!r} backend ({_origin}) needs "
        f"{missing!r}, which is not installed. Install nnodely with it, "
        f'pip install "nnodely[{_chosen}]"'
        + (
            f", or run on an installed one, {others}, by setting KERAS_BACKEND "
            f"before importing nnodely: KERAS_BACKEND={others[0]} python "
            "script.py."
            if others
            else ". The backends are tensorflow, torch and jax: choose one with "
            "the KERAS_BACKEND environment variable before importing nnodely."
        )
    ) from error


def get_backend() -> str:
    """The Keras backend nnodely runs on: ``"tensorflow"``, ``"torch"`` or
    ``"jax"``.

    The backend is chosen once, when nnodely is imported, and kept until the
    Python process ends:

    1. ``KERAS_BACKEND``, an environment variable set before the import;
    2. else the ``"backend"`` of the Keras configuration file,
       ``~/.keras/keras.json``;
    3. else ``"tensorflow"``, the default of Keras.

    Without ``KERAS_BACKEND``, a backend of 2 or 3 that is not installed gives
    way to the first installed of ``tensorflow``, ``torch`` and ``jax``.

    Returns
    -------
    str
        The name of the backend.

    See Also
    --------
    set_seed : Seeds Python, NumPy and the backend.

    Notes
    -----
    To run on another backend, set ``KERAS_BACKEND`` and start a new Python
    process, or restart the kernel of a notebook: a backend is not changed
    once Keras is imported.

    Basic Examples
    --------------
    >>> import keras
    >>> import nnodely
    >>> nnodely.get_backend() in ("tensorflow", "torch", "jax")
    True
    >>> nnodely.get_backend() == keras.backend.backend()
    True

    .. versionadded:: 2.0
    """
    return keras.backend.backend()


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
from nnodely.layers.loop import Loop
from nnodely.layers.buffer import Buffer
from nnodely.layers.localmodel import LocalModel
from nnodely.layers.function import KerasFunction
from nnodely.layers.fuzzify import Fuzzify, Gaussian, Rectangular, Triangular
from nnodely.layers.interpolation import Interpolation
from nnodely.layers.equationlearner import EquationLearner
from nnodely.layers.calculus import Derivative, DerivativeStep, Integrate, IntegrateStep
from nnodely.layers.gradient import Gradient
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
from nnodely.layers.ode import Ode
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
from nnodely.layers.trigonometric import (
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,
    Cos,
    Cosh,
    Sin,
    Sinh,
    Tan,
)
from nnodely.layers.arithmetic import (
    Abs,
    Ceil,
    Clamp,
    Cross,
    Deg2Rad,
    Exp,
    Floor,
    Log,
    Log10,
    Log2,
    Max,
    Maximum,
    Mean,
    Min,
    Minimum,
    Norm,
    Rad2Deg,
    Round,
    Sign,
    Sqrt,
    Square,
    Sum,
    Negative,
)

# Public API
__all__ = [
    "set_seed",
    "get_seed",
    "get_backend",
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
    "Loop",
    "Buffer",
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
    "Gradient",
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
    "Sinh",
    "Cosh",
    "Asinh",
    "Acosh",
    "Atanh",
    "Exp",
    "Log",
    "Log10",
    "Log2",
    "Floor",
    "Round",
    "Deg2Rad",
    "Rad2Deg",
    "Negative",
    "Abs",
    "Ceil",
    "Clamp",
    "Maximum",
    "Minimum",
    "Sign",
    "Sqrt",
    "Square",
    "Sum",
    "Mean",
    "Max",
    "Min",
    "Norm",
    "Cross",
]

# The splash, in a terminal or a notebook: NNODELY_SPLASH=0 turns it off.
from nnodely.utils.splash import show_splash  # noqa: E402

show_splash()
