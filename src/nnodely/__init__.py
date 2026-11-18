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

# The device: the CPU, where nnodely's small models run fastest, unless
# NNODELY_DEVICE is "gpu". It is set through the variable each backend reads
# when it starts; unset, one the user gave the backend is kept.
_device = _os.environ.get("NNODELY_DEVICE", "").strip().lower()
if _device not in ("", "cpu", "gpu"):
    raise ValueError(
        f"NNODELY_DEVICE is 'cpu' or 'gpu', got {_os.environ['NNODELY_DEVICE']!r}."
    )
_keras_loaded = "keras" in _sys.modules
_variable = {"torch": "KERAS_TORCH_DEVICE", "jax": "JAX_PLATFORMS"}.get(_chosen)
# The CPU, unless asked for the GPU or given a device of the backend's own.
_wants_cpu = _device == "cpu" or (
    _device == "" and (_variable is None or _variable not in _os.environ)
)
if _variable is not None:
    if _device == "":
        _os.environ.setdefault(_variable, "cpu")
    elif _device == "cpu":
        _os.environ[_variable] = "cpu"
    elif _os.environ.get(_variable, "").lower() == "cpu":
        del _os.environ[_variable]  # the backend takes its accelerator

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

import warnings as _warnings

# TensorFlow has no variable of Keras for its device: its GPUs are hidden,
# which it allows until it computes for the first time.
_restricted = True
if _device != "gpu" and keras.backend.backend() == "tensorflow":
    import tensorflow as _tf

    if _tf.config.list_physical_devices("GPU"):
        try:
            _tf.config.set_visible_devices([], "GPU")
        except RuntimeError:  # it has computed already
            _restricted = False

#: What makes a GPU available to each backend, on each system.
_GPU_HINTS = {
    ("torch", "macos"): "PyTorch computes on the Apple GPU (MPS) of Apple silicon only",
    ("torch", "linux"): "install a CUDA build of PyTorch, from pytorch.org",
    ("torch", "windows"): "install a CUDA build of PyTorch, from pytorch.org",
    ("tensorflow", "macos"): "TensorFlow has no GPU on macOS: use the torch backend",
    (
        "tensorflow",
        "linux",
    ): 'install TensorFlow with CUDA: pip install "tensorflow[and-cuda]"',
    (
        "tensorflow",
        "windows",
    ): "TensorFlow has no GPU on native Windows: run it in WSL2",
    ("jax", "macos"): "JAX has no GPU on macOS: use the torch backend",
    ("jax", "linux"): 'install JAX with CUDA: pip install "jax[cuda12]"',
    ("jax", "windows"): "JAX has no GPU on native Windows: run it in WSL2",
}


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


def get_device() -> str:
    """The device nnodely computes on: ``"cpu"``, or the accelerator of the
    backend, such as ``"cuda:0"`` or ``"mps:0"`` (PyTorch), ``"gpu:0"``
    (TensorFlow, JAX).

    The device is chosen once, when nnodely is imported, from the
    ``NNODELY_DEVICE`` environment variable:

    - unset, or ``"cpu"``: the CPU, on every backend. nnodely's models are
      small, and run faster there than on a GPU. Unset, a device given to the
      backend itself, ``KERAS_TORCH_DEVICE`` or ``JAX_PLATFORMS``, is kept;
    - ``"gpu"``: the accelerator the backend has, CUDA, or the Apple GPU
      (MPS) on PyTorch. Without one, the import raises ``RuntimeError``
      saying what makes one available on the system.

    A ``keras.device(...)`` block changes the device inside it, and so what
    this returns there.

    Returns
    -------
    str
        The device a new tensor is placed on.

    See Also
    --------
    get_backend : The Keras backend nnodely runs on.

    Notes
    -----
    Set ``NNODELY_DEVICE`` before importing nnodely, or Keras, as
    ``KERAS_BACKEND``: ``NNODELY_DEVICE=gpu python train.py``. The device of
    a model is the one it is built on, kept until the Python process ends.

    Basic Examples
    --------------
    >>> import nnodely
    >>> nnodely.get_device()  # NNODELY_DEVICE unset
    'cpu'

    .. versionadded:: 2.0
    """
    tensor = keras.ops.zeros(())
    backend = keras.backend.backend()
    if backend == "torch":
        name = str(tensor.device)
    elif backend == "jax":
        device = next(iter(tensor.devices()))
        name = f"{device.platform}:{device.id}"
    else:
        name = tensor.device.rsplit("device:", 1)[-1]
    name = name.lower()
    return "cpu" if name.split(":")[0] == "cpu" else name


if _device == "gpu":
    _backend = keras.backend.backend()
    if _backend == "tensorflow":
        import tensorflow as _tf

        # Its GPUs listed, without starting it.
        _has_gpu = bool(_tf.config.list_physical_devices("GPU"))
    else:
        _has_gpu = get_device() != "cpu"
    if not _has_gpu:
        _system = {"darwin": "macos", "win32": "windows"}.get(_sys.platform, "linux")
        raise RuntimeError(
            f"NNODELY_DEVICE=gpu, but the {_backend!r} backend has no GPU here: "
            f"{_GPU_HINTS[(_backend, _system)]}. Unset NNODELY_DEVICE to compute "
            "on the CPU."
        )
elif _wants_cpu and (_keras_loaded or not _restricted) and get_device() != "cpu":
    _cpu_variable = {
        "torch": "KERAS_TORCH_DEVICE=cpu",
        "jax": "JAX_PLATFORMS=cpu",
        "tensorflow": 'CUDA_VISIBLE_DEVICES=""',
    }[keras.backend.backend()]
    _warnings.warn(
        f"nnodely computes on {get_device()!r}, not on the CPU: Keras, or the "
        "backend, chose its device before nnodely was imported. Import nnodely "
        f"first, or set {_cpu_variable} before the backend starts.",
        stacklevel=2,
    )


# Randomness is configured before importing layers so an environment seed is
# applied before any model objects or initializers are created.
from nnodely.utils.random import get_seed, set_seed  # noqa: E402

# Core
from nnodely.core.modely import Modely
from nnodely.core.dataloader import DataLoader, OnShort

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
from nnodely.layers.fuzzify import (
    Fuzzify,
    Gaussian,
    MembershipName,
    Rectangular,
    Triangular,
)
from nnodely.layers.interpolation import Interpolation, InterpolationMode
from nnodely.layers.equationlearner import EquationFunction, EquationLearner
from nnodely.layers.calculus import (
    Derivative,
    DerivativeStep,
    Integrate,
    IntegrateStep,
    IntegrationMethod,
)
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
from nnodely.layers.ode import Ode, OdeMethod
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
    "get_device",
    "Modely",
    "DataLoader",
    "OnShort",
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
    "MembershipName",
    "KerasFunction",
    "Triangular",
    "Rectangular",
    "Gaussian",
    "Interpolation",
    "InterpolationMode",
    "EquationLearner",
    "EquationFunction",
    "Derivative",
    "DerivativeStep",
    "Gradient",
    "Integrate",
    "IntegrateStep",
    "IntegrationMethod",
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
    "OdeMethod",
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
from nnodely.utils.splash import show_splash as _show_splash  # noqa: E402

_show_splash()
