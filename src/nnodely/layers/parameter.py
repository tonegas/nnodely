<<<<<<< HEAD:nnodely/layers/parameter.py
=======
import copy
import inspect
import textwrap
>>>>>>> develop:src/nnodely/layers/parameter.py
import numpy as np
import keras

from nnodely.core.stream import Stream
from nnodely.core.dag import to_tuple


<<<<<<< HEAD:nnodely/layers/parameter.py
class Parameter(Stream):
=======
def is_numpy_float(var):
    return isinstance(var, (np.float16, np.float32, np.float64))


class Constant(NeuObj, Relation):
>>>>>>> develop:src/nnodely/layers/parameter.py
    """
    Trainable symbolic source node.

    A Parameter is a source Stream with a standalone trainable Keras weight
    stored in `self.param`.

    Shape without batch:
        seq + (time,) + dim
    """
<<<<<<< HEAD:nnodely/layers/parameter.py
    node_type = "Parameter"
=======

    @enforce_types
    def __init__(
        self,
        name: str,
        values: list | float | int | np.ndarray,
        *,
        tw: float | int | None = None,
        sw: int | None = None,
    ):
>>>>>>> develop:src/nnodely/layers/parameter.py

    def __init__(
        self,
        name: str,
        *,
        value=None,
        initializer="random_normal",
        seq=(),
        time=1,
        dim=1,
        dtype="float32",
    ):
        dim = to_tuple(dim, (1,))
        seq = to_tuple(seq, ())

<<<<<<< HEAD:nnodely/layers/parameter.py
        super().__init__(
            name=name,
            node_type=self.node_type,
            seq=seq,
            time=time,
            dim=dim,
            predecessors=[],
        )

        self.value = value
        self.initializer = initializer
        self.dtype = dtype

        self.param = None

    def build_parameter(self):
        """
        Create the trainable Keras weight if it does not exist yet.
        """
        if self.param is not None:
            return self.param

        shape = self.shape

        if self.value is not None:
            value = np.asarray(self.value, dtype=np.float32)
            if value.shape != shape:
                try:
                    value = np.reshape(value, shape)
                except Exception as e:
                    raise ValueError(
                        f"Parameter '{self.name}' value shape {value.shape} "
                        f"is incompatible with expected shape {shape}"
                    ) from e
            init = value
        else:
            if self.initializer == "zeros":
                init = np.zeros(shape, dtype=np.float32)
            elif self.initializer == "ones":
                init = np.ones(shape, dtype=np.float32)
            elif self.initializer == "random_normal":
                init = np.random.randn(*shape).astype(np.float32)
            elif self.initializer == "random_uniform":
                init = np.random.uniform(-0.05, 0.05, size=shape).astype(np.float32)
            else:
                raise ValueError(f"Unsupported initializer: {self.initializer!r}")

        self.param = keras.Variable(
            initializer=init,
            shape=shape,
            dtype=self.dtype,
            trainable=True,
            name=self.name,
        )
        return self.param

    def as_tensor(self, anchor):
        v = self.build_parameter()
        return keras.layers.Lambda(
            lambda x: v,
            output_shape=self.shape,
            name=self.name,
        )(anchor)

    @property
    def value_numpy(self):
        if self.param is None:
            return None
        return np.array(self.param)
=======
        self.dim = {}
        if tw is not None:
            check(
                len(shape) >= 2,
                ValueError,
                "The dimension must be at least 2 if tw is set.",
            )
            check(sw is None, ValueError, "If tw is set sw must be None")
            dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
            self.dim["tw"] = tw
        elif sw is not None:
            check(
                len(shape) >= 2,
                ValueError,
                "The dimension must be at least 2 if sw is set.",
            )
            self.dim["sw"] = sw
            check(
                shape[0] == self.dim["sw"],
                ValueError,
                f"The sw = {sw} is different from sw = {shape[0]} of the values.",
            )
            dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
        else:
            dimensions = (
                1
                if len(shape[0:]) == 0
                else shape[0]
                if len(shape[0:]) == 1
                else list(shape[0:])
            )

        self.dim["dim"] = dimensions

        # deepcopy dimention information inside Parameters
        self.json["Constants"][self.name] = copy.deepcopy(self.dim)
        if type(values) in (float, int):
            self.json["Constants"][self.name]["values"] = [values]
        else:
            self.json["Constants"][self.name]["values"] = values


class Parameter(NeuObj, Relation):
    """
    Represents a trainable parameter in the neural network model.

    Notes
    -----
    .. note::
        You can find some initialization functions for the 'init' and 'init_params' parameters inside the initializer module.

    Parameters
    ----------
    name : str
        The name of the parameter.
    dimensions : int, list, tuple, or None, optional
        The dimensions of the parameter. Default is None.
    tw : float or int, optional
        The time window for the parameter. Default is None.
    sw : int, optional
        The sample window for the parameter. Default is None.
    values : list, float, int, np.ndarray, or None, optional
        The values by which initialize the parameter. Default is None.
    init : Callable, optional
        A callable for initializing the parameter values. Default is None.
    init_params : dict, optional
        A dictionary of parameters for the initializer. Default is None.

    Attributes
    ----------
    name : str
        The name of the parameter.
    dim : dict
        A dictionary containing the dimensions of the parameter.
    json : dict
        A dictionary containing the configuration of the parameter.

    Examples
    --------

    .. include:: /examples_basics/parameter_module_ex/parameter.rst
    """

    @enforce_types
    def __init__(
        self,
        name: str,
        dimensions: int | list | tuple | None = None,
        *,
        tw: float | int | None = None,
        sw: int | None = None,
        values: list | float | int | np.ndarray | None = None,
        init: Callable | str | None = None,
        init_params: dict | None = None,
    ):

        NeuObj.__init__(self, name)
        dimensions = list(dimensions) if type(dimensions) is tuple else dimensions
        if values is None:
            if dimensions is None:
                dimensions = 1
            self.dim = {"dim": dimensions}
            if tw is not None:
                check(sw is None, ValueError, "If tw is set sw must be None")
                self.dim["tw"] = tw
            elif sw is not None:
                self.dim["sw"] = sw

            # deepcopy dimention information inside Parameters
            self.json["Parameters"][self.name] = copy.deepcopy(self.dim)
        else:
            values = np.array(values, dtype=NP_DTYPE)
            shape = values.shape
            values = values.tolist()

            self.dim = {}
            if tw is not None:
                check(
                    len(shape) >= 2,
                    ValueError,
                    "The dimension must be at least 2 if tw is set.",
                )
                check(sw is None, ValueError, "If tw is set sw must be None")
                dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
                self.dim["tw"] = tw
            elif sw is not None:
                check(
                    len(shape) >= 2,
                    ValueError,
                    "The dimension must be at least 2 if sw is set.",
                )
                self.dim["sw"] = sw
                check(
                    shape[0] == self.dim["sw"],
                    ValueError,
                    f"The sw = {sw} is different from sw = {shape[0]} of the values.",
                )
                dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
            else:
                dimensions = (
                    1
                    if len(shape[0:]) == 0
                    else shape[0]
                    if len(shape[0:]) == 1
                    else list(shape[0:])
                )

            self.dim["dim"] = dimensions

            # deepcopy dimention information inside Parameters
            self.json["Parameters"][self.name] = copy.deepcopy(self.dim)
            if type(values) in (int, float):
                self.json["Parameters"][self.name]["init_values"] = [values]
            else:
                self.json["Parameters"][self.name]["init_values"] = values
            self.json["Parameters"][self.name]["values"] = self.json["Parameters"][
                self.name
            ]["init_values"]

        if init is not None:
            check(
                "values" not in self.json["Parameters"][self.name],
                ValueError,
                f"The parameter {self.name} is already initialized.",
            )
            # check(inspect.isfunction(init), ValueError,f"The init parameter must be a function.")
            if inspect.isfunction(init):
                code = textwrap.dedent(inspect.getsource(init)).replace('"', "'")
                self.json["Parameters"][self.name]["init_fun"] = {
                    "code": code,
                    "name": init.__name__,
                }
            elif type(init) is str:
                self.json["Parameters"][self.name]["init_fun"] = {"name": init}
            if init_params is not None:
                self.json["Parameters"][self.name]["init_fun"]["params"] = init_params


class SampleTime:
    """
    Represents a constant value that is equal to the sample time.

    Attributes
    ----------
    name : str
        The name of the constant.
    dim : dict
        A dictionary containing the dimensions of the constant.
    json : dict
        A dictionary containing the configuration of the constant.

    Example
    -------
    .. include:: /examples_basics/parameter_module_ex/sample_time.rst
    """

    name = "SampleTime"
    g = Constant(name, values=0)

    def __new__(cls):
        SampleTime.g.dim = {"dim": 1}
        SampleTime.g.json["Constants"][SampleTime.name] = copy.deepcopy(
            SampleTime.g.dim
        )
        SampleTime.g.json["Constants"][SampleTime.name]["values"] = SampleTime.name
        return SampleTime.g
>>>>>>> develop:src/nnodely/layers/parameter.py
