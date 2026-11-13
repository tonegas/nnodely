import math
from numbers import Integral
from typing import Any, cast

import numpy as np
import keras

from nnodely.core.layer import Layer, _refuse_non_streams, _refuse_uncalled
from nnodely.core.dag import Table
from nnodely.core.shape import Shape
from nnodely.layers.input import _dimension, _positive_sizes
from nnodely.utils.printers import _table
from nnodely.utils.utils import _per_slice_initializer, _serialized_initializer


@keras.saving.register_keras_serializable(package="nnodely")
class ParameterImpl(keras.layers.Layer):
    """A value of fixed shape, the same for every sample of the batch.

    It is called with a tensor of the graph, of which it reads only the batch
    size: the value, held as ``value_shape``, comes out ``(batch,
    *value_shape)`` like every stream.
    """

    trainable_value = True

    def __init__(
        self,
        value_shape,
        value=None,
        initializer="random_normal",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.value_shape = tuple(int(axis) for axis in value_shape)
        self.value = (
            None
            if value is None
            else np.asarray(value, dtype=np.float32).reshape(self.value_shape)
        )
        self.initializer = initializer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_shape": self.value_shape,
                "value": None if self.value is None else self.value.tolist(),
                "initializer": self.initializer,
            }
        )
        return config

    def _initializer(self):
        """What the variable starts from: the value given, or else a draw of
        the initializer."""
        if self.value is None:
            return keras.initializers.get(self.initializer)
        return keras.initializers.Constant(value=self.value.tolist())

    def build(self, input_shape=None):
        self.variable = self.add_weight(
            name="value",
            shape=self.value_shape,
            initializer=self._initializer(),
            trainable=self.trainable_value,
            dtype="float32",
        )
        super().build(input_shape)

    def reinitialize(self):
        """Give the variable an initial value again, as it was first given
        one."""
        initializer = cast(keras.initializers.Initializer, self._initializer())
        self.variable.assign(initializer(tuple(self.variable.shape), dtype="float32"))

    def call(self, anchor):
        value = keras.ops.expand_dims(self.variable, 0)
        if anchor is None:
            return value
        batch = keras.ops.shape(anchor)[0]
        return keras.ops.broadcast_to(value, (batch, *self.value_shape))


@keras.saving.register_keras_serializable(package="nnodely")
class ConstantImpl(ParameterImpl):
    trainable_value = False


def _declared_shape(owner: str, dim, tw, seq) -> Shape:
    """The shape ``dim``, ``tw`` and ``seq`` declare, the sizes of the axes
    as for an Input: ``dim`` one feature by default, each axis at least 1;
    ``tw`` the window, none by default; ``seq`` the sequence axes,
    outermost first, each of a fixed length - a value holds every step - and
    only with a window."""
    sequence = () if seq is None else _positive_sizes(seq)
    if sequence is None:
        raise ValueError(
            f"{owner}: seq is the length of each sequence axis, an int or a "
            f"tuple of them, outermost first, each at least 1 - a value holds "
            f"every step of its sequence, so none is dynamic; got {seq!r}."
        )
    if sequence and tw is None:
        raise ValueError(
            f"{owner}: a sequence needs a window: declare tw as well, tw=1 "
            "for one sample per step."
        )
    if tw is not None:
        window = _positive_sizes(tw) if isinstance(tw, Integral) else None
        if window is None:
            raise ValueError(
                f"{owner}: tw is the length of the window, an int of at least "
                f"1, or None for no window; got {tw!r}."
            )
        tw = window[0]
    return Shape(dim=_dimension(owner, dim), tw=tw, seq=sequence)


def _take_axes(shape: tuple, sizes: tuple) -> tuple[int, ...] | None:
    """The axes of ``shape`` that have ``sizes``, in their order: each size
    takes the first axis of that size not taken yet, so axes of one size
    keep their order. None if a size finds none."""
    free, taken = list(range(len(shape))), []
    for size in sizes:
        axis = next((axis for axis in free if shape[axis] == size), None)
        if axis is None:
            return None
        free.remove(axis)
        taken.append(axis)
    return tuple(taken)


def _value_layout(owner: str, value, dim, tw, seq) -> tuple[np.ndarray, Shape]:
    """``value`` laid out ``(*seq, tw, *dim)``, as a tensor holds a sample,
    and the shape of its stream, declared as for an Input
    (:func:`_declared_shape`).

    Without ``dim``, ``seq`` and ``tw`` take the axes of the value of their
    sizes, and the axes they leave are its dim, in their order: all of them
    when neither is given - like a number, the value then has no window and
    applies to every sample of any stream. A dim of one when they leave none.
    A value of one axis - a list of numbers - holds as many values as the
    shape, in its order, row-major. A value of several axes has the sizes the
    shape declares, in any order: its axes are moved into place, those of one
    size keeping their order.
    """
    array = np.atleast_1d(np.asarray(value, dtype=np.float32))
    if array.size == 0:
        raise ValueError(f"{owner}: value must not be empty.")
    shape = _declared_shape(owner, dim, tw, seq)
    if dim is None:
        # A value's sequence has fixed lengths: none is None.
        sequence = cast(tuple[int, ...], shape.seq)
        outer = (*sequence, *(() if shape.tw is None else (shape.tw,)))
        taken = _take_axes(array.shape, outer)
        if taken is not None:
            rest = tuple(axis for axis in range(array.ndim) if axis not in taken)
            dims = tuple(array.shape[axis] for axis in rest) or (1,)
            array = np.transpose(array, (*taken, *rest)).reshape(*outer, *dims)
            return array, shape.replace(dim=dims)
        if array.ndim > 1:
            raise ValueError(
                f"{owner}: shape error, a value of shape {array.shape} has no "
                f"axes of the sizes seq and tw declare, (*seq, tw) = "
                f"{outer}; the axes they leave are its dim."
            )

    declared = shape.tuple
    if array.ndim == 1:
        if array.size != math.prod(declared):
            raise ValueError(
                f"{owner}: shape error, {array.size} values given where the "
                f"declared shape (*seq, tw, *dim) = {declared} holds "
                f"{math.prod(declared)}."
            )
        return array.reshape(declared), shape
    order = _take_axes(array.shape, declared)
    if order is None or len(order) != array.ndim:
        raise ValueError(
            f"{owner}: shape error, a value of shape {array.shape} does not have "
            f"the sizes of the declared shape (*seq, tw, *dim) = {declared}, "
            f"in any order. Give a value of those sizes, or the "
            f"{math.prod(declared)} values as a list."
        )
    return np.transpose(array, order), shape


class _Value(Layer):
    """What a Parameter and a Constant share: a value the model holds rather
    than reads from data, laid out and batched like any other stream.

    The two differ only in whether training changes the value.
    """

    _impl: type[ParameterImpl] = ParameterImpl
    _trainable = True
    # Not read from data: without a window it applies to any stream's.
    _is_value = True

    def __init__(self, name, value, initializer, dim, tw, seq):
        owner = type(self).__name__ + ("" if name is None else f" {name!r}")
        if value is None:
            shape = _declared_shape(owner, dim, tw, seq)
        else:
            value, shape = _value_layout(owner, value, dim, tw, seq)

        self.value = value
        self._initializer = initializer

        super().__init__(
            name=name,
            tw=shape.tw,
            dim=shape.dim,
            seq=shape.seq,
            value=None if value is None else value.tolist(),
            initializer=initializer,
        )
        # The value as it is held: its sequence axes, its window if it has
        # one, then its dim - the shape of one sample of its stream.
        self._value_shape = self.shape.tuple

    def build_layer(self):
        return self._impl(
            value_shape=self._value_shape,
            value=self.value,
            initializer=self._initializer,
            trainable=self._trainable,
            name=self.name,
        )

    def get_config(self):
        # Read from no stream, a value keeps its shape in its config rather
        # than taking it from its inputs as other layers do: the sizes of its
        # axes, and the value laid out (*seq, tw, *dim) as they declare it.
        config = super().get_config()
        config.update(self.shape.get_config())
        return config

    @property
    def _variable(self):
        return getattr(self._layer, "variable", None)

    @property
    def value_numpy(self):
        return keras.ops.convert_to_numpy(self._variable)


class Parameter(_Value):
    r"""A trainable value of a fixed shape, the same for every sample of the
    batch.

    Syntax
    ------
    ``Parameter(name, value=v)``
        Starts from the value ``v``. Every axis of ``v`` is a feature axis.
    ``Parameter(name, dim=d, tw=t, seq=s)``
        Has the shape the sizes declare. ``initializer`` draws its value.
    ``Parameter(name, value=v, dim=d, tw=t, seq=s)``
        Has the declared shape, filled with ``v``.
    ``Parameter(name, ..., trainable=False)``
        Starts frozen: training keeps its value.

    A Parameter is a stream. Use it in arithmetic with other streams, or
    pass it to a layer as one of its weights, as in ``Fir(kernel=p)``.
    Training changes its value, unless it is frozen.

    ``dim``, ``tw`` and ``seq`` are sizes, declared as for an
    :class:`Input`. Without a window, a Parameter applies to every sample
    of the window of any stream it is combined with. The Notes say how a
    ``value`` fills the shape.

    Parameters
    ----------
    name : str, optional
        The name, under which :attr:`Modely.parameters` lists it. The
        default is ``Parameter`` followed by a number.
    value : float, list or numpy.ndarray, optional
        The initial value. Without it, ``initializer`` draws the initial
        value.
    initializer : str or keras.initializers.Initializer, default "random_normal"
        The Keras initializer, by name or object, that draws the initial
        value when ``value`` is not given. It is ignored otherwise.
    tw : int, optional
        The length of the window, ``T``, at least 1. Unlike the ``tw`` of an
        Input, it is one number, not a ``(past, future)`` pair. Without it
        the Parameter has no window.
    dim : int or tuple of int, optional
        The size of each feature axis, ``D``, each at least 1. Without it:

        - one feature, ``(1,)``, when there is no ``value``;
        - the axes of ``value`` that ``tw`` and ``seq`` leave, otherwise.
    seq : int or tuple of int, optional
        The length of each sequence axis, ``S``, outermost first, each at
        least 1. A sequence needs ``tw``. No length is dynamic: the value
        holds every step.
    trainable : bool, default True
        ``False`` keeps the value as it is through training, as
        :meth:`freeze` does.

    Attributes
    ----------
    value
        The initial value, a NumPy array laid out ``(*S, T, *D)``, without
        ``T`` when there is no window. ``None`` without a ``value``.
        Training does not change it.
    value_numpy
        The current value, a NumPy array laid out as ``value``. It holds a
        value after :meth:`Modely.build`, and is
        ``array(None, dtype=object)`` before.

    Returns
    -------
    Parameter
        A stream of ``Shape(S, T, D)``, as declared or read from ``value``.
        For example, ``Parameter("w", value=[1.0, 2.0, 3.0], tw=3)`` is
        ``Shape(S=(), T=3, D=(1,))``.

    Raises
    ------
    ValueError
        If a size is not allowed ("dim is the size of each feature axis",
        "tw is the length of the window", "seq is the length of each
        sequence axis"), if ``seq`` is given without ``tw`` ("a sequence
        needs a window"), if ``value`` is empty ("value must not be empty"),
        or if ``value`` does not fill the shape ("shape error").

    See Also
    --------
    Constant : A value that training does not change.
    Input : A signal read from data.
    Modely.parameters : The Parameters of a model, as a table.
    Modely.freeze : Freezes Parameters of a model by name.
    Fir : A layer whose weights can be Parameters.

    Notes
    -----
    Like every stream, a Parameter is laid out ``(batch, *S, T, *D)``,
    without ``T`` when it has no window. Every sample of the batch holds the
    same value, ``(*S, T, *D)``.

    A ``value`` fills the shape by these rules:

    - With ``dim``, the shape is the declared one. A list of numbers holds
      exactly as many values as the shape, read in the order ``(*S, T, *D)``
      (row-major). An array of several axes has the declared sizes in any
      order: its axes are moved into place, and axes of one size keep their
      order.
    - Without ``dim``, ``tw`` and each length of ``seq`` take the first free
      axis of ``value`` of that size. The axes left are the features, in
      their order. A list whose length is no such axis fills ``(*S, T)``
      with one feature.
    - Without ``dim``, ``tw`` and ``seq``, every axis of ``value`` is a
      feature axis, and there is no window.

    ==================================================  ===========================
    Declaration                                         Shape
    ==================================================  ===========================
    ``value=2.0``                                       ``S=(), T=None, D=(1,)``
    ``value=[1.0, 2.0, 3.0]``                           ``S=(), T=None, D=(3,)``
    ``value=[[1.0, 2.0, 3.0]]``                         ``S=(), T=None, D=(1, 3)``
    ``value=[1.0, 2.0, 3.0], tw=3``                     ``S=(), T=3, D=(1,)``
    ``value=[1.0, 2.0, 3.0], tw=1, dim=3``              ``S=(), T=1, D=(3,)``
    ``value=np.ones((2, 4, 6)), tw=4``                  ``S=(), T=4, D=(2, 6)``
    ``value=np.ones((3, 2)), tw=2, dim=3``              ``S=(), T=2, D=(3,)``
    ``value=np.arange(6.0), seq=2, tw=3``               ``S=(2,), T=3, D=(1,)``
    ``dim=(3, 2), tw=4``                                ``S=(), T=4, D=(3, 2)``
    ==================================================  ===========================

    ``value`` keeps the initial value. The variable the model trains is
    :attr:`param`, and ``value_numpy`` reads it as a NumPy array.

    Basic Examples
    --------------
    A gain that multiplies every sample of a window:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Constant, Input, Modely, Output, Parameter
    >>> gain = Parameter("gain", value=2.0)
    >>> gain.shape
    Shape(S=(), T=None, D=(1,))
    >>> x = Input("x", tw=2)
    >>> model = Modely("gain_demo", inputs=[x], outputs=[Output("y", gain * x)]).build()
    >>> result = model({"x": np.array([[[1.0], [3.0]]])})
    >>> keras.ops.convert_to_numpy(result["y"])
    array([[[2.],
            [6.]]], dtype=float32)
    >>> gain.value_numpy
    array([2.], dtype=float32)

    Scope
    -----
    Without ``dim``, ``tw`` and ``seq``, every axis of the value is a feature
    axis:

    >>> Parameter("row", value=[1.0, 2.0, 3.0]).shape
    Shape(S=(), T=None, D=(3,))
    >>> Parameter("matrix", value=[[1.0, 2.0, 3.0]]).shape
    Shape(S=(), T=None, D=(1, 3))

    ``tw`` takes the axis of its size as the window, and the axes left are
    the features:

    >>> Parameter("window", value=[1.0, 2.0, 3.0], tw=3).shape
    Shape(S=(), T=3, D=(1,))
    >>> Parameter("grid", value=np.ones((2, 4, 6)), tw=4).shape
    Shape(S=(), T=4, D=(2, 6))

    With ``dim``, a list fills the shape in row-major order, and an array has
    its axes moved into place:

    >>> Parameter("ramp", value=np.arange(6.0), tw=2, dim=3).value
    array([[0., 1., 2.],
           [3., 4., 5.]], dtype=float32)
    >>> Parameter("moved", value=np.arange(6.0).reshape(3, 2), tw=2, dim=3).value
    array([[0., 2., 4.],
           [1., 3., 5.]], dtype=float32)

    A sequence of windows, and a shape drawn by the initializer:

    >>> Parameter("path", value=np.arange(6.0), seq=2, tw=3).shape
    Shape(S=(2,), T=3, D=(1,))
    >>> Parameter("drawn", dim=(3, 2), tw=4).shape
    Shape(S=(), T=4, D=(3, 2))

    Options
    -------
    ``name`` is the key of the Parameter in the table of the model:

    >>> [p.name for p in model.parameters]
    ['gain']

    ``initializer`` draws the value when there is none:

    >>> ones = Parameter("ones", dim=3, initializer="ones")
    >>> v = Input("v", dim=3, tw=1)
    >>> scaled = Modely("ones_demo", inputs=[v], outputs=[Output("s", v * ones)]).build()
    >>> ones.value_numpy
    array([1., 1., 1.], dtype=float32)

    ``trainable=False`` creates the Parameter frozen, and :meth:`unfreeze`
    releases it:

    >>> fixed = Parameter("fixed", value=1.0, trainable=False)
    >>> fixed.trainable
    False
    >>> fixed.unfreeze().trainable
    True

    ``tw`` and ``seq`` are described in Scope, ``dim`` and ``value`` above.

    Applications
    ------------
    The acceleration of a mass-spring-damper, with a trained stiffness and
    damping and a known mass:

    >>> position = Input("position", tw=1)
    >>> velocity = Input("velocity", tw=1)
    >>> force = Input("force", tw=1)
    >>> stiffness = Parameter("stiffness", value=1.0)
    >>> damping = Parameter("damping", value=0.1)
    >>> acceleration = (force - stiffness * position - damping * velocity) / 2.0
    >>> acceleration.shape
    Shape(S=(), T=1, D=(1,))

    Properties & Relations
    ----------------------
    ``param`` is the variable the model trains, and ``value_numpy`` reads it:

    >>> bool(np.all(keras.ops.convert_to_numpy(gain.param) == gain.value_numpy))
    True

    A :class:`Constant` of the same value has the same shape, but is not
    listed among the Parameters of a model:

    >>> Constant("same", value=2.0).shape
    Shape(S=(), T=None, D=(1,))

    Possible Issues
    ---------------
    A value of three numbers does not fill a window of one sample of one
    feature. Declare the features:

    >>> Parameter("bad", value=[1.0, 2.0, 3.0], tw=1)
    Traceback (most recent call last):
        ...
    ValueError: ...shape error, 3 values given where the declared shape (*seq, tw, *dim) = (1, 1) holds 1.
    >>> Parameter("good", value=[1.0, 2.0, 3.0], tw=1, dim=3).shape
    Shape(S=(), T=1, D=(3,))

    A sequence length cannot be dynamic, since the value holds every step:

    >>> Parameter("steps", seq=-1, tw=1)
    Traceback (most recent call last):
        ...
    ValueError: ...seq is the length of each sequence axis...

    The variable exists only once the model is built: before,
    :attr:`param` is ``None``.

    >>> Parameter("unbuilt", value=1.0).param is None
    True

    .. versionadded:: 2.0
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        value=None,
        initializer="random_normal",
        tw=None,
        dim=None,
        seq=None,
        trainable: bool = True,
    ):
        self._trainable = bool(trainable)
        super().__init__(name, value, initializer, dim, tw, seq)

    @property
    def trainable(self) -> bool:
        """Whether training changes the value, ``True`` unless frozen.

        Set it before the model is built or after: the next training follows
        it. A Parameter shared by several models is one value, frozen or
        trained for all of them. The flag is saved with the model.

        See Also
        --------
        Parameter.freeze : Sets it to ``False``.
        Parameter.unfreeze : Sets it to ``True``.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, trainable: bool):
        self._trainable = bool(trainable)
        if self._layer is not None:
            self._layer.trainable = self._trainable

    def freeze(self):
        """Keep the value as it is through training.

        :meth:`unfreeze` undoes it. It sets :attr:`trainable` to ``False``,
        before the model is built or after.

        Returns
        -------
        Parameter
            The Parameter itself, so the call can be chained.

        See Also
        --------
        Parameter.unfreeze : Lets training change the value again.
        Modely.freeze : Freezes Parameters of a model by name.

        Examples
        --------
        >>> from nnodely import Parameter
        >>> k = Parameter("k", value=1.0)
        >>> k.freeze().trainable
        False
        >>> k.unfreeze().trainable
        True
        """
        self.trainable = False
        return self

    def unfreeze(self):
        """Let training change the value again.

        It sets :attr:`trainable` to ``True``, before the model is built or
        after.

        Returns
        -------
        Parameter
            The Parameter itself, so the call can be chained.

        See Also
        --------
        Parameter.freeze : Keeps the value through training.
        Modely.unfreeze : Releases Parameters of a model by name.
        """
        self.trainable = True
        return self

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initializer": _serialized_initializer(self._initializer),
                "trainable": self._trainable,
            }
        )
        return config

    @property
    def param(self):
        """The Keras variable that holds the current value, ``None`` before
        :meth:`Modely.build`.

        It is laid out as ``value``, ``(*S, T, *D)``, without ``T`` when there
        is no window. Training changes it unless the Parameter is frozen.
        Read it with ``keras.ops.convert_to_numpy``, or assign it a value of
        its shape with ``param.assign``.

        Examples
        --------
        >>> import keras
        >>> import numpy as np
        >>> from nnodely import Input, Modely, Output, Parameter
        >>> k = Parameter("k", value=[1.0, 2.0])
        >>> k.param is None
        True
        >>> v = Input("v", dim=2, tw=1)
        >>> model = Modely("param_demo", inputs=[v], outputs=[Output("y", k * v)]).build()
        >>> _ = k.param.assign(np.array([3.0, 4.0]))
        >>> keras.ops.convert_to_numpy(k.param)
        array([3., 4.], dtype=float32)
        >>> k.value  # the initial value is kept
        array([1., 2.], dtype=float32)
        """
        return self._variable

    def _reinitialize(self) -> None:
        if self._layer is not None:
            self._layer.reinitialize()


class Parameters(Table):
    """The Parameters of a model, as :attr:`Modely.parameters` lists them.

    It is a list of the Parameter objects. Pick one by its name,
    ``parameters["k"]``, or by its index, ``parameters[0]``: the ``idx``
    column of the table. A name that several Parameters carry raises
    ``ValueError``.

    Printed, it is a table of their names, shapes, numbers of values and
    whether training changes them. There is a section per model when the
    models it calls have Parameters of their own. The last row totals the
    values, all and trainable.

    ``groups`` pairs the name of each model with its Parameters, the model
    itself first."""

    _kind, _attr = "Parameter", "parameters"

    def __init__(self, groups: list[tuple[str, list]], title: str = ""):
        super().__init__(parameter for _, group in groups for parameter in group)
        self._groups = groups
        self._title = title
        self._models = [model for model, group in groups for _ in group]

    def _where(self, index, item) -> str:
        return f"in {self._models[index]!r}"

    def __str__(self):
        header = ("parameter", "seq", "tw", "dim", "values", "trainable")

        def row(parameter):
            return (
                parameter.name,
                str(tuple(parameter.shape.seq)),
                str(parameter.shape.tw),
                str(tuple(parameter.shape.dim)),
                str(math.prod(parameter.shape.tuple)),
                "yes" if parameter.trainable else "no",
            )

        # A section per model only when the models it calls have some.
        sections = [
            (name, [row(parameter) for parameter in group])
            for name, group in self._groups
            if group
        ]
        if not any(group for _, group in self._groups[1:]):
            sections = [("", [row(parameter) for parameter in self])]
        values = [math.prod(p.shape.tuple) for p in self]
        trainable = sum(v for v, p in zip(values, self) if p.trainable)
        total = ("total", "", "", "", str(sum(values)), str(trainable))
        return _table(header, sections, total, indexed=True)

    __repr__ = __str__


class Constant(_Value):
    r"""A fixed value of a fixed shape, the same for every sample of the
    batch, which training does not change.

    Syntax
    ------
    ``Constant(name, value=v)``
        Holds the value ``v``. Every axis of ``v`` is a feature axis.
    ``Constant(name, value=v, dim=d, tw=t, seq=s)``
        Has the shape the sizes declare, filled with ``v``.
    ``x * 9.81``
        A number in arithmetic with a stream becomes a Constant without a
        window.

    A Constant is a stream, not a layer: use it in arithmetic with other
    streams, or pass it where a layer takes a value. It declares its shape
    and is filled by its value as a :class:`Parameter` is. Unlike a
    Parameter, it requires a ``value``, and :attr:`Modely.parameters` does
    not list it.

    Parameters
    ----------
    name : str, optional
        The name of the stream. The default is ``Constant`` followed by a
        number.
    value : float, list or numpy.ndarray
        The value. It is required. It fills the shape by the rules of
        :class:`Parameter`.
    tw : int, optional
        The length of the window, ``T``, at least 1. Without it the Constant
        has no window, and applies to every sample of any stream.
    dim : int or tuple of int, optional
        The size of each feature axis, ``D``, each at least 1. Without it,
        the axes of ``value`` that ``tw`` and ``seq`` leave.
    seq : int or tuple of int, optional
        The length of each sequence axis, ``S``, outermost first, each at
        least 1. A sequence needs ``tw``.

    Attributes
    ----------
    value
        The value, a NumPy array laid out ``(*S, T, *D)``, without ``T``
        when there is no window.
    value_numpy
        The value held by the built model, a NumPy array laid out as
        ``value``. It is ``array(None, dtype=object)`` before
        :meth:`Modely.build`.

    Returns
    -------
    Constant
        A stream of ``Shape(S, T, D)``, as declared or read from ``value``.
        For example, ``Constant("c", value=[1.0, 2.0, 3.0])`` is
        ``Shape(S=(), T=None, D=(3,))``.

    Raises
    ------
    ValueError
        If ``value`` is ``None`` ("Constant requires a value"), if a size is
        not allowed, if ``seq`` is given without ``tw``, or if ``value``
        does not fill the shape ("shape error").
    TypeError
        If ``value`` is not given.

    See Also
    --------
    Parameter : A value that training changes.
    Input : A signal read from data.

    Notes
    -----
    Like every stream, a Constant is laid out ``(batch, *S, T, *D)``,
    without ``T`` when it has no window. Every sample of the batch holds the
    same value. The table of :class:`Parameter` lists how a value fills the
    shape.

    Basic Examples
    --------------
    An offset added to each feature of every sample:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Constant, Input, Modely, Output, Parameter
    >>> offsets = Constant("offsets", value=[1.0, 2.0, 3.0])
    >>> offsets.shape
    Shape(S=(), T=None, D=(3,))
    >>> v = Input("v", dim=3, tw=1)
    >>> model = Modely("offset_demo", inputs=[v], outputs=[Output("y", v + offsets)]).build()
    >>> keras.ops.convert_to_numpy(model({"v": np.zeros((1, 1, 3))})["y"])
    array([[[1., 2., 3.]]], dtype=float32)

    Scope
    -----
    ``tw`` takes the axis of its size as the window:

    >>> Constant("weights", value=[1.0, 2.0, 3.0], tw=3).shape
    Shape(S=(), T=3, D=(1,))

    With ``dim``, the value fills the declared shape:

    >>> Constant("frame", value=[1.0, 2.0, 3.0], tw=1, dim=3).shape
    Shape(S=(), T=1, D=(3,))

    A sequence of windows:

    >>> Constant("steps", value=np.arange(6.0), seq=2, tw=3).shape
    Shape(S=(2,), T=3, D=(1,))

    Options
    -------
    ``name`` names the stream:

    >>> Constant("gravity", value=9.81).name
    'gravity'

    Applications
    ------------
    The weight of a mass of 2 kg, with the gravity a known constant:

    >>> gravity = Constant("g", value=9.81)
    >>> mass = Parameter("mass", value=2.0)
    >>> weight = mass * gravity
    >>> weight.shape
    Shape(S=(), T=None, D=(1,))

    Properties & Relations
    ----------------------
    A Constant is not among the Parameters of a model, and freezing does not
    concern it:

    >>> [p.name for p in model.parameters]
    []

    A number in arithmetic with a stream is a Constant of that value:

    >>> x = Input("x", tw=2)
    >>> same = Modely(
    ...     "same", inputs=[x], outputs=[Output("a", x * 3.0), Output("b", x * Constant("three", value=3.0))]
    ... ).build()
    >>> result = same({"x": np.array([[[1.0], [2.0]]])})
    >>> bool(np.all(keras.ops.convert_to_numpy(result["a"]) == keras.ops.convert_to_numpy(result["b"])))
    True

    Possible Issues
    ---------------
    A Constant has no initializer, so it requires a value:

    >>> Constant("empty", value=None)
    Traceback (most recent call last):
        ...
    ValueError: Constant requires a value.

    .. versionadded:: 2.0
    """

    _impl = ConstantImpl
    _trainable = False

    def __init__(self, name: str | None = None, *, value, tw=None, dim=None, seq=None):
        if value is None:
            raise ValueError("Constant requires a value.")
        super().__init__(name, value, None, dim, tw, seq)

    @property
    def constant(self):
        """The Keras variable that holds the value, ``None`` before
        :meth:`Modely.build`.

        It is laid out as ``value``, and training does not change it.
        """
        return self._variable

    def get_config(self):
        # A Constant is its value: it has no initializer.
        config = super().get_config()
        del config["initializer"]
        return config


@keras.saving.register_keras_serializable(package="nnodely")
class WeightImpl(ParameterImpl):
    """A layer's weight: a variable of the weight's own shape.

    The layer reads it whole, not one value per sample, so it comes out with
    a batch axis of one: ``(1, *weight_shape)``. The initializer fills it
    with draws of ``draw_shape``.
    """

    def __init__(self, value_shape, weight_shape, draw_shape, name=None, **kwargs):
        super().__init__(value_shape=value_shape, name=name, **kwargs)
        self.weight_shape = tuple(int(axis) for axis in weight_shape)
        self.draw_shape = tuple(int(axis) for axis in draw_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"weight_shape": self.weight_shape, "draw_shape": self.draw_shape}
        )
        return config

    def _initializer(self):
        return _per_slice_initializer(self.initializer, self.draw_shape)

    def build(self, input_shape=None):
        self.variable = self.add_weight(
            name="value",
            shape=self.weight_shape,
            initializer=self._initializer(),
            trainable=True,
            dtype="float32",
        )
        keras.layers.Layer.build(self, input_shape)

    def call(self, anchor):
        return keras.ops.expand_dims(
            keras.ops.reshape(self.variable, self.value_shape), axis=0
        )


class Weight(Parameter):
    """A layer's kernel or bias held as a Parameter.

    A layer given an initializer rather than a Parameter makes one of these
    when it is first called. Its dim is the shape of the weight, and its
    variable has exactly that shape. ``initializer`` fills it with draws of
    shape ``draw``: by default the whole weight, read as a matrix of its last
    axis by all the others, as a dense layer reads its kernel. The layer
    reads it whole rather than one value per sample, so its stream has a
    batch axis of one.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        shape,
        initializer="glorot_uniform",
        draw=None,
        trainable: bool = True,
    ):
        self._weight_shape = tuple(int(axis) for axis in shape)
        if draw is None:
            draw = (
                self._weight_shape
                if len(self._weight_shape) <= 2
                else (math.prod(self._weight_shape[:-1]), self._weight_shape[-1])
            )
        self._draw_shape = tuple(int(axis) for axis in draw)
        super().__init__(
            name,
            initializer=initializer,
            dim=self._weight_shape,
            trainable=trainable,
        )

    def build_layer(self):
        return WeightImpl(
            value_shape=self.shape.tuple,
            weight_shape=self._weight_shape,
            draw_shape=self._draw_shape,
            initializer=self._initializer,
            trainable=self._trainable,
            name=self.name,
        )

    def get_config(self):
        # Declared by its shape and the shape its initializer draws, which
        # take the place of a Parameter's value and axes.
        config = super().get_config()
        for key in ("value", "dim", "tw", "seq"):
            del config[key]
        config.update({"shape": self._weight_shape, "draw": self._draw_shape})
        return config


class _ParameterWeights(Layer):
    """A layer computing with weights held by Parameters, its predecessors
    after its input.

    The weights are named by ``_labels``, in the order its Keras layer reads
    them: ``[x, *weights]``. ``weights`` sets each one: a Parameter, used as
    given, or an initializer, by name or object, from which the layer makes
    a :class:`Weight` of the shape its input asks for, the first time it is
    called, and shares it with every later call. ``True`` takes the layer's
    default initializer for it, ``_defaults``, and ``False`` leaves it out.

    A subclass gives the shapes of its weights for an input with
    :meth:`_weight_shapes`, and the shapes their initializers draw with
    :meth:`_draw_shapes`, one for each label.
    """

    _labels: tuple[str, ...] = ("kernel", "bias")
    _defaults: dict = {"kernel": "glorot_uniform", "bias": "zeros"}

    def __init__(self, name, weights: dict, **properties):
        self._settings = dict(weights)
        # Made, or taken as given, on the first call.
        self._weights: dict[str, Parameter] | None = None
        super().__init__(name=name, **properties)

    def _weight_shapes(self, x) -> tuple[tuple[int, ...], ...]:
        raise NotImplementedError

    def _draw_shapes(self, shapes: tuple) -> tuple:
        """The shapes the initializers of weights of ``shapes`` draw, one for
        each; None for the Weight's own default."""
        return (None,) * len(shapes)

    def _parameters(self, x) -> list[Parameter]:
        """The weights for the input ``x``, made on the first call. Each must
        have the shape the input asks for, axes of size 1 aside: they leave
        the layout of the values as it is."""
        shapes = dict(zip(self._labels, self._weight_shapes(x)))
        if self._weights is None:
            draws = dict(zip(self._labels, self._draw_shapes(tuple(shapes.values()))))
            self._weights = {
                label: self._weight(
                    label, self._settings[label], shapes[label], draws[label]
                )
                for label in self._labels
                if self._settings[label] is not False
            }
        for label, weight in self._weights.items():
            if _without_ones(weight.shape.tuple) != _without_ones(shapes[label]):
                raise ValueError(
                    f"{self.name}: {label} must have the shape "
                    f"{tuple(shapes[label])}, axes of size 1 aside, got "
                    f"{weight.shape.tuple}."
                )
        return list(self._weights.values())

    def _weight(self, label, setting, shape, draw) -> Parameter:
        if isinstance(setting, Parameter):
            return setting
        if setting is True:
            setting = self._defaults[label]
        return Weight(
            f"{self.name}_{label}", shape=shape, initializer=setting, draw=draw
        )

    def __call__(self, inputs) -> Any:
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # Before the weights are sized on it.
        _refuse_non_streams(self, inputs)
        if self._one_stream and len(inputs) > 1:
            # Applied to each stream, with the same weights.
            return [self([x]) for x in inputs]
        # The weights are predecessors like the input, but configured on the
        # layer, so they are appended here and reconnected when reloading.
        # They take the place of any further stream, which would otherwise be
        # dropped without a word.
        if len(inputs) > 1:
            raise ValueError(
                f"{self.name}: {type(self).__name__} reads one stream, got "
                f"{len(inputs)}. Apply one to each, or join them first, "
                "with Concatenate or TimeConcatenate."
            )
        # Before the weights are sized on it.
        _refuse_uncalled(inputs)
        if not inputs:
            raise ValueError(
                f"{self.name}: {type(self).__name__} reads one stream, got "
                "none. Apply it to a stream first."
            )
        return self._apply([inputs[0], *self._parameters(inputs[0])])

    def get_config(self):
        # The weights are saved with the graph, as the predecessors they are:
        # an argument that sets one only says whether it is there.
        config = super().get_config()
        for label in self._labels:
            if label in config:
                config[label] = config[label] is not False
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        preds = preds or []
        layer = super().from_config(config)
        # The weights it has come back as its predecessors after its input,
        # in the order of their labels.
        weights = iter(preds[1:])
        for label in cls._labels:
            if layer._settings[label] is not False:
                layer._settings[label] = next(weights)
        return layer(preds[:1]) if preds else layer

    def _weight_nodes(self) -> list[Parameter]:
        return cast(list[Parameter], self._preds[1:]) if self._preds else []

    def _weight_value(self, label: str):
        """The variable of the Parameter holding the weight ``label``, in
        that Parameter's shape; None without it or before the model is
        built."""
        present = [name for name in self._labels if self._settings[name] is not False]
        weight = dict(zip(present, self._weight_nodes())).get(label)
        return None if weight is None else weight.param

    @property
    def kernel(self):
        """The variable of the Parameter holding the kernel, in that
        Parameter's shape; None before the model is built."""
        return self._weight_value("kernel")

    @property
    def bias(self):
        """The variable of the Parameter holding the bias, in that
        Parameter's shape; None without a bias or before the model is
        built."""
        return self._weight_value("bias")


def _without_ones(shape):
    """``shape`` without its axes of size 1."""
    return tuple(axis for axis in shape if axis != 1)


def _check_weight_size(layer_name, label, weight_shape, expected):
    """Raise unless a weight tensor ``(batch, ...)`` holds the values of a
    weight of shape ``expected``."""
    held = math.prod(int(axis) for axis in weight_shape[1:])
    if held != math.prod(expected):
        raise ValueError(
            f"{layer_name}: its {label} holds {held} values where the input asks "
            f"for {tuple(expected)}. The weights of a layer fit the input shape "
            "they were made for."
        )
