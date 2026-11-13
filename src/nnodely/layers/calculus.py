"""Calculus with respect to time: the integral and the derivative of a Stream.

They are configured first and then called on the Stream, like every other
layer::

    IntegrateStep(dt=0.1)(rate)                 # integral of rate   over its window
    Integrate(dt=0.1)(rate)                     # integral of rate   step by step
    DerivativeStep(dt=0.1)(fun)                 # d fun / d t        over its window
    Derivative(dt=0.1)(fun)                     # d fun / d t        step by step

Each integral is the inverse of the derivative beside it. ``IntegrateStep``
integrates along the rate's own window and keeps its length: the rate sample
of an interval integrates into the value at the end of that interval, and
``init`` is the value the signal had just before the window.
``DerivativeStep`` is a causal finite difference along ``fun``'s own window
that keeps its length: one derivative per sample, the oldest one reading the
``init`` condition. ``Integrate`` and ``Derivative`` are the same rules across
the steps of a ModelyLoop. The derivative with respect to an Input is
:class:`~nnodely.layers.gradient.Gradient`.
"""

from __future__ import annotations

import keras
import numpy as np

from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank, slice_axis
from nnodely.core.stream import Stream
from nnodely.layers.input import (
    Input,
    _check_history,
    _declared_seq,
    _validate_init,
)


# ======================================================================
# Integration
# ======================================================================
#: Quadrature rules, by the name the model writes. "euler" is the same
#: rectangular rule under the name it has when the window is one sample and
#: the layer is one step of a recurrence, as in ``Ode``.
_METHOD_RULE = {
    "euler": "rectangular",
    "rectangular": "rectangular",
    "trapezoidal": "trapezoidal",
}

#: What a layer of each rule is called on before the rate: the value of the
#: integral before the window, and the rate before it, which the trapezoid
#: averages with the first one.
_HISTORY = {
    "rectangular": ("init",),
    "trapezoidal": ("init", "init_rate"),
}


def _validate_dt(dt, owner: str = "IntegrateStep") -> float:
    """The time step, always given explicitly."""
    if dt is None:
        raise ValueError(
            f"{owner}: dt is required. Pass the time step of the signal, for "
            f"example {owner}(dt=0.01)."
        )
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError(f"{owner}: dt must be a number, got {type(dt).__name__}.")
    if dt <= 0:
        raise ValueError(f"{owner}: dt must be positive, got {dt}.")
    return float(dt)


def _validate_method(method, owner: str) -> str:
    if isinstance(method, Stream):
        raise TypeError(
            f"{owner} is configured first and called on the rate, like every "
            f"other layer: {owner}(method=..., dt=...)(rate)."
        )
    if method not in _METHOD_RULE:
        raise ValueError(
            f"{owner}: method must be one of {sorted(_METHOD_RULE)}, got {method!r}."
        )
    return method


def _history_and_signal(owner: str, inputs, history: tuple[str, ...]) -> list:
    """The predecessors of a layer called as ``layer(signal)`` or
    ``layer(*history, signal)``: the signal, then what it starts from - up to
    the ``history`` it reads, in that order - a number made a Constant."""
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        inputs = tuple(inputs[0])
    if not 1 <= len(inputs) <= len(history) + 1:
        given = ", ".join((*history, "signal"))
        raise TypeError(
            f"{owner} is called on the signal, or on {given} - the samples "
            f"before the first, then the signal - {owner}(...)({given}); got "
            f"{len(inputs)} arguments."
        )
    *before, signal = inputs
    values = []
    for value, name in zip(before, history):
        if not isinstance(value, Stream):
            value = Stream._coerce_operand(_validate_init(owner, value, name))
        values.append(value)
    return [signal, *values]


def _quadrature_matrix(rule: str, window_length: int, dt: float) -> np.ndarray:
    """The quadrature as the matrix that applies it to a whole window at once.

    Row ``interval`` of the increments holds the weight each rate sample has in
    the step across that interval; summing the increments up to ``i`` gives the
    value at sample ``i``, so the running integral is a single matmul - the
    same shape of operation as the derivative's banded stencil, and one the
    exporters convert without a scan.

    Sample ``i`` of the rate is the rate *of the interval ending at* ``i``,
    which is the interval a backward difference produced it from. The
    trapezoidal rule averages it with the previous sample; the previous
    sample of the first interval is the rate before the window, which the
    layer adds with the initial condition, ``dt / 2`` of it to every sample.
    """
    increments = np.zeros((window_length, window_length), dtype=np.float64)
    for interval in range(window_length):
        if rule == "trapezoidal":
            increments[interval, interval] += dt / 2.0
            if interval > 0:
                increments[interval, interval - 1] += dt / 2.0
        else:
            increments[interval, interval] += dt
    return np.cumsum(increments, axis=0).T.astype(np.float32)


@keras.saving.register_keras_serializable(package="nnodely")
class IntegrateImpl(keras.layers.Layer):
    """Running integral of a window, one integrated sample per rate sample.

    Called on the rate, then the initial condition and the rate before the
    window, if given.
    """

    def __init__(self, dt, rule, axis, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dt = float(dt)
        self.rule = rule
        self.axis = int(axis)

    def build(self, input_shape):
        shapes = (
            input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # type: ignore[list-item]
        )
        window_length = int(shapes[0][self.axis])
        operator = _quadrature_matrix(self.rule, window_length, self.dt)
        self.operator = self.add_weight(
            name="operator",
            shape=operator.shape,
            initializer=keras.initializers.Constant(operator),  # type: ignore[arg-type]
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, inputs):
        x, *history = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        tw_axis = self.axis

        result = keras.ops.tensordot(x, self.operator, axes=[[tw_axis], [0]])  # type: ignore[arg-type]
        # tensordot appends the window axis; put it back where tw belongs.
        # The axis is addressed from the front because a negative index
        # survives tracing as a negative permutation, which ONNX rejects.
        last_axis = len(result.shape) - 1
        if last_axis != tw_axis:
            result = keras.ops.moveaxis(result, last_axis, tw_axis)

        # An initial condition may leave out the sequence axes, and a scalar
        # one the dim size: adding it broadcasts along both.
        rank = len(result.shape)
        if history:
            result = result + expand_to_rank(history[0], rank)
        if len(history) > 1:
            result = result + (self.dt / 2.0) * expand_to_rank(history[1], rank)
        return result

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            return tuple(input_shape[0])
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dt": self.dt,
                "rule": self.rule,
                "axis": self.axis,
            }
        )
        return config


class IntegrateStep(Layer):
    r"""Running integral of a rate along its window: one value per sample.

    Syntax
    ------
    ``IntegrateStep(dt=dt)(rate)``
        The integral of ``rate`` at every sample of its window, from zero.
    ``IntegrateStep(dt=dt)(init, rate)``
        The same, from ``init``, the value of the integral before the window.
    ``IntegrateStep(method="trapezoidal", dt=dt)(init, init_rate, rate)``
        The trapezoidal rule, with ``init_rate`` the rate before the window.

    The window keeps its length: sample ``i`` of the result is the integral
    at the end of the interval of sample ``i`` of the rate. On a window of
    one sample, the layer is one integration step, the state update
    ``IntegrateStep(dt=dt)(v, a)``, :math:`v + \Delta t \, a`. On a longer
    window, it gives the whole integrated trajectory in one pass, with no
    rollout.

    For one rate sample per step of a :class:`ModelyLoop` or a :class:`Loop`
    body, use :class:`Integrate`, which keeps the integral from one step to
    the next.

    Parameters
    ----------
    method : str, default "euler"
        The integration rule:

        * ``"euler"``: the rectangle of each sample, its rate times ``dt``.
        * ``"rectangular"``: the same rule as ``"euler"``.
        * ``"trapezoidal"``: the trapezoid between each sample and the one
          before it.

        Only ``"euler"`` is also a method of :func:`Ode`, with the same rule.
    dt : float
        The time between two samples, a positive number. It is required:
        nothing infers it from the data. The integral is in the unit of
        ``rate`` times the unit of ``dt``.
    name : str, optional
        The name of the layer. The default is ``IntegrateStep`` followed by a
        number.

    Call arguments
    --------------
    init : Stream or float, optional
        The value of the integral before the window. It is zero when left
        out. It is one sample: a number, a :class:`Parameter` or a
        :class:`Constant` without a window, or a stream with ``T=1``. Its
        features are a scalar or those of ``rate``, and its sequence is none
        or that of ``rate``.
    init_rate : Stream or float, optional
        The rate before the window, for ``"trapezoidal"`` only, of the same
        forms as ``init``. It is zero when left out, a rate at rest before
        the window.
    rate : Stream
        A stream with a window, ``Shape(S, T, D)``.

    Returns
    -------
    Stream
        The shape of ``rate``: ``Shape(S, T, D)`` gives ``Shape(S, T, D)``.
        For example, ``Input("a", tw=50)`` gives
        ``Shape(S=(), T=50, D=(1,))``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the three ("method must be one of"), if
        ``dt`` is missing or not positive ("dt is required", "dt must be
        positive"), if ``rate`` has no window ("acts on the window of its
        stream"), or if ``init`` or ``init_rate`` is not one sample of the
        features and sequence of ``rate`` ("init holds the samples before
        the first").
    TypeError
        If ``dt`` is not a number, if the layer is called before it is
        configured, ``IntegrateStep(rate)``, or if it is called on more
        streams than its method reads ("is called on the signal, or on
        init, signal").

    See Also
    --------
    Integrate : The same integral, one step at a time.
    DerivativeStep : The time derivative along a window, its inverse.
    Ode : One step of an ODE, with Runge-Kutta methods.

    Notes
    -----
    With :math:`r_0, \dots, r_{T-1}` the rate from the oldest sample to the
    newest, :math:`y_{-1}` = ``init`` and :math:`r_{-1}` = ``init_rate``,
    sample :math:`i` of the result is, with ``"euler"`` and
    ``"rectangular"``:

    .. math:: y_i = y_{i-1} + \Delta t \, r_i

    and with ``"trapezoidal"``:

    .. math:: y_i = y_{i-1} + \frac{\Delta t}{2} \left(r_{i-1} + r_i\right)

    Sample :math:`r_i` is the rate of the interval that ends at sample
    :math:`i`, the interval the backward difference of
    :class:`DerivativeStep` computes it from. So the Euler rule inverts
    ``DerivativeStep`` exactly, with the same ``init``. The trapezoidal rule
    is second-order accurate on a smooth rate instead.

    There is no ``"rk4"`` and no ``"heun"``. These methods evaluate the
    dynamics again at a predicted state, which an integral of a given rate
    cannot do: use :func:`Ode`. The integral is a linear map of the rate and
    of the initial values, so gradients flow through it to all of them.

    Basic Examples
    --------------
    The velocity of an acceleration sampled every 0.5 s, from rest:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import DerivativeStep, Input, IntegrateStep, Modely, Output
    >>> a = Input("a", tw=4)
    >>> v = IntegrateStep(dt=0.5)(a)
    >>> model = Modely("velocity", inputs=[a], outputs=[Output("v", v)]).build()
    >>> window = np.array([[[2.0], [4.0], [6.0], [8.0]]])
    >>> keras.ops.convert_to_numpy(model({"a": window})["v"]).ravel()
    array([ 1.,  3.,  6., 10.], dtype=float32)

    Scope
    -----
    Every feature and every step of a sequence is integrated along its own
    window:

    >>> IntegrateStep(dt=0.5)(Input("w", dim=3, seq=5, tw=4)).shape
    Shape(S=(5,), T=4, D=(3,))

    On a window of one sample, the layer is one state update,
    :math:`v + \Delta t \, a`:

    >>> velocity = Input("velocity", tw=1)
    >>> acceleration = Input("acceleration", tw=1)
    >>> IntegrateStep(dt=0.5)(velocity, acceleration).shape
    Shape(S=(), T=1, D=(1,))

    Options
    -------
    ``init`` is the value before the window:

    >>> model = Modely(
    ...     "from_one", inputs=[a], outputs=[Output("v", IntegrateStep(dt=0.5)(1.0, a))]
    ... ).build()
    >>> keras.ops.convert_to_numpy(model({"a": window})["v"]).ravel()
    array([ 2.,  4.,  7., 11.], dtype=float32)

    ``method="trapezoidal"`` averages each rate with the one before it, zero
    before the window unless ``init_rate`` gives it. The first values are
    :math:`1 + 0.25 \cdot (0 + 2)` and :math:`1 + 0.25 \cdot (2 + 2)`:

    >>> at_rest = IntegrateStep(method="trapezoidal", dt=0.5)(1.0, a)
    >>> moving = IntegrateStep(method="trapezoidal", dt=0.5)(1.0, 2.0, a)
    >>> model = Modely(
    ...     "trapezoids", inputs=[a],
    ...     outputs=[Output("at_rest", at_rest), Output("moving", moving)],
    ... ).build()
    >>> result = model({"a": window})
    >>> keras.ops.convert_to_numpy(result["at_rest"]).ravel()
    array([1.5, 3. , 5.5, 9. ], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["moving"]).ravel()
    array([2. , 3.5, 6. , 9.5], dtype=float32)

    ``name`` names the layer:

    >>> IntegrateStep(dt=0.5, name="speed")(a).name
    'speed'

    Applications
    ------------
    The position of a body from a window of 50 accelerations, in one pass:

    >>> acceleration_window = Input("acceleration_window", tw=50)
    >>> v_window = IntegrateStep(dt=0.01)(acceleration_window)
    >>> x_window = IntegrateStep(dt=0.01)(v_window)
    >>> x_window.shape
    Shape(S=(), T=50, D=(1,))

    The state update of a model declared one step at a time, where
    ``velocity`` holds the value at the previous step:

    >>> v_next = velocity.feedback(IntegrateStep(dt=0.01)(velocity, acceleration))

    Properties & Relations
    ----------------------
    The Euler rule inverts :class:`DerivativeStep` with the same ``init``:

    >>> x = Input("x", tw=4)
    >>> back = IntegrateStep(dt=0.5)(3.0, DerivativeStep(dt=0.5)(3.0, x))
    >>> model = Modely("inverse", inputs=[x], outputs=[Output("x", back)]).build()
    >>> samples = np.array([[[1.0], [2.0], [4.0], [7.0]]])
    >>> keras.ops.convert_to_numpy(model({"x": samples})["x"]).ravel()
    array([1., 2., 4., 7.], dtype=float32)

    ``"rectangular"`` is ``"euler"``:

    >>> model = Modely(
    ...     "rectangles", inputs=[a],
    ...     outputs=[Output("v", IntegrateStep(method="rectangular", dt=0.5)(a))],
    ... ).build()
    >>> keras.ops.convert_to_numpy(model({"a": window})["v"]).ravel()
    array([ 1.,  3.,  6., 10.], dtype=float32)

    Possible Issues
    ---------------
    The rate needs a window. Declare one on the Input:

    >>> IntegrateStep(dt=0.5)(Input("thrust"))
    Traceback (most recent call last):
        ...
    ValueError: ...acts on the window of its stream, and 'thrust' has none...
    >>> IntegrateStep(dt=0.5)(Input("thrust", tw=1)).shape
    Shape(S=(), T=1, D=(1,))

    Only the trapezoidal rule reads the rate before the window:

    >>> IntegrateStep(dt=0.5)(1.0, 2.0, a)
    Traceback (most recent call last):
        ...
    TypeError: IntegrateStep is called on the signal, or on init, signal...got 3 arguments.

    Runge-Kutta methods belong to :func:`Ode`:

    >>> IntegrateStep(method="rk4", dt=0.5)
    Traceback (most recent call last):
        ...
    ValueError: IntegrateStep: method must be one of ['euler', 'rectangular', 'trapezoidal'], got 'rk4'.

    .. versionadded:: 2.0
    """

    def __init__(
        self,
        method: str = "euler",
        dt: float | None = None,
        name=None,
    ):
        self._method = _validate_method(method, "IntegrateStep")
        self._dt = _validate_dt(dt)
        super().__init__(name=name, method=self._method, dt=self._dt)

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------
    def __call__(self, *inputs):
        # The samples before the window are predecessors like any other,
        # after the rate, so that they are reconnected the same way when
        # reloading.
        history = _HISTORY[_METHOD_RULE[self._method]]
        return super().__call__(_history_and_signal("IntegrateStep", inputs, history))

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def output_shape(self, *inputs):
        rate, *history = inputs
        if not isinstance(rate, Stream):
            raise TypeError(
                f"{self.name}: IntegrateStep expects a Stream input, got "
                f"{type(rate).__name__}."
            )
        self._needs_window(rate)
        for value, name in zip(history, _HISTORY[_METHOD_RULE[self._method]]):
            _check_history(self.name, value, rate, 1, name)
        # One integrated sample per rate sample: the result stays aligned with
        # the signal it came from.
        return rate.shape

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def build_layer(self):
        return IntegrateImpl(
            dt=self._dt,
            rule=_METHOD_RULE[self._method],
            axis=self.shape.tw_axis,
            name=self.name,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The samples before the window were serialized after the rate, so
        # they are rebuilt with the rest of the graph.
        layer = super().from_config(config)
        return layer(*preds[1:], *preds[:1]) if preds else layer


class Integrate:
    r"""Integral of a rate one step at a time, in a ModelyLoop or a Loop body.

    Syntax
    ------
    ``Integrate(dt)(rate)``
        The integral of ``rate`` at the end of each step, from zero.
    ``Integrate(dt, init=y0)(rate)``
        The same, from ``y0``, the integral before the first step.
    ``Integrate(dt, method="trapezoidal", init=y0, init_rate=r0)(rate)``
        The trapezoidal rule, with ``r0`` the rate before the first step.

    The rate is one sample per step, ``T=1``. The integral is a state: an
    Input named ``<name>_<rate>``, fed back with the result. The
    :class:`ModelyLoop`, or the :class:`Loop` whose body holds the layer,
    feeds it back with its other states, in one rollout. The trapezoidal rule
    also keeps the rate of the previous step, in the state
    ``<name>_<rate>_rate``. These states are not your Inputs:
    :attr:`Modely.inputs` lists them as not declared.

    The rules and the results are those of :class:`IntegrateStep` on a window
    of the same samples. Built as a plain :class:`Modely`, which has no steps,
    the layer reads ``init`` and ``init_rate`` in place of the states.

    ``dt`` comes first here, and ``method`` first in :class:`IntegrateStep`:
    pass both by name to avoid mistakes.

    Parameters
    ----------
    dt : float
        The time between two steps, a positive number. It is required.
    method : str, default "euler"
        The integration rule, as in :class:`IntegrateStep`:

        * ``"euler"`` or ``"rectangular"``: the rectangle of each step.
        * ``"trapezoidal"``: the trapezoid between each step and the one
          before it.
    init : Stream or float, optional
        The integral before the first step. It is zero when left out. It can
        be a number, a :class:`Parameter`, a :class:`Constant`, or an Input of
        one sample, whose first step is read. Its features are a scalar or
        those of ``rate``, and its sequence is none or that of ``rate``. It
        is the only initial value of the state: a Loop cannot give it
        another.
    init_rate : Stream or float, optional
        The rate before the first step, for ``"trapezoidal"`` only, of the
        same forms as ``init``. It is zero when left out, a rate at rest
        before the start.
    name : str, optional
        The name of the stream the layer returns. The states are named after
        it. A second call of the same object integrates with a state of its
        own: its result is named ``<name>_2``. The default name is
        ``Integrate`` followed by a number.

    Call arguments
    --------------
    rate : Stream
        One sample per step, ``Shape(S, 1, D)``. The call takes it alone or in
        a list of one.

    Returns
    -------
    Stream
        The shape of ``rate``, ``Shape(S, 1, D)``: the integral at the end of
        this step. For example, ``Input("a", tw=1)`` gives
        ``Shape(S=(), T=1, D=(1,))``. As an output of a ModelyLoop it is a
        trajectory, one value per step.

    Raises
    ------
    ValueError
        If ``dt`` is missing or not positive, if ``method`` is not one of the
        three ("method must be one of"), if ``init_rate`` is given to another
        method than ``"trapezoidal"``, if ``rate`` has no window ("acts on the
        window of its stream") or more than one sample ("takes one rate sample
        per step"), or if ``init`` or ``init_rate`` is not one sample of the
        features and sequence of ``rate``.
    TypeError
        If ``dt`` is not a number, if ``init`` or ``init_rate`` is not a
        stream, a number or None, or if ``rate`` is not a stream.

    See Also
    --------
    IntegrateStep : The same integral along a window.
    Derivative : The derivative one step at a time, its inverse.
    ModelyLoop : A model declared one step at a time.
    Loop : The rollout of a one-step model.
    Ode : One step of an ODE, with Runge-Kutta methods.

    Notes
    -----
    At step :math:`k`, with :math:`y_{-1}` = ``init`` and :math:`r_{-1}` =
    ``init_rate``, the result is, with ``"euler"`` and ``"rectangular"``:

    .. math:: y_k = y_{k-1} + \Delta t \, r_k

    and with ``"trapezoidal"``:

    .. math:: y_k = y_{k-1} + \frac{\Delta t}{2} \left(r_{k-1} + r_k\right)

    Basic Examples
    --------------
    The velocity of an acceleration given one sample per step, every 0.5 s,
    from rest:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Integrate, IntegrateStep, Loop, Modely, ModelyLoop, Output
    >>> a = Input("a", tw=1)
    >>> v = Integrate(dt=0.5, name="v")(a)
    >>> model = ModelyLoop("velocity", inputs=[a], outputs=[Output("v", v)]).build()
    >>> steps = np.array([2.0, 4.0, 6.0, 8.0]).reshape(1, 4, 1, 1)
    >>> keras.ops.convert_to_numpy(model({"a": steps})["v"]).ravel()
    array([ 1.,  3.,  6., 10.], dtype=float32)

    The state is an Input of the model, named after the layer and the rate:

    >>> model.inputs[1].name
    'v_a'

    Options
    -------
    ``init`` can be an Input of the step: its first step is read, here 1.0.

    >>> v0 = Input("v0", tw=1)
    >>> v = Integrate(dt=0.5, init=v0)(a)
    >>> model = ModelyLoop("from_v0", inputs=[a, v0], outputs=[Output("v", v)]).build()
    >>> start = np.array([1.0, 9.0, 9.0, 9.0]).reshape(1, 4, 1, 1)
    >>> keras.ops.convert_to_numpy(model({"a": steps, "v0": start})["v"]).ravel()
    array([ 2.,  4.,  7., 11.], dtype=float32)

    ``method="trapezoidal"`` averages each rate with the one before it, which
    ``init_rate`` gives at the first step. The first value is
    :math:`1 + 0.25 \cdot (2 + 2)`:

    >>> v = Integrate(dt=0.5, method="trapezoidal", init=1.0, init_rate=2.0)(a)
    >>> model = ModelyLoop("trapezoid", inputs=[a], outputs=[Output("v", v)]).build()
    >>> keras.ops.convert_to_numpy(model({"a": steps})["v"]).ravel()
    array([2. , 3.5, 6. , 9.5], dtype=float32)

    ``name`` names the result:

    >>> Integrate(dt=0.5, name="speed")(a).name
    'speed'

    Applications
    ------------
    A mass pushed by a force, in a one-step model rolled out by a
    :class:`Loop`. The Loop finds the state of the layer in its body, and
    needs no ``feedback``:

    >>> force = Input("force", tw=1)
    >>> speed = Integrate(dt=0.5, name="speed")(force / 2.0)
    >>> body = Modely("mass", inputs=[force], outputs=[Output("speed", speed)])
    >>> forces = Input("forces", seq=3, tw=1)
    >>> rollout = Modely(
    ...     "push", inputs=[forces], outputs=[Output("speed", Loop(f=body)(forces))]
    ... ).build()
    >>> pushes = np.array([4.0, 8.0, 12.0]).reshape(1, 3, 1, 1)
    >>> keras.ops.convert_to_numpy(rollout({"forces": pushes})["speed"]).ravel()
    array([1., 3., 6.], dtype=float32)

    Properties & Relations
    ----------------------
    The same samples give the same values as :class:`IntegrateStep` on a
    window:

    >>> w = Input("w", tw=4)
    >>> v_window = IntegrateStep(dt=0.5)(w)
    >>> model = Modely("window", inputs=[w], outputs=[Output("v", v_window)]).build()
    >>> keras.ops.convert_to_numpy(model({"w": steps.reshape(1, 4, 1)})["v"]).ravel()
    array([ 1.,  3.,  6., 10.], dtype=float32)

    Built as a plain :class:`Modely`, the step reads ``init`` as the integral
    before it: :math:`1 + 0.5 \cdot 2`.

    >>> v = Integrate(dt=0.5, init=1.0)(a)
    >>> model = Modely("one_step", inputs=[a], outputs=[Output("v", v)]).build()
    >>> keras.ops.convert_to_numpy(model({"a": np.array([[[2.0]]])})["v"])
    array([[[2.]]], dtype=float32)

    Possible Issues
    ---------------
    The layer reads one sample per step. Integrate a window with
    :class:`IntegrateStep`:

    >>> Integrate(dt=0.5)(Input("long", tw=4))
    Traceback (most recent call last):
        ...
    ValueError: ...Integrate takes one rate sample per step, but the rate carries 4 time samples...

    Only the trapezoidal rule reads ``init_rate``:

    >>> Integrate(dt=0.5, init_rate=1.0)
    Traceback (most recent call last):
        ...
    ValueError: Integrate: init_rate is the rate before the first step, which only the trapezoidal method reads, not 'euler'.

    .. versionadded:: 2.0
    """

    def __init__(
        self,
        dt: float,
        method: str = "euler",
        init: Stream | float | None = None,
        init_rate: Stream | float | None = None,
        name: str | None = None,
    ):
        from nnodely.core.dag import next_name

        self._dt = _validate_dt(dt, "Integrate")
        self._method = _validate_method(method, "Integrate")
        self._init = _validate_init("Integrate", init)
        if init_rate is not None and _METHOD_RULE[method] != "trapezoidal":
            raise ValueError(
                "Integrate: init_rate is the rate before the first step, which "
                f"only the trapezoidal method reads, not {method!r}."
            )
        self._init_rate = _validate_init("Integrate", init_rate, "init_rate")
        self.name = next_name("Integrate") if name is None else name
        self._calls = 0

    def __call__(self, rate):
        if isinstance(rate, (list, tuple)):
            if len(rate) != 1:
                raise ValueError(
                    f"{self.name}: Integrate integrates one rate, got {len(rate)}."
                )
            rate = rate[0]
        if not isinstance(rate, Stream):
            raise TypeError(
                f"{self.name}: Integrate expects a Stream, got {type(rate).__name__}."
            )
        if rate.tw is None:
            raise ValueError(
                f"{self.name}: Integrate acts on the window of its stream, and "
                f"{rate.name!r} has none, {rate.shape}. Declare the Input with a "
                "window, tw=1 for one sample per step."
            )
        if rate.tw != 1:
            raise ValueError(
                f"{self.name}: Integrate takes one rate sample per step, but the "
                f"rate carries {rate.tw} time samples. Integrate a time window "
                "with IntegrateStep."
            )
        for value, name in ((self._init, "init"), (self._init_rate, "init_rate")):
            if isinstance(value, Stream):
                _check_history(self.name, value, rate, 1, name)

        # Every call integrates a rate of its own, so each has its own state.
        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"

        def hidden(name):
            node = Input(name, dim=rate.dim, seq=_declared_seq(rate), tw=1)
            node._hidden = True
            return node

        # Named after the rate it integrates, as the Input a hand-written
        # feedback would read it from.
        integral = hidden(f"{call_name}_{rate.name}")
        history = [integral]
        if _METHOD_RULE[self._method] == "trapezoidal":
            previous = hidden(f"{call_name}_{rate.name}_rate")
            previous._feed_back(rate, init=self._init_rate)
            history.append(previous)
        value = IntegrateStep(method=self._method, dt=self._dt, name=call_name)(
            *history, rate
        )
        return integral._feed_back(value, init=self._init)


# ======================================================================
# Differentiation
# ======================================================================
# ----------------------------------------------------------------------
# Finite-difference stencils
# ----------------------------------------------------------------------
def _stencil_coefficients(poly_order: int, dt: float) -> np.ndarray:
    """Weights of the backward difference of order ``poly_order``.

    They give the slope, at the newest sample, of the polynomial of degree
    ``poly_order`` through the ``poly_order + 1`` newest samples - the
    backward differentiation formula: ``(-1, 1)/dt`` for 1, ``(1, -4, 3)/(2
    dt)`` for 2, ``(-2, 9, -18, 11)/(6 dt)`` for 3, each exact on a polynomial
    of its degree. The weights are ordered oldest -> newest, as the samples of
    a window are.
    """
    offsets = np.arange(-poly_order, 1, dtype=np.float64)
    design = np.vander(offsets, poly_order + 1, increasing=True)
    # Row 1 of the inverse gives the polynomial's coefficient of t: its slope
    # at the newest sample, t = 0.
    return np.linalg.inv(design)[1] / float(dt)


def _difference_matrix(coefficients, window_length: int) -> np.ndarray:
    """The stencil as the matrix that applies it to a whole window at once.

    Sliding the stencil over a window is a banded matrix product - the
    discrete differentiation operator of the window - so the layer runs as a
    single matmul rather than a scan, and differentiates (and exports) like
    any other linear operation.
    """
    past = len(coefficients) - 1
    matrix = np.zeros((past + window_length, window_length), dtype=np.float32)
    for column in range(window_length):
        for offset, coefficient in enumerate(coefficients):
            matrix[column + offset, column] = coefficient
    return matrix


# ----------------------------------------------------------------------
# Finite difference with respect to time
# ----------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="nnodely")
class TimeDerivativeImpl(keras.layers.Layer):
    """Causal finite difference of a whole window, one derivative per sample.

    The stencil needs ``len(coefficients) - 1`` samples of history to produce
    the oldest derivative of the window; they come from the initial condition
    (a second input) or are taken as zero. Prepending them and applying the
    banded operator matrix makes the layer a single matmul over the padded
    window.
    """

    def __init__(
        self,
        coefficients,
        axis: int,
        init_tw: int = 1,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.axis = int(axis)
        self.init_tw = int(init_tw)

    @property
    def past(self) -> int:
        """Samples of history the stencil reads before the window's first."""
        return len(self.coefficients) - 1

    def build(self, input_shape):
        shapes = (
            input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # type: ignore[list-item]
        )
        window_length = int(shapes[0][self.axis])
        operator = _difference_matrix(self.coefficients, window_length)
        self.operator = self.add_weight(
            name="operator",
            shape=operator.shape,
            initializer=keras.initializers.Constant(operator),  # type: ignore[arg-type]
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, init = inputs[0], inputs[1] if len(inputs) > 1 else None
        else:
            x, init = inputs, None
        tw_axis = self.axis

        # One sample of zeros shaped like the window's first: it both stands
        # in for a missing initial condition and gives a given one the axes it
        # may be broadcasting over - a scalar initial condition carries no dim
        # size, and one may leave out the sequence axes.
        reference = keras.ops.zeros_like(slice_axis(x, tw_axis, slice(0, 1)))
        if init is None:
            history = reference
        else:
            history = expand_to_rank(init, len(reference.shape)) + reference
        if self.init_tw != self.past:
            history = keras.ops.repeat(history, self.past, axis=tw_axis)

        padded = keras.ops.concatenate([history, x], axis=tw_axis)
        result = keras.ops.tensordot(padded, self.operator, axes=[[tw_axis], [0]])  # type: ignore[arg-type]
        # tensordot appends the window axis; put it back where tw belongs.
        # The axis is addressed from the front because a negative index
        # survives tracing as a negative permutation, which ONNX rejects.
        last_axis = len(result.shape) - 1
        if last_axis == tw_axis:  # nothing follows tw, so it already is
            return result
        return keras.ops.moveaxis(result, last_axis, tw_axis)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            return tuple(input_shape[0])
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "coefficients": self.coefficients,
                "axis": self.axis,
                "init_tw": self.init_tw,
            }
        )
        return config


# ----------------------------------------------------------------------
# Symbolic layers
# ----------------------------------------------------------------------
class _FiniteDifference(Layer):
    """What :class:`DerivativeStep` and :class:`Derivative` share: the time
    step ``dt``, the backward difference of order ``poly_order``, and the
    initial condition ``init``, the sample(s) just before the first one
    differentiated.
    """

    def __init__(
        self,
        dt: float | None = None,
        poly_order: int = 1,
        name=None,
        **properties,
    ):
        layer = type(self).__name__
        self._dt = _validate_dt(dt, layer)
        if (
            isinstance(poly_order, bool)
            or not isinstance(poly_order, (int, np.integer))
            or poly_order < 1
        ):
            raise ValueError(
                f"{layer}: poly_order is the degree of the polynomial through "
                "the poly_order + 1 newest samples, a positive integer; got "
                f"{poly_order!r}."
            )
        self._poly_order = int(poly_order)
        super().__init__(
            name=name, dt=self._dt, poly_order=self._poly_order, **properties
        )

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------
    @property
    def past(self) -> int:
        """Samples of history the stencil needs before the first one."""
        return self._poly_order

    def _coefficients(self) -> np.ndarray:
        return _stencil_coefficients(self._poly_order, self._dt)

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def _check_function(self, function):
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: {type(self).__name__} expects a Stream input, got "
                f"{type(function).__name__}."
            )

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def call(self, xs):
        signature = self.input_signature(xs)
        stale = self._layer_signature is not None and self._layer_signature != signature
        if self._layer is None or stale:
            self._layer = self.build_layer()
        self._layer_signature = signature
        return self._layer(xs if len(xs) > 1 else xs[0])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The initial condition was serialized after the signal, so it is
        # rebuilt with the rest of the graph: the layer is called on it, then
        # on the signal.
        layer = super().from_config(config)
        return layer(*preds[1:], *preds[:1]) if preds else layer


class DerivativeStep(_FiniteDifference):
    r"""Time derivative of a stream along its window: one derivative per sample.

    Syntax
    ------
    ``DerivativeStep(dt)(x)``
        The derivative of ``x`` at every sample of its window, with ``x``
        zero before the window.
    ``DerivativeStep(dt)(init, x)``
        The same, with ``init`` the value of ``x`` before the window.
    ``DerivativeStep(dt, poly_order=n)(x)``
        A backward difference that reads the ``n`` samples before each one.
    ``DerivativeStep(dt)(DerivativeStep(dt)(x))``
        The second derivative.

    Each sample of the result is a backward finite difference: it reads that
    sample of ``x`` and the ones before it, never a later one. The window
    keeps its length, so the result stays aligned with ``x``. The first
    samples of the window read ``init`` in place of the samples before it.

    For one sample per step of a :class:`ModelyLoop` or a :class:`Loop`
    body, use :class:`Derivative`. For the derivative with respect to an
    Input, use :class:`Gradient`.

    Parameters
    ----------
    dt : float
        The time between two samples, a positive number. It is required:
        nothing infers it from the data. The derivative is in the unit of
        ``x`` per unit of ``dt``.
    poly_order : int, default 1
        The degree of the polynomial through the ``poly_order + 1`` newest
        samples: each derivative is its slope at the newest one. Any positive
        integer is accepted. The common ones are:

        * ``1``: :math:`(x_i - x_{i-1}) / \Delta t`, exact on a line.
        * ``2``: exact on a parabola.
        * ``3``: exact on a cubic.

        A higher order is more accurate on smooth signals, but amplifies
        noise more. Keep ``1`` for measured signals.
    name : str, optional
        The name of the layer. The default is ``DerivativeStep`` followed by
        a number.

    Call arguments
    --------------
    init : Stream or float, optional
        The value of ``x`` before the window. It is zero when left out. It
        can be:

        * a number, or a :class:`Parameter` or :class:`Constant` without a
          window: one sample;
        * a stream of one sample, ``T=1``, repeated for every sample the
          stencil reads;
        * a stream of ``poly_order`` samples, oldest first.

        Its features are a scalar or those of ``x``, and its sequence is none
        or that of ``x``. An Input needs a window, ``tw=1`` for one sample.
    x : Stream
        A stream with a window, ``Shape(S, T, D)``.

    Returns
    -------
    Stream
        The shape of ``x``: ``Shape(S, T, D)`` gives ``Shape(S, T, D)``. For
        example, a window of 4 samples of 3 features in a sequence of 5,
        ``Shape(S=(5,), T=4, D=(3,))``, gives the same shape.

    Raises
    ------
    ValueError
        If ``dt`` is missing or not positive ("dt is required", "dt must be
        positive"), if ``poly_order`` is not a positive integer, if ``x`` has
        no window ("acts on the window of its stream"), or if ``init`` has
        another number of samples, features or sequence ("init holds the
        samples before the first").
    TypeError
        If ``dt`` is not a number, if the layer is called before it is
        configured, ``DerivativeStep(x)``, or if it is called on more than
        two streams ("is called on the signal, or on init, signal").

    See Also
    --------
    Derivative : The same derivative, one step at a time.
    IntegrateStep : The running integral along a window, its inverse.
    Gradient : The derivative with respect to an Input.

    Notes
    -----
    With :math:`x_0, \dots, x_{T-1}` the window from the oldest sample to the
    newest, and :math:`p` = ``poly_order``, the samples
    :math:`x_{-p}, \dots, x_{-1}` come from ``init``. The three common
    orders, :math:`p` = 1, 2, 3, are:

    .. math::

        \dot x_i = \frac{x_i - x_{i-1}}{\Delta t}

        \dot x_i = \frac{3 x_i - 4 x_{i-1} + x_{i-2}}{2 \Delta t}

        \dot x_i = \frac{11 x_i - 18 x_{i-1} + 9 x_{i-2} - 2 x_{i-3}}{6 \Delta t}

    Order :math:`p` is the slope at :math:`x_i` of the polynomial of degree
    :math:`p` through :math:`x_{i-p}, \dots, x_i`, so it is exact on such a
    polynomial. White noise of standard deviation :math:`\sigma` on ``x``
    becomes noise of about :math:`1.4\sigma/\Delta t`,
    :math:`2.5\sigma/\Delta t` and :math:`3.8\sigma/\Delta t` on the
    derivative, for :math:`p` = 1, 2, 3. No order smooths: filter a noisy
    signal before differentiating it.

    The derivative is a linear map of ``x`` and ``init``, so gradients flow
    through it to both.

    Basic Examples
    --------------
    The velocity of a position sampled every 0.5 s. The first sample reads
    zero before the window:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import DerivativeStep, Input, IntegrateStep, Modely, Output
    >>> x = Input("x", tw=4)
    >>> v = DerivativeStep(dt=0.5)(x)
    >>> model = Modely("velocity", inputs=[x], outputs=[Output("v", v)]).build()
    >>> window = np.array([[[1.0], [2.0], [4.0], [7.0]]])
    >>> keras.ops.convert_to_numpy(model({"x": window})["v"]).ravel()
    array([2., 2., 4., 6.], dtype=float32)

    Scope
    -----
    Every feature and every step of a sequence is differentiated along its
    own window:

    >>> DerivativeStep(dt=0.5)(Input("w", dim=3, seq=5, tw=4)).shape
    Shape(S=(5,), T=4, D=(3,))

    ``init`` is a number, a stream of one sample, or a stream of
    ``poly_order`` samples, oldest first:

    >>> x0 = Input("x0", tw=1)
    >>> DerivativeStep(dt=0.5)(x0, x).shape
    Shape(S=(), T=4, D=(1,))

    Options
    -------
    ``init`` is the value before the window. From 1.0, the first derivative
    is :math:`(1 - 1) / 0.5`:

    >>> v = DerivativeStep(dt=0.5)(1.0, x)
    >>> model = Modely("from_one", inputs=[x], outputs=[Output("v", v)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": window})["v"]).ravel()
    array([0., 2., 4., 6.], dtype=float32)

    ``poly_order=2`` reads two samples before each one. Here ``init`` holds
    both, oldest first: 0.0, then 0.5. The first derivative is
    :math:`(3 \cdot 1 - 4 \cdot 0.5 + 0) / 1`:

    >>> past = Input("past", tw=2)
    >>> v2 = DerivativeStep(dt=0.5, poly_order=2)(past, x)
    >>> model = Modely("second", inputs=[x, past], outputs=[Output("v", v2)]).build()
    >>> result = model({"x": window, "past": np.array([[[0.0], [0.5]]])})
    >>> keras.ops.convert_to_numpy(result["v"]).ravel()
    array([1. , 2.5, 5. , 7. ], dtype=float32)

    ``name`` names the layer:

    >>> DerivativeStep(dt=0.5, name="velocity")(x).name
    'velocity'

    Applications
    ------------
    The velocity and the acceleration of a measured position, over the same
    window of 10 samples:

    >>> position = Input("position", tw=10)
    >>> velocity = DerivativeStep(dt=0.01)(position)
    >>> acceleration = DerivativeStep(dt=0.01)(velocity)
    >>> acceleration.shape
    Shape(S=(), T=10, D=(1,))

    Properties & Relations
    ----------------------
    :class:`IntegrateStep` with the same ``init`` gives ``x`` back, with
    ``poly_order=1``:

    >>> back = IntegrateStep(dt=0.5)(3.0, DerivativeStep(dt=0.5)(3.0, x))
    >>> model = Modely("inverse", inputs=[x], outputs=[Output("x", back)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": window})["x"]).ravel()
    array([1., 2., 4., 7.], dtype=float32)

    Possible Issues
    ---------------
    A stream without a window cannot be differentiated in time. Declare a
    window on the Input:

    >>> DerivativeStep(dt=0.5)(Input("speed"))
    Traceback (most recent call last):
        ...
    ValueError: ...acts on the window of its stream, and 'speed' has none...
    >>> DerivativeStep(dt=0.5)(Input("speed", tw=5)).shape
    Shape(S=(), T=5, D=(1,))

    ``init`` holds one sample, or the ``poly_order`` samples the stencil
    reads, and no other number:

    >>> DerivativeStep(dt=0.5)(Input("three", tw=3), x)
    Traceback (most recent call last):
        ...
    ValueError: ...init holds the samples before the first, a single sample; got 3.

    .. versionadded:: 2.0
    """

    def __init__(self, dt: float | None = None, poly_order: int = 1, name=None):
        # Its own arguments only: one it does not take is refused, rather than
        # kept as a setting that changes nothing.
        super().__init__(dt=dt, poly_order=poly_order, name=name)

    def __call__(self, *inputs):
        # The initial condition is a predecessor like any other, after the
        # signal, so that it is reconnected the same way when reloading.
        return super().__call__(
            _history_and_signal("DerivativeStep", inputs, ("init",))
        )

    def output_shape(self, *inputs):
        function, *history = inputs
        self._check_function(function)
        self._needs_window(function)
        if history:
            _check_history(self.name, history[0], function, self.past)
        # One derivative per sample: the window is preserved, so the result
        # stays aligned with the signal it came from.
        return function.shape

    def build_layer(self):
        init_tw = 1
        if len(self._preds) > 1:
            init_node = self._preds[1]
            if isinstance(init_node, Stream):
                # A value without a window is one sample.
                init_tw = init_node.shape.tw or 1
        return TimeDerivativeImpl(
            coefficients=self._coefficients(),
            axis=self.shape.tw_axis,  # type: ignore[arg-type]
            init_tw=init_tw,
            name=self.name,
        )


class Derivative(_FiniteDifference):
    r"""Time derivative of a stream one step at a time, in a ModelyLoop or a
    Loop body.

    Syntax
    ------
    ``Derivative(dt)(x)``
        The derivative of ``x`` at each step, with ``x`` zero before the
        first step.
    ``Derivative(dt, init=x0)(x)``
        The same, with ``x0`` the value of ``x`` before the first step.
    ``Derivative(dt, poly_order=n)(x)``
        A backward difference that reads ``x`` at the ``n`` previous steps.
    ``Derivative(dt)(Derivative(dt)(x))``
        The second derivative.

    ``x`` is one sample per step, ``T=1``. The layer keeps the last
    ``poly_order`` samples of ``x`` from one step to the next. They are a
    state: an Input named ``<name>_<x>``, which the :class:`ModelyLoop` or
    the :class:`Loop` feeds back with its other states. It is not one of
    your Inputs: :attr:`Modely.inputs` lists it as not declared.

    The rule and the results are those of :class:`DerivativeStep` on a window
    of the same samples. Built as a plain :class:`Modely`, which has no steps,
    the layer reads ``init`` as the previous samples.

    Parameters
    ----------
    dt : float
        The time between two steps, a positive number. It is required.
    poly_order : int, default 1
        The degree of the backward difference, a positive integer, as in
        :class:`DerivativeStep`: ``1`` reads the previous step, ``2`` the
        two previous ones.
    init : Stream or float, optional
        The value of ``x`` before the first step. It is zero when left out,
        a signal at rest before the start. It can be:

        * a number, a :class:`Parameter` or a :class:`Constant`;
        * an Input of the step, whose first step is read: one sample,
          repeated, or ``poly_order`` samples, oldest first.

        Its features are a scalar or those of ``x``, and its sequence is none
        or that of ``x``. It is the only initial value of the state: a Loop
        cannot give it another.
    name : str, optional
        The name of the layer and of the stream it returns. The state is
        named ``<name>_<x>``. A second call of the same layer differentiates
        with a state of its own: its result is named ``<name>_2``, and its
        state ``<name>_2_<x>``. The default name is ``Derivative`` followed
        by a number.

    Call arguments
    --------------
    x : Stream
        One sample per step, ``Shape(S, 1, D)``. The call takes it alone or
        in a list of one.

    Returns
    -------
    Stream
        The shape of ``x``, ``Shape(S, 1, D)``: the derivative at this step.
        For example, ``Input("x", tw=1)`` gives ``Shape(S=(), T=1, D=(1,))``.
        As an output of a ModelyLoop it is a trajectory, one value per step.

    Raises
    ------
    ValueError
        If ``dt`` is missing or not positive, if ``poly_order`` is not a
        positive integer, if ``x`` has no window ("acts on the window of its
        stream") or more than one sample ("takes one sample per step"), or if
        ``init`` has another number of samples, features or sequence ("init
        holds the samples before the first").
    TypeError
        If ``dt`` is not a number, if ``init`` is not a stream, a number or
        None, if ``x`` is not a stream, or if the layer is called before it
        is configured, ``Derivative(x)``.

    See Also
    --------
    DerivativeStep : The same derivative along a window.
    Integrate : The integral one step at a time, its inverse.
    ModelyLoop : A model declared one step at a time.
    Loop : The rollout of a one-step model.

    Notes
    -----
    At step :math:`k`, with :math:`p` = ``poly_order`` and
    :math:`x_{-p}, \dots, x_{-1}` from ``init``:

    .. math::

        \dot x_k = \frac{x_k - x_{k-1}}{\Delta t}

    for :math:`p = 1`. The higher orders are those of
    :class:`DerivativeStep`, with the step :math:`k` in place of the sample
    :math:`i`.

    Basic Examples
    --------------
    The velocity of a position given one sample per step, every 0.5 s. The
    first step reads zero before the start:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import (
    ...     Derivative, DerivativeStep, Input, Integrate, Modely, ModelyLoop, Output,
    ... )
    >>> x = Input("x", tw=1)
    >>> v = Derivative(dt=0.5, name="v")(x)
    >>> model = ModelyLoop("velocity", inputs=[x], outputs=[Output("v", v)]).build()
    >>> steps = np.array([1.0, 2.0, 4.0, 7.0]).reshape(1, 4, 1, 1)
    >>> keras.ops.convert_to_numpy(model({"x": steps})["v"]).ravel()
    array([2., 2., 4., 6.], dtype=float32)

    The state is an Input of the model, named after the layer and ``x``:

    >>> model.inputs[1].name
    'v_x'

    Options
    -------
    ``init`` is the value before the first step, and ``poly_order=2`` reads
    two previous steps:

    >>> from_one = Derivative(dt=0.5, init=1.0, name="from_one")(x)
    >>> second = Derivative(dt=0.5, poly_order=2, name="second")(x)
    >>> model = ModelyLoop(
    ...     "options", inputs=[x],
    ...     outputs=[Output("from_one", from_one), Output("second", second)],
    ... ).build()
    >>> result = model({"x": steps})
    >>> keras.ops.convert_to_numpy(result["from_one"]).ravel()
    array([0., 2., 4., 6.], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["second"]).ravel()
    array([3., 2., 5., 7.], dtype=float32)

    ``init`` can be an Input of the step: its first step is read.

    >>> x0 = Input("x0", tw=1)
    >>> Derivative(dt=0.5, init=x0)(x).shape
    Shape(S=(), T=1, D=(1,))

    Applications
    ------------
    A damper force, :math:`F = -c\,\dot x`, in a model declared one step at a
    time:

    >>> damping = -0.8 * Derivative(dt=0.01)(Input("position", tw=1))

    Properties & Relations
    ----------------------
    The same samples give the same values as :class:`DerivativeStep` on a
    window:

    >>> w = Input("w", tw=4)
    >>> v_window = DerivativeStep(dt=0.5)(w)
    >>> model = Modely("window", inputs=[w], outputs=[Output("v", v_window)]).build()
    >>> keras.ops.convert_to_numpy(model({"w": steps.reshape(1, 4, 1)})["v"]).ravel()
    array([2., 2., 4., 6.], dtype=float32)

    :class:`Integrate` with the same ``init`` gives ``x`` back:

    >>> back = Integrate(dt=0.5, init=3.0)(Derivative(dt=0.5, init=3.0)(x))
    >>> model = ModelyLoop("inverse", inputs=[x], outputs=[Output("x", back)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": steps})["x"]).ravel()
    array([1., 2., 4., 7.], dtype=float32)

    Built as a plain :class:`Modely`, the step reads ``init`` as the previous
    sample: :math:`(3 - 1) / 0.5`.

    >>> model = Modely("one_step", inputs=[x], outputs=[Output("v", from_one)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[[3.0]]])})["v"])
    array([[[4.]]], dtype=float32)

    Possible Issues
    ---------------
    The layer reads one sample per step. Differentiate a window with
    :class:`DerivativeStep`:

    >>> Derivative(dt=0.5)(Input("long", tw=4))
    Traceback (most recent call last):
        ...
    ValueError: ...Derivative takes one sample per step, but the signal carries 4 time samples...
    >>> DerivativeStep(dt=0.5)(Input("long", tw=4)).shape
    Shape(S=(), T=4, D=(1,))

    .. versionadded:: 2.0
    """

    def __init__(
        self,
        dt: float | None = None,
        poly_order: int = 1,
        init: Stream | float | None = None,
        name=None,
    ):
        self._init = _validate_init("Derivative", init)
        self._calls = 0
        super().__init__(
            dt=dt,
            poly_order=poly_order,
            name=name,
            init=self._init,
        )

    def __call__(self, function):
        if isinstance(function, (list, tuple)) and len(function) == 1:
            function = function[0]
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: Derivative expects a Stream input, got "
                f"{type(function).__name__}."
            )
        self._needs_window(function)
        if function.tw != 1:
            raise ValueError(
                f"{self.name}: Derivative takes one sample per step, but the "
                f"signal carries {function.tw} time samples. Differentiate a "
                "time window with DerivativeStep."
            )
        if isinstance(self._init, Stream):
            _check_history(self.name, self._init, function, self.past)

        # Every call differentiates a signal of its own, so each has its own
        # past.
        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        # Named after the signal whose past it holds.
        past = Input(
            f"{call_name}_{function.name}",
            dim=function.dim,
            seq=_declared_seq(function),
            tw=self.past,
        )
        past._hidden = True
        derivative = DerivativeStep(
            dt=self._dt, poly_order=self._poly_order, name=call_name
        )(past, function)
        past._feed_back(function, init=self._init)
        return derivative

    def build_layer(self):
        raise RuntimeError(
            f"{self.name}: a Derivative is the DerivativeStep it calls on an "
            "Input fed back with its signal, not a layer of its own."
        )
