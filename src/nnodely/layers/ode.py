"""Ode - fixed-step integration of a derivative function."""

from functools import reduce
from typing import Any

# Butcher tableaux as (stage coefficients, weights): one row of `a` per stage
# after the first, then the `b` row. Adding an explicit method is one entry.
TABLEAUX = {
    "euler": ((), (1.0,)),
    "midpoint": (((0.5,),), (0.0, 1.0)),
    "heun": (((1.0,),), (0.5, 0.5)),
    "rk4": (((0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)), (1 / 6, 2 / 6, 2 / 6, 1 / 6)),
}


def _derivatives(f, states, args):
    """Evaluate the derivative function on one stage."""
    derivatives = f(*states, *args)
    if not isinstance(derivatives, (list, tuple)):
        derivatives = [derivatives]
    if len(derivatives) != len(states):
        raise ValueError(
            f"Ode expects f to return one derivative per state: got "
            f"{len(derivatives)} derivatives for {len(states)} states."
        )
    return list(derivatives)


def _weighted_sum(coefficients, stages, index):
    """Combine the `index`-th derivative of every stage, skipping zero weights."""
    terms = [
        stage[index] if coefficient == 1.0 else coefficient * stage[index]
        for coefficient, stage in zip(coefficients, stages)
        if coefficient
    ]
    return reduce(lambda left, right: left + right, terms)


def Ode(f, states, dt, method="rk4", args=()) -> Any:
    r"""The states one time step later, by an explicit Runge-Kutta method.

    Syntax
    ------
    ``Ode(f, x, dt)``
        ``x`` one step ``dt`` later, by the rule ``dx/dt = f(x)``.
    ``Ode(f, [x, v], dt)``
        Several states, one stream each in a list.
    ``Ode(f, [x, v], dt, method="euler", args=(u,))``
        Another method, and streams ``f`` reads besides the states.

    ``f`` is a Python function that writes the dynamics with streams and
    layers. ``Ode`` calls it once per stage of the method, and combines the
    derivatives into the next value of each state. The result is a stream
    like any other: it trains, rolls out and exports.

    One call is one step. A trajectory is this step used as the body of a
    :class:`Loop`, or inside a :class:`ModelyLoop`.

    Parameters
    ----------
    f : callable
        The dynamics, ``f(*states, *args)``. It returns the derivative of
        each state, in the order of ``states``: a stream for one state, a
        list or a tuple for several. Create the layers with weights and the
        Parameters outside ``f`` and use them inside it: a layer created
        inside ``f`` is a new layer at every stage.
    states : Stream or list of Stream
        The state at the start of the step, one stream per state variable.
    dt : float or Stream
        The time step. A stream, such as an Input or a :class:`Parameter`,
        reads it from data or learns it.
    method : str, default "rk4"
        The Runge-Kutta method:

        * ``"euler"``: one stage, first order.
        * ``"midpoint"``: two stages, second order.
        * ``"heun"``: two stages, second order.
        * ``"rk4"``: four stages, fourth order.
    args : tuple, default ()
        Streams or numbers passed to ``f`` after the states, the same at every
        stage, such as an input force.

    Returns
    -------
    Stream or list of Stream
        The state one step later: a stream for a single stream ``states``, a
        list of streams for a list, in its order. A state ``Shape(S, T, D)``
        gives ``Shape(S, T, D)``, when its derivative has the same shape. For
        example, ``Input("x", tw=1)`` gives ``Shape(S=(), T=1, D=(1,))``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the four ("is not available; choose one
        of"), or if ``f`` does not return one derivative per state ("Ode
        expects f to return one derivative per state").

    See Also
    --------
    Loop : The rollout of a one-step model.
    ModelyLoop : A model declared one step at a time.
    IntegrateStep : The running integral of a given rate.
    Integrate : The integral of a rate, one step at a time.

    Notes
    -----
    With :math:`x' = f(x)`, the stages and the next state are:

    .. math::

        k_1 = f(x_n), \qquad
        k_i = f\left(x_n + \Delta t \sum_{j<i} a_{ij} k_j\right)

        x_{n+1} = x_n + \Delta t \sum_i b_i k_i

    The coefficients :math:`a_{ij}` and :math:`b_i` of each method are:

    * ``"euler"``: :math:`b = (1)`.
    * ``"midpoint"``: :math:`a_{21} = 1/2`, :math:`b = (0, 1)`.
    * ``"heun"``: :math:`a_{21} = 1`, :math:`b = (1/2, 1/2)`.
    * ``"rk4"``: :math:`a_{21} = a_{32} = 1/2`, :math:`a_{43} = 1`,
      :math:`b = (1/6, 1/3, 1/3, 1/6)`.

    ``"euler"`` is the rule of :class:`IntegrateStep` with
    ``method="euler"``. The other methods of ``IntegrateStep`` are not
    methods of ``Ode``.

    Basic Examples
    --------------
    A decay, :math:`x' = -x`, one Euler step of 0.5 s from 1:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, IntegrateStep, Loop, Modely, Ode, Output
    >>> x = Input("x", tw=1)
    >>> x_next = Ode(lambda x: -1.0 * x, x, dt=0.5, method="euler")
    >>> model = Modely("decay", inputs=[x], outputs=[Output("x_next", x_next)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[[1.0]]])})["x_next"])
    array([[[0.5]]], dtype=float32)

    Scope
    -----
    Several states give a list. A mass under the acceleration ``a``, read by
    ``f`` from ``args``, one Euler step:

    >>> p, v, a = Input("p", tw=1), Input("v", tw=1), Input("a", tw=1)
    >>> p_next, v_next = Ode(lambda p, v, a: (v, a), [p, v], dt=0.5, method="euler", args=(a,))
    >>> model = Modely(
    ...     "mass", inputs=[p, v, a],
    ...     outputs=[Output("p_next", p_next), Output("v_next", v_next)],
    ... ).build()
    >>> state = {"p": np.array([[[1.0]]]), "v": np.array([[[2.0]]]), "a": np.array([[[4.0]]])}
    >>> result = model(state)
    >>> keras.ops.convert_to_numpy(result["p_next"]), keras.ops.convert_to_numpy(result["v_next"])
    (array([[[2.]]], dtype=float32), array([[[4.]]], dtype=float32))

    Options
    -------
    ``method`` selects the method. Heun's is exact under a constant
    acceleration: :math:`1 + 0.5 \cdot 2 + 0.5^2 / 2 \cdot 4`.

    >>> p_next, v_next = Ode(lambda p, v, a: (v, a), [p, v], dt=0.5, method="heun", args=(a,))
    >>> model = Modely("heun", inputs=[p, v, a], outputs=[Output("p_next", p_next)]).build()
    >>> keras.ops.convert_to_numpy(model(state)["p_next"])
    array([[[2.5]]], dtype=float32)

    ``dt`` can be a stream, here read from data:

    >>> h = Input("h", tw=1)
    >>> x_next = Ode(lambda x: -1.0 * x, x, dt=h, method="midpoint")
    >>> model = Modely("step_size", inputs=[x, h], outputs=[Output("x_next", x_next)]).build()
    >>> result = model({"x": np.array([[[1.0]]]), "h": np.array([[[0.5]]])})
    >>> keras.ops.convert_to_numpy(result["x_next"])
    array([[[0.625]]], dtype=float32)

    Applications
    ------------
    A pendulum, :math:`\ddot\theta = -(g/l)\sin\theta`, simulated by a
    :class:`Loop` over 3 steps of RK4:

    >>> from nnodely import Parameter, Sin
    >>> theta, omega = Input("theta", tw=1), Input("omega", tw=1)
    >>> g_over_l = Parameter("g_over_l", value=9.81)
    >>> def pendulum(theta, omega):
    ...     return omega, -1.0 * g_over_l * Sin(theta)
    >>> theta_next, omega_next = Ode(pendulum, [theta, omega], dt=0.01)
    >>> step = Modely(
    ...     "pendulum", inputs=[theta, omega],
    ...     outputs=[Output("theta_next", theta_next), Output("omega_next", omega_next)],
    ... )
    >>> theta0 = Input("theta0", seq=3, tw=1)
    >>> angle, rate = Loop(
    ...     f=step, feedback={theta: "theta_next", omega: "omega_next"}, init={theta: theta0}
    ... )()
    >>> angle.shape
    Shape(S=(3,), T=1, D=(1,))

    Properties & Relations
    ----------------------
    The Euler step is :class:`IntegrateStep` with the derivative as the rate:

    >>> euler = Ode(lambda x: -1.0 * x, x, dt=0.5, method="euler")
    >>> integral = IntegrateStep(dt=0.5)(x, -1.0 * x)
    >>> model = Modely(
    ...     "same", inputs=[x], outputs=[Output("ode", euler), Output("integral", integral)]
    ... ).build()
    >>> result = model({"x": np.array([[[1.0]]])})
    >>> keras.ops.convert_to_numpy(result["ode"]), keras.ops.convert_to_numpy(result["integral"])
    (array([[[0.5]]], dtype=float32), array([[[0.5]]], dtype=float32))

    Possible Issues
    ---------------
    ``"trapezoidal"`` is a method of :class:`IntegrateStep`, not of ``Ode``:

    >>> Ode(lambda x: -1.0 * x, x, dt=0.5, method="trapezoidal")
    Traceback (most recent call last):
        ...
    ValueError: Ode method 'trapezoidal' is not available; choose one of ['euler', 'heun', 'midpoint', 'rk4'].

    ``f`` returns one derivative per state:

    >>> Ode(lambda p, v: v, [p, v], dt=0.5)
    Traceback (most recent call last):
        ...
    ValueError: Ode expects f to return one derivative per state: got 1 derivatives for 2 states.

    .. versionadded:: 2.0
    """
    if method not in TABLEAUX:
        raise ValueError(
            f"Ode method {method!r} is not available; choose one of {sorted(TABLEAUX)}."
        )

    single_state = not isinstance(states, (list, tuple))
    states = [states] if single_state else list(states)

    stage_coefficients, weights = TABLEAUX[method]
    stages = [_derivatives(f, states, args)]
    for coefficients in stage_coefficients:
        stage = [
            state + dt * _weighted_sum(coefficients, stages, index)
            for index, state in enumerate(states)
        ]
        stages.append(_derivatives(f, stage, args))

    results = [
        state + dt * _weighted_sum(weights, stages, index)
        for index, state in enumerate(states)
    ]
    return results[0] if single_state else results
