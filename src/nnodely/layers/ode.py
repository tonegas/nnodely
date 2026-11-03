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
    """Advance `states` by one step `dt` of the derivatives returned by `f`.

    `f` receives one stream per state, followed by `args`, and returns the
    derivative of each state in the same order. It is evaluated once per stage
    of the tableau, so the streams it builds are ordinary nnodely nodes and the
    step exports and rolls out like any other relation.

    `dt` is the step size, either a number or a stream, so it can be read from
    data or learned. `method` selects the tableau: `euler`, `midpoint`, `heun`
    or `rk4`.

    One step covers one sample interval. A trajectory is this step used as the
    body of a `Loop`, which owns the rollout.
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
