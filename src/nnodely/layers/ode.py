"""Ode - fixed-step and adaptive integration of a derivative function."""

from functools import reduce
from operator import add
from typing import Any

import keras

from nnodely.core.layer import Layer, has_batch
from nnodely.core.modely import Modely
from nnodely.core.stream import Stream
from nnodely.layers.loop import LoopOutputImpl

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


# Dormand-Prince 5(4): the stage coefficients, the fifth-order weights, and the
# difference against the embedded fourth-order weights, which is the local error
# estimate the step-size controller reads.
DOPRI5_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
DOPRI5_B = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
DOPRI5_B_ERROR = tuple(
    weight - embedded
    for weight, embedded in zip(
        DOPRI5_B,
        (
            5179 / 57600,
            0.0,
            7571 / 16695,
            393 / 640,
            -92097 / 339200,
            187 / 2100,
            1 / 40,
        ),
    )
)

# Controller gains, following the usual dopri5 defaults: back off a little from
# the predicted step, and never change it by more than these factors at once.
SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 10.0

# Locating an event divides by the change of the guard across the step. Only a
# crossing step divides by something meaningful, but the branch that is not
# taken is evaluated all the same, and without this its gradient comes back NaN.
EVENT_EPSILON = 1e-12


@keras.saving.register_keras_serializable(package="nnodely")
class OdeNetImpl(keras.layers.Layer):
    """Integrate a vector field to each requested time.

    With `dopri5` the step size is chosen by the solver rather than by the
    caller: every step is tried, its error estimated against the embedded
    fourth-order solution and accepted only when the scaled error norm is at
    most one. A rejected step shrinks and is retried, so the number of body
    evaluations depends on the values flowing through and the march cannot be
    unrolled the way `Loop` is.

    With a fixed-step tableau each reported interval is covered by `steps`
    equal substeps, so the body is evaluated a statically known number of times
    and the march unrolls into ordinary ops: that is the form that carries
    gradients on every backend and exports. It is also the only form that takes
    events, which cut a step in two around the guard crossing.

    The requested times are shared across the batch:
    one step size serves the whole batch, so each segment has a single target.
    Their count is static because it is a tensor axis, but their values are read
    at call time, which is what lets one trained field be integrated over any
    horizon without rebuilding the graph.
    """

    def __init__(
        self,
        model,
        state_input_names: tuple[str, ...],
        state_output_names: tuple[str, ...],
        model_output_names: tuple[str, ...],
        result_shapes: tuple[tuple[int, ...], ...],
        output_count: int,
        method: str = "rk4",
        steps: int = 1,
        event_output_name: str | None = None,
        reset_output_names: tuple[str, ...] = (),
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 1000,
        times_have_batch: bool = True,
        return_all_outputs: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.state_input_names = tuple(state_input_names)
        self.state_output_names = tuple(state_output_names)
        self.model_output_names = tuple(model_output_names)
        self.result_shapes = tuple(tuple(shape) for shape in result_shapes)
        self.output_count = int(output_count)
        self.method = method
        self.steps = int(steps)
        self.event_output_name = event_output_name
        self.reset_output_names = tuple(reset_output_names)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_steps = int(max_steps)
        self.times_have_batch = bool(times_have_batch)
        self.return_all_outputs = bool(return_all_outputs)
        self.output_indices = {
            name: index for index, name in enumerate(self.model_output_names)
        }

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def _select(self, states, names):
        """Evaluate the body on one stage and keep the named outputs."""
        outputs = self.model(
            {name: state for name, state in zip(self.state_input_names, states)}
        )
        if isinstance(outputs, dict):
            return [outputs[name] for name in names]
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        return [outputs[self.output_indices[name]] for name in names]

    def _field(self, states):
        """Evaluate the body on one stage."""
        return self._select(states, self.state_output_names)

    def _fixed_step(self, states, step):
        """One step of a fixed tableau, the same stages `Ode` builds inline."""
        stage_coefficients, weights = TABLEAUX[self.method]
        stages = [self._field(states)]
        for coefficients in stage_coefficients:
            probe = [
                state + step * _weighted_sum(coefficients, stages, position)
                for position, state in enumerate(states)
            ]
            stages.append(self._field(probe))
        return [
            state + step * _weighted_sum(weights, stages, position)
            for position, state in enumerate(states)
        ]

    def _event_step(self, states, step):
        """One fixed step, with an event located and applied inside it.

        The guard is interpolated linearly across the step, so the fraction of
        the step at which it crosses zero is a value the graph differentiates
        like any other: that is what puts the event time into the gradient
        instead of freezing it on the step grid. The step is then taken in two
        pieces, with the reset applied in between.

        Both branches are evaluated on every step because the choice between
        them is a `where`, and the fraction is a tensor, so each sample of the
        batch crosses at its own time.
        """
        straight = self._fixed_step(states, step)
        before = self._guard(states)
        after = self._guard(straight)
        crossing = keras.ops.logical_and(before > 0.0, after <= 0.0)
        fraction = keras.ops.clip(before / (before - after + EVENT_EPSILON), 0.0, 1.0)
        at_event = self._fixed_step(states, fraction * step)
        jumped = self._fixed_step(self._reset(at_event), (1.0 - fraction) * step)
        return [
            keras.ops.where(crossing, crossed, plain)
            for plain, crossed in zip(straight, jumped)
        ]

    def _guard(self, states):
        """The event function, positive before the event and negative after."""
        return self._select(states, (self.event_output_name,))[0]

    def _reset(self, states):
        """The discrete map the states jump through at the event."""
        return self._select(states, self.reset_output_names)

    def _dopri5_step(self, states, step):
        """One trial step: the fifth-order candidate and its error estimate."""
        stages = []
        for row in DOPRI5_A:
            probe = (
                [
                    state + step * _weighted_sum(row, stages, position)
                    for position, state in enumerate(states)
                ]
                if row
                else list(states)
            )
            stages.append(self._field(probe))

        candidates = [
            state + step * _weighted_sum(DOPRI5_B, stages, position)
            for position, state in enumerate(states)
        ]
        errors = [
            step * _weighted_sum(DOPRI5_B_ERROR, stages, position)
            for position in range(len(states))
        ]
        return candidates, errors

    def _error_norm(self, states, candidates, errors):
        """RMS of the error over every element of every state, scaled by tolerance."""
        squares, counts = [], []
        for state, candidate, error in zip(states, candidates, errors):
            scale = self.atol + self.rtol * keras.ops.maximum(
                keras.ops.abs(state), keras.ops.abs(candidate)
            )
            squares.append(keras.ops.sum(keras.ops.square(error / scale)))
            counts.append(keras.ops.cast(keras.ops.size(error), self.compute_dtype))
        return keras.ops.sqrt(reduce(add, squares) / reduce(add, counts))

    def _march(self, start, states, step, target):
        """Advance from `start` to `target`, letting the controller pick the steps."""

        def cond(time, step, count, *states):
            return keras.ops.logical_and(time < target, count < self.max_steps)

        def body(time, step, count, *states):
            # Clip so the segment ends exactly on the requested time; a clipped
            # step makes the next one start conservatively, which self-corrects.
            trial = keras.ops.minimum(step, target - time)
            candidates, errors = self._dopri5_step(states, trial)
            norm = self._error_norm(states, candidates, errors)
            accepted = norm <= 1.0
            factor = keras.ops.clip(
                SAFETY * keras.ops.maximum(norm, 1e-10) ** (-0.2),
                MIN_FACTOR,
                MAX_FACTOR,
            )
            return (
                keras.ops.where(accepted, time + trial, time),
                trial * factor,
                count + 1,
                *[
                    keras.ops.where(accepted, candidate, state)
                    for state, candidate in zip(states, candidates)
                ],
            )

        carry = keras.ops.while_loop(
            cond,
            body,
            (start, step, keras.ops.convert_to_tensor(0), *states),
            maximum_iterations=self.max_steps,
        )
        return carry[0], list(carry[3:]), carry[1]

    # ------------------------------------------------------------------
    # Keras layer
    # ------------------------------------------------------------------
    def set_method(self, method: str, steps: int | None = None):
        """Swap the tableau on a built layer, keeping the trained field.

        The graph is replayed call by call, so the next call integrates with the
        new method; a model already compiled with `jit_compile` keeps its traced
        function, and has to be recompiled for the swap to reach it.
        """
        if method != "dopri5" and method not in TABLEAUX:
            raise ValueError(
                f"OdeNet method {method!r} is not available; choose 'dopri5' or "
                f"one of {sorted(TABLEAUX)}."
            )
        if method == "dopri5" and self.event_output_name is not None:
            raise ValueError(
                "OdeNet cannot switch to 'dopri5' while an event is configured: the "
                "adaptive march has no event handling, so the swap would silently "
                "integrate straight through the event."
            )
        self.method = method
        if steps is not None:
            self.steps = int(steps)

    def compute_output_spec(self, inputs):
        """Declare the outputs, so the march is never traced just for its shapes."""
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        batch = inputs[0].shape[0]
        specs = tuple(
            keras.KerasTensor(shape=(batch, *shape), dtype=self.compute_dtype)
            for shape in self.result_shapes
        )
        if self.return_all_outputs:
            return specs
        return specs[0]

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        state_count = len(self.state_input_names)
        states = list(inputs[:state_count])
        times = inputs[state_count]

        grid = keras.ops.cast(
            keras.ops.reshape(times[0] if self.times_have_batch else times, (-1,)),
            self.compute_dtype,
        )

        # The first requested time is the initial condition.
        trajectory = [states]
        if self.method == "dopri5":
            time = grid[0]
            step = (grid[-1] - grid[0]) / 100.0
            for index in range(1, self.output_count):
                time, states, step = self._march(time, states, step, grid[index])
                trajectory.append(states)
        else:
            advance = (
                self._fixed_step if self.event_output_name is None else self._event_step
            )
            for index in range(1, self.output_count):
                step = (grid[index] - grid[index - 1]) / self.steps
                for _ in range(self.steps):
                    states = advance(states, step)
                trajectory.append(states)

        results = tuple(
            keras.ops.stack([entry[position] for entry in trajectory], axis=-1)
            for position in range(state_count)
        )
        if self.return_all_outputs:
            return results
        return results[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": keras.saving.serialize_keras_object(self.model),
                "state_input_names": self.state_input_names,
                "state_output_names": self.state_output_names,
                "model_output_names": self.model_output_names,
                "result_shapes": self.result_shapes,
                "output_count": self.output_count,
                "method": self.method,
                "steps": self.steps,
                "event_output_name": self.event_output_name,
                "reset_output_names": self.reset_output_names,
                "rtol": self.rtol,
                "atol": self.atol,
                "max_steps": self.max_steps,
                "times_have_batch": self.times_have_batch,
                "return_all_outputs": self.return_all_outputs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        return cls(**config)


class OdeNetOutput(Layer):
    """Select one state's trajectory from a multi-state OdeNet."""

    def __init__(self, odenet, index: int, state_node, result_seq):
        self.index = int(index)
        super().__init__(
            name=f"{odenet.name}_{state_node.name}",
            preds=[odenet],
            dim=state_node.dim,
            time=state_node.time,
            seq=result_seq,
            index=self.index,
        )

    def build_layer(self):
        # The same index selector the Loop outputs use.
        return LoopOutputImpl(index=self.index, name=self.name)

    def get_config(self):
        return {"name": self.name, "index": self.index}


class OdeNet(Layer):
    """Integrate a vector field.

    `f` is a built Modely holding the field: `states` maps each of its state
    inputs to the output carrying that state's derivative, so the body computes
    `dx/dt` and the solver owns the stepping. The field must be autonomous -
    every body input has to be a state - because an exogenous driver would have
    to be evaluated at the solver's own times, which are not known in advance.

    `t` is the stream of times to report, shared across the batch. 
    Its last axis fixes how many points come back, but the values
    are read at call time: the same trained field can be integrated over any
    horizon by feeding a different `t`, with no rebuild. The first time is the
    initial condition, and the result appends those points as the last sequence
    axis of each state.

    `method` picks the integrator. With `dopri5` the controller owns the step
    size, working to `rtol` and `atol` and bounded by `max_steps`: the step
    count then depends on the data, so the layer is a real loop rather than an
    unrolled graph, it does not export to ONNX, and it is not reverse-mode
    differentiable on the jax backend. With a fixed tableau - `euler`,
    `midpoint`, `heun` or `rk4` - each reported interval is covered by `steps`
    equal substeps, which makes the march an unrolled graph that trains and
    exports like any other relation. That is the mode to fit a field in: unlike
    `Ode`, the body here is a built Modely, so a weight-bearing layer is shared
    across stages instead of being rebuilt by each one.

    The two modes are the same layer, so a field fitted with `rk4` is switched
    to the adaptive march for inference with
    `model.get_layer(name).set_method("dopri5")`.

    `event` and `reset` make the field hybrid: `event` names an output of `f`
    holding a single value per sample, positive before the event and negative
    after, and `reset` maps every state to the output holding the value it jumps
    to when that guard crosses zero. The crossing is located inside the step by
    interpolating the guard, so the event time moves with the parameters and its
    gradient reaches them - a reset applied at the end of whichever step happens
    to cross would leave a bias that no amount of refinement removes. Events
    need a fixed tableau, and cost three evaluations of it per substep; at most
    one event is caught per substep.
    """

    def __init__(
        self,
        f: Modely,
        states: dict,
        t: Stream,
        initial: dict | None = None,
        method: str = "rk4",
        steps: int = 1,
        event: str | Stream | None = None,
        reset: dict | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 1000,
        name=None,
    ):
        if not f.built:
            f.build()
        if not isinstance(states, dict) or not states:
            raise ValueError(
                "OdeNet states must contain at least one state: derivative pair."
            )
        if method != "dopri5" and method not in TABLEAUX:
            raise ValueError(
                f"OdeNet method {method!r} is not available; choose 'dopri5' or "
                f"one of {sorted(TABLEAUX)}."
            )
        if int(steps) < 1:
            raise ValueError(f"OdeNet needs at least one substep, got {steps!r}.")

        pairs = [
            (
                _find_node(f.inputs, state, "state"),
                _find_node(f.outputs, derivative, "derivative"),
            )
            for state, derivative in states.items()
        ]
        state_inputs = [pair[0] for pair in pairs]
        state_outputs = [pair[1] for pair in pairs]

        drivers = [node for node in f.inputs if node not in state_inputs]
        if drivers:
            raise ValueError(
                f"OdeNet integrates an autonomous field, but {sorted(node.name for node in drivers)} "
                "are body inputs that are not states. An exogenous driver would have to be "
                "interpolated at the solver's own step times, which is not supported yet."
            )
        for node, derivative in pairs:
            if tuple(derivative.dim) != tuple(node.dim) or derivative.time != node.time:
                raise ValueError(
                    f"OdeNet state {node.name!r} -> {derivative.name!r}: the derivative must "
                    f"match the state's dim {tuple(node.dim)} and time {node.time}, got dim "
                    f"{tuple(derivative.dim)} and time {derivative.time}."
                )

        event_output, reset_outputs = _resolve_event(f, event, reset, state_inputs)
        if event_output is not None and method == "dopri5":
            raise ValueError(
                "OdeNet events need a fixed tableau: the adaptive march shares one "
                "step size across the batch, so it cannot stop at an event time per "
                "sample. Choose 'euler', 'midpoint', 'heun' or 'rk4'."
            )

        initial_streams = _resolve_initial(initial, state_inputs)
        self.output_count = _resolve_output_count(t)

        self.f = f
        self.state_inputs = state_inputs
        self.state_outputs = state_outputs
        self.initial_streams = initial_streams
        self.times = t
        self.method = method
        self.steps = int(steps)
        self.event_output = event_output
        self.reset_outputs = reset_outputs
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_steps = int(max_steps)
        self.return_all_outputs = False

        result_seq = (self.output_count,)
        super().__init__(
            name=name,
            preds=[*initial_streams, t],
            dim=state_inputs[0].dim,
            time=state_inputs[0].time,
            seq=result_seq,
        )
        self.odenet_outputs = [
            OdeNetOutput(self, index, node, result_seq)
            for index, node in enumerate(state_inputs)
        ]

    def __iter__(self):
        self.return_all_outputs = True
        return iter(self.odenet_outputs)

    def build_layer(self):
        return OdeNetImpl(
            model=self.f.model,
            state_input_names=tuple(node.name for node in self.state_inputs),
            state_output_names=tuple(node.name for node in self.state_outputs),
            model_output_names=tuple(node.name for node in self.f.outputs),
            result_shapes=tuple(
                (*node.dim, node.time, self.output_count) for node in self.state_inputs
            ),
            output_count=self.output_count,
            method=self.method,
            steps=self.steps,
            event_output_name=(
                None if self.event_output is None else self.event_output.name
            ),
            reset_output_names=tuple(node.name for node in self.reset_outputs),
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            times_have_batch=_has_batch(self.times),
            return_all_outputs=self.return_all_outputs,
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "method": self.method,
            "steps": self.steps,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
        }


def _find_node(nodes, value, kind):
    if isinstance(value, str):
        matches = [node for node in nodes if node.name == value]
        if len(matches) != 1:
            raise ValueError(f"OdeNet {kind} {value!r} was not found.")
        return matches[0]
    if value not in nodes:
        raise ValueError(
            f"OdeNet {kind} {getattr(value, 'name', value)!r} does not belong to the "
            "provided Modely."
        )
    return value


def _resolve_event(f, event, reset, state_inputs):
    """The guard output and the reset output of every state, or no event."""
    if event is None and reset is None:
        return None, []
    if event is None or reset is None:
        raise ValueError(
            "OdeNet event and reset go together: the guard says when the states "
            "jump and the reset says where they jump to."
        )

    event_output = _find_node(f.outputs, event, "event")
    if tuple(event_output.dim) != (1,) or event_output.time != 1:
        raise ValueError(
            f"OdeNet event {event_output.name!r} must be a single value per sample, "
            f"positive before the event and negative after, got dim "
            f"{tuple(event_output.dim)} and time {event_output.time}."
        )

    if not isinstance(reset, dict):
        raise TypeError("OdeNet reset must be a dict of state: output.")
    by_name = {
        key.name if isinstance(key, Stream) else key: value
        for key, value in reset.items()
    }
    expected = {node.name for node in state_inputs}
    if set(by_name) != expected:
        raise ValueError(
            "OdeNet reset must map every state to the value it jumps to; expected "
            f"{sorted(expected)}, got {sorted(by_name)}."
        )

    reset_outputs = []
    for node in state_inputs:
        output = _find_node(f.outputs, by_name[node.name], "reset")
        if tuple(output.dim) != tuple(node.dim) or output.time != node.time:
            raise ValueError(
                f"OdeNet reset {node.name!r} -> {output.name!r}: the reset must match "
                f"the state's dim {tuple(node.dim)} and time {node.time}, got dim "
                f"{tuple(output.dim)} and time {output.time}."
            )
        reset_outputs.append(output)
    return event_output, reset_outputs


_has_batch = has_batch


def _resolve_output_count(t):
    """Number of reported times, which is a tensor axis and so must be static."""
    count = t.shape.tuple[-1]
    if not isinstance(count, int) or count < 2:
        raise ValueError(
            f"OdeNet needs at least two reported times with a static count, got "
            f"{count!r} from the last axis of {t.name!r}."
        )
    return count


def _resolve_initial(initial, state_inputs):
    """Outer stream seeding each state, defaulting to the body input itself."""
    if initial is None:
        return list(state_inputs)
    if not isinstance(initial, dict):
        raise TypeError("OdeNet initial must be a dict of state: outer stream.")

    by_name = {
        key.name if isinstance(key, Stream) else key: value
        for key, value in initial.items()
    }
    expected = {node.name for node in state_inputs}
    unknown = set(by_name) - expected
    if unknown:
        raise ValueError(
            f"OdeNet initial keys {sorted(unknown)} are not states of the field."
        )

    streams = []
    for node in state_inputs:
        stream = by_name.get(node.name, node)
        if stream is not node and (
            tuple(stream.dim) != tuple(node.dim)
            or stream.time != node.time
            or tuple(stream.seq) != ()
        ):
            raise ValueError(
                f"OdeNet initial value for {node.name!r} must have dim {tuple(node.dim)}, "
                f"time {node.time} and no sequence axis, got dim {tuple(stream.dim)}, time "
                f"{stream.time} and seq {tuple(stream.seq)} from {stream.name!r}."
            )
        streams.append(stream)
    return streams
