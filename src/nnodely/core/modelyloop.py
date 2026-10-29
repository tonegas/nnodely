"""ModelyLoop: a model whose closed Inputs are fed back by a single rollout."""

from __future__ import annotations

from typing import Any, Callable, cast

import keras

from nnodely.core.dag import (
    Scope,
    _streams_from,
    flatten_closed,
    flatten_node,
    toposort_outputs,
)
from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.core.stream import Node, Stream
from nnodely.layers.input import Input
from nnodely.layers.loop import Loop
from nnodely.layers.output import Output


def _closed(roots) -> list[Input]:
    """Every closed Input ``roots`` read, and every one the streams they are
    closed with read in turn, in the order they are reached."""
    closed: list[Input] = []
    seen: set[int] = set()
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Input) and node.update is not None:
            closed.append(node)
            stack.append(node.update)
        stack.extend(reversed(node.preds))
    return closed


def _closed_across_calls(
    scope: Scope, enclosing: frozenset = frozenset()
) -> list[tuple[str, str]]:
    """The name of every closed Input a call inlines while a graph enclosing
    it inlines it too, with the name of the model called: one state per call,
    the call's is not the caller's."""
    found = []
    here = enclosing | {
        node
        for node in scope.memo
        if isinstance(node, Input) and node.update is not None
    }
    for model_call, call_scope in scope.calls.items():
        found += [
            (node.name, cast(Any, model_call).model.name)
            for node in call_scope.memo
            if node in here
        ]
        found += _closed_across_calls(call_scope, here)
    return sorted(found)


def _free_name(name: str, taken: set[str]) -> str:
    """``name``, or the first ``name_<i>`` not in ``taken``."""
    candidate, index = name, 2
    while candidate in taken:
        candidate, index = f"{name}_{index}", index + 1
    taken.add(candidate)
    return candidate


class ModelyLoop(Modely):
    """A model declared one step at a time, whose closed Inputs are all fed
    back by a single rollout::

        x, v = Input("x", sw=1), Input("v", sw=1)
        a = (F - k * x - c * v) / m
        v_next = v.close(IntegrateStep(dt=dt)(v, a))
        x_next = x.close(IntegrateStep(dt=dt)(x, v_next))
        msd = ModelyLoop("msd", inputs=[F, x], outputs=[Output("x", x_next)])

    The graph is the body of a loop: its Inputs read one step of their signal,
    and every Input it reads that is closed with :meth:`Input.close` -
    directly, or inside the models it calls, one per call - is fed back with
    the stream it is closed with. They are closed together, so the model is
    one :class:`Loop` however many feedbacks it holds, and any closed Input
    may read any other.

    Every Input the step reads and nothing closes is among ``inputs``. A
    closed Input starts from its ``init``; without one, from the first window
    of its own data when it is among ``inputs`` - zero where the data leaves
    it out - and from zero when it is not, as ``v`` above. A model the step
    calls as a block keeps the feedback of the closed Inputs it declares: the
    stream the call gives one is its initial value, read at the first step.

    Rolled out, every Input carries the horizon as its outermost sequence
    axis, ``length`` steps, or as many as the data has when ``length`` is
    None, and every output is a trajectory: its element ``k`` is computed at
    step ``k``. A closed Input that starts from data, or from an ``init``
    that is an Input, reads its first step.

    :meth:`minimize` takes the streams of the step - an output of the model,
    or an Input nothing closes - and compares their trajectories. ``body`` is
    the one-step model the loop rolls out.
    """

    def __init__(
        self,
        name: str,
        inputs: list[Input],
        outputs: list[Output],
        length: int | None = None,
    ) -> None:
        if length is not None and (
            isinstance(length, bool) or not isinstance(length, int) or length < 1
        ):
            raise ValueError(
                f"ModelyLoop {name!r}: length is a positive int, or None for the "
                f"length of the data, got {length!r}."
            )
        self.name = name
        self.length = length
        self._rolled_inputs: dict[str, Input] = {}

        # The step, with every model it calls inlined: a closed Input inside
        # one is copied once per call, together with what closes it.
        scope = Scope(pending=[], loop=True)
        step_outputs = [flatten_node(output, scope) for output in outputs]
        flatten_closed(scope)
        crossing = _closed_across_calls(scope)
        if crossing:
            state, model = crossing[0]
            raise ValueError(
                f"ModelyLoop {name!r} reads the closed Input {state!r} both in "
                f"the graph that calls {model!r} and inside it, where {model!r} "
                "does not declare it. A closed Input inside a called model is "
                "one state per call, so these would be two states. To read the "
                f"state of the call, make it an output of {model!r}; to give "
                f"{model!r} the state of the caller, declare an Input in "
                f"{model!r} for it and pass {state!r} in the call."
            )
        closed = _closed(step_outputs)
        if not closed:
            raise ValueError(
                f"ModelyLoop {name!r} reads no closed Input, so it has no "
                "feedback: close one with x.close(stream), or build it as a Modely."
            )
        # A model called twice holds a closed Input of each name per call; the
        # body reads them by name, so the copies after the first are renamed.
        names: set[str] = set()
        for node in closed:
            if node.name in names:
                node.name = _free_name(node.name, names)
                node.input = keras.Input(shape=node.shape.tuple, name=node.name)
            names.add(node.name)

        # Each closed Input is fed back from an output of the body: one the
        # model already declares for its stream, or one added for it.
        body_outputs = list(step_outputs)
        taken = {output.name for output in body_outputs}
        callback: dict[Input, Output] = {}
        for node in closed:
            output = next(
                (
                    output
                    for output in body_outputs
                    if node.update in (output, output.preds[0])
                    and output not in callback.values()
                ),
                None,
            )
            if output is None:
                output = Output(
                    _free_name(f"{node.name}_next", taken), cast(Stream, node.update)
                )
                body_outputs.append(output)
            callback[node] = output

        step_inputs: dict[str, Input] = {}
        for node in toposort_outputs(body_outputs):
            if isinstance(node, Input) and node.update is None:
                step_inputs.setdefault(node.name, node)
        listed = {node.name for node in inputs}
        read = [
            *step_inputs.values(),
            *(node.init for node in closed if isinstance(node.init, Input)),
        ]
        unlisted = sorted(
            {
                node.name
                for node in read
                if node.update is None and node.name not in listed
            }
        )
        if unlisted:
            raise ValueError(
                f"ModelyLoop {name!r} reads the Inputs {unlisted}, which nothing "
                "closes and are not among its inputs: list them in inputs, or "
                "close them with x.close(stream)."
            )

        # A closed Input starts from its init or, listed, from its data, which
        # may leave it out: then, as when it is not listed, from zero.
        initial: dict[Input, Any] = {}
        optional: set[str] = set()
        for node in closed:
            if node.init is not None:
                if node.name in listed:
                    raise ValueError(
                        f"ModelyLoop {name!r}: {node.name!r} has an initial "
                        "value - the init it is closed with, or the stream a call "
                        "gives it - and is among the inputs, which would start it "
                        "from its data: leave it out of inputs, or give it no "
                        "initial value."
                    )
                initial[node] = self._initial_value(node)
            elif node.name in listed:
                initial[node] = self._rolled_input(node)
                optional.add(node.name)
        if not step_inputs and not any(
            isinstance(value, Input) for value in initial.values()
        ):
            raise ValueError(
                f"ModelyLoop {name!r} reads no Input, in its steps or as the "
                "initial value of a closed Input, so no data sets its batch and "
                "number of steps."
            )
        self._initial = initial
        self.body = Modely(
            f"{name}_step",
            inputs=[*step_inputs.values(), *closed],
            outputs=body_outputs,
        ).build()
        # The layers were built on the copies: the streams the model was
        # declared with hand out the same ones, Parameters their values.
        for source, copy in scope.merged_memo().items():
            if isinstance(source, Layer) and isinstance(copy, Layer):
                source._layer = copy._layer
                source._layer_signature = copy._layer_signature

        loop = Loop(
            f=self.body,
            callback=callback,
            init=initial,
            length=length,
            name=f"{name}_loop",
        )({node.name: self._rolled_input(node) for node in step_inputs.values()})
        # One trajectory per output of the body, the declared outputs first.
        trajectories = list(loop)
        self._trajectories: dict[Node, Output] = {}
        rolled_outputs = []
        for output, trajectory in zip(outputs, trajectories):
            rolled = Output(output.name, trajectory)
            self._trajectories[output] = rolled
            self._trajectories.setdefault(output.preds[0], rolled)
            rolled_outputs.append(rolled)

        super().__init__(
            name,
            inputs=[self._rolled_input(node) for node in inputs],
            outputs=rolled_outputs,
        )
        self.optional_inputs = optional

    def _input_rows(self) -> list[dict]:
        """The rows of :attr:`all_inputs`: the Inputs the data feeds, shaped as
        the step reads them, then the closed Inputs the loop feeds back."""
        rows = super()._input_rows()
        for row in rows:
            row["seq"] = row["seq"][1:]  # the horizon, which the data adds
        by_name = {row["input"].name: row for row in rows}
        for node in self.body.inputs:
            if node.update is None:
                continue
            row = by_name.get(node.name)
            if row is None:
                row = {"input": node, "seq": tuple(node.seq), "declared": False}
                row["data"] = None
                rows.append(row)
            row["closed with"] = node.update.name
            start = self._initial.get(node)
            if node.name in self.optional_inputs:
                row["starts from"] = "its data, else 0"
            elif start is None:
                row["starts from"] = "0"
            elif isinstance(start, Input):
                row["starts from"] = f"{start.name}, first step"
            else:
                row["starts from"] = getattr(start, "name", str(start))
        return rows

    def _rolled_input(self, node: Input) -> Input:
        """The Input of the rollout for an Input of the step: the same name,
        the horizon its new outermost sequence axis."""
        rolled = self._rolled_inputs.get(node.name)
        if rolled is None:
            if node.time is None:
                raise ValueError(
                    f"ModelyLoop {self.name!r} reads its Input {node.name!r} one "
                    "step at a time, so its data carries the horizon as a "
                    "sequence axis, which needs a window: declare it with sw=1."
                )
            horizon = -1 if self.length is None else self.length
            rolled = Input(
                node.name,
                dim=node.dim,
                seq=(horizon, *(-1 if size is None else size for size in node.seq)),
                sw=(node.past, node.future),
            )
            self._rolled_inputs[node.name] = rolled
        return rolled

    def _initial_value(self, node: Input):
        """The init ``node`` is closed with, in the rollout: read before its
        first step."""
        init = node.init
        if not isinstance(init, Stream):
            return init
        if isinstance(init, Input) and init.update is not None:
            raise ValueError(
                f"ModelyLoop {self.name!r}: the init of {node.name!r} is the "
                f"closed Input {init.name!r}, which has no value before the "
                "first step."
            )
        if isinstance(init, Input):
            return self._rolled_input(init)
        read = sorted(
            {node.name for node in _streams_from([init]) if isinstance(node, Input)}
        )
        if read:
            raise ValueError(
                f"ModelyLoop {self.name!r}: the init of {node.name!r} is "
                f"computed from the Inputs {read}. An init is read once, before "
                "the first step: give an Input, whose first step it reads, a "
                "Parameter, a Constant or a number."
            )
        return init

    def _trajectory(self, value, minimizer: str):
        """The stream of the rollout a minimizer side of the step compares."""
        if not isinstance(value, Stream) or value._is_value or value in self.order:
            return value
        rolled = self._trajectories.get(value)
        if rolled is not None:
            return rolled
        if isinstance(value, Input) and value.update is None:
            if value in self._rolled_inputs.values():
                return value
            return self._rolled_input(value)
        raise ValueError(
            f"Minimizer {minimizer!r}: {value.name!r} is computed inside the "
            f"step of ModelyLoop {self.name!r}, where it has no trajectory. "
            "Declare it as an output of the model and minimize that."
        )

    def minimize(
        self,
        name: str,
        source: Output | Stream | str,
        target: Output | Stream | str | float | None = None,
        loss: str | dict[str, Any] | keras.losses.Loss | Callable = "mse",
        gain: float = 1.0,
        seq_weights: Any = None,
    ):
        """Register a loss between two trajectories, as :meth:`Modely.minimize`
        does. ``source`` and ``target`` are given as streams of the step - an
        output of the model, an Input nothing closes - or by name, a number
        for ``target``."""
        return super().minimize(
            name,
            self._trajectory(source, name),
            self._trajectory(target, name),
            loss=loss,
            gain=gain,
            seq_weights=seq_weights,
        )
