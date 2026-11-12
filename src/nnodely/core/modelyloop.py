"""ModelyLoop: a model whose fed-back Inputs are fed back by a single rollout."""

from __future__ import annotations

from typing import Any, Callable

import keras

from nnodely.core.modely import Modely
from nnodely.core.stream import Node, Stream
from nnodely.layers.input import Input
from nnodely.layers.loop import Loop, _Step
from nnodely.layers.output import Output


class ModelyLoop(Modely):
    """A model declared one step at a time, whose fed-back Inputs are all fed
    back by a single rollout::

        x, v = Input("x", tw=1), Input("v", tw=1)
        a = (F - k * x - c * v) / m
        v_next = v.feedback(IntegrateStep(dt=dt)(v, a))
        x_next = x.feedback(IntegrateStep(dt=dt)(x, v_next))
        msd = ModelyLoop("msd", inputs=[F, x], outputs=[Output("x", x_next)])

    The graph is the body of a :class:`Loop`: its Inputs read one step of
    their signal, and every Input it reads that is fed back with
    :meth:`Input.feedback` - directly, or inside the models it calls, one per
    call - is fed back with that stream. They are fed back together, so the
    model is one Loop however many feedbacks it holds, and any fed-back Input
    may read any other.

    Every Input the step reads and nothing feeds back is among ``inputs``. A
    fed-back Input starts from the first window of its own data when it is
    among ``inputs`` - zero where the data leaves it out - and from zero when
    it is not, as ``v`` above. The state of an :class:`Integrate`, a
    :class:`Derivative` or a :class:`Buffer` starts from the init of the
    layer. A model the step calls as a block keeps the feedback of the
    fed-back Inputs it declares: the stream the call gives one is its initial
    value, read at the first step.

    ``seq`` is the number of steps, and the outermost sequence axis the
    rollout adds to every Input and every output: with ``seq=n``, an Input of
    the step declared with ``seq=s`` is read from data of ``seq=(n, *s)``.
    None leaves that axis dynamic, as many steps as the data has. Every output
    is a trajectory: its element ``k`` is computed at step ``k``. A fed-back
    Input that starts from data, or from an init that is an Input, reads its
    first step.

    :meth:`minimize` takes the streams of the step - an output of the model,
    or an Input nothing feeds back - and compares their trajectories.
    ``body`` is the one-step model the loop rolls out.
    """

    def __init__(
        self,
        name: str,
        inputs: list[Input],
        outputs: list[Output],
        seq: int | None = None,
    ) -> None:
        if seq is not None and (
            isinstance(seq, bool) or not isinstance(seq, int) or seq < 1
        ):
            raise ValueError(
                f"ModelyLoop {name!r}: seq is a positive int, or None for the "
                f"length of the data, got {seq!r}."
            )
        self.name = name
        self._length = seq
        self._rolled_inputs: dict[str, Input] = {}

        # The step, and the one Loop that feeds back every Input it feeds back.
        step = _Step(
            Modely(f"{name}_step", inputs=inputs, outputs=outputs),
            {},
            who=f"ModelyLoop {name!r}",
            among="its inputs",
        )
        if not step.pairs:
            raise ValueError(
                f"ModelyLoop {name!r} reads no fed-back Input, so it has no "
                "feedback: feed one back with x.feedback(stream), or build it "
                "as a Modely."
            )
        # A fed-back Input among the inputs starts from its data, which may
        # leave it out: then, as when it is not listed, from zero.
        listed = {node.name for node in inputs}
        from_data = [state for state in step.states if state.name in listed]
        for state in from_data:
            if state in step.starts:
                raise ValueError(
                    f"ModelyLoop {name!r}: {state.name!r} has an initial "
                    "value - the stream a call gives it - and is among the "
                    "inputs, which would start it from its data: leave it out "
                    "of inputs, or give it no initial value."
                )
        declaration = Loop._of(
            step,
            init={state: self._rolled_input(state) for state in from_data},
            name=f"{name}_loop",
            seq=seq,
        )
        read = [*declaration._static_inputs, *declaration._seed_inputs]
        if not read and not from_data:
            raise ValueError(
                f"ModelyLoop {name!r} reads no Input, in its steps or as the "
                "initial value of a fed-back Input, so no data sets its batch "
                "and number of steps."
            )
        loop = declaration({node.name: self._rolled_input(node) for node in read})
        self.body = declaration._f

        # One trajectory per output of the body, the declared outputs first.
        trajectories = list(loop)
        self._trajectories: dict[Node, Output] = {}
        rolled_outputs = []
        for output, trajectory in zip(outputs, trajectories):
            rolled = Output(output.name, trajectory)
            self._trajectories[output] = rolled
            self._trajectories.setdefault(output._preds[0], rolled)
            rolled_outputs.append(rolled)

        super().__init__(
            name,
            inputs=[self._rolled_input(node) for node in inputs],
            outputs=rolled_outputs,
        )
        self.optional_inputs = {state.name for state in from_data}

    def _rolled_input(self, node: Input) -> Input:
        """The Input of the rollout for an Input of the step: the same name,
        the horizon its new outermost sequence axis."""
        rolled = self._rolled_inputs.get(node.name)
        if rolled is None:
            if node.tw is None:
                raise ValueError(
                    f"ModelyLoop {self.name!r} reads its Input {node.name!r} one "
                    "step at a time, so its data carries the horizon as a "
                    "sequence axis, which needs a window: declare it with tw=1."
                )
            horizon = -1 if self._length is None else self._length
            rolled = Input(
                node.name,
                dim=node.dim,
                seq=(horizon, *(-1 if size is None else size for size in node.seq)),
                tw=(node.past, node.future),
            )
            self._rolled_inputs[node.name] = rolled
        return rolled

    def _output_stream(self, output: Output) -> Node:
        """The stream of the step whose trajectory ``output`` is."""
        step = next(
            node for node, rolled in self._trajectories.items() if rolled is output
        )
        return step._preds[0]

    def _trajectory(self, value, minimizer: str):
        """The stream of the rollout a minimizer side of the step compares."""
        if not isinstance(value, Stream) or value._is_value or value in self._order:
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
        source: Output | Stream | str | tuple[str, int],
        target: Output | Stream | str | tuple[str, int] | float | None = None,
        loss: str | dict[str, Any] | keras.losses.Loss | Callable = "mse",
        gain: float = 1.0,
    ):
        """Register a loss between two trajectories, as :meth:`Modely.minimize`
        does. ``source`` and ``target`` are given as streams of the step - an
        output of the model, an Input nothing feeds back - or by name, a number
        for ``target``."""
        return super().minimize(
            name,
            self._trajectory(source, name),
            self._trajectory(target, name),
            loss=loss,
            gain=gain,
        )
