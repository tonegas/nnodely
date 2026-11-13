"""ModelyLoop: a model whose fed-back Inputs are fed back by a single rollout."""

from __future__ import annotations

from nnodely.core.modely import Modely
from nnodely.layers.input import Input
from nnodely.layers.loop import Loop, _Step
from nnodely.layers.output import Output


def ModelyLoop(
    name: str,
    inputs: list[Input],
    outputs: list[Output],
    seq: int | None = None,
) -> Modely:
    """A model declared one step at a time, whose fed-back Inputs are all fed
    back by a single rollout::

        x, v = Input("x", tw=1), Input("v", tw=1)
        a = (F - k * x - c * v) / m
        v_next = v.feedback(IntegrateStep(dt=dt)(v, a), hidden=True)
        x_next = x.feedback(IntegrateStep(dt=dt)(x, v_next))
        msd = ModelyLoop("msd", inputs=[F, x], outputs=[Output("x", x_next)])

    The graph is the body of a :class:`Loop`: its Inputs read one step of
    their signal, and every Input it reads that is fed back with
    :meth:`Input.feedback` - directly, or inside the models it calls, one per
    call - is fed back with that stream. They are fed back together, so the
    model is one Loop however many feedbacks it holds, and any fed-back Input
    may read any other.

    Every Input the step reads is among ``inputs``, but its hidden states. A
    declared fed-back Input, as ``x`` above, starts from the first window of
    its own data - zero where the data leaves it out. A hidden one, as ``v``,
    is the model's own and starts from its init, zero without one; so does
    the state of an :class:`Integrate`, a :class:`Derivative` or a
    :class:`Buffer`, from the init of the layer. A model the step calls as a
    block keeps the feedback of the fed-back Inputs it declares: the stream
    the call gives one is its initial value, read at the first step.

    ``seq`` is the number of steps, and the outermost sequence axis the
    rollout adds to every Input and every output: with ``seq=n``, an Input of
    the step declared with ``seq=s`` is read from data of ``seq=(n, *s)``.
    None leaves that axis dynamic, as many steps as the data has. Every output
    is a trajectory: its element ``k`` is computed at step ``k``. A fed-back
    Input that starts from data, or from an init that is an Input, reads its
    first step.

    The model returned is a :class:`Modely` of the rollout: its Inputs and
    outputs have the names of those of the step, and the horizon as their
    outermost sequence axis. Its :meth:`~Modely.minimize` takes the streams
    of the step too - an output, an Input nothing feeds back - each standing
    for its trajectory, and a target declared one step at a time,
    ``Input("x_meas", tw=1)``, reads one sample per step. The one-step model
    the loop rolls out is ``model.models[0]``.
    """
    if seq is not None and (
        isinstance(seq, bool) or not isinstance(seq, int) or seq < 1
    ):
        raise ValueError(
            f"ModelyLoop {name!r}: seq is a positive int, or None for the "
            f"length of the data, got {seq!r}."
        )

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

    # The Input of the rollout of each Input of the step, by the Input it
    # stands for (_Step.key): the same name, the horizon its new outermost
    # sequence axis.
    rolled_inputs: dict[int, Input] = {}

    def rolled_input(node: Input) -> Input:
        key = step.key(node)
        rolled = rolled_inputs.get(key)
        if rolled is None:
            if node.tw is None:
                raise ValueError(
                    f"ModelyLoop {name!r} reads its Input {node.name!r} one "
                    "step at a time, so its data carries the horizon as a "
                    "sequence axis, which needs a window: declare it with tw=1."
                )
            horizon = -1 if seq is None else seq
            rolled = Input(
                node.name,
                dim=node.dim,
                seq=(horizon, *(-1 if size is None else size for size in node.seq)),
                tw=(node.past, node.future),
            )
            rolled_inputs[key] = rolled
        return rolled

    # A fed-back Input among the inputs starts from its data, which may
    # leave it out: then, as when it is not listed, from zero.
    listed = {id(node) for node in inputs}
    from_data = [state for state in step.states if step.key(state) in listed]
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
        init={state: rolled_input(state) for state in from_data},
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
    loop = declaration({node: rolled_input(node) for node in read})

    # One trajectory per output of the body, the declared outputs first.
    model = Modely(
        name,
        inputs=[rolled_input(node) for node in inputs],
        outputs=[
            Output(output.name, trajectory) for output, trajectory in zip(outputs, loop)
        ],
    )
    model._optional_inputs = {rolled_input(state) for state in from_data}
    return model
