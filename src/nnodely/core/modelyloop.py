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
    r"""A :class:`Modely` of a model declared one step at a time, rolled out
    over a sequence of steps: every fed-back Input is fed back at the next
    step.

    Syntax
    ------
    ``ModelyLoop(name, inputs, outputs)``
        Rolls the step out over as many steps as the data has.
    ``ModelyLoop(name, inputs, outputs, seq=n)``
        Rolls it out over ``n`` steps.

    The graph from ``inputs`` to ``outputs`` is one step. A state is an Input
    fed back with :meth:`Input.feedback`: ``x_next = x.feedback(stream)``.
    Inside the step, ``x`` holds the value at the end of the previous step,
    and ``stream`` the value at the end of this one. Every fed-back Input the
    step reads, also inside the models it calls (one per call), is fed back
    together, in one rollout, so states may read each other. A state is
    declared among ``inputs``, or hidden, ``x.feedback(stream,
    hidden=True)``: the model's own memory.

    The model returned is a plain :class:`Modely` of the rollout: its Inputs
    and outputs have the names of those of the step, and the steps as a new
    outermost sequence axis. Every output is a trajectory: its element ``k``
    is computed at step ``k``. Everything else works as in any Modely:
    objectives, :meth:`Modely.build`, :meth:`Modely.train`, calls and
    export. The one-step model it rolls out is ``model.models[0]``, named
    ``<name>_step``.

    Parameters
    ----------
    name : str
        The name of the model. The one-step model is named ``<name>_step``.
    inputs : list of Input
        The Inputs of one step. Every Input the step reads is listed, its
        declared states too, but its hidden states:

        - a declared state starts from the first step of its own data, and
          from zero where the data leaves it out;
        - a hidden state starts from its ``init``, zero without one; so does
          the state of an :class:`Integrate`, a :class:`Derivative` or a
          :class:`Buffer`, from the ``init`` of the layer.

        A model the step calls as a block keeps the feedback of the fed-back
        Inputs it declares: the stream the call gives one is its initial
        value, read at the first step. Every Input has a window: ``tw=1``
        reads one sample per step.
    outputs : list of Output
        The streams of the step to collect over the steps.
    seq : int or None, default None
        The number of steps, at least 1. ``None`` takes as many steps as the
        data has.

    Returns
    -------
    Modely
        The model of the rollout. With ``seq=n``, an Input of the step
        declared with ``seq=s`` is read from data of ``seq=(n, *s)``. On
        arrays, an output of the step of ``Shape(S, T, D)`` gives ``(batch,
        steps, *S, T, *D)``: with ``seq=3``, a step output of ``Shape(S=(),
        T=1, D=(1,))`` gives ``(batch, 3, 1, 1)``.

    Raises
    ------
    ValueError
        If the step reads no fed-back Input ("reads no fed-back Input"). If
        ``seq`` is not a positive int or None ("seq is a positive int"). If a
        listed Input has no window ("declare it with tw=1"). If the step reads
        an Input without listing it, a state neither declared nor hidden too
        ("not among its inputs"), or lists a hidden state ("declares the
        hidden states"). If a listed fed-back Input also gets an initial value
        from the call of a block ("is among the inputs").

    See Also
    --------
    Modely : A model run one sample at a time.
    Loop : A rollout of a model inside a larger graph.
    Input : A signal read from data, fed back with ``feedback()``.
    IntegrateStep : One step of an integral.
    DataLoader : The data of a model, cut into sequences with ``seq_length``.

    Notes
    -----
    With the state :math:`x_k` at the start of step :math:`k`, the input
    :math:`u_k` and the step :math:`f`:

    .. math:: x_{k+1} = f(x_k, u_k), \qquad y_k = x_{k+1}

    where the output :math:`y` collects the stream ``x`` is fed back with,
    and :math:`x_0` is the start of the state.

    :meth:`Modely.minimize` compares the outputs of the model,
    ``model.outputs["x"]`` or their names, which are trajectories, with
    targets declared with the horizon, ``Input("x_meas", seq=-1, tw=1)``. A
    stream of the step is of another graph, and refused.

    Basic Examples
    --------------
    A running total, the hidden state ``total``, fed back with
    ``total + rate``:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, IntegrateStep, ModelyLoop, Modely, Output
    >>> rate = Input("rate", tw=1)
    >>> total = Input("total", tw=1)
    >>> total_next = total.feedback(total + rate, hidden=True)
    >>> counter = ModelyLoop("counter", inputs=[rate], outputs=[Output("total", total_next)])
    >>> counter = counter.build()
    >>> result = counter({"rate": np.ones((1, 3, 1, 1))})  # (batch, steps, T, D)
    >>> keras.ops.convert_to_numpy(result["total"]).ravel()
    array([1., 2., 3.], dtype=float32)

    Scope
    -----
    Without ``seq``, the number of steps follows the data:

    >>> keras.ops.convert_to_numpy(counter({"rate": np.ones((1, 5, 1, 1))})["total"]).ravel()
    array([1., 2., 3., 4., 5.], dtype=float32)

    The model is a :class:`Modely`, and the step model is its first model,
    which reads its states as inputs:

    >>> type(counter) is Modely
    True
    >>> counter.models[0].name, sorted(node.name for node in counter.models[0].inputs)
    ('counter_step', ['rate', 'total'])

    Options
    -------
    ``seq`` fixes the number of steps:

    >>> fixed = ModelyLoop("fixed", inputs=[rate], outputs=[Output("total", total_next)], seq=3)
    >>> fixed.outputs[0].shape
    Shape(S=(3,), T=1, D=(1,))

    A declared state starts from the first step of its data, and from zero
    when the data leaves it out:

    >>> level = Input("level", tw=1)
    >>> level_next = level.feedback(level + rate)
    >>> listed = ModelyLoop("listed", inputs=[rate, level], outputs=[Output("level", level_next)])
    >>> listed = listed.build()
    >>> data = {"rate": np.ones((1, 3, 1, 1)), "level": np.full((1, 3, 1, 1), 10.0)}
    >>> keras.ops.convert_to_numpy(listed(data)["level"]).ravel()
    array([11., 12., 13.], dtype=float32)
    >>> keras.ops.convert_to_numpy(listed({"rate": np.ones((1, 3, 1, 1))})["level"]).ravel()
    array([1., 2., 3.], dtype=float32)

    A hidden state starts from its ``init``, whatever the data holds:

    >>> memory = Input("memory", tw=1)
    >>> memory_next = memory.feedback(memory + rate, init=5.0, hidden=True)
    >>> kept = ModelyLoop("kept", inputs=[rate], outputs=[Output("memory", memory_next)])
    >>> data = {"rate": np.ones((1, 3, 1, 1)), "memory": np.full((1, 3, 1, 1), 10.0)}
    >>> keras.ops.convert_to_numpy(kept.build()(data)["memory"]).ravel()
    array([6., 7., 8.], dtype=float32)

    Applications
    ------------
    A mass of 2 kg pushed by a force, its velocity and position integrated
    with the Euler method, ``dt = 1``:

    >>> force = Input("force", tw=1)
    >>> v, x = Input("v", tw=1), Input("x", tw=1)
    >>> v_next = v.feedback(IntegrateStep(dt=1.0)(v, force / 2.0), hidden=True)
    >>> x_next = x.feedback(IntegrateStep(dt=1.0)(x, v_next), hidden=True)
    >>> mass = ModelyLoop("mass", inputs=[force], outputs=[Output("x", x_next)]).build()
    >>> keras.ops.convert_to_numpy(mass({"force": np.ones((1, 3, 1, 1))})["x"]).ravel()
    array([0.5, 1.5, 3. ], dtype=float32)

    Properties & Relations
    ----------------------
    The running total is the cumulative sum of the rates:

    >>> rates = np.array([1.0, -2.0, 4.0]).reshape(1, 3, 1, 1)
    >>> totals = keras.ops.convert_to_numpy(counter({"rate": rates})["total"]).ravel()
    >>> bool(np.all(totals == np.cumsum(rates)))
    True

    Possible Issues
    ---------------
    A step without a fed-back Input has nothing to roll out. Feed one back,
    or build it as a :class:`Modely`:

    >>> ModelyLoop("plain", inputs=[rate], outputs=[Output("o", 2.0 * rate)])
    Traceback (most recent call last):
        ...
    ValueError: ModelyLoop 'plain' reads no fed-back Input, so it has no feedback...

    A state is declared among the inputs, or hidden. One that is neither is
    refused:

    >>> s = Input("s", tw=1)
    >>> s_next = s.feedback(s + rate)
    >>> ModelyLoop("open", inputs=[rate], outputs=[Output("s", s_next)])
    Traceback (most recent call last):
        ...
    ValueError: Model 'open_step' reads the Inputs ['s'], which are not among its inputs: declare them there...

    A stream of the step has no value in the rollout. The model is
    minimized on its outputs, the trajectories:

    >>> counter.minimize("bad", total_next, Input("measured", seq=-1, tw=1))
    Traceback (most recent call last):
        ...
    ValueError: Minimizer 'bad': its source ... is a stream of the graph of 'counter_step'...
    >>> counter = counter.minimize("fit", "total", Input("measured", seq=-1, tw=1))

    .. versionadded:: 2.0
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
    # leave it out: then from zero.
    listed = {id(node) for node in inputs}
    from_data = [state for state in step.states if step.key(state) in listed]
    for state in from_data:
        if state in step.starts:
            raise ValueError(
                f"ModelyLoop {name!r}: {state.name!r} has an initial "
                "value (the stream a call gives it) and is among the "
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
    model._rolled_inputs = rolled_inputs
    return model
