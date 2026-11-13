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
    r"""A model declared one step at a time, rolled out over a sequence of
    steps: every fed-back Input is fed back at the next step.

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
    step reads is fed back together, in one rollout, so states may read each
    other.

    The rollout adds an outermost sequence axis, the steps, to every Input
    and every output. Every output is a trajectory: its element ``k`` is
    computed at step ``k``. Everything else works as in a :class:`Modely`:
    objectives, :meth:`Modely.build`, :meth:`Modely.train`, calls and
    export.

    Parameters
    ----------
    name : str
        The name of the model. The one-step model is named ``<name>_step``.
    inputs : list of Input
        The Inputs of one step. Every Input the step reads that nothing feeds
        back is listed. A fed-back Input may be listed too:

        - listed, it starts from the first step of its own data, and from
          zero where the data leaves it out;
        - not listed, it starts from zero.

        The state of an :class:`Integrate`, a :class:`Derivative` or a
        :class:`Buffer` starts from the ``init`` of the layer. Every Input
        has a window: ``tw=1`` reads one sample per step.
    outputs : list of Output
        The streams of the step to collect over the steps.
    seq : int or None, default None
        The number of steps, at least 1. ``None`` takes as many steps as the
        data has.

    Attributes
    ----------
    body : Modely
        The one-step model the rollout repeats, named ``<name>_step``.

    Returns
    -------
    dict of str to tensor, Stream or list of Stream
        As for a :class:`Modely`. On arrays, an output of the step of
        ``Shape(S, T, D)`` gives ``(batch, steps, *S, T, *D)``: with
        ``seq=3``, a step output of ``Shape(S=(), T=1, D=(1,))`` gives
        ``(batch, 3, 1, 1)``. Every Input is read the same way, with the
        steps after the batch axis.

    Raises
    ------
    ValueError
        If the step reads no fed-back Input ("reads no fed-back Input"). If
        ``seq`` is not a positive int or None ("seq is a positive int"). If a
        listed Input has no window ("declare it with tw=1"). If the step reads
        an Input nothing feeds back without listing it ("not among its
        inputs"). If a listed fed-back Input also gets an initial value from
        the call of a block ("is among the inputs").

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
    and :math:`x_0` is the start of the state. A model
    the step calls as a block keeps the feedback of the fed-back Inputs it
    declares: the stream the call gives one is its initial value, read at
    the first step.

    :meth:`minimize` takes streams of the step: an output of the model, or an
    Input nothing feeds back, such as a measured target. It compares their
    trajectories.

    Basic Examples
    --------------
    A running total, the state ``total``, fed back with ``total + rate``:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, IntegrateStep, ModelyLoop, Output
    >>> rate = Input("rate", tw=1)
    >>> total = Input("total", tw=1)
    >>> total_next = total.feedback(total + rate)
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

    The step model is ``body``:

    >>> counter.body.name, sorted(node.name for node in counter.body.inputs)
    ('counter_step', ['rate', 'total'])

    Options
    -------
    ``seq`` fixes the number of steps:

    >>> fixed = ModelyLoop("fixed", inputs=[rate], outputs=[Output("total", total_next)], seq=3)
    >>> fixed.outputs[0].shape
    Shape(S=(3,), T=1, D=(1,))

    A listed state starts from the first step of its data, and from zero
    when the data leaves it out:

    >>> listed = ModelyLoop("listed", inputs=[rate, total], outputs=[Output("total", total_next)])
    >>> listed = listed.build()
    >>> data = {"rate": np.ones((1, 3, 1, 1)), "total": np.full((1, 3, 1, 1), 10.0)}
    >>> keras.ops.convert_to_numpy(listed(data)["total"]).ravel()
    array([11., 12., 13.], dtype=float32)
    >>> keras.ops.convert_to_numpy(listed({"rate": np.ones((1, 3, 1, 1))})["total"]).ravel()
    array([1., 2., 3.], dtype=float32)

    Applications
    ------------
    A mass of 2 kg pushed by a force, its velocity and position integrated
    with the Euler method, ``dt = 1``:

    >>> force = Input("force", tw=1)
    >>> v, x = Input("v", tw=1), Input("x", tw=1)
    >>> v_next = v.feedback(IntegrateStep(dt=1.0)(v, force / 2.0))
    >>> x_next = x.feedback(IntegrateStep(dt=1.0)(x, v_next))
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

    .. versionadded:: 2.0
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
        # The Input of the rollout of each Input of the step, by the Input it
        # stands for (_Step.key).
        self._rolled_inputs: dict[int, Input] = {}

        # The step, and the one Loop that feeds back every Input it feeds back.
        step = _Step(
            Modely(f"{name}_step", inputs=inputs, outputs=outputs),
            {},
            who=f"ModelyLoop {name!r}",
            among="its inputs",
        )
        self._step = step
        if not step.pairs:
            raise ValueError(
                f"ModelyLoop {name!r} reads no fed-back Input, so it has no "
                "feedback: feed one back with x.feedback(stream), or build it "
                "as a Modely."
            )
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
        loop = declaration({node: self._rolled_input(node) for node in read})
        self.body = declaration._model

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
        self._optional_inputs = {self._rolled_input(state) for state in from_data}

    def _rolled_input(self, node: Input) -> Input:
        """The Input of the rollout for an Input of the step: the same name,
        the horizon its new outermost sequence axis."""
        key = self._step.key(node)
        rolled = self._rolled_inputs.get(key)
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
            self._rolled_inputs[key] = rolled
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
        source: Output | Stream | str,
        target: Output | Stream | str | float | None = None,
        loss: str | dict[str, Any] | keras.losses.Loss | Callable = "mse",
        gain: float = 1.0,
    ):
        """Register an objective that compares two trajectories, and return
        the model itself.

        It works as :meth:`Modely.minimize`, with streams of the step: each is
        compared over every step of the rollout.

        Parameters
        ----------
        name : str
            The name of the objective, logged as ``"loss_<name>"``.
        source : Output, Stream or str
            An output of the model, or its name.
        target : Output, Stream, str, float or None, default None
            An output of the model, an Input nothing feeds back, such as a
            measured target, the name of one, or a number. ``None`` is
            ``0.0``. An Input given here is read with the steps, as the
            inputs are.
        loss : str, dict, keras.losses.Loss or callable, default "mse"
            The loss function, as :meth:`Modely.minimize` takes it.
        gain : float, default 1.0
            The weight of the objective in the total loss, not below zero.

        Returns
        -------
        ModelyLoop
            The model itself.

        Raises
        ------
        ValueError
            If a side is a stream computed inside the step that is not an
            output ("has no trajectory"): declare it as an output. Also the
            errors of :meth:`Modely.minimize`.

        See Also
        --------
        Modely.minimize : The objectives of a model.
        Modely.train : Minimize the objectives.

        Basic Examples
        --------------
        >>> from nnodely import Input, ModelyLoop, Output
        >>> rate, total = Input("rate", tw=1), Input("total", tw=1)
        >>> total_out = Output("total", total.feedback(total + rate))
        >>> counter = ModelyLoop("counter", inputs=[rate], outputs=[total_out])
        >>> counter = counter.minimize("track", "total", Input("measured", tw=1))
        >>> print(counter.minimizers)
        ┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
        ┃ idx ┃ minimizer ┃ source ┃ target   ┃ loss               ┃ gain ┃
        ┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
        │ 0   │ track     │ total  │ measured │ mean_squared_error │ 1.0  │
        └─────┴───────────┴────────┴──────────┴────────────────────┴──────┘
        >>> counter.minimizers[0].target.shape
        Shape(S=(dyn,), T=1, D=(1,))

        Possible Issues
        ---------------
        A stream computed inside the step has no trajectory. Declare it as an
        output:

        >>> counter.minimize("inner", 2.0 * rate, 0.0)
        Traceback (most recent call last):
            ...
        ValueError: Minimizer 'inner': ... is computed inside the step of ModelyLoop 'counter', where it has no trajectory...

        .. versionadded:: 2.0
        """
        return super().minimize(
            name,
            self._trajectory(source, name),
            self._trajectory(target, name),
            loss=loss,
            gain=gain,
        )
