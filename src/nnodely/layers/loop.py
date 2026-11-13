import contextlib
import copy
from typing import Any, cast

import keras
from keras.src.backend.common import stateless_scope

from nnodely.core.dag import (
    _copies,
    Scope,
    _streams_from,
    by_name_or_index,
    flatten_fed_back,
    flatten_node,
    resolve,
    toposort,
)
from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.core.shape import (
    BATCH_AXIS,
    OUTER_AXIS,
    _same_steps,
    slice_axis,
    transpose,
)
from nnodely.core.stream import Stream, object_repr
from nnodely.layers.input import Input
from nnodely.layers.output import Output


#: Steps a dynamic rollout is declared with when no ``seq`` is given. It
#: only shapes the graph: the rollout follows the data it is given.
DYNAMIC_BUILD_LENGTH = 1


class _SeveralOutputs(ValueError, AttributeError):
    """A Loop with several outputs read as one stream. It has no one shape:
    an AttributeError too, so what only looks for a shape - ``getattr(node,
    "shape", None)`` - finds none."""


@keras.saving.register_keras_serializable(package="nnodely")
class LoopOutputImpl(keras.layers.Layer):
    """Select one tensor from a multi-output Loop."""

    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[self.index]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


class LoopOutput(Layer):
    def __init__(self, loop, index: int, output_node, result_shape):
        self._index = int(index)
        super().__init__(
            name=f"{loop.name}_{output_node.name}",
            preds=[loop],
            dim=output_node.dim,
            tw=result_shape.tw,
            seq=result_shape.seq,
            index=self._index,
        )

    def build_layer(self):
        return LoopOutputImpl(index=self._index, name=self.name)

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The outputs belong to the loop that was just rebuilt, which returns
        # them all.
        return preds[0]._loop_outputs[config["index"]]  # type: ignore[index]


@keras.saving.register_keras_serializable(package="nnodely")
class LoopImpl(keras.layers.Layer):
    """Apply a Keras model recurrently with one or more feedback states.

    The rollout runs along the outermost sequence axis of the inputs that
    carry it - the ``rolled`` ones: such an input has one more sequence axis
    than the body input it feeds, so every step consumes a single slice of it
    and the body is evaluated exactly once per step. Collected, the steps
    become the outermost sequence axis of every result.

    The body runs through a backend loop primitive - ``tf.while_loop`` on
    TensorFlow, ``keras.ops.scan`` on the others - so on TensorFlow and JAX the
    traced graph holds the body once whatever the rollout length, which keeps
    tracing and XLA compilation fast for long rollouts. Exported from
    TensorFlow, ONNX holds it as a ``Loop`` node.
    The carry holds every body output followed by one window per shifted
    feedback state.
    """

    def __init__(
        self,
        model,
        feedback_input_names: tuple[str, ...],
        feedback_output_names: tuple[str, ...],
        model_output_names: tuple[str, ...],
        static_input_names: tuple[str, ...],
        horizon: int,
        collect: bool = True,
        rolled_initial: tuple[bool, ...] = (),
        rolled_static: tuple[bool, ...] = (),
        feedback_shift_axes: tuple[int | None, ...] = (),
        feedback_state_shapes: tuple[tuple[int | None, ...] | None, ...] = (),
        result_shapes: tuple[tuple[int | None, ...], ...] = (),
        horizon_index: int | None = None,
        primary_output_index: int = 0,
        return_all_outputs: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.feedback_input_names = tuple(feedback_input_names)
        self.feedback_output_names = tuple(feedback_output_names)
        self.model_output_names = tuple(model_output_names)
        self.static_input_names = tuple(static_input_names)
        self.horizon = int(horizon)
        self.collect = bool(collect)
        self.rolled_initial = tuple(bool(rolled) for rolled in rolled_initial)
        self.rolled_static = tuple(bool(rolled) for rolled in rolled_static)
        self.feedback_shift_axes = tuple(feedback_shift_axes)
        self.feedback_state_shapes = tuple(
            None
            if shape is None
            else tuple(None if size is None else int(size) for size in shape)
            for shape in feedback_state_shapes
        )
        self.result_shapes = tuple(tuple(shape) for shape in result_shapes)
        self.horizon_index = horizon_index
        self.primary_output_index = int(primary_output_index)
        self.return_all_outputs = bool(return_all_outputs)
        self.output_indices = {
            name: index for index, name in enumerate(self.model_output_names)
        }

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _swap_steps(value):
        """Swap the batch and the outer axis: the steps to the front for the
        loop, and back after it."""
        rank = len(value.shape)
        return transpose(value, [OUTER_AXIS, BATCH_AXIS, *range(OUTER_AXIS + 1, rank)])

    @staticmethod
    def _newest_slot(window, axis):
        return slice_axis(window, axis, slice(-1, None))

    @staticmethod
    def _shift_window(window, value, axis):
        """Drop the oldest step of the window and append the new one."""
        return keras.ops.concatenate(
            [slice_axis(window, axis, slice(1, None)), value], axis=axis
        )

    @staticmethod
    def _align_to_shape(state, shape):
        """Broadcast an initial value ``(batch, ...)`` up to ``(batch, *shape)``.

        Its axes line up with the last ones of ``shape``, the way the value was
        checked against its fed-back input when the loop was declared. An axis
        of ``shape`` sized at run time, None, is left as the value has it.
        """
        while len(state.shape) < len(shape) + 1:
            state = keras.ops.expand_dims(state, axis=1)
        sizes = tuple(1 if size is None else size for size in shape)
        return state + keras.ops.zeros((1, *sizes))

    def _model_outputs(self, model_inputs):
        # Only the inputs the body reads: Keras matches a dict holding more by
        # position, feeding one input the value of another.
        taken = {tensor.name for tensor in self.model.inputs}
        outputs = self.model(
            {name: value for name, value in model_inputs.items() if name in taken}
        )
        if isinstance(outputs, dict):
            return tuple(outputs[name] for name in self.model_output_names)
        if isinstance(outputs, (list, tuple)):
            return tuple(outputs)
        return (outputs,)

    def _resolve_horizon(self, inputs):
        """Number of rollout steps.

        A rollout axis declared as ``None`` follows the sequence it is actually
        given, which is what makes a longer input roll out further; a declared
        width always wins over the tensor. Traced without a known width - an
        export, or a retrace with relaxed shapes - TensorFlow reads it at run
        time, since ``tf.while_loop`` takes its trip count as a tensor; the
        other backends trace with concrete shapes and fall back on the build
        length.
        """
        if self.horizon_index is not None:
            value = inputs[self.horizon_index]
            width = value.shape[OUTER_AXIS]
            if isinstance(width, int):
                return width
            if keras.backend.backend() == "tensorflow":
                import tensorflow as tf

                return tf.shape(value)[OUTER_AXIS]
        return self.horizon

    def compute_output_spec(self, inputs):
        """Declare the outputs instead of letting Keras trace `call`.

        The shapes are known from the graph, so the whole rollout does not have
        to be traced once more just to infer them, and a dynamic rollout keeps
        its declared length here: the tensor it is given decides the real one.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        batch = inputs[0].shape[0]
        specs = tuple(
            keras.KerasTensor(shape=(batch, *shape), dtype=self.compute_dtype)
            for shape in self.result_shapes
        )
        if self.return_all_outputs:
            return specs
        return specs[self.primary_output_index]

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        feedback_count = len(self.feedback_input_names)
        initial_values = inputs[:feedback_count]
        sources = list(inputs[feedback_count:])

        horizon = self._resolve_horizon(inputs)

        # Inputs carrying the rollout are consumed one slice of their outer
        # time per step; the others are closed over, so they are not tiled
        # horizon times.
        xs = []
        xs_positions = {}
        for index, (value, rolled) in enumerate(zip(sources, self.rolled_static)):
            if not rolled:
                continue
            xs_positions[index] = len(xs)
            xs.append(self._swap_steps(value)[:horizon])

        states = []
        windows = []
        for initial, rolled, shift_axis, state_shape in zip(
            initial_values,
            self.rolled_initial,
            self.feedback_shift_axes,
            self.feedback_state_shapes,
        ):
            state = initial
            if rolled:
                # Seeded by the first step of its outermost sequence axis.
                state = slice_axis(state, OUTER_AXIS, 0)
            if state_shape is not None:
                state = self._align_to_shape(state, state_shape)

            if shift_axis is None:
                states.append(state)
                windows.append(None)
            else:
                # The carry leaf that is fed back must match the body output, so
                # the window travels next to it and only its newest slot seeds
                # the output leaf.
                windows.append(state)
                states.append(self._newest_slot(state, shift_axis))

        def compute(step_states, step_windows, x_step):
            model_inputs = {
                name: state if window is None else window
                for name, state, window in zip(
                    self.feedback_input_names, step_states, step_windows
                )
            }
            for index, name in enumerate(self.static_input_names):
                position = xs_positions.get(index)
                model_inputs[name] = (
                    sources[index] if position is None else x_step[position]
                )
            return self._model_outputs(model_inputs)

        # A dynamic axis an initial value does not carry - a zero, a number -
        # is as long as the value fed back, which the body computes from the
        # inputs that carry it: one evaluation, the state broadcast along that
        # axis by the layer that reads it, tells how long.
        unsized = [
            index
            for index, shape in enumerate(self.feedback_state_shapes)
            if shape is not None and None in shape
        ]
        if unsized:
            first = compute(states, windows, [value[0] for value in xs])
            for index in unsized:
                name = self.feedback_output_names[index]
                reference = keras.ops.zeros_like(first[self.output_indices[name]])
                if windows[index] is None:
                    states[index] = states[index] + reference
                else:
                    # One sample, broadcast along the window too.
                    windows[index] = windows[index] + reference
                    states[index] = self._newest_slot(
                        windows[index], self.feedback_shift_axes[index]
                    )

        output_count = len(self.model_output_names)
        feedback_slots = {
            self.output_indices[name]: index
            for index, name in enumerate(self.feedback_output_names)
        }
        shifted = [
            index
            for index, axis in enumerate(self.feedback_shift_axes)
            if axis is not None
        ]

        # Outputs that are not fed back still need a carry leaf of the right
        # shape and dtype; one extra evaluation of the body provides it.
        seeds = (
            None
            if len(feedback_slots) == output_count
            else compute(states, windows, [value[0] for value in xs])
        )
        init_carry = tuple(
            states[feedback_slots[index]] if index in feedback_slots else seeds[index]  # type: ignore
            for index in range(output_count)
        ) + tuple(windows[index] for index in shifted)

        def step(carry, x_step):
            outputs = carry[:output_count]
            step_states = [
                outputs[self.output_indices[name]]
                for name in self.feedback_output_names
            ]
            step_windows = [None] * feedback_count
            for slot, index in enumerate(shifted):
                step_windows[index] = carry[output_count + slot]

            step_outputs = compute(step_states, step_windows, x_step)
            return step_outputs + tuple(
                self._shift_window(
                    step_windows[index],
                    step_outputs[
                        self.output_indices[self.feedback_output_names[index]]
                    ],
                    self.feedback_shift_axes[index],
                )
                for index in shifted
            )

        carry, trajectory = self._rollout(step, init_carry, xs, horizon, output_count)

        if self.collect:
            results = tuple(self._swap_steps(steps) for steps in trajectory)
        else:
            results = tuple(carry[:output_count])

        if self.return_all_outputs:
            return results
        return results[self.primary_output_index]

    def _rollout(self, step, init_carry, xs, horizon, output_count):
        """Run ``step`` ``horizon`` times through a backend loop primitive.

        Returns the final carry and, when collecting, the body outputs of every
        step stacked on a leading axis. On TensorFlow the loop is written by
        hand: ``keras.ops.scan`` leaves ``maximum_iterations`` unset, which XLA
        needs to compile the gradient, and requires every per-step output to
        match the carry.

        On JAX the body's non-trainable variables (e.g. BatchNorm moving
        statistics) travel in the carry: assigning them inside ``scan`` would
        leak a scan tracer out of its trace. The parent stateless scope is
        reused, since a nested one would hide the trainer's variable values.
        """
        if keras.backend.backend() != "tensorflow":
            jax = keras.backend.backend() == "jax"
            state = self.model.non_trainable_variables if jax else []
            # Torch assigns in place, so only JAX needs a scope to record them.
            own_scope = jax and not stateless_scope.in_stateless_scope()

            def scan_step(carry, x_step):
                carry, values = carry
                for variable, value in zip(state, values):
                    variable.assign(value)
                carry = step(carry, x_step)
                values = [variable.value for variable in state]
                return (carry, values), carry[:output_count] if self.collect else None

            with keras.StatelessScope() if own_scope else contextlib.nullcontext():
                (carry, values), trajectory = keras.ops.scan(
                    scan_step,
                    (init_carry, [variable.value for variable in state]),
                    xs=xs if xs else None,
                    length=horizon,
                )
            for variable, value in zip(state, values):
                variable.assign(value)
            return carry, trajectory

        import tensorflow as tf

        xs_arrays = [
            tf.TensorArray(
                value.dtype, size=horizon, element_shape=value.shape[1:]
            ).unstack(value)
            for value in xs
        ]
        ys_arrays = [
            tf.TensorArray(value.dtype, size=horizon, element_shape=value.shape)
            for value in (init_carry[:output_count] if self.collect else ())
        ]

        def body(index, carry, ys_arrays):
            carry = step(carry, [array.read(index) for array in xs_arrays])
            return (
                index + 1,
                carry,
                [array.write(index, value) for array, value in zip(ys_arrays, carry)],
            )

        _, carry, ys_arrays = tf.while_loop(
            lambda index, *_: index < horizon,
            body,
            (0, init_carry, ys_arrays),
            maximum_iterations=horizon,
        )
        return carry, tuple(array.stack() for array in ys_arrays)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": keras.saving.serialize_keras_object(self.model),
                "feedback_input_names": self.feedback_input_names,
                "feedback_output_names": self.feedback_output_names,
                "model_output_names": self.model_output_names,
                "static_input_names": self.static_input_names,
                "horizon": self.horizon,
                "collect": self.collect,
                "rolled_initial": self.rolled_initial,
                "rolled_static": self.rolled_static,
                "feedback_shift_axes": self.feedback_shift_axes,
                "feedback_state_shapes": self.feedback_state_shapes,
                "result_shapes": self.result_shapes,
                "horizon_index": self.horizon_index,
                "primary_output_index": self.primary_output_index,
                "return_all_outputs": self.return_all_outputs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        return cls(**config)


def _fed_back(roots) -> list[Input]:
    """Every fed-back Input ``roots`` read, and every one the streams they are
    fed back with read in turn, in the order they are reached."""
    found: list[Input] = []
    seen: set[int] = set()
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Input) and node.update is not None:
            found.append(node)
            stack.append(node.update)
        stack.extend(reversed(node._preds))
    return found


def _fed_back_across_calls(
    scope: Scope, enclosing: frozenset = frozenset()
) -> list[tuple[str, str]]:
    """The name of every fed-back Input a call inlines while a graph enclosing
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
            (node.name, model_call._model.name)
            for node in call_scope.memo
            if node in here
        ]
        found += _fed_back_across_calls(call_scope, here)
    return sorted(found)


def _free_name(name: str, taken: set[str]) -> str:
    """``name``, or the first ``name_<i>`` not in ``taken``."""
    candidate, index = name, 2
    while candidate in taken:
        candidate, index = f"{name}_{index}", index + 1
    taken.add(candidate)
    return candidate


def _inlined_inputs(scope: Scope) -> dict[int, Input]:
    """The Input every copy of one made inlining a graph stands for, by the
    id of the copy - in ``scope`` and the scope of every call it makes: a
    fed-back Input inside a called model is copied once per call."""
    found = {
        id(copied): node
        for node, copied in scope.memo.items()
        # A copy shares the Keras input of its Input; an Input of a called
        # model bound to another Input of the caller is that one instead.
        if isinstance(node, Input)
        and isinstance(copied, Input)
        and copied is not node
        and copied._input is node._input
    }
    for call in scope.calls.values():
        found.update(_inlined_inputs(call))
    return found


class _Step:
    """The body a Loop rolls out over the model ``f``, and its states: the
    Inputs ``feedback`` maps to an output of ``f``, then every Input the graph
    of ``f`` feeds back itself - inside the models it calls too, one per call
    - each with the output it is fed back from and what it starts from.

    The body is ``f`` when ``f`` declares every state and computes each update
    as an output. Otherwise it is the graph of ``f`` with those Inputs
    declared and an output added for every update that is not one, built as a
    Modely of the same name; ``returns`` are its outputs that ``f`` declares.
    ``who`` and ``among`` name, in the errors, what rolls ``f`` out and the
    inputs it reads.
    """

    def __init__(self, f: Modely, feedback: dict, who: str, among: str):
        self.f = f
        explicit = [
            (
                Loop._find_node(f._inputs, key, "input"),
                Loop._find_node(f._outputs, value, "output"),
            )
            for key, value in feedback.items()
        ]
        if len({id(output) for _, output in explicit}) != len(explicit):
            raise ValueError(
                "Loop cannot feed the same output back into more than one input."
            )

        # The step, with every model it calls inlined: a fed-back Input inside
        # one is copied once per call, together with the stream it is fed
        # back with.
        scope = Scope(pending=[], loop=True)
        outputs = [flatten_node(output, scope) for output in f._outputs]
        flatten_fed_back(scope)
        crossing = _fed_back_across_calls(scope)
        if crossing:
            state, model = crossing[0]
            raise ValueError(
                f"{who} reads the fed-back Input {state!r} both in the graph "
                f"that calls {model!r} and inside it, where {model!r} does not "
                "declare it. A fed-back Input inside a called model is one "
                "state per call, so these would be two states. To read the "
                f"state of the call, make it an output of {model!r}; to give "
                f"{model!r} the state of the caller, declare an Input in "
                f"{model!r} for it and pass {state!r} in the call."
            )
        found = _fed_back(outputs)
        # The Inputs and values read from the graph of f are copies; the
        # streams of the graph the Loop is called in are the originals. So
        # are the Inputs f declares, a fed-back one copied for the step.
        self._originals = {id(copy): source for source, copy in scope.values.items()}
        declared = {
            id(scope.memo[node]): node for node in f._inputs if node in scope.memo
        }
        self._originals.update(declared)
        # Every Input copied, inside the models f calls too, by its copy; and
        # every node, for what reads the graph by its objects.
        self._sources = _inlined_inputs(scope)
        self._copies = _copies(scope)

        # The states feedback= gives first, in its order. One the graph feeds
        # back too is fed back with the same stream, or with none.
        pairs: list[tuple[Input, Output]] = []
        for node, output in explicit:
            state = scope.memo.get(node, node)
            fed = outputs[f._outputs.index(output)]
            if state in found and state.update not in (fed, fed._preds[0]):
                raise ValueError(
                    f"{who}: {node.name!r} is fed back with "
                    f"{state.update.name!r} in the graph of {f.name!r}, and "
                    f"feedback= gives it {output.name!r}: an Input is fed back "
                    "with one stream."
                )
            pairs.append((state, fed))
        given = {id(state) for state, _ in pairs}
        found = [node for node in found if id(node) not in given]

        # A model called twice holds a fed-back Input of each name per call;
        # the body reads them by name, so the copies after the first are
        # renamed.
        names: set[str] = {state.name for state, _ in pairs}
        renamed = False
        for node in found:
            if node.name in names:
                node.name = _free_name(node.name, names)
                node._input = keras.Input(shape=node.shape.tuple, name=node.name)
                renamed = True
            names.add(node.name)

        # Each one the graph feeds back is fed back from an output of the
        # body: one f already declares for its update, or one added for it.
        body_outputs = list(outputs)
        taken = {output.name for output in body_outputs}
        used = {id(output) for _, output in pairs}
        for node in found:
            output = next(
                (
                    output
                    for output in body_outputs
                    if node.update in (output, output._preds[0])
                    and id(output) not in used
                ),
                None,
            )
            if output is None:
                output = Output(
                    _free_name(f"{node.name}_next", taken), cast(Stream, node.update)
                )
                body_outputs.append(output)
            used.add(id(output))
            pairs.append((node, output))

        # What the graph starts each of them from: the init of the layer that
        # keeps it, or the stream a call gives it. It is read once, before the
        # first step.
        starts: dict[Input, Any] = {}
        for node in found:
            init = node.init
            if init is None:
                continue
            if isinstance(init, Input) and init.update is not None:
                raise ValueError(
                    f"{who}: the init of {node.name!r} is the fed-back Input "
                    f"{init.name!r}, which has no value before the first step."
                )
            if isinstance(init, Stream) and not isinstance(init, Input):
                read = sorted(
                    {
                        stream.name
                        for stream in _streams_from([init])
                        if isinstance(stream, Input)
                    }
                )
                if read:
                    raise ValueError(
                        f"{who}: the init of {node.name!r} is computed from the "
                        f"Inputs {read}. An init is read once, before the first "
                        "step: give an Input, whose first step it reads, a "
                        "Parameter, a Constant or a number."
                    )
            starts[node] = init

        # An Input the step reads, or starts a state from, is an input of f:
        # that Input itself, not one of its name.
        states = [state for state, _ in pairs]
        step_inputs: dict[int, Input] = {}
        for node in toposort(body_outputs):
            if isinstance(node, Input) and node.update is None and node not in states:
                step_inputs.setdefault(self.key(node), node)
        seeds = [init for init in starts.values() if isinstance(init, Input)]
        listed = {id(node) for node in f._inputs}
        unlisted = sorted(
            {
                node.name
                for node in [*step_inputs.values(), *seeds]
                if self.key(node) not in listed
            }
        )
        if unlisted:
            raise ValueError(
                f"{who} reads the Inputs {unlisted}, which nothing feeds back, "
                f"without declaring them: list them among {among} - "
                f"model.inputs[{unlisted[0]!r}] of the model that reads them - or "
                "feed them back with x.feedback(stream)."
            )

        fits = (
            not renamed
            and len(body_outputs) == len(outputs)
            and all(id(state) in declared or state in f._inputs for state in states)
        )
        if fits:
            # f is the body as it is: its own Inputs and outputs, read from
            # data at every step - the one-step model.
            if not f.built:
                f.build()
            self.body = f
            self.returns = list(f._outputs)
            self.pairs = [
                (declared.get(id(state), state), f._outputs[outputs.index(output)])
                for state, output in pairs
            ]
            self.starts = {
                declared[id(node)]: declared[id(node)].init
                for node in starts
                if id(node) in declared
            }
        else:
            # A seed two layers start from is one Input.
            inputs = dict(step_inputs)
            for seed in seeds:
                inputs.setdefault(self.key(seed), seed)
            self.body = Modely(
                f.name, inputs=[*inputs.values(), *states], outputs=body_outputs
            )
            self.body.build()
            # The layers were built on the copies: the streams f was declared
            # with hand out the same ones, Parameters their values.
            for source, copy in scope.merged_memo().items():
                if isinstance(source, Layer) and isinstance(copy, Layer):
                    source._layer = copy._layer
                    source._layer_signature = copy._layer_signature
            # The outputs added for the updates are the loop's own: it returns
            # the ones f declares.
            self.returns = outputs
            self.pairs = pairs
            self.starts = {node: self.original(init) for node, init in starts.items()}

    @classmethod
    def restored(cls, body: Modely, feedback: list, returns: list[int]):
        """The step of a Loop loaded back: ``body`` is the one it was saved
        with, ``feedback`` pairs every state with the output it is fed back
        from - their positions among the inputs and outputs of ``body`` - and
        the states start from what the Loop was saved reading; ``returns``
        are the positions of the outputs it returns."""
        step = cls.__new__(cls)
        step.f = step.body = body
        step.returns = [body._outputs[index] for index in returns]
        step.pairs = [(body._inputs[i], body._outputs[o]) for i, o in feedback]
        step.starts = {}
        step._originals = {}
        step._sources = {}
        step._copies = {}
        return step

    @property
    def states(self) -> list[Input]:
        return [state for state, _ in self.pairs]

    def original(self, node: Any) -> Any:
        """The stream of the graph the Loop is called in that ``node``, a copy
        made inlining ``f``, stands for: ``node`` itself if it is no copy."""
        return self._originals.get(id(node), node)

    def copied_from(self, node: Any) -> Any:
        """The node of the graph of ``f`` - or of a model it calls - that
        ``node``, a node of the body, stands for: ``node`` itself if it is
        no copy."""
        return self._copies.get(id(node), node)

    def source(self, node: Any) -> Any:
        """The Input ``node`` is a copy of, made inlining ``f`` or a model it
        calls - a fed-back Input inside a called model is copied once per
        call: ``node`` itself if it is no copy."""
        return self._sources.get(id(node), self.original(node))

    def key(self, node: Any) -> int:
        """What tells two inputs apart: the Input ``node`` stands for - never
        its name, two Inputs of one name are two."""
        return id(self.source(node))

    def state(self, key) -> Input:
        """The state the ``init`` key ``key`` gives a value for: a state
        itself, or an Input of ``f`` - or its name - that is fed back."""
        if any(key is state for state in self.states):
            return key
        declared = resolve(
            key,
            [(node, index, "") for index, node in enumerate(self.f._inputs)],
            kind="input",
            owner=f"Loop init of {self.f.name!r}",
            hint="Key it by the Input itself.",
            objects=Stream,
            missing=lambda key: (
                f"Loop init key {getattr(key, 'name', key)!r} is "
                f"not an input of {self.f.name!r}: a state kept inside a block, as "
                "the one of Integrate or Derivative, starts from the init of the "
                "block, Integrate(init=...)."
            ),
        )
        name = declared.name
        state = next(
            (state for state in self.states if self.key(state) == self.key(declared)),
            None,
        )
        if state is None:
            raise ValueError(
                f"Loop init key {name!r} is not fed back: pass it among the "
                f"inputs of the Loop, Loop(...)({{{name!r}: stream}})."
            )
        return state


class Loop(Layer):
    r"""The trajectory of a one-step model rolled out over a sequence, some of
    its outputs fed back into its inputs.

    Syntax
    ------
    ``Loop(f, feedback={x: x_next}, init={x: x0})(inputs)``
        Rolls ``f`` out, its output ``x_next`` fed back into its input
        ``x``, which starts from ``x0``.
    ``Loop(f, init={x: x0})(inputs)``
        The same, for a body ``f`` that feeds its states back itself.
    ``Loop(f, ...)({u: u_seq})``
        Binds the body input ``u``, or the Input of that name, to the stream
        ``u_seq``.
    ``Loop(f, ...)([u_seq, w_seq])``
        One stream per input of the loop, in the order ``f`` declares them.
    ``Loop(f, ...)(u_seq)``
        The stream of a loop with one input.
    ``Loop(f, ..., seq=n)()``
        No stream: ``n`` steps, every body input held at its own value.

    The body ``f`` is a :class:`Modely` that computes one step. Its states
    are the Inputs fed back from one step to the next. ``feedback`` gives
    some of them, and the others are found in the graph of ``f``:

    * every Input fed back with :meth:`Input.feedback`, inside the blocks
      ``f`` calls too;
    * the states of :class:`Integrate`, :class:`Derivative` and
      :class:`Buffer`.

    The rollout runs along the outermost sequence axis of the streams that
    drive it. Such a stream has one more sequence axis than the body input it
    feeds, and each step reads one element of it. The loop returns the
    outputs of ``f`` at every step, along a new outermost sequence axis. The
    loop is configured with its body and its states, then called on the
    streams that drive it.

    Parameters
    ----------
    f : Modely
        The body, a model that computes one step. It is built if it is not
        yet. Its own objectives are not used.
    feedback : dict, optional
        ``{body input: body output}``, each given by the object or by its
        name. The output feeds the input at the next step. It adds to the
        states found in the graph of ``f``. It is needed for a body that
        feeds nothing back itself, such as a model already built or loaded,
        or to reuse one body with other feedbacks. The output has the
        features and the sequence of the input. It is one sample, ``T=1``,
        when the input has a window, and has no window when the input has
        none.
    init : dict, optional
        ``{state: initial value}``, the state given by the Input or by its
        name. A state left out starts from the ``init`` of the layer that
        keeps it, or from zero. The initial value can be:

        * a stream with one more sequence axis than the state: step 0
          reads its first step, and the others are not read;
        * a stream of the shape of the state;
        * a number.
    name : str, optional
        The name of the loop. With several outputs, each is named
        ``<name>_<output>``. The default is ``Loop`` followed by a number.
    seq : int, optional
        The number of steps. Without it, the number of steps is the length
        of the streams that drive the rollout. Give it when no stream drives
        it, or when they declare different lengths.
    collect : bool, default True
        ``True`` returns the whole trajectory, one value per step. ``False``
        returns the last step alone.

    Call arguments
    --------------
    inputs : dict, list, Stream or None, optional
        The streams of the body inputs that are not states, one of:

        * a dict ``{body input: stream}``, keyed by the Input or its name;
        * a list, one stream per input of the loop, in the order ``f``
          declares them;
        * a stream alone, for a loop with one input;
        * nothing, ``Loop(...)()``.

        A stream with one more sequence axis than its body input gives one
        value per step. A stream with the shape of the body input is held at
        every step. A body input left out is held at its own value, and is an
        Input of the model that holds the loop. An Input of ``f`` that a
        layer's ``init`` reads, as in ``Integrate(init=v0)``, is an input of
        the loop too: its stream is the initial value. A state is not an
        input: its initial value goes in ``init``.

    Returns
    -------
    Stream or several streams
        With one output in ``f``, the loop is that output, a stream. Collected,
        a body output ``Shape(S, T, D)`` becomes ``Shape((H, *S), T, D)``,
        with ``H`` steps. With ``collect=False`` it keeps ``Shape(S, T, D)``.
        For example, a body output ``Shape(S=(), T=1, D=(2,))`` rolled out
        over 20 steps gives ``Shape(S=(20,), T=1, D=(2,))``. A rollout driven
        by a dynamic sequence, ``seq=-1``, gives ``S=(dyn,)``.

        With several outputs in ``f``, the loop is all of them, in the order
        of ``f``. Unpack them, ``position, velocity = Loop(...)(inputs)``, or
        pick one, by position, ``loop[1]``, or by its name in ``f``,
        ``loop["velocity"]``. An output that the loop adds for a state of a
        layer, such as the integral of :class:`Integrate`, is not returned.

    Raises
    ------
    ValueError
        If ``f`` has no state ("has no fed-back Input"), if an ``init`` key
        is not a state ("is not fed back"), if a state is among the inputs
        of the call ("is fed back: give its initial value with init="), if
        the number of steps is unknown ("cannot determine the rollout
        length"), if a list of the call has another length, if a feedback
        output does not match its input, or if a loop with several outputs
        is read as one stream ("is read as one stream").
    TypeError
        If ``f`` is not a Modely, if ``feedback`` or ``init`` is not a dict,
        or if the call is given something other than a dict, a list or a
        stream.

    See Also
    --------
    ModelyLoop : A model declared one step at a time, a Loop over its step.
    Input.feedback : Feeds an Input back with its next value.
    Ode : One step of an ODE, a body to roll out.
    Integrate : An integral that keeps its state in the body.
    SeqSelect : One step of a trajectory.

    Notes
    -----
    At step :math:`k`, the body reads the state :math:`x_k` and the inputs
    :math:`u_k`. The output fed back is the next state, which the step
    :math:`k + 1` reads:

    .. math:: x_{k+1} = f(x_k, u_k)

    Step 0 reads the first step of the initial value, :math:`x_0`. So the
    trajectory of an output fed back is :math:`x_1, \dots, x_H`, one step
    ahead of the states the steps read. For the body ``x_next = x + u`` of
    the Basic Examples, from 10:

    ====  ===========  ===========  =================
    step  state ``x``  input ``u``  output ``x_next``
    ====  ===========  ===========  =================
    0     10           1            11
    1     11           2            13
    2     13           3            16
    3     16           4            20
    ====  ===========  ===========  =================

    This is why the target of such an output is the data shifted one step:
    with ``X`` the recorded states, the initial values are ``X[:-1]`` and the
    targets ``X[1:]``. An output that is not fed back is computed from
    :math:`x_k`, and aligns with the states the steps read.

    A state with a window, ``tw=n``, is fed back by shifting: the oldest
    sample is dropped and the new one appended. After ``n`` steps the window
    holds only values the loop computed. Its window reads no future
    samples.

    A Loop can be the body of another Loop. Each rollout adds its own
    outermost sequence axis.

    Basic Examples
    --------------
    The state ``x`` accumulates the input ``u``, from 10:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Integrate, Loop, Modely, ModelyLoop, Output, TimeSelect
    >>> x, u = Input("x", tw=1), Input("u", tw=1)
    >>> step = Modely("step", inputs=[x, u], outputs=[Output("x_next", x + u)])
    >>> x0 = Input("x0", seq=4, tw=1)
    >>> u_seq = Input("u_seq", seq=4, tw=1)
    >>> trajectory = Loop(f=step, feedback={x: "x_next"}, init={x: x0})({u: u_seq})
    >>> trajectory.shape
    Shape(S=(4,), T=1, D=(1,))
    >>> sim = Modely("sim", inputs=[x0, u_seq], outputs=[Output("x", trajectory)]).build()
    >>> start = np.array([10.0, 0.0, 0.0, 0.0]).reshape(1, 4, 1, 1)
    >>> pushes = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    >>> keras.ops.convert_to_numpy(sim({"x0": start, "u_seq": pushes})["x"]).ravel()
    array([11., 13., 16., 20.], dtype=float32)

    Step 0 reads the first step of ``x0``, 10, and no other.

    Scope
    -----
    The call takes a list, one stream per input in the order of ``f``, or
    the stream alone, or a dict keyed by names:

    >>> rollout = Loop(f=step, feedback={x: "x_next"}, init={x: x0})
    >>> rollout([u_seq]).shape, rollout(u_seq).shape, rollout({"u": u_seq}).shape
    (Shape(S=(4,), T=1, D=(1,)), Shape(S=(4,), T=1, D=(1,)), Shape(S=(4,), T=1, D=(1,)))

    Called with nothing, ``u`` is held at its own value. It is an Input of
    the model that holds the loop:

    >>> held = rollout()
    >>> sim = Modely("held", inputs=[x0, u], outputs=[Output("x", held)]).build()
    >>> result = sim({"x0": start, "u": np.array([[[1.0]]])})
    >>> keras.ops.convert_to_numpy(result["x"]).ravel()
    array([11., 12., 13., 14.], dtype=float32)

    A dynamic sequence, ``seq=-1``, rolls out as many steps as the data has:

    >>> u_any = Input("u_any", seq=-1, tw=1)
    >>> Loop(f=step, feedback={x: "x_next"}, init={x: 10.0})(u_any).shape
    Shape(S=(dyn,), T=1, D=(1,))

    A state with a window is shifted. Here the body adds 1 to the newest
    sample of a window of 3, which starts from ``[1, 2, 3]``:

    >>> w = Input("w", tw=3)
    >>> newest = TimeSelect(-1)(w)
    >>> shift = Modely("shift", inputs=[w], outputs=[Output("w_next", newest + 1.0)])
    >>> w0 = Input("w0", seq=4, tw=3)
    >>> shifted = Loop(f=shift, feedback={w: "w_next"}, init={w: w0})()
    >>> sim = Modely("windows", inputs=[w0], outputs=[Output("w", shifted)]).build()
    >>> windows = np.tile(np.array([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1), (1, 4, 1, 1))
    >>> keras.ops.convert_to_numpy(sim({"w0": windows})["w"]).ravel()
    array([4., 5., 6., 7.], dtype=float32)

    A body with several outputs gives one stream each:

    >>> p, v, force = Input("p", tw=1), Input("v", tw=1), Input("force", tw=1)
    >>> v_next = v + 0.5 * force
    >>> p_next = p + 0.5 * v_next
    >>> motion = Modely(
    ...     "motion", inputs=[p, v, force],
    ...     outputs=[Output("position", p_next), Output("velocity", v_next)],
    ... )
    >>> forces = Input("forces", seq=3, tw=1)
    >>> moved = Loop(f=motion, feedback={p: "position", v: "velocity"}, name="moved")(forces)
    >>> position, velocity = moved
    >>> position.name, velocity.name
    ('moved_position', 'moved_velocity')
    >>> moved[1].name, moved["velocity"].name
    ('moved_velocity', 'moved_velocity')
    >>> sim = Modely(
    ...     "push", inputs=[forces],
    ...     outputs=[Output("position", position), Output("velocity", velocity)],
    ... ).build()
    >>> result = sim({"forces": np.full((1, 3, 1, 1), 2.0)})
    >>> keras.ops.convert_to_numpy(result["velocity"]).ravel()
    array([1., 2., 3.], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["position"]).ravel()
    array([0.5, 1.5, 3. ], dtype=float32)

    Options
    -------
    ``feedback`` takes names too, and ``init`` takes a number. ``seq`` gives
    the number of steps when no stream drives the rollout:

    >>> counted = Loop(f=step, feedback={"x": "x_next"}, init={"x": 10.0}, seq=3)()
    >>> sim = Modely("counted", inputs=[u], outputs=[Output("x", counted)]).build()
    >>> keras.ops.convert_to_numpy(sim({"u": np.array([[[1.0]]])})["x"]).ravel()
    array([11., 12., 13.], dtype=float32)

    ``collect=False`` returns the last step alone, which is the last step of
    the trajectory:

    >>> last = Loop(f=step, feedback={x: "x_next"}, init={x: x0}, collect=False)(u_seq)
    >>> last.shape
    Shape(S=(), T=1, D=(1,))
    >>> sim = Modely(
    ...     "last", inputs=[x0, u_seq], outputs=[Output("x", trajectory), Output("last", last)]
    ... ).build()
    >>> result = sim({"x0": start, "u_seq": pushes})
    >>> keras.ops.convert_to_numpy(result["last"]), keras.ops.convert_to_numpy(result["x"])[0, -1]
    (array([[[20.]]], dtype=float32), array([[20.]], dtype=float32))

    ``name`` names the loop:

    >>> Loop(f=step, feedback={x: "x_next"}, init={x: x0}, name="sum")(u_seq).name
    'sum'

    Applications
    ------------
    A mass pushed by a force, integrated in its body by :class:`Integrate`.
    The Loop finds the state of the layer, and binds the Input ``v0`` that
    starts it to a stream of the outer model:

    >>> f_k, v0 = Input("f_k", tw=1), Input("v0", tw=1)
    >>> speed = Integrate(dt=0.5, init=v0, name="speed")(f_k / 2.0)
    >>> mass = Modely("mass", inputs=[f_k, v0], outputs=[Output("speed", speed)])
    >>> f_seq, v_start = Input("f_seq", seq=-1, tw=1), Input("v_start", tw=1)
    >>> Loop(f=mass)({f_k: f_seq, v0: v_start}).shape
    Shape(S=(dyn,), T=1, D=(1,))

    Properties & Relations
    ----------------------
    A :class:`ModelyLoop` is a Loop over its step. Declared with
    :meth:`Input.feedback`, the same step gives the same trajectory:

    >>> xs, us = Input("xs", tw=1), Input("us", tw=1)
    >>> xs_next = xs.feedback(xs + us)
    >>> declared = ModelyLoop("declared", inputs=[us, xs], outputs=[Output("x", xs_next)])
    >>> result = declared.build()({"us": pushes, "xs": start})
    >>> keras.ops.convert_to_numpy(result["x"]).ravel()
    array([11., 13., 16., 20.], dtype=float32)

    Possible Issues
    ---------------
    Without a stream that drives the rollout, the number of steps is
    unknown. Give ``seq``:

    >>> Loop(f=step, feedback={x: "x_next"})()
    Traceback (most recent call last):
        ...
    ValueError: Loop cannot determine the rollout length: no initial value or input carries a rollout, so pass seq= to Loop.

    A state starts from ``init``, and is not an input of the call:

    >>> Loop(f=step, feedback={x: "x_next"})({x: x0})
    Traceback (most recent call last):
        ...
    ValueError: Loop input 'x' is fed back: give its initial value with init=, Loop(..., init={'x': stream}).

    A loop with several outputs is not one stream. Unpack it, or pick one:

    >>> try:
    ...     Output("both", moved)
    ... except ValueError as error:
    ...     print(error)
    Loop 'moved' returns the 2 outputs ['position', 'velocity'] of 'motion', and is read as one stream...
    >>> Output("velocity", moved["velocity"]).name
    'velocity'

    .. versionadded:: 2.0
    """

    # Set by __call__ on the loop node it returns.
    _loop_outputs: list[LoopOutput]

    def __init__(
        self,
        f: Modely,
        feedback: dict | None = None,
        init: dict | None = None,
        name=None,
        seq: int | None = None,
        collect: bool = True,
    ):
        if not isinstance(f, Modely):
            raise TypeError(f"Loop rolls out a Modely, f=, got {type(f).__name__}.")
        if feedback is not None and not isinstance(feedback, dict):
            raise TypeError("Loop feedback must be a dict of body input: output.")
        if init is not None and not isinstance(init, dict):
            raise TypeError(
                "Loop init must be a dict of fed-back input: initial value."
            )
        step = _Step(
            f,
            feedback or {},
            who=f"Loop over {f.name!r}",
            among=f"the inputs of {f.name!r}",
        )
        if not step.pairs:
            raise ValueError(
                f"Loop: {f.name!r} has no fed-back Input: declare one with "
                "x.feedback(stream), or give feedback=."
            )
        self._declare(step, init or {}, name, seq, collect)

    @classmethod
    def _of(cls, step: _Step, init: dict, name=None, seq=None, collect=True):
        """The Loop over a step already found."""
        loop = cls.__new__(cls)
        loop._declare(step, init, name, seq, collect)
        return loop

    def _declare(self, step: _Step, init: dict, name, seq, collect):
        self._step = step
        self._model = model = step.body
        self._feedback_inputs = step.states
        self._feedback_outputs = [output for _, output in step.pairs]
        self._init = self._resolve_init(init, step)
        # A body input its outputs never read, such as a target, is no input
        # of the loop.
        read = {id(model._original(node)) for node in self._model.inference_inputs}
        self._static_inputs = [
            node
            for node in model._inputs
            if node not in self._feedback_inputs and id(node) in read
        ]
        # An Input the body starts a state from, and does not read: an input
        # of the loop all the same, whose outer stream is that initial value.
        # One per Input, which several states may start from.
        static = {step.key(node) for node in self._static_inputs}
        self._seed_inputs = list(
            {
                step.key(start): start
                for start in step.starts.values()
                if isinstance(start, Input) and step.key(start) not in static
            }.values()
        )
        self._feedback_shift_axes = self._resolve_feedback(step.pairs)
        self._length = None if seq is None else int(seq)
        self._collect = bool(collect)
        # The outputs of f; with more than one, the loop is all of them, and
        # no stream.
        self._returns = step.returns
        self._return_all_outputs = len(self._returns) > 1
        super().__init__(name=name)
        # Unbound until called: it has no shape to be used as a stream with.
        self._shape = None

    @property
    def shape(self):
        self._check_bound()
        if self._return_all_outputs:
            names = [output.name for output in self._returns]
            raise _SeveralOutputs(
                f"Loop {self.name!r} returns the {len(names)} outputs {names} of "
                f"{self._model.name!r}, and is read as one stream: unpack them, "
                f"{', '.join(names)} = Loop(...)(inputs), or pick one, loop[i]."
            )
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    # The streams a bound loop reads are its predecessors, and nothing else:
    # a copy rewired to other streams - the loop inlined in the graph of a
    # call to the model that holds it - reads those, and so does what is
    # drawn or listed of it.
    @property
    def _initial_values(self) -> list[Stream]:
        """The initial value of every state, in their order: the first
        predecessors."""
        return list(self._preds[: len(self._feedback_inputs)])  # type: ignore

    @property
    def _static_sources(self) -> list[Stream]:
        """The outer stream of every body input that is not fed back, in the
        order of the static inputs: the predecessors after the initial
        values."""
        return list(self._preds[len(self._feedback_inputs) :])  # type: ignore

    def __call__(self, inputs: dict | list | tuple | Stream | None = None):
        """Bind the loop to its outer inputs, seeded with its ``init``.

        ``inputs`` is a dict keyed by body input or input name, a list with
        one stream per input of the loop, in the order ``f`` declares them,
        or the stream of a loop with one input. Every call returns a new loop
        node; the body, and so its weights, is the one the loop was declared
        with.
        """
        node = copy.copy(self)
        node._source = node
        node._layer = node._layer_signature = None
        static_sources, bound = node._resolve_sources({} if inputs is None else inputs)
        node._preds = [*node._resolve_initial_values(bound), *static_sources]
        node._configure()

        result_shapes = [node._result_shape(output) for output in self._model._outputs]
        node.shape = result_shapes[self._model._outputs.index(self._returns[0])]
        node._loop_outputs = [
            LoopOutput(node, index, output, result_shapes[index])
            for index, output in enumerate(self._model._outputs)
            if output in self._returns
        ]
        return node

    # ------------------------------------------------------------------
    # Axis configuration
    # ------------------------------------------------------------------
    def _configure(self):
        self._rolled_initial = [
            self._rolled(node, value)
            for node, value in zip(self._feedback_inputs, self._initial_values)
        ]
        self._rolled_static = [
            self._rolled(node, source)
            for node, source in zip(self._static_inputs, self._static_sources)
        ]
        for node, source, rolled in zip(
            self._static_inputs, self._static_sources, self._rolled_static
        ):
            step = source.shape.without_outer_seq() if rolled else source.shape
            if source is not node and (
                step.tw != node.tw or not _same_steps(step.seq, node.seq)
            ):
                raise ValueError(
                    f"Loop input {node.name!r} expects the sequence and window "
                    f"{node.shape}, or one more sequence axis for the rollout, got "
                    f"{source.shape} from {source.name!r}."
                )

        sources = [*self._initial_values, *self._static_sources]
        rolled = [*self._rolled_initial, *self._rolled_static]
        loop_sources = [
            (index, source)
            for index, (source, is_rolled) in enumerate(zip(sources, rolled))
            if is_rolled
        ]
        # The inputs that drive the rollout set its number of steps. An initial
        # value is read at its first step only, so it sets them only when no
        # input carries a rollout.
        driving = [
            item for item in loop_sources if item[0] >= len(self._initial_values)
        ] or loop_sources
        self._horizon = self._resolve_length(driving)

        # The rollout needs a static number of steps, so a dynamic rollout is
        # only read from the tensor that carries it, at call time.
        self._horizon_index = next(
            (index for index, source in driving if source.shape.seq[0] is None),
            None,
        )
        self._feedback_state_shapes = [
            self._state_shape(node, value, is_rolled)
            for node, value, is_rolled in zip(
                self._feedback_inputs, self._initial_values, self._rolled_initial
            )
        ]

    def _resolve_length(self, loop_sources):
        widths = {
            int(source.shape.seq[0])
            for _, source in loop_sources
            if source.shape.seq[0] is not None
        }
        if self._length is not None:
            return self._length
        if len(widths) == 1:
            return next(iter(widths))
        if widths:
            raise ValueError(
                f"Loop rollout inputs declare different lengths {sorted(widths)}; "
                "pass seq= to pin the number of steps."
            )
        if loop_sources:
            # Every rollout input is dynamic, so the data decides the number of
            # steps; the graph is only declared with one.
            return DYNAMIC_BUILD_LENGTH
        raise ValueError(
            "Loop cannot determine the rollout length: no initial value or input "
            "carries a rollout, so pass seq= to Loop."
        )

    def _result_shape(self, output):
        """The shape of an output once rolled out: collected, the rollout is
        its new outermost sequence axis."""
        if self._collect:
            # A rollout as long as its data is declared dynamic, like the
            # inputs it follows; the length it is built with is a placeholder.
            dynamic = self._length is None and self._horizon_index is not None
            return output.shape.with_seq(None if dynamic else self._horizon)
        return output.shape

    # ------------------------------------------------------------------
    # Declaration helpers
    # ------------------------------------------------------------------
    def _resolve_init(self, init: dict, step: _Step) -> dict[Input, Any]:
        """The initial value ``init`` gives each state: only a state takes
        one, and one that starts from the init of its layer takes no other."""
        resolved: dict[Input, Any] = {}
        for key, value in init.items():
            state = step.state(key)
            if state in step.starts:
                raise ValueError(
                    f"Loop init gives {state.name!r} an initial value, and it "
                    "has one already: the stream a call gives it. A state "
                    "starts from one value: give one of them."
                )
            resolved[state] = value
        return resolved

    @staticmethod
    def _rolled(node, source):
        """Whether ``source`` feeds ``node`` one step of its outermost
        sequence axis per step of the rollout: it has one more sequence axis
        than the body reads, the others the same - a dynamic one matching any
        length."""
        return source.shape.seq_rank == node.shape.seq_rank + 1 and _same_steps(
            source.seq[1:], node.seq
        )

    @staticmethod
    def _state_shape(node, value, rolled):
        """Shape an initial value must be broadcast to, or None if it fits.

        A dynamic axis of the input is 1 where the value carries it, kept as
        it comes, and None where it does not - a zero, a number, a value of
        fewer sequence axes: the value fed back sizes it at run time. A
        dynamic axis of the value matches any length of the input's.
        """
        if value.shape.tw is None and node.tw is not None and not value._is_value:
            raise ValueError(
                f"Loop initial value {value.name!r} has no window, {value.shape}, "
                f"and the fed-back input {node.name!r} has one, {node.shape}: "
                "declare the Input with tw=1."
            )
        source = value.shape.without_outer_seq().tuple if rolled else value.shape.tuple
        target = tuple(node.shape.tuple)
        if source == target:
            return None
        incompatible = len(source) > len(target) or any(
            None not in (size, target_size) and size not in (1, target_size)
            for size, target_size in zip(reversed(source), reversed(target))
        )
        if incompatible:
            raise ValueError(
                f"Loop initial value {value.name!r} has shape {source}, which cannot "
                f"feed the fed-back input {node.name!r} of shape {target}."
            )
        padded = (1,) * (len(target) - len(source)) + tuple(source)
        return tuple(
            target_size if target_size is not None else None if size == 1 else 1
            for size, target_size in zip(padded, target)
        )

    @staticmethod
    def _resolve_feedback(pairs):
        """Time axis each feedback output is shifted into, or None to replace."""
        shift_axes = []
        for node, output in pairs:
            if tuple(output.dim) != tuple(node.dim):
                raise ValueError(
                    f"Loop feedback {node.name!r} -> {output.name!r}: dim "
                    f"{tuple(output.dim)} does not match the input dim "
                    f"{tuple(node.dim)}."
                )
            # The output becomes the input at the next step, so it must carry
            # the same sequence; a dynamic axis is only known at run time.
            output_seq, node_seq = tuple(output.seq), tuple(node.seq)
            if len(output_seq) != len(node_seq) or any(
                None not in (size, node_size) and size != node_size
                for size, node_size in zip(output_seq, node_seq)
            ):
                raise ValueError(
                    f"Loop feedback {node.name!r} -> {output.name!r}: the output "
                    f"is {output.shape}, the input {node.shape}: the sequence must "
                    "be the same."
                )
            # Without a window, the output replaces the input as it is.
            if node.tw is None or output.tw is None:
                if (node.tw, output.tw) != (None, None):
                    raise ValueError(
                        f"Loop feedback {node.name!r} -> {output.name!r}: the "
                        f"output is {output.shape}, the input {node.shape}: both "
                        "have a window, or neither."
                    )
                shift_axes.append(None)
                continue
            # The output is the next sample of the input: a window of outputs
            # would be shifted in as one step.
            if output.tw != 1:
                raise ValueError(
                    f"Loop feedback {node.name!r} -> {output.name!r}: the output "
                    f"must be a single sample, the next one of the input, got a "
                    f"window of {output.tw}."
                )
            # The window is shifted to end at the sample fed back, so it cannot
            # read past the current one.
            if node.future:
                raise ValueError(
                    f"Loop feedback {node.name!r} -> {output.name!r}: the input "
                    f"window reads {node.future} future samples, but a fed-back "
                    "input can only read up to its current sample."
                )
            if node.tw == 1:
                shift_axes.append(None)
            else:
                shift_axes.append(node.shape.tw_axis)
        return shift_axes

    def _call_order(self) -> list[Input]:
        """The inputs of the loop, those a call binds, in the order ``f``
        declares them: the ones the body reads that are not fed back, and the
        ones it starts a state from."""
        bindable = {
            self._step.key(n): n for n in (*self._static_inputs, *self._seed_inputs)
        }
        return [
            bindable[id(node)] for node in self._step.f._inputs if id(node) in bindable
        ]

    def _resolve_sources(self, inputs):
        """Outer stream feeding each body input that is not fed back, and
        every input of the loop by the id of its node - those a state starts
        from too. A key is a body input, its Input of ``f``, or its name."""
        if isinstance(inputs, (Stream, list, tuple)):
            # One stream per input of the loop, in the order f declares them.
            streams = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
            order = self._call_order()
            if len(streams) != len(order):
                given = (
                    f"{len(streams)} streams"
                    if isinstance(inputs, (list, tuple))
                    else "one stream"
                )
                raise ValueError(
                    f"Loop {self.name!r} has {len(order)} inputs "
                    f"{[node.name for node in order]}, the inputs of "
                    f"{self._model.name!r} it does not feed back: called with a list "
                    "it takes one stream for each, in that order, and with a "
                    f"stream alone it has one; got {given}."
                )
            inputs = dict(zip(order, streams))
        if not isinstance(inputs, dict):
            raise TypeError(
                "Loop is called with a dict of body input: outer stream, a list "
                "of streams, one per input in the order the body declares "
                f"them, or the stream of its one input; got {type(inputs).__name__}."
            )

        known = [*self._static_inputs, *self._seed_inputs]
        by_node: dict[int, Stream] = {}
        for key, value in inputs.items():
            key_name = key.name if isinstance(key, Stream) else key
            if not isinstance(value, Stream):
                raise TypeError(
                    f"Loop input {key_name!r} must be bound to a Stream, got {value!r}."
                )
            by_node[id(self._loop_input(key, known))] = value

        bound = {}
        for node in known:
            # A body input left unbound is handed on to the body as it is.
            original = self._step.original(node)
            source: Stream = by_node[id(node)] if id(node) in by_node else original
            if source is not original and tuple(source.dim) != tuple(node.dim):
                raise ValueError(
                    f"Loop input {node.name!r} expects dim {tuple(node.dim)}, got dim "
                    f"{tuple(source.dim)} from {source.name!r}."
                )
            bound[id(node)] = source
        return [bound[id(node)] for node in self._static_inputs], bound

    def _loop_input(self, key, known: list[Input]) -> Input:
        """The input of the loop a call's key binds: a body input itself, the
        Input of ``f`` it stands for, or a name - one input alone carrying
        it. A state is refused: it starts from its init."""
        states = list(self._feedback_inputs)
        key = resolve(
            key,
            [(node, None, "") for node in [*known, *states]],
            kind="input",
            owner=f"Loop {self.name!r}",
            hint="Pass the Input itself.",
            objects=Stream,
            same=lambda node, key: self._step.key(node) == self._step.key(key),
            missing=lambda key: (
                f"Loop input {getattr(key, 'name', key)!r} is not "
                "an input the body model reads."
            ),
        )
        if any(key is state for state in states):
            raise ValueError(
                f"Loop input {key.name!r} is fed back: give its initial value "
                f"with init=, Loop(..., init={{{key.name!r}: stream}})."
            )
        return key

    def _resolve_initial_values(self, bound):
        """Seed of every state: its init, the init of the layer that keeps it
        - an input of the loop is its outer stream - or zero."""
        from nnodely.layers.parameter import Constant

        values = []
        for node in self._feedback_inputs:
            if node in self._init:
                value = self._init[node]
            else:
                start = self._step.starts.get(node)
                if isinstance(start, Input):
                    # The input of the loop that Input is.
                    seed = next(
                        seed
                        for seed in (*self._static_inputs, *self._seed_inputs)
                        if self._step.key(seed) == self._step.key(start)
                    )
                    value = bound[id(seed)]
                else:
                    value = 0.0 if start is None else start
            values.append(
                value if isinstance(value, Stream) else Constant(name=None, value=value)
            )
        return values

    def _state_starts(self) -> list[tuple[Input, Stream, Any, bool]]:
        """Every state of the bound loop, with the stream it is fed back with,
        what it starts from - a stream, a number, or None for zero - and
        whether that is read at its first step."""
        starts = []
        for state, output, value, rolled in zip(
            self._feedback_inputs,
            self._feedback_outputs,
            self._initial_values,
            self._rolled_initial,
        ):
            given = self._init.get(state, self._step.starts.get(state))
            starts.append(
                (
                    state,
                    output._preds[0] if state.update is None else state.update,
                    value if isinstance(given, Stream) else given,
                    rolled,
                )
            )
        return starts

    @staticmethod
    def _find_node(nodes, value, kind):
        # A name several of them carry is refused, listing them: the body is
        # built only if its inputs and outputs have names of their own.
        return resolve(
            value,
            [(node, None, f"(its {kind} {index})") for index, node in enumerate(nodes)],
            kind=kind,
            owner="Loop feedback",
            hint=f"Pass the {kind} itself.",
            objects=Stream,
            missing=lambda value: (
                f"feedback {kind} {value!r} was not found."
                if isinstance(value, str)
                else f"feedback {kind} {value.name!r} does not belong to the "
                "provided Modely."
            ),
        )

    def __repr__(self):
        """Called, a loop of one output is the stream it returns. Not called
        yet, or with several outputs - no one stream - it is the loop."""
        if self._shape is None:
            return object_repr(self, name=self.name, model=self._model, bound=False)
        if self._return_all_outputs:
            outputs = [output.name for output in self._returns]
            return object_repr(
                self, name=self.name, model=self._model, bound=True, outputs=outputs
            )
        return Stream.__repr__(self)

    def _check_bound(self):
        if self._shape is None:
            raise ValueError(
                f"Loop {self.name!r} is not bound: call it with its inputs "
                "dict, Loop(...)(inputs), before using it as a stream."
            )

    def _outputs(self) -> list:
        """The streams the loop returns: one per output of ``f``, the loop
        itself when it has one."""
        self._check_bound()
        return self._loop_outputs if self._return_all_outputs else [self]

    def __iter__(self):
        return iter(self._outputs())

    def __getitem__(self, key):
        """An output by position, as in a list, or by the name it has in
        ``f`` - one or the other, never both. A name several outputs carry
        raises, listing them with their position."""
        outputs = self._outputs()
        owner = f"Loop {self.name!r}"
        if not by_name_or_index(key, owner):
            return outputs[key]
        picked = resolve(
            key,
            [(output, index, "") for index, output in enumerate(self._returns)],
            kind="output",
            owner=owner,
            hint="Pick it by its position, loop[i], or give it a name of its "
            "own in the body, output.name = 'new_name'.",
        )
        # The loop returns one stream per output of f, in its order.
        return outputs[self._returns.index(picked)]

    def build_layer(self):
        model = self._model
        return LoopImpl(
            model=model.inference_model,
            # The keys the body was built with: renamed since, its Inputs and
            # outputs keep them in its Keras model.
            feedback_input_names=tuple(
                model._key(node) for node in self._feedback_inputs
            ),
            feedback_output_names=tuple(
                model._key(node) for node in self._feedback_outputs
            ),
            model_output_names=tuple(model._key(node) for node in model._outputs),
            static_input_names=tuple(model._key(node) for node in self._static_inputs),
            horizon=self._horizon,
            collect=self._collect,
            rolled_initial=tuple(self._rolled_initial),
            rolled_static=tuple(self._rolled_static),
            feedback_shift_axes=tuple(self._feedback_shift_axes),
            feedback_state_shapes=tuple(self._feedback_state_shapes),
            result_shapes=tuple(
                self._result_shape(output).tuple for output in model._outputs
            ),
            horizon_index=self._horizon_index,
            primary_output_index=model._outputs.index(self._returns[0]),
            return_all_outputs=self._return_all_outputs,
            name=self.name,
        )

    def get_config(self):
        config = super().get_config()
        model = self._model
        config.update(
            {
                # By position among the inputs and outputs of the body.
                "feedback": [
                    [model._inputs.index(node), model._outputs.index(output)]
                    for node, output in zip(
                        self._feedback_inputs, self._feedback_outputs
                    )
                ],
                "returns": [model._outputs.index(output) for output in self._returns],
                "seq": self._length,
                "collect": self._collect,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # `f` is the reloaded body (see ModelSerializer), and the feedback
        # every state, found or given when it was saved. The predecessors are
        # the initial value of every state, then the source of every other
        # body input it reads, in the body's order. A body input left unbound
        # was its own source; the reloaded body is a copy of its own, so it is
        # bound to the predecessor saved for it - the same Input, loaded once
        # for every file of the tree that reads it.
        config = dict(config)
        body = config.pop("f")
        feedback = config.pop("feedback")
        step = _Step.restored(body, feedback, config.pop("returns"))
        preds = list(preds or [])
        loop = cls._of(
            step, init=dict(zip(step.states, preds[: len(feedback)])), **config
        )
        return loop(dict(zip(loop._static_inputs, preds[len(feedback) :])))
