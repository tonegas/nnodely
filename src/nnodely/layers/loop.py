import contextlib
import copy
from typing import Any, cast

import keras
from keras.src.backend.common import stateless_scope

from nnodely.core.dag import (
    _copies,
    Scope,
    _streams_from,
    flatten_fed_back,
    flatten_node,
    toposort_outputs,
)
from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.core.names import resolve
from nnodely.core.shape import (
    BATCH_AXIS,
    OUTER_AXIS,
    _same_steps,
    slice_axis,
    transpose,
)
from nnodely.core.stream import Stream
from nnodely.layers.input import Input
from nnodely.layers.output import Output


#: Steps a dynamic rollout is declared with when no ``length`` is given. It
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
            (node.name, cast(Any, model_call)._model.name)
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

        # An Input is read from the data of its name: one the step reads, or
        # starts a state from, is an input of f declared under that name.
        states = [state for state, _ in pairs]
        step_inputs: dict[str, Input] = {}
        for node in toposort_outputs(body_outputs):
            if isinstance(node, Input) and node.update is None and node not in states:
                step_inputs.setdefault(node.name, node)
        seeds = [init for init in starts.values() if isinstance(init, Input)]
        listed = {node.name for node in f._inputs}
        unlisted = sorted(
            {
                node.name
                for node in [*step_inputs.values(), *seeds]
                if node.name not in listed
            }
        )
        if unlisted:
            raise ValueError(
                f"{who} reads the Inputs {unlisted}, which nothing feeds back "
                f"and are not among {among}: declare them there, or feed them "
                "back with x.feedback(stream)."
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
                inputs.setdefault(seed.name, seed)
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
            (state for state in self.states if self.original(state) is declared),
            None,
        )
        if state is None:
            raise ValueError(
                f"Loop init key {name!r} is not fed back: pass it among the "
                f"inputs of the Loop, Loop(...)({{{name!r}: stream}})."
            )
        return state


class Loop(Layer):
    """Roll out a Modely, feeding some of its outputs back onto its inputs.

    ::

        Loop(f=step, init={state: s0})({force: F_seq})
        Loop(f=body, feedback={state: next_state}, init={state: s0})({force: F_seq})

    The states - the Inputs fed back - are found in the graph of ``f``: every
    Input fed back with :meth:`Input.feedback`, inside the models it calls
    too, and the state of an :class:`~nnodely.Integrate`, a
    :class:`~nnodely.Derivative` or a :class:`~nnodely.Buffer`. ``feedback``
    (body input: body output) adds to them, and is the only way to give the
    feedback of a model that is already built or loaded, or to reuse one step
    with different feedbacks. An update that is not an output of ``f`` is
    added to the outputs of the body, which the loop does not return.

    The loop is declared with its body and ``init`` (state: initial value),
    and bound to the outer graph by calling it with ``inputs`` (body input:
    outer stream) - only the inputs that are not fed back. As for a Modely,
    they can be a list instead, one stream per input of the loop in the
    order ``f`` declares them, or the stream alone when there is one::

        Loop(f=step, init={state: s0})([F_seq, u_seq])  # f declares force, u
        Loop(f=step, init={state: s0})(F_seq)            # f declares force

    A state starts from its ``init``, or from the init of the layer that
    keeps it, or from zero without one; two are an error. A layer's init that
    is an Input of ``f``, as in ``Integrate(init=x0)``, is an input of the
    loop: its outer stream is the initial value. A body input missing from a
    dict keeps its own value at every step.

    The rollout lives outside the body: every stream that carries it has one
    more sequence axis than the body input it feeds, its outermost, which the
    rollout consumes one step at a time. A state takes that stream through
    ``init`` (step 0 seeds the state), an exogenous input through
    ``inputs``.

    ``length`` pins the number of steps: the length of the outermost
    sequence axis the rollout reads from the streams that carry it and,
    collected, adds to every output. It is required only when no initial
    value or input carries a rollout. A rollout declared dynamic
    (an Input with ``seq=-1``) follows the length of the sequence it is actually
    given, and without ``length`` its streams are declared with
    ``DYNAMIC_BUILD_LENGTH`` steps.

    ``collect`` returns the whole trajectory, the rollout the new outermost
    sequence axis of every output; ``collect=False`` returns only the final
    step.

    The loop returns every output of ``f``, in its order. With one, the loop
    is that output, a stream. With more, it is all of them: unpack them,
    ``x, v = Loop(f=step)(inputs)``, or pick one, ``loop[1]``; used where one
    stream is read - an Output, an operator, a layer - it raises.

    A state declared with a window (``tw=n``) is fed back by shifting: the
    oldest step is dropped and the new output appended, so after ``n`` steps
    the window is entirely self-predicted. A fed-back output is therefore a
    single sample, and the window of its input reads no future samples. A
    state without a window is replaced by an output without one.
    """

    # Set by __call__ on the loop node it returns.
    _loop_outputs: list[LoopOutput]

    def __init__(
        self,
        f: Modely,
        feedback: dict | None = None,
        init: dict | None = None,
        name=None,
        length: int | None = None,
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
        self._declare(step, init or {}, name, length, collect)

    @classmethod
    def _of(cls, step: _Step, init: dict, name=None, length=None, collect=True):
        """The Loop over a step already found."""
        loop = cls.__new__(cls)
        loop._declare(step, init, name, length, collect)
        return loop

    def _declare(self, step: _Step, init: dict, name, length, collect):
        self._step = step
        self._f = step.body
        self._feedback_inputs = step.states
        self._feedback_outputs = [output for _, output in step.pairs]
        self._init = self._resolve_init(init, step)
        # A body input its outputs never read, such as a target, is no input
        # of the loop.
        read = {id(self._f._original(node)) for node in self._f.inference_inputs}
        self._static_inputs = [
            node
            for node in self._f._inputs
            if node not in self._feedback_inputs and id(node) in read
        ]
        # An Input the body starts a state from, and does not read: an input
        # of the loop all the same, whose outer stream is that initial value.
        # One per name: Inputs of one name read the same data.
        static = {node.name for node in self._static_inputs}
        self._seed_inputs = list(
            {
                start.name: start
                for start in step.starts.values()
                if isinstance(start, Input) and start.name not in static
            }.values()
        )
        self._feedback_shift_axes = self._resolve_feedback(step.pairs)
        self._length = None if length is None else int(length)
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
                f"{self._f.name!r}, and is read as one stream: unpack them, "
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

        result_shapes = [node._result_shape(output) for output in self._f._outputs]
        node.shape = result_shapes[self._f._outputs.index(self._returns[0])]
        node._loop_outputs = [
            LoopOutput(node, index, output, result_shapes[index])
            for index, output in enumerate(self._f._outputs)
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
                "pass length= to pin the number of steps."
            )
        if loop_sources:
            # Every rollout input is dynamic, so the data decides the number of
            # steps; the graph is only declared with one.
            return DYNAMIC_BUILD_LENGTH
        raise ValueError(
            "Loop cannot determine the rollout length: no initial value or input "
            "carries a rollout, so pass length= to Loop."
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
            node.name: node for node in (*self._static_inputs, *self._seed_inputs)
        }
        return [
            bindable[node.name]
            for node in self._step.f._inputs
            if node.name in bindable
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
                    f"{self._f.name!r} it does not feed back: called with a list "
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
            same=lambda node, key: node is key or self._step.original(node) is key,
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
                    # The input of the loop reading the data of its name.
                    seed = next(
                        seed
                        for seed in (*self._static_inputs, *self._seed_inputs)
                        if seed is start or seed.name == start.name
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
        if self._shape is None or not self._return_all_outputs:
            return super().__repr__()
        # Several outputs: no one shape to print.
        return (
            f"Loop(name={self.name!r}, "
            f"outputs={[output.name for output in self._returns]}, "
            f"predecessors={[pred.name for pred in self._preds]})"
        )

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

    def __getitem__(self, index: int):
        return self._outputs()[index]

    def build_layer(self):
        return LoopImpl(
            model=self._f.inference_model,
            # The keys the body was built with: renamed since, its Inputs and
            # outputs keep them in its Keras model.
            feedback_input_names=tuple(
                self._f._key(node) for node in self._feedback_inputs
            ),
            feedback_output_names=tuple(
                self._f._key(node) for node in self._feedback_outputs
            ),
            model_output_names=tuple(self._f._key(node) for node in self._f._outputs),
            static_input_names=tuple(
                self._f._key(node) for node in self._static_inputs
            ),
            horizon=self._horizon,
            collect=self._collect,
            rolled_initial=tuple(self._rolled_initial),
            rolled_static=tuple(self._rolled_static),
            feedback_shift_axes=tuple(self._feedback_shift_axes),
            feedback_state_shapes=tuple(self._feedback_state_shapes),
            result_shapes=tuple(
                self._result_shape(output).tuple for output in self._f._outputs
            ),
            horizon_index=self._horizon_index,
            primary_output_index=self._f._outputs.index(self._returns[0]),
            return_all_outputs=self._return_all_outputs,
            name=self.name,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # By position among the inputs and outputs of the body.
                "feedback": [
                    [self._f._inputs.index(node), self._f._outputs.index(output)]
                    for node, output in zip(
                        self._feedback_inputs, self._feedback_outputs
                    )
                ],
                "returns": [self._f._outputs.index(output) for output in self._returns],
                "length": self._length,
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
        # bound to the node of the same name in the graph the loop is loaded in.
        config = dict(config)
        body = config.pop("f")
        feedback = config.pop("feedback")
        step = _Step.restored(body, feedback, config.pop("returns"))
        preds = list(preds or [])
        loop = cls._of(
            step, init=dict(zip(step.states, preds[: len(feedback)])), **config
        )
        return loop(dict(zip(loop._static_inputs, preds[len(feedback) :])))
