import contextlib

import keras
from keras.src.backend.common import stateless_scope

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.core.stream import Shape, Stream


#: Steps a dynamic rollout is declared with when no ``length`` is given. It
#: only shapes the graph: the rollout follows the data it is given.
DYNAMIC_BUILD_LENGTH = 1


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
    def __init__(self, loop, index: int, output_node, result_seq):
        self.index = int(index)
        super().__init__(
            name=f"{loop.name}_{output_node.name}",
            preds=[loop],
            dim=output_node.dim,
            time=output_node.time,
            seq=result_seq,
            index=self.index,
        )

    def build_layer(self):
        return LoopOutputImpl(index=self.index, name=self.name)

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The outputs belong to the loop that was just rebuilt, and selecting
        # one of them is what unpacking the loop does.
        loop = preds[0]  # type: ignore[index]
        loop.return_all_outputs = True
        return loop.loop_outputs[config["index"]]


@keras.saving.register_keras_serializable(package="nnodely")
class LoopImpl(keras.layers.Layer):
    """Apply a Keras model recurrently with one or more feedback states.

    The rollout axis is the last sequence axis of the inputs that carry it: such
    an input is one sequence rank deeper than the body input it feeds, so every
    step consumes a single slice and the body is evaluated exactly once per step.

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
        callback_input_names: tuple[str, ...],
        callback_output_names: tuple[str, ...],
        model_output_names: tuple[str, ...],
        static_input_names: tuple[str, ...],
        horizon: int,
        collect: bool = True,
        initial_sequence_axes: tuple[int | None, ...] = (),
        static_loop_axes: tuple[int | None, ...] = (),
        output_sequence_axes: tuple[int, ...] = (),
        callback_shift_axes: tuple[int | None, ...] = (),
        callback_state_shapes: tuple[tuple[int, ...] | None, ...] = (),
        result_shapes: tuple[tuple[int | None, ...], ...] = (),
        horizon_index: int | None = None,
        horizon_axis: int | None = None,
        primary_output_index: int = 0,
        return_all_outputs: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.callback_input_names = tuple(callback_input_names)
        self.callback_output_names = tuple(callback_output_names)
        self.model_output_names = tuple(model_output_names)
        self.static_input_names = tuple(static_input_names)
        self.horizon = int(horizon)
        self.collect = bool(collect)
        self.initial_sequence_axes = tuple(initial_sequence_axes)
        self.static_loop_axes = tuple(static_loop_axes)
        self.output_sequence_axes = tuple(output_sequence_axes)
        self.callback_shift_axes = tuple(callback_shift_axes)
        self.callback_state_shapes = tuple(
            None if shape is None else tuple(int(size) for size in shape)
            for shape in callback_state_shapes
        )
        self.result_shapes = tuple(tuple(shape) for shape in result_shapes)
        self.horizon_index = horizon_index
        self.horizon_axis = horizon_axis
        self.primary_output_index = int(primary_output_index)
        self.return_all_outputs = bool(return_all_outputs)
        self.output_indices = {
            name: index for index, name in enumerate(self.model_output_names)
        }

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _move_scan_axis(value, axis):
        rank = len(value.shape)
        permutation = list(range(1, rank))
        permutation.insert(axis, 0)
        return keras.ops.transpose(value, permutation)

    @staticmethod
    def _move_axis_front(value, axis):
        rank = len(value.shape)
        permutation = [axis] + [index for index in range(rank) if index != axis]
        return keras.ops.transpose(value, permutation)

    @staticmethod
    def _newest_slot(window, axis):
        slices = [slice(None)] * len(window.shape)
        slices[axis] = slice(-1, None)
        return window[tuple(slices)]

    @staticmethod
    def _shift_window(window, value, axis):
        """Drop the oldest step of the window and append the new one."""
        slices = [slice(None)] * len(window.shape)
        slices[axis] = slice(1, None)
        return keras.ops.concatenate([window[tuple(slices)], value], axis=axis)

    @staticmethod
    def _align_to_shape(state, shape):
        """Broadcast an initial value ``(batch, ...)`` up to ``(batch, *shape)``.

        Its axes line up with the last ones of ``shape``, the way the value was
        checked against its callback input when the loop was declared.
        """
        while len(state.shape) < len(shape) + 1:
            state = keras.ops.expand_dims(state, axis=1)
        return state + keras.ops.zeros((1, *shape))

    def _model_outputs(self, model_inputs):
        outputs = self.model(model_inputs)
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
            width = value.shape[self.horizon_axis]
            if isinstance(width, int):
                return width
            if keras.backend.backend() == "tensorflow":
                import tensorflow as tf

                return tf.shape(value)[self.horizon_axis]
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

        callback_count = len(self.callback_input_names)
        initial_values = inputs[:callback_count]
        sources = list(inputs[callback_count:])

        horizon = self._resolve_horizon(inputs)

        # Inputs carrying the rollout axis are consumed one slice per step; the
        # others are closed over, so they are not tiled horizon times.
        xs = []
        xs_positions = {}
        for index, (value, axis) in enumerate(zip(sources, self.static_loop_axes)):
            if axis is None:
                continue
            xs_positions[index] = len(xs)
            xs.append(self._move_axis_front(value, axis)[:horizon])

        states = []
        windows = []
        for initial, initial_axis, shift_axis, state_shape in zip(
            initial_values,
            self.initial_sequence_axes,
            self.callback_shift_axes,
            self.callback_state_shapes,
        ):
            state = initial
            if initial_axis is not None:
                state = keras.ops.take(state, 0, axis=initial_axis)
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
                    self.callback_input_names, step_states, step_windows
                )
            }
            for index, name in enumerate(self.static_input_names):
                position = xs_positions.get(index)
                model_inputs[name] = (
                    sources[index] if position is None else x_step[position]
                )
            return self._model_outputs(model_inputs)

        output_count = len(self.model_output_names)
        callback_slots = {
            self.output_indices[name]: index
            for index, name in enumerate(self.callback_output_names)
        }
        shifted = [
            index
            for index, axis in enumerate(self.callback_shift_axes)
            if axis is not None
        ]

        # Outputs that are not fed back still need a carry leaf of the right
        # shape and dtype; one extra evaluation of the body provides it.
        seeds = (
            None
            if len(callback_slots) == output_count
            else compute(states, windows, [value[0] for value in xs])
        )
        init_carry = tuple(
            states[callback_slots[index]] if index in callback_slots else seeds[index]  # type: ignore
            for index in range(output_count)
        ) + tuple(windows[index] for index in shifted)

        def step(carry, x_step):
            outputs = carry[:output_count]
            step_states = [
                outputs[self.output_indices[name]]
                for name in self.callback_output_names
            ]
            step_windows = [None] * callback_count
            for slot, index in enumerate(shifted):
                step_windows[index] = carry[output_count + slot]

            step_outputs = compute(step_states, step_windows, x_step)
            return step_outputs + tuple(
                self._shift_window(
                    step_windows[index],
                    step_outputs[
                        self.output_indices[self.callback_output_names[index]]
                    ],
                    self.callback_shift_axes[index],
                )
                for index in shifted
            )

        carry, trajectory = self._rollout(step, init_carry, xs, horizon, output_count)

        if self.collect:
            results = tuple(
                self._move_scan_axis(trajectory[position], axis)
                for position, axis in enumerate(self.output_sequence_axes)
            )
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
                "callback_input_names": self.callback_input_names,
                "callback_output_names": self.callback_output_names,
                "model_output_names": self.model_output_names,
                "static_input_names": self.static_input_names,
                "horizon": self.horizon,
                "collect": self.collect,
                "initial_sequence_axes": self.initial_sequence_axes,
                "static_loop_axes": self.static_loop_axes,
                "output_sequence_axes": self.output_sequence_axes,
                "callback_shift_axes": self.callback_shift_axes,
                "callback_state_shapes": self.callback_state_shapes,
                "result_shapes": self.result_shapes,
                "horizon_index": self.horizon_index,
                "horizon_axis": self.horizon_axis,
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


class Loop(Layer):
    """Roll out a Modely by closing one or more of its outputs onto its inputs.

    ::

        Loop(f=body, callback={state: next_state})({state: s0}, {force: F_seq})

    The loop is declared with its body and callback, and bound to the outer
    graph by calling it with two dicts: ``initial`` (callback input: seed) and
    ``inputs`` (body input: outer stream). A callback input missing from
    ``initial`` starts at zero; a body input missing from ``inputs`` keeps its
    own value at every step.

    The rollout axis lives outside the body: every stream that carries it is one
    sequence rank deeper than the body input it feeds, and its last sequence axis
    is the one the rollout consumes. A feedback input takes that stream through
    ``initial`` (step 0 seeds the state), an exogenous one through ``inputs``.

    ``length`` pins the number of steps. It is required only when no initial
    value or input carries a rollout axis. A rollout declared dynamic
    (``seq=-1``) follows the length of the sequence it is actually given, and
    without ``length`` its streams are declared with ``DYNAMIC_BUILD_LENGTH``
    steps.

    ``collect`` returns the whole trajectory, with the rollout axis appended as
    the last sequence axis of every output; ``collect=False`` returns only the
    final step.

    A feedback input declared with a time window (``x.sw(n)``) is closed by
    shifting: the oldest step is dropped and the new output appended, so after
    ``n`` steps the window is entirely self-predicted.
    """

    def __init__(
        self,
        f: Modely,
        callback: dict,
        name=None,
        length: int | None = None,
        collect: bool = True,
    ):
        if not f.built:
            f.build()
        if not isinstance(callback, dict) or not callback:
            raise ValueError(
                "Loop callback must contain at least one input: output pair."
            )

        callback_pairs = [
            (
                self._find_node(f.inputs, input_value, "input"),
                self._find_node(f.outputs, output_value, "output"),
            )
            for input_value, output_value in callback.items()
        ]
        callback_inputs = [pair[0] for pair in callback_pairs]
        callback_outputs = [pair[1] for pair in callback_pairs]
        if len(set(node.name for node in callback_outputs)) != len(callback_outputs):
            raise ValueError(
                "Loop cannot feed the same output back into more than one input."
            )

        self.f = f
        self.callback = callback
        self.callback_inputs = callback_inputs
        self.callback_outputs = callback_outputs
        self.static_inputs = [node for node in f.inputs if node not in callback_inputs]
        self.callback_shift_axes = self._resolve_feedback(callback_pairs)
        self.length = None if length is None else int(length)
        self.collect = bool(collect)
        self.return_all_outputs = False
        super().__init__(name=name)
        # Unbound until called: it has no shape to be used as a stream with.
        self._shape = None

    @property
    def shape(self):
        if self._shape is None:
            raise ValueError(
                f"Loop {self.name!r} is not bound: call it with its initial and "
                "inputs dicts, Loop(...)(initial, inputs), before using it as a "
                "stream."
            )
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    def __call__(self, initial: dict, inputs: dict={}):  # type: ignore[override]
        """Bind the loop to its initial values and outer inputs.

        Every call returns a new loop node; the body, and so its weights, is
        the one the loop was declared with.
        """
        node = Loop(
            f=self.f,
            callback=self.callback,
            name=self.name,
            length=self.length,
            collect=self.collect,
        )
        node.initial_values = self._resolve_initial_values(
            initial, self.callback_inputs
        )
        node.static_sources = self._resolve_sources(
            inputs, self.static_inputs, self.callback_inputs
        )
        node._configure()

        result_sequences = [node._result_sequence(output) for output in self.f.outputs]
        node.preds = [*node.initial_values, *node.static_sources]
        node.shape = Shape(
            dim=self.f.outputs[0].dim,
            time=self.f.outputs[0].time,
            seq=result_sequences[0],
        )
        node.loop_outputs = [
            LoopOutput(node, index, output, result_sequences[index])
            for index, output in enumerate(self.f.outputs)
        ]
        return node

    # ------------------------------------------------------------------
    # Axis configuration
    # ------------------------------------------------------------------
    def _configure(self):
        self.initial_sequence_axes = [
            self._loop_axis(node, value)
            for node, value in zip(self.callback_inputs, self.initial_values)
        ]
        self.static_loop_axes = [
            self._loop_axis(node, source)
            for node, source in zip(self.static_inputs, self.static_sources)
        ]
        for node, source, axis in zip(
            self.static_inputs, self.static_sources, self.static_loop_axes
        ):
            if source is node or axis is not None:
                continue
            if tuple(source.seq) != tuple(node.seq):
                raise ValueError(
                    f"Loop input {node.name!r} expects seq {tuple(node.seq)} or one "
                    f"extra rollout axis, got {tuple(source.seq)} from "
                    f"{source.name!r}."
                )

        sources = [*self.initial_values, *self.static_sources]
        loop_axes = [*self.initial_sequence_axes, *self.static_loop_axes]
        loop_sources = [
            (index, source, axis)
            for index, (source, axis) in enumerate(zip(sources, loop_axes))
            if axis is not None
        ]
        self.horizon = self._resolve_length(loop_sources)

        # The rollout needs a static number of steps, so a dynamic rollout axis
        # is only read from the tensor that carries it, at call time.
        self.horizon_index, self.horizon_axis = next(
            (
                (index, axis)
                for index, source, axis in loop_sources
                if source.seq[-1] is None
            ),
            (None, None),
        )
        self.output_sequence_axes = [
            output.shape.dim_rank + 2 + len(output.seq) for output in self.f.outputs
        ]
        self.callback_state_shapes = [
            self._state_shape(node, value, axis)
            for node, value, axis in zip(
                self.callback_inputs, self.initial_values, self.initial_sequence_axes
            )
        ]

    def _resolve_length(self, loop_sources):
        widths = {
            int(source.seq[-1])
            for _, source, _ in loop_sources
            if source.seq[-1] is not None
        }
        if self.length is not None:
            return self.length
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
            "carries a rollout axis, so pass length= to Loop."
        )

    def _result_sequence(self, output):
        if self.collect:
            return (*tuple(output.seq), self.horizon)
        return tuple(output.seq)

    # ------------------------------------------------------------------
    # Declaration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _loop_axis(node, source):
        """Tensor axis of ``source`` feeding ``node`` one slice per step, if any."""
        node_seq, source_seq = tuple(node.seq), tuple(source.seq)
        if len(source_seq) != len(node_seq) + 1:
            return None
        if source_seq[: len(node_seq)] != node_seq:
            return None
        return source.shape.dim_rank + 2 + len(node_seq)

    @staticmethod
    def _state_shape(node, value, axis):
        """Shape an initial value must be broadcast to, or None if it fits."""
        source = tuple(value.shape.tuple)
        if axis is not None:
            source = source[:-1]
        target = tuple(node.shape.tuple)
        if source == target:
            return None
        incompatible = len(source) > len(target) or any(
            size not in (1, target_size)
            for size, target_size in zip(reversed(source), reversed(target))
        )
        if incompatible or any(size is None for size in target):
            raise ValueError(
                f"Loop initial value {value.name!r} has shape {source}, which cannot "
                f"feed callback input {node.name!r} of shape {target}."
            )
        return target

    @staticmethod
    def _resolve_feedback(callback_pairs):
        """Time axis each feedback output is shifted into, or None to replace."""
        shift_axes = []
        for node, output in callback_pairs:
            if tuple(output.dim) != tuple(node.dim):
                raise ValueError(
                    f"Loop callback {node.name!r} -> {output.name!r}: dim "
                    f"{tuple(output.dim)} does not match the input dim "
                    f"{tuple(node.dim)}."
                )
            if output.time == node.time:
                shift_axes.append(None)
            elif output.time == 1 and node.time > 1:
                shift_axes.append(node.shape.dim_rank + 1)
            else:
                raise ValueError(
                    f"Loop callback {node.name!r} -> {output.name!r}: the output must "
                    f"either cover the whole time window of the input ({node.time}) or "
                    f"a single step to shift into it, got {output.time}."
                )
        return shift_axes

    @staticmethod
    def _resolve_sources(inputs, static_inputs, callback_inputs):
        """Outer stream feeding each body input that is not fed back."""
        if not isinstance(inputs, dict):
            raise TypeError("Loop inputs must be a dict of body input: outer stream.")

        by_name = {}
        for key, value in inputs.items():
            key_name = key.name if isinstance(key, Stream) else key
            if not isinstance(value, Stream):
                raise TypeError(
                    f"Loop input {key_name!r} must be bound to a Stream, got {value!r}."
                )
            by_name[key_name] = value

        callback_names = {node.name for node in callback_inputs}
        static_names = {node.name for node in static_inputs}
        for key_name in by_name:
            if key_name in callback_names:
                raise ValueError(
                    f"Loop input {key_name!r} is a callback input; its rollout source "
                    "comes from initial, not from inputs."
                )
            if key_name not in static_names:
                raise ValueError(
                    f"Loop input {key_name!r} is not an input of the body model."
                )

        sources = []
        for node in static_inputs:
            source = by_name.get(node.name, node)
            if source is not node and (
                tuple(source.dim) != tuple(node.dim) or source.time != node.time
            ):
                raise ValueError(
                    f"Loop input {node.name!r} expects dim {tuple(node.dim)} and time "
                    f"{node.time}, got dim {tuple(source.dim)} and time {source.time} "
                    f"from {source.name!r}."
                )
            sources.append(source)
        return sources

    @staticmethod
    def _resolve_initial_values(initial, callback_inputs):
        """Seed of every callback input; the ones not given start at zero."""
        if not isinstance(initial, dict):
            raise TypeError(
                "Loop initial must be a dict of callback input: initial value."
            )
        by_name = {
            key.name if isinstance(key, Stream) else key: value
            for key, value in initial.items()
        }
        expected = {node.name for node in callback_inputs}
        unknown = set(by_name) - expected
        if unknown:
            raise ValueError(
                f"Loop initial keys {sorted(unknown)} are not callback inputs; "
                f"expected a subset of {sorted(expected)}."
            )
        values = [by_name.get(node.name, 0.0) for node in callback_inputs]

        from nnodely.layers.constant import Constant

        return [
            value if isinstance(value, Stream) else Constant(name=None, value=value)
            for value in values
        ]

    @staticmethod
    def _find_node(nodes, value, kind):
        if isinstance(value, str):
            matches = [node for node in nodes if node.name == value]
            if len(matches) != 1:
                raise ValueError(f"callback {kind} {value!r} was not found.")
            return matches[0]
        if value not in nodes:
            raise ValueError(
                f"callback {kind} {getattr(value, 'name', value)!r} "
                "does not belong to the provided Modely."
            )
        return value

    def __iter__(self):
        self.return_all_outputs = True
        return iter(self.loop_outputs)

    def build_layer(self):
        return LoopImpl(
            model=self.f.inference_model,
            callback_input_names=tuple(node.name for node in self.callback_inputs),
            callback_output_names=tuple(node.name for node in self.callback_outputs),
            model_output_names=tuple(node.name for node in self.f.outputs),
            static_input_names=tuple(node.name for node in self.static_inputs),
            horizon=self.horizon,
            collect=self.collect,
            initial_sequence_axes=tuple(self.initial_sequence_axes),
            static_loop_axes=tuple(self.static_loop_axes),
            output_sequence_axes=tuple(self.output_sequence_axes),
            callback_shift_axes=tuple(self.callback_shift_axes),
            callback_state_shapes=tuple(self.callback_state_shapes),
            result_shapes=tuple(
                (*output.dim, output.time, *self._result_sequence(output))
                for output in self.f.outputs
            ),
            horizon_index=self.horizon_index,
            horizon_axis=self.horizon_axis,
            primary_output_index=self.f.outputs.index(self.callback_outputs[0]),
            return_all_outputs=self.return_all_outputs,
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "callback": {
                node.name: output.name
                for node, output in zip(self.callback_inputs, self.callback_outputs)
            },
            "length": self.length,
            "collect": self.collect,
        }

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # `f` is the reloaded body (see ModelSerializer). The predecessors are
        # the initial value of every callback, then the source of every other
        # body input in the body's order. A body input left unbound was its own
        # source; the reloaded body is a copy of its own, so it is bound to the
        # node of the same name in the graph the loop is loaded in.
        config = dict(config)
        body, callback = config.pop("f"), config.pop("callback")
        preds = list(preds or [])
        static_names = [node.name for node in body.inputs if node.name not in callback]
        return cls(f=body, callback=callback, **config)(
            dict(zip(callback, preds[: len(callback)])),
            dict(zip(static_names, preds[len(callback) :])),
        )
