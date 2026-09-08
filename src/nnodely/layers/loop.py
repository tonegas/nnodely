import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.core.stream import Stream


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

    def get_config(self):
        return {"name": self.name, "index": self.index}


@keras.saving.register_keras_serializable(package="nnodely")
class LoopImpl(keras.layers.Layer):
    """Apply a Keras model recurrently with one or more feedback states.

    The rollout axis is the last sequence axis of the inputs that carry it: such
    an input is one sequence rank deeper than the body input it feeds, so every
    step consumes a single slice and the body is evaluated exactly once per step.

    On the TensorFlow backend ``keras.ops.scan`` requires the per-step output of
    the step function to match the carry in structure, shape and dtype, so the
    carry holds every body output followed by one window per shifted feedback
    state, and the step function returns it unchanged as its per-step output.
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
        batch_reference_index: int | None = None,
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
        self.batch_reference_index = batch_reference_index
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
    def _first_step(scan_values):
        if isinstance(scan_values, list):
            return [value[0] for value in scan_values]
        return scan_values[0]

    @staticmethod
    def _align_to_shape(state, shape, reference):
        """Broadcast an initial value up to ``(batch, *shape)``."""
        while len(state.shape) < len(shape) + 1:
            state = keras.ops.expand_dims(state, axis=0)
        state = state + keras.ops.zeros((1, *shape))
        if reference is None:
            return state
        # A constant initial value carries no batch axis, while the carry has to
        # match the body output, so borrow the batch from a batched input.
        batch = keras.ops.sum(
            reference * 0.0, axis=tuple(range(1, len(reference.shape)))
        )
        return state + keras.ops.reshape(batch, (-1, *([1] * len(shape))))

    def _model_outputs(self, model_inputs):
        outputs = self.model(model_inputs)
        if isinstance(outputs, dict):
            return tuple(outputs[name] for name in self.model_output_names)
        if isinstance(outputs, (list, tuple)):
            return tuple(outputs)
        return (outputs,)

    def _resolve_horizon(self, inputs):
        """Number of rollout steps, always a Python int.

        ``keras.ops.scan`` needs a static trip count, so the horizon is never
        read symbolically. A rollout axis declared as ``None`` follows the
        sequence it is actually given, which is what makes a longer input roll
        out further; a declared width always wins over the tensor.
        """
        if self.horizon_index is not None:
            width = inputs[self.horizon_index].shape[self.horizon_axis]
            if isinstance(width, int):
                return width
        return self.horizon

    def compute_output_spec(self, inputs):
        """Declare the outputs instead of letting Keras trace `call`.

        The JAX backend infers shapes by tracing with a symbolic dimension, which
        `keras.ops.scan` rejects because its trip count has to be static. The
        shapes are known from the graph anyway, and a dynamic rollout keeps its
        declared length here: the tensor it is given decides the real one.
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
        batch_reference = (
            None
            if self.batch_reference_index is None
            else inputs[self.batch_reference_index]
        )

        # Inputs carrying the rollout axis are consumed one slice per step; the
        # others are closed over, so they are not tiled horizon times.
        xs = []
        xs_positions = {}
        for index, (value, axis) in enumerate(zip(sources, self.static_loop_axes)):
            if axis is None:
                continue
            xs_positions[index] = len(xs)
            xs.append(self._move_axis_front(value, axis)[:horizon])
        scan_values = xs if xs else keras.ops.zeros((horizon,))

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
                state = self._align_to_shape(state, state_shape, batch_reference)

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
            else compute(states, windows, self._first_step(scan_values))
        )
        init_carry = tuple(
            states[callback_slots[index]] if index in callback_slots else seeds[index]
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
            new_carry = step_outputs + tuple(
                self._shift_window(
                    step_windows[index],
                    step_outputs[
                        self.output_indices[self.callback_output_names[index]]
                    ],
                    self.callback_shift_axes[index],
                )
                for index in shifted
            )
            return new_carry, new_carry

        final, scanned = keras.ops.scan(
            step,
            init=init_carry,
            xs=scan_values,
            length=horizon,
        )

        if self.collect:
            results = tuple(
                self._move_scan_axis(value, axis)
                for value, axis in zip(scanned[:output_count], self.output_sequence_axes)
            )
        else:
            results = tuple(final[:output_count])

        if self.return_all_outputs:
            return results
        return results[self.primary_output_index]

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
                "batch_reference_index": self.batch_reference_index,
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

    The rollout axis lives outside the body: every stream that carries it is one
    sequence rank deeper than the body input it feeds, and its last sequence axis
    is the one the rollout consumes. A feedback input takes that stream through
    ``initial`` (step 0 seeds the state), an exogenous one through ``inputs``.
    Body inputs left unbound keep their own value at every step.

    ``length`` pins the number of steps. It is required when no rollout input
    declares a concrete width, and a rollout declared dynamic (``seq=(None,)``)
    follows the length of the sequence it is actually given.

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
        initial: Stream | float | int | dict = 0.0,
        inputs: dict | None = None,
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

        initial_values = self._resolve_initial_values(initial, callback_inputs)
        static_inputs = [node for node in f.inputs if node not in callback_inputs]
        static_sources = self._resolve_sources(inputs, static_inputs, callback_inputs)

        self.f = f
        self.callback = dict(callback_pairs)
        self.callback_inputs = callback_inputs
        self.callback_outputs = callback_outputs
        self.static_inputs = static_inputs
        self.static_sources = static_sources
        self.initial_values = initial_values
        self.callback_shift_axes = self._resolve_feedback(callback_pairs)
        self.length = None if length is None else int(length)
        self.collect = bool(collect)
        self.return_all_outputs = False
        self._configure()

        result_sequences = [self._result_sequence(output) for output in f.outputs]
        first_output = f.outputs[0]
        super().__init__(
            name=name,
            preds=[*initial_values, *static_sources],
            dim=first_output.dim,
            time=first_output.time,
            seq=result_sequences[0],
        )
        self.loop_outputs = [
            LoopOutput(self, index, output, result_sequences[index])
            for index, output in enumerate(f.outputs)
        ]

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

        # `keras.ops.scan` needs a static trip count, so a dynamic rollout axis
        # is only read from the tensor that carries it, at call time.
        self.horizon_index, self.horizon_axis = next(
            (
                (index, axis)
                for index, source, axis in loop_sources
                if source.seq[-1] is None
            ),
            (None, None),
        )
        self.batch_reference_index = next(
            (index for index, source in enumerate(sources) if self._has_batch(source)),
            None,
        )
        if self.batch_reference_index is None:
            raise ValueError(
                "Loop needs at least one input with a batch axis: pass a stream "
                "derived from an Input through initial= or inputs=."
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
        raise ValueError(
            "Loop cannot determine the rollout length. `keras.ops.scan` needs a "
            "static number of steps, so either pass length= to Loop or give one of "
            "its rollout inputs a concrete seq=."
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
    def _has_batch(node):
        """False for constants and parameters, which have no batch axis."""
        return not (isinstance(node, Layer) and len(node.preds) == 0)

    @classmethod
    def _state_shape(cls, node, value, axis):
        """Shape an initial value must be broadcast to, or None if it fits."""
        source = tuple(value.shape.tuple)
        if axis is not None:
            source = source[:-1]
        target = tuple(node.shape.tuple)
        if source == target and cls._has_batch(value):
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
        if inputs is None:
            return list(static_inputs)
        if not isinstance(inputs, dict):
            raise ValueError("Loop inputs must be a dict of body input: outer stream.")

        by_name = {}
        for key, value in inputs.items():
            key_name = key.name if isinstance(key, Stream) else key
            if not isinstance(value, Stream):
                raise ValueError(
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
        if isinstance(initial, dict):
            by_name = {
                key.name if isinstance(key, Stream) else key: value
                for key, value in initial.items()
            }
            expected = {node.name for node in callback_inputs}
            if set(by_name) != expected:
                raise ValueError(
                    "Loop initial keys must match callback inputs; "
                    f"expected {sorted(expected)}, got {sorted(by_name)}."
                )
            values = [by_name[node.name] for node in callback_inputs]
        elif len(callback_inputs) == 1:
            values = [initial]
        else:
            raise ValueError("Multiple Loop callbacks require an initial value dict.")

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
            model=self.f.model,
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
            batch_reference_index=self.batch_reference_index,
            primary_output_index=self.f.outputs.index(self.callback_outputs[0]),
            return_all_outputs=self.return_all_outputs,
            name=self.name,
        )

    def get_config(self):
        return {"name": self.name, "horizon": self.horizon}
