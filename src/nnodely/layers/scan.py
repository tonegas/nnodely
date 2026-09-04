import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.core.stream import Stream
from nnodely.layers.input import Input


@keras.saving.register_keras_serializable(package="nnodely")
class ScanOutputImpl(keras.layers.Layer):
    """Select one tensor from a multi-output Scan."""

    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[self.index]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


class ScanOutput(Layer):
    def __init__(self, scan, index: int, output_node, result_seq):
        self.index = int(index)
        super().__init__(
            name=f"{scan.name}_{output_node.name}",
            preds=[scan],
            dim=output_node.dim,
            time=output_node.time,
            seq=result_seq,
            index=self.index,
        )

    def build_layer(self):
        return ScanOutputImpl(index=self.index, name=self.name)

    def get_config(self):
        return {"name": self.name, "index": self.index}


@keras.saving.register_keras_serializable(package="nnodely")
class ScanImpl(keras.layers.Layer):
    """Apply a Keras model recurrently with one or more feedback states."""

    def __init__(
        self,
        model,
        callback_input_names: tuple[str, ...],
        callback_output_names: tuple[str, ...],
        model_output_names: tuple[str, ...],
        static_input_names: tuple[str, ...],
        horizon: int,
        collect_outputs: bool = False,
        initial_sequence_axes: tuple[int | None, ...] = (),
        expand_callback_axes: tuple[int | None, ...] = (),
        output_step_axes: tuple[int | None, ...] = (),
        output_sequence_axes: tuple[int | None, ...] = (),
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
        self.collect_outputs = bool(collect_outputs)
        self.initial_sequence_axes = tuple(initial_sequence_axes)
        self.expand_callback_axes = tuple(expand_callback_axes)
        self.output_step_axes = tuple(output_step_axes)
        self.output_sequence_axes = tuple(output_sequence_axes)
        self.primary_output_index = int(primary_output_index)
        self.return_all_outputs = bool(return_all_outputs)

    @staticmethod
    def _select_step(value, selector, axis, horizon):
        selector_shape = [1] * len(value.shape)
        selector_shape[axis] = horizon
        selector = keras.ops.reshape(selector, selector_shape)
        return keras.ops.sum(value * selector, axis=axis)

    @staticmethod
    def _move_scan_axis(value, axis):
        rank = len(value.shape)
        permutation = list(range(1, rank))
        permutation.insert(axis, 0)
        return keras.ops.transpose(value, permutation)

    def call(self, inputs):
        callback_count = len(self.callback_input_names)
        initial_values = inputs[:callback_count]
        sequence_anchors = inputs[callback_count : 2 * callback_count]
        static_inputs = inputs[2 * callback_count :]

        states = []
        for initial, anchor, initial_axis, expand_axis in zip(
            initial_values,
            sequence_anchors,
            self.initial_sequence_axes,
            self.expand_callback_axes,
        ):
            state = initial
            if initial_axis is not None:
                state = keras.ops.take(state, 0, axis=initial_axis)

            if self.collect_outputs:
                target_rank = len(anchor.shape) - int(expand_axis is not None)
                while len(state.shape) < target_rank:
                    state = keras.ops.expand_dims(state, axis=0)
                if expand_axis is not None:
                    state_anchor = keras.ops.take(anchor, 0, axis=expand_axis)
                    state = state + keras.ops.zeros_like(state_anchor)
            else:
                while len(state.shape) < len(anchor.shape):
                    state = keras.ops.expand_dims(state, axis=0)
                state = state + keras.ops.zeros_like(anchor)
            states.append(state)

        output_indices = {
            name: index for index, name in enumerate(self.model_output_names)
        }

        def step(carry, selector):
            model_inputs = {}
            for name, value, anchor, expand_axis in zip(
                self.callback_input_names,
                carry,
                sequence_anchors,
                self.expand_callback_axes,
            ):
                if expand_axis is not None:
                    value = keras.ops.expand_dims(value, axis=expand_axis)
                    value = value + keras.ops.zeros_like(anchor)
                model_inputs[name] = value
            model_inputs.update(zip(self.static_input_names, static_inputs))

            model_outputs = self.model(model_inputs)
            if isinstance(model_outputs, dict):
                output_values = tuple(
                    model_outputs[name] for name in self.model_output_names
                )
            elif isinstance(model_outputs, (list, tuple)):
                output_values = tuple(model_outputs)
            else:
                output_values = (model_outputs,)

            step_outputs = tuple(
                self._select_step(value, selector, axis, self.horizon)
                if axis is not None
                else value
                for value, axis in zip(output_values, self.output_step_axes)
            )
            new_carry = tuple(
                step_outputs[output_indices[name]]
                for name in self.callback_output_names
            )
            return new_carry, step_outputs

        scan_values = (
            keras.ops.eye(self.horizon)
            if any(axis is not None for axis in self.output_step_axes)
            else keras.ops.zeros((self.horizon,))
        )
        _, scanned_outputs = keras.ops.scan(
            step,
            init=tuple(states),
            xs=scan_values,
            length=self.horizon,
        )

        if self.collect_outputs:
            results = tuple(
                self._move_scan_axis(value, axis)
                for value, axis in zip(
                    scanned_outputs,
                    self.output_sequence_axes,
                )
            )
        else:
            last_outputs = tuple(value[-1] for value in scanned_outputs)
            results = tuple(
                keras.ops.take(value, -1, axis=axis) if axis is not None else value
                for value, axis in zip(last_outputs, self.output_sequence_axes)
            )

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
                "collect_outputs": self.collect_outputs,
                "initial_sequence_axes": self.initial_sequence_axes,
                "expand_callback_axes": self.expand_callback_axes,
                "output_step_axes": self.output_step_axes,
                "output_sequence_axes": self.output_sequence_axes,
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


class Scan(Layer):
    """Close one or more outputs of a built Modely onto its sequenced inputs."""

    def __init__(
        self,
        f: Modely,
        callback: dict,
        initial: Stream | Input | float | int | dict = 0.0,
        name=None,
    ):
        if not f.built:
            raise ValueError("Modely body must be built before creating Scan.")
        if not isinstance(callback, dict) or not callback:
            raise ValueError(
                "Scan callback must contain at least one input: output pair."
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

        callback_seq = tuple(callback_inputs[0].seq)
        if not callback_seq or callback_seq[0] is None:
            raise ValueError("Scan callback inputs require a fixed sequence dimension.")
        if any(tuple(node.seq) != callback_seq for node in callback_inputs[1:]):
            raise ValueError("All Scan callback inputs must have the same seq shape.")

        initial_values = self._resolve_initial_values(initial, callback_inputs)
        static_inputs = [node for node in f.inputs if node not in callback_inputs]
        sequenced_static = [node for node in static_inputs if node.seq]

        self.f = f
        self.callback = dict(callback_pairs)
        self.callback_inputs = callback_inputs
        self.callback_outputs = callback_outputs
        self.static_inputs = static_inputs
        self.initial_values = initial_values
        self.return_all_outputs = False
        self.collect_outputs = bool(sequenced_static) or all(
            value is node for value, node in zip(initial_values, callback_inputs)
        )

        self.initial_sequence_axes = [None] * len(callback_inputs)
        self.expand_callback_axes = [None] * len(callback_inputs)
        deeper_sources = [
            node
            for node in [*initial_values, *static_inputs]
            if len(node.seq) > len(callback_seq)
        ]

        if self.collect_outputs and deeper_sources:
            driver = max(deeper_sources, key=lambda node: len(node.seq))
            if tuple(driver.seq[: len(callback_seq)]) != callback_seq:
                raise ValueError(
                    f"Scan sequence {driver.seq} is incompatible with callback "
                    f"sequence {callback_seq}."
                )
            self.horizon = int(driver.seq[len(callback_seq)])
            self.remaining_seq = tuple(driver.seq)
            for index, value in enumerate(initial_values):
                if len(value.seq) > len(callback_seq):
                    self.initial_sequence_axes[index] = (  # type: ignore
                        value.shape.dim_rank + 2 + len(callback_seq)
                    )
            self.output_step_axes = [None] * len(f.outputs)
            self.output_sequence_axes = [
                output.shape.dim_rank + 2 + len(callback_seq) for output in f.outputs
            ]
        elif self.collect_outputs:
            self.horizon = int(callback_seq[0])
            self.remaining_seq = callback_seq
            for index, (value, callback_input) in enumerate(
                zip(initial_values, callback_inputs)
            ):
                axis = callback_input.shape.dim_rank + 2
                if value.seq:
                    self.initial_sequence_axes[index] = axis
                self.expand_callback_axes[index] = axis
            self.output_step_axes = [
                output.shape.dim_rank + 2 if output.seq else None
                for output in f.outputs
            ]
            self.output_sequence_axes = [
                output.shape.dim_rank + 2 for output in f.outputs
            ]
        else:
            self.horizon = int(callback_seq[0])
            self.remaining_seq = tuple(callback_seq[1:])
            self.output_step_axes = [None] * len(f.outputs)
            self.output_sequence_axes = [
                output.shape.dim_rank + 2 if output.seq else None
                for output in f.outputs
            ]

        result_sequences = [self._result_sequence(output) for output in f.outputs]
        first_output = f.outputs[0]
        super().__init__(
            name=name,
            preds=[*initial_values, *callback_inputs, *static_inputs],
            dim=first_output.dim,
            time=first_output.time,
            seq=result_sequences[0],
        )
        self.scan_outputs = [
            ScanOutput(self, index, output, result_sequences[index])
            for index, output in enumerate(f.outputs)
        ]

    def _result_sequence(self, output):
        if self.collect_outputs:
            return self.remaining_seq
        return tuple(output.seq[1:]) if output.seq else ()

    @staticmethod
    def _resolve_initial_values(initial, callback_inputs):
        if isinstance(initial, dict):
            by_name = {
                key.name if isinstance(key, Input) else key: value
                for key, value in initial.items()
            }
            expected = {node.name for node in callback_inputs}
            if set(by_name) != expected:
                raise ValueError(
                    "Scan initial keys must match callback inputs; "
                    f"expected {sorted(expected)}, got {sorted(by_name)}."
                )
            values = [by_name[node.name] for node in callback_inputs]
        elif len(callback_inputs) == 1:
            values = [initial]
        else:
            raise ValueError("Multiple Scan callbacks require an initial value dict.")

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
        return iter(self.scan_outputs)

    def build_layer(self):
        return ScanImpl(
            model=self.f.model,
            callback_input_names=tuple(node.name for node in self.callback_inputs),
            callback_output_names=tuple(node.name for node in self.callback_outputs),
            model_output_names=tuple(node.name for node in self.f.outputs),
            static_input_names=tuple(node.name for node in self.static_inputs),
            horizon=self.horizon,
            collect_outputs=self.collect_outputs,
            initial_sequence_axes=tuple(self.initial_sequence_axes),
            expand_callback_axes=tuple(self.expand_callback_axes),
            output_step_axes=tuple(self.output_step_axes),
            output_sequence_axes=tuple(self.output_sequence_axes),
            primary_output_index=self.f.outputs.index(self.callback_outputs[0]),
            return_all_outputs=self.return_all_outputs,
            name=self.name,
        )

    def get_config(self):
        return {"name": self.name, "horizon": self.horizon}
