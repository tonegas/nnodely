import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely


@keras.saving.register_keras_serializable(package="nnodely")
class ModelClosedLoopImpl(keras.layers.Layer):
    """Repeatedly evaluate a model and feed selected streams back to its inputs."""

    def __init__(
        self,
        model,
        callbacks: dict[str, str],
        output_names: tuple[str, ...],
        input_time_axes: dict[str, int],
        steps: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.callbacks = dict(callbacks)
        self.output_names = tuple(output_names)
        self.input_time_axes = dict(input_time_axes)
        self.steps = int(steps)

    def call(self, inputs, training=None):
        states = dict(inputs)
        for _ in range(self.steps):
            values = self.model(states, training=training)
            if not isinstance(values, dict):
                values = dict(zip(self.model.output_names, values))

            next_states = dict(states)
            for input_name, stream_name in self.callbacks.items():
                state = states[input_name]
                feedback = values[stream_name]
                time_axis = self.input_time_axes[input_name]
                slices = [slice(None)] * len(state.shape)
                slices[time_axis] = slice(1, None)
                next_states[input_name] = keras.ops.concatenate(
                    [state[tuple(slices)], feedback], axis=time_axis
                )
            states = next_states

        return {name: values[name] for name in self.output_names}  # type: ignore

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": keras.saving.serialize_keras_object(self.model),
                "callbacks": self.callbacks,
                "output_names": self.output_names,
                "input_time_axes": self.input_time_axes,
                "steps": self.steps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        return cls(**config)


@keras.saving.register_keras_serializable(package="nnodely")
class LoopImpl(keras.layers.Layer):
    """Unroll a model while shifting its feedback into a temporal window."""

    def __init__(
        self,
        model,
        callback_input_name: str,
        callback_output_name: str,
        static_input_names: tuple[str, ...],
        steps: int,
        time_axis: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.callback_input_name = callback_input_name
        self.callback_output_name = callback_output_name
        self.static_input_names = tuple(static_input_names)
        self.steps = int(steps)
        self.time_axis = int(time_axis)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        state, *static_inputs = inputs

        for _ in range(self.steps):
            model_inputs = {self.callback_input_name: state}
            model_inputs.update(zip(self.static_input_names, static_inputs))
            feedback = self.model(model_inputs)
            if isinstance(feedback, dict):
                feedback = feedback[self.callback_output_name]

            slices = [slice(None)] * len(state.shape)
            slices[self.time_axis] = slice(1, None)
            state = keras.ops.concatenate(
                [state[tuple(slices)], feedback],
                axis=self.time_axis,
            )

        return state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": keras.saving.serialize_keras_object(self.model),
                "callback_input_name": self.callback_input_name,
                "callback_output_name": self.callback_output_name,
                "static_input_names": self.static_input_names,
                "steps": self.steps,
                "time_axis": self.time_axis,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        return cls(**config)


class Loop(Layer):
    """Unroll a built Modely over a temporal feedback window."""

    def __init__(
        self,
        f: Modely,
        callback: dict,
        steps: int | None = None,
        name=None,
    ):
        if not f.built:
            raise ValueError("Loop Modely body must be built before creating Loop.")
        if not isinstance(callback, dict) or len(callback) != 1:
            raise ValueError(
                "Loop callback must contain exactly one input: output pair."
            )

        callback_input, callback_output = next(iter(callback.items()))
        callback_input = self._find_node(f.inputs, callback_input, "input")
        callback_output = self._find_node(f.outputs, callback_output, "output")

        expected_output_shape = (
            tuple(callback_input.dim) + (1,) + tuple(callback_input.seq)
        )
        if callback_output.shape.tuple != expected_output_shape:
            raise ValueError(
                "Loop callback output must produce one temporal sample with shape "
                f"{expected_output_shape}, got {callback_output.shape.tuple}."
            )

        if steps is None:
            steps = callback_input.time
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("Loop steps must be a positive integer.")

        self.f = f
        self.callback = {callback_input: callback_output}
        self.callback_input = callback_input
        self.callback_output = callback_output
        self.static_inputs = [node for node in f.inputs if node is not callback_input]
        self.steps = steps

        super().__init__(
            name=name,
            preds=[callback_input, *self.static_inputs],
            dim=callback_input.dim,
            time=callback_input.time,
            seq=callback_input.seq,
        )

    @staticmethod
    def _find_node(nodes, value, kind):
        if isinstance(value, str):
            matches = [node for node in nodes if node.name == value]
            if len(matches) != 1:
                raise ValueError(f"Loop callback {kind} {value!r} was not found.")
            return matches[0]
        if value not in nodes:
            raise ValueError(
                f"Loop callback {kind} {getattr(value, 'name', value)!r} "
                "does not belong to the provided Modely."
            )
        return value

    def build_layer(self):
        return LoopImpl(
            model=self.f.model,
            callback_input_name=self.callback_input.name,
            callback_output_name=self.callback_output.name,
            static_input_names=tuple(node.name for node in self.static_inputs),
            steps=self.steps,
            time_axis=self.callback_input.shape.dim_rank + 1,
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "steps": self.steps,
        }
