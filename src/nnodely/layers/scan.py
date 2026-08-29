import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.core.stream import Stream
from nnodely.layers.input import Input


@keras.saving.register_keras_serializable(package="nnodely")
class ScanImpl(keras.layers.Layer):
    """Apply a Keras model recurrently and return the final feedback value."""

    def __init__(
        self,
        model,
        callback_input_name: str,
        callback_output_name: str,
        static_input_names: tuple[str, ...],
        horizon: int,
        sequence_axis: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.callback_input_name = callback_input_name
        self.callback_output_name = callback_output_name
        self.static_input_names = tuple(static_input_names)
        self.horizon = int(horizon)
        self.sequence_axis = int(sequence_axis)

    def call(self, inputs):
        initial, sequence_anchor, *static_inputs = inputs

        state = initial
        while len(state.shape) < len(sequence_anchor.shape):
            state = keras.ops.expand_dims(state, axis=0)
        state = state + keras.ops.zeros_like(sequence_anchor)

        def step(carry, _):
            model_inputs = {self.callback_input_name: carry}
            model_inputs.update(zip(self.static_input_names, static_inputs))
            outputs = self.model(model_inputs)
            if isinstance(outputs, dict):
                outputs = outputs[self.callback_output_name]
            return outputs, outputs

        final_state, _ = keras.ops.scan(
            step,
            init=state,
            xs=None,
            length=self.horizon,
        )
        return keras.ops.take(final_state, -1, axis=self.sequence_axis)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": keras.saving.serialize_keras_object(self.model),
                "callback_input_name": self.callback_input_name,
                "callback_output_name": self.callback_output_name,
                "static_input_names": self.static_input_names,
                "horizon": self.horizon,
                "sequence_axis": self.sequence_axis,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        return cls(**config)


class Scan(Layer):
    """Close one output of a built Modely onto one of its sequenced inputs."""

    def __init__(
        self,
        f: Modely,
        callback: dict,
        initial: Stream | Input | float | int = 0.0,
        name=None,
    ):
        if not f.built:
            raise ValueError("Loop Modely body must be built before creating Loop.")
        if (
            not isinstance(callback, dict) or len(callback) != 1
        ):  # TODO: in future support multiple feedbacks
            raise ValueError(
                "Scan callback must contain exactly one input: output pair."
            )

        callback_input, callback_output = next(iter(callback.items()))
        callback_input = self._find_node(f.inputs, callback_input, "input")
        callback_output = self._find_node(f.outputs, callback_output, "output")

        if not callback_input.seq or callback_input.seq[0] is None:
            raise ValueError(
                "Scan callback input must have a fixed first sequence dimension, "
                f"got seq={callback_input.seq}."
            )
        # if callback_output.shape.tuple != callback_input.shape.tuple:
        #     raise ValueError(
        #         "Scan callback output shape must match its input shape, got "
        #         f"{callback_output.shape.tuple} and {callback_input.shape.tuple}."
        #     )

        static_inputs = [node for node in f.inputs if node is not callback_input]
        sequenced_static = [node.name for node in static_inputs if node.seq]
        if sequenced_static:
            raise ValueError(
                "Scan currently supports non-feedback inputs without seq only; "
                f"got {sequenced_static}."
            )

        if not isinstance(initial, Stream):
            from nnodely.layers.constant import Constant

            initial = Constant(name=None, value=initial)

        self.f = f
        self.callback = {callback_input: callback_output}
        self.callback_input = callback_input
        self.callback_output = callback_output
        self.static_inputs = static_inputs
        self.initial = initial
        self.horizon = int(callback_input.seq[0])
        self.remaining_seq = tuple(callback_input.seq[1:])

        super().__init__(
            name=name,
            preds=[initial, callback_input, *static_inputs],
            dim=callback_input.dim,
            time=callback_input.time,
            seq=self.remaining_seq,
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
        return ScanImpl(
            model=self.f.model,
            callback_input_name=self.callback_input.name,
            callback_output_name=self.callback_output.name,
            static_input_names=tuple(node.name for node in self.static_inputs),
            horizon=self.horizon,
            sequence_axis=self.callback_input.shape.dim_rank + 2,
            name=self.name,
        )

    def get_config(self):
        return {
            "name": self.name,
            "horizon": self.horizon,
        }
