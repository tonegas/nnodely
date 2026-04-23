import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely


@keras.saving.register_keras_serializable(package="nnodely")
class LoopImpl(keras.layers.Layer):
    def __init__(self, submodel, loop_out_name, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.submodel = submodel
        self.loop_out_name = loop_out_name

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "submodel": keras.saving.serialize_keras_object(self.submodel),
                "loop_out_name": self.loop_out_name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["submodel"] = keras.saving.deserialize_keras_object(config["submodel"])
        return cls(**config)

    def _pad_to_horizon(self, x, horizon):
        s = keras.ops.shape(x)[1]
        pad_len = horizon - s

        def pad_fn():
            pad_shape = keras.ops.concatenate(
                [
                    keras.ops.shape(x)[:1],
                    keras.ops.reshape(pad_len, (1,)),
                    keras.ops.shape(x)[2:],
                ],
                axis=0,
            )
            zeros = keras.ops.zeros(pad_shape, dtype=x.dtype)
            return keras.ops.concatenate([x, zeros], axis=1)

        return keras.ops.cond(pad_len > 0, pad_fn, lambda: x)

    def _make_valid_mask(self, x, horizon):
        s = keras.ops.shape(x)[1]
        t = keras.ops.arange(horizon)
        return keras.ops.cast(t < s, "bool")

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        fn_input_names = [node.name for node in self.submodel.inputs]
        if len(inputs) != len(fn_input_names):
            raise ValueError(
                f"{self.name}: expected {len(fn_input_names)} inputs, got {len(inputs)}"
            )

        seq_lengths = [keras.ops.shape(x)[1] for x in inputs]
        horizon = seq_lengths[0]
        for s in seq_lengths[1:]:
            horizon = keras.ops.maximum(horizon, s)

        padded_inputs = [self._pad_to_horizon(x, horizon) for x in inputs]
        valid_masks = [self._make_valid_mask(x, horizon) for x in inputs]

        xs = [
            keras.ops.transpose(x, [1, 0] + list(range(2, len(x.shape))))
            for x in padded_inputs
        ]

        loop_in_idx = 0

        def step(carry, x_t_pack):
            prev_y = carry
            step_inputs = {}

            for idx, input_name in enumerate(fn_input_names):
                x_t = x_t_pack[0][idx]

                if idx == loop_in_idx:
                    x_used = prev_y
                else:
                    x_used = x_t

                step_inputs[input_name] = x_used

            y = self.submodel(step_inputs)

            if isinstance(y, dict):
                y_t = y[self.loop_out_name]
            else:
                y_t = y

            return y_t, y_t

        xs_pack = (
            [x for x in xs],
            [m for m in valid_masks],
        )

        sample_input = padded_inputs[0]
        batch = keras.ops.shape(sample_input)[0]

        step_out_node = self.submodel.outputs[0]
        step_out_shape = [1 if dim is None else dim for dim in step_out_node.shape[1:]]

        init_shape = keras.ops.concatenate(
            [
                keras.ops.reshape(batch, (1,)),
                keras.ops.convert_to_tensor(step_out_shape, dtype="int32"),
            ],
            axis=0,
        )
        init = keras.ops.zeros(init_shape, dtype=sample_input.dtype)
        _, ys = keras.ops.scan(step, init, xs_pack)

        ys = keras.ops.transpose(ys, [1, 0] + list(range(2, len(ys.shape))))
        return ys


class Loop(Layer):
    """
    Roll out a one-step Modely over the first seq axis.

    First stable version restrictions:
    - f must be a Modely
    - f must have exactly one output
    - closed_loop must have exactly one {output: input} pair
    - each ingress stream must have exactly one seq axis
    - the order of Loop(...) inputs must match f.inputs
    """

    node_type = "Loop"

    def __init__(self, f: Modely, closed_loop: dict, name=None):
        if len(f.outputs) != 1:
            raise ValueError("Loop currently supports Modely with exactly one output")

        if len(closed_loop) != 1:
            raise ValueError("Loop currently supports exactly one closed_loop pair")

        self.f = f
        self.closed_loop = dict(closed_loop)

        # extract the single mapping
        self.loop_out_stream, self.loop_in_stream = next(iter(self.closed_loop.items()))

        super().__init__(name=name, f=f, closed_loop=self.closed_loop)

    def output_shape(self, seqs, times, dims):
        # all ingress streams must have exactly one seq axis
        for seq in seqs:
            if len(seq) != 1:
                raise ValueError(
                    f"{self.name}: each Loop input must have exactly one seq axis, got seq={seq}"
                )

        horizon = max(seq[0] for seq in seqs)
        out_node = self.f.outputs[0]
        return (horizon,), out_node.time, out_node.dim

    def build_layer(self):
        f = self.f
        # loop_in_name = self.loop_in_stream.name
        loop_out_name = self.loop_out_stream.name

        if f._model is None:
            f.build()

        # fn_input_names = [node.name for node in f.inputs]
        # if loop_in_name not in fn_input_names:
        #     raise ValueError(f"{self.name}: closed-loop input '{loop_in_name}' is not among f.inputs={fn_input_names}")
        fn_output_names = [node.name for node in f.outputs]
        if loop_out_name not in fn_output_names:
            raise ValueError(
                f"{self.name}: closed-loop output '{loop_out_name}' is not among f.outputs={fn_output_names}"
            )

        self._layer = LoopImpl(f._model, loop_out_name, name=self.name)
        return self._layer
