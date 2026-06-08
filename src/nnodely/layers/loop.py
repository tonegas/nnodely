import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely


class LoopImpl(keras.layers.Layer):
    def __init__(
        self,
        submodel,
        closed_loop: dict[str, str],
        name=None,
        inputs=None,
        longest_seq_idx=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.submodel = submodel
        self.closed_loop = closed_loop
        self.inputs = inputs
        self.longest_seq_idx = longest_seq_idx

        # Validate input sequence dimensions to be the same for all inputs, except the zero default (non-sequence) value.
        seq_dims = set()
        for inp in self.inputs:  # type:ignore
            if inp.seq != ():
                seq_dims.add(inp.seq[0])  # leftmost seq dimension
        if len(seq_dims) > 1:
            raise ValueError(
                f"LoopImpl: all inputs must have the same seq dimensions or not have a sequence, got {seq_dims}"
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "submodel": keras.saving.serialize_keras_object(self.submodel),
                "closed_loop": self.closed_loop,
                "inputs": self.inputs,
                "longest_seq_idx": self.longest_seq_idx,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["submodel"] = keras.saving.deserialize_keras_object(config["submodel"])
        return cls(**config)

    def _shift_time_window(self, prev_tensor, new_tensor, idx):
        # Shift the time window by one step.
        # For example, if time dimension is at index 2 and shape is (batch, features, time, seq), we want to shift along axis=2.
        if (
            new_tensor.shape[self.inputs[idx].time_index] is not None  # type:ignore
            and new_tensor.shape[self.inputs[idx].time_index] == 1  # type:ignore
        ):
            return keras.ops.concatenate(
                [prev_tensor[..., 1:], new_tensor],
                axis=self.inputs[idx].time_index,  # type:ignore
            )
        else:  # new_tensor has a bigger time window
            # Check if the new_tensor has the same time dimension as prev_tensor.
            if (
                new_tensor.shape[self.inputs[idx].time_index]  # type:ignore
                != prev_tensor.shape[self.inputs[idx].time_index]  # type:ignore
            ):
                raise ValueError(
                    f"Cannot shift time window: new_tensor time dimension {new_tensor.shape[self.inputs[idx].time_index]} is not equal to prev_tensor time dimension {prev_tensor.shape[self.inputs[idx].time_index]}"  # type:ignore
                )
            return new_tensor

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # Determine horizon from the known sequence metadata stored during build.
        print(
            f"LoopImpl.call: self.longest_seq_idx={self.longest_seq_idx}, inputs shapes={[keras.ops.shape(inp) for inp in inputs]}"
        )  # type:ignore
        horizon = keras.ops.shape(inputs[self.longest_seq_idx])[-1]  # type:ignore

        # Prepare initial step inputs by taking the first time step from each input sequence.
        step_inputs = {}
        y = None
        for idx, inp in enumerate(self.submodel.inputs):
            if (
                self.inputs[idx].seq == ()  # type:ignore
            ):  # No sequence dimension, broadcast the input across the horizon.
                step_inputs[inp.name] = inputs[idx]
            else:
                step_inputs[inp.name] = inputs[idx][..., 0]

        # Iteratively call the submodel for each time step, updating closed-loop inputs with previous outputs as needed.
        outputs = {}
        for t in range(horizon):
            for idx, inp in enumerate(self.inputs):  # type:ignore
                if t > 0 and inp.name in self.closed_loop:
                    if self.inputs[idx].time == 1:  # type:ignore
                        step_inputs[self.submodel.inputs[idx].name] = (
                            y[self.closed_loop[inp.name]] if isinstance(y, dict) else y
                        )
                    else:  # If the closed-loop input has a time dimension, the time window must be shifted.
                        step_inputs[self.submodel.inputs[idx].name] = (
                            self._shift_time_window(
                                step_inputs[self.submodel.inputs[idx].name],
                                y[self.closed_loop[inp.name]]
                                if isinstance(y, dict)
                                else y,
                                idx,
                            )
                        )
                else:
                    if (
                        self.inputs[idx].seq == ()  # type:ignore
                    ):  # No sequence dimension, broadcast the input across the horizon.
                        step_inputs[self.submodel.inputs[idx].name] = inputs[idx]
                    elif (
                        inputs[idx].shape[-1] is not None and inputs[idx].shape[-1] > t
                    ):
                        step_inputs[self.submodel.inputs[idx].name] = inputs[idx][
                            ..., t
                        ]
                    else:
                        step_inputs[self.submodel.inputs[idx].name] = inputs[idx][
                            ..., -1
                        ]

            y = self.submodel(step_inputs)

            for out_name, out_value in (
                y.items() if isinstance(y, dict) else {"output": y}.items()
            ):
                # out_value = (
                #     out_value
                #     if len(keras.ops.shape(inputs[self.longest_seq_idx])[-1]) < 1  # type:ignore
                #     else keras.ops.expand_dims(out_value, axis=-1)
                # )
                out_value = keras.ops.expand_dims(out_value, axis=-1)
                if out_name not in outputs:
                    outputs[out_name] = out_value
                else:
                    outputs[out_name] = keras.ops.concatenate(
                        [outputs[out_name], out_value], axis=-1
                    )

        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs

    # Alternative implementation of the call method that uses keras.ops.scan. (Not currently used, slower than the for-loop version)
    # def call_scan(self, inputs):
    #     if not isinstance(inputs, (list, tuple)):
    #         inputs = [inputs]

    #     horizon = self.longest_seq[-1] if self.longest_seq else 1

    #     def _slice_to_horizon(tensor):
    #         seq_len = keras.ops.shape(tensor)[-1]
    #         indices = keras.ops.minimum(
    #             keras.ops.arange(horizon, dtype=tf.int32),
    #             tf.cast(seq_len - 1, tf.int32),
    #         )
    #         return tf.gather(tensor, indices, axis=-1)

    #     def _move_loop_axis_front(tensor):
    #         rank = tensor.shape.rank
    #         perm = [rank - 1] + list(range(rank - 1))
    #         return tf.transpose(tensor, perm=perm)

    #     def _flatten_output(output):
    #         return next(iter(output.values())) if isinstance(output, dict) else output

    #     step_tensors = [_move_loop_axis_front(_slice_to_horizon(t)) for t in inputs]
    #     init_step = {
    #         self.submodel.inputs[idx].name: step_tensors[idx][0]
    #         for idx in range(len(inputs))
    #     }

    #     y0 = _flatten_output(self.submodel(init_step))
    #     if horizon == 1:
    #         return (
    #             y0 if len(self.longest_seq) < 1 else keras.ops.expand_dims(y0, axis=-1)  # type:ignore
    #         )

    #     xs = [tensor[1:] for tensor in step_tensors]

    #     def step_fn(prev_output, step_values):
    #         step_inputs = {}
    #         for idx, inp in enumerate(self.inputs):  # type:ignore
    #             if inp.name in self.closed_loop:
    #                 step_inputs[self.submodel.inputs[idx].name] = prev_output
    #             else:
    #                 step_inputs[self.submodel.inputs[idx].name] = step_values[idx]

    #         y = _flatten_output(self.submodel(step_inputs))
    #         return y, y

    #     _, scanned_outputs = keras.ops.scan(step_fn, y0, xs=xs, length=horizon - 1)
    #     outputs = keras.ops.concatenate(
    #         [keras.ops.expand_dims(y0, axis=0), scanned_outputs], axis=0
    #     )
    #     perm = list(range(1, outputs.shape.rank)) + [0]  # type:ignore
    #     return tf.transpose(outputs, perm=perm)


class Loop(Layer):
    """
    Roll out a one-step Modely over the rightmost sequence axis.

    Semantics:
    - The layer unrolls over the rightmost sequence axis of its inputs (axis=1).
    - Inputs without a sequence axis are broadcast across the horizon.
    - Inputs with seq=1 are repeated across the horizon.
    - If inputs have multiple seq dimensions (nested loops), only the rightmost is
      iterated by this Loop; remaining seq dims are passed through to the inner model.
    - The closed-loop mapped input is updated each step with the submodel output.
    """

    def __init__(self, f: Modely, closed_loop: dict, name=None):
        if len(f.outputs) != 1:
            raise ValueError("Loop currently supports Modely with exactly one output")

        if len(closed_loop) != 1:
            raise ValueError("Loop currently supports exactly one closed_loop pair")

        self.f = f
        self.closed_loop = dict(closed_loop)

        super().__init__(name=name, f=f, closed_loop=self.closed_loop)

    def output_shape(self, *inputs):
        # Determine loop horizon from the deepest sequence input.
        seq_inputs = [inp.seq for inp in inputs if len(inp.seq) > 0]
        if seq_inputs:
            longest_seq = max(seq_inputs, key=len)
            horizon = longest_seq[-1]
        else:
            horizon = None

        out_node = self.f.outputs[0]
        out_seq = tuple(out_node.seq) + (horizon,)
        return out_seq, out_node.time, out_node.dim

    def build_layer(self):
        # save the index of the longest sequence input for use during call.
        sequences = [
            inp.seq[-1] if len(inp.seq) > 0 else inp.seq
            for inp in self.inputs  # type:ignore
        ]
        tmp_max = -1
        for idx, seq in enumerate(sequences):
            if seq is not None:
                if seq != () and seq > tmp_max:
                    tmp_max = seq
                    self.longest_seq_idx = idx
            else:
                self.longest_seq_idx = idx

        if self.f.model is None:
            self.f.build()

        fn_output_names = [node.name for node in self.f.outputs]
        if any(out not in fn_output_names for out in self.closed_loop.values()):
            raise ValueError(
                f"{self.name}: closed-loop outputs: '{[out for out in self.closed_loop.values() if out not in fn_output_names]}' not in f.outputs={fn_output_names}"
            )
        return LoopImpl(
            self.f.model,
            self.closed_loop,
            name=self.name,
            inputs=self.inputs,
            longest_seq_idx=self.longest_seq_idx,
        )
