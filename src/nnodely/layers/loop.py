import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.layers.input import Input
from nnodely.core.stream import Node


class LoopImpl(keras.layers.Layer):
    def __init__(
        self,
        submodel,
        closed_loop: dict[str, str],
        initial_values: dict[str, str],
        inputs,
        submodel_output_names,
        name,
        longest_seq_idx,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.submodel = submodel
        self.closed_loop = closed_loop
        self.initial_values = initial_values
        self.inputs = inputs
        self.submodel_output_names = submodel_output_names
        self.longest_seq_idx = longest_seq_idx

        # Validate input sequence dimensions to be the same for all inputs, except the zero default (non-sequence) value.
        seq_dims = set()
        for inp in self.inputs:  # type:ignore
            if inp.seq != ():
                seq_dims.add(inp.seq[-1])  # rightmost seq dimension
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
                "initial_values": self.initial_values,
                "inputs": self.inputs,
                "longest_seq_idx": self.longest_seq_idx,
                "submodel_output_names": self.submodel_output_names,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["submodel"] = keras.saving.deserialize_keras_object(config["submodel"])
        return cls(**config)

    def _shift_time_window(self, prev_tensor, new_tensor, idx):
        time_axis = self.inputs[idx].time_index  # type:ignore

        # If the incoming tensor represents a single new timestep,
        # append it after dropping the oldest timestep.
        new_time_dim = new_tensor.shape[time_axis]

        if new_time_dim == 1:
            rank = len(prev_tensor.shape)

            slices = [slice(None)] * rank
            slices[time_axis] = slice(1, None)

            shifted = prev_tensor[tuple(slices)]

            return keras.ops.concatenate(
                [shifted, new_tensor],
                axis=time_axis,
            )

        # Otherwise expect a complete replacement window.
        prev_time_dim = prev_tensor.shape[time_axis]

        if (
            new_time_dim is not None
            and prev_time_dim is not None
            and new_time_dim != prev_time_dim
        ):
            raise ValueError(
                f"Cannot shift time window: "
                f"new_tensor time dimension {new_time_dim} "
                f"is not equal to prev_tensor time dimension {prev_time_dim}"
            )

        return new_tensor

    def call_for(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # Determine horizon from the known sequence metadata stored during build.
        horizon = (
            keras.ops.shape(inputs[self.longest_seq_idx])[-1]
            if self.longest_seq_idx is not None and inputs[self.longest_seq_idx].shape[-1] is not None
            else 1 # TODO: handle dynamic horizon from dataset or a special parameter in training
        )

        # Prepare initial step inputs by taking the first time step from each input sequence.
        step_inputs = {}
        y = None
        for idx, inp in enumerate(self.submodel.inputs):
            if self.inputs[idx].name in self.initial_values:  # type:ignore
                step_inputs[inp.name] = (
                    inputs[idx] if self.inputs[idx].seq == () else inputs[idx][..., 0]  # type:ignore
                )
            else:
                step_inputs[inp.name] = inputs[idx][..., 0]

        # Iteratively call the submodel for each time step, updating closed-loop inputs with previous outputs as needed.
        outputs = {}

        for t in range(horizon):
            for idx, inp in enumerate(self.inputs):  # type:ignore
                if t > 0 and inp.name in self.closed_loop:
                    if (
                        inp.time == 1 or inp.time == ()
                    ):  # If the closed-loop input has no time dimension, we can directly use the previous output without shifting.
                        loop_value = (
                            y[self.closed_loop[inp.name]] if isinstance(y, dict) else y
                        )
                        if (
                            getattr(loop_value.shape, "rank", None)  # type:ignore
                            == len(self.inputs[idx].shape) + 1  # type:ignore
                        ):
                            loop_value = loop_value[:, 0]  # type:ignore
                        step_inputs[self.submodel.inputs[idx].name] = loop_value
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
                else:  # Not a closed-loop input or first time step, take the appropriate time slice from the input sequence.
                    if (
                        inp.seq == ()
                    ):  # No sequence dimension, broadcast the input across the horizon.
                        step_inputs[self.submodel.inputs[idx].name] = inputs[idx]
                    elif (
                        inputs[idx].shape[-1] is not None and inputs[idx].shape[-1] > t
                    ):  # If the input sequence has enough time steps, take the t-th step.
                        step_inputs[self.submodel.inputs[idx].name] = inputs[idx][
                            ..., t
                        ]
                    else:  # Input sequence is shorter than the horizon, repeat the last time step.
                        step_inputs[self.submodel.inputs[idx].name] = inputs[idx][
                            ..., -1
                        ]

            y = self.submodel(step_inputs)

            for out_name, out_value in (
                y.items() if isinstance(y, dict) else {"output": y}.items()
            ):
                out_value = keras.ops.expand_dims(out_value, axis=-1)
                if out_name not in outputs:
                    outputs[out_name] = [out_value]
                else:
                    outputs[out_name].append(out_value)

        if len(outputs) == 1:
            return next(iter(outputs.values()))
        else:
            for out_name in outputs:
                outputs[out_name] = keras.ops.stack(outputs[out_name], axis=-1)
        return outputs

    def call(self, inputs):
        """
        Roll out the submodel over the horizon using keras.ops.scan.

        Carry  - a flat list of tensors, one per closed-loop input, holding the
                "previous output" that will be fed back at the next step.
        xs     - a list of tensors shaped [horizon, ...] (loop axis moved to
                front), one per input, containing the pre-sliced per-step
                values.  For non-sequence inputs (seq == ()) we still create a
                repeated slice so scan sees a uniform xs structure.

        At each step the step_fn
        1. rebuilds the full step_inputs dict,
        2. replaces closed-loop slots with the corresponding carry tensor,
        3. calls self.submodel,
        4. returns the updated carry and the raw output dict / tensor.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # ------------------------------------------------------------------ #
        # 1.  Determine horizon                                                #
        # ------------------------------------------------------------------ #
        if (
            self.longest_seq_idx is not None
            and inputs[self.longest_seq_idx].shape[-1] is not None
        ):
            horizon = inputs[self.longest_seq_idx].shape[-1]
        else:
            horizon = keras.ops.shape(inputs[self.longest_seq_idx])[-1] if self.longest_seq_idx is not None else 1

        # ------------------------------------------------------------------ #
        # 2.  Build xs: [horizon, ...] tensors for every input slot           #
        #                                                                      #
        #  * seq == ()  → broadcast the single tensor across all steps        #
        #  * seq != ()  → move the time axis (always the last axis) to front, #
        #                 pad / truncate to `horizon` steps                    #
        # ------------------------------------------------------------------ #
        def _prepare_xs(tensor, inp_meta):
            """Return a tensor with shape [horizon, *spatial_dims]."""
            if inp_meta.seq == ():
                # Non-sequence input: repeat the same value for every step.
                # Result shape: [horizon, *tensor.shape]
                expanded = keras.ops.expand_dims(tensor, axis=0)           # [1, ...]
                multiples = [horizon] + [1] * len(tensor.shape)
                return keras.ops.tile(expanded, multiples)                 # [horizon, ...]

            # Sequence input: last axis is time.
            # Move it to the front so scan can iterate over it.
            rank = len(tensor.shape)
            perm = [rank - 1] + list(range(rank - 1))                     # time first
            t_first = keras.ops.transpose(tensor, perm)                    # [T, ...]

            seq_len = tensor.shape[-1]
            if seq_len is not None and seq_len >= horizon:
                # Enough steps – just slice.
                slices = [slice(None)] * (rank)
                slices[0] = slice(0, horizon)
                return t_first[tuple(slices)]                              # [horizon, ...]

            # Sequence shorter than horizon: repeat last element to pad.
            last = t_first[-1:]                                            # [1, ...]
            pad_len = horizon - (seq_len if seq_len is not None else keras.ops.shape(t_first)[0])
            pad = keras.ops.tile(last, [pad_len] + [1] * (rank - 1))
            return keras.ops.concatenate([t_first, pad], axis=0)           # [horizon, ...]

        xs = [_prepare_xs(inp, self.inputs[idx]) for idx, inp in enumerate(inputs)]

        # ------------------------------------------------------------------ #
        # 3.  Build initial carry                                            #
        # ------------------------------------------------------------------ #
        sub_inp_name_to_carry_idx = {inp: idx for idx, inp in enumerate(self.closed_loop)}
        inp_name_to_idx = {inp.name: idx for idx, inp in enumerate(self.inputs)}
        out_name_to_idx = {out: idx for idx, out in enumerate(self.submodel_output_names)}

        # initial_values dict maps input-name → initial tensor (first xs step).
        init_carry = [xs[inp_name_to_idx[name]][0] for name in self.initial_values.values()]

        # ------------------------------------------------------------------ #
        # 4.  Define the scan step function                                  #
        # ------------------------------------------------------------------ #

        def step_fn(carry, x_step):
            """
            carry : list of tensors - previous closed-loop outputs.
            x_step: list of tensors - current time-step slice for every input.
            Returns (new_carry, y_step).
            """
            step_inputs = {}
            for idx, inp_meta in enumerate(self.inputs):
                submodel_name = self.submodel.inputs[idx].name
                if submodel_name in self.closed_loop:
                    # Closed-loop slot: use carry (previous output).
                    prev_out = carry[sub_inp_name_to_carry_idx[submodel_name]]

                    if inp_meta.time == 1 or inp_meta.time == ():
                        # No time window – direct feedback.
                        # Remove trailing singleton if the output has one more dim.
                        if len(prev_out.shape) == len(self.inputs[idx].shape) + 1:
                            prev_out = prev_out[:, 0]
                        step_inputs[submodel_name] = prev_out
                    else:
                        # Time window – shift: drop oldest, append newest output.
                        step_inputs[submodel_name] = self._shift_time_window(
                            x_step[idx],    # current window (already updated in xs)
                            prev_out,
                            idx,
                        )
                else:
                    step_inputs[submodel_name] = x_step[idx]

            y = self.submodel(step_inputs)
            y = [v for v in (y.values() if isinstance(y, dict) else [y])]

            # Build new carry from the model outputs.
            new_carry = [y[out_name_to_idx[name]] for name in self.closed_loop.values()]
            return new_carry, y

        # ------------------------------------------------------------------ #
        # 5.  Run keras.ops.scan                                             #
        # ------------------------------------------------------------------ #
        # xs for scan must be a structure where the leading axis is the loop
        # axis; we pass a list of [horizon, ...] tensors.
        _, ys = keras.ops.scan(
            step_fn,
            init_carry,
            xs=xs,
            length=horizon,
        )

        # ------------------------------------------------------------------- #
        # 6.  Post-process outputs                                            #
        #                                                                     #
        #  scan stacks outputs along axis 0 → shape [horizon, batch, *dims].  #
        #  We need to move the horizon axis to the last position to match the #
        #  for-loop version: [batch, *dims, horizon].                         #
        # ------------------------------------------------------------------- #
        def _horizon_to_last(tensor):
            """Move axis 0 (horizon) to the last position."""
            rank = len(keras.ops.shape(tensor)) if not isinstance(tensor, (list, tuple)) else len(tensor)
            perm = list(range(1, rank)) + [0]
            return keras.ops.transpose(tensor, perm)

        if isinstance(ys, dict):
            if len(ys) == 1:
                key = next(iter(ys))
                return _horizon_to_last(ys[key])
            return {k: _horizon_to_last(v) for k, v in ys.items()}
        else:
            # Single output tensor from scan: shape [horizon, batch, *dims].
            if len(ys) == 1:
                # Only one output key – unwrap list if scan returned a list.
                out = ys[0] if isinstance(ys, (list, tuple)) else ys
                return _horizon_to_last(out)
            return tuple(_horizon_to_last(v) for v in ys)

def _solve_dict_names(d: dict[str | Input, str | Node]) -> dict[str, str]:
    result = {}
    for key, value in d.items():
        if isinstance(key, Input):
            key_name = key.name
        elif isinstance(key, str):
            key_name = key
        else:
            raise ValueError("closed_loop keys must be strings or Input instances.")

        if isinstance(value, Node):
            value_name = value.name
        elif isinstance(value, str):
            value_name = value
        else:
            raise ValueError("closed_loop values must be strings or Node instances.")

        result[key_name] = value_name
    return result

def _sort_dict_by_keys(d: dict[str, str], keys: list[str]) -> dict[str, str]:
    return {k: d[k] for k in keys if k in d}

class Loop(Layer):
    """
    Roll out a one-step Modely over the rightmost sequence axis.
    """

    def __init__(
        self,
        f: Modely,
        closed_loop: dict[str | Input, str | Node],
        initial_values: dict[str | Input, str | Node],
        name=None,
    ):
        self.f = f

        self.closed_loop = _solve_dict_names(closed_loop)
        self.initial_values = _solve_dict_names(initial_values)

        # check that closed_loop and initial_values keys are valid inputs to f
        f_input_names = [node.name for node in self.f.inputs]
        for key in self.closed_loop.keys():
            if key not in f_input_names:
                raise ValueError(
                    f"Loop: closed_loop key '{key}' not in f.inputs={f_input_names}"
                )

        for key in self.initial_values.keys():
            if key not in f_input_names:
                raise ValueError(
                    f"Loop: initial_values key '{key}' not in f.inputs={f_input_names}"
                )

        # check outputs in closed_loop values are valid outputs of f
        self.f_output_names = [node.name for node in self.f.outputs]
        for out in self.closed_loop.values():
            if out not in self.f_output_names:
                raise ValueError(
                    f"Loop: closed_loop output '{out}' not in f.outputs={self.f_output_names}"
                )

        # check if closed_loop and initial_values keys are valid inputs to f
        for f_inp in f_input_names:
            if f_inp in self.closed_loop and f_inp not in self.initial_values:
                raise ValueError(
                    f"Loop: closed_loop key '{f_inp}' must also be in initial_values."
                )
            if f_inp in self.initial_values and f_inp not in self.closed_loop:
                raise ValueError(
                    f"Loop: initial_values key '{f_inp}' must also be in closed_loop."
                )

        self.closed_loop = _sort_dict_by_keys(self.closed_loop, f_input_names)
        self.initial_values = _sort_dict_by_keys(self.initial_values, f_input_names)

        super().__init__(
            name=name,
            f=f,
            closed_loop=self.closed_loop,
            initial_values=self.initial_values,
        )

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
        return [(out_node.dim, out_node.time, out_seq) for out_node in self.f.outputs]

    def build_layer(self):
        if not self.f.built:
            self.f.build()
        
        # check inputs in initial_values values are valid outputs of the outer model
        input_names = [node.name for node in self.inputs]
        for inp in self.initial_values.values():
            if inp not in input_names:
                raise ValueError(
                    f"Loop: initial_values input '{inp}' not in outer model inputs={input_names}"
                )

        # save the index of the longest sequence input for use during call.
        sequences = [
            inp.seq[-1] if len(inp.seq) > 0 else inp.seq
            for inp in self.inputs
        ]
        tmp_max = -1
        self.longest_seq_idx = None
        for idx, seq in enumerate(sequences):
            if seq is not None:
                if seq != () and seq > tmp_max:
                    tmp_max = seq
                    self.longest_seq_idx = idx
            else:
                self.longest_seq_idx = idx

        fn_output_names = [node.name for node in self.f.outputs]
        if any(out not in fn_output_names for out in self.closed_loop.values()):
            raise ValueError(
                f"{self.name}: closed-loop outputs: '{[out for out in self.closed_loop.values() if out not in fn_output_names]}' not in f.outputs={fn_output_names}"
            )

        return LoopImpl(
            self.f.model,
            self.closed_loop,
            self.initial_values,
            inputs=self.inputs,
            name=self.name,
            longest_seq_idx=self.longest_seq_idx,
            submodel_output_names=self.f_output_names
        )