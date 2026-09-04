import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.layers.input import Input
from nnodely.core.stream import Node


class _LoopInputSpec:
    """
    Serializable view of a Loop input.

    LoopImpl only ever reads name/dim/time/seq off the nnodely nodes, and those
    nodes carry the whole DAG with them, so they cannot go into get_config.
    """

    def __init__(self, name, dim, time, seq):
        self.name = name
        self.dim = tuple(dim)
        self.time = tuple(time) if isinstance(time, (list, tuple)) else time
        self.seq = tuple(seq)

    @classmethod
    def from_node(cls, node):
        return cls(node.name, node.dim, node.time, node.seq)

    # Mirrors Stream.shape / Stream.time_index.
    @property
    def shape(self):
        time = (self.time,) if isinstance(self.time, int) else self.time
        return self.dim + time + self.seq

    @property
    def time_index(self):
        return len(self.dim) + 1 if isinstance(self.time, int) else None

    def get_config(self):
        return {
            "name": self.name,
            "dim": list(self.dim),
            "time": self.time,
            "seq": list(self.seq),
        }


@keras.saving.register_keras_serializable(package="nnodely")
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
        length=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.submodel = submodel
        self.closed_loop = closed_loop
        self.initial_values = initial_values
        self.inputs = [
            inp if isinstance(inp, _LoopInputSpec) else _LoopInputSpec.from_node(inp)
            for inp in inputs
        ]
        self.submodel_output_names = submodel_output_names
        self.longest_seq_idx = longest_seq_idx
        self.length = length

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
                "inputs": [inp.get_config() for inp in self.inputs],
                "longest_seq_idx": self.longest_seq_idx,
                "submodel_output_names": self.submodel_output_names,
                "length": self.length,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["submodel"] = keras.saving.deserialize_keras_object(config["submodel"])
        config["inputs"] = [_LoopInputSpec(**spec) for spec in config["inputs"]]
        return cls(**config)

    def _resolve_horizon(self, inputs):
        """
        Number of rollout steps, always a Python int.

        `keras.ops.scan` lowers to `jax.lax.scan`, whose trip count must be
        static, so the horizon is never read symbolically. When the loop axis of
        the driving input is concrete (jit tracing and eager calls) it wins, so
        feeding a longer sequence rolls out further; otherwise (symbolic shape
        inference on a `seq=(None,)` input) fall back to the declared length.
        """
        if self.longest_seq_idx is not None:
            seq_len = inputs[self.longest_seq_idx].shape[-1]
            if isinstance(seq_len, int):
                return seq_len
        return self.length

    def compute_output_spec(self, inputs):
        """
        Declare the output spec directly so Keras never traces `call` with
        polymorphic dimensions.
        """
        horizon = self.length
        specs = tuple(
            keras.KerasTensor(
                shape=(inputs[0].shape[0],) + tuple(out.shape[1:]) + (horizon,),
                dtype=out.dtype,
            )
            for out in self.submodel.outputs
        )
        return specs[0] if len(specs) == 1 else specs

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

    def call(self, inputs):
        """
        Roll out the submodel over the horizon using keras.ops.scan.

        Carry  - a flat list of tensors, one per closed-loop input, holding the
                "previous output" that will be fed back at the next step.
        xs     - a list of tensors shaped [horizon, ...] (loop axis moved to
                front), one per *sequence* input, containing the pre-sliced
                per-step values.  Non-sequence inputs (seq == ()) are closed
                over by step_fn instead, so they are not tiled horizon times.

        At each step the step_fn
        1. rebuilds the full step_inputs dict,
        2. replaces closed-loop slots with the corresponding carry tensor,
        3. calls self.submodel,
        4. returns the updated carry and the raw output dict / tensor.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # ------------------------------------------------------------------ #
        # 1.  Determine horizon (a Python int, see _resolve_horizon)          #
        # ------------------------------------------------------------------ #
        horizon = self._resolve_horizon(inputs)

        # ------------------------------------------------------------------ #
        # 2.  Build xs: [horizon, ...] tensors for every sequence input       #
        #                                                                      #
        #  The time axis (always the last one) is moved to the front, then     #
        #  padded / truncated to `horizon` steps.  Non-sequence inputs stay    #
        #  out of xs entirely and are closed over by step_fn.                  #
        # ------------------------------------------------------------------ #
        def _prepare_xs(tensor):
            """Return a tensor with shape [horizon, *spatial_dims]."""
            rank = len(tensor.shape)
            perm = [rank - 1] + list(range(rank - 1))                     # time first
            t_first = keras.ops.transpose(tensor, perm)                    # [T, ...]

            seq_len = tensor.shape[-1]
            if seq_len >= horizon:
                # Enough steps – just slice.
                return t_first[:horizon]                                   # [horizon, ...]

            # Sequence shorter than horizon: repeat last element to pad.
            last = t_first[-1:]                                            # [1, ...]
            pad = keras.ops.tile(last, [horizon - seq_len] + [1] * (rank - 1))
            return keras.ops.concatenate([t_first, pad], axis=0)           # [horizon, ...]

        # Position of each sequence input inside xs; non-sequence inputs absent.
        xs_pos = {}
        xs = []
        for idx, inp_meta in enumerate(self.inputs):  # type:ignore
            if inp_meta.seq != ():
                xs_pos[idx] = len(xs)
                xs.append(_prepare_xs(inputs[idx]))

        def _step_slice(idx, x_step):
            """Current-step value for input `idx`, from xs or from the closure."""
            return x_step[xs_pos[idx]] if idx in xs_pos else inputs[idx]

        # ------------------------------------------------------------------ #
        # 3.  Build initial carry                                            #
        # ------------------------------------------------------------------ #
        sub_inp_name_to_carry_idx = {inp: idx for idx, inp in enumerate(self.closed_loop)}
        inp_name_to_idx = {inp.name: idx for idx, inp in enumerate(self.inputs)}
        out_name_to_idx = {out: idx for idx, out in enumerate(self.submodel_output_names)}

        # initial_values dict maps input-name → initial tensor (first xs step).
        init_carry = []
        for name in self.initial_values.values():
            idx = inp_name_to_idx[name]
            init_carry.append(xs[xs_pos[idx]][0] if idx in xs_pos else inputs[idx])

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
                            _step_slice(idx, x_step),  # current window
                            prev_out,
                            idx,
                        )
                else:
                    step_inputs[submodel_name] = _step_slice(idx, x_step)

            y = self.submodel(step_inputs)
            # Index by name: dict iteration order is not a contract.
            if isinstance(y, dict):
                y = [y[name] for name in self.submodel_output_names]
            elif isinstance(y, (list, tuple)):
                y = list(y)
            else:
                y = [y]

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
        length=None,
    ):
        self.f = f

        self.closed_loop = _solve_dict_names(closed_loop)
        self.initial_values = _solve_dict_names(initial_values)
        self.length = length

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
            length=length,
        )

    def _horizon(self, inputs):
        """
        Static rollout length: `length` pins it, otherwise the longest concrete
        sequence axis. None when neither is available.
        """
        if self.length is not None:
            return self.length
        static = [
            inp.seq[-1]
            for inp in inputs
            if len(inp.seq) > 0 and isinstance(inp.seq[-1], int)
        ]
        return max(static) if static else None

    def output_shape(self, *inputs):
        horizon = self._horizon(inputs)

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

        horizon = self._horizon(self.inputs)
        if horizon is None:
            raise ValueError(
                f"{self.name}: cannot determine the rollout length. "
                f"`keras.ops.scan` needs a static number of steps, so either pass "
                f"`length=` to Loop or give one of its inputs a concrete `seq=`."
            )

        return LoopImpl(
            self.f.model,
            self.closed_loop,
            self.initial_values,
            inputs=self.inputs,
            name=self.name,
            longest_seq_idx=self.longest_seq_idx,
            submodel_output_names=self.f_output_names,
            length=horizon,
        )