import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely


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
        if not isinstance(f, Modely):
            raise TypeError("Loop currently supports only f=Modely")

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
        loop_in_name = self.loop_in_stream.name
        loop_out_name = self.loop_out_stream.name

        if f._model is None:
            f.build()

        fn_input_names = [node.name for node in f.inputs]
        if loop_in_name not in fn_input_names:
            raise ValueError(
                f"{self.name}: closed-loop input '{loop_in_name}' is not among f.inputs={fn_input_names}"
            )

        class _LoopImpl(keras.layers.Layer):
            def __init__(self, outer, name=None):
                super().__init__(name=name)
                self.outer = outer

            def _pad_to_horizon(self, x, horizon):
                # x shape: (batch, S, ...)
                s = keras.ops.shape(x)[1]
                pad_len = horizon - s

                def pad_fn():
                    pad_shape = keras.ops.concatenate(
                        [
                            keras.ops.shape(x)[:1],              # batch
                            keras.ops.reshape(pad_len, (1,)),   # padded seq
                            keras.ops.shape(x)[2:],             # rest
                        ],
                        axis=0,
                    )
                    zeros = keras.ops.zeros(pad_shape, dtype=x.dtype)
                    return keras.ops.concatenate([x, zeros], axis=1)

                return keras.ops.cond(pad_len > 0, pad_fn, lambda: x)

            def _make_valid_mask(self, x, horizon):
                # x shape: (batch, S, ...)
                s = keras.ops.shape(x)[1]
                t = keras.ops.arange(horizon)
                return keras.ops.cast(t < s, "bool")  # (horizon,)

            def call(self, inputs):
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]

                if len(inputs) != len(fn_input_names):
                    raise ValueError(
                        f"{self.outer.name}: expected {len(fn_input_names)} inputs, got {len(inputs)}"
                    )

                # infer common horizon
                seq_lengths = [keras.ops.shape(x)[1] for x in inputs]
                horizon = seq_lengths[0]
                for s in seq_lengths[1:]:
                    horizon = keras.ops.maximum(horizon, s)

                padded_inputs = [self._pad_to_horizon(x, horizon) for x in inputs]
                valid_masks = [self._make_valid_mask(x, horizon) for x in inputs]

                # scan wants leading scan axis first:
                # (batch, S, ...) -> (S, batch, ...)
                xs = [
                    keras.ops.transpose(x, [1, 0] + list(range(2, len(x.shape))))
                    for x in padded_inputs
                ]

                loop_in_idx = fn_input_names.index(loop_in_name)

                def step(carry, x_t_pack):
                    prev_y = carry
                    step_inputs = {}

                    for idx, input_name in enumerate(fn_input_names):
                        x_t = x_t_pack[0][idx]   # current tensor slice
                        valid_t = x_t_pack[1][idx]  # scalar bool for this step

                        if idx == loop_in_idx:
                            # before sequence end -> use dataset value
                            # after sequence end  -> use previous output
                            x_used = keras.ops.cond(valid_t, lambda: x_t, lambda: prev_y)
                        else:
                            # shorter non-looped inputs are already padded with zeros
                            x_used = x_t

                        step_inputs[input_name] = x_used

                    y = f._model(step_inputs)

                    if isinstance(y, dict):
                        y_t = y[loop_out_name]
                    else:
                        y_t = y

                    # carry == y_t ; in TF backend this is the easiest stable case
                    return y_t, y_t

                # scan over both data slices and masks
                xs_pack = (
                    [x for x in xs],           # per-step input slices
                    [m for m in valid_masks],  # per-step validity flags
                )

                # initial carry: zeros like one step of output
                sample_input = padded_inputs[0]          # (batch, S, ...)
                batch = keras.ops.shape(sample_input)[0]
                out_shape = self.outer.shape             # symbolic shape without batch
                init_shape = keras.ops.concatenate(
                    [keras.ops.reshape(batch, (1,)), keras.ops.convert_to_tensor(out_shape)],
                    axis=0,
                )
                init = keras.ops.zeros(init_shape, dtype=sample_input.dtype)

                _, ys = keras.ops.scan(step, init, xs_pack)

                # ys shape: (S, batch, ...)
                ys = keras.ops.transpose(ys, [1, 0] + list(range(2, len(ys.shape))))
                return ys

        self._layer = _LoopImpl(self, name=self.name)
        return self._layer