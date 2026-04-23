"""Local model layer."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer
from nnodely.layers.fir import FirImpl


@keras.saving.register_keras_serializable(package="nnodely")
class LocalModelImpl(keras.layers.Layer):
    def __init__(
        self,
        function_name="Fir",
        out_features=1,
        use_bias=True,
        num_models=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.function_name = str(function_name)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.num_models = int(num_models) if num_models is not None else None
        self.local_functions = []

        if self.function_name.lower() != "fir":
            raise ValueError(
                f"Unsupported LocalModel function {self.function_name!r}. "
                "Currently supported: 'Fir'."
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "function_name": self.function_name,
                "out_features": self.out_features,
                "use_bias": self.use_bias,
                "num_models": self.num_models,
            }
        )
        return config

    def _make_local_function(self, idx):
        if self.function_name.lower() == "fir":
            return FirImpl(
                out_features=self.out_features,
                use_bias=self.use_bias,
                name=f"{self.name}_local_{idx}",
            )
        raise ValueError(
            f"Unsupported LocalModel function {self.function_name!r}. "
            "Currently supported: 'Fir'."
        )

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
            raise ValueError(
                "LocalModel expects a list of inputs: [local_inputs..., activations]."
            )

        local_input_shapes = list(input_shape[:-1])
        self.num_models = len(local_input_shapes)

        self.local_functions = [
            self._make_local_function(idx) for idx in range(self.num_models)
        ]

        for local_fn, shape in zip(self.local_functions, local_input_shapes):
            local_fn.build(shape)

        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
            raise ValueError(
                "LocalModel expects a list of inputs: [local_inputs..., activations]."
            )

        local_inputs = list(inputs[:-1])
        activations = inputs[-1]

        if len(local_inputs) != len(self.local_functions):
            raise ValueError(
                f"LocalModel received {len(local_inputs)} local inputs but was built with {len(self.local_functions)}."
            )

        local_outputs = [
            local_fn(local_input)
            for local_fn, local_input in zip(self.local_functions, local_inputs)
        ]

        stacked = keras.ops.stack(local_outputs, axis=2)

        if activations.shape[-1] is not None and activations.shape[-1] != len(
            local_inputs
        ):
            raise ValueError(
                f"Activation width ({activations.shape[-1]}) must match number of local inputs ({len(local_inputs)})."
            )

        weights = keras.ops.cast(activations, stacked.dtype)
        weights = keras.ops.expand_dims(weights, axis=-1)
        weighted = stacked * weights
        return keras.ops.sum(weighted, axis=2)


class LocalModel(Layer):
    """Apply one function per input and aggregate outputs with fuzzy activations."""

    # node_type = "LocalModel"

    def __init__(
        self,
        function="Fir",
        out_features=1,
        use_bias=True,
        name=None,
    ):
        self.function = str(function)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)

        super().__init__(
            name=name,
            function=self.function,
            out_features=self.out_features,
            use_bias=self.use_bias,
        )

    def __call__(self, *inputs):
        if not inputs:
            raise TypeError("LocalModel expects at least one input")

        first = inputs[0]

        if isinstance(first, dict):
            if len(inputs) != 2:
                raise ValueError(
                    "LocalModel with dict inputs expects exactly two arguments: dict inputs and activations."
                )
            ordered_inputs = [first[key] for key in first]
            return super().__call__(*ordered_inputs, inputs[1])

        if len(inputs) == 2 and isinstance(first, (list, tuple)):
            return super().__call__(*list(first), inputs[1])

        return super().__call__(*inputs)

    def output_shape(self, seqs, times, dims):
        if len(seqs) < 2:
            raise ValueError(
                "LocalModel expects at least one local input plus one activation input."
            )

        local_seqs = seqs[:-1]
        local_times = times[:-1]
        local_dims = dims[:-1]
        activation_dim = dims[-1]

        if len(local_seqs[0]) != 0:
            raise NotImplementedError(
                "LocalModel currently supports local inputs with seq=() only."
            )

        for dim in local_dims:
            if len(dim) != 1:
                raise ValueError(
                    f"LocalModel currently expects 1D feature inputs, got dim={dim}."
                )

        if len(activation_dim) != 1:
            raise ValueError(
                f"LocalModel activation input must have one feature axis, got dim={activation_dim}."
            )

        if activation_dim[0] != len(local_dims):
            raise ValueError(
                f"LocalModel activation width ({activation_dim[0]}) must match number of local inputs ({len(local_dims)})."
            )

        ref_seq = local_seqs[0]
        ref_time = local_times[0]
        for seq, time in zip(local_seqs[1:], local_times[1:]):
            if seq != ref_seq or time != ref_time:
                raise ValueError(
                    "All LocalModel local inputs must share the same shape."
                )

        return ref_seq, 1, (self.out_features,)

    def build_layer(self):
        num_models = max(0, len(self.predecessors) - 1)
        self._layer = LocalModelImpl(
            function_name=self.function,
            out_features=self.out_features,
            use_bias=self.use_bias,
            num_models=num_models,
            name=self.name,
        )
        return self._layer
