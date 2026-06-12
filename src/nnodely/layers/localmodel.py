from __future__ import annotations


from nnodely.layers.time_ops import Select
from nnodely.core.layer import Add
from nnodely.core.modely import Modely
from nnodely.layers.input import Input
from nnodely.layers.output import Output


# @keras.saving.register_keras_serializable(package="nnodely")
# class LocalModelImpl(keras.layers.Layer):
#     def __init__(
#         self,
#         function_name="Fir",
#         out_features=1,
#         use_bias=True,
#         num_models=None,
#         name=None,
#         **kwargs,
#     ):
#         super().__init__(name=name, **kwargs)
#         self.function_name = str(function_name)
#         self.out_features = int(out_features)
#         self.use_bias = bool(use_bias)
#         self.num_models = int(num_models) if num_models is not None else None
#         self.local_functions = []

#         if self.function_name.lower() != "fir":
#             raise ValueError(
#                 f"Unsupported LocalModel function {self.function_name!r}. "
#                 "Currently supported: 'Fir'."
#             )

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "function_name": self.function_name,
#                 "out_features": self.out_features,
#                 "use_bias": self.use_bias,
#                 "num_models": self.num_models,
#             }
#         )
#         return config

#     def _make_local_function(self, idx):
#         if self.function_name.lower() == "fir":
#             return FirImpl(
#                 out_features=self.out_features,
#                 use_bias=self.use_bias,
#                 name=f"{self.name}_local_{idx}",
#             )
#         raise ValueError(
#             f"Unsupported LocalModel function {self.function_name!r}. "
#             "Currently supported: 'Fir'."
#         )

#     def build(self, input_shape):
#         if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
#             raise ValueError(
#                 "LocalModel expects a list of inputs: [local_inputs..., activations]."
#             )

#         local_input_shapes = list(input_shape[:-1])
#         self.num_models = len(local_input_shapes)

#         self.local_functions = [
#             self._make_local_function(idx) for idx in range(self.num_models)
#         ]

#         for local_fn, shape in zip(self.local_functions, local_input_shapes):
#             local_fn.build(shape)

#         super().build(input_shape)

#     def call(self, inputs):
#         if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
#             raise ValueError(
#                 "LocalModel expects a list of inputs: [local_inputs..., activations]."
#             )

#         local_inputs = list(inputs[:-1])
#         activations = inputs[-1]

#         if len(local_inputs) != len(self.local_functions):
#             raise ValueError(
#                 f"LocalModel received {len(local_inputs)} local inputs but was built with {len(self.local_functions)}."
#             )

#         local_outputs = [
#             local_fn(local_input)
#             for local_fn, local_input in zip(self.local_functions, local_inputs)
#         ]

#         stacked = keras.ops.stack(local_outputs, axis=2)

#         if activations.shape[-1] is not None and activations.shape[-1] != len(
#             local_inputs
#         ):
#             raise ValueError(
#                 f"Activation width ({activations.shape[-1]}) must match number of local inputs ({len(local_inputs)})."
#             )

#         weights = keras.ops.cast(activations, stacked.dtype)
#         weights = keras.ops.expand_dims(weights, axis=-1)
#         weighted = stacked * weights
#         return keras.ops.sum(weighted, axis=2)


# class LocalModel(Layer):
#     """Apply one function per input and aggregate outputs with fuzzy activations."""

#     def __init__(
#         self,
#         function="Fir",
#         out_features=1,
#         use_bias=True,
#         name=None,
#     ):
#         self.function = str(function)
#         self.out_features = int(out_features)
#         self.use_bias = bool(use_bias)

#         super().__init__(
#             name=name,
#             function=self.function,
#             out_features=self.out_features,
#             use_bias=self.use_bias,
#         )

#     def __call__(self, *inputs):
#         if not inputs:
#             raise TypeError("LocalModel expects at least one input")

#         # first = inputs[0]

#         # if isinstance(first, dict):
#         #     if len(inputs) != 2:
#         #         raise ValueError(
#         #             "LocalModel with dict inputs expects exactly two arguments: dict inputs and activations."
#         #         )
#         #     ordered_inputs = [first[key] for key in first]
#         #     return super().__call__(*ordered_inputs, inputs[1])

#         # if len(inputs) == 2 and isinstance(first, (list, tuple)):
#         #     return super().__call__(*list(first), inputs[1])

#         return super().__call__(*inputs)

#     def output_shape(self, *inputs):
#         return inputs[0].seq, inputs[0].time, (self.out_features,)

#     def build_layer(self):
#         num_models = max(0, len(self.preds) - 1)
#         return LocalModelImpl(
#             function_name=self.function,
#             out_features=self.out_features,
#             use_bias=self.use_bias,
#             num_models=num_models,
#             name=self.name,
#         )


class LocalModel:
    """
    High-level abstraction for a local model built using only nnodely blocks
    """

    def __init__(
        self,
        input_function,
        output_function=None,
        name: str | None = None,
    ):
        self.input_function = input_function
        self.output_function = output_function
        self.name = name

    def __call__(self, activation):
        ret = []
        local = Input("local_input")
        for i in range(activation.dim[0]):
            x = self.input_function([local]) * Select(idx=i, axis=0)([activation])
            if self.output_function is not None:
                x = self.output_function([x])
            ret.append(x)
        ret = Add()(ret)
        out = Output("local_output", ret)
        return Modely(name=f"{self.name}", inputs=[local], outputs=[out])
