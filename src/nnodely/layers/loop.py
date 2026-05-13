import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely

class LoopImpl(keras.layers.Layer):
    def __init__(self, submodel, closed_loop: dict[str, str], name=None, inputs=None, longest_seq=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.submodel = submodel
        self.closed_loop = closed_loop
        self.inputs = inputs
        self.longest_seq = longest_seq

        # Validate input sequence dimensions to be the same for all inputs, except the zero default (non-sequence) value.
        seq_dims = set()
        for inp in self.inputs:
            print(f"LoopImpl init - input '{inp.name}' seq: {inp.seq}")
            if inp.seq != ():
                seq_dims.add(inp.seq[0]) # leftmost seq dimension
        if len(seq_dims) > 1:
            raise ValueError(f"LoopImpl: all inputs must have the same seq dimensions or not have a sequence, got {seq_dims}")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "submodel": keras.saving.serialize_keras_object(self.submodel),
                "closed_loop": self.closed_loop,
                "inputs": self.inputs,
                "longest_seq": self.longest_seq,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["submodel"] = keras.saving.deserialize_keras_object(config["submodel"])
        return cls(**config)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        #determine horizon from all looped inputs (leftmost seq axis)
        horizon = max(x.shape[-1] if x.shape[-1] is not None else 1 for x in inputs) if inputs else 1
        
        # for loop
        step_inputs = {}
        for idx, inp in enumerate(self.submodel.inputs):
            step_inputs[inp.name] = inputs[idx][..., 0]

        outputs = {}
        for t in range(horizon):
            for idx, inp in enumerate(self.inputs):
                if t > 0 and inp.name in self.closed_loop.keys():
                    step_inputs[self.submodel.inputs[idx].name] = y[self.closed_loop[inp.name]] if isinstance(y, dict) else y
                else:
                    step_inputs[self.submodel.inputs[idx].name] = inputs[idx][..., t] if inputs[idx].shape[-1] is not None and inputs[idx].shape[-1] > t else inputs[idx][..., -1]

            y = self.submodel(step_inputs)
            # print(f"Loop step {t+1}/{horizon} - inputs: {[f'{inp.name}:{step_inputs[inp.name]}' for inp in self.submodel.inputs]} - outputs: {next(iter(y.items()))[1]}")

            for out_name, out_value in (y.items() if isinstance(y, dict) else {"output": y}.items()):
                out_value = out_value if len(self.longest_seq) < 1 else keras.ops.expand_dims(out_value, axis=-1)
                if out_name not in outputs:
                    outputs[out_name] = out_value
                else:
                    outputs[out_name] = keras.ops.concatenate([outputs[out_name], out_value], axis=-1)

        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs

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
        # output follows the first input leftmost seq (looped axis)
        if len(inputs) > 0 and len(inputs[0].seq) > 0:
            horizon = inputs[0].seq[0]
        else:
            horizon = 1
        out_node = self.f.outputs[0]
        out_seq = (horizon,) + tuple(out_node.seq)
        return out_seq, out_node.time, out_node.dim

    def build_layer(self):
        self.longest_seq = max(inp.seq for inp in self.inputs)

        if self.f._model is None:
            self.f.build()

        fn_output_names = [node.name for node in self.f.outputs]
        if any(out not in fn_output_names for out in self.closed_loop.values()):
            raise ValueError(
                f"{self.name}: closed-loop outputs: '{[out for out in self.closed_loop.values() if out not in fn_output_names]}' not in f.outputs={fn_output_names}"
            )
        return LoopImpl(
            self.f._model, self.closed_loop, name=self.name, inputs=self.inputs, longest_seq=self.longest_seq
        )
