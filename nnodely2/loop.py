"""
Loop: Layer to incapsulate a model in a loop.
"""
import keras
from nnodely.layer import Layer

from keras import ops

@keras.saving.register_keras_serializable(package="nnodely")
class _LoopLayer(keras.layers.Layer):
    def __init__(self, cell_model, iterations:int|str="auto", **kwargs):
        super().__init__(**kwargs)
        self.cell_model = cell_model
        self.iterations = iterations

    def call(self, inputs):

        # inputs = (batch, seq1, seq2, ..., const1, const2...)
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # dummy carry iniziale (non usato, ma necessario per scan)
        carry0 = ops.zeros((ops.shape(inputs[0])[0], 1))  # (B, carry_dim=1)
        sequences = []
        constants = []

        # detect sequence inputs (rank >= 4 assumed: B,T,...)
        for inp in inputs:
            if ops.ndim(inp) >= 4:
                sequences.append(inp)
            else:
                constants.append(inp)

        if sequences:
            if self.iterations == "auto":
                N = ops.shape(sequences[0])[1]
            else:
                N = self.iterations
        else:
            if self.iterations == "auto":
                raise ValueError("iterations must be specified if no sequences are provided")
            N = self.iterations

        def step(carry, i):

            step_inputs = []

            # recurrent state
            #step_inputs.append(carry)

            # timestep slices
            for seq in sequences:
                step_inputs.append(ops.take(seq, i, axis=1))

            # constants
            step_inputs.extend(constants)

            out = self.cell_model(step_inputs)

            # allow multi-output
            if isinstance(out, (list, tuple)):
                new_carry = out[0]
                y = out[1]
            else:
                new_carry = out
                y = out

            return new_carry, y

        _, outputs = ops.scan(
            step,
            carry0,
            ops.arange(N),
        )

        # transpose from (T,B,...) → (B,T,...)
        outputs = ops.transpose(
            outputs,
            axes=(1, 0) + tuple(range(2, ops.ndim(outputs)))
        )

        return outputs

class Loop(Layer):
    """Loop."""

    def __init__(self, cell_model, *, iterations:int|str='auto', name:str|None=None, **kwargs):
        super().__init__(name=name)
        self._cell_model = cell_model
        self._iterations = iterations

    def build_layer(self, **kwargs):
        """Costruisce il layer Keras Loop con il modello cell_model e numero di iterazioni."""
        
        self._layer = _LoopLayer(
            cell_model=self._cell_model,
            iterations=self._iterations,
            name=self.name,
        )
        return self._layer

    def call(self, x):
        return self._layer(x)
