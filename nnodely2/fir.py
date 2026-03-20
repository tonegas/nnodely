"""
Fir - Finite Impulse Response layer. Comprime T->1.
"""

import functools
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from nnodely.dag import to_tuple
from nnodely.layer import Layer

@keras.saving.register_keras_serializable(package="nnodely")
class _FirLayer(keras.Layer):
    """
    Layer Keras che implementa Fir senza sublayer (Sequential).
    Appare come singolo nodo in plot_model con expand_nested=True.
    """

    def __init__(self, T, inp_dim, out_features, use_bias=True,
                 kernel_initializer=None, bias_initializer=None, **kwargs):
        super().__init__(**kwargs)
        self._T = T
        self._inp_dim = tuple(inp_dim)
        self._out_features = out_features
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer or "glorot_uniform"
        self._bias_initializer = bias_initializer or "zeros"
        self._D_prod = functools.reduce(lambda a, b: a * b, self._inp_dim, 1)

    def build(self, input_shape):
        # kernel: (T, out_features) - applicato sull'ultima dim dell'input
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self._T, self._out_features),
            initializer=self._kernel_initializer,
            trainable=True,
        )
        if self._use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self._out_features,),
                initializer=self._bias_initializer,
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x):
        T, inp_dim, out = self._T, self._inp_dim, self._out_features
        D_prod = self._D_prod

        if len(inp_dim) == 1 and inp_dim[0] == 1:
            # (B, T, 1) -> (B, T)
            x = keras.ops.reshape(x, (-1, T))
            # (B, T) @ (T, out) + bias
            x = keras.ops.matmul(x, self.kernel)
            if self._use_bias:
                x = x + self.bias
            # (B, out) -> (B, 1, out) o (B, 1, 1)
            return keras.ops.reshape(x, (-1, 1, out))
        else:
            # (B, T, *inp_dim) -> transpose -> (B, *inp_dim, T)
            perm = (0,) + tuple(range(2, 2 + len(inp_dim))) + (1,)
            x = keras.ops.transpose(x, perm)
            # (B, *inp_dim, T) -> (B, D_prod, T)
            x = keras.ops.reshape(x, (-1, D_prod, T))
            # (B, D_prod, T) @ (T, out) -> (B, D_prod, out)
            x = keras.ops.matmul(x, self.kernel)
            if self._use_bias:
                x = x + self.bias
            # (B, D_prod, out) -> (B, 1, *inp_dim) o (B, 1, *inp_dim, out)
            out_shape = (-1, 1) + inp_dim + ((out,) if out > 1 else ())
            return keras.ops.reshape(x, out_shape)


class Fir(Layer):
    """FIR layer. Comprime time T->1."""

    def __init__(self, out_features: int = 1, *, use_bias: bool = True,
                 kernel_initializer=None, bias_initializer=None, name: str = None):
        super().__init__(name=name, use_bias=use_bias)
        self._out_features = out_features
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    def output_shape(self, seq, time, dim):
        """Comprime T->1. out_dim = (out_features,) se dim=(1,), altrimenti dim + (out_features,)."""
        dim = to_tuple(dim[0], (1,))
        seq = seq[0]
        if self._out_features == 1:
            return (seq, 1, dim)
        if dim == (1,):
            return (seq, 1, (self._out_features,))
        return (seq, 1, dim + (self._out_features,))

    def _get_initializer(self, which):
        """Converte Parameter in keras.initializers.Constant se necessario."""
        init = self._kernel_initializer if which == 'kernel' else self._bias_initializer
        if init is None:
            return None
        from nnodely.parameter import _is_parameter
        if _is_parameter(init):
            return keras.initializers.Constant(init.initial_value)
        return init

    def build_layer(self):
        """Compress T->1. Usa self.time e self.dim (impostati dal chiamante)."""
        T = self.time
        inp_dim = to_tuple(self.dim, (1,))
        ki = self._get_initializer('kernel')
        bi = self._get_initializer('bias')
        self._layer = _FirLayer(
            T=T,
            inp_dim=inp_dim,
            out_features=self._out_features,
            use_bias=self.use_bias,
            kernel_initializer=ki,
            bias_initializer=bi,
            name=self.name,
        )
        return self._layer

    def call(self, x):
        return self._layer(x)
