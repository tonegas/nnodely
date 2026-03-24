"""
Fir - Finite Impulse Response layer. Comprime T->1.
"""

import os

import keras
from keras import ops
from nnodely.dag import to_tuple
from nnodely.layer import Layer
import tensorflow as tf

class Fir(Layer):
    def __init__(self, out_features=1, init_negexp=False, **kwargs):
        super().__init__(**kwargs)
        self._out_features = int(out_features)
        self.init_negexp = init_negexp

        if self._out_features < 1:
            raise ValueError("out_features must be >= 1")

    def build_layer(self):
        if self.dim:
            in_dim = to_tuple(self.dim[0], (1,))
            self._dim_rank = len(in_dim)

        if self.init_negexp:
            t = tf.cast(tf.range(self.time[0]), tf.float32)
            w = tf.exp(-3.0 * t)
            w = 0.1 * w / tf.reduce_sum(w)
            w = tf.reverse(w, axis=[0])
            w = tf.reshape(w, (self.time[0], 1))
            w = tf.tile(w, [1, self._out_features])

            initializer = tf.keras.initializers.Constant(w)
        else:
            initializer = tf.keras.initializers.RandomUniform(0.0, 1.0)

        self.kernel = keras.Variable(
            name="fir_weight",
            shape=(self.time[0], self._out_features),
            trainable=True,
            initializer=initializer,
        )

    def call(self, x):
        # x expected as (B, S1, S2, ..., T, D1, D2, ...)
        # Move time to the last axis so matmul contracts only along T.
        len_x = len(x.shape)
        perm = list(range(len_x))
        time_pos = perm.pop(len_x - self._dim_rank - 1)  # time axis is before dim axes
        perm.append(time_pos)
        x_t_last = ops.transpose(x, perm)
        #print(f"len_x: {len_x}, perm: {perm}, time_pos: {time_pos}, x_t_last shape: {getattr(x_t_last, 'shape', None)}.")
        # (..., T) @ (T, out_features) -> (..., out_features)
        y = ops.matmul(x_t_last, self.kernel)

        # Insert time axis back as length 1 (time compressed to one step).
        y = ops.expand_dims(y, axis=time_pos)

        if self._out_features == 1:
            y = ops.squeeze(y, axis=-1)
        if self.dim[0] == (1,):
            y = ops.squeeze(y, axis=-2)
        return y

    def output_shape(self, seq, time, dim):
        out_seq = seq[0] if seq else None
        out_time = 1  # FIR comprime T->1
        out_dim = (self._out_features,) if dim[0] == (1,) else dim[0]+(self._out_features,)
        return out_seq, out_time, out_dim
