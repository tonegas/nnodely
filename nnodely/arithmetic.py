"""
Operazioni aritmetiche binarie: Add, Sub, Mul, Div.
Richiedono stream con stesse dimensioni (seq, time, dim).
"""

# import keras
# from nnodely.dag import next_name, to_tuple, get_seq_time_dim
# from nnodely.stream import Stream
# from nnodely.layer import LayerBase, _is_stream


# def _check_compatible(s1, s2, op_name: str):
#     """Verifica che due stream/input abbiano dimensioni compatibili."""
#     seq1, t1, d1 = get_seq_time_dim(s1)
#     seq2, t2, d2 = get_seq_time_dim(s2)
#     # Se uno è Input (no time), usa i valori dell'altro
#     if t1 is None and t2 is not None:
#         seq1, t1 = seq2, t2
#     elif t2 is None and t1 is not None:
#         seq2, t2 = seq1, t1
#     if seq1 != seq2:
#         raise ValueError(
#             f"{op_name}: seq incompatibili: {seq1} vs {seq2}"
#         )
#     if t1 != t2:
#         raise ValueError(
#             f"{op_name}: time incompatibili: {t1} vs {t2}"
#         )
#     if d1 != d2:
#         raise ValueError(
#             f"{op_name}: dim incompatibili: {d1} vs {d2}"
#         )


# class _BinaryOp(LayerBase):
#     """
#     Base per operazioni binarie. Crea Stream con predecessors=[s1, s2].
#     node_type='Stream' per compatibilità con Layer (Fir, ecc.).
#     """

#     node_type = 'Stream'
#     output_prefix = 'Arith'

#     def __init__(self, op_name: str, keras_op):
#         self.op_name = op_name
#         self._keras_op = keras_op
#         self._layer = None
#         self.name = next_name(op_name)
#         self.predecessors = []

#     def __call__(self, a, b):
#         if not _is_stream(a) or not _is_stream(b):
#             raise TypeError(
#                 f"{self.op_name} richiede due Stream/Input, non {type(a)}, {type(b)}"
#             )
#         _check_compatible(a, b, self.op_name)
#         # Usa seq, time, dim da chi li ha (Input non ha time, Stream sì)
#         sa, ta, da = get_seq_time_dim(a)
#         sb, tb, db = get_seq_time_dim(b)
#         seq = sa or sb
#         time = ta or tb
#         dim = da if da != (1,) else db
#         out_name = next_name(self.output_prefix)
#         out_stream = Stream(
#             out_name, self.node_type,
#             seq=tuple(seq), time=time, dim=dim,
#             predecessors=[a, b],
#             layer=self
#         )
#         return out_stream

#     def build_layer(self):
#         """Le op aritmetiche non hanno parametri, usano keras.layers."""
#         self._layer = self._keras_op()
#         return self._layer

#     def call(self, inputs):
#         """inputs: lista [x1, x2] di tensori."""
#         if self._layer is None:
#             self.build_layer()
#         return self._layer(inputs)


# class Add(_BinaryOp):
#     """Addizione element-wise. stream1 + stream2."""

#     def __init__(self):
#         super().__init__('Add', keras.layers.Add)


# class Subtract(_BinaryOp):
#     """Sottrazione element-wise. stream1 - stream2."""

#     def __init__(self):
#         super().__init__('Subtract', keras.layers.Subtract)


# class Multiply(_BinaryOp):
#     """Moltiplicazione element-wise. stream1 * stream2."""

#     def __init__(self):
#         super().__init__('Multiply', keras.layers.Multiply)


# class Divide(_BinaryOp):
#     """Divisione element-wise. stream1 / stream2."""

#     def __init__(self):
#         # Keras non ha layers.Divide, usiamo Lambda
#         super().__init__('Divide', lambda: keras.layers.Lambda(lambda xs: xs[0] / xs[1]))
