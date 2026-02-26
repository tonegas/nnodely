.. _relation_module:

Stream-level composition
================================

Streams represent signals inside the model graph and are created automatically
when you operate on an :obj:`Input`, :obj:`Parameter`, or :obj:`Constant`.
Use stream methods to manipulate signal timing and routing before they are
bound to outputs or composed into sub-models.

Common Stream operators:

- **Closed-loop and connect** : create feedback or feedforward wiring from a
  signal to an :class:`Input` using :meth:`~nnodely.basic.relation.Stream.closedLoop`
  and :meth:`~nnodely.basic.relation.Stream.connect`. These return a new
  :class:`~nnodely.basic.relation.Stream` representing the connected signal.

- **Delays / z / time windows** : postpone or window a stream with
  :meth:`~nnodely.basic.relation.Stream.delay`, :meth:`~nnodely.basic.relation.Stream.z`,
  :meth:`~nnodely.basic.relation.Stream.tw` and :meth:`~nnodely.basic.relation.Stream.sw`.

- **Sampling windows** : select past samples or time intervals with
  :meth:`~nnodely.basic.relation.Stream.sw` (sample window) and
  :meth:`~nnodely.basic.relation.Stream.tw` (time window). These are the
  primitives used by temporal building blocks (FIR, recurrent windows, etc.).
  
.. automodule:: nnodely.basic.relation
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.basic.relation.Stream
        :members:
        :no-inherited-members:
