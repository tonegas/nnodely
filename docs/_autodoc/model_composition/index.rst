.. _nnodely-model-composition:

Model Composition
==================================

Beyond individual models, **nnodely** supports composition of multiple neural
models and fine-grained wiring of signals. Composition can be performed at two
distinct levels:

- **Stream-level composition** : local operations on signal streams (time windows,
  delays, closed-loop connections). Implemented by the :class:`~nnodely.basic.relation.Stream`
  API (methods such as :meth:`~nnodely.basic.relation.Stream.connect`,
  :meth:`~nnodely.basic.relation.Stream.closedLoop`, :meth:`~nnodely.basic.relation.Stream.delay`,
  :meth:`~nnodely.basic.relation.Stream.s`, :meth:`~nnodely.basic.relation.Stream.sw`,
  :meth:`~nnodely.basic.relation.Stream.tw`, :meth:`~nnodely.basic.relation.Stream.z`).

- **Model-level composition** : composing and connecting independently defined
  models. Provided by the :class:`~nnodely.operators.composer.Composer`
  interface (methods such as :meth:`~nnodely.operators.composer.Composer.addModel`,
  :meth:`~nnodely.operators.composer.Composer.removeModel`,
  :meth:`~nnodely.operators.composer.Composer.addConnect`,
  :meth:`~nnodely.operators.composer.Composer.addClosedLoop`,
  :meth:`~nnodely.operators.composer.Composer.removeConnection`,
  and :meth:`~nnodely.operators.composer.Composer.neuralizeModel`).

- **Dynamically** : connection are created dynamically at execution time, during training, analysis, or inference.

.. Stream-level example
.. ^^^^^^^^^^^^^^^^^^^^

.. .. code-block:: python

..     from nnodely import Modely, Input, Output
..     from nnodely.modules import Fir

..     model = Modely()
..     x = Input('x')

..     fir_layer = Fir(b=True)
..     y = Output('out', fir_layer(x.tw(0.05)))

.. Model-level example
.. ^^^^^^^^^^^^^^^^^^^^

.. .. code-block:: python

..     from nnodely import Modely, Input, Output
..     from nnodely.modules import Fir

..     model = Modely()
..     x = Input('x'); y = Input('y')

..     # model A
..     outA = Output('A_out', Fir(x.last()))
..     model.addModel('subA', [outA])

..     # model B
..     outB = Output('B_out', Fir(y.last()))
..     model.addModel('subB', [outB])

..     # connect A_out to input y of subB (global composer connect)
..     model.addConnect(outA, y)

..     # close loop from B_out to input x
..     model.addClosedLoop(outB, x)

..     # finalize neural model (prepare for training/inference)
..     model.neuralizeModel(sample_time=0.1)

.. toctree::
   :maxdepth: 1

   relation_module
   composer_module
   modely_execution_model