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

Stream-level composition
------------------------------------------------

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

.. Stream-level example
.. ^^^^^^^^^^^^^^^^^^^^

.. .. code-block:: python

..     from nnodely import Modely, Input, Output
..     from nnodely.modules import Fir

..     model = Modely()
..     x = Input('x')

..     fir_layer = Fir(b=True)
..     y = Output('out', fir_layer(x.tw(0.05)))

Model-level composition
------------------------------------------------

Model-level composition deals with independently defined sub-models and their
interactions. Sub-models are logical groups of output streams that can be added,
removed, connected, or frozen independently. Use the Composer API to manage
these operations and to prepare the whole definition for neuralization.

Key Composer operators:

- **addModel / removeModel** : register and unregister named sub-models.
  See :meth:`~nnodely.operators.composer.Composer.addModel` and
  :meth:`~nnodely.operators.composer.Composer.removeModel`.

- **addConnect / addClosedLoop / removeConnection** : connect an output stream
  of one model to an input of another (or the same) model; create closed-loop
  feedback between streams and inputs. See
  :meth:`~nnodely.operators.composer.Composer.addConnect`,
  :meth:`~nnodely.operators.composer.Composer.addClosedLoop`, and
  :meth:`~nnodely.operators.composer.Composer.removeConnection`.

- **neuralizeModel** : finalize the model definition and translate it into a
  trainable neural representation (builds time windows, slices, and internal
  tensors required for training/inference). See
  :meth:`~nnodely.operators.composer.Composer.neuralizeModel`.

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

Composition at Training and Inference Time
------------------------------------------------
In addition to static composition through the Composer API, nnodely also allows model composition to be defined dynamically at execution time, during training, analysis, or inference.

In this case, connections and closed loops between signals and models can be specified directly as arguments of high-level execution methods such as train, trainAndAnalyze, and inference.

This approach is useful when:

- The same base models are reused in different configurations

- Feedback and interconnections must change across experiments

- Rapid prototyping and testing of architectures is required

.. code-block:: python

    msd.trainAndAnalyze(models='PID', closed_loop={'x':'x_n', 'x_m':'x_n'}, connect={'F':'F_PID'}, ...)

Contents
---------
.. toctree::
   :maxdepth: 2

   relation_module
   composer_module
   modely_execution_model