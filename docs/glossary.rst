Glossary
========

One concept has one name, in the guide, in the reference pages and in the
error messages. Where a method name uses another word, as ``minimize`` and
``minimizers`` do for objectives, the entry says so.

.. glossary::
   :sorted:

   stream
      A node of the graph of a model: an :class:`~nnodely.Input`, a
      :class:`~nnodely.Parameter`, a :class:`~nnodely.Constant`, or the result
      of a layer or of an operator. A stream holds no values while the graph
      is declared, only a :term:`shape`.

   shape
      The axes of a stream after the batch axis, written
      ``Shape(S, T, D)``: the :term:`sequence` axes ``S``, the :term:`window`
      ``T`` and the :term:`features` ``D``. A tensor of the stream is laid
      out as ``(batch, *S, T, *D)``, without ``T`` when there is no window.

   window
      The consecutive samples a stream carries, ``T``, declared with ``tw``.
      ``tw=3`` is the last three samples up to the current one; ``tw=(3, 1)``
      adds the next one. A stream without a window holds one sample and no
      ``T`` axis. Windows are counted in samples, not in seconds.

   sequence
      The axes before the window, ``S``, declared with ``seq``, outermost
      first: ``seq=5`` is five consecutive windows. A :class:`~nnodely.Loop`
      rolls out along the outermost one. A *dynamic* sequence, ``seq=-1``,
      takes the length of the data.

   features
      The axes of one sample, ``D``, declared with ``dim``: ``dim=3`` is a
      vector of three values, ``dim=(3, 2)`` a 3x2 matrix. The default is one
      feature, ``D=(1,)``.

   sample
      One row of the data a model reads: for every input, the window, or the
      sequence of windows, it declares. The batch axis stacks samples.

   simulation
      One contiguous recording of the signals in a
      :class:`~nnodely.DataLoader`. A window never spans two simulations.

   objective
      A loss the training reduces, registered with
      :meth:`~nnodely.Modely.minimize` between a *source* stream and a
      *target*. Its name labels its loss in the training history. The method
      names call it a *minimizer*: :attr:`~nnodely.Modely.minimizers`,
      :meth:`~nnodely.Modely.remove_minimizer`.

   target
      What an objective compares its source with: an Input read from the
      data, a number, or nothing, which drives the source to zero.

   block
      A :class:`~nnodely.Modely` called on streams inside another model. It
      brings its weights, and its outputs become streams of the larger graph.

   build
      :meth:`~nnodely.Modely.build` turns the graph into a Keras model and
      creates its weights. Objectives and feedback are declared before it.

   rollout
      The repeated execution of a one-step model along the outermost sequence
      axis, each step reading the results of the one before. Its length is
      the *horizon*.

   state
      An Input fed back with a value the model computes, so that the next
      step reads it. Also called a *fed-back Input*. Declared with
      :meth:`Input.feedback() <nnodely.Input.feedback>`, or with the
      ``feedback`` of a :class:`~nnodely.Loop`.

   body
      The one-step model a :class:`~nnodely.Loop` or a
      :class:`~nnodely.ModelyLoop` rolls out.

   step
      One execution of a body in a rollout, ``k``. In a
      :class:`~nnodely.ModelyLoop`, a state holds its value at the end of step
      ``k - 1``, and the stream it is fed back with its value at the end of
      step ``k``.

   mask
      Marks the real steps of rollouts padded to a common length, when
      simulations of different lengths are loaded whole. Training and
      validation leave the padded steps out.

   channel
      One feature of a compared signal. Validation scores every channel on
      its own scale.
