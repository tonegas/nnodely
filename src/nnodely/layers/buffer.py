"""Buffer: the last values of a signal computed one sample per step."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank
from nnodely.core.stream import Stream
from nnodely.layers.input import Input, _check_history, _declared_seq, _validate_init
from nnodely.layers.partitions import TimeSelect


@keras.saving.register_keras_serializable(package="nnodely")
class BufferImpl(keras.layers.Layer):
    """The past samples of a buffer followed by the current one, along the
    window; a current sample without a window gets one. The past is broadcast
    to the current sample's shape: an init may be a value, or samples of a
    scalar, and a loop starts a state from zero before the first step tells
    the length of a dynamic axis. With ``repeat``, it is an init of one
    sample, repeated that many times along the window."""

    def __init__(self, axis: int, expand: bool, repeat: int = 0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.expand = bool(expand)
        self.repeat = int(repeat)

    def call(self, inputs):
        *past, current = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        if self.expand:
            current = keras.ops.expand_dims(current, self.axis)
        if not past:
            return current
        history = expand_to_rank(past[0], len(current.shape))
        history = history + keras.ops.zeros_like(current)
        if self.repeat:
            history = keras.ops.repeat(history, self.repeat, axis=self.axis)
        return keras.ops.concatenate([history, current], axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "expand": self.expand, "repeat": self.repeat})
        return config


class Buffer(Layer):
    r"""The last ``tw`` values of a signal computed one sample per step, the
    current one included.

    Syntax
    ------
    ``Buffer(tw=n)(u)``
        The window of the last ``n`` values of ``u``, zero before the first
        step.
    ``Buffer(tw=n, init=c)(u)``
        The same, with the values before the first step given by ``c``.

    At step :math:`k` of a :class:`ModelyLoop` or a :class:`Loop` body,
    ``Buffer(tw=3)(u)`` holds :math:`[u_{k-2}, u_{k-1}, u_k]`. Its newest
    sample is ``u`` at this step, as if the reader were connected to ``u``.
    An Input fed back with ``u`` is different: its window ends at the
    previous step.

    Use it where a block reads a window of a signal that the step computes
    one sample at a time. For example, a plant reads the last three commands
    of a controller. The past ``tw - 1`` values are a state: an Input named
    ``<name>_<u>``, which the ModelyLoop or the Loop feeds back with its
    other states. Built as a plain :class:`Modely`, which has no steps, the
    layer reads ``init`` as the past values.

    Parameters
    ----------
    tw : int
        The number of values in the window, a positive integer. It is
        required. ``1`` holds the current value alone, with no past.
    init : Stream or float, optional
        The values before the first step. They are zero when left out. It can
        be:

        * a number, a :class:`Parameter` or a :class:`Constant`: one value,
          repeated;
        * an Input of the step, whose first step is read: one sample,
          repeated, or ``tw - 1`` samples, oldest first.

        Its features are a scalar or those of ``u``, and its sequence is none
        or that of ``u``. It is the only initial value of the state: a Loop
        cannot give it another. ``tw=1`` takes no ``init``.
    name : str, optional
        The name of the layer and of the window it returns. The state is
        named ``<name>_<u>``. The default name is ``Buffer`` followed by a
        number.

    Call arguments
    --------------
    u : Stream
        One sample per step, ``Shape(S, 1, D)``, or a stream without a
        window, ``Shape(S, None, D)``. The call takes it alone or in a list of
        one.

    Returns
    -------
    Stream
        ``Shape(S, tw, D)``: the window of the last ``tw`` values. For
        example, ``Buffer(tw=3)`` on ``Input("u", tw=1)`` gives
        ``Shape(S=(), T=3, D=(1,))``.

    Raises
    ------
    ValueError
        If ``tw`` is not a positive integer ("tw is a positive int"), if
        ``init`` is given with ``tw=1`` ("nothing for init to start"), if
        ``u`` has a window of more than one sample ("collects a signal
        computed one sample per step"), or if ``init`` has another number of
        samples, features or sequence ("init holds the samples before the
        first").
    TypeError
        If the layer is called before it is configured, ``Buffer(u)``, if
        ``u`` is not a stream, or if ``init`` is not a stream, a number or
        None.

    See Also
    --------
    ModelyLoop : A model declared one step at a time.
    Loop : The rollout of a one-step model.
    Derivative : A time derivative that keeps past samples the same way.
    Input.feedback : A state that reads the previous step.

    Notes
    -----
    At step :math:`k`, with :math:`n` = ``tw`` and the values
    :math:`u_{-n+1}, \dots, u_{-1}` from ``init``:

    .. math:: y_k = [u_{k-n+1}, \dots, u_{k-1}, u_k]

    Basic Examples
    --------------
    The last three values of a signal, step by step:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Buffer, Fir, Input, Modely, ModelyLoop, Output
    >>> u = Input("u", tw=1)
    >>> window = Buffer(tw=3, name="last3")(u)
    >>> model = ModelyLoop("history", inputs=[u], outputs=[Output("w", window)]).build()
    >>> steps = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    >>> keras.ops.convert_to_numpy(model({"u": steps})["w"])[0, :, :, 0]
    array([[0., 0., 1.],
           [0., 1., 2.],
           [1., 2., 3.],
           [2., 3., 4.]], dtype=float32)

    The state is an Input of the model, named after the layer and ``u``:

    >>> model.inputs[1].name
    'last3_u'

    Scope
    -----
    A signal without a window is collected in the same way, and each
    feature has a window of its own:

    >>> Buffer(tw=3)(Input("level")).shape
    Shape(S=(), T=3, D=(1,))
    >>> Buffer(tw=4)(Input("v", dim=2, tw=1)).shape
    Shape(S=(), T=4, D=(2,))

    Options
    -------
    ``init`` is a number, repeated, or an Input of ``tw - 1`` samples, oldest
    first, of which the first step is read:

    >>> repeated = Buffer(tw=3, init=0.5)(u)
    >>> past = Input("past", tw=2)
    >>> given = Buffer(tw=3, init=past)(u)
    >>> model = ModelyLoop(
    ...     "starts", inputs=[u, past],
    ...     outputs=[Output("repeated", repeated), Output("given", given)],
    ... ).build()
    >>> start = np.tile(np.array([-2.0, -1.0]).reshape(1, 1, 2, 1), (1, 4, 1, 1))
    >>> result = model({"u": steps, "past": start})
    >>> keras.ops.convert_to_numpy(result["repeated"])[0, :2, :, 0]
    array([[0.5, 0.5, 1. ],
           [0.5, 1. , 2. ]], dtype=float32)
    >>> keras.ops.convert_to_numpy(result["given"])[0, :2, :, 0]
    array([[-2., -1.,  1.],
           [-1.,  1.,  2.]], dtype=float32)

    ``tw=1`` holds the current value alone:

    >>> Buffer(tw=1)(u).shape
    Shape(S=(), T=1, D=(1,))

    Applications
    ------------
    A proportional controller drives a tank whose level reads the last
    three commands:

    >>> forces = Input("forces", tw=3)
    >>> level = Input("level", tw=1)
    >>> level_next = level.feedback(0.9 * level + Fir(bias=False)(forces), hidden=True)
    >>> tank = Modely("tank", inputs=[forces], outputs=[Output("level_next", level_next)])
    >>> reference, measured = Input("reference", tw=1), Input("measured", tw=1)
    >>> command = Output("command", 0.5 * (reference - measured))
    >>> measured_next = measured.feedback(tank([Buffer(tw=3)(command)]), hidden=True)
    >>> control = ModelyLoop(
    ...     "control", inputs=[reference], outputs=[command, Output("level", measured)]
    ... )

    Properties & Relations
    ----------------------
    Built as a plain :class:`Modely`, the window reads ``init`` as the past
    values:

    >>> model = Modely("one_step", inputs=[u], outputs=[Output("w", repeated)]).build()
    >>> keras.ops.convert_to_numpy(model({"u": np.array([[[7.0]]])})["w"]).ravel()
    array([0.5, 0.5, 7. ], dtype=float32)

    Possible Issues
    ---------------
    The signal is computed one sample per step. A stream that already has a
    window needs no buffer:

    >>> Buffer(tw=3)(Input("recorded", tw=3))
    Traceback (most recent call last):
        ...
    ValueError: ...a Buffer collects a signal computed one sample per step, and 'recorded' carries a window of 3.

    The layer is configured first, then called on the signal:

    >>> Buffer(u)
    Traceback (most recent call last):
        ...
    TypeError: Buffer is configured first and called on the signal, like every other layer: Buffer(tw=3)(u).

    .. versionadded:: 2.0
    """

    def __init__(self, tw: int, init: Stream | float | None = None, name=None):
        if isinstance(tw, Stream):
            raise TypeError(
                "Buffer is configured first and called on the signal, like every "
                "other layer: Buffer(tw=3)(u)."
            )
        if isinstance(tw, bool) or not isinstance(tw, int) or tw < 1:
            raise ValueError(f"Buffer: tw is a positive int, got {tw!r}.")
        if tw == 1 and init is not None:
            raise ValueError(
                "Buffer: tw=1 holds the current sample alone, no past one, so "
                f"there is nothing for init to start; got init={init!r}."
            )
        self._tw = tw
        self._init = _validate_init("Buffer", init)
        self._calls = 0
        super().__init__(name=name)
        # Saved with the layer's arguments. Passed as tw it would be the
        # window of the layer's own shape, which the signal it reads sets.
        self._properties["tw"] = tw

    def __call__(self, stream):  # type: ignore[override]
        if isinstance(stream, (list, tuple)) and len(stream) == 1:
            stream = stream[0]
        if not isinstance(stream, Stream):
            raise TypeError(
                f"{self.name}: Buffer is called on one Stream, got "
                f"{type(stream).__name__}."
            )
        if stream.tw not in (None, 1):
            raise ValueError(
                f"{self.name}: a Buffer collects a signal computed one sample per "
                f"step, and {stream.name!r} carries a window of {stream.tw}."
            )
        if self._tw == 1:
            return super().__call__([stream])
        if isinstance(self._init, Stream):
            _check_history(self.name, self._init, stream, self._tw - 1)

        # Every call buffers a signal of its own, so each has its own past.
        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        # Named after the signal whose past it holds.
        past = Input(
            f"{call_name}_{stream.name}",
            dim=stream.dim,
            seq=_declared_seq(stream),
            tw=self._tw - 1,
        )
        past._hidden = True
        window = super().__call__([past, stream])
        past._feed_back(
            stream if stream.tw == 1 else TimeSelect(-1)(window), init=self._init
        )
        return window

    def output_shape(self, *inputs):
        return inputs[-1].shape.replace(tw=self._tw)

    def build_layer(self):
        repeat = 0
        if len(self._preds) > 1:
            past = self._preds[0].shape  # type: ignore[union-attr]
            # Built as a Modely, the past is read as the init: unless it has
            # the shape of the Input it stands for, it is broadcast to it.
            if past.tuple != self.shape.replace(tw=self._tw - 1).tuple:
                repeat = 1 if past.tw == self._tw - 1 else self._tw - 1
        return BufferImpl(
            axis=self.shape.tw_axis,  # type: ignore[arg-type]
            expand=self._preds[-1].shape.tw is None,  # type: ignore[union-attr]
            repeat=repeat,
            name=self.name,
        )

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The past values were saved as the Input this node reads first, so
        # the node is rebuilt on its predecessors, without an Input of its own.
        layer = super().from_config(config)
        return Layer.__call__(layer, list(preds)) if preds else layer
