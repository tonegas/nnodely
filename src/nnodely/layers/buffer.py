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
    """The last ``sw`` values of a signal computed one sample per step, the
    current one included::

        u_window = Buffer(sw=3)(u)            # [u[k-2], u[k-1], u[k]]
        u_window = Buffer(sw=3, init=0.5)(u)

    Where a model reads a window of a signal that another computes one sample
    per step - a controller's command fed to a plant that reads its last three
    - the buffer holds the past ones. The newest sample is the signal at this
    step, which reaches the model within the step, as if connected to it; the
    window shifts after the step, when the newest becomes a past one. An
    Input fed back with the signal is the other way round: its window ends at
    the previous step.

    The past ``sw - 1`` values are a hidden Input named after the block and
    the signal, ``<name>_<signal>`` (``Buffer3_x`` for ``Buffer(sw=3)(x)``),
    fed back with the signal, by the :class:`~nnodely.ModelyLoop` or
    :class:`~nnodely.Loop` like any other. ``init`` holds them before the
    first step, as the ``init`` of :class:`~nnodely.Derivative` holds the
    samples its stencil reads, and is their only initial value: a number, a
    Parameter or Constant, or an Input of the step - one sample, repeated, or
    ``sw - 1`` of them, oldest first, its dim a scalar or the signal's and
    its sequence none or the signal's; ``None`` for zeros. Built as a
    :class:`~nnodely.Modely` instead, nothing feeds them back: the step reads
    ``init`` as them, as the first step of a loop does, and no Input of its
    own. The signal is one sample, ``sw=1``, or has no window; ``sw=1``
    holds no past, and takes no ``init``.
    """

    def __init__(self, sw: int, init: Stream | float | None = None, name=None):
        if isinstance(sw, Stream):
            raise TypeError(
                "Buffer is configured first and called on the signal, like every "
                "other layer: Buffer(sw=3)(u)."
            )
        if isinstance(sw, bool) or not isinstance(sw, int) or sw < 1:
            raise ValueError(f"Buffer: sw is a positive int, got {sw!r}.")
        if sw == 1 and init is not None:
            raise ValueError(
                "Buffer: sw=1 holds the current sample alone, no past one, so "
                f"there is nothing for init to start; got init={init!r}."
            )
        self._sw = sw
        self._init = _validate_init("Buffer", init)
        self._calls = 0
        super().__init__(name=name, sw=sw)

    def __call__(self, stream):  # type: ignore[override]
        if isinstance(stream, (list, tuple)) and len(stream) == 1:
            stream = stream[0]
        if not isinstance(stream, Stream):
            raise TypeError(
                f"{self.name}: Buffer is called on one Stream, got "
                f"{type(stream).__name__}."
            )
        if stream.time not in (None, 1):
            raise ValueError(
                f"{self.name}: a Buffer collects a signal computed one sample per "
                f"step, and {stream.name!r} carries a window of {stream.time}."
            )
        if self._sw == 1:
            return super().__call__([stream])
        if isinstance(self._init, Stream):
            _check_history(self.name, self._init, stream, self._sw - 1)

        # Every call buffers a signal of its own, so each has its own past.
        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        # Named after the signal whose past it holds.
        past = Input(
            f"{call_name}_{stream.name}",
            dim=stream.dim,
            seq=_declared_seq(stream),
            sw=self._sw - 1,
        )
        past._hidden = True
        window = super().__call__([past, stream])
        past._feed_back(
            stream if stream.time == 1 else TimeSelect(-1)(window), init=self._init
        )
        return window

    def output_shape(self, *inputs):
        return inputs[-1].shape.replace(time=self._sw)

    def build_layer(self):
        repeat = 0
        if len(self._preds) > 1:
            past = self._preds[0].shape  # type: ignore[union-attr]
            # Built as a Modely, the past is read as the init: unless it has
            # the shape of the Input it stands for, it is broadcast to it.
            if past.tuple != self.shape.replace(time=self._sw - 1).tuple:
                repeat = 1 if past.time == self._sw - 1 else self._sw - 1
        return BufferImpl(
            axis=self.shape.time_axis,  # type: ignore[arg-type]
            expand=self._preds[-1].shape.time is None,  # type: ignore[union-attr]
            repeat=repeat,
            name=self.name,
        )

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The past values were saved as the Input this node reads first, so
        # the node is rebuilt on its predecessors, without an Input of its own.
        layer = super().from_config(config)
        return Layer.__call__(layer, list(preds)) if preds else layer
