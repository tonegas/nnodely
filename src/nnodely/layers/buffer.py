"""Buffer: the last values of a signal computed one sample per step."""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer
from nnodely.core.stream import Stream
from nnodely.layers.input import Input, _validate_init
from nnodely.layers.partitions import TimeSelect


@keras.saving.register_keras_serializable(package="nnodely")
class BufferImpl(keras.layers.Layer):
    """The past samples of a buffer followed by the current one, along the
    window; a current sample without a window gets one."""

    def __init__(self, axis: int, expand: bool, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = int(axis)
        self.expand = bool(expand)

    def call(self, inputs):
        *past, current = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        if self.expand:
            current = keras.ops.expand_dims(current, self.axis)
        if not past:
            return current
        return keras.ops.concatenate([past[0], current], axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "expand": self.expand})
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
    Input closed with the signal is the other way round: its window ends at
    the previous step.

    The past ``sw - 1`` values are an Input closed with the signal, fed back
    by the :class:`~nnodely.ModelyLoop` like any other. ``init`` holds them
    before the first step: a number, a Parameter or Constant, or an Input -
    one sample, repeated, or ``sw - 1`` of them, oldest first; ``None`` for
    zeros. The signal is one sample, ``sw=1``, or has no window.
    """

    def __init__(self, sw: int, init: Stream | float | None = None, name=None):
        if isinstance(sw, Stream):
            raise TypeError(
                "Buffer is configured first and called on the signal, like every "
                "other layer: Buffer(sw=3)(u)."
            )
        if isinstance(sw, bool) or not isinstance(sw, int) or sw < 1:
            raise ValueError(f"Buffer: sw is a positive int, got {sw!r}.")
        self.sw = sw
        self.init = _validate_init("Buffer", init)
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
        if self.sw == 1:
            return super().__call__([stream])

        # Every call buffers a signal of its own, so each has its own past.
        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        past = Input(
            f"{call_name}_past",
            dim=stream.dim,
            seq=tuple(-1 if length is None else length for length in stream.seq)
            or None,
            sw=self.sw - 1,
        )
        window = super().__call__([past, stream])
        past.close(
            stream if stream.time == 1 else TimeSelect(-1)(window), init=self.init
        )
        return window

    def output_shape(self, *inputs):
        return inputs[-1].shape.replace(time=self.sw)

    def build_layer(self):
        return BufferImpl(
            axis=self.shape.time_axis,  # type: ignore[arg-type]
            expand=self.preds[-1].shape.time is None,  # type: ignore[union-attr]
            name=self.name,
        )

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The past values were saved as the Input this node reads first, so
        # the node is rebuilt on its predecessors, without an Input of its own.
        layer = cls(**config)
        if not preds:
            return layer
        return Layer.__call__(layer, list(preds))
