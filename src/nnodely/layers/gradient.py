"""Gradient of a scalar model with respect to its input.

`Derivative` differentiates a stream along its own time axis with a finite
difference. This differentiates one stream with respect to another, which is a
different operation and needs the backend's autodiff rather than a stencil:
it is what `torch.autograd.grad(H.sum(), x, create_graph=True)` does in every
Hamiltonian network, and what any physics-informed model needs to read a field
off a learned potential.

The scalar comes in as a built `Modely` rather than as a stream, for the same
reason `Loop` and `OdeNet` take one: the tape has to watch the input while the
scalar is computed, so the computation must be replayable. Passing a stream
that was already evaluated would be too late. A built model also means every
evaluation shares one set of weights instead of rebuilding the network.

Keras exposes no backend-agnostic way to differentiate an output with respect
to an input - `keras.ops` has only `custom_gradient` and `stop_gradient` - so
the three backends are dispatched explicitly below.

The backward pass is materialised in the traced graph, which saves and reloads
like any other relation but does not export to ONNX: the gradient ops it leaves
behind (`BroadcastGradientArgs`, `TanhGrad`, `StridedSliceGrad`) have no ONNX
equivalent. That is the trade `Derivative` avoids by using a stencil, and the
reason to keep the two operators separate rather than folding one into the other.
"""

from __future__ import annotations

import keras

from nnodely.core.layer import Layer
from nnodely.core.modely import Modely
from nnodely.core.stream import Stream


def _gradient_tensorflow(total, x):
    import tensorflow as tf

    with tf.GradientTape() as tape:
        tape.watch(x)
        value = total(x)
    return tape.gradient(value, x)


def _gradient_torch(total, x):
    import torch

    # Prediction runs under `no_grad`, which would leave the scalar detached and
    # the gradient undefined, so the tape is re-enabled for this block alone.
    with torch.enable_grad():
        # A tensor already carrying history is differentiated where it is; only
        # one that carries none - a raw input - needs to be made a leaf first.
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        return torch.autograd.grad(total(x), x, create_graph=True)[0]


def _gradient_jax(total, x):
    import jax

    return jax.grad(total)(x)


_GRADIENTS = {
    "tensorflow": _gradient_tensorflow,
    "torch": _gradient_torch,
    "jax": _gradient_jax,
}


@keras.saving.register_keras_serializable(package="nnodely")
class GradientImpl(keras.layers.Layer):
    """dy/dx of a scalar model, read off the backend's tape.

    Summing y over the batch is the usual trick: the samples are independent, so
    the gradient of the sum is each sample's own gradient. The tape is read
    inside whatever tape the optimizer is holding, which is what keeps dy/dx
    itself differentiable with respect to the weights of the model.
    """

    def __init__(self, model=None, input_name="", output_name="", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.input_name = input_name
        self.output_name = output_name

    def compute_output_spec(self, x):
        # The gradient has the shape of the input, and declaring it keeps Keras
        # from tracing a tape over symbolic tensors just to learn that.
        return keras.KerasTensor(shape=x.shape, dtype=self.compute_dtype)

    def _total(self, x):
        outputs = self.model({self.input_name: x})
        value = outputs[self.output_name] if isinstance(outputs, dict) else outputs
        return keras.ops.sum(value)

    def call(self, x):
        backend = keras.backend.backend()
        gradient = _GRADIENTS.get(backend)
        if gradient is None:
            raise NotImplementedError(
                f"Gradient has no autodiff dispatch for the {backend!r} Keras backend; "
                f"choose one of {sorted(_GRADIENTS)}."
            )
        return gradient(self._total, x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": keras.saving.serialize_keras_object(self.model),
                "input_name": self.input_name,
                "output_name": self.output_name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        return cls(**config)


class Gradient(Layer):
    """Stream holding dy/dx, where `f` is a built Modely with one scalar output::

        Gradient(f, x)

    `f` takes a single input, which `x` feeds, and returns a single value per
    sample. The result has the shape of `x`: one partial derivative per element.

    `x` is any stream of the right shape, not only an Input window, so the same
    `f` can be differentiated at several points of one graph - at the state and
    at an equilibrium, say - with the weights shared between them.
    """

    def __init__(self, f: Modely, x: Stream, name=None):
        if not f.built:
            f.build()
        if len(f.inputs) != 1:
            raise ValueError(
                f"Gradient differentiates a model of one input, but {f.name!r} has "
                f"{len(f.inputs)}: {sorted(node.name for node in f.inputs)}."
            )
        if len(f.outputs) != 1:
            raise ValueError(
                f"Gradient needs a single output to differentiate, but {f.name!r} has "
                f"{len(f.outputs)}: {sorted(node.name for node in f.outputs)}."
            )
        output = f.outputs[0]
        if tuple(output.dim) != (1,) or output.time != 1:
            raise ValueError(
                f"Gradient needs a scalar output, but {output.name!r} has dim "
                f"{tuple(output.dim)} and time {output.time}."
            )
        self.f = f
        super().__init__(name=name, preds=[x], dim=x.dim, time=x.time, seq=x.seq)

    def build_layer(self):
        return GradientImpl(
            model=self.f.model,
            input_name=self.f.inputs[0].name,
            output_name=self.f.outputs[0].name,
            name=self.name,
        )

    def get_config(self):
        return {"name": self.name}
