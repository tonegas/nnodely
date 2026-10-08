"""TorchModule: a ``torch.nn.Module`` used as a layer, on every Keras backend."""

from __future__ import annotations

import base64
import contextlib
import io
from functools import partial

import keras
import numpy as np

from nnodely.core.layer import Layer


def _module_state(module):
    """The parameters, then the buffers, of ``module`` by their dotted names."""
    return dict(module.named_parameters()), dict(module.named_buffers())


def _keras_dtype(tensor):
    import torch

    # JAX keeps 32 bits unless told otherwise, so integer buffers (a
    # BatchNorm's batch counter) are held as int32 and cast back for torch.
    if tensor.is_floating_point():
        return keras.config.floatx()
    return "bool" if tensor.dtype == torch.bool else "int32"


def _probe(module, dim):
    """The feature shape ``module`` returns for one sample of shape ``dim``.

    Run on the "meta" device first, which computes shapes without arithmetic;
    modules holding plain tensors outside their state cannot, and run for real.
    """
    import torch

    params, buffers = _module_state(module)
    was_training = module.training
    module.eval()
    try:
        try:
            state = {
                key: torch.empty_like(value, device="meta")
                for key, value in {**params, **buffers}.items()
            }
            x = torch.empty((1, *dim), device="meta")
            y = torch.func.functional_call(module, state, (x,))
        except Exception:
            device = next(iter(params.values()), torch.empty(0)).device
            with torch.no_grad():
                y = module(torch.zeros((1, *dim), device=device))
    finally:
        module.train(was_training)
    if not isinstance(y, torch.Tensor):
        raise ValueError(
            f"TorchModule: the module must return one tensor, got {type(y).__name__}."
        )
    if y.dim() < 2:
        raise ValueError(
            "TorchModule: the module must return at least one feature axis after "
            f"the batch axis, got shape {tuple(y.shape)}."
        )
    return tuple(int(axis) for axis in y.shape[1:])


def _encode_module(module) -> str:
    import torch

    buffer = io.BytesIO()
    torch.save(module, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_module(data: str):
    import torch
    from keras.src.saving.serialization_lib import in_safe_mode

    # Unpickling runs code, as Keras's own TorchModuleWrapper warns.
    if in_safe_mode():
        raise ValueError(
            "Loading a TorchModule unpickles a torch.nn.Module, which can run "
            "arbitrary code, and is refused in safe mode. If you trust the file, "
            "pass safe_mode=False or call keras.config.enable_unsafe_deserialization()."
        )
    return torch.load(
        io.BytesIO(base64.b64decode(data)), map_location="cpu", weights_only=False
    )


def _fold(x, dim_rank):
    """``[batch, *dim, time, *seq]`` -> ``[batch * time * prod(seq), *dim]``.

    Returns the leading ``[batch, time, *seq]`` shape too, to unfold with.
    """
    rank = len(x.shape)
    x = keras.ops.transpose(x, [0, *range(1 + dim_rank, rank), *range(1, 1 + dim_rank)])
    lead = keras.ops.shape(x)[: rank - dim_rank]
    return keras.ops.reshape(x, (-1, *x.shape[rank - dim_rank :])), lead


def _unfold(y, lead):
    """``[batch * time * prod(seq), *out]`` -> ``[batch, *out, time, *seq]``."""
    y = keras.ops.reshape(y, (*lead, *y.shape[1:]))
    rank, lead_rank = len(y.shape), len(lead)
    return keras.ops.transpose(y, [0, *range(lead_rank, rank), *range(1, lead_rank)])


@keras.saving.register_keras_serializable(package="nnodely")
class TorchModuleImpl(keras.layers.Layer):
    """Applies a ``torch.nn.Module`` to every time and sequence step.

    The module's parameters and buffers are held as Keras variables, so Keras
    trains and saves them; the module only supplies its forward pass, run with
    ``torch.func.functional_call`` on those variables. On the torch backend
    that call is native. On tensorflow and jax it runs on the host through
    ``tf.numpy_function`` / ``jax.pure_callback``, and its gradient is
    computed by torch autograd, replaying the forward pass with the same
    random seed so that dropout draws the same mask.
    """

    def __init__(self, module, dim, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if isinstance(module, str):
            module = _decode_module(module)
        # Out of torch's module registry: on the torch backend a Keras layer is
        # a torch module, which would adopt this one and move it to its device.
        object.__setattr__(self, "module", module)
        self.dim = tuple(dim)
        self.out_dim = _probe(module, self.dim)
        # XLA cannot compile a host callback on tensorflow; jax can.
        self.supports_jit = keras.backend.backend() != "tensorflow"

        params, buffers = _module_state(module)
        self._names = [*params, *buffers]
        self._n_params = len(params)
        self._torch_dtypes = [value.dtype for value in {**params, **buffers}.values()]
        self._state = [
            self.add_weight(
                shape=tuple(value.shape),
                dtype=_keras_dtype(value),
                initializer=keras.initializers.Constant(value.detach().cpu().numpy()),
                trainable=key in params and value.requires_grad,
                name=key.replace(".", "_"),
            )
            for key, value in {**params, **buffers}.items()
        ]
        if keras.backend.backend() != "torch":
            self.seed_generator = keras.random.SeedGenerator()

    def compute_output_shape(self, input_shape):
        dim_rank = len(self.dim)
        return (input_shape[0], *self.out_dim, *input_shape[1 + dim_rank :])

    def call(self, x, training=None):
        # A frozen module runs in inference mode, as a frozen Keras
        # BatchNormalization does: its statistics stay those it was given.
        training = bool(training) and self.trainable
        x, lead = _fold(x, len(self.dim))
        backend = keras.backend.backend()
        if backend == "torch":
            y, buffers = self._torch_call(x, training)
        else:
            seed = keras.random.randint(
                (), 0, 2**31 - 1, seed=self.seed_generator, dtype="int32"
            )
            state = [keras.ops.convert_to_tensor(v) for v in self._state]
            apply = self._tf_call if backend == "tensorflow" else self._jax_call
            y, buffers = apply(x, seed, state, training)
            # Running statistics move only while training, as in Keras.
            if training:
                for variable, value in zip(self._state[self._n_params :], buffers):
                    variable.assign(value)
        return _unfold(y, lead)

    # ------------------------------------------------------------------
    # torch backend
    # ------------------------------------------------------------------
    def _torch_call(self, x, training):
        import torch

        # The variables are torch tensors here: autograd reaches them, and a
        # BatchNorm updates its running statistics in them in place.
        state = {key: variable.value for key, variable in zip(self._names, self._state)}
        self.module.train(training)
        y = torch.func.functional_call(self.module, state, (x,))
        return y, None

    # ------------------------------------------------------------------
    # tensorflow and jax backends: the module runs on the host
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _torch_inputs(self, seed, state):
        import torch

        device = next(self.module.parameters(), torch.empty(0)).device
        tensors = [
            torch.tensor(np.asarray(value), dtype=dtype, device=device)
            for value, dtype in zip(state, self._torch_dtypes)
        ]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            yield device, tensors

    def _forward_host(self, training, x, seed, *state):
        import torch

        with self._torch_inputs(seed, state) as (device, tensors), torch.no_grad():
            self.module.train(training)
            y = torch.func.functional_call(
                self.module,
                dict(zip(self._names, tensors)),
                (torch.tensor(np.asarray(x), device=device),),
            )
        buffers = [
            value.cpu().numpy().astype(variable.dtype)
            for value, variable in zip(
                tensors[self._n_params :], self._state[self._n_params :]
            )
        ]
        return (y.cpu().numpy().astype(x.dtype), *buffers)

    def _backward_host(self, training, x, seed, dy, *state):
        import torch

        with self._torch_inputs(seed, state) as (device, tensors):
            x_t = torch.tensor(np.asarray(x), device=device, requires_grad=True)
            params = tensors[: self._n_params]
            for param in params:
                param.requires_grad_(True)
            self.module.train(training)
            y = torch.func.functional_call(
                self.module, dict(zip(self._names, tensors)), (x_t,)
            )
            grads = torch.autograd.grad(
                y,
                [x_t, *params],
                torch.tensor(np.asarray(dy), device=device),
                allow_unused=True,
            )
        return tuple(
            np.zeros(value.shape, dtype=value.dtype)
            if grad is None
            else grad.cpu().numpy().astype(value.dtype)
            for grad, value in zip(grads, [x, *state[: self._n_params]])
        )

    def _tf_call(self, x, seed, state, training):
        import tensorflow as tf

        n = self._n_params
        buffer_vars = self._state[n:]
        out_shape = (x.shape[0], *self.out_dim)

        @tf.custom_gradient
        def apply(x, *state):
            outputs = tf.numpy_function(
                partial(self._forward_host, training),
                [x, seed, *state],
                [x.dtype, *[variable.dtype for variable in buffer_vars]],
                stateful=True,
            )
            # A single output comes back as a tensor, not a list.
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            outputs[0].set_shape(out_shape)
            for output, variable in zip(outputs[1:], buffer_vars):
                output.set_shape(variable.shape)

            def grad(dy, *_):
                grads = tf.numpy_function(
                    partial(self._backward_host, training),
                    [x, seed, dy, *state],
                    [x.dtype, *[value.dtype for value in state[:n]]],
                    stateful=True,
                )
                if not isinstance(grads, (list, tuple)):
                    grads = [grads]
                for gradient, value in zip(grads, [x, *state[:n]]):
                    gradient.set_shape(value.shape)
                return [*grads, *[None] * len(buffer_vars)]

            return outputs, grad

        # Read into plain tensors: a variable reaching the function would make
        # tf.custom_gradient expect its gradient through a `variables` argument.
        outputs = apply(x, *[tf.identity(value) for value in state])
        return outputs[0], outputs[1:]

    def _jax_call(self, x, seed, state, training):
        import jax

        n = self._n_params
        spec = lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype)  # noqa: E731
        forward_spec = (
            jax.ShapeDtypeStruct((x.shape[0], *self.out_dim), x.dtype),
            *[spec(value) for value in state[n:]],
        )

        def forward(x, seed, state):
            return jax.pure_callback(
                partial(self._forward_host, training),
                forward_spec,
                x,
                seed,
                *state,
                vmap_method="sequential",
            )

        @jax.custom_vjp
        def apply(x, seed, state):
            return forward(x, seed, state)

        def apply_fwd(x, seed, state):
            return forward(x, seed, state), (x, seed, state)

        def apply_bwd(residuals, cotangents):
            x, seed, state = residuals
            grads = jax.pure_callback(
                partial(self._backward_host, training),
                tuple(spec(value) for value in [x, *state[:n]]),
                x,
                seed,
                cotangents[0],
                *state,
                vmap_method="sequential",
            )
            # Integer inputs take float0 cotangents; buffers take none.
            zero = lambda value: (  # noqa: E731
                np.zeros(value.shape, jax.dtypes.float0)
                if not jax.numpy.issubdtype(value.dtype, jax.numpy.floating)
                else jax.numpy.zeros_like(value)
            )
            return (
                grads[0],
                zero(seed),
                [*grads[1:], *[zero(value) for value in state[n:]]],
            )

        apply.defvjp(apply_fwd, apply_bwd)
        outputs = apply(x, seed, state)
        return outputs[0], outputs[1:]

    # ------------------------------------------------------------------
    # torch module access and serialization
    # ------------------------------------------------------------------
    def to_torch(self):
        """The wrapped module, given the values its variables hold now."""
        import torch

        state = {
            key: torch.as_tensor(variable.numpy(), dtype=dtype)
            for key, variable, dtype in zip(
                self._names, self._state, self._torch_dtypes
            )
        }
        self.module.load_state_dict(state, strict=False)
        return self.module

    def get_config(self):
        config = super().get_config()
        config.update({"module": _encode_module(self.to_torch()), "dim": self.dim})
        return config


class TorchModule(Layer):
    """A ``torch.nn.Module`` applied to a stream, on any Keras backend.

    The module is applied to every time and sequence step of the stream, and
    sees the stream's feature axes as its sample shape: a stream of shape
    ``[batch, *dim, time, *seq]`` reaches it as ``[N, *dim]`` with
    ``N = batch * time * prod(seq)``, and the ``[N, *out]`` it returns comes
    back as ``[batch, *out, time, *seq]``. An image stream
    ``Input("img", dim=(3, 224, 224))`` therefore feeds a vision model the
    ``[N, 3, 224, 224]`` batch it expects.

    Its parameters and buffers become Keras variables: the module is trained
    with the rest of the model and saved with it, and :meth:`to_torch` gives
    it back with the trained values. Set ``trainable=False`` to keep it fixed,
    as a pretrained feature extractor.

    On the tensorflow and jax backends the module runs on the host through a
    callback, so tensors are copied to and from torch at every call, and
    tensorflow cannot compile the model with XLA. Saving pickles the module,
    which loading unpickles: load only files you trust.
    """

    def __init__(self, module, trainable: bool = True, name=None):
        if isinstance(module, str):
            module = _decode_module(module)
        self.module = module
        self.trainable = bool(trainable)
        super().__init__(name=name, module=module, trainable=self.trainable)

    def output_shape(self, *inputs):  # type: ignore
        dim, time, seq = inputs[0].shape.dimensions
        return _probe(self.module, dim), time, seq

    def build_layer(self):
        return TorchModuleImpl(
            self.module,
            dim=self.preds[0].shape.dim,  # type: ignore
            trainable=self.trainable,
            name=self.name,
        )

    def to_torch(self):
        """The wrapped module, with the values the model trained into it."""
        if self._layer is not None:
            return self._layer.to_torch()
        return self.module

    def get_config(self):
        return {
            "name": self.name,
            "module": _encode_module(self.to_torch()),
            "trainable": self.trainable,
        }
