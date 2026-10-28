"""Derivatives of a Stream: with respect to an Input, or with respect to time.

They are configured first and then called on the Stream to differentiate, like
every other layer::

    Differentiate(order=1, respect_to=x)(fun)   # d fun / d x   (automatic differentiation)
    DerivativeStep(order=1, dt=0.1)(fun)        # d fun / d t   over its window
    Derivative(order=1, dt=0.1)(fun)            # d fun / d t   step by step

``Differentiate`` is exact: the sub-graph that produces ``fun`` is differentiated
by the backend's automatic differentiation, so any relation in between
(activations, Fir/Linear weights, a whole sub-network) is taken into account.
``DerivativeStep`` is a causal finite difference along ``fun``'s own window
that keeps its length: one derivative per sample, the oldest one reading the
``init`` condition; it is the inverse of ``IntegrateStep``. ``Derivative`` is
the same stencil across the steps of a ModelyLoop, the inverse of
``Integrate``.
"""

from __future__ import annotations

import math

import keras
import numpy as np

from nnodely.core.dag import _source_input_tensors, _source_names
from nnodely.core.layer import Layer
from nnodely.core.shape import expand_to_rank, slice_axis
from nnodely.core.stream import Stream
from nnodely.layers.input import Input
from nnodely.layers.integrate import _signal_and_init

#: Only first and second derivatives are supported, in both modes.
_SUPPORTED_ORDERS = (1, 2)


# ----------------------------------------------------------------------
# Finite-difference stencils
# ----------------------------------------------------------------------
def _stencil_coefficients(
    order: int, window: int, poly_order: int, dt: float
) -> np.ndarray:
    """Weights estimating the ``order``-th derivative at the newest sample.

    A polynomial of degree ``poly_order`` is fitted by least squares to the
    ``window`` most recent samples and differentiated at the newest one - the
    Savitzky-Golay construction. With the minimum window the fit is exact and
    the weights are the textbook backward differences ((-1, 1)/dt for the
    first derivative, (1, -2, 1)/dt^2 for the second); a longer window turns
    the same expression into a smoothing estimate, which is what measured
    signals need. The weights are ordered oldest -> newest, as the samples
    of a window are.
    """
    offsets = np.arange(-(window - 1), 1, dtype=np.float64)
    design = np.vander(offsets, poly_order + 1, increasing=True)
    # Row `order` of the pseudo-inverse gives the fitted polynomial's
    # coefficient of t^order; the derivative adds its factorial.
    fit = np.linalg.pinv(design)
    return math.factorial(order) * fit[order] / float(dt) ** order


def _operator_matrix(coefficients, window_length: int) -> np.ndarray:
    """The stencil as the matrix that applies it to a whole window at once.

    Sliding the stencil over a window is a banded matrix product - the
    discrete differentiation operator of the window - so the layer runs as a
    single matmul rather than a scan, and differentiates (and exports) like
    any other linear operation.
    """
    past = len(coefficients) - 1
    matrix = np.zeros((past + window_length, window_length), dtype=np.float32)
    for column in range(window_length):
        for offset, coefficient in enumerate(coefficients):
            matrix[column + offset, column] = coefficient
    return matrix


# ----------------------------------------------------------------------
# Graph introspection
# ----------------------------------------------------------------------
def _find_input(node, name: str):
    """The Input named ``name`` among ``node``'s ancestors, if any."""
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Input) and current.name == name:
            return current
        stack.extend(getattr(current, "preds", None) or [])
    return None


# ----------------------------------------------------------------------
# Automatic differentiation with respect to an Input
# ----------------------------------------------------------------------
def _differentiate(function_model, inputs, wrt_index: int, order: int, training):
    """``order``-th derivative of ``function_model`` w.r.t. its ``wrt_index``-th input.

    The backend's own autodiff is used, so the result is differentiable again
    - both to nest derivatives and to train through them.
    """
    backend = keras.backend.backend()
    arguments = list(inputs)

    def evaluate(value):
        call_arguments = list(arguments)
        call_arguments[wrt_index] = value
        return function_model(call_arguments, training=training)

    if backend == "tensorflow":
        import tensorflow as tf

        def nested(remaining, value):
            if remaining == 0:
                return evaluate(value)
            with tf.GradientTape() as tape:
                tape.watch(value)
                inner = nested(remaining - 1, value)
            return tape.gradient(
                inner, value, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )

        return nested(order, arguments[wrt_index])

    if backend == "torch":
        import torch

        # Inference and the symbolic build run under torch.no_grad(), where
        # nothing records a backward pass; a derivative is part of this
        # model's forward pass, so it has to opt back in.
        with torch.enable_grad():
            value = arguments[wrt_index]
            if not value.requires_grad:
                value = value.detach().requires_grad_(True)
            result = evaluate(value)
            for _ in range(order):
                gradient = torch.autograd.grad(
                    result,
                    value,
                    grad_outputs=torch.ones_like(result),
                    create_graph=True,
                    allow_unused=True,
                )[0]
                if gradient is None:
                    gradient = torch.zeros_like(value)
                result = gradient
        return result

    if backend == "jax":
        import jax
        import jax.numpy as jnp

        def gradient_of(function, value):
            output, pullback = jax.vjp(function, value)
            return pullback(jnp.ones_like(output))[0]

        function = evaluate
        for _ in range(order):
            function = (lambda inner: lambda value: gradient_of(inner, value))(function)
        return function(arguments[wrt_index])

    raise NotImplementedError(
        "Differentiate with respect to an Input needs automatic differentiation, "
        "which the 'tensorflow', 'torch' and 'jax' Keras backends provide but "
        f"{backend!r} does not. Differentiate with respect to time instead, or "
        "select another backend with KERAS_BACKEND."
    )


@keras.saving.register_keras_serializable(package="nnodely")
class InputDerivativeImpl(keras.layers.Layer):
    """Differentiates a sub-graph with respect to one of the model's inputs.

    The layer is applied to the sub-graph's own source tensors, not to the
    value of the relation it differentiates: a value carries no derivative,
    so the relation has to be re-evaluated under the backend's autodiff.
    The sub-model shares its layers - and therefore its weights - with the
    graph that produced it.
    """

    #: Evaluating a backward pass constrains ONNX export: its shape arithmetic
    #: only converts once the batch axis is a number (see Modely.export_onnx).
    _traces_backward_pass = True

    def __init__(self, function_model, wrt_index: int, order: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.function_model = function_model
        self.wrt_index = int(wrt_index)
        self.order = int(order)

    def call(self, inputs, training=None):
        return _differentiate(
            self.function_model, inputs, self.wrt_index, self.order, training
        )

    def compute_output_shape(self, input_shape):
        return input_shape[self.wrt_index]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "function_model": keras.saving.serialize_keras_object(
                    self.function_model
                ),
                "wrt_index": self.wrt_index,
                "order": self.order,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["function_model"] = keras.saving.deserialize_keras_object(
            config["function_model"]
        )
        return cls(**config)


# ----------------------------------------------------------------------
# Finite difference with respect to time
# ----------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="nnodely")
class TimeDerivativeImpl(keras.layers.Layer):
    """Causal finite difference of a whole window, one derivative per sample.

    The stencil needs ``len(coefficients) - 1`` samples of history to produce
    the oldest derivative of the window; they come from the initial condition
    (a second input) or are taken as zero. Prepending them and applying the
    banded operator matrix makes the layer a single matmul over the padded
    window.
    """

    def __init__(
        self,
        coefficients,
        axis: int,
        init_time: int = 1,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.axis = int(axis)
        self.init_time = int(init_time)

    @property
    def past(self) -> int:
        """Samples of history the stencil reads before the window's first."""
        return len(self.coefficients) - 1

    def build(self, input_shape):
        shapes = (
            input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # type: ignore[list-item]
        )
        window_length = int(shapes[0][self.axis])
        operator = _operator_matrix(self.coefficients, window_length)
        self.operator = self.add_weight(
            name="operator",
            shape=operator.shape,
            initializer=keras.initializers.Constant(operator),  # type: ignore[arg-type]
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, init = inputs[0], inputs[1] if len(inputs) > 1 else None
        else:
            x, init = inputs, None
        time_axis = self.axis

        # One sample of zeros shaped like the window's first: it both stands
        # in for a missing initial condition and gives a given one the axes it
        # may be broadcasting over - a scalar initial condition carries no dim
        # size, and one may leave out the sequence axes.
        reference = keras.ops.zeros_like(slice_axis(x, time_axis, slice(0, 1)))
        if init is None:
            history = reference
        else:
            history = expand_to_rank(init, len(reference.shape)) + reference
        if self.init_time != self.past:
            history = keras.ops.repeat(history, self.past, axis=time_axis)

        padded = keras.ops.concatenate([history, x], axis=time_axis)
        result = keras.ops.tensordot(padded, self.operator, axes=[[time_axis], [0]])  # type: ignore[arg-type]
        # tensordot appends the window axis; put it back where time belongs.
        # The axis is addressed from the front because a negative index
        # survives tracing as a negative permutation, which ONNX rejects.
        last_axis = len(result.shape) - 1
        if last_axis == time_axis:  # nothing follows time, so it already is
            return result
        return keras.ops.moveaxis(result, last_axis, time_axis)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            return tuple(input_shape[0])
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "coefficients": self.coefficients,
                "axis": self.axis,
                "init_time": self.init_time,
            }
        )
        return config


# ----------------------------------------------------------------------
# Symbolic layers
# ----------------------------------------------------------------------
def _check_order(layer: str, order) -> int:
    if order not in _SUPPORTED_ORDERS:
        raise ValueError(
            f"{layer} supports order in {list(_SUPPORTED_ORDERS)}, got {order}."
        )
    return int(order)


class Differentiate(Layer):
    """
    Derivative of a Stream with respect to an Input::

        Differentiate(order=1|2, respect_to=x)(fun)     # d fun / d x

    ``fun`` is differentiated with respect to the input as it is declared,
    its whole window if it has one, by the backend's automatic
    differentiation, so every relation between the two - including trainable
    weights - is taken into account, and the result is itself differentiable
    (it can be nested, and trained through). The output has the shape of the
    input: for a window longer than one sample it is the derivative of the
    summed relation with respect to each sample, which is the usual
    reverse-mode (vector-Jacobian) reading. ``fun`` must actually depend on
    the input. For a derivative over time, see
    :class:`DerivativeStep` and :class:`Derivative`.

    Two notes:

    * It reads the input where it is declared. A model used as a reusable
      block has its own inputs replaced by the streams it is called with, so
      a Differentiate inside such a block no longer has an input to read and says
      so rather than guessing.

    * ONNX export records the backward pass in the exported graph. Its shape
      arithmetic only converts with a fixed batch axis, so
      ``Modely.export_onnx`` fixes it (see its ``batch_size`` argument); and
      the Torch backend, whose exporter traces a forward pass only, refuses
      such a model outright - export it under the TensorFlow backend (JAX
      exports no ONNX at all).
    """

    def __init__(self, order: int = 1, respect_to: Input | None = None, name=None):
        self.order = _check_order("Differentiate", order)
        if not isinstance(respect_to, Input):
            hint = (
                " For a derivative over time, use DerivativeStep(dt=...) or "
                "Derivative(dt=...)."
                if isinstance(respect_to, (int, float))
                and not isinstance(respect_to, bool)
                else ""
            )
            raise TypeError(
                "Differentiate: respect_to must be the Input to differentiate against, "
                f"got {type(respect_to).__name__}.{hint}"
            )
        self.respect_to = respect_to
        self.wrt_name = respect_to.name
        super().__init__(name=name, order=self.order, respect_to=self.respect_to)

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def output_shape(self, *inputs):
        function = inputs[0]
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: Differentiate expects a Stream input, got "
                f"{type(function).__name__}."
            )
        if len(inputs) != 1:
            raise ValueError(
                f"{self.name}: Differentiate differentiates exactly one Stream, "
                f"got {len(inputs)}."
            )
        # Reverse-mode differentiation yields the shape of the variable, not
        # of the relation.
        return self.respect_to.shape

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def build_layer(self):
        raise RuntimeError(
            f"{self.name}: a Differentiate is built from the graph it "
            "differentiates, not from its input value."
        )

    def call(self, xs):
        tensor = xs[0]
        sources = _source_input_tensors(tensor)
        names = _source_names(sources)
        if self.wrt_name not in names:
            raise ValueError(
                f"{self.name}: the differentiated relation does not depend on "
                f"input {self.wrt_name!r} (it reads {names}), so its derivative "
                "is not defined by the graph. Inside a model used as a reusable "
                "block this is what binding does: the block's own inputs are "
                "replaced by the streams it is called with, so differentiate in "
                "the model where the input is declared."
            )
        if self._layer is None:
            function_model = keras.Model(
                inputs=sources,
                outputs=tensor,
                name=f"{self.name}_function",
            )
            self._layer = InputDerivativeImpl(
                function_model=function_model,
                wrt_index=names.index(self.wrt_name),
                order=self.order,
                name=self.name,
            )
        return self._layer(sources)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self):
        return {"name": self.name, "order": self.order, "respect_to": self.wrt_name}

    @classmethod
    def from_config(cls, config: dict, preds=None):
        if not preds:
            raise ValueError(
                "A Differentiate needs the relation it differentiates to be rebuilt first."
            )
        respect_to = _find_input(preds[0], config["respect_to"])
        if respect_to is None:
            raise ValueError(
                f"Differentiate {config['name']!r}: input {config['respect_to']!r} was "
                "not found among the ancestors of the relation it differentiates."
            )
        layer = cls(
            order=config.get("order", 1), respect_to=respect_to, name=config["name"]
        )
        return layer(preds[0])


class _FiniteDifference(Layer):
    """What :class:`DerivativeStep` and :class:`Derivative` share: a causal
    finite-difference stencil of ``order``, ``window`` and ``poly_order``, the
    time step ``dt``, and the initial condition ``init``, the sample(s) just
    before the first one differentiated.
    """

    def __init__(
        self,
        order: int = 1,
        dt: float | None = None,
        window: int | None = None,
        poly_order: int | None = None,
        name=None,
        **properties,
    ):
        layer = type(self).__name__
        self.order = _check_order(layer, order)
        if dt is None:
            raise ValueError(f"{layer} requires dt, the time step.")
        if not isinstance(dt, (int, float)) or isinstance(dt, bool):
            raise TypeError(f"{layer}: dt must be a number, got {type(dt).__name__}.")
        if dt <= 0:
            raise ValueError(f"{layer}: the time step dt must be positive, got {dt}.")
        self.dt = float(dt)

        self.window = self.order + 1 if window is None else int(window)
        # The lowest degree that can represent the derivative at all, so
        # widening the window always buys smoothing; a higher degree buys
        # accuracy back at the cost of noise.
        self.poly_order = self.order if poly_order is None else int(poly_order)
        if self.window < self.order + 1:
            raise ValueError(
                f"{layer}: a derivative of order {self.order} reads at least "
                f"{self.order + 1} samples, got window={self.window}."
            )
        if not self.order <= self.poly_order <= self.window - 1:
            raise ValueError(
                f"{layer}: poly_order must be between the derivative's order "
                f"({self.order}) and window - 1 ({self.window - 1}), got "
                f"{self.poly_order}."
            )

        super().__init__(
            name=name,
            order=self.order,
            dt=self.dt,
            window=self.window,
            poly_order=self.poly_order,
            **properties,
        )

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------
    @property
    def past(self) -> int:
        """Samples of history the stencil needs before the first one."""
        return self.window - 1

    def _coefficients(self) -> np.ndarray:
        return _stencil_coefficients(self.order, self.window, self.poly_order, self.dt)

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def _check_function(self, function):
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: {type(self).__name__} expects a Stream input, got "
                f"{type(function).__name__}."
            )

    def _validate_init(self, function, init, seq, where: str):
        """``init`` holds 1 sample (reused for each) or the ``past`` the
        stencil reads, a scalar or the function's dim, and no sequence or
        ``seq``, the one ``where`` names. A value - a number, a Parameter or
        a Constant - without a window is one sample, and has no sequence."""
        init_dim, samples = tuple(init.shape.dim), init.shape.time
        if samples is None:
            if not init._is_value:
                raise ValueError(
                    f"{self.name}: init has no window, {init.shape}: the samples "
                    "before the first need one - declare the Input with sw=1 - "
                    "unless init is a number, a Parameter or a Constant."
                )
            seq = ()
        elif samples not in (1, self.past):
            raise ValueError(
                f"{self.name}: init must carry 1 sample (reused for each) or the "
                f"{self.past} the stencil reads before the first, got {samples}."
            )
        if init_dim not in ((1,), tuple(function.shape.dim)):
            raise ValueError(
                f"{self.name}: init has dim {init_dim}, which is neither a "
                f"scalar nor the differentiated relation's {tuple(function.shape.dim)}."
            )
        if tuple(init.shape.seq) not in ((), tuple(seq)):
            raise ValueError(
                f"{self.name}: init has the sequence {tuple(init.shape.seq)}, "
                f"which is neither empty nor {where}, {tuple(seq)}."
            )

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def call(self, xs):
        signature = self.input_signature(xs)
        stale = self._layer_signature is not None and self._layer_signature != signature
        if self._layer is None or stale:
            self._layer = self.build_layer()
        self._layer_signature = signature
        return self._layer(xs if len(xs) > 1 else xs[0])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self):
        return {
            "name": self.name,
            "order": self.order,
            "dt": self.dt,
            "window": self.window,
            "poly_order": self.poly_order,
        }

    @classmethod
    def from_config(cls, config: dict, preds=None):
        # The initial condition was serialized as this node's second
        # predecessor, so it is rebuilt with the rest of the graph: the layer
        # is called on it, then on the signal.
        layer = cls(
            order=config.get("order", 1),
            dt=config["dt"],
            window=config.get("window"),
            poly_order=config.get("poly_order"),
            name=config["name"],
        )
        if preds:
            return layer(*preds[::-1])
        return layer


class DerivativeStep(_FiniteDifference):
    """
    Derivative of a Stream with respect to time, over its own window::

        DerivativeStep(order=1|2, dt=0.1)(fun)           # d fun / d t
        DerivativeStep(order=1|2, dt=0.1)(init, fun)

    A causal finite difference along ``fun``'s time axis, with the time step
    ``dt`` given explicitly as for :class:`IntegrateStep`. **The window's length is
    preserved**: a window of n samples gives n derivatives, the i-th one
    estimated from the samples up to i, so the result stays aligned with the
    signal and composes with ``IntegrateStep``, whose cumulative form is its
    inverse: integrating ``DerivativeStep(dt=dt)(x0, x)`` with the same
    ``init`` returns ``x`` (exactly for the default first-order, two-sample
    stencil). A one-sample window is one step, ``(x - init) / dt``. Across
    the steps of a ModelyLoop, see :class:`Derivative`; for a derivative
    with respect to an Input, see :class:`Differentiate`.

    * ``init``, given first when the layer is called - the initial condition,
      the sample(s) just before the window, which the oldest derivative needs.
      A Stream (typically the state the window continues, so a rollout stays
      continuous across chunks), or a number (kept as a Constant); called on
      the signal alone, the layer reads zero before it. A Stream of one sample
      is reused for every sample of history a longer stencil needs.

    * ``window`` - how many samples each derivative reads, ``order + 1`` by
      default (the textbook backward difference). A longer window fits a
      polynomial of degree ``poly_order`` by least squares and differentiates
      that: still causal, still one matmul, but a *smoothing* estimate. This
      is the knob that matters for measured mechanical signals, where the
      two-sample difference multiplies sensor noise by sqrt(2)/dt - at
      dt = 1 ms, a 0.1 mm encoder ripple becomes 0.14 m/s of velocity. Its
      noise gain falls roughly as ``window^-3/2`` (0.32/dt at ``window=5``,
      0.13/dt at ``window=9``), paid for with a group delay of about
      ``(window - 1) / 2`` samples: widen it until the delay approaches the
      time constant of the dynamics being modelled, not beyond.

    * ``poly_order`` - the degree of that fit, the derivative's own ``order``
      by default (maximum smoothing). Raising it removes delay and bias at
      the price of noise: ``window=3, poly_order=2`` is the second-order
      accurate backward difference (BDF2), exact on a parabola, but it
      amplifies noise about 1.8x more than the plain two-sample difference.
      Smooth simulated signals want the higher degree, measured ones the
      default.
    """

    def __call__(self, *inputs):
        # The initial condition is a predecessor like any other, after the
        # signal, so that it is reconnected the same way when reloading.
        return super().__call__(_signal_and_init("DerivativeStep", inputs))

    def output_shape(self, *inputs):
        function = inputs[0]
        self._check_function(function)
        self._needs_window(function)
        if len(inputs) > 1:
            self._validate_init(
                function, inputs[1], function.shape.seq, "the differentiated relation's"
            )
        # One derivative per sample: the window is preserved, so the result
        # stays aligned with the signal it came from.
        return function.shape

    def build_layer(self):
        init_time = 1
        if len(self.preds) > 1:
            init_node = self.preds[1]
            if isinstance(init_node, Stream):
                # A value without a window is one sample.
                init_time = init_node.shape.time or 1
        return TimeDerivativeImpl(
            coefficients=self._coefficients(),
            axis=self.shape.time_axis,  # type: ignore[arg-type]
            init_time=init_time,
            name=self.name,
        )


class Derivative(_FiniteDifference):
    """
    Derivative of a Stream with respect to time, one step at a time, in a
    :class:`~nnodely.ModelyLoop`::

        Derivative(order=1|2, dt=0.1, init=x0)(fun)     # d fun / d t

    The counterpart of :class:`DerivativeStep` across the steps of a
    ModelyLoop, as :class:`Integrate` is :class:`IntegrateStep`'s: ``fun`` is
    one sample of a step, ``T=1`` and no sequence, and the ``window - 1``
    samples before it are an Input closed with ``fun``, which the ModelyLoop
    feeds back with all its other closed Inputs - no Loop of its own. The
    result is the derivative at this step, estimated from the steps up to it.
    The stencil is ``DerivativeStep``'s - ``order``, ``window`` and
    ``poly_order`` mean the same - so laid out along a time window instead,
    the same samples give the same derivatives, and ``Integrate`` with the
    same ``init`` inverts it (exactly for the default first-order, two-sample
    stencil).

    ``init`` is the value just before the first step, as :meth:`Input.close`
    takes it: one sample (reused for every step of history a longer stencil
    reads) or the ``window - 1`` samples it reads, oldest first, along its
    window, its dim a scalar or ``fun``'s - a number, a Parameter or a
    Constant, or an Input, whose first step the ModelyLoop reads; ``None``
    for zero. Built as a :class:`~nnodely.Modely` instead, nothing feeds the
    past samples back: they are read from data, as every closed Input is.
    """

    def __init__(
        self,
        order: int = 1,
        dt: float | None = None,
        init: Stream | float | None = None,
        window: int | None = None,
        poly_order: int | None = None,
        name=None,
    ):
        self.init = init
        self._calls = 0
        super().__init__(
            order=order,
            dt=dt,
            window=window,
            poly_order=poly_order,
            name=name,
            init=init,
        )

    def __call__(self, function):
        if isinstance(function, (list, tuple)) and len(function) == 1:
            function = function[0]
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: Derivative expects a Stream input, got "
                f"{type(function).__name__}."
            )
        if function.seq:
            raise ValueError(
                f"{self.name}: Derivative takes the signal of one step of a "
                f"ModelyLoop, and this signal carries a sequence, {function.shape}. "
                "Declare it one sample per step, an Input with sw=1 and no seq, "
                "and build the model with ModelyLoop."
            )
        self._needs_window(function)
        if function.time != 1:
            raise ValueError(
                f"{self.name}: Derivative takes one sample per step, but the "
                f"signal carries {function.time} time samples. Differentiate a "
                "time window with DerivativeStep."
            )
        if isinstance(self.init, Stream):
            self._validate_init(function, self.init, (), "the signal's")

        self._calls += 1
        call_name = self.name if self._calls == 1 else f"{self.name}_{self._calls}"
        past = Input(f"{call_name}_past", dim=function.dim, sw=self.past)
        derivative = DerivativeStep(
            order=self.order,
            dt=self.dt,
            window=self.window,
            poly_order=self.poly_order,
            name=f"{call_name}_step",
        )(past, function)
        past.close(function, init=self.init)
        return derivative

    def build_layer(self):
        raise RuntimeError(
            f"{self.name}: a Derivative is the DerivativeStep it calls on an "
            "Input closed with its signal, not a layer of its own."
        )
