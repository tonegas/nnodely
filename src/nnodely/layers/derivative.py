"""Derivative of a Stream, with respect to an Input or with respect to time.

``Derivative(order=..., respect_to=...)`` is configured first and then called
on the Stream to differentiate, like every other layer::

    Derivative(order=1, respect_to=x)(fun)     # d fun / d x   (automatic differentiation)
    Derivative(order=1, respect_to=0.1)(fun)   # d fun / d t   (finite difference, dt = 0.1)

With an ``Input`` the derivative is exact: the sub-graph that produces ``fun``
is differentiated by the backend's automatic differentiation, so any relation
in between (activations, Fir/Linear weights, a whole sub-network) is taken
into account. With a float the argument *is* the time step, and the
derivative is a causal finite difference along ``fun``'s own time axis that
keeps the window's length: one derivative per sample, the oldest one reading
the ``init`` condition.
"""

from __future__ import annotations

import math

import keras
import numpy as np

from nnodely.core.layer import Layer
from nnodely.core.stream import Stream
from nnodely.layers.input import Input

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
    signals need. The weights are ordered oldest -> newest, matching the
    SampleWindow time-axis convention.
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
def _source_input_tensors(tensor):
    """The model-input tensors ``tensor`` is computed from.

    The Keras graph is walked rather than the nnodely DAG because the two do
    not always agree: a Constant or Parameter is wired to an arbitrary anchor
    input to give it a batch context, so the Keras sub-graph of a relation can
    need a tensor its symbolic predecessors never mention. Sources are ordered
    by name so a reloaded model resolves them exactly like the saved one did.
    """
    sources: dict[str, object] = {}
    seen: set[int] = set()
    stack = [tensor]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        history = getattr(current, "_keras_history", None)
        if history is None:
            continue
        operation = history[0]
        if isinstance(operation, keras.layers.InputLayer):
            sources[operation.name] = current
            continue
        node = operation._inbound_nodes[history[1]]
        stack.extend(node.input_tensors)
    return [sources[name] for name in sorted(sources)]


def _source_names(tensors):
    return [tensor._keras_history[0].name for tensor in tensors]


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
        "Derivative with respect to an Input needs automatic differentiation, "
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
        dim_rank: int,
        init_time: int = 1,
        init_has_batch: bool = True,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.dim_rank = int(dim_rank)
        self.init_time = int(init_time)
        self.init_has_batch = bool(init_has_batch)

    @property
    def past(self) -> int:
        """Samples of history the stencil reads before the window's first."""
        return len(self.coefficients) - 1

    def build(self, input_shape):
        shapes = (
            input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # type: ignore[list-item]
        )
        window_length = int(shapes[0][1 + self.dim_rank])
        operator = _operator_matrix(self.coefficients, window_length)
        self.operator = self.add_weight(
            name="operator",
            shape=operator.shape,
            initializer=keras.initializers.Constant(operator),  # type: ignore[arg-type]
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def _time_slice(self, x, start, stop):
        slices = [slice(None)] * len(x.shape)
        slices[1 + self.dim_rank] = slice(start, stop)
        return x[tuple(slices)]

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, init = inputs[0], inputs[1] if len(inputs) > 1 else None
        else:
            x, init = inputs, None
        time_axis = 1 + self.dim_rank

        # One sample of zeros shaped like the window's first: it both stands
        # in for a missing initial condition and gives a given one the batch
        # (and dim) axes it may be broadcasting over - a Constant carries
        # neither, and a scalar initial condition carries no dim axis.
        reference = keras.ops.zeros_like(self._time_slice(x, 0, 1))
        if init is None:
            history = reference
        else:
            if not self.init_has_batch:
                init = keras.ops.expand_dims(init, axis=0)
            history = init + reference
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
                "dim_rank": self.dim_rank,
                "init_time": self.init_time,
                "init_has_batch": self.init_has_batch,
            }
        )
        return config


# ----------------------------------------------------------------------
# Symbolic layer
# ----------------------------------------------------------------------
class Derivative(Layer):
    """
    Derivative of a Stream, with respect to an Input or with respect to time::

        Derivative(order=1|2, respect_to=x)(fun)     # d fun / d x
        Derivative(order=1|2, respect_to=0.1)(fun)   # d fun / d t, with dt = 0.1

    ``respect_to`` is an ``Input`` or the time step:

    * **An Input.** ``fun`` is differentiated with respect to that input's
      whole temporal window by the backend's automatic differentiation, so
      every relation between the two - including trainable weights - is taken
      into account, and the result is itself differentiable (it can be nested,
      and trained through). The output has the shape of the input's window:
      for a window longer than one sample it is the derivative of the summed
      relation with respect to each sample, which is the usual reverse-mode
      (vector-Jacobian) reading. ``fun`` must actually depend on the input.

    * **A float.** The float *is* ``dt``, and the derivative is a causal
      finite difference along ``fun``'s own time axis. **The window's length is
      preserved**: a window of n samples gives n derivatives, the i-th one
      estimated from the samples up to i, so the result stays aligned with the
      signal and composes with ``Integrate`` (whose cumulative form is its
      exact inverse: integrating ``Derivative(x)`` returns ``x - x[0]``).

    The time derivative takes two further arguments:

    * ``init`` - the initial condition, the sample(s) just before the window,
      which the oldest derivative needs. A Stream (typically the state the
      window continues, so a rollout stays continuous across chunks), a number
      (kept as a Constant), or ``None`` for zero. A Stream of one sample is
      reused for every sample of history a longer stencil needs.

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

    Two notes on differentiating with respect to an Input:

    * It reads the input where it is declared. A model used as a reusable
      block has its own inputs replaced by the streams it is called with, so
      a Derivative inside such a block no longer has an input to read and
      says so rather than guessing.

    * ONNX export records the backward pass in the exported graph. Its shape
      arithmetic only converts with a fixed batch axis, so
      ``Modely.export_onnx`` fixes it (see its ``batch_size`` argument); and
      the Torch backend, whose exporter traces a forward pass only, refuses
      such a model outright - export it under the TensorFlow or JAX backend.
    """

    def __init__(
        self,
        order: int = 1,
        respect_to: Input | float | None = None,
        init: Stream | float | None = None,
        window: int | None = None,
        poly_order: int | None = None,
        name=None,
    ):
        if order not in _SUPPORTED_ORDERS:
            raise ValueError(
                f"Derivative supports order in {list(_SUPPORTED_ORDERS)}, got {order}."
            )
        if respect_to is None:
            raise ValueError(
                "Derivative requires respect_to: an Input to differentiate "
                "against, or a float with the time step dt."
            )
        if isinstance(respect_to, Input):
            self.respect_to = respect_to
            self.wrt_name = respect_to.name
            self.dt = None
        elif isinstance(respect_to, (int, float)) and not isinstance(respect_to, bool):
            if respect_to <= 0:
                raise ValueError(
                    f"Derivative: the time step dt must be positive, got {respect_to}."
                )
            self.respect_to = float(respect_to)
            self.wrt_name = None
            self.dt = float(respect_to)
        else:
            raise TypeError(
                "Derivative: respect_to must be an Input or a float (the time "
                f"step dt), got {type(respect_to).__name__}."
            )

        self.order = int(order)
        self.window = self.order + 1 if window is None else int(window)
        # The lowest degree that can represent the derivative at all, so
        # widening the window always buys smoothing; a higher degree buys
        # accuracy back at the cost of noise.
        self.poly_order = self.order if poly_order is None else int(poly_order)
        self.init = init

        if self.dt is None:
            if init is not None or window is not None or poly_order is not None:
                raise ValueError(
                    "Derivative: init, window and poly_order describe a finite "
                    "difference over time. A derivative with respect to an Input "
                    "is exact and reads no window of its own."
                )
        else:
            if self.window < self.order + 1:
                raise ValueError(
                    f"Derivative: a derivative of order {self.order} reads at least "
                    f"{self.order + 1} samples, got window={self.window}."
                )
            if not self.order <= self.poly_order <= self.window - 1:
                raise ValueError(
                    f"Derivative: poly_order must be between the derivative's order "
                    f"({self.order}) and window - 1 ({self.window - 1}), got "
                    f"{self.poly_order}."
                )

        super().__init__(
            name=name,
            order=self.order,
            respect_to=self.respect_to,
            init=self.init,
            window=self.window if self.dt is not None else None,
            poly_order=self.poly_order if self.dt is not None else None,
        )

    # ------------------------------------------------------------------
    # Symbolic graph logic
    # ------------------------------------------------------------------
    @property
    def past(self) -> int:
        """Samples of history the stencil needs before the window's first."""
        return self.window - 1

    def _init_stream(self):
        """The initial condition as a node of the graph, if there is one."""
        if self.init is None or isinstance(self.init, Stream):
            return self.init
        return Stream._coerce_operand(float(self.init))

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        # The initial condition is a predecessor like any other, but it is
        # configured on the layer rather than passed at the call, so it is
        # appended here and reconnected the same way when reloading.
        if inputs and all(isinstance(value, Stream) for value in inputs):
            init = self._init_stream()
            inputs = [inputs[0]] if init is None else [inputs[0], init]
        return super().__call__(inputs)

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def output_shape(self, *inputs):
        function = inputs[0]
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: Derivative expects a Stream input, got "
                f"{type(function).__name__}."
            )
        if self.dt is None:
            if len(inputs) != 1:
                raise ValueError(
                    f"{self.name}: Derivative differentiates exactly one Stream, "
                    f"got {len(inputs)}."
                )
            # Reverse-mode differentiation yields the shape of the variable, not
            # of the relation: one derivative per sample of the input's window.
            wrt = self.respect_to
            return wrt.shape.dim, wrt.shape.time, wrt.shape.seq  # type: ignore[union-attr]

        if len(inputs) > 1:
            self._validate_init(function, inputs[1])
        # One derivative per sample: the window is preserved, so the result
        # stays aligned with the signal it came from.
        return function.shape.dim, function.shape.time, function.shape.seq

    def _validate_init(self, function, init):
        init_dim, init_seq = tuple(init.shape.dim), tuple(init.shape.seq)
        if init.shape.time not in (1, self.past):
            raise ValueError(
                f"{self.name}: init must carry 1 sample (reused for each) or the "
                f"{self.past} the stencil reads before the window, got "
                f"{init.shape.time}."
            )
        if init_dim not in ((1,), tuple(function.shape.dim)):
            raise ValueError(
                f"{self.name}: init has dim {init_dim}, which is neither a "
                f"scalar nor the differentiated relation's {tuple(function.shape.dim)}."
            )
        if init_seq not in ((), tuple(function.shape.seq)):
            raise ValueError(
                f"{self.name}: init has seq {init_seq}, which is neither "
                f"empty nor the differentiated relation's {tuple(function.shape.seq)}."
            )

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def build_layer(self):
        if self.dt is None:
            raise RuntimeError(
                f"{self.name}: a Derivative with respect to an Input is built "
                "from the graph it differentiates, not from its input value."
            )
        init_time, init_has_batch = 1, True
        if len(self.preds) > 1:
            init_node = self.preds[1]
            init_time = init_node.shape.time if isinstance(init_node, Stream) else 1
            # A Constant or Parameter is a leaf of the graph and carries no
            # batch axis, so the initial condition it holds broadcasts instead.
            init_has_batch = not (
                isinstance(init_node, Layer) and len(init_node.preds) == 0
            )
        return TimeDerivativeImpl(
            coefficients=_stencil_coefficients(
                self.order, self.window, self.poly_order, self.dt
            ),
            dim_rank=len(self.dim),
            init_time=init_time,
            init_has_batch=init_has_batch,
            name=self.name,
        )

    def call(self, xs):
        if self.dt is not None:
            signature = self.input_signature(xs)
            stale = (
                self._layer_signature is not None and self._layer_signature != signature
            )
            if self._layer is None or stale:
                self._layer = self.build_layer()
            self._layer_signature = signature
            return self._layer(xs if len(xs) > 1 else xs[0])

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
        config = {
            "name": self.name,
            "order": self.order,
            "respect_to": self.wrt_name if self.dt is None else self.dt,
        }
        if self.dt is not None:
            config.update({"window": self.window, "poly_order": self.poly_order})
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        respect_to = config["respect_to"]
        if isinstance(respect_to, str):
            if not preds:
                raise ValueError(
                    "A Derivative with respect to an Input needs the relation it "
                    "differentiates to be rebuilt first."
                )
            resolved = _find_input(preds[0], respect_to)
            if resolved is None:
                raise ValueError(
                    f"Derivative {config['name']!r}: input {respect_to!r} was not "
                    "found among the ancestors of the relation it differentiates."
                )
            respect_to = resolved

        # The initial condition was serialized as this node's second
        # predecessor, so it is rebuilt with the rest of the graph.
        layer = cls(
            order=config.get("order", 1),
            respect_to=respect_to,
            init=preds[1] if preds and len(preds) > 1 else None,
            window=config.get("window"),
            poly_order=config.get("poly_order"),
            name=config["name"],
        )
        if preds:
            return layer(preds[0])
        return layer
