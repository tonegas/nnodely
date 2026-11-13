"""The derivative of a Stream with respect to an Input.

It is configured first and then called on the Stream to differentiate, like
every other layer::

    Gradient(order=1, wrt=x)(fun)        # d fun / d x   (automatic differentiation)

``Gradient`` is exact: the sub-graph that produces ``fun`` is differentiated
by the backend's automatic differentiation, so any relation in between
(activations, Fir/Linear weights, a whole sub-network) is taken into account.
The derivatives with respect to time are
:class:`~nnodely.layers.calculus.DerivativeStep` and
:class:`~nnodely.layers.calculus.Derivative`.
"""

from __future__ import annotations

import keras

from nnodely.core.dag import _source_input_tensors
from nnodely.core.layer import Layer
from nnodely.core.stream import Stream
from nnodely.layers.input import Input

#: Only first and second derivatives are supported.
_SUPPORTED_ORDERS = (1, 2)


# ----------------------------------------------------------------------
# Graph introspection
# ----------------------------------------------------------------------
def _input_ancestors(node) -> list[Input]:
    """Every Input among ``node``'s ancestors, each once, in the order a walk
    from ``node`` reaches them: the same for a graph saved and loaded."""
    seen: set[int] = set()
    found: list[Input] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Input):
            found.append(current)
        stack.extend(getattr(current, "_preds", None) or [])
    return found


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
        "A Gradient with respect to an Input needs automatic differentiation, "
        "which the 'tensorflow', 'torch' and 'jax' Keras backends provide but "
        f"{backend!r} does not. Take a Derivative with respect to time instead, or "
        "select another backend with KERAS_BACKEND."
    )


@keras.saving.register_keras_serializable(package="nnodely")
class GradientImpl(keras.layers.Layer):
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
# Symbolic layer
# ----------------------------------------------------------------------
class Gradient(Layer):
    r"""Derivative of a stream with respect to an Input, by automatic
    differentiation.

    Syntax
    ------
    ``Gradient(wrt=x)(y)``
        The derivative of ``y`` with respect to the Input ``x``.
    ``Gradient(order=2, wrt=x)(y)``
        The second derivative.

    The derivative is exact: the backend differentiates every operation
    between ``x`` and ``y``, trainable weights included. The result is a
    stream like any other: it can be differentiated again, used in an
    objective, and trained through. This is how a physical law becomes an
    objective, as in physics-informed neural networks.

    The result has the shape of ``x``. When ``y`` has several values, a
    window or several features, the result is the derivative of their sum
    with respect to each value of ``x``. For the derivative with respect to
    time, use :class:`DerivativeStep` or :class:`Derivative`.

    Parameters
    ----------
    order : int, default 1
        The order of the derivative, ``1`` or ``2``.
    wrt : Input
        The Input to differentiate with respect to, as declared in the model
        that holds the layer. It is required.
    name : str, optional
        The name of the layer. The default is ``Gradient`` followed by a
        number.

    Call arguments
    --------------
    y : Stream
        A stream computed from ``wrt``, of any shape. The call takes it alone
        or in a list of one.

    Returns
    -------
    Stream
        The shape of ``wrt``, whatever the shape of ``y``. For example,
        ``Input("x")`` gives ``Shape(S=(), T=None, D=(1,))``, and
        ``Input("x", tw=2)`` gives ``Shape(S=(), T=2, D=(1,))``.

    Raises
    ------
    ValueError
        If ``order`` is not 1 or 2 ("Gradient supports order in [1, 2]"),
        or, when the model is built, if ``y`` does not depend on ``wrt``
        ("does not depend on input").
    TypeError
        If ``wrt`` is not an Input ("wrt must be the Input to differentiate
        against"). A number points to :class:`DerivativeStep`.

    See Also
    --------
    DerivativeStep : The time derivative along a window.
    Derivative : The time derivative one step at a time.
    Modely.export_onnx : The export to ONNX, and its batch size.

    Notes
    -----
    For an Input :math:`x` and a stream :math:`y` of values :math:`y_j`:

    .. math:: g = \frac{\partial}{\partial x} \sum_j y_j

    ``order=2`` differentiates :math:`g` in the same way. For a scalar
    :math:`y` of a scalar :math:`x`, these are :math:`dy/dx` and
    :math:`d^2y/dx^2`.

    The layer reads ``wrt`` where it is declared. A model called as a block
    inside another reads the streams of the call in place of its own Inputs:
    a Gradient inside the block has no Input left to differentiate against,
    and raises when the model is built. Differentiate in the model that
    declares the Input.

    ONNX export records the backward pass in the exported graph:

    * TensorFlow exports it with the batch axis fixed, to 1 by default.
    * PyTorch refuses to export it.
    * JAX exports no model to ONNX.

    Basic Examples
    --------------
    The derivative of :math:`x^3` is :math:`3x^2`:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Fir, Gradient, Input, Modely, Output, Parameter
    >>> x = Input("x")
    >>> dy_dx = Gradient(wrt=x)(x * x * x)
    >>> model = Modely("cube", inputs=[x], outputs=[Output("dy_dx", dy_dx)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0], [2.0]])})["dy_dx"])
    array([[ 3.],
           [12.]], dtype=float32)

    Scope
    -----
    For an Input with a window, the result has one derivative per sample.
    The derivative of a filter with respect to its window is its kernel:

    >>> w = Input("w", tw=2)
    >>> filtered = Fir(kernel=Parameter("k", value=[[3.0], [5.0]]), bias=False)(w)
    >>> model = Modely("fir", inputs=[w], outputs=[Output("g", Gradient(wrt=w)(filtered))])
    >>> keras.ops.convert_to_numpy(model.build()({"w": np.array([[[1.0], [2.0]]])})["g"])
    array([[[3.],
            [5.]]], dtype=float32)

    Options
    -------
    ``order=2`` gives the second derivative, :math:`6x`:

    >>> d2y_dx2 = Gradient(order=2, wrt=x)(x * x * x)
    >>> model = Modely("second", inputs=[x], outputs=[Output("d2y", d2y_dx2)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0], [2.0]])})["d2y"])
    array([[ 6.],
           [12.]], dtype=float32)

    ``name`` names the layer:

    >>> Gradient(wrt=x, name="slope")(x * x).name
    'slope'

    Applications
    ------------
    A physical law as an objective, called a minimizer in the method names.
    The model :math:`u = k t^2` must satisfy :math:`\dot u = 2t`. The
    objective has no target: training drives the residual to zero, and so
    ``k`` toward 1.

    >>> t = Input("t")
    >>> k = Parameter("k_law", value=3.0)
    >>> u = k * t * t
    >>> law = Modely("law", inputs=[t], outputs=[Output("u", u)])
    >>> law = law.minimize("physics", Gradient(wrt=t)(u) - 2.0 * t).build()

    Properties & Relations
    ----------------------
    ``order=2`` is the derivative of the derivative:

    >>> twice = Gradient(wrt=x)(Gradient(wrt=x)(x * x * x))
    >>> model = Modely("twice", inputs=[x], outputs=[Output("d2y", twice)]).build()
    >>> keras.ops.convert_to_numpy(model({"x": np.array([[1.0], [2.0]])})["d2y"])
    array([[ 6.],
           [12.]], dtype=float32)

    Possible Issues
    ---------------
    The derivative with respect to time is not a Gradient:

    >>> Gradient(wrt=0.01)
    Traceback (most recent call last):
        ...
    TypeError: Gradient: wrt must be the Input to differentiate against, got float. For a derivative over time, use DerivativeStep(dt=...) or Derivative(dt=...).

    The stream must depend on ``wrt``. The error comes when the model is
    built:

    >>> z = Input("z")
    >>> Modely("unrelated", inputs=[x, z], outputs=[Output("g", Gradient(wrt=z)(x * x))]).build()
    Traceback (most recent call last):
        ...
    ValueError: ...the differentiated relation does not depend on input 'z' (it reads ['x'])...

    .. versionadded:: 2.0
    """

    def __init__(self, order: int = 1, wrt: Input | None = None, name=None):
        if order not in _SUPPORTED_ORDERS:
            raise ValueError(
                f"Gradient supports order in {list(_SUPPORTED_ORDERS)}, got {order}."
            )
        self._order = int(order)
        if not isinstance(wrt, Input):
            hint = (
                " For a derivative over time, use DerivativeStep(dt=...) or "
                "Derivative(dt=...)."
                if isinstance(wrt, (int, float)) and not isinstance(wrt, bool)
                else ""
            )
            raise TypeError(
                "Gradient: wrt must be the Input to differentiate against, "
                f"got {type(wrt).__name__}.{hint}"
            )
        self._wrt = wrt
        super().__init__(name=name, order=self._order, wrt=self._wrt)

    # ------------------------------------------------------------------
    # Shape logic
    # ------------------------------------------------------------------
    def output_shape(self, *inputs):
        function = inputs[0]
        if not isinstance(function, Stream):
            raise TypeError(
                f"{self.name}: Gradient expects a Stream input, got "
                f"{type(function).__name__}."
            )
        if len(inputs) != 1:
            raise ValueError(
                f"{self.name}: Gradient differentiates exactly one Stream, "
                f"got {len(inputs)}."
            )
        # Reverse-mode differentiation yields the shape of the variable, not
        # of the relation.
        return self._wrt.shape

    # ------------------------------------------------------------------
    # Keras layer logic
    # ------------------------------------------------------------------
    def build_layer(self):
        raise RuntimeError(
            f"{self.name}: a Gradient is built from the graph it "
            "differentiates, not from its input value."
        )

    def call(self, xs):
        tensor = xs[0]
        sources = _source_input_tensors(tensor)
        # The Input by its Keras input, the one its copy in the graph reads.
        wrt_index = next(
            (
                index
                for index, source in enumerate(sources)
                if source is self._wrt._input
            ),
            None,
        )
        if wrt_index is None:
            names = [source.name for source in sources]
            raise ValueError(
                f"{self.name}: the differentiated relation does not depend on "
                f"input {self._wrt.name!r} (it reads {names}), so its derivative "
                "is not defined by the graph. Inside a model used as a reusable "
                "block this is what binding does: the block's own inputs are "
                "replaced by the streams it is called with, so differentiate in "
                "the model where the input is declared."
            )
        # Built for the relation of this application: one built for another -
        # in an earlier graph, or by another call of the block holding it -
        # would compute that one. Its weights are those of the relation's
        # layers, shared through their own nodes, so none is lost.
        function_model = keras.Model(
            inputs=sources,
            outputs=tensor,
            name=f"{self.name}_function",
        )
        self._layer = GradientImpl(
            function_model=function_model,
            wrt_index=wrt_index,
            order=self._order,
            name=self.name,
        )
        return self._layer(sources)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self):
        # The Input is saved by its position among the Inputs the relation it
        # differentiates is computed from, and found there again.
        config = super().get_config()
        # A copy of the Input - in a flattened graph - shares its Keras input.
        config["wrt"] = next(
            index
            for index, node in enumerate(_input_ancestors(self._preds[0]))
            if node._input is self._wrt._input
        )
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        if not preds:
            raise ValueError(
                "A Gradient needs the relation it differentiates to be rebuilt first."
            )
        wrt = _input_ancestors(preds[0])[config["wrt"]]
        return super().from_config({**config, "wrt": wrt}, preds=preds[:1])
