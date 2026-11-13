"""KerasFunction: a custom function, a Keras layer, as an nnodely layer."""

from __future__ import annotations

import weakref
from typing import cast

import keras

from nnodely.core.layer import Layer


@keras.saving.register_keras_serializable(package="nnodely")
class KerasFunctionImpl(keras.layers.Layer):
    """Calls ``function``, a Keras layer, with its inputs as separate
    arguments."""

    def __init__(self, function, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.function = (
            function
            if isinstance(function, keras.layers.Layer)
            else keras.saving.deserialize_keras_object(function)
        )

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return self.function(*inputs)
        return self.function(inputs)

    def get_config(self):
        config = super().get_config()
        config["function"] = keras.saving.serialize_keras_object(self.function)
        return config


class KerasFunction(Layer):
    r"""Stream computed by a custom Keras layer: any function written as a
    ``keras.layers.Layer``, as a layer of the model.

    Syntax
    ------
    ``KerasFunction(F)(x)``
        Makes the Keras layer class ``F`` without arguments, and calls it on
        ``x``.
    ``KerasFunction(F(...))(x)``
        Calls the Keras layer instance ``F(...)`` on ``x``.
    ``KerasFunction(F)([a, b])``
        Calls the Keras layer with the streams as separate arguments,
        ``call(a, b)``.

    Write the computation in the ``call`` method of a ``keras.layers.Layer``
    subclass, with ``keras.ops``, so that it runs on every backend. The
    model saves and loads the layer with the rest.

    Parameters
    ----------
    function : keras.layers.Layer subclass or instance
        The Keras layer that computes the stream:

        - A class is made here, without arguments, anew for every
          KerasFunction.
        - An instance is used as it is. Its weights are shared wherever it
          is used.
    name : str, optional
        The name of the layer. The default is ``KerasFunction`` followed by
        a number.

    Call arguments
    --------------
    x : Stream or list of Stream
        One stream, or a list of streams passed to ``call`` as separate
        arguments, in order. Each arrives as a tensor laid out
        ``(batch, *seq, T, *D)``, without ``T`` for a stream without a
        window.

    Returns
    -------
    Stream
        A stream whose shape is read off the result of ``call``. It keeps
        the sequence axes, has a window if a stream it reads has one, and
        its remaining axes are the features. For example, a ``call`` that
        sums the 3 features of ``Shape(S=(), T=2, D=(3,))`` with
        ``keepdims=True`` gives ``Shape(S=(), T=2, D=(1,))``.

    Raises
    ------
    TypeError
        If ``function`` is not a ``keras.layers.Layer`` subclass or instance
        ("function is a keras.layers.Layer subclass or instance"). If it is
        a class that cannot be made without arguments ("cannot be made
        without arguments; pass an instance").

    See Also
    --------
    Fuzzify : Memberships, which also take Keras layers as functions.
    EquationLearner : Learned combination of elementary functions.
    Linear : Learned affine map of the features.

    Notes
    -----
    The tensors are passed to ``call`` as they are. Two streams of
    different shapes broadcast as Keras broadcasts, from the last axis.

    The weights of the Keras layer are Keras weights, trained with the
    model. They are not :class:`Parameter` objects: :attr:`Modely.parameters`
    does not list them, and :meth:`Modely.freeze` does not freeze them. Set
    the ``trainable`` attribute of the Keras layer instead.
    ``Modely.build(keep_weights=False)`` gives them their initial values
    again, from a copy of the layer made from its config.

    The class of the Keras layer is made known to Keras, so a saved model
    loads again in the same session. To load it in another session,
    define the class there first, decorated with
    ``@keras.saving.register_keras_serializable()``. A class with arguments
    needs a ``get_config`` that returns them.

    Basic Examples
    --------------
    A function of two streams, :math:`a^2 + b`:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, KerasFunction, Modely, Output
    >>> @keras.saving.register_keras_serializable()
    ... class SquarePlus(keras.layers.Layer):
    ...     def call(self, a, b):
    ...         return a * a + b
    >>> a, b = Input("a", tw=2), Input("b", tw=2)
    >>> y = KerasFunction(SquarePlus)([a, b])
    >>> model = Modely("custom", inputs=[a, b], outputs=[Output("y", y)]).build()
    >>> result = model({"a": np.array([[[1.0], [2.0]]]), "b": np.array([[[3.0], [4.0]]])})
    >>> keras.ops.convert_to_numpy(result["y"])  # 1 + 3, 4 + 4
    array([[[4.],
            [8.]]], dtype=float32)

    Scope
    -----
    The output shape is read off the result. A Euclidean norm of the
    features gives one feature:

    >>> @keras.saving.register_keras_serializable()
    ... class Norm(keras.layers.Layer):
    ...     def call(self, x):
    ...         return keras.ops.sqrt(keras.ops.sum(x * x, axis=-1, keepdims=True))
    >>> v = Input("v", dim=3, tw=1)
    >>> length = KerasFunction(Norm)(v)
    >>> length.shape
    Shape(S=(), T=1, D=(1,))
    >>> model = Modely("norm", inputs=[v], outputs=[Output("length", length)]).build()
    >>> keras.ops.convert_to_numpy(model({"v": np.array([[[3.0, 4.0, 0.0]]])})["length"])
    array([[[5.]]], dtype=float32)

    A sequence is kept, and a stream without a window has none:

    >>> KerasFunction(Norm)(Input("s", dim=3, seq=4, tw=2)).shape
    Shape(S=(4,), T=2, D=(1,))
    >>> KerasFunction(Norm)(Input("nw", dim=3)).shape
    Shape(S=(), T=None, D=(1,))

    Options
    -------
    ``function`` takes an instance, for a layer with arguments:

    >>> @keras.saving.register_keras_serializable()
    ... class Gain(keras.layers.Layer):
    ...     def __init__(self, gain, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.gain = gain
    ...     def call(self, x):
    ...         return self.gain * x
    ...     def get_config(self):
    ...         return {**super().get_config(), "gain": self.gain}
    >>> doubled = KerasFunction(Gain(2.0))(a)
    >>> model = Modely("gain", inputs=[a], outputs=[Output("y", doubled)]).build()
    >>> keras.ops.convert_to_numpy(model({"a": np.array([[[1.0], [2.0]]])})["y"])
    array([[[2.],
            [4.]]], dtype=float32)

    ``name`` names the layer:

    >>> KerasFunction(Norm, name="norm")(v).name
    'norm'

    Applications
    ------------
    The aerodynamic drag of a vehicle from the 3 components of its
    velocity, :math:`-c \, |v| \, v`:

    >>> @keras.saving.register_keras_serializable()
    ... class Drag(keras.layers.Layer):
    ...     def call(self, v):
    ...         speed = keras.ops.sqrt(keras.ops.sum(v * v, axis=-1, keepdims=True))
    ...         return -0.4 * speed * v
    >>> velocity = Input("velocity", dim=3, tw=1)
    >>> KerasFunction(Drag)(velocity).shape
    Shape(S=(), T=1, D=(3,))

    Properties & Relations
    ----------------------
    A model saved with a KerasFunction loads with it, and computes the same:

    >>> model = Modely("saved", inputs=[a, b], outputs=[Output("y", y)])
    >>> model.save("custom_model")
    >>> loaded = Modely.load("custom_model").build()
    >>> result = loaded({"a": np.array([[[1.0], [2.0]]]), "b": np.array([[[3.0], [4.0]]])})
    >>> keras.ops.convert_to_numpy(result["y"])
    array([[[4.],
            [8.]]], dtype=float32)

    The weights of a Keras layer are not Parameters:

    >>> @keras.saving.register_keras_serializable()
    ... class Scale(keras.layers.Layer):
    ...     def build(self, input_shape):
    ...         self.k = self.add_weight(shape=(1,), initializer="ones")
    ...     def call(self, x):
    ...         return self.k * x
    >>> scale = Scale()
    >>> model = Modely("weights", inputs=[a], outputs=[Output("y", KerasFunction(scale)(a))]).build()
    >>> len(model.parameters), len(scale.weights)
    (0, 1)

    Freeze them with the ``trainable`` attribute of the Keras layer:

    >>> scale.trainable = False
    >>> scale.trainable = True  # trained again

    Possible Issues
    ---------------
    A class with arguments cannot be made here: pass an instance.

    >>> KerasFunction(Gain)
    Traceback (most recent call last):
        ...
    TypeError: KerasFunction: Gain cannot be made without arguments; pass an instance.

    A plain Python function is not a Keras layer. Write it as the ``call``
    of one:

    >>> KerasFunction(lambda x: 2.0 * x)
    Traceback (most recent call last):
        ...
    TypeError: KerasFunction: function is a keras.layers.Layer subclass or instance, got ...

    .. versionadded:: 2.0
    """

    def __init__(self, function, name=None):
        if isinstance(function, dict):  # as a saved config holds it
            function = keras.saving.deserialize_keras_object(function)
        elif isinstance(function, type) and issubclass(function, keras.layers.Layer):
            try:
                function = function()
            except TypeError as exc:
                raise TypeError(
                    f"KerasFunction: {function.__name__} cannot be made without "
                    "arguments; pass an instance."
                ) from exc
        if not isinstance(function, keras.layers.Layer):
            raise TypeError(
                "KerasFunction: function is a keras.layers.Layer subclass or "
                f"instance, got {function!r}."
            )
        # Known to Keras by the name it is saved under, so a load finds it.
        keras.saving.get_custom_objects()[
            keras.saving.get_registered_name(type(function))
        ] = type(function)
        self._function = function
        super().__init__(name=name, function=function)
        # One Keras layer is one layer however many times it is wrapped: every
        # wrapper of an instance applies it, so they save and load as one.
        self._source = _WRAPPERS.setdefault(id(function), self._source)

    def build_layer(self):
        return KerasFunctionImpl(function=self._function, name=self.name)

    def _reinitialize(self) -> None:
        # The Keras layer holds its weights itself: one made again from its
        # config, and built as it was, gives them their initial values.
        function = self._function
        fresh = type(function).from_config(function.get_config())
        fresh.build_from_config(function.get_build_config())
        if len(fresh.weights) != len(function.weights):
            raise ValueError(
                f"{self.name}: the weights of {type(function).__name__} cannot "
                f"be initialized again: made again from its config, it has "
                f"{len(fresh.weights)} weights where it has "
                f"{len(function.weights)}. Give it a get_config() that makes it "
                "again, or a build() that makes its weights."
            )
        for variable, value in zip(function.weights, fresh.weights):
            variable.assign(value)

    def get_config(self):
        # The Keras layer as it is declared: the shape it was first built for
        # is not part of it, the model builds it again.
        config = super().get_config()
        function = cast(dict, keras.saving.serialize_keras_object(self._function))
        config["function"] = {k: v for k, v in function.items() if k != "build_config"}
        return config


# The first wrapper of every Keras layer still wrapped, by the layer's id: a
# live wrapper keeps its layer alive, so the id is not reused meanwhile.
_WRAPPERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _is_keras_layer(function) -> bool:
    return isinstance(function, keras.layers.Layer) or (
        isinstance(function, type) and issubclass(function, keras.layers.Layer)
    )
