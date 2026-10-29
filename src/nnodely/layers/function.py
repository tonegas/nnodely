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
    """
    A custom function in the graph: an nnodely layer computing with
    ``function``, a Keras layer - a ``keras.layers.Layer`` subclass, made
    here without arguments, or an instance. Its streams are passed to
    ``call`` as separate arguments, laid out ``(batch, *seq, time, *dim)``,
    and its output is a stream whose shape is read off the result::

        class Shifted(keras.layers.Layer):
            def call(self, x, y):
                return x * x + y

        shifted = KerasFunction(Shifted)([a, b])

    A class is made anew for every KerasFunction; an instance keeps its
    weights, shared wherever it is used. Weights of the Keras layer are Keras
    weights, trained with the model but not :class:`Parameter` objects: neither
    :meth:`Modely.freeze` nor :attr:`Modely.parameters` reaches them - set
    the layer's ``trainable`` instead.

    The model saves, loads and exports with it. The class is made known to
    Keras, so it loads again in the same session; in another, define it -
    decorated with ``@keras.saving.register_keras_serializable()`` - before
    loading. :class:`Fuzzify` wraps the Keras layers it is given in one.
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
        self.function = function
        super().__init__(name=name, function=function)
        # One Keras layer is one layer however many times it is wrapped: every
        # wrapper of an instance applies it, so they save and load as one.
        self._source = _WRAPPERS.setdefault(id(function), self._source)

    def build_layer(self):
        return KerasFunctionImpl(function=self.function, name=self.name)

    def get_config(self):
        # The Keras layer as it is declared: the shape it was first built for
        # is not part of it, the model builds it again.
        function = cast(dict, keras.saving.serialize_keras_object(self.function))
        return {
            "name": self.name,
            "function": {k: v for k, v in function.items() if k != "build_config"},
        }


# The first wrapper of every Keras layer still wrapped, by the layer's id: a
# live wrapper keeps its layer alive, so the id is not reused meanwhile.
_WRAPPERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _is_keras_layer(function) -> bool:
    return isinstance(function, keras.layers.Layer) or (
        isinstance(function, type) and issubclass(function, keras.layers.Layer)
    )
