"""
GetItem - Extract a value from a dictionary by key.
Useful for extracting specific outputs from layers that return dictionaries.
"""

import keras
from nnodely.core.layer import Layer


class GetItemImpl(keras.layers.Layer):
    """Keras layer that extracts a value from a dictionary."""

    def __init__(self, key, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.key = key

    def get_config(self):
        config = super().get_config()
        config.update({"key": self.key})
        return config

    def call(self, inputs):
        # inputs should be a dictionary
        if isinstance(inputs, dict):
            return inputs[self.key]
        else:
            # If it's not a dictionary, just return it as-is
            return inputs


class GetItem(Layer):
    """
    Layer that extracts a specific key from a dictionary returned by another layer.

    Usage:
        loop_output = loop([inputs])  # Returns dict
        pos_output = GetItem(key="Ypos_pred")(loop_output)
    """

    def __init__(self, key: str, name=None):
        self.key = key
        super().__init__(name=name, key=key)

    def build_layer(self):
        return GetItemImpl(self.key, name=self.name)

    def output_shape(self, *inputs):
        # Keep the same shape as the input
        return inputs[0].dimensions
