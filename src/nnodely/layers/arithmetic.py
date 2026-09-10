"""Arithmetic layers for nnodely."""

import keras

from nnodely.core.layer import Layer

@keras.saving.register_keras_serializable(package="nnodely")
class ArithmeticImpl(keras.layers.Layer):
    def __init__(self, operation: str, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.operation = operation

    def call(self, inputs):
        return getattr(keras.ops, self.operation)(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"operation": self.operation})
        return config


class Arithmetic(Layer):
    operation = ""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return ArithmeticImpl(operation=self.operation, name=self.name)

    def get_config(self):
        return {"name": self.name}


class Exp(Arithmetic):
    """Wrapper for exponential transform."""

    operation = "exp"
    
class Log(Arithmetic):
    """Wrapper for natural logarithm transform."""

    operation = "log"

class Log10(Arithmetic):
    """Wrapper for base 10 logarithm transform."""

    operation = "log10"

class Sqrt(Arithmetic):
    """Wrapper for square root transform."""

    operation = "sqrt"

class Abs(Arithmetic):
    """Wrapper for absolute value transform."""

    operation = "abs"

class Floor(Arithmetic):
    """Wrapper for floor transform."""

    operation = "floor"

class Ceil(Arithmetic):
    """Wrapper for ceiling transform."""

    operation = "ceil"

## Note: The following class is commented out, as it is not currently in Keras ops.
# .
# class Rad2Deg(Arithmetic):
#     """Wrapper for radians to degrees transform."""

#     operation = "rad2deg"

class Deg2Rad(Arithmetic):
    """Wrapper for degrees to radians transform."""

    operation = "deg2rad"

class Sign(Arithmetic):
    """Wrapper for sign transform."""

    operation = "sign"
