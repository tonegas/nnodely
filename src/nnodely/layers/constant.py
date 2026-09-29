from nnodely.layers.parameter import ConstantImpl, _Value


class Constant(_Value):
    """
    Non-trainable symbolic constant layer.

    ``value`` sets its shape: a number or a vector is one time step of its
    ``dim``, a matrix is ``(dim, time)``, and further axes are ``seq`` axes.
    Numbers used in arithmetic with a stream become constants automatically.

    Like every stream it is laid out with a batch axis first, the same value
    for every sample::

        (batch, *dim, time, *seq)
    """

    _impl = ConstantImpl

    def __init__(self, name: str | None = None, *, value):
        if value is None:
            raise ValueError("Constant requires a value.")
        super().__init__(name, value, None, None, None, None)

    @property
    def constant(self):
        return self._variable

    def get_config(self):
        config = super().get_config()
        config.update({"value": self.value.tolist()})  # type: ignore
        return config

    @classmethod
    def from_config(cls, config: dict, preds=None):
        return cls(name=config["name"], value=config["value"])
