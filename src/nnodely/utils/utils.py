# This file contains utility functions for the nnodely library.
import keras
import numpy as np
from typing import Any, Callable

SUPPORTED_OPTIMIZERS = {
    "sgd",
    "rmsprop",
    "adam",
    "adamw",
    "adagrad",
    "adadelta",
    "adamax",
    "adafactor",
    "nadam",
    "ftrl",
    "lion",
}


# def _flatten_dict(d):
#     # Flatten a dictionary of the form {key: value} where value can be a string or a dict of the same form.
#     if not isinstance(d, dict):
#         return d

#     result = {}
#     for key, value in d.items():
#         if isinstance(value, dict):
#             nested = _flatten_dict(value)
#             for nested_key, nested_value in nested.items():
#                 result[nested_key] = nested_value
#         else:
#             result[key] = value
#     return result


def _resolve_loss(
    loss: str | dict[str, Any] | keras.losses.Loss | Callable,
) -> keras.losses.Loss | Callable:
    """Resolve any loss supported by the installed Keras version."""
    try:
        resolved = keras.losses.get(loss)
    except Exception as exc:
        raise ValueError(f"Unknown or invalid Keras loss {loss!r}.") from exc

    if not callable(resolved):
        raise TypeError("loss must resolve to a callable Keras loss.")
    return resolved


def _mask_padded_targets(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mark the padded rollout steps of a target array with NaN.

    The mask is one row per sample, one column per rollout step; the array it
    marks is [samples, *dim, time, *seq], whose last axis is that same rollout.
    """
    view = mask.reshape(mask.shape[0], *([1] * (values.ndim - 2)), mask.shape[1])
    return np.where(view, values, np.nan).astype(values.dtype)


@keras.saving.register_keras_serializable(package="nnodely")
class MaskedLoss(keras.losses.Loss):
    """Drop the padded target steps, marked with NaN, from a wrapped loss.

    Both sides are zeroed on a padded step, so it contributes exactly zero
    whatever the wrapped loss is. Zeroing the prediction too is what keeps the
    loss finite: a rollout that runs past the end of its simulation is driven by
    the model alone and can overflow, and copying that into the target would
    make the loss ``inf - inf``. The reduction still divides by the padded
    width, so every real step weighs the same across simulations of different
    lengths.

    It is a registered Loss rather than a closure because ``compile`` keeps it
    in the model state, which has to survive an export/import round trip.
    """

    def __init__(self, loss, name="masked_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss = keras.losses.get(loss)

    def call(self, y_true, y_pred):
        valid = keras.ops.logical_not(keras.ops.isnan(y_true))
        zero = keras.ops.zeros_like(y_pred)
        y_true = keras.ops.where(valid, y_true, zero)
        y_pred = keras.ops.where(valid, y_pred, zero)
        if callable(self.loss):
            return self.loss(y_true, y_pred)
        else:
            raise TypeError("Wrapped loss must be callable.")

    def get_config(self):
        config = super().get_config()
        config["loss"] = keras.losses.serialize(self.loss)
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["loss"] = keras.losses.deserialize(config["loss"])
        return cls(**config)


def _resolve_optimizer(
    optimizer: str | dict[str, Any] | keras.optimizers.Optimizer | None,
    learning_rate: float,
    optimizer_kwargs: dict[str, Any] | None,
) -> keras.optimizers.Optimizer:
    if optimizer is None:
        optimizer = "adam"

    if isinstance(optimizer, str):
        if optimizer.lower() not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unknown or invalid Keras optimizer {optimizer!r}. the following optimizers are supported: [{', '.join(SUPPORTED_OPTIMIZERS)}]"
            )
        config = keras.optimizers.serialize(keras.optimizers.get(optimizer))
        if type(config) is dict and "config" in config:
            config["config"].update(optimizer_kwargs or {})
            config["config"]["learning_rate"] = learning_rate
        return keras.optimizers.deserialize(config)

    if optimizer_kwargs:
        raise ValueError(
            "optimizer_kwargs can only be used when optimizer is a name. "
            "Configure optimizer instances or serialized configurations "
            "before passing them to train()."
        )

    try:
        resolved = keras.optimizers.get(optimizer)
    except Exception as exc:
        raise ValueError("Invalid Keras optimizer configuration or instance.") from exc

    if not isinstance(resolved, keras.optimizers.Optimizer):
        raise TypeError(
            "optimizer must resolve to an instance of keras.optimizers.Optimizer."
        )
    return resolved
