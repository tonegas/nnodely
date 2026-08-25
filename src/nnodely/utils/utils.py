# This file contains utility functions for the nnodely library.
import keras
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
