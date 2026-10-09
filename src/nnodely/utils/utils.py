# This file contains utility functions for the nnodely library.
import math

import keras
from typing import Any, Callable, Collection, cast

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


def _serialized_initializer(initializer):
    """An initializer as a saved config can hold it: its name, or its config."""
    if isinstance(initializer, str):
        return initializer
    return keras.initializers.serialize(initializer)


def _per_slice_initializer(initializer, rank: int):
    """Draw every slice of the last ``rank`` axes from ``initializer`` as if
    it owned its own tensor.

    Variance scaling reads the fans from the whole tensor, so initialising
    ``[cells, features, out_features]`` in one shot would divide the scale by
    the number of cells: a cell only ever sees ``features`` inputs.
    """
    base = cast(keras.initializers.Initializer, keras.initializers.get(initializer))

    def initialize(shape, dtype=None):
        # A random initializer draws its seed once and then repeats itself, so
        # reusing one instance would give every slice the same values.
        config = base.get_config()
        slices = []
        for index in range(math.prod(shape[: len(shape) - rank])):
            slice_config = dict(config)
            if slice_config.get("seed") is not None:
                slice_config["seed"] = config["seed"] + index
            slices.append(
                type(base).from_config(slice_config)(
                    shape[len(shape) - rank :], dtype=dtype
                )
            )
        return keras.ops.reshape(keras.ops.stack(slices, axis=0), shape)

    return initialize


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


def _weighted_sequence_loss(loss, y_true, y_pred, weights):
    """Mean of ``loss`` with each step of the last axis scaled by ``weights``.

    A Keras loss reduces the last axis, which is the sequence the weights are
    laid along: a trailing axis of one is added for it to reduce instead, so
    every step keeps its own value. The weights have mean one, so uniform
    weights give back the unweighted loss.
    """
    if isinstance(loss, keras.losses.Loss):
        loss = loss.call  # one value per element, before the Loss reduces them
    values = loss(keras.ops.expand_dims(y_true, -1), keras.ops.expand_dims(y_pred, -1))
    return keras.ops.mean(values * keras.ops.cast(weights, values.dtype))


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


def _resolve_early_stopping(
    early_stopping: str | dict[str, Any] | keras.callbacks.Callback | None,
    early_stopping_kwargs: dict[str, Any] | None,
    logged: Collection[str],
) -> keras.callbacks.Callback | None:
    """Turn the ``early_stopping`` argument of :meth:`train` into a callback.

    ``logged`` holds the quantities training logs, the ones a Keras
    ``EarlyStopping`` can monitor.
    """
    if early_stopping is None:
        if early_stopping_kwargs:
            raise ValueError(
                "early_stopping_kwargs needs early_stopping to name the "
                "quantity to monitor."
            )
        return None

    if isinstance(early_stopping, str):
        callback = keras.callbacks.EarlyStopping(
            monitor=early_stopping, **(early_stopping_kwargs or {})
        )
    elif early_stopping_kwargs:
        raise ValueError(
            "early_stopping_kwargs can only be used when early_stopping is the "
            "name of the quantity to monitor. Configure callbacks and "
            "configurations before passing them to train()."
        )
    elif isinstance(early_stopping, dict):
        callback = keras.callbacks.EarlyStopping(**early_stopping)
    elif isinstance(early_stopping, keras.callbacks.Callback):
        callback = early_stopping
    else:
        raise TypeError(
            "early_stopping must be the name of a monitored quantity, a "
            "keras.callbacks.EarlyStopping configuration, or a "
            "keras.callbacks.Callback."
        )

    # Keras only warns, at every epoch, about a quantity that is never logged,
    # and never stops: a misspelled name, or a validation loss without val_data.
    if (
        isinstance(callback, keras.callbacks.EarlyStopping)
        and callback.monitor not in logged
    ):
        hint = (
            " Pass val_data to monitor a validation loss."
            if callback.monitor.startswith("val_")
            and not any(name.startswith("val_") for name in logged)
            else ""
        )
        raise ValueError(
            f"Early stopping monitors {callback.monitor!r}, which training "
            f"does not log: it logs {sorted(logged)}.{hint}"
        )
    # Every quantity training logs is a loss, but Keras infers a direction only
    # for the names "loss" and "val_loss": "fit_loss" would raise in "auto" mode.
    if isinstance(callback, keras.callbacks.EarlyStopping) and callback.mode == "auto":
        callback.mode = "min"
    return callback
