"""Early stopping in Modely.train: Keras's EarlyStopping, or a callback of one's own."""

import keras
import numpy as np
import pytest

from nnodely import DataLoader, Input, Linear, Modely, Output
from conftest import to_numpy

# With a min_delta this large only the first epoch is an improvement, so a
# Keras EarlyStopping stops after exactly 1 + patience epochs.
NEVER_BETTER = 1e9


def _fit():
    """y = 2x, to fit with one linear layer under the minimizer "fit"."""
    x = Input("es_x")
    y = Output("es_y", Linear(out_features=1)(x.last()))
    model = Modely("early_stopping", inputs=[x], outputs=[y])
    model.minimize("fit", y, Input("es_target").last())
    model.build()
    values = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    data = DataLoader(model, source={"es_x": values, "es_target": 2.0 * values})
    return model, data


def test_without_early_stopping_every_epoch_runs():
    model, data = _fit()

    history = model.train(data, epochs=4, batch_size=16, printer=None)

    assert len(history["loss"]) == 4


def test_a_monitored_name_stops_with_keras_early_stopping():
    model, data = _fit()

    history = model.train(
        data,
        epochs=50,
        batch_size=16,
        printer=None,
        early_stopping="loss",
        early_stopping_kwargs={"patience": 2, "min_delta": NEVER_BETTER},
    )

    assert len(history["loss"]) == 3


def test_a_minimizer_validation_loss_can_be_monitored():
    model, data = _fit()

    history = model.train(
        data,
        val_data=data,
        epochs=50,
        batch_size=16,
        printer=None,
        early_stopping="val_fit_loss",
        early_stopping_kwargs={"patience": 1, "min_delta": NEVER_BETTER},
    )

    assert len(history["val_fit_loss"]) == 2


def test_a_configuration_builds_keras_early_stopping():
    model, data = _fit()

    history = model.train(
        data,
        epochs=50,
        batch_size=16,
        printer=None,
        early_stopping={
            "monitor": "fit_loss",
            "patience": 1,
            "min_delta": NEVER_BETTER,
        },
    )

    assert len(history["loss"]) == 2


def test_a_keras_early_stopping_instance_restores_the_best_weights():
    model, data = _fit()
    weights_per_epoch = []
    recorder = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: weights_per_epoch.append(
            [to_numpy(w) for w in model.model.get_weights()]  # type: ignore[union-attr]
        )
    )
    stopper = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=1,
        min_delta=NEVER_BETTER,  # type: ignore[arg-type]
        restore_best_weights=True,
    )

    history = model.train(
        data, epochs=50, batch_size=16, lr=0.1, printer=recorder, early_stopping=stopper
    )

    # Only the first epoch counted as an improvement: its weights come back.
    assert len(history["loss"]) == 2
    restored = [to_numpy(w) for w in model.model.get_weights()]  # type: ignore[union-attr]
    for weight, best, last in zip(restored, *weights_per_epoch):
        np.testing.assert_allclose(weight, best)
        assert not np.allclose(best, last)


class StopBelow(keras.callbacks.Callback):
    """A custom early stopping: stop once a logged loss falls below a level."""

    def __init__(self, monitor: str, threshold: float):
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if (logs or {})[self.monitor] < self.threshold:
            assert self.model is not None
            self.model.stop_training = True


def test_a_custom_callback_stops_training_when_it_says_so():
    model, data = _fit()

    history = model.train(
        data,
        epochs=500,
        batch_size=16,
        lr=0.05,
        printer=None,
        early_stopping=StopBelow("fit_loss", 1e-3),
    )

    losses = history["fit_loss"]
    assert len(losses) < 500
    assert losses[-1] < 1e-3
    assert all(loss >= 1e-3 for loss in losses[:-1])


def test_the_legacy_printer_shows_the_epoch_training_stopped_at(capsys):
    model, data = _fit()

    model.train(
        data,
        epochs=100,  # a row every 5 epochs
        batch_size=16,
        printer="legacy",
        early_stopping="loss",
        early_stopping_kwargs={"patience": 1, "min_delta": NEVER_BETTER},
    )

    assert "2/100" in capsys.readouterr().out


@pytest.mark.parametrize(
    "early_stopping, kwargs, error, message",
    [
        ("los", None, ValueError, "monitors 'los', which training does not log"),
        ("val_loss", None, ValueError, "Pass val_data to monitor a validation loss"),
        (
            keras.callbacks.EarlyStopping(),  # monitors val_loss by default
            None,
            ValueError,
            "Pass val_data",
        ),
        ({"monitor": "loss"}, {"patience": 2}, ValueError, "only be used when"),
        (None, {"patience": 2}, ValueError, "needs early_stopping"),
        (5, None, TypeError, "early_stopping must be"),
    ],
    ids=[
        "misspelled",
        "validation_without_val_data",
        "keras_default_without_val_data",
        "kwargs_with_a_configuration",
        "kwargs_alone",
        "not_a_callback",
    ],
)
def test_an_early_stopping_that_could_never_work_is_refused(
    early_stopping, kwargs, error, message
):
    # Keras itself only warns at every epoch about a quantity that is never
    # logged, and trains to the end.
    model, data = _fit()

    with pytest.raises(error, match=message):
        model.train(
            data,
            epochs=3,
            printer=None,
            early_stopping=early_stopping,
            early_stopping_kwargs=kwargs,
        )
