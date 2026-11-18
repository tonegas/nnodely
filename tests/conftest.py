import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
# Tests must never select an interactive GUI backend. In particular, the
# macOS backend can abort a test process when validation creates a figure.
os.environ["MPLBACKEND"] = "Agg"
# What a model prints (its tables) is plain text, whatever runs the tests.
os.environ["NO_COLOR"] = "1"

import pytest

from typing import Any, cast

# Before Keras: nnodely chooses the device, the CPU, which the torch backend
# reads when Keras imports it.
import nnodely
import keras
import numpy.typing as npt


def to_numpy(value: Any) -> npt.NDArray[Any]:
    return cast(
        npt.NDArray[Any],
        keras.ops.convert_to_numpy(value),
    )


@pytest.fixture(autouse=True)
def seed():
    nnodely.set_seed(42)


# Ends a training once its loss stops falling. Most trainings here reach what
# their test checks within a few dozen epochs, then spend the rest of their
# budget polishing a loss already far below any tolerance.
CONVERGED: dict[str, Any] = {
    "early_stopping": {
        "monitor": "loss",
        "patience": 10,
        "min_delta": 1e-10,
        "restore_best_weights": True,
    },
}
