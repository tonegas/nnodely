import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
# Tests must never select an interactive GUI backend. In particular, the
# macOS backend can abort a test process when validation creates a figure.
os.environ["MPLBACKEND"] = "Agg"

import pytest

from typing import Any, cast

import keras
import numpy.typing as npt
import nnodely


def to_numpy(value: Any) -> npt.NDArray[Any]:
    return cast(
        npt.NDArray[Any],
        keras.ops.convert_to_numpy(value),
    )


@pytest.fixture(autouse=True)
def seed():
    nnodely.set_seed(42)
