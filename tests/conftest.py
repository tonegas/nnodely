import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

from typing import Any, cast

import keras
import numpy.typing as npt


def to_numpy(value: Any) -> npt.NDArray[Any]:
    return cast(
        npt.NDArray[Any],
        keras.ops.convert_to_numpy(value),
    )


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)
