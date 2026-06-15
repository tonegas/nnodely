import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)
