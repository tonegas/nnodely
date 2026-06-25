import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pytest

# def pytest_addoption(parser):
#     parser.addoption(
#         "--keras-backend",
#         action="store",
#         default="tensorflow",
#     )

# def pytest_sessionstart(session):
#     backend = (
#             session.config.getoption("--keras-backend")
#             or os.environ.get("KERAS_BACKEND")
#             or "tensorflow"
#     )

#     os.environ["KERAS_BACKEND"] = backend


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)
