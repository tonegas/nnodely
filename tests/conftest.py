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


# Ends a training once its loss stops falling. Most trainings here reach what
# their test checks within a few dozen epochs, then spend the rest of their
# budget polishing a loss already far below any tolerance.
CONVERGED = {
    "early_stopping": "loss",
    "early_stopping_kwargs": {
        "patience": 10,
        "min_delta": 1e-10,
        "restore_best_weights": True,
    },
}


@pytest.fixture(autouse=True)
def torch_on_cpu():
    # Keras's torch backend picks the Apple GPU (MPS) whenever there is one,
    # and runs eagerly: each of the many small ops of a model is then its own
    # GPU kernel launch, which makes the suite about twice as slow as the CPU.
    if keras.backend.backend() != "torch":
        yield
        return
    with keras.device("cpu"):
        yield


# Keras reaches ONNX from jax through jax2tf, but jax 0.4.36 removed graph
# serialization, so jax2tf now always emits a single opaque XlaCallModule node
# that tf2onnx has no converter for. Nothing about the model matters here: the
# whole jax -> ONNX path is unavailable until Keras gains a native exporter.
requires_onnx_export = pytest.mark.skipif(
    keras.backend.backend() == "jax",
    reason="Keras cannot export ONNX from the jax backend: jax>=0.4.36 makes "
    "jax2tf emit an XlaCallModule node that tf2onnx cannot convert",
)
