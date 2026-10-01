import importlib
import os
import random

import numpy as np
import pytest

from conftest import to_numpy
import nnodely
from nnodely import Input, Linear, Modely, Output


def _initialized_linear_weights(seed: int, suffix: str):
    nnodely.set_seed(seed)
    x = Input(f"seed_x_{suffix}", dim=3)
    linear = Linear(out_features=2, name=f"seed_linear_{suffix}")(x.last())
    Modely(
        f"seed_model_{suffix}",
        inputs=[x],
        outputs=[Output(f"seed_out_{suffix}", linear)],
    ).build()
    return to_numpy(linear.kernel), to_numpy(linear.bias)


def test_set_seed_repeats_python_numpy_and_keras_initialization():
    nnodely.set_seed(1234)
    python_values = [random.random() for _ in range(3)]
    numpy_values = np.random.random(3)
    first_kernel, first_bias = _initialized_linear_weights(1234, "first")

    nnodely.set_seed(1234)
    assert [random.random() for _ in range(3)] == python_values
    np.testing.assert_array_equal(np.random.random(3), numpy_values)
    second_kernel, second_bias = _initialized_linear_weights(1234, "second")

    np.testing.assert_array_equal(second_kernel, first_kernel)
    np.testing.assert_array_equal(second_bias, first_bias)
    assert nnodely.get_seed() == 1234
    assert os.environ["NNODELY_SEED"] == "1234"


def test_different_seeds_change_keras_initialization():
    first_kernel, _ = _initialized_linear_weights(100, "different_first")
    second_kernel, _ = _initialized_linear_weights(101, "different_second")

    assert not np.array_equal(first_kernel, second_kernel)


@pytest.mark.parametrize("seed", [True, 1.5, "42", None])
def test_set_seed_rejects_non_integer_values(seed):
    with pytest.raises(TypeError, match="seed must be an integer"):
        nnodely.set_seed(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_set_seed_rejects_values_outside_common_backend_range(seed):
    with pytest.raises(ValueError, match=r"between 0 and 2\*\*32 - 1"):
        nnodely.set_seed(seed)


def test_environment_seed_is_applied_when_nnodely_is_imported(monkeypatch):
    monkeypatch.setenv("NNODELY_SEED", "2468")
    random_config = importlib.import_module("nnodely.utils.random")

    importlib.reload(random_config)

    assert random_config.get_seed() == 2468
