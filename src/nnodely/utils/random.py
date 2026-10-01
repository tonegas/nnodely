"""Global random-seed configuration for nnodely."""

from __future__ import annotations

import os
import random

import keras
import numpy as np


SEED_ENVIRONMENT_VARIABLE = "NNODELY_SEED"
_seed: int | None = None


def set_seed(seed: int) -> None:
    """Set the random seed used by nnodely and the active Keras backend.

    Python, NumPy, and the selected backend are seeded together. The value is
    also stored in ``NNODELY_SEED`` so child processes inherit the same
    configuration.

    Call this before constructing layers or models when reproducible weight
    initialization and training are required.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}.")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1.")

    global _seed
    _seed = seed
    os.environ[SEED_ENVIRONMENT_VARIABLE] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    backend = keras.backend.backend()
    if backend == "tensorflow":
        import tensorflow

        tensorflow.random.set_seed(seed)
    elif backend == "torch":
        import torch

        torch.manual_seed(seed)


def get_seed() -> int | None:
    """Return the seed configured through nnodely, if any."""
    return _seed


def _configure_seed_from_environment() -> None:
    value = os.environ.get(SEED_ENVIRONMENT_VARIABLE)
    if value is None:
        return

    try:
        seed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{SEED_ENVIRONMENT_VARIABLE} must contain an integer, got {value!r}."
        ) from exc
    set_seed(seed)


_configure_seed_from_environment()
