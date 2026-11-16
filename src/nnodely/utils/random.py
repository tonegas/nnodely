"""Global random-seed configuration for nnodely."""

from __future__ import annotations

import os
import random

import keras
import numpy as np


SEED_ENVIRONMENT_VARIABLE = "NNODELY_SEED"
_seed: int | None = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and the Keras backend, for reproducible runs.

    Syntax
    ------
    ``set_seed(seed)``
        Seeds every source of randomness nnodely draws from.

    It seeds the ``random`` module of Python, NumPy, and the backend.
    TensorFlow and PyTorch are seeded directly. On JAX, Keras draws its
    seeds from the Python generator. It also sets ``NNODELY_SEED``,
    so child processes inherit the seed. Call it before creating layers,
    Parameters and models: the initial weights, the shuffling of the data
    and the training are then the same from one run to the next.

    nnodely reads ``NNODELY_SEED`` when it is imported. Setting it before
    the import has the effect of calling ``set_seed`` with its value.

    Parameters
    ----------
    seed : int
        The seed, from 0 to ``2**32 - 1``.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``seed`` is not an int ("seed must be an integer"). A bool is
        refused too.
    ValueError
        If ``seed`` is out of range ("seed must be between 0 and
        2**32 - 1"), or if ``NNODELY_SEED`` holds no integer when nnodely
        is imported.

    See Also
    --------
    get_seed : The seed set last.
    Modely.train : The training, whose shuffling the seed fixes.

    Basic Examples
    --------------
    The same seed gives the same draws:

    >>> import numpy as np
    >>> from nnodely import get_seed, set_seed
    >>> set_seed(42)
    >>> first = np.random.rand(3)
    >>> set_seed(42)
    >>> bool(np.all(np.random.rand(3) == first))
    True
    >>> get_seed()
    42

    Properties & Relations
    ----------------------
    The seed is also in the environment of the process:

    >>> import os
    >>> os.environ["NNODELY_SEED"]
    '42'

    Possible Issues
    ---------------
    A seed is an integer:

    >>> set_seed(1.5)
    Traceback (most recent call last):
        ...
    TypeError: seed must be an integer, got float.

    .. versionadded:: 2.0
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
    """Return the seed set last by :func:`set_seed`, or ``None``.

    Syntax
    ------
    ``get_seed()``
        The seed in use.

    Returns
    -------
    int or None
        The seed of the last call of :func:`set_seed`, or the value of
        ``NNODELY_SEED`` when nnodely was imported. ``None`` when no seed was
        set: the run is then not reproducible.

    See Also
    --------
    set_seed : Seeds Python, NumPy and the backend.

    Basic Examples
    --------------
    >>> from nnodely import get_seed, set_seed
    >>> set_seed(7)
    >>> get_seed()
    7

    .. versionadded:: 2.0
    """
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
