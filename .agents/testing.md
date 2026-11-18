# Testing

## Running

```bash
uv run pytest tests -m "not slow"                       # fast suite
uv run pytest tests/test_layers.py -k atan2              # one area
uv run pytest tests                                      # everything
KERAS_BACKEND=jax uv run pytest tests -m "not slow"      # another backend
```

`tests/conftest.py` sets `KERAS_BACKEND=tensorflow` when it is not set, and
keeps the torch backend on the CPU.

## Writing tests

- **Every behaviour change comes with a test**; a bug fix with one that fails
  without the fix. Changes to documentation, comments or tooling alone need
  none. Run it before the fix to see it fail.
- **Name the test after the behaviour**, as a sentence:
  `test_atan2_gives_the_angle_of_the_point_in_every_quadrant`, not
  `test_atan2_2`.
- **Put it with its kin**: one file per area (`test_layers.py`,
  `test_save_export.py`, `test_dataset.py`, ...). Add a
  file only for a new area. Tests of names, clashes and renames go in
  `test_name_tag.py`.
- **Check values, not only shapes**: compare against NumPy with
  `np.testing.assert_allclose`.
- **Files go in `tmp_path`**, never in the working directory.
- **Mark slow tests**: a test that takes 1 s or more gets
  `@pytest.mark.slow`. Check with `--durations=10`.
- **Backend limits are skips, not deletions**: use a `skipif` marker like
  the one of `test_train_with_loop_under_xla` in `tests/test_train.py`, with
  a reason that says why the backend cannot do it.
- Keep a test independent of the others: no shared state, no order.

## What the suite also checks

- `tests/test_docs.py` (slow) runs every docstring example as a doctest,
  every code block of `docs/getting_started.rst` and `docs/guide/*.rst` from
  top to bottom, and the hello world of the README. Changing what something
  prints means updating those examples.
- `tests/test_packaging.py` checks what an installed nnodely ships with.

## CI

| Pull request to | Runs |
| --- | --- |
| `develop` | fast suite, Python 3.13, 3 OSes x 3 backends; ruff, pyright |
| `main` | full suite, Python 3.10-3.13, 3 OSes x 3 backends; ruff, pyright |

Both also run coverage on Ubuntu, one job per backend: the fast suite for
`develop`, the full suite for `main`.
