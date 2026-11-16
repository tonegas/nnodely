# Contributing

Thanks for your interest in **nnodely**! 🎉
We welcome contributions of all kinds — bug reports, documentation fixes,
examples, and new features.

## Contributing with an AI assistant

If you use an AI coding assistant (Claude Code, Cursor, Copilot, Codex, ...),
point it to [`AGENTS.md`](AGENTS.md) and the [`.agents/`](.agents/) folder.
They hold the rules an agent must follow to write features, tests and
documentation for nnodely. To *use* the library rather than change it, see
[`NNODELY_AI_GUIDE.md`](NNODELY_AI_GUIDE.md).

## Setup

All you need to do is:

- install [uv](https://docs.astral.sh/uv/)
- clone the repo
- run `uv sync --all-extras --dev`, which installs the three Keras backends,
  the ONNX tools and the development tools
- install the hooks `uv run pre-commit install --install-hooks`

You are now up and running, just make sure to run everything via `uv` (e.g.,
`uv run ...`). Always sync with `--all-extras --dev`: a plain `uv sync` removes
the backends, which are optional extras.

The backend is chosen with the `KERAS_BACKEND` environment variable
(`tensorflow`, `torch` or `jax`); the tests default to `tensorflow`.

### IDE

We have config files for the IDEs we use, so `VSCode`, `PyCharm`, and `neovim`
should be ready to go.

- `VSCode`: you can find a `.vscode` folder which should prompt you to install
  the necessary plugins and enable them as we expect them to be. Nonetheless
  know that it will use `PyLance` and not bare `pyright` so you will see
  different diagnostics. It should pick up the `uv` managed environment right
  away.
- `PyCharm`: enable `pyright` and `ruff` and you should be good to go. It
  should also pick up `uv` automatically.
- `neovim`: you have to add the `pyright` and `ruff` (e.g., using `mason`) and
  enable them (e.g., `vim.lsp.enable(...)`). To pick up `uv` just run `uv run
nvim .` in the project root.

## Tests

```bash
uv run pytest tests -m "not slow"                  # the fast suite, under a minute
uv run pytest tests                                # everything
KERAS_BACKEND=torch uv run pytest tests -m "not slow"
scripts/test_all_backends.sh                       # every backend in turn
```

- A test that takes 1 s or more is marked `@pytest.mark.slow`.
- Every behaviour change comes with a test; a bug fix with one that fails
  without it. Changes to documentation, comments or tooling alone need none.
- A test for one backend only is marked so it is skipped on the others (see
  `requires_onnx_export` in `tests/conftest.py`).

## Documentation

The documentation is built by Read the Docs with warnings as errors. To build
it locally the same way:

```bash
KERAS_BACKEND=jax uv run --with-requirements docs/requirements.txt \
    sphinx-build -W -b html docs docs/_build/html
```

## Code Style

This project follows a strict and largely automated Python code style. The goal
is to keep the codebase consistent, readable, and easy to review, while
minimising style-related discussion in PRs.

In short: **let the tools do the work**.

---

## General Principles

- Prefer clarity over cleverness.
- Keep functions and classes small and focused.
- Be consistent with existing code.

---

## Python Version

We support Python 3.10 and later (`requires-python` in `pyproject.toml` has no
upper bound). Pull requests to `main` are tested on 3.10 to 3.13 with the full
suite; pull requests to `develop` only on 3.13, with the fast suite (the full
suite runs only for coverage, on Ubuntu). Tests will
catch most of the version specific behaviour, but please keep it in mind.

---

## Linting, Formatting and Type Checking

This project uses `ruff` for both linting and formatting, and `pyright` for
type checking.

- All code **must** pass `ruff` and `pyright` checks.
- Formatting is enforced via `ruff format`.
- Do **not** manually fight the formatter.

The `pre-commit` hooks take care of this (they also run the whole test suite,
slow tests included, on every commit that touches Python files), but you can
run the same checks the CI runs manually with:

```bash
uv run ruff check --fix
uv run ruff format
uv run pyright
```

### Ignoring Rules and Formatting

Disabling rules or formatting should be rare and justified:

```python
from .layers import Fir  # noqa: F401  # re-exported as part of the public API

# fmt: off
table = [
    ("short",      1),
    ("muchlonger", 2),
]
# fmt: on
```

---

## Naming Conventions

We use:

- `snake_case` for functions and variables
- `CamelCase` for classes
- `UPPERCASE` for constants

---

## Type Hints

- Required unless impractical.
- Make custom types whenever your type gets too big, for example:

  ```python
  # This horrible mess
  list[dict[str, list[int]]]

  # Should become
  CustomType: TypeAlias = list[dict[str, list[int]]]
  ```

- Heavily prefer strong typing (e.g., `Enum` and `dataclass`), for example:

  ```python
  # Instead of this
  def func(flag: str) -> None:
    ...

  # Do this
  class Flag(Enum):
    ...

  def func(flag: Flag) -> None:
    ...
  ```

---

## Docstrings

We use the NumPy docstring style, which the documentation renders through
`sphinx.ext.napoleon`. An example:

```python
"""Short description of the function.

Parameters
----------
param:
    What the parameter is.

Returns
-------
What the function returns.
"""
```

Public classes and methods are documented: their docstrings are the API
reference.

We suggest **not** writing docstrings by hand, but using one of the many
editor plugins. Here are the ones we use for each editor.

> Types go in the type hints, which pyright checks. The reference pages do
> not show the hints (`autodoc_typehints = "none"` in `docs/conf.py`), so
> every entry of `Parameters` also states its type and default as plain
> text, `features : int, default 1`, saying the same as the hint.

### VSCode

We use the
[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
plugin, with the "numpy" docstring format.

### PyCharm

Simply enable the "NumPy" docstring format in "Settings > Python >
Tools > Integrated Tools".

### neovim

We use the [neogen](https://github.com/danymat/neogen) plugin, configured as such:

```lua
require("neogen").setup({
    snippet_engine = "nvim",
    languages = {
        python = {
            template = {
                annotation_convention = "numpydoc",
            },
        },
    },
})
```

---

## Comments

A comment explains why, never what: the code itself says what it does. Write
one only when the code alone cannot tell the reason, such as a Keras quirk, a
backend limit, a design choice, or the exclusion of a static checker rule.

---

## Branching

### Branching Model

- `main` is the default branch
  - Always stable.
  - Always releasable.
  - Protected (no direct commits).
- `develop` is the rolling branch
  - Put all development not quite stable yet.
  - Protected (no direct commits).
- All work happens on branches created from `develop`.
- A release is prepared on a `release/x.y.z` branch, merged into `main` and
  tagged `vx.y.z`; the tag publishes the package.

### Branch Naming

We follow [this](https://conventionalbranch.org).

---

## Commit Messages

We follow [this](https://www.conventionalcommits.org/en/v1.0.0/).

---

## GitHub Actions

Testing GitHub Actions is a pain, but it becomes easier if you test at least
some of their functionality with [act](https://github.com/nektos/act).

For example, to test the `codecov.yml` action just setup `act` and run: `act
pull_request --workflows .github/workflows/codecov.yml` (the event is needed,
since `act` simulates a `push` by default and the workflow runs on pull
requests only).
