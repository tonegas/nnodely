# Contributing

Thanks for your interest in **nnodely**! 🎉
We're just getting started and welcome contributions of all kinds — bug
reports, documentation fixes, examples, and new features.

## Setup

All you need to do is:

- install [uv](https://docs.astral.sh/uv/)
- clone the repo
- run `uv sync --dev`
- install the hooks `uv run pre-commit install --install-hooks`

You are now up and running, just make sure to run everything via `uv` (e.g.,
`uv run ...`).

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
nvim .` in the project root or install the `venv-selector` plugin.

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

We target all the [supported Python
versions](https://devguide.python.org/versions/). Tests will catch most of the
version specific behaviour, but please keep it in mind.

---

## Linting and Formatting

This project uses `ruff` for both linting and formatting.

- All code **must** pass `ruff` checks.
- Formatting is enforced via `ruff format`.
- Do **not** manually fight the formatter.

A `pre-commit` hook will take care of this, but you can also run it manually with:

```bash
uv run ruff check --fix
uv run ruff format
```

### Ignoring Rules and Formatting

Disabling rules or formatting should be rare and justified:

```python
value = legacy_call()  # noqa: PLW0603  # required by external API

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

We use the `reST` docstring style, an example of which is:

```python
"""Example reST docstring

:param param: a parameter
:return: what the function returns
"""
```

I strongly suggest you to **not** write docstrings by hand, but rather use one
of the many editor plugins. Hereafter some examples for each editor we use.

> We do not put types in the docstrings! You **have** to put type hints
> whenever you can.

### VSCode

We use the
[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
plugin, with the "sphinx-notypes" docstring format.

### PyCharm

Simply enable the "reStructuredText" docstring format in "Settings > Python >
Tools > Integrated Tools".

### neovim

We use the [neogen](https://github.com/danymat/neogen) plugin, configured as such:

```lua
require("neogen").setup({
    snippet_engine = "nvim",
    languages = {
        python = {
            template = {
                annotation_convention = "reST",
            },
        },
    },
})
```

---

## Comments

Please refrain from writing useless comments. As a rule of thumb, you should
always be able to understand code without needing comments. As such, write
comments only when absolutely necessary, like to explain a particular exclusion
of a static checker rule.

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

### Branch Naming

We follow [this](https://conventional-branch.github.io).

---

## Commit Messages

We follow [this](https://www.conventionalcommits.org/en/v1.0.0/).

---

## GitHub Actions

Testing GitHub Actions is a pain, but it becomes easier if you test at least
some of their functionality with [act](https://github.com/nektos/act).

For example, to test the `codecov.xml` action just setup `act` and run: `act
--workflows .github/workflows/codecov.yml`.
