# Code Style

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

This project uses **ruff** as the primary tool for both linting and formatting.

- All code **must** pass `ruff` checks.
- Formatting is enforced via `ruff format`.
- Do **not** manually fight the formatter.

A `pre-commit` hook will take care of this, but you can also run it manually with:

```bash
ruff check .
ruff format .
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

> Needs to be understood.

---

## Type Hints

- Always use type hints unless absolutely unfeasible.
- Make custom types whenever your type gets too big, for example:

  ```python
  # This horrible mess
  list[dict[str, list[int]]]

  # Should become
  CustomType = list[dict[str, list[int]]]
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

Docstrings and Comments
 • Use docstrings for public modules, classes, and functions
 • Follow the project’s configured docstring style
 • Comments should explain why, not what
 • Avoid obvious or redundant comments

---

## Pre-commit Hooks

This project uses pre-commit hooks to enforce good behaviour. You are expected
to install and run them locally:

```bash
pre-commit install
pre-commit run --all-files
```

Note that CI will reject code that does not pass pre-commit checks.
