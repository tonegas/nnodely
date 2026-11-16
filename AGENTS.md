# AGENTS.md

Instructions for AI coding agents that **change** nnodely: its code, tests or
documentation. Humans start from [`CONTRIBUTING.md`](CONTRIBUTING.md); this
file is its digest for agents, and when the two disagree `CONTRIBUTING.md`
wins.

nnodely builds Model-Structured Neural Networks on Keras 3 (backends
TensorFlow, PyTorch, JAX; Python 3.10-3.13). A model is a graph of streams
declared in Python and compiled to a Keras model by `Modely.build()`.

## Read before working

| Task | Read |
| --- | --- |
| Any change to `src/`, above all one that reads, compares or assigns a name | [`.agents/architecture.md`](.agents/architecture.md) (rules on names and identity) |
| A new layer, or a change to one | [`.agents/new-layer.md`](.agents/new-layer.md) |
| Tests | [`.agents/testing.md`](.agents/testing.md) |
| Docstrings, guide, tutorials, changelog | [`.agents/documentation.md`](.agents/documentation.md) |

## How to work

- **Think before coding.** State your assumptions. If a request has several
  readings, present them rather than picking one silently. If something is
  unclear, stop, name what confuses you, and ask. If a simpler approach
  exists, say so, and push back when warranted.
- **Simplicity first.** Write the minimum code that solves the problem: no
  features, abstractions, options or error handling beyond what was asked or
  what can actually happen. If 200 lines could be 50, rewrite them; if a
  senior engineer would call it overcomplicated, simplify it.
- **Surgical changes.** Every changed line traces to the request. Do not
  improve adjacent code, comments or formatting, and do not refactor what
  works; match the existing style. Remove the imports, variables and
  functions your change left unused. Mention other dead code, do not delete
  it.
- **Goal-driven execution.** Turn the task into a check and loop until it
  passes: a bug becomes a test that reproduces it, a refactor keeps the tests
  passing before and after. For a multi-step task, state a short plan, each
  step with the check that verifies it.

## Commands

Always run through `uv`, and sync with `--all-extras --dev`: a plain
`uv sync` removes the backends.

```bash
uv sync --all-extras --dev                    # setup
uv run pytest tests -m "not slow"             # fast suite, under a minute
uv run pytest tests                           # full suite, with the doctests
KERAS_BACKEND=torch uv run pytest tests -m "not slow"   # another backend
uv run ruff check --fix && uv run ruff format # lint and format
uv run pyright                                # type check
```

Documentation build, as Read the Docs runs it (warnings are errors):

```bash
KERAS_BACKEND=jax uv run --with-requirements docs/requirements.txt \
    sphinx-build -W -b html docs docs/_build/html
```

## Definition of done

A change is finished when all of these hold:

1. `uv run ruff check`, `uv run ruff format --check` and `uv run pyright` pass.
2. `uv run pytest tests -m "not slow"` passes on the default backend
   (`tensorflow`). If the change touches backend-specific code (Keras ops,
   shapes, export), it also passes with `KERAS_BACKEND=torch` and `jax`.
3. Every behaviour change has a test; a bug fix has a test that fails
   without it.
4. A public class or method changed or added has its docstring updated, and
   its examples run as doctests (`tests/test_docs.py`, a slow test: run it
   with `uv run pytest tests/test_docs.py`).
5. A user-visible change has an entry in the topmost version section of
   [`CHANGELOG.md`](CHANGELOG.md), the release being prepared.

## Hard rules

- **Do not change the public API by accident.** The public names are those in
  `__all__` of `src/nnodely/__init__.py`. Renaming or removing one is a
  breaking change: say so in the changelog under `### Changed` or
  `### Removed`.
- **Do not break saved models silently.** If the saved format changes, bump
  `ModelSerializer.VERSION` in `src/nnodely/core/registry.py` and say so in the
  changelog ("The save format is version N").
- **Every feature works on every backend.** Write it with `keras.ops`
  whenever possible. Where Keras offers no common operation (gradients,
  recurrent loops, seeding: see `layers/gradient.py`), import `tensorflow`,
  `torch` or `jax` inside the function, with one branch per backend. A
  feature that cannot be implemented on a backend raises a clear error there,
  saying why and what to use instead (see the jax branch of `export_onnx` in
  `core/export.py`); its tests are skipped on that backend with a marker
  (see `requires_onnx_export` in `tests/conftest.py`), never deleted.
- **Error messages say what to do.** A `ValueError` or `TypeError` names the
  layer or stream, states what is wrong, and tells the user how to fix it.
  Follow the existing messages in `src/nnodely/core/layer.py`.
- **Comments explain why, not what.** Write one only when the code alone
  cannot tell the reason (a Keras quirk, a backend limit, a design choice).
- **No dashes as parentheses** (" - " inserting a clause mid-sentence) in
  comments, docstrings, error messages or docs: use a separate sentence, a
  list or brackets.
- **Type hints everywhere practical.** The reference pages do not show them,
  so each entry of a docstring's `Parameters` also states its type and
  default as text, `features : int, default 1`, saying the same as the hint.
- **Do not touch** `experimental/` (scratch area, ignored by tooling),
  `docs/api/generated/` (built), `uv.lock` by hand (run `uv lock`).
- **Do not add `__init__.py` to the subpackages** of `src/nnodely/`: they are
  namespace packages, and the docs test walks them by path.

## Git

- Branch from `develop` (protected, PR only), or from `main` for a
  `hotfix/`; names follow
  [Conventional Branch](https://conventionalbranch.org): `feat/`, `fix/`,
  `hotfix/`, `chore/`, then lowercase words joined by `-`
  (`feat/seq-reverse`). `main` receives only `release/x.y.z` and
  `hotfix/x.y.z` branches. The full model, with the merge method of each
  branch, is under "Branching" in `CONTRIBUTING.md`.
- Commit messages follow
  [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
  `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- End every commit an agent writes with a `Co-authored-by:` trailer that
  names the assistant.
- Never commit, push or open a PR unless the person you work with asks for
  it. Show the commit message first.
- The pre-commit hooks run ruff, pyright and the **whole** test suite on every
  commit that touches Python files: expect a commit to take minutes.
