# Documentation

How to write and review the nnodely documentation: the Sphinx guide, the
docstrings, the tutorials, the README and the changelog.

The models are the **MATLAB** and **Wolfram Language** reference pages. Every
public name has a page with a fixed structure: the syntax at the top, the
arguments one by one, and runnable examples from the simplest case to the
edge cases. Two principles follow:

1. **A reference page stands on its own.** A reader arriving from a search
   understands the arguments without reading a guide.
2. **Examples are the core.** Every option has at least one example that
   uses it.

The documentation is built with warnings as errors and `nitpicky = True`:
every cross-reference must resolve. The docs **run**: docstring examples are
doctests, and guide pages are scripts (`tests/test_docs.py`, on every
backend).

## 1. Where things go

| Level | Purpose | Place | Question it answers |
| --- | --- | --- | --- |
| Getting started | A new user's first working model | `docs/getting_started.rst` | "How do I start?" |
| User guide | A concept or workflow on a concrete case | `docs/guide/*.rst` | "How do I do X?" |
| Reference | Every public name, exactly | docstrings, listed in `docs/api/*.rst` | "What does `Fir` take and return?" |
| Tutorials | A topic taught as a notebook | `docs/tutorial/NN_name.ipynb` | "Show me, step by step." |
| Glossary | One term, one definition | `docs/glossary.rst` | "What is a *rollout*?" |
| Changelog | What changed for the user | `CHANGELOG.md` | "What breaks if I upgrade?" |

Rules:

- A guide links to the reference pages for the details and does **not**
  copy their rules. Each fine-grained rule lives in one place only.
- Every public name (`nnodely.__all__`) has a reference page, listed in an
  `.. autosummary::` block of `docs/api/*.rst`. A name documented as public
  is importable from `nnodely`.

## 2. Reference pages (docstrings)

NumPy style, rendered by `sphinx.ext.napoleon`. The extra sections are
declared in `napoleon_custom_sections` of `docs/conf.py`. `Fir`
(`layers/fir.py`) and `Sin` (`layers/trigonometric.py`) are the canonical
examples: copy their layout.

Sections, in this order; leave out only those that make no sense for the
name. A new docstring follows the whole template. When you change a name
whose docstring does not follow it yet, update only the sections your change
touches, and mention the missing ones instead of rewriting them.

```
<One-line summary: what it returns, not how it is implemented.>

Syntax
------
``Fir(features=1)(x)``
    One sentence per calling form.

<Description: 1-3 short paragraphs.>

Parameters
----------
features : int, default 1
    What it does. Allowed values. Units.

Call arguments            (layers only: the streams of the call)
--------------
x : Stream
    Its required shape.

Returns
-------
Stream
    The shape rule in Shape(S, T, D) notation, with a numeric case.

Raises
------
ValueError
    When, with the error text abridged.

See Also
--------
Linear : One line.

Notes
-----
.. math:: the formula, the domain, the gradient.

Basic Examples
--------------
Scope
-----
Options
-------
Applications
------------
Properties & Relations
----------------------
Possible Issues
---------------

.. versionadded:: 2.0
```

Rules for the syntax:

- **The signature is visible** on the page, never `(*args, **kwargs)`. A
  constructor wrapped by the layer metaclass exposes it through
  `__signature__`.
- Layers have **two stages**, configuration and call: the syntax shows both,
  `Fir(features=2)(x)`. Every calling form is on its own line, with its own
  sentence.

Rules for `Parameters`:

- Every argument of the signature appears, `name` included, with its
  **type**, **default** and **allowed values**, as text (`int, default 1`).
- Constraints are a list, not one long sentence.
- The text says the same as the type hint: the pages do not show the hints
  (`autodoc_typehints = "none"`), so this text is the only type the reader
  sees.

Rules for `Returns`: always the shape rule in `Shape(S=..., T=..., D=...)`
notation, with a numeric case.

`See Also` has 2-5 entries. A change of behaviour adds
`.. versionchanged:: X.Y` with one line.

## 3. The example ladder

The example groups are top-level sections, in this order:

| Group | Content | Minimum |
| --- | --- | --- |
| Basic Examples | The most common use, 3-6 lines, output printed | 1 |
| Scope | Every kind of accepted input: with and without a window, `dim>1`, `seq`, a dynamic sequence | 1 per shape variant |
| Options | One example for each constructor argument | 1 per argument |
| Applications | A fragment of a physical model (mass-spring-damper, vehicle, pendulum) | 1 |
| Properties & Relations | Checkable identities: `IntegrateStep` inverts `DerivativeStep`; `Sin(x)` is `Sin()(x)` | 1 if a relation exists |
| Possible Issues | Wrong use, the error shown, the fix | 1 per frequent error |

Rules for every example:

1. **Self-contained.** Each docstring runs in an empty namespace: it imports
   everything it uses (`keras`, `numpy as np`, the nnodely names).
2. **Visible output.** Doctest style, `>>>`, with what it prints.
   `ELLIPSIS` and `NORMALIZE_WHITESPACE` are on.
3. **Numbers checkable by hand** in basic examples: windows of 2-3 samples,
   known kernels.
4. **No hidden state.** An example that changes state (freeze, assign) does
   not leave the following ones depending on it.
5. **Converted results.** `keras.ops.convert_to_numpy(...)` before printing,
   so the output shows arrays, not backend tensors.
6. **Files** go to the current directory, which the test sets to a temporary
   one.

## 4. User guide pages

Every page of `docs/guide/` follows this plan (stated in
`docs/guide/index.rst`):

1. Title: a task ("Training a model"), not a class name.
2. Goal in 2 lines: what the reader can do at the end.
3. Prerequisites: the pages to read first.
4. The example system: equations and data, one per page, reused in every
   section.
5. Numbered steps, each: short explanation, code, output, comment on the
   result.
6. "Common errors": a table of error, cause, fix.
7. "See also": the reference pages used on the page.

Rules:

- Code is `.. code-block:: python`. The output it prints follows as
  `.. code-block:: text` with `:class: output`, and must match it, compared
  as a doctest (`...` and whitespace differences allowed).
- The blocks of a page run in order, as one script, on every backend. A block
  that must not run (it needs ONNX, or shows an error) gets
  `:class: skip-test`, and the page says why.
- Summary tables (layers, metrics, export support) link to the reference
  page of each entry.

## 5. Tutorials

Notebooks in `docs/tutorial/`, shown with their saved outputs: the build does
not run them. After changing a notebook, or the behaviour it shows (a printed
key, a file name), run it from top to bottom and save it with its outputs.
If you cannot run it, say so in the change.

## 6. Changelog

[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries go in the
topmost version section, the release being prepared (all the work of a
release branch is released together, so there is no `[Unreleased]`), under
`### Added`, `### Changed`, `### Fixed` or `### Removed`. An entry describes
the change from the previous release as the user sees it: a fix to something
new in the same release is no entry, its entry simply describes the right
behaviour. Each entry starts with its subject in bold (the API name, or a
short headline), then says what changed for the user and what to do about
it. A change to the save format says "The save
format is version N".

## 7. Writing style

The same voice in guides, docstrings, error messages and the changelog.

1. **One sentence, one idea.** At most about 25 words. Split chains of
   subordinate clauses ("..., which..., as..., so...").
2. **No dashes as parentheses** to insert a clause mid-sentence ("a value -
   a number, a Parameter - applies"): use a list or a separate sentence.
   This holds everywhere, code comments and error messages included. A dash
   separating a title from its subtitle, or a label from its description
   ("nnodely validation - history"), is no aside and stays.
3. **The rule first, then the exception**, each in its own sentence.
4. **Present tense, second person for instructions**: "Call `build()` before
   creating the DataLoader."
5. **Lists** for options, constraints and allowed values; **tables** for
   comparisons (Loop vs ModelyLoop vs Ode).
6. **Formulas** in `.. math::` for every numerical operation (Fir,
   IntegrateStep, DerivativeStep, metrics).
7. **No implementation details** on user pages: no internal classes
   (`FirImpl`, `FuzzifyImpl`) or private constants, unless the user sees or
   uses them.

## 8. One terminology

One concept, one term, everywhere: guides, docstrings, error messages and
key names. The definitions are in `docs/glossary.rst`; use its terms and add
one there before using it.

| Term | Meaning | Not to be used as a synonym |
| --- | --- | --- |
| stream | A node of the graph: an Input, the result of a layer, a Parameter, a Constant | node, tensor, relation |
| window (`tw`, `T`) | The consecutive samples a stream reads | time, time axis |
| sequence (`seq`, `S`) | The axes outside the window; the outermost one is the rollout's | trajectory, horizon (only for the length) |
| rollout | Repeated execution of a step along the outermost sequence axis | loop, unroll |
| state | An Input that receives the value of the previous step through `feedback()` | callback, closed input |
| objective | A term registered with `minimize()`; its name is the one in the history | loss (the function an objective uses) |
| block | A Modely called on streams inside another model | sub-model, model call |
| simulation | One contiguous recording in the DataLoader | run, file, episode |

Where an API name uses another word (`minimize`, `minimizers` for
objectives), the page says so once and then uses the chosen term.

## 9. Building

```bash
KERAS_BACKEND=jax uv run --with-requirements docs/requirements.txt \
    sphinx-build -W -b html docs docs/_build/html
uv run pytest tests/test_docs.py      # doctests and guide pages
```

## 10. Review checklist

For every reference page:

- [ ] First sentence says what it returns.
- [ ] Explicit signature, no `*args, **kwargs`.
- [ ] Every calling form under `Syntax`.
- [ ] `Parameters` with type, default and allowed values for every argument.
- [ ] `Returns` with the shape rule.
- [ ] `Raises` for the documented errors.
- [ ] At least one `Basic Examples` with output, and one example per option.
- [ ] `Possible Issues` for the frequent errors.
- [ ] `See Also`.
- [ ] `.. versionadded::`.
- [ ] The docs build passes with `-W` (no broken references, no napoleon
      warnings).

For every guide page:

- [ ] Runs from top to bottom in `tests/test_docs.py` on every backend.
- [ ] No example leaves state that changes the following ones.
- [ ] Printed outputs match the produced ones.
- [ ] Glossary terms used consistently.

Across documents:

- [ ] README, guide and docstrings give the same signatures, defaults and
      lists (EquationLearner functions, optimizers,
      integration methods).
- [ ] Every name presented as public is importable from `nnodely`.
