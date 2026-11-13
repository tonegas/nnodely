"""The documentation runs as it is written.

The examples of the reference pages, the docstrings, run as doctests: what
they print is what the page shows. Every page of the guide runs from top to
bottom, its Python blocks one after the other, and a block followed by an
``output`` block prints that output.
"""

import contextlib
import doctest
import importlib
import io
import re
from pathlib import Path

import pytest

import nnodely

FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = [DOCS / "getting_started.rst", *sorted((DOCS / "guide").glob("*.rst"))]

# The subpackages have no __init__.py, which pkgutil would need to walk them.
SOURCE = Path(nnodely.__file__).parent
MODULES = sorted(
    ".".join(("nnodely", *path.relative_to(SOURCE).with_suffix("").parts))
    for path in SOURCE.rglob("*.py")
    if path.name != "__init__.py"
)


@pytest.mark.slow
@pytest.mark.parametrize("module", MODULES)
def test_the_examples_of_the_reference_pages_print_what_they_show(
    module, tmp_path, monkeypatch
):
    # Each docstring runs in a namespace of its own, empty: an example
    # imports what it uses, as a reader copying it has to.
    monkeypatch.chdir(tmp_path)
    tests = doctest.DocTestFinder(exclude_empty=True).find(
        importlib.import_module(module), module, globs={}
    )
    runner = doctest.DocTestRunner(optionflags=FLAGS)
    report = io.StringIO()
    for test in tests:
        runner.run(test, out=report.write)
    assert runner.failures == 0, report.getvalue()


_DIRECTIVE = re.compile(r"^(\s*)\.\. code-block:: (\w+)\s*$")


def _blocks(page: Path):
    """Every code block of a page: its language, options, code and line."""
    lines = page.read_text().splitlines()
    i = 0
    while i < len(lines):
        match = _DIRECTIVE.match(lines[i])
        if not match:
            i += 1
            continue
        indent, language, start = len(match.group(1)), match.group(2), i + 1
        i += 1
        options = []
        while i < len(lines) and lines[i].strip().startswith(":"):
            options.append(lines[i].strip())
            i += 1
        body = []
        while i < len(lines) and (
            not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
        ):
            body.append(lines[i])
            i += 1
        width = min(
            (len(line) - len(line.lstrip()) for line in body if line.strip()),
            default=0,
        )
        code = "\n".join(line[width:] for line in body).strip("\n")
        yield language, options, code, start


def _runnable(page: Path):
    """The Python blocks of a page, each with the output it shows, if any."""
    blocks = list(_blocks(page))
    for index, (language, options, code, line) in enumerate(blocks):
        if language != "python" or ":class: skip-test" in options:
            continue
        expected = None
        if index + 1 < len(blocks):
            _, next_options, next_code, _ = blocks[index + 1]
            if ":class: output" in next_options:
                expected = next_code + "\n"
        yield code, expected, line


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.stem)
def test_every_page_of_the_guide_runs_and_prints_what_it_shows(
    page, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__main__"}
    checker = doctest.OutputChecker()
    for code, expected, line in _runnable(page):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(code, f"{page.name}:{line}", "exec"), namespace)
        if expected is not None:
            got = printed.getvalue()
            assert checker.check_output(expected, got, FLAGS), (
                f"{page.name}:{line} prints\n{got}\nwhile the page shows\n{expected}"
            )


@pytest.mark.slow
def test_the_hello_world_of_the_readme_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert blocks
    for code in blocks:
        exec(compile(code, "README.md", "exec"), {"__name__": "__main__"})
