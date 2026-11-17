"""What an installed nnodely carries, and what it says when it cannot run."""

import os
import re
import subprocess
import sys
from importlib.metadata import metadata, requires, version
from pathlib import Path

import pytest

import nnodely
from nnodely import Fir, Input, Modely, Output

_PACKAGE = Path(nnodely.__file__).parent


def test_the_version_is_the_installed_one():
    assert nnodely.__version__ == version("nnodely")
    # importlib's helpers are not part of the package's namespace.
    assert not hasattr(nnodely, "version")
    assert not hasattr(nnodely, "PackageNotFoundError")


def test_the_package_is_marked_as_typed():
    assert (_PACKAGE / "py.typed").is_file()


def _requirements(extra=None):
    """The requirements of the installed nnodely: its own, or an extra's."""
    marker = re.compile(r"extra\s*==\s*['\"]([\w-]+)['\"]")
    found = []
    for requirement in requires("nnodely") or []:
        match = marker.search(requirement)
        if (match.group(1) if match else None) == extra:
            found.append(requirement)
    return found


def test_every_installation_exports_and_runs_onnx():
    # Once an onnx extra of its own: ONNX export is part of every model, and
    # validate_onnx runs the file.
    names = [re.split(r"[<>=; ]", requirement)[0] for requirement in _requirements()]

    assert "onnx" in names and "onnxruntime" in names


@pytest.mark.parametrize(
    "backend, converter",
    [("tensorflow", "tf2onnx"), ("torch", "onnxscript"), ("jax", "jax2onnx")],
)
def test_every_backend_extra_brings_its_onnx_converter(backend, converter):
    # Once an extra apart, onnx and onnx-jax: nnodely[jax] could not export.
    assert any(r.startswith(converter) for r in _requirements(backend))


def test_keras_is_one_that_exports_onnx_from_every_backend():
    # Once capped below 3.14: the export now relies on its jax2onnx path and
    # on its torch exporter with fixed shapes.
    keras_requirement = next(
        requirement
        for requirement in requires("nnodely") or []
        if requirement.startswith("keras")
    )

    assert ">=3.15.1" in keras_requirement and "<" not in keras_requirement


def test_newer_pythons_are_not_turned_away():
    # Once capped: on a Python above the cap, pip fell back to an old release
    # that declared none.
    assert "<" not in metadata("nnodely")["Requires-Python"]


def test_the_html_logo_ships_with_the_package(tmp_path):
    # Once read from the repository's imgs/ folder: an installed copy had none.
    logo = _PACKAGE / "utils" / "templates" / "logo.png"
    x = Input("logo_x", tw=2)
    model = Modely("logo", inputs=[x], outputs=[Output("logo_out", Fir(1)([x]))])

    model.build().export_html(tmp_path)

    assert (tmp_path / "imgs" / "logo_info.png").read_bytes() == logo.read_bytes()
    assert 'src="./imgs/logo_info.png"' in (tmp_path / "logo.html").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_a_missing_backend_is_named_with_how_to_install_one(backend):
    # Once a bare "No module named 'tensorflow'" from inside Keras.
    code = f"import sys; sys.modules[{backend!r}] = None; import nnodely"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "KERAS_BACKEND": backend, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        encoding="utf-8",
    )

    assert result.returncode != 0
    assert f"backend needs {backend!r}, which is not installed" in result.stderr
    assert 'pip install "nnodely[tensorflow]"' in result.stderr
