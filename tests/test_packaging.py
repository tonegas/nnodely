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


def test_the_package_is_marked_as_typed():
    assert (_PACKAGE / "py.typed").is_file()


def test_the_onnx_extra_brings_the_tensorflow_exporter():
    # Once a development dependency only: Keras converts a TensorFlow graph
    # with tf2onnx, so nnodely[tensorflow,onnx] could not export ONNX.
    onnx_extra = [
        requirement
        for requirement in requires("nnodely") or []
        if re.search(r"extra\s*==\s*['\"]onnx['\"]", requirement)
    ]

    assert any(requirement.startswith("tf2onnx") for requirement in onnx_extra)


def test_newer_pythons_are_not_turned_away():
    # Once capped: on a Python above the cap, pip fell back to an old release
    # that declared none.
    assert "<" not in metadata("nnodely")["Requires-Python"]


def test_the_html_logo_ships_with_the_package(tmp_path):
    # Once read from the repository's imgs/ folder: an installed copy had none.
    logo = _PACKAGE / "utils" / "templates" / "logo.png"
    x = Input("logo_x")
    model = Modely("logo", inputs=[x], outputs=[Output("logo_out", Fir(1)([x.sw(2)]))])

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
