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
    # run_onnx runs the file.
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


def _import_nnodely(
    tmp_path, missing=(), backend=None, config=None, env=None, before=""
):
    """Import nnodely in a new process where the backends ``missing`` are not
    installed, with ``KERAS_BACKEND`` set to ``backend`` (unset for None), a
    Keras configuration of its own, whose backend is ``config``, the variables
    ``env`` and the code ``before`` run first: what it prints, the backend and
    the device it runs on, or its error. The device variables are unset but
    those ``env`` sets."""
    home = tmp_path / "keras_home"
    home.mkdir(parents=True)
    if config is not None:
        (home / "keras.json").write_text(f'{{"backend": "{config}"}}')
    variables = {**os.environ, "KERAS_HOME": str(home), "PYTHONIOENCODING": "utf-8"}
    for name in (
        "KERAS_BACKEND",
        "NNODELY_DEVICE",
        "KERAS_TORCH_DEVICE",
        "JAX_PLATFORMS",
    ):
        variables.pop(name, None)
    if backend is not None:
        variables["KERAS_BACKEND"] = backend
    variables.update(env or {})
    code = (
        f"import sys; sys.modules.update(dict.fromkeys({list(missing)!r})); {before}"
        "import nnodely; print(nnodely.get_backend(), nnodely.get_device())"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=variables,
        capture_output=True,
        encoding="utf-8",
    )


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_a_missing_backend_is_named_with_how_to_install_one(tmp_path, backend):
    # Once a bare "No module named 'tensorflow'" from inside Keras, then a
    # message that listed every backend without saying which was chosen.
    result = _import_nnodely(tmp_path, missing=[backend], backend=backend)

    assert result.returncode != 0
    assert (
        f"{backend!r} backend (set by KERAS_BACKEND) needs {backend!r}" in result.stderr
    )
    assert f'pip install "nnodely[{backend}]"' in result.stderr
    # The others are installed here: the error names them.
    others = [name for name in ("tensorflow", "torch", "jax") if name != backend]
    assert f"run on an installed one, {others}" in result.stderr


def test_without_any_backend_the_error_says_how_to_install_one(tmp_path):
    result = _import_nnodely(tmp_path, missing=["tensorflow", "torch", "jax"])

    assert result.returncode != 0
    assert "'tensorflow' backend (the default) needs 'tensorflow'" in result.stderr
    assert 'pip install "nnodely[tensorflow]"' in result.stderr
    assert "choose one with the KERAS_BACKEND environment variable" in result.stderr


@pytest.mark.slow
@pytest.mark.parametrize(
    "missing, config, expected",
    [
        ([], None, "tensorflow"),  # the default of Keras
        ([], "jax", "jax"),  # the backend of its configuration file
        (["tensorflow"], None, "torch"),  # the default missing: the first installed
        (["tensorflow", "torch"], None, "jax"),  # nnodely[jax] alone
        (["jax"], "jax", "tensorflow"),  # the configured one missing
    ],
)
def test_a_backend_not_installed_gives_way_to_an_installed_one(
    tmp_path, missing, config, expected
):
    # Once the default, tensorflow, even where only jax was installed: the
    # import failed, telling to install a backend.
    result = _import_nnodely(tmp_path, missing=missing, config=config)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2] == expected


@pytest.mark.slow
def test_keras_backend_is_the_users_choice_over_the_configuration(tmp_path):
    result = _import_nnodely(tmp_path, backend="jax", config="torch")

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2] == "jax"


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
@pytest.mark.parametrize("device", [None, "cpu", " CPU "])
def test_nnodely_computes_on_the_cpu_unless_asked_for_the_gpu(
    tmp_path, backend, device
):
    # Once the Apple GPU on torch, where a model of nnodely's size trains up
    # to four times slower than on the CPU, and any GPU on the others.
    env = {} if device is None else {"NNODELY_DEVICE": device}
    result = _import_nnodely(tmp_path, backend=backend, env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == [backend, "cpu"]


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_nnodely_device_gpu_takes_the_accelerator_or_says_how_to_have_one(
    tmp_path, backend
):
    result = _import_nnodely(tmp_path, backend=backend, env={"NNODELY_DEVICE": "gpu"})

    # Whether this machine has a GPU for the backend decides which.
    if result.returncode == 0:
        assert result.stdout.split()[-1] != "cpu"
    else:
        assert f"NNODELY_DEVICE=gpu, but the {backend!r} backend has no GPU here" in (
            result.stderr
        )
        assert "Unset NNODELY_DEVICE to compute on the CPU" in result.stderr


@pytest.mark.slow
def test_a_device_given_to_the_backend_is_kept_unless_nnodely_device_is_set(tmp_path):
    # "meta", a device of torch on every machine, stands for one the user
    # gives it, such as "cuda:1".
    kept = _import_nnodely(
        tmp_path / "kept", backend="torch", env={"KERAS_TORCH_DEVICE": "meta"}
    )
    overridden = _import_nnodely(
        tmp_path / "overridden",
        backend="torch",
        env={"KERAS_TORCH_DEVICE": "meta", "NNODELY_DEVICE": "cpu"},
    )

    assert kept.returncode == 0 and overridden.returncode == 0, kept.stderr
    assert kept.stdout.split()[-1] == "meta"
    assert overridden.stdout.split()[-1] == "cpu"


def test_an_unknown_device_is_refused(tmp_path):
    result = _import_nnodely(tmp_path, env={"NNODELY_DEVICE": "cuda"})

    assert result.returncode != 0
    assert "NNODELY_DEVICE is 'cpu' or 'gpu', got 'cuda'" in result.stderr


@pytest.mark.slow
def test_a_device_keras_chose_before_nnodely_is_named_in_a_warning(tmp_path):
    # Keras imported first reads its device then: nnodely cannot move it.
    result = _import_nnodely(
        tmp_path,
        backend="torch",
        env={"KERAS_TORCH_DEVICE": "meta", "NNODELY_DEVICE": "cpu"},
        before="import keras; ",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-1] == "meta"
    assert "nnodely computes on 'meta', not on the CPU" in result.stderr
    assert "set KERAS_TORCH_DEVICE=cpu before the backend starts" in result.stderr
