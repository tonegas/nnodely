import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Tuple
import unittest

import torch
from torch import Tensor, nn


def deterministic_initializer(indexes, params_size, dict_param):
    scale = float(dict_param.get("scale", 0.03))
    value = 0
    for axis, index in enumerate(indexes):
        value += (axis + 3) * (index + 5) * 1103515245
    normalized = float((value + 12345) % 4096) / 2048.0 - 1.0
    return scale * normalized


def zero_initializer(indexes, params_size, dict_param):
    return 0.0


def probe_function(history_signal, current_signal, gain, bias):
    return gain * torch.tanh(history_signal + current_signal) + bias


def build_probe_model(history_len: int, hidden_dim: int, sample_time: float):
    from nnodely import Concatenate, Constant, Fir, Input, Linear, Modely, Output, Parameter, ParamFun, Relu, Select, Tanh, clearNames

    clearNames()
    measurement = Input("measurement")
    command = Input("command")
    encoded_measurement = Relu(
        Fir(
            hidden_dim,
            b=True,
            W_init=deterministic_initializer,
            W_init_params={"scale": 0.05},
            b_init=zero_initializer,
            b_init_params={},
        )(measurement.sw(history_len))
    )
    encoded_command = Relu(
        Linear(
            hidden_dim,
            b=True,
            W_init=deterministic_initializer,
            W_init_params={"scale": 0.04},
            b_init=zero_initializer,
            b_init_params={},
        )(command.last())
    )
    features = Concatenate(encoded_measurement, encoded_command)
    hidden = Relu(
        Linear(
            hidden_dim,
            b=True,
            W_init=deterministic_initializer,
            W_init_params={"scale": 0.03},
            b_init=zero_initializer,
            b_init_params={},
        )(features)
    )
    raw = Linear(
        2,
        b=True,
        W_init=deterministic_initializer,
        W_init_params={"scale": 0.02},
        b_init=zero_initializer,
        b_init_params={},
    )(hidden)
    bounded = Tanh(Select(raw, 0))
    residual = ParamFun(
        probe_function,
        parameters_and_constants=[
            Parameter("probe_gain", values=0.7),
            Constant("probe_bias", values=0.05),
        ],
    )(bounded, Select(raw, 1))
    model = Modely(visualizer=None)
    model.addModel(
        "native_torchscript_probe",
        [
            Output("prediction", residual),
            Output("auxiliary", Select(raw, 1)),
        ],
    )
    model.neuralizeModel(sample_time=sample_time)
    return model


def export_probe_model(export_dir: Path, name: str, history_len: int, hidden_dim: int, sample_time: float) -> Path:
    model_dir = export_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model = build_probe_model(history_len, hidden_dim, sample_time)
    model.exportPythonModel(name=name, model_folder=str(model_dir))
    return model_dir / f"{name}.py"


def load_tracer_model(path: Path) -> nn.Module:
    module_name = f"nnodely_native_torchscript_probe_{os.getpid()}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exported model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TracerModel()


def make_inputs(batch_size: int, history_len: int, requires_grad: bool) -> Dict[str, Tensor]:
    measurement = torch.randn(batch_size, history_len, 1)
    command = torch.randn(batch_size, history_len, 1)
    if requires_grad:
        measurement.requires_grad_()
        command.requires_grad_()
    return {"measurement": measurement, "command": command}


def output_dict(result: Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return result[0]


def backward_probe(module: nn.Module, inputs: Dict[str, Tensor]) -> float:
    module.zero_grad(set_to_none=True)
    outputs = output_dict(module(inputs))
    loss = outputs["prediction"].pow(2).mean() + 0.1 * outputs["auxiliary"].pow(2).mean()
    loss.backward()
    if not any(parameter.grad is not None for parameter in module.parameters()):
        raise RuntimeError("no parameter gradients found")
    if not all(tensor.grad is not None for tensor in inputs.values()):
        raise RuntimeError("missing input gradient")
    return float(loss.detach())


def max_output_difference(eager: nn.Module, scripted: nn.Module, inputs: Dict[str, Tensor]) -> float:
    with torch.no_grad():
        eager_outputs = output_dict(eager(inputs))
        scripted_outputs = output_dict(scripted(inputs))
    return max(float((value - scripted_outputs[key]).abs().max()) for key, value in eager_outputs.items())


class NativeTorchScriptExportTest(unittest.TestCase):
    def test_export_python_model_generates_native_torchscript(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exported_model = export_probe_model(tmp_path / "export", "native_torchscript_probe", 5, 8, 0.05)
            tracer = load_tracer_model(exported_model)

            eager_loss = backward_probe(tracer.train(), make_inputs(3, 5, True))
            scripted = torch.jit.script(tracer)
            scripted_loss = backward_probe(scripted.train(), make_inputs(3, 5, True))

            comparison_inputs = make_inputs(3, 5, False)
            self.assertLessEqual(max_output_difference(tracer.eval(), scripted.eval(), comparison_inputs), 1e-6)
            artifact = tmp_path / "native_torchscript_probe.pt"
            torch.jit.save(scripted, str(artifact))
            loaded = torch.jit.load(str(artifact), map_location="cpu")
            self.assertLessEqual(max_output_difference(tracer.eval(), loaded.eval(), comparison_inputs), 1e-6)
            self.assertTrue(torch.isfinite(torch.tensor([eager_loss, scripted_loss])).all())


if __name__ == "__main__":
    unittest.main()
