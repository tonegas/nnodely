"""TorchModule runs a torch.nn.Module on every backend, so each test compares
nnodely against the same module run by torch itself: the forward pass, a
training step with its BatchNorm statistics, and what a save brings back."""

import copy

import keras
import numpy as np
import pytest

from conftest import to_numpy
from nnodely import DataLoader, Input, Linear, Modely, Output, TorchModule

torch = pytest.importorskip("torch")
nn = torch.nn


def _cnn(dropout=0.0):
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


def _fit_model(name, module, trainable=True, head=False):
    image = Input(f"{name}_img", dim=(3, 8, 8))
    target = Input(f"{name}_t", dim=2)
    features = TorchModule(module, trainable=trainable)(image.last())
    output = Output(f"{name}_y", Linear(2)(features) if head else features)
    model = Modely(name, inputs=[image], outputs=[output])
    model.minimize("error", source=output, target=target.last())
    return model.build(), features


def _images(count, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(count, 3, 8, 8)).astype(np.float32),
        rng.normal(size=(count, 2)).astype(np.float32),
    )


def test_module_sees_every_time_step_as_a_sample():
    module = _cnn().eval()
    image = Input("frames", dim=(3, 8, 8))
    features = TorchModule(module)(image.sw(2))
    model = Modely("frames", inputs=[image], outputs=[Output("out", features)]).build()
    assert features.shape.dimensions == ((2,), 2, ())

    frames = np.random.default_rng(0).normal(size=(4, 3, 8, 8, 2)).astype(np.float32)
    result = to_numpy(model({"frames": frames})["out"])

    with torch.no_grad():
        flat = torch.tensor(frames).permute(0, 4, 1, 2, 3).reshape(8, 3, 8, 8)
        expected = module(flat).reshape(4, 2, 2).permute(0, 2, 1).numpy()
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_a_training_step_matches_torch():
    # One full-batch SGD step: the gradient reaches every parameter, and the
    # BatchNorm running statistics move as they do in torch.
    module = _cnn()
    reference = copy.deepcopy(module)
    model, features = _fit_model("sgd_step", module)
    images, targets = _images(6)

    data = DataLoader(model, source={"sgd_step_img": images, "sgd_step_t": targets})
    model.train(data, epochs=1, batch_size=6, optimizer="sgd", lr=0.1, shuffle=False)

    reference.train()
    loss = ((reference(torch.tensor(images)) - torch.tensor(targets)) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter -= 0.1 * parameter.grad

    trained = features.to_torch().state_dict()
    for key, expected in reference.state_dict().items():
        np.testing.assert_allclose(
            trained[key].float().numpy(), expected.float().numpy(), atol=1e-5
        )


def test_dropout_replays_its_mask_for_the_gradient():
    # y = 2 * w * mask on a constant input, so loss = 4 w^2 p for the kept
    # fraction p, and the gradient it implies is 2 * loss / w - which holds
    # only if the backward pass drops the samples the forward pass dropped.
    module = nn.Sequential(nn.Dropout(0.5), nn.Linear(1, 1, bias=False))
    x = Input("drop_x")
    dropped = TorchModule(module)(x.last())
    out = Output("drop_y", dropped)
    model = Modely("drop", inputs=[x], outputs=[out])
    model.minimize("error", source=out, target=0.0)
    model.build()

    data = DataLoader(model, source={"drop_x": np.ones(64, np.float32)})
    w = module[1].weight.item()
    history = model.train(data, epochs=1, batch_size=64, optimizer="sgd", lr=0.1)

    trained = dropped.to_torch()[1].weight.item()
    np.testing.assert_allclose(trained, w - 0.1 * 2 * history["loss"][0] / w, rtol=1e-4)


def test_a_frozen_module_keeps_its_weights():
    module = _cnn()
    initial = copy.deepcopy(module.state_dict())
    model, features = _fit_model("frozen", module, trainable=False, head=True)
    # Only the Linear head trains.
    assert model.model is not None
    assert len(model.model.trainable_weights) == 2

    images, targets = _images(6)
    data = DataLoader(model, source={"frozen_img": images, "frozen_t": targets})
    model.train(data, epochs=2, batch_size=2)

    for key, value in features.to_torch().state_dict().items():
        assert torch.equal(value, initial[key]), key


def test_save_and_load_bring_the_module_back(tmp_path):
    model, _ = _fit_model("saved", _cnn())
    images = _images(2)[0][..., None]  # one time step
    expected = to_numpy(model({"saved_img": images})["saved_y"])

    model.save(tmp_path / "saved")
    loaded = Modely.load(tmp_path / "saved")

    result = to_numpy(loaded({"saved_img": images})["saved_y"])
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.skipif(
    keras.backend.backend() != "torch",
    reason="only the torch backend exports a TorchModule as ONNX operators",
)
def test_export_onnx_on_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    model, _ = _fit_model("onnx_torch", _cnn())
    # The torch exporter fixes the batch size to one.
    images = _images(1)[0][..., None]
    expected = to_numpy(model({"onnx_torch_img": images})["onnx_torch_y"])

    model.export_onnx(tmp_path)
    actual = Modely.validate_onnx(
        tmp_path / "onnx_torch.onnx", {"onnx_torch_img": images}, return_dict=True
    )
    assert isinstance(actual, dict)
    np.testing.assert_allclose(to_numpy(actual["onnx_torch_y"]), expected, atol=1e-5)


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="the tensorflow backend runs the module as a Python callback",
)
def test_export_onnx_on_tensorflow_says_it_is_unavailable(tmp_path):
    model, _ = _fit_model("onnx_tf", _cnn())

    with pytest.raises(NotImplementedError, match="TorchModule"):
        model.export_onnx(tmp_path)
    assert not (tmp_path / "onnx_tf.onnx").exists()
