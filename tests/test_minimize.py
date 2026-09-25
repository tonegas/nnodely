"""Minimizers: registration, removal, and the loss they actually train on.

A minimizer compares a ``source`` stream (the prediction, ``y_pred``) with a
``target`` stream (the reference, ``y_true``) through a Keras loss. Either side
can be a model Output, an intermediate Stream, an Input or one of its windows,
and the target can also be a number, turned into a Constant.

Loss values are checked against a numpy reference computed straight from the
CSV files in ``tests/datasets``, never from the DataLoader or the model, so a
minimizer wired to the wrong signal cannot agree with it. The loss the trainer
sees is read from one full-batch epoch at learning rate zero: the weights do
not move, so the epoch loss is exactly the loss at the initial weights.
"""

import glob
import os

import keras
import numpy as np
import pandas as pd
import pytest

from nnodely import (
    Constant,
    DataLoader,
    Input,
    Linear,
    Loop,
    Modely,
    Output,
    Parameter,
    Tanh,
)
from conftest import to_numpy


DATASETS = os.path.join(os.path.dirname(__file__), "datasets")
FORMAT = {
    "x": "data_1",
    "y": "data_3",
    "x2": ["data_1", "data_2"],
    "y2": ["data_3", "a"],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _column(*names, past=1, future=0, shift=0):
    """CSV columns, one row per window the DataLoader keeps.

    ``past``/``future`` are the deepest windows of the whole model, which set
    the range of end samples every input is aligned to. ``shift=0`` reads the
    last sample of the window, ``shift=1`` the sample after it (``next()``).
    """
    first = max(past - 1, 0)
    chunks = []
    for path in sorted(glob.glob(os.path.join(DATASETS, "*.csv"))):
        frame = pd.read_csv(path)
        values = np.stack([frame[name].to_numpy(np.float32) for name in names], axis=-1)
        chunks.append(values[first + shift : len(values) - future + shift])
    values = np.concatenate(chunks)
    return values[:, 0] if len(names) == 1 else values


def _mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def _huber(y_true, y_pred, delta=1.0):
    error = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    quadratic = 0.5 * error**2
    linear = delta * error - 0.5 * delta**2
    return float(np.mean(np.where(error <= delta, quadratic, linear)))


def _mean_error(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


@keras.saving.register_keras_serializable(package="nnodely_test_minimize")
def signed_error_loss(y_true, y_pred):
    """Asymmetric loss: it tells the prediction from the reference."""
    return keras.ops.mean(y_pred - y_true, axis=-1)


def _load(model):
    return DataLoader(model, source=DATASETS, format=FORMAT)


def _training_loss(model, data):
    """Total loss the trainer computes at the current weights."""
    history = model.train(
        train_data=data,
        epochs=1,
        batch_size=len(data),
        optimizer="sgd",
        lr=0.0,
        shuffle=False,
        printer=None,
    )
    return history["loss"][0]


def _validation_losses(model, data):
    result = model.validate(data)
    return {name: signal.metrics["loss"] for name, signal in result.signals.items()}


def _fit(model, data, epochs, lr):
    return model.train(
        train_data=data,
        epochs=epochs,
        batch_size=len(data),
        optimizer="sgd",
        lr=lr,
        shuffle=False,
        printer=None,
    )


def _f32(values):
    return np.asarray(values, dtype=np.float32)


def _value(parameter):
    return float(np.ravel(to_numpy(parameter.value_numpy))[0])


def _gain_model(gain=2.0):
    """``pred = gain * x``: a prediction whose value is known before training."""
    x = Input("x")
    k = Parameter(value=[gain])
    pred = x.last() * k
    out = Output("pred", pred)
    return x, k, pred, out


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------


def test_minimize_returns_the_model_for_chaining():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    returned = model.minimize("a", out, Input("y").last()).minimize("b", out, 1.0)

    assert returned is model
    assert [m["name"] for m in model.minimizers] == ["a", "b"]


def test_minimize_stores_source_target_and_resolved_loss():
    x, _, _, out = _gain_model()
    target = Input("y").last()
    model = Modely("model", inputs=[x], outputs=[out])

    model.minimize("error", out, target, loss="mae")

    [minimizer] = model.minimizers
    assert minimizer["name"] == "error"
    assert minimizer["source"] is out
    assert minimizer["target"] is target
    assert callable(minimizer["loss"])
    value = minimizer["loss"](_f32([[1.0]]), _f32([[4.0]]))
    np.testing.assert_allclose(to_numpy(value), [3.0])


def test_minimize_default_loss_is_mse():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    model.minimize("error", out, Input("y").last())

    value = model.minimizers[0]["loss"](_f32([[1.0]]), _f32([[4.0]]))
    np.testing.assert_allclose(to_numpy(value), [9.0])


def test_minimize_without_target_compares_against_a_zero_constant():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    model.minimize("error", out)

    target = model.minimizers[0]["target"]
    assert isinstance(target, Constant)
    np.testing.assert_allclose(target.value, [0.0])


@pytest.mark.parametrize(
    "number",
    [3.0, 3, np.float32(3.0), np.float64(3.0)],
    ids=["float", "int", "numpy_float32", "numpy_float64"],
)
def test_minimize_numeric_target_becomes_a_constant(number):
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    model.minimize("error", out, number)

    target = model.minimizers[0]["target"]
    assert isinstance(target, Constant)
    np.testing.assert_allclose(target.value, [3.0])


def test_minimize_keeps_an_explicit_constant_target():
    x, _, _, out = _gain_model()
    three = Constant("three", value=[3.0])
    model = Modely("model", inputs=[x], outputs=[out])

    model.minimize("error", out, three)

    assert model.minimizers[0]["target"] is three


def test_minimize_with_an_existing_name_replaces_it_and_warns():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, Input("y").last())
    model.minimize("other", out, 2.0)

    with pytest.warns(UserWarning, match="'error' already exists"):
        model.minimize("error", out, 1.0, loss="mae")

    assert [m["name"] for m in model.minimizers] == ["error", "other"]
    replaced = model.minimizers[0]
    np.testing.assert_allclose(replaced["target"].value, [1.0])
    value = replaced["loss"](_f32([[1.0]]), _f32([[4.0]]))
    np.testing.assert_allclose(to_numpy(value), [3.0])


def test_a_replaced_minimizer_is_the_one_trained():
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, y.last())
    with pytest.warns(UserWarning):
        model.minimize("error", out, 3.0)
    model.build()
    data = _load(model)

    X = _column("data_1")
    np.testing.assert_allclose(_training_loss(model, data), _mse(3.0, 2 * X), rtol=1e-5)
    np.testing.assert_allclose(
        _validation_losses(model, data)["error"], _mse(3.0, 2 * X), rtol=1e-5
    )


@pytest.mark.parametrize(
    "source", [3.0, None, [object()]], ids=["number", "none", "list"]
)
def test_minimize_rejects_a_source_that_is_not_a_stream_or_a_name(source):
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    with pytest.raises(TypeError, match="source"):
        model.minimize("error", source, 1.0)
    assert model.minimizers == []


def test_minimize_rejects_a_target_that_is_not_a_stream_a_name_or_a_number():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    with pytest.raises(TypeError, match="target"):
        model.minimize("error", out, [object()])  # type: ignore
    assert model.minimizers == []


def test_minimize_resolves_the_names_of_the_model_streams():
    x, _, pred, out = _gain_model()
    y = Input("y")
    reference = Output("y_ref", y.last())
    model = Modely("model", inputs=[x, y], outputs=[out, reference])

    model.minimize("outputs", "pred", "y_ref")
    model.minimize("stream_and_input", pred.name, "y")

    by_name = {m["name"]: m for m in model.minimizers}
    assert by_name["outputs"]["source"] is out
    assert by_name["outputs"]["target"] is reference
    assert by_name["stream_and_input"]["source"] is pred
    assert by_name["stream_and_input"]["target"] is y


def test_minimize_by_name_trains_on_the_named_streams():
    x, _, _, out = _gain_model()
    y = Input("y")
    reference = Output("y_ref", y.last())
    model = Modely("model", inputs=[x, y], outputs=[out, reference])
    model.minimize("error", "pred", "y_ref")
    model.build()
    data = _load(model)

    X, Y = _column("data_1"), _column("data_3")
    np.testing.assert_allclose(_training_loss(model, data), _mse(Y, 2 * X), rtol=1e-5)


@pytest.mark.parametrize("side", ["source", "target"])
def test_minimize_rejects_a_name_the_model_does_not_have(side):
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    arguments = {"source": out, "target": 1.0, side: "missing"}
    with pytest.raises(ValueError, match=f"{side} 'missing'"):
        model.minimize("error", **arguments)
    assert model.minimizers == []


def test_minimize_rejects_a_name_shared_by_several_streams():
    # One named layer applied twice gives two streams that carry its name.
    x = Input("x")
    shared = Linear(out_features=1, name="shared_linear")
    first = Output("first", shared(x.last()))
    second = Output("second", shared(x.sw(2)))
    model = Modely("model", inputs=[x], outputs=[first, second])

    with pytest.raises(ValueError, match="ambiguous"):
        model.minimize("error", "shared_linear", 1.0)
    assert model.minimizers == []

    # The Outputs keep distinct names, so they remain nameable.
    model.minimize("error", "first", 1.0)
    assert model.minimizers[0]["source"] is first


def test_minimize_rejects_an_unknown_loss():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    with pytest.raises(ValueError, match="loss"):
        model.minimize("error", out, 1.0, loss="not_a_keras_loss")
    assert model.minimizers == []


# -----------------------------------------------------------------------------
# Removal
# -----------------------------------------------------------------------------


def test_remove_minimizer_keeps_the_others_in_order():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("a", out, 1.0)
    model.minimize("b", out, 2.0)
    model.minimize("c", out, 3.0)

    model.remove_minimizer("b")

    assert [m["name"] for m in model.minimizers] == ["a", "c"]


def test_remove_minimizer_with_an_unknown_name_warns_and_changes_nothing():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("a", out, 1.0)

    with pytest.warns(UserWarning, match="'missing'"):
        model.remove_minimizer("missing")
    assert [m["name"] for m in model.minimizers] == ["a"]


def test_a_removed_name_can_be_registered_again():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, 1.0)

    model.remove_minimizer("error")
    model.minimize("error", out, 2.0)

    [minimizer] = model.minimizers
    np.testing.assert_allclose(minimizer["target"].value, [2.0])


def test_train_after_removing_every_minimizer_raises():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, Input("y").last())
    model.build()
    data = _load(model)

    model.remove_minimizer("error")

    with pytest.raises(ValueError, match="No minimizers"):
        model.train(train_data=data, epochs=1, printer=None)


# -----------------------------------------------------------------------------
# Loss values: which signals a minimizer compares
#
# Every case returns the model and the loss each minimizer must report, from
# a model whose prediction is ``pred = 2 * x`` before training.
# -----------------------------------------------------------------------------


def _case_output_vs_input_window():
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, y.last())
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(Y, 2 * X)}


def _case_stream_vs_input_window():
    x, _, pred, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", pred, y.last())
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(Y, 2 * X)}


def _case_output_vs_input():
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, y)
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(Y, 2 * X)}


def _case_input_vs_output():
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", y, out)
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(2 * X, Y)}


def _case_input_window_vs_output():
    # The orientation examples/mass_estimation uses: the measured signal first.
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x, y], outputs=[out])
    model.minimize("error", y.last(), out)
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(2 * X, Y)}


def _case_input_window_vs_input_window():
    # No trainable weight on either side: a baseline between two measurements.
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", x.last(), y.last())
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(Y, X)}


def _case_output_vs_float():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, 3.0)
    X = _column("data_1")
    return model, {"error": _mse(3.0, 2 * X)}


def _case_output_vs_int():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, 3)
    X = _column("data_1")
    return model, {"error": _mse(3.0, 2 * X)}


def _case_output_vs_none():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out)
    X = _column("data_1")
    return model, {"error": _mse(0.0, 2 * X)}


def _case_output_vs_constant():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, Constant("three", value=[3.0]))
    X = _column("data_1")
    return model, {"error": _mse(3.0, 2 * X)}


def _case_stream_vs_float():
    x, _, pred, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", pred, 3.0)
    X = _column("data_1")
    return model, {"error": _mse(3.0, 2 * X)}


def _case_output_vs_computed_stream():
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, y.last() * 0.5)
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(0.5 * Y, 2 * X)}


def _case_output_vs_layer_of_one_input():
    # A single-input layer on the target: the target is tanh(y), not y.
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, Tanh()(y.last()))
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(np.tanh(Y), 2 * X)}


def _case_output_vs_output():
    x, _, _, out = _gain_model()
    other = Output("pred_unit", x.last() * Parameter(value=[1.0]))
    model = Modely("model", inputs=[x], outputs=[out, other])
    model.minimize("error", out, other)
    X = _column("data_1")
    return model, {"error": _mse(X, 2 * X)}


def _case_output_vs_output_of_an_input_window():
    x, _, _, out = _gain_model()
    y = Input("y")
    reference = Output("y_ref", y.last())
    model = Modely("model", inputs=[x, y], outputs=[out, reference])
    model.minimize("error", out, reference)
    X, Y = _column("data_1"), _column("data_3")
    return model, {"error": _mse(Y, 2 * X)}


def _case_one_source_two_targets():
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("to_y", out, y.last())
    model.minimize("to_zero", out, 0.0)
    X, Y = _column("data_1"), _column("data_3")
    return model, {"to_y": _mse(Y, 2 * X), "to_zero": _mse(0.0, 2 * X)}


def _case_one_target_two_sources():
    x, _, _, out = _gain_model()
    y = Input("y")
    target = y.last()
    other = Output("pred_unit", x.last() * Parameter(value=[1.0]))
    model = Modely("model", inputs=[x], outputs=[out, other])
    model.minimize("double", out, target)
    model.minimize("unit", other, target, loss="mae")
    X, Y = _column("data_1"), _column("data_3")
    return model, {"double": _mse(Y, 2 * X), "unit": _mae(Y, X)}


def _case_target_narrower_than_its_input():
    # y carries a 3-sample window elsewhere; the target is its last sample only.
    x, _, _, out = _gain_model()
    y = Input("y")
    history = Output("y_history", y.sw(3))
    model = Modely("model", inputs=[x, y], outputs=[out, history])
    model.minimize("error", out, y.last())
    X, Y = _column("data_1", past=3), _column("data_3", past=3)
    return model, {"error": _mse(Y, 2 * X)}


def _case_next_target_with_a_past_window():
    # Autoregressive shape: y's past is an input, y's next sample the target.
    x, _, _, out = _gain_model()
    y = Input("y")
    history = Output("y_history", y.sw(2))
    model = Modely("model", inputs=[x, y], outputs=[out, history])
    model.minimize("error", out, y.next())
    X = _column("data_1", past=2, future=1)
    Y_next = _column("data_3", past=2, future=1, shift=1)
    return model, {"error": _mse(Y_next, 2 * X)}


def _case_next_source_vs_output():
    # examples/mass_estimation: minimize(name, signal.next(), Output)
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x, y], outputs=[out])
    model.minimize("error", y.next(), out)
    X = _column("data_1", future=1)
    Y_next = _column("data_3", future=1, shift=1)
    return model, {"error": _mse(2 * X, Y_next)}


def _case_multi_dimensional_signals():
    x2 = Input("x2", dim=2)
    y2 = Input("y2", dim=2)
    pred = x2.last() * Parameter(value=[2.0])
    out = Output("pred", pred)
    model = Modely("model", inputs=[x2], outputs=[out])
    model.minimize("to_y", out, y2.last())
    model.minimize("to_three", out, 3.0)
    X = _column("data_1", "data_2")
    Y = _column("data_3", "a")
    return model, {"to_y": _mse(Y, 2 * X), "to_three": _mse(3.0, 2 * X)}


LOSS_CASES = {
    "output_vs_input_window": _case_output_vs_input_window,
    "stream_vs_input_window": _case_stream_vs_input_window,
    "output_vs_input": _case_output_vs_input,
    "input_vs_output": _case_input_vs_output,
    "input_window_vs_output": _case_input_window_vs_output,
    "input_window_vs_input_window": _case_input_window_vs_input_window,
    "output_vs_float": _case_output_vs_float,
    "output_vs_int": _case_output_vs_int,
    "output_vs_none": _case_output_vs_none,
    "output_vs_constant": _case_output_vs_constant,
    "stream_vs_float": _case_stream_vs_float,
    "output_vs_computed_stream": _case_output_vs_computed_stream,
    "output_vs_layer_of_one_input": _case_output_vs_layer_of_one_input,
    "output_vs_output": _case_output_vs_output,
    "output_vs_output_of_an_input_window": _case_output_vs_output_of_an_input_window,
    "one_source_two_targets": _case_one_source_two_targets,
    "one_target_two_sources": _case_one_target_two_sources,
    "target_narrower_than_its_input": _case_target_narrower_than_its_input,
    "next_target_with_a_past_window": _case_next_target_with_a_past_window,
    "next_source_vs_output": _case_next_source_vs_output,
    "multi_dimensional_signals": _case_multi_dimensional_signals,
}


# Two measurements compared with each other can be scored, but not trained on.
TRAINING_CASES = {
    name: case
    for name, case in LOSS_CASES.items()
    if name != "input_window_vs_input_window"
}


@pytest.mark.parametrize("case", TRAINING_CASES)
def test_training_loss_compares_the_declared_signals(case):
    model, expected = TRAINING_CASES[case]()
    model.build()
    data = _load(model)

    loss = _training_loss(model, data)

    np.testing.assert_allclose(loss, sum(expected.values()), rtol=1e-5)


@pytest.mark.parametrize("case", LOSS_CASES)
def test_validation_loss_compares_the_declared_signals(case):
    model, expected = LOSS_CASES[case]()
    model.build()
    data = _load(model)

    losses = _validation_losses(model, data)

    assert set(losses) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(losses[name], value, rtol=1e-5, err_msg=name)


def test_training_on_no_trainable_weight_is_refused():
    """A network with weights, but no minimizer that reaches any of them."""
    model, _ = _case_input_window_vs_input_window()
    model.build()
    data = _load(model)

    with pytest.raises(ValueError, match="trainable weight"):
        _training_loss(model, data)


def test_a_minimizer_without_trainable_weights_warns_and_adds_a_constant():
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("fit", out, y.last())
    model.minimize("baseline", x.last(), y.last())
    model.build()
    data = _load(model)

    with pytest.warns(UserWarning, match="'baseline'.*no trainable weight"):
        loss = _training_loss(model, data)

    X, Y = _column("data_1"), _column("data_3")
    np.testing.assert_allclose(loss, _mse(Y, 2 * X) + _mse(Y, X), rtol=1e-5)


# -----------------------------------------------------------------------------
# Loss types
# -----------------------------------------------------------------------------


LOSS_TYPES = {
    "mse_name": ("mse", _mse),
    "mae_name": ("mae", _mae),
    "huber_name": ("huber", _huber),
    "loss_instance": (keras.losses.MeanAbsoluteError(), _mae),
    "loss_config": ({"class_name": "MeanSquaredError", "config": {}}, _mse),
    "custom_callable": (signed_error_loss, _mean_error),
}


@pytest.mark.parametrize("kind", LOSS_TYPES)
def test_each_loss_type_is_the_one_trained_and_validated(kind):
    loss, reference = LOSS_TYPES[kind]
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, y.last(), loss=loss)
    model.build()
    data = _load(model)
    X, Y = _column("data_1"), _column("data_3")
    expected = reference(Y, 2 * X)

    np.testing.assert_allclose(_training_loss(model, data), expected, rtol=1e-5)
    np.testing.assert_allclose(
        _validation_losses(model, data)["error"], expected, rtol=1e-5
    )


@pytest.mark.parametrize("swapped", [False, True], ids=["source_first", "target_first"])
def test_source_is_the_prediction_and_target_the_reference(swapped):
    """An asymmetric loss sees ``y_true = target`` and ``y_pred = source``."""
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    if swapped:
        model.minimize("error", y.last(), out, loss=signed_error_loss)
    else:
        model.minimize("error", out, y.last(), loss=signed_error_loss)
    model.build()
    data = _load(model)
    X, Y = _column("data_1"), _column("data_3")
    expected = _mean_error(2 * X, Y) if swapped else _mean_error(Y, 2 * X)

    np.testing.assert_allclose(_training_loss(model, data), expected, rtol=1e-5)
    np.testing.assert_allclose(
        _validation_losses(model, data)["error"], expected, rtol=1e-5
    )


# -----------------------------------------------------------------------------
# Training: the gradient reaches the weights behind each side
# -----------------------------------------------------------------------------


def _proportional_data(model, gain=3.0, samples=8):
    x = np.linspace(0.5, 1.5, samples, dtype=np.float32)
    return DataLoader(model, source={"x": x, "y": gain * x})


def test_float_target_drives_the_source_to_it():
    x = Input("x")
    k = Parameter(value=[1.0])
    out = Output("pred", x.last() * k)
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, 6.0)
    model.build()
    data = DataLoader(model, source={"x": np.full(8, 2.0, dtype=np.float32)})

    _fit(model, data, epochs=60, lr=0.05)

    np.testing.assert_allclose(_value(k), 3.0, atol=1e-3)


@pytest.mark.parametrize("swapped", [False, True], ids=["source_first", "target_first"])
def test_output_is_trained_whichever_side_it_is_declared_on(swapped):
    x = Input("x")
    y = Input("y")
    k = Parameter(value=[1.0])
    out = Output("pred", x.last() * k)
    model = Modely("model", inputs=[x, y], outputs=[out])
    if swapped:
        model.minimize("error", y.last(), out)
    else:
        model.minimize("error", out, y.last())
    model.build()

    _fit(model, _proportional_data(model), epochs=300, lr=0.2)

    np.testing.assert_allclose(_value(k), 3.0, atol=1e-3)


def test_computed_target_is_trained_against():
    x = Input("x")
    y = Input("y")
    k = Parameter(value=[1.0])
    out = Output("pred", x.last() * k)
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, y.last() * 0.5)
    model.build()

    _fit(model, _proportional_data(model, gain=6.0), epochs=300, lr=0.2)

    np.testing.assert_allclose(_value(k), 3.0, atol=1e-3)


def test_output_vs_output_trains_both_sides():
    x = Input("x")
    y = Input("y")
    k_fit = Parameter(value=[1.0])
    k_follow = Parameter(value=[0.0])
    fit = Output("fit", x.last() * k_fit)
    follow = Output("follow", x.last() * k_follow)
    model = Modely("model", inputs=[x], outputs=[fit, follow])
    model.minimize("fit_error", fit, y.last())
    model.minimize("follow_error", follow, fit)
    model.build()
    data = DataLoader(
        model,
        source={
            "x": np.ones(8, dtype=np.float32),
            "y": np.full(8, 3.0, dtype=np.float32),
        },
    )

    _fit(model, data, epochs=300, lr=0.1)

    np.testing.assert_allclose(_value(k_fit), 3.0, atol=1e-3)
    np.testing.assert_allclose(_value(k_follow), 3.0, atol=1e-3)


@pytest.mark.parametrize("rebuild", [True, False], ids=["rebuilt", "not_rebuilt"])
def test_a_removed_minimizer_no_longer_trains_its_weights(rebuild):
    x = Input("x")
    y = Input("y")
    k_kept = Parameter(value=[1.0])
    k_removed = Parameter(value=[1.0])
    kept = Output("kept", x.last() * k_kept)
    removed = Output("removed", x.last() * k_removed)
    model = Modely("model", inputs=[x], outputs=[kept, removed])
    model.minimize("kept_error", kept, y.last())
    model.minimize("removed_error", removed, 10.0)
    model.build()

    model.remove_minimizer("removed_error")
    if rebuild:
        model.build()
    _fit(model, _proportional_data(model), epochs=300, lr=0.2)

    np.testing.assert_allclose(_value(k_kept), 3.0, atol=1e-3)
    np.testing.assert_allclose(_value(k_removed), 1.0)


def test_removed_minimizer_is_not_validated():
    x, _, _, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("kept", out, y.last())
    model.minimize("removed", out, 3.0)
    model.build()
    model.remove_minimizer("removed")
    data = _load(model)

    losses = _validation_losses(model, data)

    X, Y = _column("data_1"), _column("data_3")
    assert set(losses) == {"kept"}
    np.testing.assert_allclose(losses["kept"], _mse(Y, 2 * X), rtol=1e-5)
    np.testing.assert_allclose(_training_loss(model, data), _mse(Y, 2 * X), rtol=1e-5)


def test_minimizer_added_after_build_is_trained_after_rebuilding():
    x, k, pred, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("to_y", out, y.last())
    model.build()

    model.minimize("stream_to_three", pred, 3.0)
    model.build()
    data = _load(model)

    X, Y = _column("data_1"), _column("data_3")
    expected = _mse(Y, 2 * X) + _mse(3.0, 2 * X)
    np.testing.assert_allclose(_training_loss(model, data), expected, rtol=1e-5)


def test_training_a_minimizer_added_after_build_asks_for_a_rebuild():
    """A minimizer on a stream the built graph does not expose needs build()."""
    x, _, pred, out = _gain_model()
    y = Input("y")
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("to_y", out, y.last())
    model.build()
    data = _load(model)

    model.minimize("stream_to_three", pred, 3.0)

    with pytest.raises(ValueError, match="build"):
        model.train(train_data=data, epochs=1, printer=None)


def test_streams_of_different_shapes_are_not_broadcast():
    """One sample against a whole window is an error, not a broadcast loss."""
    x, _, _, _ = _gain_model()
    y = Input("y")
    window = Output("window", x.sw(3) * Parameter(value=[2.0]))
    model = Modely("model", inputs=[x], outputs=[window])
    model.minimize("error", window, y.last())
    model.build()
    data = _load(model)

    with pytest.raises(ValueError, match="same shape"):
        _training_loss(model, data)


# -----------------------------------------------------------------------------
# Gains: the weight of each minimizer in the total loss
# -----------------------------------------------------------------------------


def test_minimize_default_gain_is_one():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    model.minimize("error", out, 1.0)

    assert model.minimizers[0]["gain"] == 1.0


def test_gains_weight_the_total_loss_but_not_the_logged_losses():
    """As Keras does with loss_weights: each minimizer logs its own loss."""
    x, _, _, out = _gain_model()
    y = Input("y")
    unit = Output("pred_unit", x.last() * Parameter(value=[1.0]))
    model = Modely("model", inputs=[x], outputs=[out, unit])
    model.minimize("half", out, y.last(), gain=0.5)
    model.minimize("double", unit, 3.0, gain=2)
    model.build()
    data = _load(model)
    X, Y = _column("data_1"), _column("data_3")
    half, double = _mse(Y, 2 * X), _mse(3.0, X)

    history = model.train(
        train_data=data,
        epochs=1,
        batch_size=len(data),
        optimizer="sgd",
        lr=0.0,
        shuffle=False,
        printer=None,
    )

    np.testing.assert_allclose(history["loss"][0], 0.5 * half + 2 * double, rtol=1e-5)
    np.testing.assert_allclose(history["pred_loss"][0], half, rtol=1e-5)
    np.testing.assert_allclose(history["pred_unit_loss"][0], double, rtol=1e-5)
    losses = _validation_losses(model, data)
    np.testing.assert_allclose(losses["half"], half, rtol=1e-5)
    np.testing.assert_allclose(losses["double"], double, rtol=1e-5)


@pytest.mark.parametrize("gain", [0.0, 0.5, 1.0, 2.0])
def test_a_gain_scales_the_step_its_minimizer_takes(gain):
    # loss = gain * (k - 3)^2 at x = 1: one SGD step moves k by 2 * lr * gain * 2
    x = Input("x")
    k = Parameter(value=[1.0])
    out = Output("pred", x.last() * k)
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, 3.0, gain=gain)
    model.build()
    data = DataLoader(model, source={"x": np.ones(4, dtype=np.float32)})

    _fit(model, data, epochs=1, lr=0.1)

    np.testing.assert_allclose(_value(k), 1.0 + 0.4 * gain, atol=1e-6)


def test_gains_set_the_trade_off_between_conflicting_minimizers():
    # gain 1 pulls pred to 0, gain 2 pulls it to 3: the optimum is (0 + 2 * 3) / 3
    x = Input("x")
    k = Parameter(value=[1.0])
    out = Output("pred", x.last() * k)
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("to_zero", out, 0.0, gain=1)
    model.minimize("to_three", out, 3.0, gain=2)
    model.build()
    data = DataLoader(model, source={"x": np.ones(8, dtype=np.float32)})

    _fit(model, data, epochs=60, lr=0.1)

    np.testing.assert_allclose(_value(k), 2.0, atol=1e-4)


@pytest.mark.parametrize(
    "gain, error",
    [
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (True, TypeError),
        ("2", TypeError),
    ],
    ids=["negative", "nan", "bool", "str"],
)
def test_minimize_rejects_an_invalid_gain(gain, error):
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    with pytest.raises(error, match="gain"):
        model.minimize("error", out, 1.0, gain=gain)
    assert model.minimizers == []


# -----------------------------------------------------------------------------
# Sequence weights: the weight of each step along the last axis
# -----------------------------------------------------------------------------


def _windows(name, width):
    """CSV column as sliding windows of ``width`` samples, oldest first."""
    return np.stack(
        [_column(name, past=width, shift=s) for s in range(1 - width, 1)], axis=-1
    )


def _window_model(**minimize_options):
    """``pred = 2 * x.sw(3)`` against ``y.sw(3)``: a known error at every step."""
    x = Input("x")
    y = Input("y")
    out = Output("pred", x.sw(3) * Parameter(value=[2.0]))
    model = Modely("model", inputs=[x], outputs=[out])
    model.minimize("error", out, y.sw(3), **minimize_options)
    model.build()
    X, Y = _windows("data_1", 3), _windows("data_3", 3)
    return model, X, Y


def _elementwise(kind, y_true, y_pred):
    error = np.asarray(y_pred) - np.asarray(y_true)
    if kind == "mae":
        return np.abs(error)
    return error**2


@pytest.mark.parametrize("kind", ["mse", "mae", "mse_instance"])
def test_seq_weights_weigh_each_step_trained_logged_and_validated(kind):
    weights = np.exp(0.5 * np.arange(3))
    loss = keras.losses.MeanSquaredError() if kind == "mse_instance" else kind
    model, X, Y = _window_model(loss=loss, seq_weights=weights)
    data = _load(model)
    normalized = weights / weights.mean()
    expected = float(np.mean(normalized * _elementwise(kind.split("_")[0], Y, 2 * X)))

    history = model.train(
        train_data=data,
        epochs=1,
        batch_size=len(data),
        optimizer="sgd",
        lr=0.0,
        shuffle=False,
        printer=None,
    )

    np.testing.assert_allclose(history["loss"][0], expected, rtol=1e-5)
    np.testing.assert_allclose(history["pred_loss"][0], expected, rtol=1e-5)
    np.testing.assert_allclose(
        _validation_losses(model, data)["error"], expected, rtol=1e-5
    )


def test_uniform_seq_weights_give_the_unweighted_loss():
    model, X, Y = _window_model(seq_weights=[4.0, 4.0, 4.0])

    np.testing.assert_allclose(
        _training_loss(model, _load(model)), _mse(Y, 2 * X), rtol=1e-5
    )


def test_seq_weights_on_the_last_step_only_give_its_error():
    model, X, Y = _window_model(seq_weights=[0.0, 0.0, 1.0])

    np.testing.assert_allclose(
        _training_loss(model, _load(model)), _mse(Y[:, -1], 2 * X[:, -1]), rtol=1e-5
    )


def test_seq_weights_are_stored_normalized_to_mean_one():
    model, _, _ = _window_model(seq_weights=[1.0, 2.0, 3.0])

    np.testing.assert_allclose(model.minimizers[0]["seq_weights"], [0.5, 1.0, 1.5])


def test_minimize_default_seq_weights_are_none():
    x, _, _, out = _gain_model()
    model = Modely("model", inputs=[x], outputs=[out])

    model.minimize("error", out, 1.0)

    assert model.minimizers[0]["seq_weights"] is None


def test_seq_weights_weigh_the_rollout_steps_of_a_loop():
    # c = 0, d = 2: the loop predicts 2 * y at every step, whatever it feeds back
    x, y = Input("x"), Input("y")
    step = x.last() * Parameter("c", value=[0.0]) + y.last() * Parameter(
        "d", value=[2.0]
    )
    body = Modely("body", inputs=[x, y], outputs=[Output("step", step)]).build()
    x_seq, y_seq, t_seq = (
        Input("x_seq", seq=4),
        Input("y_seq", seq=4),
        Input("t_seq", seq=4),
    )
    loop = Loop(
        f=body, callback={"x": "step"}, initial={"x": x_seq}, inputs={"y": y_seq}
    )
    out = Output("rollout", loop)
    model = Modely("model", inputs=[x_seq, y_seq], outputs=[out])
    weights = np.exp(np.arange(4))
    model.minimize("error", out, t_seq, seq_weights=weights)
    model.build()
    rng = np.random.default_rng(0)
    raw = {
        name: rng.normal(size=20).astype(np.float32)
        for name in ("x_seq", "y_seq", "t_seq")
    }
    data = DataLoader(model, source=raw)
    Y = np.stack([raw["y_seq"][i : i + 4] for i in range(17)])
    T = np.stack([raw["t_seq"][i : i + 4] for i in range(17)])
    expected = float(np.mean(weights / weights.mean() * (2 * Y - T) ** 2))

    np.testing.assert_allclose(_training_loss(model, data), expected, rtol=1e-5)


@pytest.mark.parametrize(
    "seq_weights",
    [
        [1.0, 2.0],
        [1.0, -1.0, 1.0],
        [1.0, float("nan"), 1.0],
        [0.0, 0.0, 0.0],
        [[1.0, 1.0, 1.0]],
    ],
    ids=["wrong_length", "negative", "nan", "all_zero", "two_dimensional"],
)
def test_minimize_rejects_invalid_seq_weights(seq_weights):
    x, y = Input("x"), Input("y")
    out = Output("pred", x.sw(3) * Parameter(value=[2.0]))
    model = Modely("model", inputs=[x], outputs=[out])

    with pytest.raises(ValueError, match="seq_weights"):
        model.minimize("error", out, y.sw(3), seq_weights=seq_weights)
    assert model.minimizers == []


def test_callable_seq_weights_are_evaluated_with_the_sequence_length():
    model, X, Y = _window_model(seq_weights=lambda n: np.exp(0.5 * np.arange(n)))
    weights = np.exp(0.5 * np.arange(3))
    expected = _mse(0, np.sqrt(weights / weights.mean()) * (2 * X - Y))

    np.testing.assert_allclose(_training_loss(model, _load(model)), expected, rtol=1e-5)


def _dynamic_loop_model(seq_weights):
    """A rollout of dynamic length predicting ``2 * y`` at every step."""
    x, y = Input("x"), Input("y")
    step = x.last() * Parameter("c", value=[0.0]) + y.last() * Parameter(
        "d", value=[2.0]
    )
    body = Modely("body", inputs=[x, y], outputs=[Output("step", step)]).build()
    x_seq, y_seq = Input("x_seq", seq=-1), Input("y_seq", seq=-1)
    t_seq = Input("t_seq", seq=-1)
    loop = Loop(
        f=body,
        callback={"x": "step"},
        initial={"x": x_seq},
        inputs={"y": y_seq},
        length=3,
    )
    out = Output("rollout", loop)
    model = Modely("model", inputs=[x_seq, y_seq], outputs=[out])
    model.minimize("error", out, t_seq, seq_weights=seq_weights)
    model.build()
    rng = np.random.default_rng(0)
    raw = {
        name: rng.normal(size=20).astype(np.float32)
        for name in ("x_seq", "y_seq", "t_seq")
    }
    return model, raw


def _rollout_loss(raw, width, weights):
    samples = 20 - width + 1
    Y = np.stack([raw["y_seq"][i : i + width] for i in range(samples)])
    T = np.stack([raw["t_seq"][i : i + width] for i in range(samples)])
    return float(np.mean(weights / weights.mean() * (2 * Y - T) ** 2))


@pytest.mark.parametrize("width", [5, 8])
def test_callable_seq_weights_follow_the_length_of_a_dynamic_rollout(width):
    model, raw = _dynamic_loop_model(lambda n: np.exp(0.3 * np.arange(n)))
    data = DataLoader(model, source=raw, seq_length=width)
    expected = _rollout_loss(raw, width, np.exp(0.3 * np.arange(width)))

    np.testing.assert_allclose(_training_loss(model, data), expected, rtol=1e-5)
    np.testing.assert_allclose(
        _validation_losses(model, data)["error"], expected, rtol=1e-5
    )


def test_array_seq_weights_on_a_dynamic_rollout_match_the_data_length():
    # The stream declares length=3, but the rollout follows the data: 5 steps
    weights = np.array([1.0, 1.0, 2.0, 3.0, 5.0])
    model, raw = _dynamic_loop_model(weights)
    data = DataLoader(model, source=raw, seq_length=5)

    np.testing.assert_allclose(
        _training_loss(model, data), _rollout_loss(raw, 5, weights), rtol=1e-5
    )


def test_minimize_rejects_a_callable_giving_the_wrong_number_of_weights():
    x, y = Input("x"), Input("y")
    out = Output("pred", x.sw(3) * Parameter(value=[2.0]))
    model = Modely("model", inputs=[x], outputs=[out])

    with pytest.raises(ValueError, match="seq_weights"):
        model.minimize("error", out, y.sw(3), seq_weights=lambda n: np.ones(n + 1))
    assert model.minimizers == []
