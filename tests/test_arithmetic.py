from nnodely import Input, Modely, Output
from nnodely.layers.arithmetic import (
    Abs,
    Ceil,
    Clamp,
    Deg2Rad,
    Exp,
    Floor,
    Log,
    Log10,
    Sign,
    Sqrt,
    Sum,
    Negative
)
from conftest import to_numpy
import numpy as np
import pytest


@pytest.mark.parametrize(
    "layer_class, reference, low, high",
    [
        (Exp, np.exp, -2.0, 2.0),
        (Log, np.log, 0.5, 5.0),
        (Log10, np.log10, 0.5, 5.0),
        (Sqrt, np.sqrt, 0.5, 5.0),
        (Abs, np.abs, -3.0, 3.0),
        (Floor, np.floor, -3.0, 3.0),
        (Ceil, np.ceil, -3.0, 3.0),
        (Sign, np.sign, -3.0, 3.0),
        (Negative, np.negative, -3.0, 3.0),
        (Deg2Rad, np.deg2rad, -180.0, 180.0)
    ],
)
def test_arithmetic(layer_class, reference, low, high):
    # ------- Test arithmetic layers -------
    x = Input("x", dim=1)
    window = 3
    out = Output("out", layer_class()([x.sw(window)]))
    model = Modely(
        f"arithmetic_{layer_class.__name__.lower()}_model",
        inputs=[x],
        outputs=[out],
    )
    model.build()

    # ------- Model inference -------
    batch_size = 1
    dummy_input_x = np.random.uniform(low, high, size=(batch_size, 1, window))
    result = model({"x": dummy_input_x})
    assert "out" in result
    assert result["out"].shape == (batch_size, 1, window)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        reference(dummy_input_x),
        rtol=1e-5,
        atol=1e-5,
    )


def test_arithmetic_chain():
    # ------- Test chained arithmetic layers -------
    x = Input("x", dim=1)
    out = Output("out", Log(name="log")(Abs()(Exp()([x.sw(2)]))))
    model = Modely("arithmetic_chain_model", inputs=[x], outputs=[out])
    model.build()

    dummy_input_x = np.random.uniform(-2.0, 2.0, size=(1, 1, 2))
    np.testing.assert_allclose(
        to_numpy(model({"x": dummy_input_x})["out"]),
        np.log(np.abs(np.exp(dummy_input_x))),
        rtol=1e-5,
        atol=1e-5,
    )


def test_arithmetic_save_load(tmp_path):
    # ------- Test arithmetic layer serialization -------
    x = Input("x", dim=1)
    out = Output("out", Sqrt(name="sqrt")([x.sw(3)]))
    model = Modely("arithmetic_save_model", inputs=[x], outputs=[out])
    model.build()

    inputs = {"x": np.random.uniform(0.5, 5.0, size=(1, 1, 3))}
    expected = model(inputs)["out"]

    path = tmp_path / "arithmetic.nnodely"
    model.save(path)
    restored = Modely.load(path)
    np.testing.assert_allclose(
        to_numpy(restored(inputs)["out"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    restored_layer = next(
        node for node in restored.flatten().order if isinstance(node, Sqrt)
    )
    assert restored_layer.get_config() == {"name": "sqrt"}


@pytest.mark.parametrize(
    "minimum, maximum",
    [(-1.0, 1.0), (-1.0, None), (None, 1.0)],
)
def test_clamp(minimum, maximum):
    # ------- Test clamp bounds -------
    x = Input("x", dim=1)
    window = 3
    out = Output("out", Clamp(min=minimum, max=maximum)([x.sw(window)]))
    model = Modely("arithmetic_clamp_model", inputs=[x], outputs=[out])
    model.build()

    dummy_input_x = np.random.uniform(-3.0, 3.0, size=(1, 1, window))
    result = model({"x": dummy_input_x})
    assert result["out"].shape == (1, 1, window)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        np.clip(
            dummy_input_x,
            -np.inf if minimum is None else minimum,
            np.inf if maximum is None else maximum,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_clamp_invalid_bounds():
    # ------- Test clamp bounds validation -------
    with pytest.raises(ValueError):
        Clamp(min=1.0, max=-1.0)


@pytest.mark.parametrize(
    "axis, reference_axis, expected_dim",
    [
        (None, (1, 2), (1, 1)),
        (0, 1, (1, 3)),
        (1, 2, (2, 1)),
        (-1, 2, (2, 1)),
    ],
)
def test_sum(axis, reference_axis, expected_dim):
    # ------- Test sum over the dim axes -------
    x = Input("x", dim=(2, 3))
    window = 4
    out = Output("out", Sum(axis=axis)([x.sw(window)]))
    model = Modely("arithmetic_sum_model", inputs=[x], outputs=[out])
    model.build()

    batch_size = 5
    dummy_input_x = np.random.uniform(-2.0, 2.0, size=(batch_size, 2, 3, window))
    result = model({"x": dummy_input_x})
    assert result["out"].shape == (batch_size, *expected_dim, window)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        dummy_input_x.sum(axis=reference_axis, keepdims=True),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sum_axis_out_of_bounds():
    # ------- Test sum axis validation -------
    x = Input("x", dim=(2, 3))
    with pytest.raises(ValueError):
        Sum(axis=2)([x.sw(2)])


def test_clamp_sum_save_load(tmp_path):
    # ------- Test clamp and sum serialization -------
    x = Input("x", dim=(2, 3))
    out = Output(
        "out", Sum(name="sum")(Clamp(min=-0.5, max=0.5, name="clamp")([x.sw(3)]))
    )
    model = Modely("arithmetic_clamp_sum_save_model", inputs=[x], outputs=[out])
    model.build()

    inputs = {"x": np.random.uniform(-2.0, 2.0, size=(1, 2, 3, 3))}
    expected = model(inputs)["out"]

    path = tmp_path / "arithmetic_clamp_sum.nnodely"
    model.save(path)
    restored = Modely.load(path)
    np.testing.assert_allclose(
        to_numpy(restored(inputs)["out"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    nodes = {node.name: node for node in restored.flatten().order}
    assert nodes["clamp"].get_config() == {
        "name": "clamp",
        "min": -0.5,
        "max": 0.5,
    }
    assert nodes["sum"].get_config() == {"name": "sum", "axis": None}
