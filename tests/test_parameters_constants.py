import numpy as np
import pytest
from conftest import to_numpy

from nnodely import Input, Output, Modely, Parameter, Constant, Exp


def test_parameter_constant_shapes():
    p = Parameter("param1", value=[1.0])
    c = Constant("const1", value=[1.0])

    assert p.shape.dim == (1,)
    assert c.shape.dim == (1,)


def test_parameter_shape_is_inferred_like_constant():
    parameter = Parameter(
        "parameter_sequence", value=np.ones((2, 3, 4), dtype=np.float32)
    )
    parameter_with_axes = Parameter(
        "parameter_with_axes",
        value=np.ones((1, 3, 4), dtype=np.float32),
        dim=0,
        time=1,
        seq=2,
    )

    assert parameter.shape.dimensions == ((2, 3, 4), 1, ())
    assert parameter_with_axes.shape.dimensions == ((1,), 3, (4,))


def test_parameter_constant_model_inference():
    x = Input("x", dim=1)
    parameter = Parameter("param1", value=[1.0])
    constant = Constant("const1", value=[1.0])

    y = x.sw(1) * parameter + constant
    out = Output("x_out", y)

    model = Modely("model", inputs=[x], outputs=[out])
    model.build()
    model.summary()
    from pprint import pprint

    print("model order:")
    pprint(model.order)

    dummy_x = np.ones((4, 1, 1), dtype=np.float32)
    result = model({"x": dummy_x})

    assert "x_out" in result
    assert result["x_out"].shape == (4, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["x_out"]),
        np.ones((4, 1, 1), dtype=np.float32) * (1.0 * 1.0 + 1.0),
        rtol=1e-5,
        atol=1e-5,
    )
    # assert np.allclose(
    #     keras.ops.convert_to_numpy(result["x_out"]),
    #     [[[2.0]], [[2.0]], [[2.0]], [[2.0]]],
    # )
    # assert np.allclose(result["x_out"], [[[2.0]], [[2.0]], [[2.0]], [[2.0]]])
    assert constant.constant is not None
    assert constant.constant.shape == (1, 1)
    np.testing.assert_allclose(
        to_numpy(constant.constant),
        np.array([[1.0]]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert parameter.param is not None
    assert parameter.param.shape == (1, 1)
    np.testing.assert_allclose(
        to_numpy(parameter.param),
        np.array([[1.0]]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_parameter_through_a_layer_keeps_the_batch_axis():
    x = Input("x", dim=4)
    parameter = Parameter("param_exp", value=np.zeros(4, dtype=np.float32))

    y = x.last() * Exp()([parameter])
    model = Modely("model_param_exp", inputs=[x], outputs=[Output("x_out", y)])
    model.build()

    batch = np.arange(3 * 4, dtype=np.float32).reshape(3, 4, 1)
    result = to_numpy(model({"x": batch})["x_out"])

    assert result.shape == (3, 4, 1)
    np.testing.assert_allclose(result, batch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "value, dimensions",
    [
        (2.0, ((1,), 1, ())),
        ([1.0, 2.0, 3.0], ((3,), 1, ())),
        (np.ones((2, 4)), ((2, 4), 1, ())),
        (np.ones((2, 4, 3)), ((2, 4, 3), 1, ())),
    ],
    ids=["number", "vector", "matrix", "three_axes"],
)
def test_a_value_is_laid_out_dim_time_seq(value, dimensions):
    # Unless told otherwise, every axis of a value is a dim axis and the value
    # holds one time step - for both classes alike.
    for node in (Constant(value=value), Parameter(value=value)):
        assert node.shape.dimensions == dimensions
        assert node.value is not None
        assert node.value.shape == node.shape.tuple


@pytest.mark.parametrize(
    "axes, dimensions, order",
    [
        ({"dim": 0, "time": 1, "seq": 2}, ((2,), 3, (4,)), (0, 1, 2)),
        # The axes time and seq leave are the dim axes.
        ({"time": 1}, ((2, 4), 3, ()), (0, 2, 1)),
        ({"seq": (1, 2)}, ((2,), 1, (3, 4)), (0, 1, 2)),
        # dim names its axes in the order it wants them.
        ({"dim": (2, 0), "time": 1}, ((4, 2), 3, ()), (2, 0, 1)),
        ({"dim": -1, "time": 0, "seq": 1}, ((4,), 2, (3,)), (2, 0, 1)),
    ],
    ids=["all_three", "time_only", "seq_only", "dim_reordered", "negative"],
)
def test_dim_time_and_seq_name_the_axes_of_a_value(axes, dimensions, order):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    for node in (Constant(value=values, **axes), Parameter(value=values, **axes)):
        assert node.shape.dimensions == dimensions
        # The value is laid out (*dim, time, *seq): the axes moved into place,
        # and a time step of one added where the value has no time axis.
        assert node.value is not None
        np.testing.assert_array_equal(
            node.value, np.transpose(values, order).reshape(node.shape.tuple)
        )


def test_a_value_without_dim_axes_keeps_a_dim_of_one():
    window = Parameter("window", value=[1.0, 2.0, 3.0, 4.0], time=0)

    assert window.shape.dimensions == ((1,), 4, ())


def test_a_vector_value_becomes_a_column():
    constant = Constant("column", value=[1.0, 2.0, 3.0])

    np.testing.assert_array_equal(constant.value, [[1.0], [2.0], [3.0]])


def test_a_parameter_without_value_is_drawn_with_the_shape_it_is_given():
    x = Input("drawn_x")
    parameter = Parameter("drawn", dim=(3, 2), time=4, seq=5, initializer="ones")
    model = Modely(
        "drawn_model",
        inputs=[x],
        outputs=[Output("drawn_x_out", x.last()), Output("drawn_out", parameter)],
    ).build()

    assert parameter.shape.dimensions == ((3, 2), 4, (5,))
    assert parameter.param is not None
    assert tuple(parameter.param.shape) == (3, 2, 4, 5)
    result = model({"drawn_x": np.zeros((2, 1, 1), dtype=np.float32)})["drawn_out"]
    np.testing.assert_allclose(to_numpy(result), np.ones((2, 3, 2, 4, 5)))


def test_dim_time_and_seq_must_name_each_axis_of_a_value_once():
    values = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="time is one axis"):
        Parameter("two_times", value=values, time=(1, 2))
    with pytest.raises(ValueError, match="dim axis 3 is not an axis"):
        Parameter("out_of_range", value=values, dim=3)
    # Axis 2 is left out, and then axis 1 taken twice.
    with pytest.raises(ValueError, match="must take each axis"):
        Parameter("left_out", value=values, dim=0, time=1)
    with pytest.raises(ValueError, match="must take each axis"):
        Constant("taken_twice", value=values, dim=(0, 1), time=1, seq=2)


def test_a_matrix_value_given_its_time_axis_is_a_dim_by_time_stream():
    # time=1: the columns of the matrix are its time steps, the rows its dim.
    x = Input("matrix_x", dim=3)
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    parameter = Parameter("matrix_parameter", value=values, time=1)
    model = Modely(
        "matrix_model", inputs=[x], outputs=[Output("matrix_out", x.sw(4) + parameter)]
    ).build()

    assert parameter.shape.dimensions == ((3,), 4, ())
    batch = np.ones((2, 3, 4), dtype=np.float32)
    result = to_numpy(model({"matrix_x": batch})["matrix_out"])
    np.testing.assert_allclose(result, batch + values)


def test_only_a_parameter_is_trained():
    x = Input("trained_x")
    parameter = Parameter("trained_gain", value=[1.0])
    constant = Constant("fixed_gain", value=[1.0])
    model = Modely(
        "trained_model",
        inputs=[x],
        outputs=[Output("trained_out", x.last() * parameter * constant)],
    ).build()

    assert model.model is not None
    trainable = model.model.trainable_weights
    fixed = model.model.non_trainable_weights
    assert any(weight is parameter.param for weight in trainable)
    assert not any(weight is constant.constant for weight in trainable)
    assert any(weight is constant.constant for weight in fixed)


def test_a_constant_requires_a_value_and_names_its_axes_like_a_parameter():
    with pytest.raises(ValueError, match="requires a value"):
        Constant("no_value", value=None)
    with pytest.raises(ValueError, match="must not be empty"):
        Constant("empty_value", value=[])
    # With a value, dim is an axis of it, not a size.
    with pytest.raises(ValueError, match="dim axis 1 is not an axis"):
        Constant("given_dim", value=[1.0], dim=1)


def test_values_given_their_axes_round_trip_through_save(tmp_path):
    x = Input("axes_x")
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    parameter = Parameter("axes_parameter", value=values, dim=(2, 0), time=1)
    constant = Constant("axes_constant", value=[[1.0, 2.0, 3.0]], time=1)
    model = Modely(
        "axes_model",
        inputs=[x],
        outputs=[
            Output("axes_x_out", x.last()),
            Output("axes_parameter_out", parameter),
            Output("axes_constant_out", constant),
        ],
    ).build()
    data = {"axes_x": np.zeros((1, 1, 1), dtype=np.float32)}
    before = model(data)

    model.save(tmp_path / "axes", weights=False)
    restored = Modely.load(tmp_path / "axes")
    nodes = {node.name: node for node in restored.order}

    restored_parameter, restored_constant = (
        nodes["axes_parameter"],
        nodes["axes_constant"],
    )
    assert isinstance(restored_parameter, Parameter)
    assert isinstance(restored_constant, Constant)
    assert restored_parameter.shape.dimensions == ((4, 2), 3, ())
    assert restored_constant.shape.dimensions == ((1,), 3, ())
    after = restored(data)
    for name in ("axes_parameter_out", "axes_constant_out"):
        np.testing.assert_array_equal(to_numpy(after[name]), to_numpy(before[name]))
