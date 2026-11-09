import numpy as np
import pytest
from conftest import to_numpy

from nnodely.core.dag import _parameters
from nnodely import (
    Constant,
    DataLoader,
    Exp,
    Fir,
    Input,
    Linear,
    Loop,
    Modely,
    Output,
    Parameter,
)
from nnodely import TimeSelect


def test_parameter_constant_shapes():
    p = Parameter("param1", value=[1.0])
    c = Constant("const1", value=[1.0])

    assert p.shape.dim == (1,)
    assert c.shape.dim == (1,)


def test_parameter_and_constant_declaration():

    ## possible way to create a Constant (value is mandatory)
    c1 = Constant("const1", value=[1.0])
    c2 = Constant(
        "const2",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        time=2,
        dim=5,
    )
    c3 = Constant(
        "const3",
        value=[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
        time=2,
        dim=5,
    )
    c4 = Constant("const4", value=np.ones((2, 3), dtype=np.float32), time=2, dim=3)
    c5 = Constant(
        "const5", value=np.ones((2, 3, 4), dtype=np.float32), time=2, dim=(3, 4)
    )
    c6 = Constant(
        "const6",
        value=np.ones((2, 3, 4, 5), dtype=np.float32),
        seq=5,
        time=3,
        dim=(2, 4),
    )
    c7 = Constant(
        "const7",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        seq=2,
        time=2,
        dim=3,
    )
    c8 = Constant(
        "const8",
        value=np.ones((2, 3, 4, 5), dtype=np.float32),
        seq=5,
        time=3,
        dim=(2, 4),
    )
    c9 = Constant("const9", value=np.ones((6), dtype=np.float32), time=2, dim=3)

    ## constant error
    with pytest.raises(ValueError, match="requires a value"):
        Constant("no_value", value=None)
    with pytest.raises(ValueError, match="shape error"):
        Constant("error1", value=[1.0, 2.0, 3.0], time=2, dim=5)
    with pytest.raises(ValueError, match="shape error"):
        Constant(
            "error2",
            value=np.ones((2, 3, 4), dtype=np.float32),
            seq=2,
            time=2,
            dim=(3, 4),
        )

    # possible way to create a Parameter (value is optional)
    p1 = Parameter("param1", value=[1.0])
    p2 = Parameter(
        "param2",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        time=2,
        dim=5,
    )
    p3 = Parameter(
        "param3",
        value=[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
        time=2,
        dim=5,
    )
    p4 = Parameter("param4", value=np.ones((2, 3), dtype=np.float32), time=2, dim=3)
    p5 = Parameter(
        "param5", value=np.ones((2, 3, 4), dtype=np.float32), time=2, dim=(3, 4)
    )
    p6 = Parameter(
        "param6",
        value=np.ones((2, 3, 4, 5), dtype=np.float32),
        seq=5,
        time=3,
        dim=(2, 4),
    )
    p7 = Parameter(
        "param7",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        seq=2,
        time=2,
        dim=3,
    )
    p8 = Parameter(
        "param8",
        value=np.ones((2, 3, 4, 5), dtype=np.float32),
        seq=5,
        time=3,
        dim=(2, 4),
    )
    p9 = Parameter("param9", value=np.ones((6), dtype=np.float32), time=2, dim=3)
    p10 = Parameter("param10", dim=3, time=2, seq=2, initializer="ones")
    p11 = Parameter("param11", dim=(3, 4), time=2, seq=2, initializer="ones")
    p12 = Parameter("param12", dim=(3, 4, 5), time=2, seq=2, initializer="ones")

    # Each takes the shape it declares, its value laid out (*seq, time, *dim).
    declared = [
        ((c1, p1), ((), None, (1,))),
        ((c2, p2, c3, p3), ((), 2, (5,))),
        ((c4, p4, c9, p9), ((), 2, (3,))),
        ((c5, p5), ((), 2, (3, 4))),
        ((c6, p6, c8, p8), ((5,), 3, (2, 4))),
        ((c7, p7), ((2,), 2, (3,))),
        ((p10,), ((2,), 2, (3,))),
        ((p11,), ((2,), 2, (3, 4))),
        ((p12,), ((2,), 2, (3, 4, 5))),
    ]
    for nodes, dimensions in declared:
        for node in nodes:
            assert node.shape.dimensions == dimensions
            if node.value is not None:
                assert node.value.shape == node.shape.tuple
    # A list of values and the matrix of them are one value.
    np.testing.assert_array_equal(c2.value, c3.value)
    np.testing.assert_array_equal(p7.value, np.arange(1, 13).reshape(2, 2, 3))
    assert all(node.value is None for node in (p10, p11, p12))


def test_parameter_shape_is_inferred_like_constant():
    parameter = Parameter(
        "parameter_sequence", value=np.ones((2, 3, 4), dtype=np.float32)
    )
    parameter_with_axes = Parameter(
        "parameter_with_axes",
        value=np.ones((1, 3, 4), dtype=np.float32),
        seq=4,
        time=3,
        dim=1,
    )

    assert parameter.shape.dimensions == ((), None, (2, 3, 4))
    assert parameter_with_axes.shape.dimensions == ((4,), 3, (1,))


def test_parameter_constant_model_inference():
    x = Input("x", dim=1, sw=1)
    parameter = Parameter("param1", value=[1.0])
    constant = Constant("const1", value=[1.0])

    y = x * parameter + constant
    out = Output("x_out", y)

    model = Modely("model", inputs=[x], outputs=[out])
    model.build()
    model.summary()
    from pprint import pprint

    print("model order:")
    pprint(model._order)

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
    assert constant.constant.shape == (1,)
    np.testing.assert_allclose(
        to_numpy(constant.constant),
        np.array([1.0]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert parameter.param is not None
    assert parameter.param.shape == (1,)
    np.testing.assert_allclose(
        to_numpy(parameter.param),
        np.array([1.0]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_parameter_through_a_layer_keeps_the_batch_axis():
    x = Input("x", dim=4, sw=1)
    parameter = Parameter("param_exp", value=np.zeros(4, dtype=np.float32))

    y = x * Exp()([parameter])
    model = Modely("model_param_exp", inputs=[x], outputs=[Output("x_out", y)])
    model.build()

    batch = np.arange(3 * 4, dtype=np.float32).reshape(3, 1, 4)
    result = to_numpy(model({"x": batch})["x_out"])

    assert result.shape == (3, 1, 4)
    np.testing.assert_allclose(result, batch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "value, dimensions",
    [
        (2.0, ((), None, (1,))),
        ([1.0, 2.0, 3.0], ((), None, (3,))),
        (np.ones((2, 4)), ((), None, (2, 4))),
        (np.ones((2, 4, 3)), ((), None, (2, 4, 3))),
    ],
    ids=["number", "vector", "matrix", "three_axes"],
)
def test_a_value_is_laid_out_dim_time_seq(value, dimensions):
    # Unless told otherwise, every axis of a value is a dim axis and the value
    # has no window - for both classes alike. It is held as given.
    for node in (Constant(value=value), Parameter(value=value)):
        assert node.shape.dimensions == dimensions
        assert node.value is not None
        assert node.value.shape == node.shape.dim


@pytest.mark.parametrize(
    "sizes, dimensions, order",
    [
        ({"seq": 2, "time": 3, "dim": 4}, ((2,), 3, (4,)), (0, 1, 2)),
        ({"seq": 4, "time": 3, "dim": 2}, ((4,), 3, (2,)), (2, 1, 0)),
        ({"time": 3, "dim": (2, 4)}, ((), 3, (2, 4)), (1, 0, 2)),
        ({"time": 3, "dim": (4, 2)}, ((), 3, (4, 2)), (1, 2, 0)),
        ({"seq": 3, "time": 2, "dim": 4}, ((3,), 2, (4,)), (1, 0, 2)),
    ],
    ids=["in_order", "all_moved", "time_first", "dim_reordered", "seq_first"],
)
def test_a_value_of_several_axes_is_moved_into_the_declared_order(
    sizes, dimensions, order
):
    # seq, time and dim are sizes, as for an Input: a value has them, in
    # any order, and is laid out (*seq, time, *dim) - its axes moved into
    # place, not its values read again in another order.
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    for node in (Constant(value=values, **sizes), Parameter(value=values, **sizes)):
        assert node.shape.dimensions == dimensions
        assert node.value is not None
        np.testing.assert_array_equal(node.value, np.transpose(values, order))


def test_axes_of_one_size_keep_their_order():
    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    # (3, 2, 2) declared: the window takes the axis of 3, the two axes of 2
    # are the dim, the first of them first.
    node = Parameter("same_sizes", value=values, time=3, dim=(2, 2))
    assert node.shape.dimensions == ((), 3, (2, 2))
    np.testing.assert_array_equal(node.value, np.transpose(values, (1, 0, 2)))


def test_a_list_of_values_fills_the_declared_shape_in_order():
    # As many values as the shape holds, laid out (*seq, time, *dim),
    # row-major: the first dim values are the first sample.
    values = np.arange(12, dtype=np.float32)
    for node in (
        Constant("flat_constant", value=values.tolist(), seq=2, time=2, dim=3),
        Parameter("flat_parameter", value=values, seq=2, time=2, dim=3),
    ):
        assert node.shape.dimensions == ((2,), 2, (3,))
        np.testing.assert_array_equal(node.value, values.reshape(2, 2, 3))


def test_a_list_and_its_matrix_give_one_value_in_a_model():
    x = Input("filled_x", sw=1)
    flat = Constant("filled_flat", value=list(range(10)), time=2, dim=5)
    matrix = Constant("filled_matrix", value=np.arange(10).reshape(2, 5), time=2, dim=5)
    moved = Constant("filled_moved", value=np.arange(10).reshape(2, 5).T, time=2, dim=5)
    model = Modely(
        "filled_model",
        inputs=[x],
        outputs=[
            Output("filled_x_out", x),
            Output("filled_flat_out", flat),
            Output("filled_matrix_out", matrix),
            Output("filled_moved_out", moved),
        ],
    ).build()

    result = model({"filled_x": np.zeros((3, 1, 1), dtype=np.float32)})
    expected = np.broadcast_to(np.arange(10).reshape(2, 5), (3, 2, 5))
    for name in ("filled_flat_out", "filled_matrix_out", "filled_moved_out"):
        np.testing.assert_array_equal(to_numpy(result[name]), expected)


@pytest.mark.parametrize(
    "shape, sizes, dimensions, order",
    [
        ((2, 4, 6), {"time": 4}, ((), 4, (2, 6)), (1, 0, 2)),
        ((2, 3, 5, 4), {"seq": 5, "time": 3}, ((5,), 3, (2, 4)), (2, 1, 0, 3)),
        ((4, 2, 6), {"time": 4}, ((), 4, (2, 6)), (0, 1, 2)),
        # Axes of one size keep their order: the window takes the first.
        ((2, 3, 2), {"time": 2}, ((), 2, (3, 2)), (0, 1, 2)),
        # Neither seq nor time: every axis is a dim axis, no window.
        ((2, 4, 6), {}, ((), None, (2, 4, 6)), (0, 1, 2)),
    ],
    ids=["time", "seq_and_time", "time_in_place", "same_sizes", "none"],
)
def test_without_dim_the_axes_seq_and_time_leave_are_the_dim(
    shape, sizes, dimensions, order
):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    for node in (Constant(value=values, **sizes), Parameter(value=values, **sizes)):
        assert node.shape.dimensions == dimensions
        assert node.value is not None
        np.testing.assert_array_equal(node.value, np.transpose(values, order))


def test_without_dim_a_value_that_seq_and_time_take_whole_has_one_feature():
    assert Parameter(
        "taken_list", value=[1.0, 2.0, 3.0, 4.0], time=4
    ).shape.dimensions == (
        (),
        4,
        (1,),
    )
    matrix = Constant("taken_matrix", value=np.ones((2, 4)), seq=2, time=4)
    assert matrix.shape.dimensions == ((2,), 4, (1,))
    # A list whose length is no axis of seq or time fills them, a dim of one.
    flat = Constant("taken_flat", value=np.arange(8.0), seq=2, time=4)
    assert flat.shape.dimensions == ((2,), 4, (1,))
    np.testing.assert_array_equal(flat.value, np.arange(8.0).reshape(2, 4, 1))


def test_a_value_whose_left_axes_are_its_dim_computes_and_round_trips(tmp_path):
    # time=4 takes the axis of 4; the dim is the axes left, (2, 6), in order.
    values = np.arange(48, dtype=np.float32).reshape(2, 4, 6)
    x = Input("left_x", dim=(2, 6), sw=4)
    offset = Constant("left_offset", value=values, time=4)
    model = Modely(
        "left_model", inputs=[x], outputs=[Output("left_out", x + offset)]
    ).build()
    data = {"left_x": np.zeros((3, 4, 2, 6), dtype=np.float32)}
    expected = np.broadcast_to(values.transpose(1, 0, 2), (3, 4, 2, 6))

    np.testing.assert_array_equal(to_numpy(model(data)["left_out"]), expected)
    model.save(tmp_path / "left", weights=False)
    restored = Modely.load(tmp_path / "left")
    np.testing.assert_array_equal(to_numpy(restored(data)["left_out"]), expected)


def test_a_value_without_dim_axes_keeps_a_dim_of_one():
    window = Parameter("window", value=[1.0, 2.0, 3.0, 4.0], time=4)

    assert window.shape.dimensions == ((), 4, (1,))


def test_a_vector_value_becomes_a_column():
    constant = Constant("column", value=[1.0, 2.0, 3.0])

    np.testing.assert_array_equal(constant.value, [1.0, 2.0, 3.0])


def test_a_parameter_without_value_is_drawn_with_the_shape_it_is_given():
    x = Input("drawn_x", sw=1)
    parameter = Parameter("drawn", dim=(3, 2), time=4, seq=5, initializer="ones")
    model = Modely(
        "drawn_model",
        inputs=[x],
        outputs=[Output("drawn_x_out", x), Output("drawn_out", parameter)],
    ).build()

    assert parameter.shape.dimensions == ((5,), 4, (3, 2))
    assert parameter.param is not None
    assert tuple(parameter.param.shape) == (5, 4, 3, 2)
    result = model({"drawn_x": np.zeros((2, 1, 1), dtype=np.float32)})["drawn_out"]
    np.testing.assert_allclose(to_numpy(result), np.ones((2, 5, 4, 3, 2)))


@pytest.mark.parametrize("cls", [Parameter, Constant])
def test_a_value_that_does_not_fill_the_declared_shape_is_refused(cls):
    # A list of another number of values.
    with pytest.raises(ValueError, match="3 values given where .* holds 10"):
        cls("short", value=[1.0, 2.0, 3.0], time=2, dim=5)
    # As many values, but sizes no order of the axes gives: one axis more,
    # or another size.
    with pytest.raises(ValueError, match="shape error, a value of shape"):
        cls("extra_axis", value=np.ones((2, 3, 2, 2, 5)), seq=5, time=3, dim=(2, 4))
    with pytest.raises(ValueError, match="shape error, a value of shape"):
        cls("other_sizes", value=np.ones((2, 3, 4)), seq=2, time=2, dim=(3, 4))
    # Without dim, time must find an axis of its size.
    with pytest.raises(ValueError, match="has no axes of the sizes seq and time"):
        cls("no_window_axis", value=np.ones((2, 3)), time=4)
    # A list without dim fills a dim of one.
    with pytest.raises(ValueError, match="12 values given where .* holds 4"):
        cls("list_no_dim", value=list(range(12)), time=4)


@pytest.mark.parametrize("cls", [Parameter, Constant])
def test_dim_time_and_seq_are_declared_as_for_an_input(cls):
    def make(**sizes):
        return cls("declared", value=np.ones(6), **sizes)

    for dim in (0, (2, 0), (), -1, 2.0):
        with pytest.raises(ValueError, match="dim is the size of each feature"):
            make(time=6, dim=dim)
    for time in (0, -1, (2, 3), 1.5, True):
        with pytest.raises(ValueError, match="time is the length of the window"):
            make(time=time, dim=3)
    # A sequence needs a window, and holds every step: none is dynamic.
    with pytest.raises(ValueError, match="a sequence needs a window"):
        make(seq=2, dim=3)
    for seq in (-1, "dyn", 0, (2, None)):
        with pytest.raises(ValueError, match="seq is the length of each sequence"):
            make(seq=seq, time=1, dim=3)


def test_a_parameter_without_value_is_declared_by_its_sizes_alone():
    assert Parameter("only_dim", dim=(3, 4)).shape.dimensions == ((), None, (3, 4))
    assert Parameter("one_feature").shape.dimensions == ((), None, (1,))
    assert Parameter("sized", seq=2, time=2, dim=3).shape.dimensions == (
        (2,),
        2,
        (3,),
    )
    with pytest.raises(ValueError, match="a sequence needs a window"):
        Parameter("no_window", seq=2)
    with pytest.raises(ValueError, match="dim is the size of each feature"):
        Parameter("no_feature", dim=0)


def test_an_input_has_at_least_one_feature():
    for dim in (0, (2, 0), (), -1):
        with pytest.raises(ValueError, match="dim is the size of each feature"):
            Input("featureless", dim=dim, sw=1)
    assert Input("featured", dim=(2, 3), sw=1).shape.dimensions == ((), 1, (2, 3))


def test_a_matrix_value_given_its_time_axis_is_a_dim_by_time_stream():
    # time=4, dim=3: the columns of the 3 x 4 matrix are its time steps, the
    # rows its dim.
    x = Input("matrix_x", dim=3, sw=4)
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    parameter = Parameter("matrix_parameter", value=values, time=4, dim=3)
    model = Modely(
        "matrix_model", inputs=[x], outputs=[Output("matrix_out", x + parameter)]
    ).build()

    assert parameter.shape.dimensions == ((), 4, (3,))
    batch = np.ones((2, 4, 3), dtype=np.float32)
    result = to_numpy(model({"matrix_x": batch})["matrix_out"])
    np.testing.assert_allclose(result, batch + values.T)


def test_only_a_parameter_is_trained():
    x = Input("trained_x", sw=1)
    parameter = Parameter("trained_gain", value=[1.0])
    constant = Constant("fixed_gain", value=[1.0])
    model = Modely(
        "trained_model",
        inputs=[x],
        outputs=[Output("trained_out", x * parameter * constant)],
    ).build()

    assert model.model is not None
    trainable = model.model.trainable_weights
    fixed = model.model.non_trainable_weights
    assert any(weight is parameter.param for weight in trainable)
    assert not any(weight is constant.constant for weight in trainable)
    assert any(weight is constant.constant for weight in fixed)


def test_a_constant_requires_a_value_and_declares_its_sizes_like_a_parameter():
    with pytest.raises(ValueError, match="requires a value"):
        Constant("no_value", value=None)
    with pytest.raises(ValueError, match="must not be empty"):
        Constant("empty_value", value=[])
    # With a value, dim is a size, as without one.
    assert Constant("given_dim", value=[1.0], dim=1).shape.dimensions == (
        (),
        None,
        (1,),
    )
    with pytest.raises(ValueError, match="shape error"):
        Constant("given_dim_2", value=[1.0], dim=2)


def test_values_given_their_sizes_round_trip_through_save(tmp_path):
    x = Input("axes_x", sw=1)
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    # Its axes moved into place when declared; saved as laid out.
    parameter = Parameter("axes_parameter", value=values, time=3, dim=(4, 2))
    constant = Constant("axes_constant", value=[1.0, 2.0, 3.0], time=3)
    model = Modely(
        "axes_model",
        inputs=[x],
        outputs=[
            Output("axes_x_out", x),
            Output("axes_parameter_out", parameter),
            Output("axes_constant_out", constant),
        ],
    ).build()
    data = {"axes_x": np.zeros((1, 1, 1), dtype=np.float32)}
    before = model(data)

    model.save(tmp_path / "axes", weights=False)
    restored = Modely.load(tmp_path / "axes")
    nodes = {node.name: node for node in restored._order}

    restored_parameter, restored_constant = (
        nodes["axes_parameter"],
        nodes["axes_constant"],
    )
    assert isinstance(restored_parameter, Parameter)
    assert isinstance(restored_constant, Constant)
    assert restored_parameter.shape.dimensions == ((), 3, (4, 2))
    assert restored_constant.shape.dimensions == ((), 3, (1,))
    after = restored(data)
    for name in ("axes_parameter_out", "axes_constant_out"):
        np.testing.assert_array_equal(to_numpy(after[name]), to_numpy(before[name]))


# ---------------------------------------------------------------------------
# Freezing: a Parameter, or every Parameter of a model
# ---------------------------------------------------------------------------


def _trainable_ids(model):
    assert model.model is not None
    return {id(weight) for weight in model.model.trainable_weights}


def test_freeze_and_unfreeze_a_parameter_or_a_whole_model():
    x = Input("freeze_x", dim=(2, 3), sw=2)
    # A Fir kernel is laid out (*dim, window, features), as its values are.
    a = Parameter("freeze_a", value=np.array([0.5] * 12).reshape(2, 3, 2))
    b = Parameter("freeze_b", dim=(2, 3, 2))
    fir1 = Fir(features=1, kernel=a, bias=False)([x])
    fir2 = Fir(features=1, kernel=b, bias=False)([x])
    model = Modely(
        "freeze_model", inputs=[x], outputs=[Output("freeze_out", fir1 + fir2)]
    ).build()
    both = {id(a.param), id(b.param)}

    assert model.freeze() is model
    assert (a.trainable, b.trainable) == (False, False)
    assert _trainable_ids(model) == set()
    model.unfreeze()
    assert (a.trainable, b.trainable) == (True, True)
    assert _trainable_ids(model) == both

    assert a.freeze() is a
    assert _trainable_ids(model) == {id(b.param)}
    b.freeze()
    assert _trainable_ids(model) == set()
    a.unfreeze()
    assert (a.trainable, b.trainable) == (True, False)
    assert _trainable_ids(model) == {id(a.param)}


def test_a_parameter_frozen_before_build_is_built_frozen():
    x = Input("prebuild_x", sw=1)
    a = Parameter("prebuild_a", value=[1.0]).freeze()
    b = Parameter("prebuild_b", value=[1.0], trainable=False)
    c = Parameter("prebuild_c", value=[1.0])
    model = Modely(
        "prebuild_model",
        inputs=[x],
        outputs=[Output("prebuild_out", x * a + x * b + c)],
    ).build()

    assert _trainable_ids(model) == {id(c.param)}


@pytest.mark.slow
def test_a_frozen_parameter_keeps_its_value_through_training():
    # out = a x + b x against 5 on x = 1: the gradient of (a + b - 5)^2 is
    # 2 (a + b - 5) for either, so one SGD step at lr 0.1 from a = b = 1
    # moves the trainable one by 0.6.
    x = Input("frozen_x", sw=1)
    a = Parameter("frozen_a", value=[1.0])
    b = Parameter("frozen_b", value=[1.0])
    out = Output("frozen_out", x * a + x * b)
    model = Modely("frozen_model", inputs=[x], outputs=[out])
    model.minimize("frozen_fit", out, Input("frozen_y", sw=1))
    model.build()
    data = DataLoader(
        model,
        source={"frozen_x": np.ones(4), "frozen_y": np.full(4, 5.0)},
    )

    def step():
        model.train(data, epochs=1, batch_size=4, optimizer="sgd", lr=0.1, printer=None)

    a.freeze()
    step()
    np.testing.assert_allclose(to_numpy(a.param).ravel(), [1.0])
    np.testing.assert_allclose(to_numpy(b.param).ravel(), [1.6], rtol=1e-5)

    # Now the other way round: 2 (1 + 1.6 - 5) = -4.8, so a goes to 1.48.
    a.unfreeze()
    b.freeze()
    step()
    np.testing.assert_allclose(to_numpy(a.param).ravel(), [1.48], rtol=1e-5)
    np.testing.assert_allclose(to_numpy(b.param).ravel(), [1.6], rtol=1e-5)


def test_model_freeze_reaches_layer_weights_called_models_and_loop_bodies():
    # A block called as a model, with a Parameter of its own.
    u = Input("reach_u", sw=1)
    gain = Parameter("reach_gain", value=[2.0])
    block = Modely(
        "reach_block", inputs=[u], outputs=[Output("reach_block_out", u * gain)]
    ).build()
    # A Loop body whose Linear makes its own weights.
    q = Input("reach_q", sw=1)
    body = Modely(
        "reach_body",
        inputs=[q],
        outputs=[Output("reach_next", Linear(features=1)(q))],
    ).build()
    seed = Input("reach_seed", seq=3, sw=1)
    loop = Loop(f=body, feedback={q: "reach_next"}, init={q: seed})()

    x = Input("reach_x", sw=2)
    model = Modely(
        "reach_model",
        inputs=[x, seed],
        outputs=[
            Output("reach_fir", Fir(features=1)([x])),
            Output("reach_called", block([TimeSelect(-1)(x)])),
            Output("reach_loop", loop),
        ],
    ).build()
    weights = _trainable_ids(model)
    # The Fir kernel and bias, the gain, the Linear kernel and bias.
    assert len(weights) == 5

    model.freeze()
    assert _trainable_ids(model) == set()
    assert not gain.trainable
    model.unfreeze()
    assert _trainable_ids(model) == weights


def _frozen(model):
    return sorted(
        parameter.name for parameter in model.parameters if not parameter.trainable
    )


def _still_trainable(model):
    return {
        id(parameter.param) for parameter in model.parameters if parameter.trainable
    }


def test_freeze_and_unfreeze_parameters_by_name():
    x = Input("named_x", sw=1)
    k = Parameter("named_k", value=[1.0])
    g = Parameter("named_g", value=[2.0])
    h = Parameter("named_h", value=[3.0])
    linear = Linear(features=1, name="named_lin")(x)
    model = Modely(
        "named", inputs=[x], outputs=[Output("named_out", x * k + x * g + h + linear)]
    ).build()

    # A list of names, or one; the weights a layer made, by theirs.
    assert model.freeze(["named_k", "named_g"]) is model
    assert _frozen(model) == ["named_g", "named_k"]
    model.freeze("named_lin_kernel")
    assert _frozen(model) == ["named_g", "named_k", "named_lin_kernel"]
    assert model.unfreeze(["named_k", "named_lin_kernel"]) is model
    assert _frozen(model) == ["named_g"]
    assert _trainable_ids(model) == _still_trainable(model)


def test_freeze_and_unfreeze_a_model_it_reaches_by_name():
    u = Input("reached_u", sw=1)
    inner_gain = Parameter("reached_inner_gain", value=[3.0])
    inner = Modely(
        "reached_inner",
        inputs=[u],
        outputs=[Output("reached_inner_out", u * inner_gain)],
    )
    v = Input("reached_v", sw=1)
    gain = Parameter("reached_gain", value=[2.0])
    block = Modely(
        "reached_block",
        inputs=[v],
        outputs=[Output("reached_block_out", inner([v]) * gain)],
    )
    q = Input("reached_q", sw=1)
    body = Modely(
        "reached_body",
        inputs=[q],
        outputs=[Output("reached_next", Linear(features=1, name="reached_lin")(q))],
    )
    seed = Input("reached_seed", seq=3, sw=1)
    loop = Loop(f=body, feedback={q: "reached_next"}, init={q: seed})()
    x = Input("reached_x", sw=1)
    k = Parameter("reached_k", value=[1.0])
    model = Modely(
        "reached",
        inputs=[x, seed],
        outputs=[
            Output("reached_called", block([x]) * k),
            Output("reached_looped", loop),
        ],
    ).build()

    # A block it calls takes the block it calls in turn along; that one
    # alone is picked by its own name.
    model.freeze("reached_block")
    assert _frozen(model) == ["reached_gain", "reached_inner_gain"]
    model.unfreeze("reached_inner")
    assert _frozen(model) == ["reached_gain"]
    # The body of a Loop, with the weights its layer made, and a Parameter,
    # in one list.
    model.freeze(["reached_body", "reached_k"])
    assert _frozen(model) == [
        "reached_gain",
        "reached_k",
        "reached_lin_bias",
        "reached_lin_kernel",
    ]
    model.unfreeze(["reached_body", "reached_block"])
    assert _frozen(model) == ["reached_k"]
    assert _trainable_ids(model) == _still_trainable(model)


def test_freeze_refuses_a_name_it_cannot_resolve_and_changes_nothing():
    x = Input("unresolved_x", sw=1)
    k = Parameter("unresolved_k", value=[1.0])
    shared = Parameter("unresolved_shared", value=[1.0])
    w = Input("unresolved_w", sw=1)
    named_alike = Modely(
        "unresolved_shared", inputs=[w], outputs=[Output("unresolved_w_out", w * 2.0)]
    )
    model = Modely(
        "unresolved",
        inputs=[x],
        outputs=[Output("unresolved_out", x * k + named_alike([x]) * shared)],
    )

    # Every name is checked first: the known one is not frozen either.
    with pytest.raises(ValueError, match="no Parameter and no model named 'nope'"):
        model.freeze(["unresolved_k", "nope"])
    assert k.trainable
    with pytest.raises(ValueError, match="both named 'unresolved_shared'"):
        model.unfreeze("unresolved_shared")
    with pytest.raises(TypeError, match="got Parameter"):
        model.freeze([k])  # type: ignore[list-item]
    assert _frozen(model) == []


def test_the_trainable_flag_is_saved_with_the_model(tmp_path):
    x = Input("saved_flag_x", sw=2)
    frozen = Parameter("saved_frozen", value=[2.0]).freeze()
    free = Parameter("saved_free", value=[3.0])
    fir = Fir(features=1, name="saved_flag_fir")([x])
    fir._preds[2].freeze()  # its bias, a Parameter it made
    model = Modely(
        "saved_flag_model",
        inputs=[x],
        outputs=[Output("saved_flag_out", TimeSelect(-1)(x) * frozen + free + fir)],
    ).build()

    model.save(tmp_path / "saved_flag")
    restored = Modely.load(tmp_path / "saved_flag")
    parameters = {p.name: p for p in _parameters(restored)}

    assert not parameters["saved_frozen"].trainable
    assert parameters["saved_free"].trainable
    assert parameters["saved_flag_fir_kernel"].trainable
    assert not parameters["saved_flag_fir_bias"].trainable
    assert len(_trainable_ids(restored)) == 2


def test_model_parameters_lists_every_parameter_and_prints_a_table():
    x = Input("table_x", dim=(2, 3), sw=2)
    # A Fir kernel is laid out (*dim, window, features), as its values are.
    a = Parameter("a", value=np.full((2, 3, 2), 0.5))
    b = Parameter("b", dim=(2, 3, 2)).freeze()
    c = Parameter("c", value=np.ones(4), time=4)
    fir1 = Fir(features=1, kernel=a, bias=False)([x])
    fir2 = Fir(features=1, kernel=b, bias=False)([x])
    linear = Linear(features=2, name="lin")([fir1 + fir2])
    model = Modely(
        "table_model",
        inputs=[x],
        outputs=[Output("table_out", linear), Output("table_c", c)],
    )

    # A list of the Parameter objects, in the order the graph reaches them -
    # before the model is built too.
    parameters = model.parameters
    assert isinstance(parameters, list)
    assert parameters[:2] == [a, b]
    assert [p.name for p in parameters] == ["a", "b", "lin_kernel", "lin_bias", "c"]

    model.build()
    assert str(model.parameters) == "\n".join(
        [
            "Parameters of table_model",
            "┏━━━━━━━━━━━━┳━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓",
            "┃ name       ┃ seq ┃ time ┃ dim       ┃ values ┃ trainable ┃",
            "┡━━━━━━━━━━━━╇━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩",
            "│ a          │ ()  │ None │ (2, 3, 2) │ 12     │ yes       │",
            "│ b          │ ()  │ None │ (2, 3, 2) │ 12     │ no        │",
            "│ lin_kernel │ ()  │ None │ (2, 3, 2) │ 12     │ yes       │",
            "│ lin_bias   │ ()  │ None │ (2,)      │ 2      │ yes       │",
            "│ c          │ ()  │ 4    │ (1,)      │ 4      │ yes       │",
            "└────────────┴─────┴──────┴───────────┴────────┴───────────┘",
            " 5 parameters, 42 values, 30 trainable",
        ]
    )
    assert repr(model.parameters) == str(model.parameters)


def test_model_parameters_prints_a_table_per_model():
    u = Input("per_model_u", sw=1)
    shared = Parameter("per_model_shared", value=2.0)
    frozen = Parameter("per_model_frozen", value=1.0).freeze()
    body = Modely(
        "per_model_body",
        inputs=[u],
        outputs=[Output("per_model_body_out", u * shared + frozen)],
    )
    z = Input("per_model_z", sw=1)
    top = Modely(
        "per_model_top",
        inputs=[z],
        outputs=[Output("per_model_out", body([z]) * shared)],
    )

    # Still one list, in the order the graph reaches them.
    assert top.parameters == [shared, frozen]
    top_rule = "┏━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓"
    header = "┃ name             ┃ seq ┃ time ┃ dim  ┃ values ┃ trainable ┃"
    open_columns = "├──────────────────┬─────┬──────┬──────┬────────┬───────────┤"
    shared_row = "│ per_model_shared │ ()  │ None │ (1,) │ 1      │ yes       │"
    frozen_row = "│ per_model_frozen │ ()  │ None │ (1,) │ 1      │ no        │"
    bottom = "└──────────────────┴─────┴──────┴──────┴────────┴───────────┘"
    # A row of the model's name opens its Parameters. One the model shares
    # with a block it calls is listed once, under the model, the first to
    # reach it.
    assert str(top.parameters) == "\n".join(
        [
            "Parameters of per_model_top",
            top_rule,
            header,
            "┡━━━━━━━━━━━━━━━━━━┻━━━━━┻━━━━━━┻━━━━━━┻━━━━━━━━┻━━━━━━━━━━━┩",
            "│ per_model_top: 1 parameters, 1 values, 1 trainable        │",
            open_columns,
            shared_row,
            "├──────────────────┴─────┴──────┴──────┴────────┴───────────┤",
            "│ per_model_body: 1 parameters, 1 values, 0 trainable       │",
            open_columns,
            frozen_row,
            bottom,
            " 2 parameters, 2 values, 1 trainable",
        ]
    )
    # The block alone is one model: one table.
    assert str(body.parameters).splitlines()[3:] == [
        "┡━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩",
        shared_row,
        frozen_row,
        bottom,
        " 2 parameters, 2 values, 1 trainable",
    ]
    # A model without Parameters of its own still names the block they are of.
    wrap = Modely(
        "per_model_wrap",
        inputs=[z],
        outputs=[Output("per_model_wrap_out", body([z]))],
    )
    assert str(wrap.parameters).splitlines()[4] == (
        "│ per_model_body: 2 parameters, 2 values, 1 trainable       │"
    )
