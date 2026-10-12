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


# ---------------------------------------------------------------------------
# Freezing: a Parameter, or every Parameter of a model
# ---------------------------------------------------------------------------


def _trainable_ids(model):
    assert model.model is not None
    return {id(weight) for weight in model.model.trainable_weights}


def test_freeze_and_unfreeze_a_parameter_or_a_whole_model():
    x = Input("freeze_x", dim=(2, 3))
    a = Parameter(
        "freeze_a", value=np.array([0.5] * 12).reshape(2, 3, 2), dim=(0, 1), time=2
    )
    b = Parameter("freeze_b", dim=(2, 3, 2))
    fir1 = Fir(features=1, kernel=a, bias=False)([x.sw(2)])
    fir2 = Fir(features=1, kernel=b, bias=False)([x.sw(2)])
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
    x = Input("prebuild_x")
    a = Parameter("prebuild_a", value=[1.0]).freeze()
    b = Parameter("prebuild_b", value=[1.0], trainable=False)
    c = Parameter("prebuild_c", value=[1.0])
    model = Modely(
        "prebuild_model",
        inputs=[x],
        outputs=[Output("prebuild_out", x.last() * a + x.last() * b + c)],
    ).build()

    assert _trainable_ids(model) == {id(c.param)}


@pytest.mark.slow
def test_a_frozen_parameter_keeps_its_value_through_training():
    # out = a x + b x against 5 on x = 1: the gradient of (a + b - 5)^2 is
    # 2 (a + b - 5) for either, so one SGD step at lr 0.1 from a = b = 1
    # moves the trainable one by 0.6.
    x = Input("frozen_x")
    a = Parameter("frozen_a", value=[1.0])
    b = Parameter("frozen_b", value=[1.0])
    out = Output("frozen_out", x.last() * a + x.last() * b)
    model = Modely("frozen_model", inputs=[x], outputs=[out])
    model.minimize("frozen_fit", out, Input("frozen_y").last())
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
    u = Input("reach_u")
    gain = Parameter("reach_gain", value=[2.0])
    block = Modely(
        "reach_block", inputs=[u], outputs=[Output("reach_block_out", u.last() * gain)]
    ).build()
    # A Loop body whose Linear makes its own weights.
    q = Input("reach_q")
    body = Modely(
        "reach_body",
        inputs=[q],
        outputs=[Output("reach_next", Linear(features=1)(q.last()))],
    ).build()
    seed = Input("reach_seed", seq=3)
    loop = Loop(f=body, callback={q: "reach_next"}, init={q: seed})()

    x = Input("reach_x")
    model = Modely(
        "reach_model",
        inputs=[x, seed],
        outputs=[
            Output("reach_fir", Fir(features=1)([x.sw(2)])),
            Output("reach_called", block([x.last()])),
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


def test_the_trainable_flag_is_saved_with_the_model(tmp_path):
    x = Input("saved_flag_x")
    frozen = Parameter("saved_frozen", value=[2.0]).freeze()
    free = Parameter("saved_free", value=[3.0])
    fir = Fir(features=1, name="saved_flag_fir")([x.sw(2)])
    fir.preds[2].freeze()  # its bias, a Parameter it made
    model = Modely(
        "saved_flag_model",
        inputs=[x],
        outputs=[Output("saved_flag_out", x.last() * frozen + free + fir)],
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
    x = Input("table_x", dim=(2, 3))
    a = Parameter("a", value=np.full((2, 3, 2), 0.5), dim=(0, 1), time=2)
    b = Parameter("b", dim=(2, 3, 2)).freeze()
    c = Parameter("c", value=np.ones((1, 4)), seq=1)
    fir1 = Fir(features=1, kernel=a, bias=False)([x.sw(2)])
    fir2 = Fir(features=1, kernel=b, bias=False)([x.sw(2)])
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
            "name        dim        time  seq   values  trainable",
            "----------  ---------  ----  ----  ------  ---------",
            "a           (2, 3)     2     -     12      yes",
            "b           (2, 3, 2)  1     -     12      no",
            "lin_kernel  (2, 3, 2)  1     -     12      yes",
            "lin_bias    (2,)       1     -     2       yes",
            "c           (1,)       1     (4,)  4       yes",
            "5 parameters, 42 values, 30 trainable",
        ]
    )
    assert repr(model.parameters) == str(model.parameters)
