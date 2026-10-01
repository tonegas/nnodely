import warnings

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
    parameter_with_dim = Parameter(
        "parameter_with_dim", value=np.ones((1, 3, 4), dtype=np.float32), dim=1
    )

    assert parameter.shape.dimensions == ((2,), 3, (4,))
    assert parameter_with_dim.shape.dimensions == ((1,), 3, (4,))


def test_parameter_constant_model_inference():
    x = Input("x", dim=1)
    parameter = Parameter("param1", value=[[1.0]])
    constant = Constant("const1", value=[[1.0]])

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
    parameter = Parameter("param_exp", value=np.zeros((4, 1), dtype=np.float32))

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
        (np.ones((2, 4)), ((2,), 4, ())),
        (np.ones((2, 4, 3)), ((2,), 4, (3,))),
    ],
    ids=["number", "vector", "matrix", "sequence"],
)
def test_a_value_is_laid_out_dim_time_seq(value, dimensions):
    # A number or a vector is one time step of its dim, a matrix is (dim,
    # time), and further axes are sequence axes - for both classes alike.
    for node in (Constant(value=value), Parameter(value=value)):
        assert node.shape.dimensions == dimensions
        assert node.value is not None
        assert node.value.shape == node.shape.tuple


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


def test_a_value_overrides_the_shape_given_with_it():
    with pytest.warns(UserWarning, match="overrides the dim, time"):
        overridden = Parameter("overridden", value=np.ones((2, 3)), dim=4, time=1)
    assert overridden.shape.dimensions == ((2,), 3, ())

    # An axis given the size the value has is not overridden: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agreeing = Parameter("agreeing", value=np.ones((2, 3)), dim=2)
    assert agreeing.shape.dimensions == ((2,), 3, ())


def test_a_matrix_value_given_with_its_dim_is_a_dim_by_time_stream():
    # dim=3 was once read as a number of axes: the stream became (3, 1) and
    # the model failed to reshape its twelve values.
    x = Input("matrix_x", dim=3)
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    parameter = Parameter("matrix_parameter", value=values, dim=3)
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


def test_a_constant_takes_its_shape_from_its_value_only():
    with pytest.raises(ValueError, match="requires a value"):
        Constant("no_value", value=None)
    with pytest.raises(ValueError, match="must not be empty"):
        Constant("empty_value", value=[])
    with pytest.raises(TypeError):
        Constant("given_dim", value=[1.0], dim=1)  # type: ignore[call-arg]
