import numpy as np
from conftest import to_numpy

from nnodely import Input, Output, Modely, Parameter, Constant


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
