import numpy as np

from nnodely import Input, Output, Modely, Parameter, Constant


def test_parameter_constant_shapes():
    p = Parameter("param1", value=[1.0])
    c = Constant("const1", value=[1.0])

    assert p.shape.dim == (1,)
    assert c.shape.dim == (1,)


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
    assert np.allclose(
        result["x_out"].cpu().detach().numpy(), [[[2.0]], [[2.0]], [[2.0]], [[2.0]]]
    )
    # assert np.allclose(result["x_out"], [[[2.0]], [[2.0]], [[2.0]], [[2.0]]])
    assert constant.constant is not None
    assert constant.constant.shape == (1, 1)
    assert np.allclose(constant.constant.numpy(), [[1.0]])
    assert parameter.param is not None
    assert parameter.param.shape == (1, 1)
    assert np.allclose(parameter.param.numpy(), [[1.0]])


if __name__ == "__main__":
    test_parameter_constant_model_inference()
