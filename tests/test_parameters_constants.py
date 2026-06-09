import numpy as np

from nnodely import Input, Output, Modely, Parameter, Constant


def test_parameter_constant_shapes():
    p = Parameter("param1", value=[1.0])
    c = Constant("const1", value=[1.0])

    assert p.shape == (1,)
    assert c.shape == (1,)


def test_parameter_constant_model_inference():
    x = Input("x", dim=1)
    parameter = Parameter("param1", value=[1.0])
    constant = Constant("const1", value=[1.0])

    y = x.sw(1) * parameter + constant
    out = Output("x_out", y)

    model = Modely("model", inputs=[x], outputs=[out])
    model.build()

    dummy_x = np.ones((4, 1, 1), dtype=np.float32)
    result = model({"x": dummy_x})

    assert "x_out" in result
    assert result["x_out"].shape == (4, 1, 1)
    assert np.allclose(
        result["x_out"].cpu().detach().numpy(), [[[2.0]], [[2.0]], [[2.0]], [[2.0]]]
    )
    assert parameter.param is not None
    if parameter.param is not None:
        assert parameter.param.shape == (1, 1)
        assert np.allclose(parameter.param.numpy(), [[1.0]])
    assert np.allclose(constant.value_numpy, [[1.0]])
