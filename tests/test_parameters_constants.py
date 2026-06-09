import numpy as np

from nnodely import Input, Output, Modely, Parameter, Constant


def test_parameter_constant_shapes():
    p = Parameter("param1", value=[1.0])
    c = Constant("const1", value=[1.0])

    assert p.shape == (1,)
    assert c.shape == (1,)


def test_parameter_constant_model_inference():
    x = Input("x", dim=1)
    param = Parameter("param1", value=[1.0])
    const = Constant("const1", value=[1.0])

    y = x.sw(1) * param + const
    out = Output("x_out", y)

    model = Modely("model", inputs=[x], outputs=[out])
    model.build()

    dummy_x = np.ones((4, 1, 1), dtype=np.float32)
    result = model({"x": dummy_x})

    assert "x_out" in result
    assert result["x_out"].shape == (4, 1, 1)
