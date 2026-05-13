import numpy as np

from nnodely import Input, Output, Fir, Modely


def test_model_composition():
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    result_fir = Fir(out_features=2)([x.sw(10) + y.sw(10)])
    x_out = Output("x_pred", result_fir)

    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    z = Input("z", dim=1)
    z_fir = Fir(out_features=1)([model1([z.sw(10), z.sw(10)])])
    z_out = Output("z_pred", z_fir)

    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    dummy_z = np.ones((3, 10, 1), dtype=np.float32)
    result = model2([dummy_z])

    assert "z_pred" in result
    assert result["z_pred"].shape == (3, 1, 1)
