from nnodely import Modely, Input, Output, Fir
import numpy as np
import pytest

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def window_size():
    return 5

def test_basic_model_build_and_inference(batch_size, window_size):
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    x_stream = x.sw(window_size)
    y_stream = y.sw(window_size)

    result_fir = Fir(out_features=2)([x_stream + y_stream])
    x_out = Output("x_pred", result_fir)

    model = Modely("model1", inputs=[x, y], outputs=[x_out])
    model.build()

    dummy_x = np.ones((batch_size, 1, window_size), dtype=np.float32)
    dummy_y = np.ones((batch_size, 1, window_size), dtype=np.float32)

    result = model([dummy_x, dummy_y])

    assert "x_pred" in result
    assert result["x_pred"].shape == (batch_size, 2, 1)


def test_model_inference_with_composed_model(batch_size, window_size):
    # ------- Model definition and building -------
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    x_stream = x.sw(window_size)
    y_stream = y.sw(window_size)

    fir = Fir(out_features=2)
    result_fir = fir([x_stream + y_stream])

    x_out = Output("x_pred", result_fir)
    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    # ------- Model composition -------
    z = Input("z", dim=1)
    z_stream = z.sw(window_size)
    z_fir = Fir(out_features=1)(model1([z_stream, z_stream]))
    z_out = Output("z_pred", z_fir)
    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    # ------- Model inference -------
    dummy_input_x = np.ones((batch_size, 1, window_size), dtype=np.float32)
    dummy_input_y = np.ones((batch_size, 1, window_size), dtype=np.float32)
    dummy_input_z = np.ones((batch_size, 1, window_size), dtype=np.float32)

    result1 = model1([dummy_input_x, dummy_input_y])
    assert "x_pred" in result1
    assert result1["x_pred"].shape == (batch_size, 2, 1)

    result2 = model2([dummy_input_z])
    assert "z_pred" in result2
    assert result2["z_pred"].shape == (batch_size, 1, 1)
