from nnodely import Input, Output, Modely
from nnodely.layers.fuzzify import Fuzzify
import numpy as np


def test_fuzzify():
    x = Input("x", dim=1)
    fuzzy_rectangular = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")
    fuzzy_triangular = Fuzzify(centers=[0.0, 0.5, 1.0], function="triangular")
    x_out_rectangular = Output("x_pred_rectangular", fuzzy_rectangular([x.sw(1)]))
    x_out_triangular = Output("x_pred_triangular", fuzzy_triangular([x.sw(1)]))
    model1 = Modely("model1", inputs=[x], outputs=[x_out_rectangular, x_out_triangular])
    model1.build()

    # ------- Model inference -------
    dummy_input_x = np.array([[[-7]], [[0.5]], [[0.8]], [[5.0]]], dtype=np.float32)
    result1 = model1({"x": dummy_input_x})

    assert "x_pred_rectangular" in result1
    assert result1["x_pred_rectangular"].shape == (4, 3, 1)
    assert np.allclose(
        result1["x_pred_rectangular"].cpu().numpy(),
        [
            [[0.0], [0.0], [0.0]],
            [[0.0], [1.0], [0.0]],
            [[0.0], [0.0], [1.0]],
            [[0.0], [0.0], [0.0]],
        ],
    )
    assert "x_pred_triangular" in result1
    assert result1["x_pred_triangular"].shape == (4, 3, 1)
    assert np.allclose(
        result1["x_pred_triangular"].cpu().numpy(),
        [
            [[0.0], [0.0], [0.0]],
            [[0.0], [1.0], [0.0]],
            [[0.0], [0.4], [0.6]],
            [[0.0], [0.0], [0.0]],
        ],
    )


# def test_local_model():
#     # ------- High-level Blocks (Local Models) with Multi-inputs -------
#     x = Input("x", dim=1)
#     k = Input("k", dim=1)
#     g = Constant("gravity", value=[9.81])

#     fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")([k.sw(1)])
#     local_model = LocalModel(
#         input_function=Fir(out_features=1),
#         output_function=ReLU(),
#         name="local_model",
#     )(activation=fuzzy_k)

#     out = Output("out", local_model([x]) + g)
#     model = Modely("model_with_local", inputs=[x, k], outputs=[out])
#     model.build()

#     # ------- Model inference -------
#     batch_size = 3
#     dummy_input_x = np.ones((batch_size, 5, 5), dtype=np.float32) * 1.0
#     dummy_input_y = np.ones((batch_size, 5, 5), dtype=np.float32) * 2.0
#     dummy_input_z = np.ones((batch_size, 5, 5), dtype=np.float32) * 3.0
#     dummy_input_k = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)

#     batch_size = 3
#     dummy_input_x = np.ones((batch_size, 1, 1), dtype=np.float32) * 1.0
#     dummy_input_k = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)

#     result = model({"x": dummy_input_x, "k": dummy_input_k})

#     print("Model - Input shape x:", dummy_input_x.shape)
#     print("Model - Input shape k:", dummy_input_k.shape)
#     print("Model - Output shape:", result["out"].shape)
#     print("Model - Output:", result)


if __name__ == "__main__":
    test_fuzzify()
