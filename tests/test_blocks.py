from nnodely import Input, Output, Constant, Modely
from nnodely.layers.localmodel import LocalModel
from nnodely.layers.fuzzify import Fuzzify
from nnodely.layers.fir import Fir
from nnodely.layers.activations import ReLU


def test_local_model():
    pass
    # ------- High-level Blocks (Local Models) with Multi-inputs -------
    # x = Input("x", dim=1)
    # k = Input("k", dim=1)
    # g = Constant("gravity", value=[9.81])

    # fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")([k.sw(1)])
    # local_model = LocalModel(
    #     input_function=Fir(out_features=1),
    #     output_function=ReLU(),
    #     name="local_model",
    # )(activation=fuzzy_k)

    # out = Output("out", local_model([x]) + g)
    # model = Modely("model_with_local", inputs=[x, k], outputs=[out])
    # model.build()

    # ------- Model inference -------
    # batch_size = 3
    # dummy_input_x = np.ones((batch_size, 5, 5), dtype=np.float32) * 1.0
    # dummy_input_y = np.ones((batch_size, 5, 5), dtype=np.float32) * 2.0
    # dummy_input_z = np.ones((batch_size, 5, 5), dtype=np.float32) * 3.0
    # dummy_input_k = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)

    # batch_size = 3
    # dummy_input_x = np.ones((batch_size, 1, 1), dtype=np.float32) * 1.0
    # dummy_input_k = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)

    # result = model({"x": dummy_input_x,"k": dummy_input_k})

    # print("Model - Input shape x:", dummy_input_x.shape)
    # print("Model - Input shape k:", dummy_input_k.shape)
    # print("Model - Output shape:", result["out"].shape)
    # print("Model - Output:", result)
