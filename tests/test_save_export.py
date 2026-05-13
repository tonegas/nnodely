import os


from nnodely import Input, Output, Fir, Modely


def test_save_model(tmp_path):
    # ------- Model definition and building -------
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    x_stream = x.sw(10)
    y_stream = y.sw(10)

    fir = Fir(out_features=2)
    result_fir = fir([x_stream + y_stream])

    x_out = Output("x_pred", result_fir)
    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    # ------- Model composition -------
    z = Input("z", dim=1)
    z_stream = z.sw(10)
    z_fir = Fir(out_features=1)([model1([z_stream, z_stream])])
    z_out = Output("z_pred", z_fir)
    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    # ------- Model inference -------
    # dummy_input_x = np.ones((4, 10, 1), dtype=np.float32)
    # dummy_input_y = np.ones((4, 10, 1), dtype=np.float32)
    # dummy_input_z = np.ones((4, 10, 1), dtype=np.float32)

    # result1 = model1([dummy_input_x, dummy_input_y])
    # result2 = model2([dummy_input_z])

    # ------- Save/load model inference -------
    model1.save(os.path.join(tmp_path, "model1"))
    model2.save(os.path.join(tmp_path, "model2"))
    loaded_model1 = Modely.load(os.path.join(tmp_path, "model1"))
    loaded_model2 = Modely.load(os.path.join(tmp_path, "model2"))
    loaded_model1.build()
    loaded_model2.build()
    # loaded_result1 = loaded_model1([dummy_input_x, dummy_input_y])
    # loaded_result2 = loaded_model2([dummy_input_z])
    # assert result1["x_pred"] == loaded_result1["x_pred"]
    # assert result2["z_pred"] == loaded_result2["z_pred"]


# def test_export_html(tmp_path):
#     x = Input(name="x", dim=1)
#     y = Input(name="y", dim=1)
#     r1 = x.sw(1) + y.sw(1)
#     out1 = Output("out1", r1)
#     model1 = Modely(name="model1", inputs=[x, y], outputs=[out1])
#     model1.build()

#     z = Input(name="z", dim=1, seq=5)
#     const = Constant("const", value=2.0)
#     r2 = z.sw(1) * const
#     loop_fn = Loop(f=model1, closed_loop={out1: z})
#     out = Output("out", loop_fn([z, r2]))
#     model = Modely(name="model", inputs=[z], outputs=[out])
#     model.build()

#     dummy_input_z = np.ones((4, 5, 1, 1), dtype=np.float32)
#     model_result = model({"z": dummy_input_z})
#     assert "out" in model_result
#     assert model_result["out"].shape == (4, 5, 1, 1)

#     # ------- Model export to HTML -------
#     model.save(filename=os.path.join(tmp_path, "model_k"))
#     new_model = Modely.load(filename=os.path.join(tmp_path, "model_k"))
#     new_model.build()

#     # ------- New model inference -------
#     result2 = new_model({"z": dummy_input_z})
#     assert "out" in result2
#     assert result2["out"].shape == model_result["out"].shape
#     assert np.allclose(result2["out"], model_result["out"], atol=1e-5)
