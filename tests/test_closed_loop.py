# import numpy as np

# from nnodely import Input, Output, Modely, Loop, Constant


# def test_closed_loop(batch_size):
#     # ------- Model with closed loop connections -------

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

#     # ------- Model inference -------
#     dummy_input_x = np.ones((batch_size, 1, 1), dtype=np.float32)
#     dummy_input_y = np.ones((batch_size, 1, 1), dtype=np.float32)
#     dummy_input_z = np.ones((batch_size, 5, 1, 1), dtype=np.float32)

#     result1 = model1({"x": dummy_input_x, "y": dummy_input_y})
#     assert "out1" in result1
#     assert result1["out1"].shape == (batch_size, 1, 1)

#     result2 = model({"z": dummy_input_z})
#     assert "out" in result2
#     assert result2["out"].shape == (batch_size, 5, 1, 1)
