import os
os.environ.setdefault("KERAS_BACKEND", "jax")

from nnodely import Input, Output, Modely, Loop, Parameter
import numpy as np


from nnodely.layers.fir import Fir


# def dummy_input(shape, method="random"):
#     if method == "random":
#         return np.random.rand(*shape).astype(np.float32)
#     elif method == "zeros":
#         return np.zeros(shape, dtype=np.float32)
#     elif method == "ones":
#         return np.ones(shape, dtype=np.float32)
#     elif method == "sequential":
#         return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1


def test_closed_loop(tmp_path):
    # ------- Model with closed loop connections -------
    batch_size = 10
    # Define a simple model to be used in the loop
    _x = Input(name="_x", dim=1)
    _y = Input(name="_y", dim=1)
    c = Parameter("c", dim=1, value=1.0)
    d = Parameter("d", dim=1, value=1.0)
    r1 = _x * c + _y * d
    out1 = Output("out1", r1)
    model_add = Modely(name="model1", inputs=[_x, _y], outputs=[out1])
    model_add.build()

#     # Option 1: use Loop layer directly in the output definition

#     x = Input(name="x", dim=1)
#     y = Input(name="y", dim=1, seq=4)
#     loop_fn = Loop(
#         f=model_add,
#         closed_loop={"_x": "out1"},
#         initial_values={"_x": x},
#         name="loop_model_add",
#     )  # add initial condition
#     out = Output("out", loop_fn([x, y]))
#     model_in = Modely(name="model", inputs=[x, y], outputs=[out])
#     model_in.build()

#     # Create a nested loop model
#     w = Input(name="w", dim=1)
#     z = Input(name="z", dim=1, seq=(4, 2))
#     loop_fn2 = Loop(
#         f=model_in,
#         closed_loop={"y": "out"},
#         initial_values={"y": z},
#         name="loop_model_in",
#     )
#     out_w = Output("out_w", loop_fn2([w, z]))
#     model_out = Modely(name="model_with_loop_w", inputs=[w, z], outputs=[out_w])

#     model_out.minimize(
#         "error",
#         source=out_w,
#         target=Input("w_target", dim=1, seq=(4, 2)).sw(1),
#         loss="mse",
#     )

#     model_out.build()
#     print(model_out)
#     model_in.plot(to_file=os.path.join(tmp_path, "model_with_loop.png"))
#     model_out.plot(to_file=os.path.join(tmp_path, "model_with_loop_w.png"))
#     model_out.export_html(os.path.join("html", "model_with_loop_w.html"))

#     # ------- Model inference -------
#     batch_size = 1
#     dummy_input_x = dummy_input((batch_size, 1), method="ones")
#     dummy_input_y = dummy_input((batch_size, 1, 4), method="sequential")

    result_in = model_in({"x": dummy_input_x, "y": dummy_input_y})
    assert "out" in result_in
    assert result_in["out"].shape == (batch_size, 1, 4)
    #assert result_in["out"] == dummy_input_x * 1.0 + dummy_input_y * 1.0
    dummy_input_w = dummy_input((batch_size, 1), method="ones")
    dummy_input_z = dummy_input((batch_size, 1, 4, 2), method="sequential")
    result_out = model_out({"z": dummy_input_z, "w": dummy_input_w})
    assert "out_w" in result_out
    assert result_out["out_w"].shape == (batch_size, 1, 4, 2)


# def test_sw_closed_loop():
#     # Define a simple model to be used in the loop
#     _x = Input(name="_x", dim=1)
#     _y = Input(name="_y", dim=1)
#     d = Parameter("d", dim=1)
#     fir = Fir(out_features=1)([_x.sw(2)])
#     r1 = fir + _y.sw(1) * d
#     out1 = Output("out1", r1)
#     model_add = Modely(name="model1", inputs=[_x, _y], outputs=[out1])
#     model_add.build()

#     # Option 1: use Loop layer directly in the output definition
#     x = Input(name="x", dim=1)
#     y = Input(name="y", dim=1, seq=4)
#     loop_fn = Loop(
#         f=model_add,
#         closed_loop={"_x": "out1"},
#         initial_values={"_x": x},
#         name="loop_model_add",
#     )
#     out = Output("out", loop_fn([x.sw(2), y.sw(1)]))
#     model_in = Modely(name="model", inputs=[x, y], outputs=[out])
#     model_in.build()

#     # Create a nested loop model
#     w = Input(name="w", dim=1)
#     z = Input(name="z", dim=1, seq=(4, 2))
#     loop_fn2 = Loop(
#         f=model_in,
#         closed_loop={"y": "out"},
#         initial_values={"y": z},
#         name="loop_model_in",
#     )
#     out_w = Output("out_w", loop_fn2([w.sw(2), z.sw(1)]))
#     model_out = Modely(name="model_with_loop_w", inputs=[w, z], outputs=[out_w])
#     model_out.build()

#     model_in.plot(to_file="html/model_with_loop.png")
#     model_out.plot(to_file="html/model_with_loop_w.png")

#     # model_out.train(train_data=data_train, epochs=100, batch_size=64, lr=1e-3)
#     # ------- Model inference -------
#     batch_size = 1
#     dummy_input_x = dummy_input((batch_size, 1, 2), method="ones")
#     dummy_input_y = dummy_input((batch_size, 1, 1, 4), method="sequential")

#     result_in = model_in({"x": dummy_input_x, "y": dummy_input_y})
#     print("Result of model_in shape:", result_in["out"].shape)
#     dummy_input_w = dummy_input((batch_size, 1, 2), method="ones")
#     dummy_input_z = dummy_input((batch_size, 1, 1, 4, 2), method="sequential")
#     result_out = model_out({"z": dummy_input_z, "w": dummy_input_w})

#     assert "out" in result_in
#     assert result_in["out"].shape == (batch_size, 1, 1, 4)

#     assert "out_w" in result_out
#     assert result_out["out_w"].shape == (batch_size, 1, 1, 4, 2)


# def test_sw_2_closed_loop():
#     # Define a simple model to be used in the loop
#     _x = Input(name="_x", dim=1)
#     _y = Input(name="_y", dim=1)
#     d = Parameter("d", dim=1)
#     fir = Fir(out_features=1)([_x.sw(2)])
#     r1 = fir + _y.sw(2) * d
#     out1 = Output("out1", r1)
#     model_add = Modely(name="model1", inputs=[_x, _y], outputs=[out1])
#     model_add.build()

#     # Option 1: use Loop layer directly in the output definition
#     x = Input(name="x", dim=1)
#     y = Input(name="y", dim=1, seq=4)
#     loop_fn = Loop(
#         f=model_add,
#         closed_loop={"_x": "out1"},
#         initial_values={"_x": x},
#         name="loop_model_add",
#     )
#     out = Output("out", loop_fn([x.sw(2), y.sw(2)]))
#     model_in = Modely(name="model_loop_add", inputs=[x, y], outputs=[out])
#     model_in.build()

#     # Create a nested loop model
#     w = Input(name="w", dim=1)
#     z = Input(name="z", dim=1, seq=(4, 2))
#     loop_fn2 = Loop(
#         f=model_in,
#         closed_loop={"y": "out"},
#         initial_values={"y": z},
#         name="loop_model_in",
#     )
#     out_w = Output("out_w", loop_fn2([w.sw(2), z.sw(2)]))
#     model_out = Modely(name="model_with_loop_w", inputs=[w, z], outputs=[out_w])

#     print(model_out)
#     model_out.build()

#     model_in.plot(to_file="html/model_with_loop.png")
#     model_out.plot(to_file="html/model_with_loop_w.png")
#     model_out.export_html("html/model_with_loop_w.html")

#     # ------- Model inference -------
#     batch_size = 1
#     dummy_input_x = dummy_input((batch_size, 1, 2), method="ones")
#     dummy_input_y = dummy_input((batch_size, 1, 2, 4), method="sequential")

#     result_in = model_in({"x": dummy_input_x, "y": dummy_input_y})

#     dummy_input_w = dummy_input((batch_size, 1, 2), method="ones")
#     dummy_input_z = dummy_input((batch_size, 1, 2, 4, 2), method="sequential")
#     result_out = model_out({"z": dummy_input_z, "w": dummy_input_w})

#     assert "out" in result_in
#     assert result_in["out"].shape == (batch_size, 1, 2, 4)

#     assert "out_w" in result_out
#     assert result_out["out_w"].shape == (batch_size, 1, 2, 4, 2)

if __name__ == "__main__":
    test_closed_loop(tmp_path="html")