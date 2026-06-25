import os

import pytest
from torch.nn import Linear
os.environ.setdefault("KERAS_BACKEND", "jax")

from nnodely import Input, Output, Modely, Loop, Parameter, DataLoader
import numpy as np


from nnodely.layers.fir import Fir


def dummy_input(shape, method="random"):
    if method == "random":
        return np.random.rand(*shape).astype(np.float32)
    elif method == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif method == "ones":
        return np.ones(shape, dtype=np.float32)
    elif method == "sequential":
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1


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

    # Option 1: use Loop layer directly in the output definition

    x = Input(name="x", dim=1)
    y = Input(name="y", dim=1, seq=4)
    loop_fn = Loop(
        f=model_add,
        closed_loop={"_x": "out1"},
        initial_values={"_x": x},
        name="loop_model_add",
    )  # add initial condition
    out = Output("out", loop_fn([x, y]))
    model_in = Modely(name="model", inputs=[x, y], outputs=[out])
    model_in.build()

    # Create a nested loop model
    w = Input(name="w", dim=1)
    z = Input(name="z", dim=1, seq=(4, 2))
    loop_fn2 = Loop(
        f=model_in,
        closed_loop={"y": "out"},
        initial_values={"y": z},
        name="loop_model_in",
    )
    out_w = Output("out_w", loop_fn2([w, z]))
    model_out = Modely(name="model_with_loop_w", inputs=[w, z], outputs=[out_w])

    model_out.minimize(
        "error",
        source=out_w,
        target=Input("w_target", dim=1, seq=(4, 2)).sw(1),
        loss="mse",
    )

    model_out.build()
    model_in.plot(to_file=os.path.join(tmp_path, "model_with_loop.png"))
    model_out.plot(to_file=os.path.join(tmp_path, "model_with_loop_w.png"))
    model_out.export_html(os.path.join("html", "model_with_loop_w.html"))

    # ------- Model inference -------
    batch_size = 1
    dummy_input_x = dummy_input((batch_size, 1), method="ones")
    dummy_input_y = dummy_input((batch_size, 1, 4), method="sequential")

    result_in = model_in({"x": dummy_input_x, "y": dummy_input_y})
    assert "out" in result_in
    assert result_in["out"].shape == (batch_size, 1, 4)
    #assert result_in["out"] == dummy_input_x * 1.0 + dummy_input_y * 1.0
    dummy_input_w = dummy_input((batch_size, 1), method="ones")
    dummy_input_z = dummy_input((batch_size, 1, 4, 2), method="sequential")
    result_out = model_out({"z": dummy_input_z, "w": dummy_input_w, "w_target": dummy_input_z})
    assert "out_w" in result_out
    assert result_out["out_w"].shape == (batch_size, 1, 4, 2)


def test_sw_closed_loop():
    # Define a simple model to be used in the loop
    _x = Input(name="_x", dim=1)
    _y = Input(name="_y", dim=1)
    d = Parameter("d", dim=1)
    fir = Fir(out_features=1)([_x.sw(2)])
    r1 = fir + _y.sw(1) * d
    out1 = Output("out1", r1)
    model_add = Modely(name="model1", inputs=[_x, _y], outputs=[out1])
    model_add.build()

    # Option 1: use Loop layer directly in the output definition
    x = Input(name="x", dim=1)
    y = Input(name="y", dim=1, seq=4)
    loop_fn = Loop(
        f=model_add,
        closed_loop={"_x": "out1"},
        initial_values={"_x": x},
        name="loop_model_add",
    )
    out = Output("out", loop_fn([x.sw(2), y.sw(1)]))
    model_in = Modely(name="model", inputs=[x, y], outputs=[out])
    model_in.build()

    # Create a nested loop model
    w = Input(name="w", dim=1)
    z = Input(name="z", dim=1, seq=(4, 2))
    loop_fn2 = Loop(
        f=model_in,
        closed_loop={"y": "out"},
        initial_values={"y": z},
        name="loop_model_in",
    )
    out_w = Output("out_w", loop_fn2([w.sw(2), z.sw(1)]))
    model_out = Modely(name="model_with_loop_w", inputs=[w, z], outputs=[out_w])
    model_out.build()

    model_in.plot(to_file="html/model_with_loop.png")
    model_out.plot(to_file="html/model_with_loop_w.png")

    # model_out.train(train_data=data_train, epochs=100, batch_size=64, lr=1e-3)
    # ------- Model inference -------
    batch_size = 1
    dummy_input_x = dummy_input((batch_size, 1, 2), method="ones")
    dummy_input_y = dummy_input((batch_size, 1, 1, 4), method="sequential")

    result_in = model_in({"x": dummy_input_x, "y": dummy_input_y})
    print("Result of model_in shape:", result_in["out"].shape)
    dummy_input_w = dummy_input((batch_size, 1, 2), method="ones")
    dummy_input_z = dummy_input((batch_size, 1, 1, 4, 2), method="sequential")
    result_out = model_out({"z": dummy_input_z, "w": dummy_input_w})

    assert "out" in result_in
    assert result_in["out"].shape == (batch_size, 1, 1, 4)

    assert "out_w" in result_out
    assert result_out["out_w"].shape == (batch_size, 1, 1, 4, 2)


def test_sw_2_closed_loop():
    # Define a simple model to be used in the loop
    _x = Input(name="_x", dim=1)
    _y = Input(name="_y", dim=1)
    d = Parameter("d", dim=1)
    fir = Fir(out_features=1)([_x.sw(2)])
    r1 = fir + _y.sw(2) * d
    out1 = Output("out1", r1)
    model_add = Modely(name="model1", inputs=[_x, _y], outputs=[out1])
    model_add.build()

    # Option 1: use Loop layer directly in the output definition
    x = Input(name="x", dim=1)
    y = Input(name="y", dim=1, seq=4)
    loop_fn = Loop(
        f=model_add,
        closed_loop={"_x": "out1"},
        initial_values={"_x": x},
        name="loop_model_add",
    )
    out = Output("out", loop_fn([x.sw(2), y.sw(2)]))
    model_in = Modely(name="model_loop_add", inputs=[x, y], outputs=[out])
    model_in.build()

    # Create a nested loop model
    w = Input(name="w", dim=1)
    z = Input(name="z", dim=1, seq=(4, 2))
    loop_fn2 = Loop(
        f=model_in,
        closed_loop={"y": "out"},
        initial_values={"y": z},
        name="loop_model_in",
    )
    out_w = Output("out_w", loop_fn2([w.sw(2), z.sw(2)]))
    model_out = Modely(name="model_with_loop_w", inputs=[w, z], outputs=[out_w])

    print(model_out)
    model_out.build()

    model_in.plot(to_file="html/model_with_loop.png")
    model_out.plot(to_file="html/model_with_loop_w.png")
    model_out.export_html("html/model_with_loop_w.html")

    # ------- Model inference -------
    batch_size = 1
    dummy_input_x = dummy_input((batch_size, 1, 2), method="ones")
    dummy_input_y = dummy_input((batch_size, 1, 2, 4), method="sequential")

    result_in = model_in({"x": dummy_input_x, "y": dummy_input_y})

    dummy_input_w = dummy_input((batch_size, 1, 2), method="ones")
    dummy_input_z = dummy_input((batch_size, 1, 2, 4, 2), method="sequential")
    dummy_input_w_target = dummy_input((batch_size, 1, 2, 4, 2), method="sequential")
    result_out = model_out({"z": dummy_input_z, "w": dummy_input_w, "w_target": dummy_input_w_target})

    assert "out" in result_in
    assert result_in["out"].shape == (batch_size, 1, 2, 4)

    assert "out_w" in result_out
    assert result_out["out_w"].shape == (batch_size, 1, 2, 4, 2)

@pytest.mark.skipif(os.environ.get("KERAS_BACKEND") == "torch", reason="torch is too slow")
def test_simple_model(tmp_path):
    # Define a simple model to be used in the loop
    _x = Input(name="_x", dim=1)
    _z = Input(name="_z", dim=1)
    w = Parameter("w", dim=1, value=3.5)
    out1 = Output("y", _x + w + _z)
    model_add = Modely(name="model1", inputs=[_x, _z], outputs=[out1])

    # Option 1: use Loop layer directly in the output definition
    x = Input(name="x", dim=1, seq=4)
    z = Input(name="z", dim=1)
    loop_fn = Loop(
        f=model_add,
        closed_loop={"_x": "y"},
        initial_values={"_x": x},
        name="loop_model_add",
    )
    out = Output("out", loop_fn([x, z]))
    model_in = Modely(name="simple_loop_model", inputs=[x, z], outputs=[out])

    model_in.minimize(
        "error",
        source=out,
        target=Input("x_target", dim=1, seq=4),
        loss="mse",
    )
    model_in.build()
    model_in.export_html(os.path.join(tmp_path, "simple_loop_model.html"))

    def compute_targets(x):
        # Compute the target values for the closed loop model
        w_value = 4
        y = np.zeros_like(x)
        for t in range(x.shape[-1]):
            if t == 0:
                y[:,t] = x[:,t] + w_value + 1
            else:
                y[:,t] = y[:,t - 1] + w_value + 1
        return y

    xs = dummy_input((3000, 1, 4), method="sequential")
    zs = dummy_input((3000, 1), method="ones")
    dataset = {'x': xs,
               'z': zs,
               'x_target': [compute_targets(v) for v in xs]}
    
    res = model_in({"x": dummy_input((1, 1, 4), method="ones")+7, "z": dummy_input((1, 1), method="ones"), "x_target": dummy_input((1, 1, 4), method="sequential")})
    print("Result of simple model:", res["out"])
    data_train = DataLoader(model_in, source=dataset)
    model_in.train(train_data=data_train, epochs=100, batch_size=64, lr=5e-3)
    print("Weights of simple model:", model_in.model.get_weights())

    res = model_in({"x": dummy_input((1, 1, 4), method="ones")+7, "z": dummy_input((1, 1), method="ones"), "x_target": dummy_input((1, 1, 4), method="sequential")})
    print("Result of simple model:", res["out"])

    assert model_in.model.get_weights()[0] == pytest.approx(4.0, rel=1e-2)


@pytest.mark.skipif(os.environ.get("KERAS_BACKEND") == "torch", reason="torch is too slow")
def test_simple_model2(tmp_path):
    # Define a simple model to be used in the loop
    _x = Input(name="_x", dim=1)
    _z = Input(name="_z", dim=1)
    w = Parameter("w", dim=1)
    k = Parameter("k", dim=1)
    out1 = Output("y", _x + w)
    out2 = Output("y2", _z - k)
    model_add = Modely(name="model1", inputs=[_x, _z], outputs=[out1, out2])

    # Option 1: use Loop layer directly in the output definition
    x = Input(name="x", dim=1, seq=4)
    z = Input(name="z", dim=1, seq=4)
    loop_fn = Loop(
        f=model_add,
        closed_loop={"_x": "y", "_z": "y"},
        initial_values={"_x": x, "_z": z},
        name="loop_model_add",
    )
    loop_out1, loop_out2 = loop_fn([x, z])
    out1 = Output("out1", loop_out1)
    out2 = Output("out2", loop_out2)
    model_in = Modely(name="simple_loop_model", inputs=[x, z], outputs=[out1, out2])

    model_in.minimize(
        "error_x",
        source=out1,
        target=Input("x_target", dim=1, seq=4),
        loss="mse",
    )
    model_in.minimize(
        "error_z",
        source=out2,
        target=Input("z_target", dim=1, seq=4),
        loss="mse",
    )
    model_in.build()
    model_in.export_html(os.path.join(tmp_path, "simple_loop_model.html"))

    def compute_targets(x, z):
        # Compute the target values for the closed loop model
        w_value = 4
        k_value = 2
        y = np.zeros_like(x)
        y1 = np.zeros_like(x)
        for t in range(x.shape[-1]):
            if t == 0:
                y[:,t] = x[:,t] + w_value
                y1[:,t] = z[:,t] - k_value
            else:
                y[:,t] = y[:,t - 1] + w_value
                y1[:,t] = y[:,t - 1] - k_value
        return y, y1

    xs = dummy_input((3000, 1, 4), method="sequential")
    zs = dummy_input((3000, 1, 4), method="ones")
    y = [compute_targets(v, z) for v, z in zip(xs, zs)]
    ys, ys1 = zip(*y)
    dataset = {'x': xs,
               'z': zs,
               'x_target': ys,
               'z_target': ys1}
    data_train = DataLoader(model_in, source=dataset)

    print("Weights of simple model:", model_in.model.get_weights())
    model_in.train(train_data=data_train, epochs=100, batch_size=64, lr=5e-3)

    res = model_in({"x": dummy_input((1, 1, 4), method="ones")+7, "z": dummy_input((1, 1, 4), method="ones"),
                    "x_target": dummy_input((1, 1, 4), method="sequential"),
                    "z_target": dummy_input((1, 1, 4), method="sequential")})
    print("Result of simple model:", res["out1"], res["out2"])

    print("Weights of simple model:", model_in.model.get_weights())
    assert model_in.model.get_weights()[0] == pytest.approx(4.0, rel=1e-2)
    assert model_in.model.get_weights()[1] == pytest.approx(2.0, rel=1e-2)

if __name__ == "__main__":
    test_simple_model(tmp_path="html")
    test_simple_model2(tmp_path="html")