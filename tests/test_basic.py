import numpy as np

from nnodely.core.modely import Modely
from nnodely.core.stream import Stream
from nnodely.layers.input import Input
from nnodely.layers.output import Output
from nnodely.layers.constant import Constant

def dummy_data(batch_size=1, time_steps=1, feature_dim=1):
    return np.ones((batch_size, time_steps, feature_dim), dtype=np.float32)

def test_modely():
    x = Input(name="x", dim=(1,))
    out = Output("out", x.sw(1) + x.sw(1))

    model = Modely("Model", outputs=out)
    model.build()

    ret = model({"x": dummy_data()})
    assert np.allclose(ret['out'].numpy(), dummy_data() * 2)

def test_basic_operations():
    x = Input(name="x", dim=(1,))
    y = Input(name="y", dim=(1,))
    x_1 = x.sw(1)
    y_1 = y.sw(1)

    add_out = Output("add_out", x_1 + y_1)
    sub_out = Output("sub_out", x_1 - y_1)
    mul_out = Output("mul_out", x_1 * y_1)
    div_out = Output("div_out", x_1 / y_1)

    model = Modely("Model", outputs=[add_out, sub_out, mul_out, div_out])
    model.build()

    res = model({"x": dummy_data(batch_size=4), "y": dummy_data(batch_size=4)})
    assert np.allclose(res['add_out'].numpy(), dummy_data(batch_size=4) * 2)
    assert np.allclose(res['sub_out'].numpy(), dummy_data(batch_size=4) * 0)
    assert np.allclose(res['mul_out'].numpy(), dummy_data(batch_size=4))
    assert np.allclose(res['div_out'].numpy(), dummy_data(batch_size=4))

def test_basic_operations_with_constants():
    x = Input(name="x", dim=(1,))
    two = Constant(name="two", value=2)
    x_1 = x.sw(1)

    add_out = Output("add_out", x_1 + two)
    sub_out = Output("sub_out", x_1 - 3)
    mul_out = Output("mul_out", x_1 * 3)
    div_out = Output("div_out", x_1 / 4)

    model = Modely("Model", outputs=[add_out, sub_out, mul_out, div_out])
    model.build()

    res = model({"x": dummy_data(batch_size=4)})
    assert len(Stream._literal_constant_cache) == 2 # Only two constants (3 and 4) should be cached
    assert np.allclose(res['add_out'].numpy(), dummy_data(batch_size=4) + 2)
    assert np.allclose(res['sub_out'].numpy(), dummy_data(batch_size=4) - 3)
    assert np.allclose(res['mul_out'].numpy(), dummy_data(batch_size=4) * 3)
    assert np.allclose(res['div_out'].numpy(), dummy_data(batch_size=4) / 4)

# def test_model_composition():
#     x = Input(name="x", dim=1)
#     y = Input(name="y", dim=1)

#     # Model 1
#     add_out = x.sw(1) + y.sw(1)
#     out = Output("out", add_out)
#     model1 = Modely("Model", outputs=out)
#     model1.build()

#     # Model 2 that uses Model 1
#     z = Input(name="z", dim=1)
#     z_1 = z.sw(1)
#     out1 = model1({"x": z_1, "y": z_1})+5
#     print("out1:", out1)
#     out2 = Output("out2", out1)
#     model2 = Modely("Model2", outputs=out2)
#     model2.build()

#     res = model2({"z": dummy_data(batch_size=4)})
#     assert np.allclose(res['out2'].numpy(), dummy_data(batch_size=4) + dummy_data(batch_size=4) + 5)

#     # ------- Save/load model inference -------
#     model2.save("model2")
#     loaded_model = Modely.load("model2")
#     res_loaded = loaded_model({"z": dummy_data(batch_size=4)})
#     assert np.allclose(res_loaded['out2'].numpy(), dummy_data(batch_size=4) + dummy_data(batch_size=4) + 5)