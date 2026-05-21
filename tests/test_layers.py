from nnodely import Input, Output, Modely
from nnodely.layers.trigonometric import Sin, Cos, Tan, Asin, Acos, Atan
from nnodely.layers.activations import (
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    GELU,
    PReLU,
    Softmax,
    Swish,
    Softplus,
)
from nnodely.layers.concatenate import Concatenate, TimeConcatenate
from nnodely.layers.parameter import Parameter
from nnodely.layers.constant import Constant
from nnodely.layers.fir import Fir
from nnodely.layers.fuzzify import Fuzzify
import numpy as np

def test_trigonometric():
    # ------- Test trigonometric layers -------
    x = Input("x", dim=1)

    sin_out = Output("out_sin", Sin()([x.sw(1)]))
    cos_out = Output("out_cos", Cos()([x.sw(3)]))
    tan_out = Output("out_tan", Tan()([x.sw(1)]))
    asin_out = Output("out_asin", Asin()([x.sw(4)]))
    acos_out = Output("out_acos", Acos()([x.sw(1)]))
    atan_out = Output("out_atan", Atan()([x.sw(2)]))
    model = Modely(
        "trig_model",
        inputs=[x],
        outputs=[sin_out, cos_out, tan_out, asin_out, acos_out, atan_out],
    )
    model.build()

    # ------- Model inference -------
    batch_size = 1
    max_time = 4
    dummy_input_x = np.random.rand(batch_size, 1, max_time) * 2 - 1  # Random values in range [-1, 1]
    result = model({"x": dummy_input_x})
    assert "out_sin" in result
    assert result["out_sin"].shape == (batch_size, 1, 1)
    assert np.allclose(result["out_sin"].detach().cpu().numpy(), np.sin(dummy_input_x[:, :, -1:]))
    assert "out_cos" in result
    assert result["out_cos"].shape == (batch_size, 1, 3)
    assert np.allclose(result["out_cos"].detach().cpu().numpy(), np.cos(dummy_input_x[:, :, -3:]))
    assert "out_tan" in result
    assert result["out_tan"].shape == (batch_size, 1, 1)
    assert np.allclose(result["out_tan"].detach().cpu().numpy(), np.tan(dummy_input_x[:, :, -1:]))
    assert "out_asin" in result
    assert result["out_asin"].shape == (batch_size, 1, 4)
    assert np.allclose(result["out_asin"].detach().cpu().numpy(), np.arcsin(dummy_input_x[:, :, -4:]))
    assert "out_acos" in result
    assert result["out_acos"].shape == (batch_size, 1, 1)
    assert np.allclose(result["out_acos"].detach().cpu().numpy(), np.arccos(dummy_input_x[:, :, -1:]))
    assert "out_atan" in result
    assert result["out_atan"].shape == (batch_size, 1, 2)
    assert np.allclose(result["out_atan"].detach().cpu().numpy(), np.arctan(dummy_input_x[:, :, -2:]))

def test_activations():
    # ------- Test activation layers -------
    x = Input("x", dim=1)
    relu_out = Output("out_relu", ReLU(max_value=1.0)([x.sw(1)]))
    elu_out = Output("out_elu", ELU(alpha=1.0)([x.sw(2)]))
    leaky_relu_out = Output("out_leaky_relu", LeakyReLU(negative_slope=0.3)([x.sw(1)]))
    prelu_out = Output("out_prelu", PReLU()([x.sw(1)]))
    sigmoid_out = Output("out_sigmoid", Sigmoid()([x.sw(5)]))
    tanh_out = Output("out_tanh", Tanh()([x.sw(1)]))
    softmax_out = Output("out_softmax", Softmax(axis=-1)([x.sw(3)]))
    swish_out = Output("out_swish", Swish()([x.sw(1)]))
    gelu_out = Output("out_gelu", GELU()([x.sw(1)]))
    softplus_out = Output("out_softplus", Softplus()([x.sw(1)]))
    model = Modely(
        "activation_model",
        inputs=[x],
        outputs=[
            relu_out,
            elu_out,
            leaky_relu_out,
            prelu_out,
            sigmoid_out,
            tanh_out,
            softmax_out,
            swish_out,
            gelu_out,
            softplus_out,
        ],
    )
    model.build()


def test_layers():
    x = Input("x", dim=1)
    param = Parameter("param1", dim=1)
    const = Constant("const1", value=[1.0])

    add = param + const
    mul = param * const
    sub = param - const
    div = param / const

    Fir_out = Fir(out_features=1)([x.sw(2)])
    out_add = Output("out_add", add)
    out_mul = Output("out_mul", mul)
    out_sub = Output("out_sub", sub)
    out_div = Output("out_div", div)
    out_fir = Output("out_fir", Fir_out)
    out_concat = Output(
        "out_concat", TimeConcatenate(name="time_concat")([x.sw(2), Fir_out])
    )
    param2 = Parameter("param2", dim=(3, 2))
    param3 = Parameter("param3", dim=(1, 2))
    param4 = Parameter("param4", dim=(1, 2))
    out_concat2 = Output(
        "out_concat2", Concatenate(name="concat", axis=0)([param2, param3])
    )
    out_concat3 = Output(
        "out_concat3", Concatenate(name="concat2", axis=1)([param3, param4])
    )
    out_concat4 = Output(
        "out_concat4", Concatenate(name="concat3", axis=0)([param2, param3, param4])
    )
    model = Modely(
        "model",
        inputs=[x],
        outputs=[
            out_add,
            out_mul,
            out_sub,
            out_div,
            out_fir,
            out_concat,
            out_concat2,
            out_concat3,
            out_concat4,
        ],
    )
    model.build()


def test_fuzzify():
    x = Input("x", dim=1)
    x_stream = x.sw(1)
    fuzzy = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")
    x_out = Output("x_pred", fuzzy([x_stream]))
    model1 = Modely("model1", inputs=[x], outputs=[x_out])
    model1.build()

    # ------- Model inference -------
    dummy_input_x = np.array([[[-7]], [[0.5]], [[0.8]]], dtype=np.float32)
    result1 = model1({"x": dummy_input_x})
    assert "x_pred" in result1
    assert result1["x_pred"].shape == (3, 1, 3)
