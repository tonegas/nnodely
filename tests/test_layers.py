from nnodely import (
    Constant,
    Concatenate,
    Input,
    Interpolation,
    Modely,
    Output,
    Parameter,
    TimeConcatenate,
    TimeSelect,
)
from nnodely.layers.trigonometric import Sin, Cos, Tan, Asin, Acos, Atan
from nnodely.layers.fir import Fir
from nnodely.core.layer import Layer
import keras
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
import numpy as np
import pytest
from conftest import to_numpy


class DoubleFeaturesImpl(keras.layers.Layer):
    def call(self, x):
        return keras.ops.concatenate([x, x], axis=1)


class DoubleFeatures(Layer):
    """Test layer relying entirely on Layer.output_shape()."""

    def build_layer(self):
        return DoubleFeaturesImpl(name=self.name)


def test_automatic_layer_output_shape():
    x = Input("automatic_shape_input", dim=2)
    doubled = DoubleFeatures(name="double_features")(x.sw(3))

    assert doubled.dim == (4,)
    assert doubled.time == 3
    assert doubled.seq == ()

    model = Modely(
        "automatic_shape_model",
        inputs=[x],
        outputs=[Output("automatic_shape_output", doubled)],
    ).build()
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    result = to_numpy(
        model({"automatic_shape_input": values})["automatic_shape_output"]
    )

    assert result.shape == (1, 4, 3)
    np.testing.assert_allclose(result[:, :2], values)
    np.testing.assert_allclose(result[:, 2:], values)


def test_time_select():
    x = Input("time_select_input")
    selected = TimeSelect(idx=3, name="time_select")(x.sw(5))

    assert selected.dim == (1,)
    assert selected.time == 1
    assert selected.seq == ()

    model = Modely(
        "time_select_model",
        inputs=[x],
        outputs=[Output("time_select_output", selected)],
    ).build()
    values = np.arange(5, dtype=np.float32).reshape((1, 1, 5))
    result = model({"time_select_input": values})

    assert result["time_select_output"].shape == (1, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["time_select_output"]),
        np.array([[[3.0]]], dtype=np.float32),
    )


def test_linear_interpolation():
    x = Input("interpolation_input")
    interpolated = Interpolation(
        x_points=[2.0, 0.0, 1.0],
        y_points=[4.0, 0.0, 1.0],
        mode="linear",
        name="linear_interpolation",
    )(x.sw(5))

    assert interpolated.dimensions == ((1,), 5, ())
    model = Modely(
        "linear_interpolation_model",
        inputs=[x],
        outputs=[Output("interpolated", interpolated)],
    ).build()

    values = np.array([[[-1.0, 0.5, 1.0, 1.5, 3.0]]], dtype=np.float32)
    result = to_numpy(model({"interpolation_input": values})["interpolated"])
    expected = np.array([[[0.0, 0.5, 1.0, 2.5, 4.0]]], dtype=np.float32)

    assert result.shape == values.shape
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_polynomial_interpolation(tmp_path):
    x = Input("polynomial_input")
    x_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
    y_points = [value**2 + 2.0 * value + 3.0 for value in x_points]
    interpolated = Interpolation(
        x_points=x_points,
        y_points=y_points,
        mode="polynomial",
        name="polynomial_interpolation",
    )(x.sw(7))
    model = Modely(
        "polynomial_interpolation_model",
        inputs=[x],
        outputs=[Output("interpolated", interpolated)],
    ).build()

    values = np.array([[[-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]]], dtype=np.float32)
    clipped = np.clip(values, x_points[0], x_points[-1])
    expected = clipped**2 + 2.0 * clipped + 3.0
    result = to_numpy(model({"polynomial_input": values})["interpolated"])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    export_path = tmp_path / "polynomial_interpolation.keras"
    model.export_keras(export_path)
    restored = Modely.import_keras(export_path)
    restored_result = to_numpy(
        restored({"polynomial_input": values})["interpolated"]  # type: ignore
    )
    np.testing.assert_allclose(restored_result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("x_points", "y_points", "mode"),
    [
        ([0.0], [1.0], "linear"),
        ([0.0, 1.0], [1.0], "linear"),
        ([[0.0, 1.0]], [[1.0, 2.0]], "linear"),
        ([0.0, 0.0], [1.0, 2.0], "linear"),
        ([0.0, 1.0], [1.0, 2.0], "cubic"),
    ],
)
def test_interpolation_validation(x_points, y_points, mode):
    with pytest.raises(ValueError):
        Interpolation(x_points=x_points, y_points=y_points, mode=mode)


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
    dummy_input_x = (
        np.random.rand(batch_size, 1, max_time) * 2 - 1
    )  # Random values in range [-1, 1]
    result = model({"x": dummy_input_x})
    assert "out_sin" in result
    assert result["out_sin"].shape == (batch_size, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["out_sin"]),
        np.sin(dummy_input_x[:, :, -1:]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "out_cos" in result
    assert result["out_cos"].shape == (batch_size, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result["out_cos"]),
        np.cos(dummy_input_x[:, :, -3:]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "out_tan" in result
    assert result["out_tan"].shape == (batch_size, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["out_tan"]),
        np.tan(dummy_input_x[:, :, -1:]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "out_asin" in result
    assert result["out_asin"].shape == (batch_size, 1, 4)
    np.testing.assert_allclose(
        to_numpy(result["out_asin"]),
        np.arcsin(dummy_input_x[:, :, -4:]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "out_acos" in result
    assert result["out_acos"].shape == (batch_size, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["out_acos"]),
        np.arccos(dummy_input_x[:, :, -1:]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "out_atan" in result
    assert result["out_atan"].shape == (batch_size, 1, 2)
    np.testing.assert_allclose(
        to_numpy(result["out_atan"]),
        np.arctan(dummy_input_x[:, :, -2:]),
        rtol=1e-5,
        atol=1e-5,
    )


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


def test_concatenate_multiple_tensors_on_multidimensional_axis():
    x = Input("concat_x", dim=(2, 2))
    y = Input("concat_y", dim=(1, 2))
    z = Input("concat_z", dim=(3, 2))
    concatenated = Concatenate(axis=0, name="dim_concat")([x.sw(2), y.sw(2), z.sw(2)])
    model = Modely(
        "dim_concatenate_model",
        inputs=[x, y, z],
        outputs=[Output("result", concatenated)],
    ).build()

    x_data = np.arange(8, dtype=np.float32).reshape((1, 2, 2, 2))
    y_data = (10 + np.arange(4, dtype=np.float32)).reshape((1, 1, 2, 2))
    z_data = (20 + np.arange(12, dtype=np.float32)).reshape((1, 3, 2, 2))
    result = to_numpy(
        model({"concat_x": x_data, "concat_y": y_data, "concat_z": z_data})["result"]
    )

    assert concatenated.dim == (6, 2)
    assert concatenated.time == 2
    assert result.shape == (1, 6, 2, 2)
    np.testing.assert_array_equal(
        result, np.concatenate([x_data, y_data, z_data], axis=1)
    )


def test_concatenate_supports_second_and_negative_dim_axes():
    x = Input("axis_x", dim=(2, 1))
    y = Input("axis_y", dim=(2, 2))
    z = Input("axis_z", dim=(2, 3))
    positive = Concatenate(axis=1)([x.last(), y.last(), z.last()])
    negative = Concatenate(axis=-1)([x.last(), y.last(), z.last()])
    model = Modely(
        "axis_concatenate_model",
        inputs=[x, y, z],
        outputs=[Output("positive", positive), Output("negative", negative)],
    ).build()

    x_data = np.arange(2, dtype=np.float32).reshape((1, 2, 1, 1))
    y_data = (10 + np.arange(4, dtype=np.float32)).reshape((1, 2, 2, 1))
    z_data = (20 + np.arange(6, dtype=np.float32)).reshape((1, 2, 3, 1))
    result = model({"axis_x": x_data, "axis_y": y_data, "axis_z": z_data})
    expected = np.concatenate([x_data, y_data, z_data], axis=2)

    assert positive.dim == negative.dim == (2, 6)
    np.testing.assert_array_equal(to_numpy(result["positive"]), expected)
    np.testing.assert_array_equal(to_numpy(result["negative"]), expected)


def test_time_concatenate_multiple_tensors():
    x = Input("time_x", dim=2)
    y = Input("time_y", dim=2)
    z = Input("time_z", dim=2)
    concatenated = TimeConcatenate(name="time_concat")([x.sw(2), y.sw(1), z.sw(3)])
    model = Modely(
        "time_concatenate_model",
        inputs=[x, y, z],
        outputs=[Output("result", concatenated)],
    ).build()

    x_data = np.arange(4, dtype=np.float32).reshape((1, 2, 2))
    y_data = (10 + np.arange(2, dtype=np.float32)).reshape((1, 2, 1))
    z_data = (20 + np.arange(6, dtype=np.float32)).reshape((1, 2, 3))
    result = to_numpy(
        model({"time_x": x_data, "time_y": y_data, "time_z": z_data})["result"]
    )

    assert concatenated.dim == (2,)
    assert concatenated.time == 6
    assert result.shape == (1, 2, 6)
    np.testing.assert_array_equal(
        result, np.concatenate([x_data, y_data, z_data], axis=2)
    )


def test_concatenate_rejects_incompatible_shapes():
    x = Input("invalid_concat_x", dim=(2, 2))
    y = Input("invalid_concat_y", dim=(1, 3))

    with pytest.raises(ValueError, match="matching dimensions"):
        Concatenate(axis=0)([x.last(), y.last()])

    with pytest.raises(ValueError, match="matching dim and seq"):
        TimeConcatenate()([x.last(), y.last()])

    with pytest.raises(ValueError, match="at least two inputs"):
        Concatenate()([x.last()])


def test_fir_simple():
    input1 = Input("in1")
    rel1 = Fir(out_features=1)([input1.last()])
    fun = Output("out", rel1)

    test = Modely(name="test_model", inputs=[input1], outputs=[fun])
    test.build()


def test_double_fir_simple():
    input1 = Input("in1")
    rel1 = Fir(out_features=1)([input1.sw(5)])
    rel2 = Fir(out_features=1)([input1.sw(1)])
    fun = Output("out", rel1 + rel2)

    test = Modely(name="test_model", inputs=[input1], outputs=[fun])
    test.build()


def test_fir_tw():
    input1 = Input("in1")
    input2 = Input("in2")
    rel1 = Fir(out_features=1)([input1.sw(5)])
    rel2 = Fir(out_features=1)([input1.sw(1)])
    rel3 = Fir(out_features=1)([input2.sw(1)])
    rel4 = Fir(out_features=1)([input2.sw([2, 2])])
    fun = Output("out", rel1 + rel2 + rel3 + rel4)

    test = Modely(name="test_model", inputs=[input1, input2], outputs=[fun])
    test.build()


def test_fir_tw2():
    input1 = Input("in1")
    rel3 = Fir(out_features=1)([input1.sw(5)])
    rel4 = Fir(out_features=1)([input1.sw([2, 2])])
    rel5 = Fir(out_features=1)([input1.sw([3, 3])])
    rel6 = Fir(out_features=1)([input1.sw([3, 0])])
    rel7 = Fir(out_features=1)([input1.sw(3)])
    fun = Output("out", rel3 + rel4 + rel5 + rel6 + rel7)

    test = Modely(name="test_model", inputs=[input1], outputs=[fun])
    test.build()


def test_fir_tw3():
    input1 = Input("in1")
    rel3 = Fir(out_features=1)([input1.sw(5)])
    rel4 = Fir(out_features=1)([input1.sw([1, 3])])
    rel5 = Fir(out_features=1)([input1.sw([4, 1])])
    fun = Output("out", rel3 + rel4 + rel5)

    test = Modely(name="test_model", inputs=[input1], outputs=[fun])
    test.build()
