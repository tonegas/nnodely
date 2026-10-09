from nnodely import (
    BatchNorm,
    DataLoader,
    Linear,
    Constant,
    Concatenate,
    Input,
    Interpolation,
    Modely,
    Output,
    Parameter,
    Range,
    Select,
    TimeConcatenate,
    TimeRange,
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


def test_time_range():
    x = Input("time_range_input")
    selected = TimeRange(start=1, end=4, name="time_range")(x.sw(5))

    assert selected.dim == (1,)
    assert selected.time == 3
    assert selected.seq == ()

    model = Modely(
        "time_range_model",
        inputs=[x],
        outputs=[Output("time_range_output", selected)],
    ).build()
    values = np.arange(5, dtype=np.float32).reshape((1, 1, 5))
    result = model({"time_range_input": values})

    assert result["time_range_output"].shape == (1, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result["time_range_output"]),
        np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32),
    )


def test_range():
    x = Input("range_input", dim=4)
    selected = Range(start=1, end=3, axis=0, name="range")(x.last())

    assert selected.dim == (2,)
    assert selected.time == 1
    assert selected.seq == ()

    model = Modely(
        "range_model",
        inputs=[x],
        outputs=[Output("range_output", selected)],
    ).build()
    values = np.arange(4, dtype=np.float32).reshape((1, 4, 1))
    result = model({"range_input": values})

    assert result["range_output"].shape == (1, 2, 1)
    np.testing.assert_allclose(
        to_numpy(result["range_output"]),
        np.array([[[1.0], [2.0]]], dtype=np.float32),
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
    model.export_keras(tmp_path, "polynomial_interpolation")
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


def test_batchnorm_inference_and_training():
    x = Input("batchnorm_input", dim=2)
    normalized = BatchNorm(name="batchnorm")([x.sw(3)])

    assert normalized.dim == (2,)
    assert normalized.time == 3
    assert normalized.seq == ()

    model = Modely(
        "batchnorm_model",
        inputs=[x],
        outputs=[Output("out_batchnorm", normalized)],
    ).build()
    assert model.model is not None
    values = np.random.rand(4, 2, 3).astype(np.float32) * 10.0

    # Untrained moving statistics (mean 0, variance 1) leave the input untouched.
    inference = to_numpy(
        model.model({"batchnorm_input": values}, training=False)["out_batchnorm"]
    )
    assert inference.shape == (4, 2, 3)
    np.testing.assert_allclose(inference, values, rtol=1e-3, atol=1e-3)

    # In training mode each feature is normalized over the batch and time axes.
    training = to_numpy(
        model.model({"batchnorm_input": values}, training=True)["out_batchnorm"]
    )
    np.testing.assert_allclose(training.mean(axis=(0, 2)), np.zeros(2), atol=1e-5)
    np.testing.assert_allclose(training.std(axis=(0, 2)), np.ones(2), atol=1e-3)


def test_batchnorm_axis_and_moving_statistics():
    x = Input("batchnorm_axis_input", dim=2)
    normalized = BatchNorm(axis=-1, name="batchnorm_axis")([x.sw(3)])
    model = Modely(
        "batchnorm_axis_model",
        inputs=[x],
        outputs=[Output("out_batchnorm_axis", normalized)],
    ).build()
    assert model.model is not None
    values = np.random.rand(4, 2, 3).astype(np.float32) * 10.0
    result = to_numpy(
        model.model({"batchnorm_axis_input": values}, training=True)[
            "out_batchnorm_axis"
        ]
    )

    # Normalizing on the time axis gives one statistic per time step instead.
    np.testing.assert_allclose(result.mean(axis=(0, 1)), np.zeros(3), atol=1e-5)

    moving_mean = to_numpy(normalized._layer.moving_mean)
    assert moving_mean.shape == (3,)
    assert np.any(moving_mean != 0.0)


# ---------------------------------------------------------------------------
# Linear takes the arguments it declares, and keeps them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argument",
    [{"output_dimension": 4}, {"dim": 4}],
    ids=["misspelled", "shape_of_the_node"],
)
def test_linear_rejects_an_argument_it_does_not_declare(argument):
    # Once kept aside and ignored: output_dimension built one output, and dim
    # set the shape of the node rather than the size of the projection.
    with pytest.raises(TypeError, match=next(iter(argument))):
        Linear(**argument)


def test_a_frozen_linear_stays_frozen_through_a_keras_file(tmp_path):
    x = Input("frozen_x", dim=2)
    model = Modely(
        "frozen",
        inputs=[x],
        outputs=[
            Output("frozen_out", Linear(out_features=3, name="frozen_linear")(x.last()))
        ],
    ).build()
    assert model.inference_model is not None
    model.inference_model.get_layer("frozen_linear").trainable = False

    model.export_keras(tmp_path, "frozen")
    restored = Modely.import_keras(tmp_path / "frozen.keras")

    assert restored.get_layer("frozen_linear").trainable is False  # type: ignore


def test_linear_initializers_are_saved_with_the_architecture(tmp_path):
    x = Input("initialized_x", dim=2)
    linear = Linear(
        out_features=3, kernel="ones", bias="zeros", name="initialized_linear"
    )(x.last())
    model = Modely(
        "initialized", inputs=[x], outputs=[Output("initialized_out", linear)]
    ).build()

    model.save(tmp_path / "initialized", weights=False)
    restored = {node.name: node for node in Modely.load(tmp_path / "initialized").order}
    restored_linear = restored["initialized_linear"]

    assert isinstance(restored_linear, Linear)
    np.testing.assert_allclose(to_numpy(restored_linear.kernel), np.ones((2, 3)))
    np.testing.assert_allclose(to_numpy(restored_linear.bias), np.zeros(3))


def test_linear_takes_its_kernel_and_bias_from_parameters():
    x = Input("linear_param_x", dim=3)
    w = Parameter("linear_param_w", value=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = Parameter("linear_param_b", value=[0.5, -1.0])
    linear = Linear(out_features=2, kernel=w, bias=b)([x.sw(2)])
    unbiased = Linear(out_features=2, kernel=w, bias=False)([x.sw(2)])
    model = Modely(
        "linear_param_model",
        inputs=[x],
        outputs=[
            Output("linear_param_out", linear),
            Output("linear_param_unbiased", unbiased),
        ],
    ).build()

    # The Parameters are predecessors of the Linear, and the only weights.
    assert linear.preds[1] is w and linear.preds[2] is b
    assert linear.kernel is w.param and linear.bias is b.param
    assert unbiased.bias is None
    assert model.model is not None
    assert {id(weight) for weight in model.model.trainable_weights} == {
        id(w.param),
        id(b.param),
    }

    values = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    result = model({"linear_param_x": values})
    # Each sample [a, b, c] -> [a + c, b + c], plus the bias.
    np.testing.assert_allclose(
        to_numpy(result["linear_param_unbiased"]), [[[6.0, 8.0], [8.0, 10.0]]]
    )
    np.testing.assert_allclose(
        to_numpy(result["linear_param_out"]), [[[6.5, 8.5], [7.0, 9.0]]]
    )


def test_linear_rejects_parameters_of_the_wrong_shape():
    x = Input("linear_bad_x", dim=3)
    with pytest.raises(ValueError, match=r"kernel must have the shape \(3, 2\)"):
        Linear(out_features=2, kernel=Parameter("linear_bad_w", dim=(2, 3)))([x.last()])
    with pytest.raises(ValueError, match=r"bias must have the shape \(2,\)"):
        Linear(out_features=2, bias=Parameter("linear_bad_b", dim=3))([x.last()])


def test_linear_with_parameters_round_trips_through_save_and_keras(tmp_path):
    x = Input("linear_kept_x", dim=2)
    w = Parameter("linear_kept_w", dim=(2, 3))
    linear = Linear(out_features=3, kernel=w, bias="ones", name="linear_kept")(
        [x.last()]
    )
    model = Modely(
        "linear_kept", inputs=[x], outputs=[Output("linear_kept_out", linear)]
    ).build()
    values = np.random.default_rng(11).normal(size=(2, 2, 1)).astype(np.float32)
    before = to_numpy(model({"linear_kept_x": values})["linear_kept_out"])

    model.save(tmp_path / "linear_kept")
    restored = Modely.load(tmp_path / "linear_kept")
    restored_linear = {node.name: node for node in restored.order}["linear_kept"]
    assert isinstance(restored_linear, Linear)
    assert [pred.name for pred in restored_linear.preds[1:]] == [
        "linear_kept_w",
        "linear_kept_bias",
    ]
    np.testing.assert_allclose(
        to_numpy(restored({"linear_kept_x": values})["linear_kept_out"]),
        before,
        rtol=1e-5,
        atol=1e-5,
    )

    model.export_keras(tmp_path, "linear_kept")
    imported = Modely.import_keras(tmp_path / "linear_kept.keras")
    np.testing.assert_allclose(
        to_numpy(imported({"linear_kept_x": values})["linear_kept_out"]),  # type: ignore
        before,
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Linear on a multi-dimensional input: the whole dim, or one axis of it
# ---------------------------------------------------------------------------


def _linear_values(shape, seed):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def test_linear_projects_the_whole_dim_by_default():
    # (batch, 2, 3, time, seq) -> (batch, out_features, time, seq): every one
    # of the 6 features of a sample feeds every output.
    x = Input("lin_dim_x", dim=(2, 3), seq=2)
    summed = Linear(out_features=2, kernel="ones", bias=False, name="lin_dim_sum")(
        x.sw(4)
    )
    drawn = Linear(out_features=2, name="lin_dim_drawn")(x.sw(4))
    model = Modely(
        "lin_dim_model",
        inputs=[x],
        outputs=[Output("lin_dim_sum", summed), Output("lin_dim_drawn", drawn)],
    ).build()
    assert summed.shape.dimensions == ((2,), 4, (2,))
    assert drawn.kernel is not None and drawn.bias is not None
    assert tuple(drawn.kernel.shape) == (2, 3, 2)
    assert tuple(drawn.bias.shape) == (2,)

    values = _linear_values((5, 2, 3, 4, 2), seed=12)
    result = model({"lin_dim_x": values})

    # With a kernel of ones, both outputs are the sum of the 6 features.
    total = values.sum(axis=(1, 2))
    np.testing.assert_allclose(
        to_numpy(result["lin_dim_sum"]), np.stack([total, total], axis=1), rtol=1e-5
    )
    expected = np.einsum("bijts,ijo->bots", values, to_numpy(drawn.kernel)) + to_numpy(
        drawn.bias
    ).reshape(1, 2, 1, 1)
    np.testing.assert_allclose(
        to_numpy(result["lin_dim_drawn"]), expected, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "axis, out_dim, kernel_shape, subscripts",
    [
        (0, (2, 3), (2, 2), "bijts,io->bojts"),
        (1, (2, 2), (3, 2), "bijts,jo->biots"),
        (-1, (2, 2), (3, 2), "bijts,jo->biots"),
    ],
)
def test_linear_with_an_axis_projects_that_axis_alone(
    tmp_path, axis, out_dim, kernel_shape, subscripts
):
    # The same matrix along the axis, at every position of the other one;
    # out_features takes the place of the axis.
    x = Input("lin_axis_x", dim=(2, 3), seq=2)
    linear = Linear(out_features=2, axis=axis, name="lin_axis")(x.sw(4))
    model = Modely(
        "lin_axis_model", inputs=[x], outputs=[Output("lin_axis_out", linear)]
    ).build()
    assert linear.shape.dimensions == (out_dim, 4, (2,))
    assert linear.kernel is not None and linear.bias is not None
    assert tuple(linear.kernel.shape) == kernel_shape

    values = _linear_values((5, 2, 3, 4, 2), seed=13)
    result = to_numpy(model({"lin_axis_x": values})["lin_axis_out"])
    bias_shape = [1] * result.ndim
    bias_shape[1 + axis % 2] = 2
    expected = np.einsum(subscripts, values, to_numpy(linear.kernel)) + to_numpy(
        linear.bias
    ).reshape(bias_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    # The axis is part of the architecture: saved and loaded with it.
    model.save(tmp_path / "lin_axis")
    restored = Modely.load(tmp_path / "lin_axis")
    np.testing.assert_allclose(
        to_numpy(restored({"lin_axis_x": values})["lin_axis_out"]),
        result,
        rtol=1e-5,
        atol=1e-5,
    )


def test_linear_rejects_an_axis_its_input_does_not_have():
    x = Input("lin_bad_axis_x", dim=(2, 3))
    for axis in (2, -3):
        with pytest.raises(ValueError, match=f"axis {axis} is not an axis"):
            Linear(out_features=2, axis=axis)(x.last())


def test_linear_draws_a_whole_dim_kernel_as_one_matrix():
    # fan_out of the (50, 2, 1) kernel is the 1 output, a bound of sqrt(3).
    # Read as a (2, 1) kernel of a 50-wide convolution, Keras would count
    # 50 outputs instead, a bound of about 0.24.
    x = Input("lin_draw_x", dim=(50, 2))
    linear = Linear(
        out_features=1,
        kernel=keras.initializers.VarianceScaling(
            mode="fan_out", distribution="uniform"
        ),
        bias=False,
    )(x.last())
    Modely("lin_draw", inputs=[x], outputs=[Output("lin_draw_out", linear)]).build()

    kernel = np.abs(to_numpy(linear.kernel))
    assert kernel.max() <= np.sqrt(3.0)
    assert kernel.max() > 0.5


# ---------------------------------------------------------------------------
# Select only takes an index its dim axis has
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dim, axis, idx",
    [(3, 0, 3), (3, 0, 5), (3, 0, -4), ((4, 3), 1, 3), ((4, 3), 1, -4)],
    ids=["one_past", "far_past", "one_before", "second_axis", "second_axis_negative"],
)
def test_select_rejects_an_index_out_of_its_dim_axis(dim, axis, idx):
    # Once accepted: the stream came out with an empty dim axis.
    x = Input("select_bounds_x", dim=dim)
    with pytest.raises(ValueError, match=f"idx {idx} out of bounds"):
        Select(idx=idx, axis=axis)([x.last()])


def test_select_takes_every_index_of_its_dim_axis():
    x = Input("select_every_x", dim=3)
    outputs = [
        Output(f"select_{index + 3}", Select(idx=index)([x.last()]))
        for index in range(-3, 3)
    ]
    model = Modely("select_every", inputs=[x], outputs=outputs).build()

    result = model(
        {"select_every_x": np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)}
    )

    picked = [
        float(to_numpy(result[f"select_{index + 3}"]).ravel()[0])
        for index in range(-3, 3)
    ]
    assert picked == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Softmax normalizes the whole sample by default
# ---------------------------------------------------------------------------


def test_softmax_normalizes_every_value_of_each_sample():
    # Once over the last tensor axis only: on x.last() that is the time axis,
    # one value long, so every output was 1.
    x = Input("softmax_whole_x", dim=3)
    y = Input("softmax_whole_y", dim=2, seq=2)
    model = Modely(
        "softmax_whole",
        inputs=[x, y],
        outputs=[
            Output("softmax_dim", Softmax()([x.last()])),
            Output("softmax_all", Softmax()([y.sw(3)])),
        ],
    ).build()
    rng = np.random.default_rng(0)
    x_values = rng.normal(size=(2, 3, 1)).astype(np.float32)
    y_values = rng.normal(size=(2, 2, 3, 2)).astype(np.float32)

    result = model({"softmax_whole_x": x_values, "softmax_whole_y": y_values})

    for name, values in (("softmax_dim", x_values), ("softmax_all", y_values)):
        flat = values.reshape(len(values), -1)
        expected = np.exp(flat) / np.exp(flat).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(
            to_numpy(result[name]).reshape(len(values), -1), expected, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# Fir filters the time axis of every element alone, one sequence step at a time
# ---------------------------------------------------------------------------


def _fir_weights(fir, kernel_shape, bias_shape):
    kernel = np.arange(np.prod(kernel_shape), dtype=np.float32)
    kernel = kernel.reshape(kernel_shape) / 10.0
    fir.kernel.assign(kernel)
    fir.bias.assign(np.full(bias_shape, 0.5, dtype=np.float32))
    return kernel


def _fir_expected(values, kernel, bias=0.5):
    """``values [batch, *dim, time]`` filtered by ``kernel [*dim, time, out]``
    (or ``[time, out]``, shared), as ``[batch, out, *dim]``."""
    out = np.einsum(
        "b...t,...to->bo...",
        values,
        np.broadcast_to(kernel, values.shape[1:] + kernel.shape[-1:]),
    )
    return out + bias


@pytest.mark.parametrize(
    "dim, out_features, expected",
    [
        ((1,), 1, (1,)),
        ((1,), 4, (4,)),
        ((2,), 1, (2,)),
        ((2,), 3, (3, 2)),
        ((2, 3), 1, (2, 3)),
        ((2, 3), 4, (4, 2, 3)),
    ],
)
@pytest.mark.parametrize("shared_kernel", [False, True])
def test_fir_adds_a_channel_axis_in_front_of_the_dim(
    dim, out_features, expected, shared_kernel
):
    x = Input("fir_dim_x", dim=dim)
    fir = Fir(out_features=out_features, shared_kernel=shared_kernel)([x.sw(3)])
    model = Modely("fir_dim_model", inputs=[x], outputs=[Output("fir_dim_out", fir)])
    model.build()
    assert fir.shape.dimensions == (expected, 1, ())

    values = np.ones((2, *dim, 3), dtype=np.float32)
    result = to_numpy(model({"fir_dim_x": values})["fir_dim_out"])
    assert result.shape == (2, *expected, 1)


def test_fir_filters_every_element_with_its_own_kernel():
    x = Input("fir_own_x", dim=(2, 3))
    fir = Fir(out_features=4, name="fir_own")([x.sw(5)])
    model = Modely("fir_own_model", inputs=[x], outputs=[Output("fir_own_out", fir)])
    model.build()
    assert fir.kernel is not None and fir.bias is not None
    assert tuple(fir.kernel.shape) == (2, 3, 5, 4)
    assert tuple(fir.bias.shape) == (2, 3, 4)

    kernel = _fir_weights(fir, (2, 3, 5, 4), (2, 3, 4))
    values = np.random.default_rng(3).normal(size=(2, 2, 3, 5)).astype(np.float32)
    result = to_numpy(model({"fir_own_x": values})["fir_own_out"])

    assert result.shape == (2, 4, 2, 3, 1)
    np.testing.assert_allclose(
        result[..., 0], _fir_expected(values, kernel), rtol=1e-5, atol=1e-5
    )
    # Elements are never mixed: element (0, 0) depends on its own window only.
    changed = values.copy()
    changed[:, 1, 2] += 1.0
    changed_result = to_numpy(model({"fir_own_x": changed})["fir_own_out"])
    np.testing.assert_allclose(changed_result[:, :, 0, 0], result[:, :, 0, 0])


def test_fir_with_a_shared_kernel_filters_every_element_alike():
    x = Input("fir_shared_x", dim=(2, 3))
    fir = Fir(out_features=4, shared_kernel=True, name="fir_shared")([x.sw(5)])
    model = Modely(
        "fir_shared_model", inputs=[x], outputs=[Output("fir_shared_out", fir)]
    )
    model.build()
    assert fir.kernel is not None and fir.bias is not None
    assert tuple(fir.kernel.shape) == (5, 4)
    assert tuple(fir.bias.shape) == (4,)

    kernel = _fir_weights(fir, (5, 4), (4,))
    values = np.random.default_rng(4).normal(size=(2, 2, 3, 5)).astype(np.float32)
    result = to_numpy(model({"fir_shared_x": values})["fir_shared_out"])

    assert result.shape == (2, 4, 2, 3, 1)
    np.testing.assert_allclose(
        result[..., 0], _fir_expected(values, kernel), rtol=1e-5, atol=1e-5
    )


def test_fir_with_one_channel_keeps_the_dim():
    x = Input("fir_one_x", dim=(2, 3))
    fir = Fir(out_features=1, name="fir_one")([x.sw(4)])
    model = Modely("fir_one_model", inputs=[x], outputs=[Output("fir_one_out", fir)])
    model.build()

    kernel = _fir_weights(fir, (2, 3, 4, 1), (2, 3, 1))
    values = np.random.default_rng(5).normal(size=(2, 2, 3, 4)).astype(np.float32)
    result = to_numpy(model({"fir_one_x": values})["fir_one_out"])

    assert result.shape == (2, 2, 3, 1)
    np.testing.assert_allclose(
        result[..., 0], _fir_expected(values, kernel)[:, 0], rtol=1e-5, atol=1e-5
    )


def test_fir_keeps_the_sequence_axes():
    # Once flattened into the projection: every output mixed every step of
    # the sequence, and the seq axis was gone from the result.
    x = Input("fir_seq_x", dim=2, seq=4)
    fir = Fir(out_features=3, name="fir_seq")([x.sw(3)])
    model = Modely("fir_seq_model", inputs=[x], outputs=[Output("fir_seq_out", fir)])
    model.build()
    assert fir.shape.dimensions == ((3, 2), 1, (4,))

    kernel = _fir_weights(fir, (2, 3, 3), (2, 3))
    values = np.random.default_rng(1).normal(size=(2, 2, 3, 4)).astype(np.float32)
    result = to_numpy(model({"fir_seq_x": values})["fir_seq_out"])

    assert result.shape == (2, 3, 2, 1, 4)
    for step in range(4):
        np.testing.assert_allclose(
            result[:, :, :, 0, step],
            _fir_expected(values[..., step], kernel),
            rtol=1e-5,
            atol=1e-5,
        )


def test_fir_without_sequence_axes_filters_the_whole_window():
    x = Input("fir_plain_x")
    fir = Fir(out_features=3, name="fir_plain")([x.sw(3)])
    model = Modely(
        "fir_plain_model", inputs=[x], outputs=[Output("fir_plain_out", fir)]
    )
    model.build()

    kernel = _fir_weights(fir, (3, 3), (3,))
    values = np.random.default_rng(2).normal(size=(2, 1, 3)).astype(np.float32)
    result = to_numpy(model({"fir_plain_x": values})["fir_plain_out"])

    assert result.shape == (2, 3, 1)
    np.testing.assert_allclose(
        result[:, :, 0], values.reshape(2, -1) @ kernel + 0.5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "dim, out_features, expected",
    [((1,), 2, (2,)), ((2, 3), 2, (2, 2, 3)), ((2, 3), 1, (2, 3))],
)
def test_fir_follows_a_dynamic_sequence_length(dim, out_features, expected):
    x = Input("fir_dynamic_x", dim=dim, seq=-1)
    fir = Fir(out_features=out_features, name="fir_dynamic")([x.sw(2)])
    model = Modely(
        "fir_dynamic_model", inputs=[x], outputs=[Output("fir_dynamic_out", fir)]
    )
    model.build()

    for length in (3, 5):
        values = np.ones((1, *dim, 2, length), dtype=np.float32)
        result = to_numpy(model({"fir_dynamic_x": values})["fir_dynamic_out"])
        assert result.shape == (1, *expected, 1, length)


def test_fir_shared_kernel_is_saved_with_the_architecture(tmp_path):
    x = Input("fir_saved_x", dim=(2, 3))
    fir = Fir(out_features=4, shared_kernel=True, name="fir_saved")([x.sw(3)])
    model = Modely(
        "fir_saved", inputs=[x], outputs=[Output("fir_saved_out", fir)]
    ).build()
    values = np.random.default_rng(6).normal(size=(2, 2, 3, 3)).astype(np.float32)
    before = to_numpy(model({"fir_saved_x": values})["fir_saved_out"])

    model.save(tmp_path / "fir_saved")
    restored = Modely.load(tmp_path / "fir_saved")
    restored_fir = {node.name: node for node in restored.order}["fir_saved"]

    assert isinstance(restored_fir, Fir) and restored_fir.shared_kernel
    assert restored_fir.kernel is not None
    assert tuple(restored_fir.kernel.shape) == (3, 4)
    np.testing.assert_allclose(
        to_numpy(restored({"fir_saved_x": values})["fir_saved_out"]),
        before,
        rtol=1e-5,
        atol=1e-5,
    )


def test_fir_draws_the_filter_of_each_element_alone():
    # Glorot takes its fans from the (time, out_features) filter of one
    # element: 4 + 2 here, a bound of 1. Drawn as one (10, 10, 4, 2) tensor,
    # the fans would count the 100 elements too, a bound of about 0.14.
    x = Input("fir_draw_x", dim=(10, 10))
    fir = Fir(out_features=2, name="fir_draw")([x.sw(4)])
    Modely("fir_draw", inputs=[x], outputs=[Output("fir_draw_out", fir)]).build()

    kernel = to_numpy(fir.kernel)
    assert np.abs(kernel).max() <= 1.0
    assert np.abs(kernel).max() > 0.5
    assert not np.allclose(kernel[0, 0], kernel[0, 1])
    np.testing.assert_allclose(to_numpy(fir.bias), np.zeros((10, 10, 2)))


def test_fir_initializers_are_saved_with_the_architecture(tmp_path):
    x = Input("fir_init_x", dim=2)
    fir = Fir(
        out_features=3,
        kernel="ones",
        bias=keras.initializers.Constant(0.5),
        name="fir_init",
    )([x.sw(4)])
    model = Modely(
        "fir_init", inputs=[x], outputs=[Output("fir_init_out", fir)]
    ).build()
    np.testing.assert_allclose(to_numpy(fir.kernel), np.ones((2, 4, 3)))
    np.testing.assert_allclose(to_numpy(fir.bias), np.full((2, 3), 0.5))

    model.save(tmp_path / "fir_init", weights=False)
    restored = {node.name: node for node in Modely.load(tmp_path / "fir_init").order}
    restored_fir = restored["fir_init"]

    assert isinstance(restored_fir, Fir)
    np.testing.assert_allclose(to_numpy(restored_fir.kernel), np.ones((2, 4, 3)))
    np.testing.assert_allclose(to_numpy(restored_fir.bias), np.full((2, 3), 0.5))


# ---------------------------------------------------------------------------
# Fir with its kernel and bias given as Parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shared_kernel, kernel_dim, bias_dim",
    [(False, (3, 3, 4), (3, 4)), (True, (3, 4), (4,))],
)
def test_fir_takes_its_kernel_and_bias_from_parameters(
    shared_kernel, kernel_dim, bias_dim
):
    x = Input("fir_param_x", dim=3)
    w = Parameter("fir_param_w", dim=kernel_dim)
    b = Parameter("fir_param_b", dim=bias_dim)
    fir = Fir(out_features=4, kernel=w, bias=b, shared_kernel=shared_kernel)([x.sw(3)])
    model = Modely(
        "fir_param_model", inputs=[x], outputs=[Output("fir_param_out", fir)]
    ).build()

    # The Parameters are predecessors of the Fir, and the only weights.
    assert fir.preds[1] is w and fir.preds[2] is b
    assert fir.kernel is w.param and fir.bias is b.param
    assert model.model is not None
    assert {id(weight) for weight in model.model.trainable_weights} == {
        id(w.param),
        id(b.param),
    }

    # Assigning the Parameters sets the filter.
    assert w.param is not None and b.param is not None
    kernel = np.arange(np.prod(kernel_dim), dtype=np.float32).reshape(kernel_dim)
    w.param.assign(kernel.reshape(w.param.shape) / 10.0)
    b.param.assign(np.full(b.param.shape, 0.5, dtype=np.float32))
    values = np.random.default_rng(8).normal(size=(2, 3, 3)).astype(np.float32)
    result = to_numpy(model({"fir_param_x": values})["fir_param_out"])

    assert result.shape == (2, 4, 3, 1)
    np.testing.assert_allclose(
        result[..., 0], _fir_expected(values, kernel / 10.0), rtol=1e-5, atol=1e-5
    )


def test_fir_takes_a_parameter_whose_shape_differs_only_by_axes_of_size_one():
    x = Input("fir_ones_x")
    # A matrix value is (dim, time): (3, 2) here, the (time, out) kernel.
    w = Parameter("fir_ones_w", value=[[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    fir = Fir(out_features=2, kernel=w, bias=False)([x.sw(3)])
    model = Modely(
        "fir_ones_model", inputs=[x], outputs=[Output("fir_ones_out", fir)]
    ).build()
    assert fir.bias is None

    values = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    result = to_numpy(model({"fir_ones_x": values})["fir_ones_out"])
    # Channel 0 reads the oldest sample, channel 1 twice the newest.
    np.testing.assert_allclose(result.ravel(), [1.0, 6.0], rtol=1e-5)


def test_firs_given_one_parameter_share_its_weight(tmp_path):
    x = Input("fir_twin_x")
    y = Input("fir_twin_y")
    w = Parameter("fir_twin_w", value=np.ones((3, 1)))
    first = Fir(out_features=1, kernel=w, bias=False)([x.sw(3)])
    second = Fir(out_features=1, kernel=w, bias=False)([y.sw(3)])
    model = Modely(
        "fir_twin_model",
        inputs=[x, y],
        outputs=[Output("fir_twin_first", first), Output("fir_twin_second", second)],
    ).build()
    data = {
        "fir_twin_x": np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32),
        "fir_twin_y": np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32),
    }

    # One variable, held by the Parameter, behind both Firs.
    assert first.kernel is w.param and second.kernel is w.param
    assert model.model is not None and len(model.model.trainable_weights) == 1

    # Changed through one Fir, the kernel changes for the other as well.
    first.kernel.assign(np.array([[0.0], [0.0], [2.0]], dtype=np.float32))
    result = model(data)
    np.testing.assert_allclose(to_numpy(result["fir_twin_first"]).ravel(), [6.0])
    np.testing.assert_allclose(to_numpy(result["fir_twin_second"]).ravel(), [60.0])

    # Still one weight once saved and loaded.
    model.save(tmp_path / "fir_twin")
    restored = Modely.load(tmp_path / "fir_twin")
    assert restored.model is not None
    assert len(restored.model.trainable_weights) == 1
    restored_result = restored(data)
    for name in ("fir_twin_first", "fir_twin_second"):
        np.testing.assert_allclose(
            to_numpy(restored_result[name]), to_numpy(result[name])
        )


def test_fir_rejects_parameters_it_cannot_use():
    x = Input("fir_bad_x", dim=3)
    # (time, out) is the kernel shared by every element, not one per element.
    with pytest.raises(ValueError, match=r"kernel must have the shape \(3, 3, 4\)"):
        Fir(out_features=4, kernel=Parameter("fir_bad_w", dim=(3, 4)))([x.sw(3)])
    with pytest.raises(ValueError, match=r"bias must have the shape \(3, 4\)"):
        Fir(out_features=4, bias=Parameter("fir_bad_b", dim=(4,)))([x.sw(3)])


@pytest.mark.slow
def test_fir_trains_the_parameters_it_is_given():
    x = Input("fir_train_x")
    w = Parameter("fir_train_w", value=np.zeros((3, 1)))
    b = Parameter("fir_train_b", value=[0.0])
    fir = Fir(out_features=1, kernel=w, bias=b)([x.sw(3)])
    model = Modely(
        "fir_train_model", inputs=[x], outputs=[Output("fir_train_out", fir)]
    )
    model.minimize("fit", fir, Input("fir_train_y").last())
    model.build()

    signal = np.random.default_rng(9).normal(size=100).astype(np.float32)
    data = DataLoader(model, source={"fir_train_x": signal, "fir_train_y": signal})
    model.train(data, epochs=1, batch_size=16, lr=1e-2, printer=None)

    assert not np.allclose(to_numpy(w.param), 0.0)
    assert not np.allclose(to_numpy(b.param), 0.0)


@pytest.mark.parametrize("given", [("kernel",), ("bias",), ("kernel", "bias")])
def test_fir_with_parameters_round_trips_through_save_and_keras(tmp_path, given):
    x = Input("fir_kept_x", dim=2)
    w = Parameter("fir_kept_w", dim=(2, 3, 4)) if "kernel" in given else None
    b = Parameter("fir_kept_b", dim=(2, 4)) if "bias" in given else None
    fir = Fir(
        out_features=4,
        kernel="glorot_uniform" if w is None else w,
        bias=True if b is None else b,
        name="fir_kept",
    )([x.sw(3)])
    model = Modely(
        "fir_kept", inputs=[x], outputs=[Output("fir_kept_out", fir)]
    ).build()
    values = np.random.default_rng(10).normal(size=(2, 2, 3)).astype(np.float32)
    before = to_numpy(model({"fir_kept_x": values})["fir_kept_out"])

    model.save(tmp_path / "fir_kept")
    restored = Modely.load(tmp_path / "fir_kept")
    restored_fir = {node.name: node for node in restored.order}["fir_kept"]
    assert isinstance(restored_fir, Fir)
    assert [pred.name for pred in restored_fir.preds[1:]] == [
        pred.name for pred in fir.preds[1:]
    ]
    np.testing.assert_allclose(
        to_numpy(restored({"fir_kept_x": values})["fir_kept_out"]),
        before,
        rtol=1e-5,
        atol=1e-5,
    )

    model.export_keras(tmp_path, "fir_kept")
    imported = Modely.import_keras(tmp_path / "fir_kept.keras")
    np.testing.assert_allclose(
        to_numpy(imported({"fir_kept_x": values})["fir_kept_out"]),  # type: ignore
        before,
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# A layer with nothing to configure is applied as it is created: Sin(x)
# ---------------------------------------------------------------------------

import nnodely  # noqa: E402

_APPLIED_ON_CREATION = [
    "Abs", "Acos", "Asin", "Atan", "Ceil", "Cos", "Deg2Rad", "Exp", "Floor",
    "Log", "Log10", "Negative", "Sigmoid", "Sign", "Sin", "Softplus", "Sqrt",
    "Swish", "Tan", "Tanh",
]  # fmt: skip


@pytest.mark.parametrize("layer_name", _APPLIED_ON_CREATION)
def test_a_layer_without_configuration_is_applied_on_creation(layer_name):
    layer = getattr(nnodely, layer_name)
    x = Input(f"direct_{layer_name}_x")
    window = x.sw(4)
    direct = layer(window)
    called = layer()(window)
    assert isinstance(direct, layer)
    assert direct.preds == [window]  # applied to the stream it was created with
    model = Modely(
        f"direct_{layer_name}",
        inputs=[x],
        outputs=[
            Output(f"direct_{layer_name}_out", direct),
            Output(f"called_{layer_name}_out", called),
        ],
    ).build()

    values = np.array([[[0.1, 0.4, 0.6, 0.9]]], dtype=np.float32)  # in every domain
    result = model({f"direct_{layer_name}_x": values})
    np.testing.assert_allclose(
        to_numpy(result[f"direct_{layer_name}_out"]),
        to_numpy(result[f"called_{layer_name}_out"]),
    )


def test_time_concatenate_is_applied_on_creation_to_a_list():
    x = Input("direct_join_x")
    joined = TimeConcatenate([x.sw(2), x.sw(3)])
    assert isinstance(joined, TimeConcatenate)
    assert joined.time == 5


def test_a_name_passed_positionally_still_only_configures():
    configured = Sin("direct_named_sin")
    assert configured.name == "direct_named_sin"
    assert configured.preds == []
    x = Input("direct_named_x")
    assert configured(x.sw(2)).name == "direct_named_sin"


@pytest.mark.parametrize(
    "layer",
    [ReLU, LeakyReLU, ELU, GELU, PReLU, Softmax, Linear, Fir, Concatenate],
)
def test_a_layer_with_settings_is_configured_before_it_is_called(layer):
    assert layer._applied_on_creation is False
