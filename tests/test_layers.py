from nnodely import (
    BatchNorm,
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
        out_features=3,
        initializer="ones",
        bias_initializer="zeros",
        name="initialized_linear",
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
# Fir compresses dim and time, one sequence step at a time
# ---------------------------------------------------------------------------


def _fir_kernel(fir, in_features, out_features):
    kernel = np.arange(in_features * out_features, dtype=np.float32)
    kernel = kernel.reshape(in_features, out_features) / 10.0
    fir.kernel.assign(kernel)
    fir.bias.assign(np.full(out_features, 0.5, dtype=np.float32))
    return kernel


def test_fir_keeps_the_sequence_axes():
    # Once flattened into the projection as well: every output mixed every
    # step of the sequence, and the seq axis was gone from the result.
    x = Input("fir_seq_x", dim=2, seq=4)
    fir = Fir(out_features=3, name="fir_seq")([x.sw(3)])
    model = Modely("fir_seq_model", inputs=[x], outputs=[Output("fir_seq_out", fir)])
    model.build()
    assert fir.shape.dimensions == ((3,), 1, (4,))

    kernel = _fir_kernel(fir, 2 * 3, 3)
    values = np.random.default_rng(1).normal(size=(2, 2, 3, 4)).astype(np.float32)
    result = to_numpy(model({"fir_seq_x": values})["fir_seq_out"])

    assert result.shape == (2, 3, 1, 4)
    for step in range(4):
        expected = values[..., step].reshape(2, -1) @ kernel + 0.5
        np.testing.assert_allclose(result[:, :, 0, step], expected, rtol=1e-5)


def test_fir_without_sequence_axes_projects_the_whole_window():
    x = Input("fir_plain_x", dim=2)
    fir = Fir(out_features=3, name="fir_plain")([x.sw(3)])
    model = Modely(
        "fir_plain_model", inputs=[x], outputs=[Output("fir_plain_out", fir)]
    )
    model.build()

    kernel = _fir_kernel(fir, 2 * 3, 3)
    values = np.random.default_rng(2).normal(size=(2, 2, 3)).astype(np.float32)
    result = to_numpy(model({"fir_plain_x": values})["fir_plain_out"])

    assert result.shape == (2, 3, 1)
    np.testing.assert_allclose(
        result[:, :, 0], values.reshape(2, -1) @ kernel + 0.5, rtol=1e-5
    )


def test_fir_follows_a_dynamic_sequence_length():
    x = Input("fir_dynamic_x", seq=-1)
    fir = Fir(out_features=2, name="fir_dynamic")([x.sw(2)])
    model = Modely(
        "fir_dynamic_model", inputs=[x], outputs=[Output("fir_dynamic_out", fir)]
    )
    model.build()

    for length in (3, 5):
        values = np.ones((1, 1, 2, length), dtype=np.float32)
        assert to_numpy(model({"fir_dynamic_x": values})["fir_dynamic_out"]).shape == (
            1,
            2,
            1,
            length,
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
