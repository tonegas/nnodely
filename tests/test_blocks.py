from nnodely import (
    Input,
    Output,
    Modely,
    ReLU,
    Linear,
    EquationLearner,
    Sin,
    Cos,
    Fir,
    Select,
    Tanh,
    IntegrateStep,
    Integrate,
    Derivative,
    DerivativeStep,
    Loop,
    ModelyLoop,
    DataLoader,
    Parameter,
)

from nnodely.core.layer import Add, Layer
from nnodely.layers.fir import FirImpl
from nnodely.layers.localmodel import LocalModel
from nnodely.layers.fuzzify import Fuzzify, Gaussian, Rectangular, Triangular
from nnodely.core.layer import Identity
import keras
import numpy as np
from conftest import requires_onnx_export, to_numpy

import pytest
from nnodely import TimeRange, TimeSelect


def test_fuzzify(tmp_path):
    from nnodely.layers.trigonometric import Sin, Cos
    from nnodely.layers.fuzzify import Rectangular, Triangular, Gaussian
    import keras

    def gaussian(distance, sigma=1.0 / 6.0):
        # With centers 0, 0.5 and 1 every sigma is (0.5 + 0.5) / 6: three
        # sigmas reach the next centers.
        return np.exp(-0.5 * (distance / sigma) ** 2)

    class CustomFunction(keras.layers.Layer):
        def call(self, x):
            return keras.ops.power(x, 2)

    class CustomFunctionMultiInputs(keras.layers.Layer):
        def call(self, x, y):
            return keras.ops.power(x, 2) + y

    x = Input("x", dim=1, sw=1)
    fuzzy_rectangular = Fuzzify(centers=[0.0, 0.5, 1.0], function=Rectangular)
    fuzzy_triangular = Fuzzify(centers=[0.0, 0.5, 1.0], function=Triangular)
    # The functions themselves, not normalized.
    fuzzy_gaussian = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=Gaussian, normalize=False
    )
    fuzzy_sin = Fuzzify(centers=[0.0, 0.5, 1.0], function=Sin, normalize=False)
    fuzzy_cos = Fuzzify(centers=[0.0, 0.5, 1.0], function=Cos, normalize=False)
    fuzzy_multiple = Fuzzify(
        centers=[0.0, 0.5, 1.0],
        function=[Rectangular, Triangular, Gaussian],
        normalize=False,
    )
    fuzzy_custom = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=CustomFunction, normalize=False
    )
    # Bounded by the outer centers, x beyond them is in the outer set; or by
    # 0.2 and 0.9, where two triangles meet.
    bounded = {
        function: Fuzzify(
            centers=[0.0, 0.5, 1.0], function=function, bounds=True, normalize=True
        )
        for function in (Rectangular, Triangular, Gaussian)
    }
    fuzzy_inner = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=Triangular, bounds=(0.2, 0.9)
    )
    # A custom function reads x - center alone, as a class or an instance.
    for function in (CustomFunctionMultiInputs, CustomFunctionMultiInputs()):
        with pytest.raises(TypeError, match="reads one argument"):
            Fuzzify(centers=[0.0, 0.5, 1.0], function=function)
    x_out_rectangular = Output("x_pred_rectangular", fuzzy_rectangular([x]))
    x_out_triangular = Output("x_pred_triangular", fuzzy_triangular([x]))
    x_out_gaussian = Output("x_pred_gaussian", fuzzy_gaussian([x]))
    x_out_sin = Output("x_pred_sin", fuzzy_sin([x]))
    x_out_cos = Output("x_pred_cos", fuzzy_cos([x]))
    x_out_multiple = Output("x_pred_multiple", fuzzy_multiple([x]))
    x_out_custom = Output("x_pred_custom", fuzzy_custom([x]))
    x_out_bounded = [
        Output(f"x_bounded_{function.__name__}", fuzzy([x]))
        for function, fuzzy in bounded.items()
    ]
    x_out_inner = Output("x_pred_inner", fuzzy_inner([x]))
    model1 = Modely(
        "model1",
        inputs=[x],
        outputs=[
            x_out_rectangular,
            x_out_triangular,
            x_out_gaussian,
            x_out_sin,
            x_out_cos,
            x_out_multiple,
            x_out_custom,
            *x_out_bounded,
            x_out_inner,
        ],
    )
    model1.build()

    # ------- Model inference -------
    dummy_input_x = np.array([[[-7]], [[0.5]], [[0.8]], [[5.0]]], dtype=np.float32)
    dummy_input_y = np.array([[[1.0]], [[0.5]], [[0.0]], [[-3.0]]], dtype=np.float32)
    result1 = model1({"x": dummy_input_x, "y": dummy_input_y})

    assert "x_pred_rectangular" in result1
    assert result1["x_pred_rectangular"].shape == (4, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_rectangular"]),
        np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0]],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "x_pred_triangular" in result1
    assert result1["x_pred_triangular"].shape == (4, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_triangular"]),
        np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.4, 0.6]],
                [[0.0, 0.0, 0.0]],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "x_pred_gaussian" in result1
    assert result1["x_pred_gaussian"].shape == (4, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_gaussian"]),
        np.array(
            [
                [[gaussian(x - c) for c in (0.0, 0.5, 1.0)]]
                for x in (-7.0, 0.5, 0.8, 5.0)
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "x_pred_sin" in result1
    assert result1["x_pred_sin"].shape == (4, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_sin"]),
        np.array(
            [
                [[np.sin(-7.0), np.sin(-7.0 - 0.5), np.sin(-7.0 - 1.0)]],
                [[np.sin(0.5), np.sin(0.5 - 0.5), np.sin(0.5 - 1.0)]],
                [[np.sin(0.8), np.sin(0.8 - 0.5), np.sin(0.8 - 1.0)]],
                [[np.sin(5.0), np.sin(5.0 - 0.5), np.sin(5.0 - 1.0)]],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "x_pred_cos" in result1
    assert result1["x_pred_cos"].shape == (4, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_cos"]),
        np.array(
            [
                [[np.cos(-7.0), np.cos(-7.0 - 0.5), np.cos(-7.0 - 1.0)]],
                [[np.cos(0.5), np.cos(0.5 - 0.5), np.cos(0.5 - 1.0)]],
                [[np.cos(0.8), np.cos(0.8 - 0.5), np.cos(0.8 - 1.0)]],
                [[np.cos(5.0), np.cos(5.0 - 0.5), np.cos(5.0 - 1.0)]],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "x_pred_multiple" in result1
    assert result1["x_pred_multiple"].shape == (4, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_multiple"]),
        np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 1.0, gaussian(-0.5)]],
                [[0.0, 0.4, gaussian(-0.2)]],
                [[0.0, 0.0, 0.0]],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "x_pred_custom" in result1
    assert result1["x_pred_custom"].shape == (4, 1, 3)
    np.testing.assert_allclose(
        to_numpy(result1["x_pred_custom"]),
        np.array(
            [
                [[np.power(-7.0, 2), np.power(-7.5, 2), np.power(-8.0, 2)]],
                [[np.power(0.5, 2), np.power(0.0, 2), np.power(-0.5, 2)]],
                [[np.power(0.8, 2), np.power(0.3, 2), np.power(-0.2, 2)]],
                [[np.power(5.0, 2), np.power(4.5, 2), np.power(4.0, 2)]],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )

    # Bounded, -7 reads 0 and 5 reads 1: no sample is in no set, and the
    # Gaussians, normalized, add up to one.
    bells = np.array(
        [[gaussian(x - c) for c in (0.0, 0.5, 1.0)] for x in (0.0, 0.5, 0.8, 1.0)]
    )
    expected = {
        "x_bounded_Rectangular": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        "x_bounded_Triangular": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.4, 0.6], [0.0, 0.0, 1.0]],
        "x_bounded_Gaussian": bells / bells.sum(axis=1, keepdims=True),
        # -7 reads 0.2 and 5 reads 0.9.
        "x_pred_inner": [[0.6, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.4, 0.6], [0.0, 0.2, 0.8]],
    }  # fmt: skip
    for name, values in expected.items():
        assert result1[name].shape == (4, 1, 3)
        np.testing.assert_allclose(
            to_numpy(result1[name])[:, 0], values, rtol=1e-5, atol=1e-6
        )

    model1.plot(out_dir=tmp_path, filename="fuzzify")
    model1.export_html(out_dir=tmp_path, filename="fuzzify")


def test_fuzzify_multi_temporal():
    from nnodely.layers.trigonometric import Sin, Cos
    from nnodely.layers.fuzzify import Rectangular, Triangular, Gaussian
    import keras

    def gaussian(distance, sigma=1.0 / 6.0):
        # With centers 0, 0.5 and 1 every sigma is (0.5 + 0.5) / 6: three
        # sigmas reach the next centers.
        return np.exp(-0.5 * (distance / sigma) ** 2)

    class CustomFunction(keras.layers.Layer):
        def call(self, x):
            return keras.ops.power(x, 2)

    x = Input("x", dim=1, sw=2)
    x_window = x  # temporal window of length 2
    fuzzy_rectangular = Fuzzify(centers=[0.0, 0.5, 1.0], function=Rectangular)
    fuzzy_triangular = Fuzzify(centers=[0.0, 0.5, 1.0], function=Triangular)
    # The functions themselves, not normalized.
    fuzzy_gaussian = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=Gaussian, normalize=False
    )
    fuzzy_sin = Fuzzify(centers=[0.0, 0.5, 1.0], function=Sin, normalize=False)
    fuzzy_cos = Fuzzify(centers=[0.0, 0.5, 1.0], function=Cos, normalize=False)
    fuzzy_multiple = Fuzzify(
        centers=[0.0, 0.5, 1.0],
        function=[Rectangular, Triangular, Gaussian],
        normalize=False,
    )
    fuzzy_custom = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=CustomFunction, normalize=False
    )
    # Bounded by the outer centers, and normalized at each step on its own.
    fuzzy_bounded_triangular = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=Triangular, bounds=True, normalize=True
    )
    fuzzy_bounded_gaussian = Fuzzify(
        centers=[0.0, 0.5, 1.0], function=Gaussian, bounds=True, normalize=True
    )
    x_out_rectangular = Output("x_pred_rectangular", fuzzy_rectangular([x_window]))
    x_out_triangular = Output("x_pred_triangular", fuzzy_triangular([x_window]))
    x_out_gaussian = Output("x_pred_gaussian", fuzzy_gaussian([x_window]))
    x_out_sin = Output("x_pred_sin", fuzzy_sin([x_window]))
    x_out_cos = Output("x_pred_cos", fuzzy_cos([x_window]))
    x_out_multiple = Output("x_pred_multiple", fuzzy_multiple([x_window]))
    x_out_custom = Output("x_pred_custom", fuzzy_custom([x_window]))
    x_out_bounded_triangular = Output(
        "x_bounded_triangular", fuzzy_bounded_triangular([x_window])
    )
    x_out_bounded_gaussian = Output(
        "x_bounded_gaussian", fuzzy_bounded_gaussian([x_window])
    )
    model1 = Modely(
        "model1",
        inputs=[x],
        outputs=[
            x_out_rectangular,
            x_out_triangular,
            x_out_gaussian,
            x_out_sin,
            x_out_cos,
            x_out_multiple,
            x_out_custom,
            x_out_bounded_triangular,
            x_out_bounded_gaussian,
        ],
    )
    model1.build()

    # ------- Model inference -------
    # Two samples of a two-step window, oldest first: the first far from every
    # center, the second in a different set at each step.
    samples_x = ((-7.0, -6.0), (0.5, 0.8))
    samples_y = ((1.0, 3.0), (0.5, -1.0))
    centers = (0.0, 0.5, 1.0)
    dummy_input_x = np.array(samples_x, dtype=np.float32).reshape(2, 2, 1)
    dummy_input_y = np.array(samples_y, dtype=np.float32).reshape(2, 2, 1)
    result1 = model1({"x": dummy_input_x, "y": dummy_input_y})

    def rectangular(distance):
        # Each rectangle reaches halfway to the next center: 0.25 either side.
        return float(abs(distance) < 0.25)

    def triangular(distance):
        # Each triangle reaches the next centers, 0.5 away.
        return max(0.0, 1.0 - abs(distance) / 0.5)

    def matrix(function, bounds=(-np.inf, np.inf)):
        """function(x_t - c, y_t) on every center (rows) at every step
        (columns), for each sample: (batch, centers, time); x_t held within
        the bounds."""
        return np.array(
            [
                [
                    [
                        function(np.clip(x_t, *bounds) - c, y_t)
                        for x_t, y_t in zip(xs, ys)
                    ]
                    for c in centers
                ]
                for xs, ys in zip(samples_x, samples_y)
            ]
        )

    expected = {
        "x_pred_rectangular": matrix(lambda d, _: rectangular(d)),
        "x_pred_triangular": matrix(lambda d, _: triangular(d)),
        "x_pred_gaussian": matrix(lambda d, _: gaussian(d)),
        "x_pred_sin": matrix(lambda d, _: np.sin(d)),
        "x_pred_cos": matrix(lambda d, _: np.cos(d)),
        "x_pred_custom": matrix(lambda d, _: d**2),
        "x_bounded_triangular": matrix(lambda d, _: triangular(d), (0.0, 1.0)),
    }
    # The Gaussians of each step, x = -7 and -6 read as 0, add up to one.
    bells = matrix(lambda d, _: gaussian(d), (0.0, 1.0))
    expected["x_bounded_gaussian"] = bells / bells.sum(axis=1, keepdims=True)
    # One function per center: Rectangular on 0, Triangular on 0.5, Gaussian on 1.
    expected["x_pred_multiple"] = np.stack(
        [
            expected["x_pred_rectangular"][:, 0],
            expected["x_pred_triangular"][:, 1],
            expected["x_pred_gaussian"][:, 2],
        ],
        axis=1,
    )
    # The second sample at a glance: x = 0.5 is the middle set, x = 0.8 the
    # last one's rectangle and 0.4 / 0.6 of the middle and last triangles.
    np.testing.assert_allclose(
        expected["x_pred_rectangular"][1], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    np.testing.assert_allclose(
        expected["x_pred_triangular"][1],
        [[0.0, 0.0], [1.0, 0.4], [0.0, 0.6]],
        atol=1e-12,
    )
    # The first, bounded: in the first set at both steps.
    np.testing.assert_allclose(
        expected["x_bounded_triangular"][0], [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )

    for name, values in expected.items():
        assert name in result1
        # A matrix per sample: 2 time steps x 3 centers.
        assert result1[name].shape == (2, 2, 3)
        np.testing.assert_allclose(
            to_numpy(result1[name]), values.transpose(0, 2, 1), rtol=1e-5, atol=1e-5
        )


def test_fuzzify_with_trainable_centers():
    x = Input("x", dim=1, sw=1)

    class CustomFunction(keras.layers.Layer):
        # A trainable custom function: k is a Keras weight, 2 to begin with.
        # (An nnodely Parameter is a node of the graph, not made in a call.)
        def build(self, input_shape):
            self.k = self.add_weight(
                shape=(), initializer=keras.initializers.Constant(2.0), name="k"
            )

        def call(self, x):
            return self.k * x

    fir_input = Input("fir_input", dim=1, sw=3)
    # 4 centers for each sample: 0, 0.5, 1 and 1.5 times its last fir_input.
    fir = Fir(
        features=4,
        kernel=Parameter(
            "fir_kernel", value=[[0.0] * 4, [0.0] * 4, [0.0, 0.5, 1.0, 1.5]]
        ),
        bias=False,
    )(fir_input)
    # 1 center for each sample: minus its last fir_input.
    fir2 = Fir(
        features=1,
        kernel=Parameter("fir2_kernel", value=[[0.0], [0.0], [-1.0]]),
        bias=False,
    )(fir_input)
    # Bounded by the lowest and the highest of each sample's centers.
    fuzzy_fir = Fuzzify(centers=fir, function=Triangular, bounds=True)
    ## Fuzzy can take a Parameter that is monodimensional and compatible with the centers
    fuzzy_param = Fuzzify(
        centers=Parameter(name="param", value=[0.0, 0.5, 1.0, 1.5]),
        function=Rectangular,
    )

    a, b, c = (
        Parameter(name="a", value=[0.0]),
        Parameter(name="b", value=[0.5]),
        Parameter(name="c", value=[1.0]),
    )
    fuzzy_multiple_param = Fuzzify(centers=[a, b, c], function=Sin, normalize=False)
    fuzzy_fir2 = Fuzzify(
        centers=[fir2, a], function=Triangular, normalize=False
    )  # this has two centers because fir2 exit with 1 and parameter has one

    fuzzy_trainable_function = Fuzzify(
        centers=[0.0, 0.5, 1.0, 1.5], function=CustomFunction, normalize=False
    )
    fuzzy_trainable_function_and_centers = Fuzzify(
        centers=Parameter(name="param2", value=[0.0, 0.5, 1.0, 1.5]),
        function=CustomFunction,
        normalize=False,
    )
    x_out_fir = Output("out_fir", fuzzy_fir(x))
    x_out_param = Output("out_param", fuzzy_param(x))
    x_out_fir2 = Output("out_fir2", fuzzy_fir2(x))
    x_out_multiple = Output("out_multiple", fuzzy_multiple_param(x))
    x_out_custom = Output("out_custom", fuzzy_trainable_function(x))
    x_out_multiple_custom = Output(
        "out_multiple_custom", fuzzy_trainable_function_and_centers(x)
    )

    model1 = Modely(
        "model1",
        inputs=[x, fir_input],
        outputs=[
            x_out_fir,
            x_out_param,
            x_out_fir2,
            x_out_multiple,
            x_out_custom,
            x_out_multiple_custom,
        ],
    )
    model1.build()
    # The centers are Parameters of the model, trained with it.
    assert {"param", "param2", "a", "b", "c"} <= {
        parameter.name for parameter in model1.parameters
    }

    # ------- Model inference -------
    dummy_input_x = np.array([[[-7]], [[0.5]], [[1.2]]], dtype=np.float32)
    # The Fir centers of the samples: 0, 0.5, 1, 1.5 twice, then 0, 1, 2, 3;
    # and the fir2 ones -1, -1, -2.
    dummy_input_fir = np.array(
        [[[0.0], [0.0], [1.0]], [[0.0], [0.0], [1.0]], [[0.0], [0.0], [2.0]]],
        dtype=np.float32,
    )
    result1 = model1({"x": dummy_input_x, "fir_input": dummy_input_fir})

    xs = np.array([-7.0, 0.5, 1.2]).reshape(3, 1, 1)
    centers = np.array([0.0, 0.5, 1.0, 1.5]).reshape(1, 1, 4)
    expected = {
        # Triangles on each sample's centers: -7 reads 0, its lowest one,
        # and 1.2 is 0.8 of the way back from 2 to 1 on the last one.
        "out_fir": [[[1.0, 0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0, 0.0]],
                    [[0.0, 0.8, 0.2, 0.0]]],
        # Rectangles reaching halfway to the next center, 0.25 either side.
        "out_param": [[[0.0, 0.0, 0.0, 0.0]],
                      [[0.0, 1.0, 0.0, 0.0]],
                      [[0.0, 0.0, 1.0, 0.0]]],
        # Two triangles on -1 and 0, then -2 and 0, reaching each other.
        "out_fir2": [[[0.0, 0.0]],
                     [[0.0, 0.5]],
                     [[0.0, 0.4]]],
        "out_multiple": np.sin(xs - np.array([0.0, 0.5, 1.0]).reshape(1, 1, 3)),
        # k (x - c) with k = 2, on fixed and on trainable centers alike.
        "out_custom": 2.0 * (xs - centers),
        "out_multiple_custom": 2.0 * (xs - centers),
    }  # fmt: skip
    for name, values in expected.items():
        assert result1[name].shape == np.shape(values)
        np.testing.assert_allclose(
            to_numpy(result1[name]), values, rtol=1e-5, atol=1e-5
        )


def test_fuzzify_on_moving_centers_matches_fixed_centers():
    # Untrained, centers in a Parameter give what the same numbers give - uneven
    # ones too, on a time window and a dynamic sequence - and so do a lone
    # center and a list of numbers and scalar Parameters, bounded or not.
    x = Input("same_x", seq=-1, sw=2)
    values = [0.0, 0.5, 2.0]
    pairs = {
        "triangular": (
            Fuzzify(values, function=Triangular),
            Fuzzify(Parameter("same_t", value=values), function=Triangular),
        ),
        "rectangular": (
            Fuzzify(values, function=Rectangular),
            Fuzzify(Parameter("same_r", value=values), function=Rectangular),
        ),
        "gaussian": (
            Fuzzify(values, function=Gaussian),
            Fuzzify(Parameter("same_g", value=values), function=Gaussian),
        ),
        "mixed": (
            Fuzzify(values, function=[Gaussian, Triangular, Rectangular]),
            Fuzzify(
                Parameter("same_m", value=values),
                function=[Gaussian, Triangular, Rectangular],
            ),
        ),
        "sin": (
            Fuzzify(values, function=Sin),
            Fuzzify(Parameter("same_s", value=values), function=Sin),
        ),
        "lone": (
            Fuzzify([0.5], function=Triangular),
            Fuzzify(Parameter("same_lone", value=[0.5]), function=Triangular),
        ),
        "listed": (
            Fuzzify(values, function=Triangular),
            Fuzzify(
                [0.0, Parameter("same_mid", value=[0.5]), 2.0], function=Triangular
            ),
        ),
        "bounded": (
            Fuzzify(values, function=Gaussian, bounds=True),
            Fuzzify(Parameter("same_b", value=values), function=Gaussian, bounds=True),
        ),
        "inner": (
            Fuzzify(values, function=Triangular, bounds=(0.2, 1.5)),
            Fuzzify(
                Parameter("same_i", value=values),
                function=Triangular,
                bounds=(0.2, 1.5),
            ),
        ),
    }
    outputs = []
    # One window, with the Input's sequence, read by every pair.
    window = x
    for key, (fixed, moving) in pairs.items():
        fixed_out, moving_out = fixed([window]), moving([window])
        assert moving_out.shape.dimensions == fixed_out.shape.dimensions
        outputs += [Output(f"same_{key}_fixed", fixed_out)]
        outputs += [Output(f"same_{key}_moving", moving_out)]
    model = Modely("same_model", inputs=[x], outputs=outputs).build()

    for length in (3, 5):
        samples = np.random.default_rng(length).uniform(-0.5, 2.5, (4, 1, 2, length))
        samples = samples.transpose(0, 3, 2, 1)  # (batch, seq, time, dim)
        result = model({"same_x": samples.astype(np.float32)})
        for key in pairs:
            np.testing.assert_allclose(
                to_numpy(result[f"same_{key}_moving"]),
                to_numpy(result[f"same_{key}_fixed"]),
                rtol=1e-5,
                atol=1e-6,
            )


def test_fuzzify_rejects_centers_it_cannot_use():
    x = Input("bad_centers_x", seq=3, sw=1)
    u = Input("bad_centers_u", sw=3)
    with pytest.raises(ValueError, match="vector of shape"):
        Fuzzify(
            Parameter("bad_matrix", value=[[0.0, 1.0], [2.0, 3.0]]), function=Triangular
        )
    with pytest.raises(ValueError, match="vector of shape"):
        Fuzzify(u, function=Triangular)  # a time window, not one sample
    with pytest.raises(ValueError, match="center 1 has dim"):
        Fuzzify([0.0, Parameter("bad_pair", value=[0.5, 1.0])], function=Triangular)
    with pytest.raises(ValueError, match="2 functions for 3 centers"):
        Fuzzify(Parameter("bad_count", value=[0.0, 0.5, 1.0]), function=[Sin, Cos])
    # Centers with a sequence axis follow the signal's.
    s = Input("bad_centers_s", seq=2, sw=1)
    with pytest.raises(ValueError, match="with a sequence"):
        Fuzzify(Fir(features=2)(s), function=Triangular)([x])


@pytest.mark.slow
def test_fuzzify_centers_follow_the_gradient():
    # x - c on two trainable centers, against x - c on 1 and 3: the mean
    # squared error is ((c0 - 1)^2 + (c1 - 3)^2) / 2 whatever x, so an SGD
    # step at lr 0.5 moves every center halfway to its target.
    x = Input("step_x", sw=1)
    centers = Parameter("step_centers", value=[0.0, 1.0])
    shifted = Output(
        "step_out", Fuzzify(centers, function=Identity, normalize=False)([x])
    )
    model = Modely("step_model", inputs=[x], outputs=[shifted])
    target = Fuzzify([1.0, 3.0], function=Identity, normalize=False)([x])
    model.minimize("step_fit", shifted, target)
    model.build()
    data = DataLoader(model, source={"step_x": np.array([[0.3], [-1.2]], np.float32)})

    for expected in ([0.5, 2.0], [0.75, 2.5]):
        model.train(data, epochs=1, batch_size=2, optimizer="sgd", lr=0.5, printer=None)
        np.testing.assert_allclose(np.ravel(centers.value_numpy), expected, rtol=1e-6)  # type: ignore
    # Frozen, they stay where they are.
    centers.freeze()
    model.train(data, epochs=1, batch_size=2, optimizer="sgd", lr=0.5, printer=None)
    np.testing.assert_allclose(np.ravel(centers.value_numpy), [0.75, 2.5], rtol=1e-6)  # type: ignore

    # Centers computed by a Fir, u times its kernel with u = 2, against 2 and
    # 6: d/dk of ((2 k0 - 2)^2 + (2 k1 - 6)^2) / 2 is (4 k0 - 4, 4 k1 - 12),
    # so an SGD step at lr 0.125 moves the kernel from (0, 1) halfway to (1, 3).
    y, u = Input("fir_step_x", sw=1), Input("fir_step_u", sw=1)
    kernel = Parameter("fir_step_kernel", value=[[0.0, 1.0]])
    computed = Fir(features=2, kernel=kernel, bias=False)(u)
    out = Output(
        "fir_step_out", Fuzzify(computed, function=Identity, normalize=False)([y])
    )
    model = Modely("fir_step", inputs=[y, u], outputs=[out])
    target = Fuzzify([2.0, 6.0], function=Identity, normalize=False)([y])
    model.minimize("fir_step_fit", out, target)
    model.build()
    data = DataLoader(
        model,
        source={
            "fir_step_x": np.array([[0.3]], np.float32),
            "fir_step_u": np.array([[2.0]], np.float32),
        },
    )
    model.train(data, epochs=1, batch_size=1, optimizer="sgd", lr=0.125, printer=None)
    np.testing.assert_allclose(np.ravel(kernel.value_numpy), [0.5, 2.0], rtol=1e-6)  # type: ignore


@pytest.mark.slow
def test_fuzzify_learns_where_its_triangles_sit():
    # Triangles on 0, 1 and 2 fitted to triangles on 0, 1.4 and 2: the
    # middle center moves to 1.4, the outer ones stay where they are.
    x = Input("learn_x", sw=1)
    centers = Parameter("learn_centers", value=[0.0, 1.0, 2.0])
    learned = Output("learn_out", Fuzzify(centers, function=Triangular)([x]))
    model = Modely("learn_model", inputs=[x], outputs=[learned])
    model.minimize(
        "learn_fit", learned, Fuzzify([0.0, 1.4, 2.0], function=Triangular)([x])
    )
    model.build()
    samples = np.linspace(-0.5, 2.5, 61, dtype=np.float32).reshape(-1, 1)
    data = DataLoader(model, source={"learn_x": samples})

    model.train(
        data, epochs=100, batch_size=61, optimizer="adam", lr=0.05, printer=None
    )
    np.testing.assert_allclose(
        np.ravel(centers.value_numpy),  # type: ignore
        [0.0, 1.4, 2.0],
        atol=0.01,
    )


# ---------------------------------------------------------------------------
# The membership functions, layers of their own, and the Fuzzify edge cases
# ---------------------------------------------------------------------------


def test_membership_functions_are_layers_of_their_own():
    x = Input("member_x", sw=5)
    window = x
    # Centered at 0: each reads the window shifted by its center.
    layers = {
        "triangle": (Triangular(radius=0.5), 0.5),
        "leaning": (Triangular(radius=(1.0, 2.0)), 1.0),
        "rectangle": (Rectangular(radius=0.25), 0.5),
        "lopsided": (Rectangular(radius=(0.25, 1.0)), 0.5),
        "bell": (Gaussian(sigma=0.25), 0.5),
    }
    outputs = {key: layer([window - c]) for key, (layer, c) in layers.items()}
    model = Modely(
        "member_model",
        inputs=[x],
        outputs=[Output(f"member_{key}", node) for key, node in outputs.items()],
    ).build()
    assert all(node.shape.dimensions == ((), 5, (1,)) for node in outputs.values())

    samples = np.array([0.0, 0.25, 0.5, 1.4, 2.0], dtype=np.float32)
    result = model({"member_x": samples.reshape(1, 5, 1)})
    expected = {
        "triangle": [0.0, 0.5, 1.0, 0.0, 0.0],
        # Up from 0 to 1 at 1, down to 0 at 3.
        "leaning": [0.0, 0.25, 0.5, 0.8, 0.5],
        # 1 on [0.25, 0.75), then [0.25, 1.5): the left edge is in.
        "rectangle": [0.0, 1.0, 1.0, 0.0, 0.0],
        "lopsided": [0.0, 1.0, 1.0, 1.0, 0.0],
        "bell": np.exp(-0.5 * ((samples - 0.5) / 0.25) ** 2),
    }
    for key, values in expected.items():
        np.testing.assert_allclose(
            to_numpy(result[f"member_{key}"]).ravel(), values, rtol=1e-5, atol=1e-6
        )


def test_membership_functions_keep_the_shape_of_any_stream():
    x = Input("member_shape_x", dim=(2, 3), seq=-1, sw=2)
    bell = Gaussian()([x])
    for node in (Triangular()([x]), Rectangular()([x]), bell):
        assert node.shape.dimensions == ((None,), 2, (2, 3))
    model = Modely(
        "member_shape", inputs=[x], outputs=[Output("member_shape_bell", bell)]
    ).build()

    for length in (3, 5):
        values = np.random.default_rng(length).normal(size=(1, 2, 3, 2, length))
        values = values.transpose(0, 4, 3, 1, 2)  # (batch, seq, time, *dim)
        result = to_numpy(model({"member_shape_x": values})["member_shape_bell"])
        np.testing.assert_allclose(result, np.exp(-0.5 * values**2), rtol=1e-5)


def test_membership_functions_reject_widths_they_cannot_have():
    with pytest.raises(ValueError, match="must not be negative"):
        Triangular(radius=-1.0)
    with pytest.raises(ValueError, match="number or a \\(left, right\\) pair"):
        Rectangular(radius=(1.0, 2.0, 3.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be positive"):
        Gaussian(sigma=0.0)


def test_fuzzify_with_uneven_centers_leans_its_triangles_and_adds_up_to_one():
    x = Input("uneven_x", sw=1)
    fuzzy = Fuzzify([0.0, 1.0, 3.0], function=Triangular)([x])
    model = Modely(
        "uneven_model", inputs=[x], outputs=[Output("uneven_out", fuzzy)]
    ).build()

    values = np.linspace(0.0, 3.0, 13, dtype=np.float32)
    result = to_numpy(model({"uneven_x": values.reshape(-1, 1, 1)})["uneven_out"])
    result = result[:, 0]
    np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1e-6)
    # x = 2, halfway from 1 to 3: the triangle of 1 reaches 3, that of 3 reaches 1.
    np.testing.assert_allclose(result[8], [0.0, 0.5, 0.5], atol=1e-6)


def test_fuzzify_sorts_its_centers_with_their_functions():
    x = Input("sorted_x", sw=1)
    fuzzy = Fuzzify([1.0, 0.0], function=[Gaussian, Rectangular])([x])
    model = Modely(
        "sorted_model", inputs=[x], outputs=[Output("sorted_out", fuzzy)]
    ).build()

    # Sorted, center 0 keeps its Rectangular and center 1 its Gaussian, whose
    # sigma is (1 + 1) / 6: at x = 0, three sigmas away.
    result = to_numpy(model({"sorted_x": np.zeros((1, 1, 1))})["sorted_out"])
    np.testing.assert_allclose(result.ravel(), [1.0, np.exp(-4.5)], rtol=1e-5)


def test_fuzzify_with_a_single_center():
    x = Input("single_x", sw=1)
    fuzzy = Fuzzify([0.0], function=Triangular, normalize=True, name="single_fuzzy")(
        [x]
    )
    raw = Fuzzify([0.0], function=Triangular, normalize=False, name="single_raw")([x])
    model = Modely(
        "single_model",
        inputs=[x],
        outputs=[Output("single_out", fuzzy), Output("single_raw_out", raw)],
    ).build()
    assert (fuzzy.name, raw.name) == ("single_fuzzy", "single_raw")
    assert fuzzy.shape.dimensions == raw.shape.dimensions == ((), 1, (1,))

    # A lone triangle reaches 1 on either side; normalized, a sample is in
    # its set or in none.
    values = np.array([-0.5, 0.0, 0.5, 1.5], dtype=np.float32).reshape(-1, 1, 1)
    result = model({"single_x": values})
    np.testing.assert_allclose(
        to_numpy(result["single_raw_out"]).ravel(), [0.5, 1.0, 0.5, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        to_numpy(result["single_out"]).ravel(), [1.0, 1.0, 1.0, 0.0], atol=1e-6
    )


def test_fuzzify_with_a_layer_instance_or_a_weighted_layer_class():
    x = Input("instance_x", sw=1)
    # An instance is shared by every center and reads x - center.
    bell = Fuzzify([0.0, 1.0], function=Gaussian(sigma=0.5), normalize=False)([x])
    # A class is made anew for each center: two Linear, with their own weights,
    # shared again by a second call of the same Fuzzify.
    linear = Fuzzify([0.0, 1.0], function=Linear, name="linear_fuzzy")
    first, second = linear([x]), linear([x])
    model = Modely(
        "instance_model",
        inputs=[x],
        outputs=[
            Output("instance_bell", bell),
            Output("instance_first", first),
            Output("instance_second", second),
        ],
    ).build()

    assert [parameter.name for parameter in model.parameters] == [
        "linear_fuzzy_Linear_0_kernel",
        "linear_fuzzy_Linear_0_bias",
        "linear_fuzzy_Linear_1_kernel",
        "linear_fuzzy_Linear_1_bias",
    ]
    result = model({"instance_x": np.full((1, 1, 1), 0.5, dtype=np.float32)})
    np.testing.assert_allclose(
        to_numpy(result["instance_bell"]).ravel(),
        np.exp(-0.5 * (np.array([0.5, -0.5]) / 0.5) ** 2),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["instance_first"]), to_numpy(result["instance_second"])
    )


def test_fuzzify_follows_a_time_window_and_a_dynamic_sequence():
    x = Input("dynamic_fuzzy_x", seq=-1, sw=4)
    placed = Fuzzify([0.0, 1.0], function=Triangular, bounds=True)([x])
    waves = Fuzzify([0.0, 1.0], function=Sin, normalize=False)([x])
    assert placed.shape.dimensions == ((None,), 4, (2,))
    assert waves.shape.dimensions == ((None,), 4, (2,))
    model = Modely(
        "dynamic_fuzzy",
        inputs=[x],
        outputs=[Output("dynamic_placed", placed), Output("dynamic_waves", waves)],
    ).build()

    for length in (3, 5):
        values = np.random.default_rng(length).uniform(-1, 2, size=(1, 1, 4, length))
        values = values.transpose(0, 3, 2, 1)  # (batch, seq, time, dim)
        result = model({"dynamic_fuzzy_x": values.astype(np.float32)})
        np.testing.assert_allclose(
            to_numpy(result["dynamic_waves"]),
            np.concatenate([np.sin(values - 0.0), np.sin(values - 1.0)], axis=-1),
            rtol=1e-5,
            atol=1e-6,
        )
        # Bounded and normalized: every sample at every step adds up to one.
        placed_values = to_numpy(result["dynamic_placed"])
        assert placed_values.shape == (1, length, 4, 2)
        np.testing.assert_allclose(placed_values.sum(axis=-1), 1.0, rtol=1e-6)


def test_fuzzify_takes_a_membership_name_for_its_class():
    x = Input("named_x", sw=1)
    named = Fuzzify([0.0, 0.5, 1.0], function=["triangular", "Rectangular", "GAUSSIAN"])
    classes = Fuzzify([0.0, 0.5, 1.0], function=[Triangular, Rectangular, Gaussian])
    model = Modely(
        "named_model",
        inputs=[x],
        outputs=[
            Output("named_out", named([x])),
            Output("classes_out", classes([x])),
        ],
    ).build()

    values = np.linspace(-0.5, 1.5, 9, dtype=np.float32).reshape(-1, 1, 1)
    result = model({"named_x": values})
    np.testing.assert_array_equal(
        to_numpy(result["named_out"]), to_numpy(result["classes_out"])
    )


def test_fuzzify_rejects_what_it_cannot_use():
    x = Input("bad_fuzzy_x", sw=1)
    v = Input("bad_fuzzy_v", dim=2, sw=1)
    with pytest.raises(ValueError, match="at least one center"):
        Fuzzify([], function=Triangular)
    with pytest.raises(ValueError, match="2 functions for 3 centers"):
        Fuzzify([0.0, 1.0, 2.0], function=[Triangular, Gaussian])
    with pytest.raises(ValueError, match="Unsupported fuzzify function"):
        Fuzzify([0.0], function="Trapezoidal")
    with pytest.raises(TypeError, match="layer class or instance"):
        Fuzzify([0.0], function=np.sin)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cannot be made without arguments"):
        Fuzzify([0.0, 1.0], function=Select)([x])
    with pytest.raises(ValueError, match="scalar signal"):
        Fuzzify([0.0, 1.0], function=Triangular)([v])
    with pytest.raises(TypeError, match="reads one stream"):
        Fuzzify([0.0, 1.0], function=Triangular)([x, x])
    with pytest.raises(TypeError, match="reads one stream"):
        Fuzzify([0.0, 1.0], function=Triangular)([])
    for bounds in ((1.0, 0.0), (0.0,), (0.0, 1.0, 2.0), ("a", 1.0), (True, 1.0), 0.5):
        with pytest.raises(ValueError, match="bounds is True, False or"):
            Fuzzify([0.0, 1.0], function=Triangular, bounds=bounds)  # type: ignore[arg-type]

    class NeedsArguments(keras.layers.Layer):
        def __init__(self, power):
            super().__init__()
            self.power = power

    with pytest.raises(TypeError, match="cannot be made without arguments"):
        Fuzzify([0.0, 1.0], function=NeedsArguments)([x])

    # A custom function reads x - center alone; arguments with a default,
    # such as training, are not counted.
    class TwoArguments(keras.layers.Layer):
        def call(self, x, y, training=None):
            return x + y

    class OneArgument(keras.layers.Layer):
        def call(self, x, training=None):
            return x

    with pytest.raises(TypeError, match="TwoArguments.call needs 2: x, y"):
        Fuzzify([0.0, 1.0], function=TwoArguments)
    Fuzzify([0.0, 1.0], function=OneArgument)([x])


def test_fuzzify_shares_a_keras_layer_instance_between_its_centers():
    class Scaled(keras.layers.Layer):
        def build(self, input_shape):
            self.gain = self.add_weight(shape=(), initializer="ones", name="gain")

        def call(self, x):
            return self.gain * x

    x = Input("keras_instance_x", sw=1)
    scaled = Scaled()
    fuzzy = Fuzzify([0.0, 1.0, 2.0], function=scaled, normalize=False)([x])
    model = Modely(
        "keras_instance", inputs=[x], outputs=[Output("keras_instance_out", fuzzy)]
    ).build()

    # One gain for every center: (x - c) times the same weight.
    assert model.model is not None and len(model.model.trainable_weights) == 1
    scaled.gain.assign(3.0)
    result = model({"keras_instance_x": np.full((1, 1, 1), 2.0, dtype=np.float32)})
    np.testing.assert_allclose(
        to_numpy(result["keras_instance_out"]).ravel(), [6.0, 3.0, 0.0], rtol=1e-6
    )


def test_fuzzify_is_one_layer():
    # The bounds, every membership and their normalization are one layer,
    # reading the signal and the centers that move.
    x = Input("one_layer_x", sw=3)
    centers = Parameter("one_layer_centers", value=[0.0, 1.0, 2.0])
    middle = Parameter("one_layer_middle", value=[1.0])
    moving = Fuzzify(centers, function=Triangular, bounds=True)([x])
    listed = Fuzzify([0.0, middle, 2.0], function=Gaussian, bounds=(0.0, 2.0))([x])
    model = Modely(
        "one_layer",
        inputs=[x],
        outputs=[
            Output("one_layer_moving", moving),
            Output("one_layer_listed", listed),
        ],
    ).build()

    assert isinstance(moving, Layer) and moving.preds == [x, centers]
    assert listed.preds == [x, middle]
    assert sorted(type(node).__name__ for node in model.order) == [
        "Fuzzify",
        "Fuzzify",
        "Input",
        "Output",
        "Output",
        "Parameter",
        "Parameter",
    ]


def test_placed_memberships_follow_their_centers():
    # Sized by the centers as they are, the sets move with them, their
    # widths too: the centers of a triangle are those of its neighbors.
    x = Input("follow_x", sw=1)
    centers = Parameter("follow_centers", value=[0.0, 1.0, 2.0])
    model = Modely(
        "follow",
        inputs=[x],
        outputs=[
            Output(
                "follow_triangles",
                Fuzzify(centers, function=Triangular, normalize=False)([x]),
            ),
            Output(
                "follow_boxes",
                Fuzzify(centers, function=Rectangular, normalize=False)([x]),
            ),
        ],
    ).build()
    values = {"follow_x": np.array([[[1.0]], [[2.5]]], dtype=np.float32)}

    result = model(values)
    np.testing.assert_allclose(
        to_numpy(result["follow_triangles"])[:, 0], [[0, 1, 0], [0, 0, 0.5]]
    )
    np.testing.assert_allclose(
        to_numpy(result["follow_boxes"])[:, 0], [[0, 1, 0], [0, 0, 0]]
    )
    # The middle center moves to 2: the triangles of 0 and 2 reach each
    # other, 2 away, and the rectangle of 3 starts halfway to 2.
    centers.param.assign(np.array([0.0, 2.0, 3.0], dtype=np.float32))  # type: ignore
    result = model(values)
    np.testing.assert_allclose(
        to_numpy(result["follow_triangles"])[:, 0], [[0.5, 0.5, 0], [0, 0.5, 0.5]]
    )
    np.testing.assert_allclose(
        to_numpy(result["follow_boxes"])[:, 0], [[0, 1, 0], [0, 0, 1]]
    )


def test_membership_widths_take_a_parameter():
    x = Input("width_x", sw=1)
    radius = Parameter("width_radius", value=[1.0, 2.0])  # (left, right)
    sigma = Parameter("width_sigma", value=[0.5])
    leaning = Triangular(radius=radius)([x - 1.0])
    bell = Gaussian(sigma=sigma)([x])
    # An instance in a Fuzzify: the same widths on every center.
    fuzzy = Fuzzify([0.0, 2.0], function=Triangular(radius=radius), normalize=False)
    shifted = fuzzy([x])
    model = Modely(
        "width",
        inputs=[x],
        outputs=[
            Output("width_leaning", leaning),
            Output("width_bell", bell),
            Output("width_fuzzy", shifted),
        ],
    ).build()
    # The Parameters are predecessors, of the Fuzzify too.
    assert leaning.preds[1] is radius and bell.preds[1] is sigma
    assert shifted.preds == [x, radius]

    values = {"width_x": np.array([[[0.5]], [[2.0]]], dtype=np.float32)}
    result = model(values)
    expected = {
        # Up from 0 to 1 at 1, down to 0 at 3.
        "width_leaning": [0.5, 0.5],
        "width_bell": np.exp(-0.5 * (np.array([0.5, 2.0]) / 0.5) ** 2),
        "width_fuzzy": [[0.75, 0.0], [0.0, 1.0]],
    }
    for name, value in expected.items():
        np.testing.assert_allclose(
            to_numpy(result[name]).reshape(np.shape(value)), value, rtol=1e-5
        )
    radius.param.assign(np.array([0.5, 0.5], dtype=np.float32))  # type: ignore
    np.testing.assert_allclose(
        to_numpy(model(values)["width_leaning"]).ravel(), [0.0, 0.0]
    )

    with pytest.raises(ValueError, match="Parameter of dim"):
        Triangular(radius=Parameter("width_bad", value=[1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="Parameter of dim"):
        Gaussian(sigma=radius)  # a pair is no sigma
    with pytest.raises(ValueError, match="Parameter of dim"):
        Rectangular(radius=x)  # nor an input a width


@pytest.mark.slow
def test_membership_widths_train():
    # Triangles of radius 0.5 and a bell of sigma 0.5 fitted to a radius and
    # a sigma of 1.
    x = Input("train_width_x", sw=1)
    radius = Parameter("train_radius", value=[0.5])
    sigma = Parameter("train_sigma", value=[0.5])
    triangles = Output(
        "train_triangles",
        Fuzzify([0.0, 1.0, 2.0], function=Triangular(radius=radius), normalize=False)(
            [x]
        ),
    )
    bell = Output("train_bell", Gaussian(sigma=sigma)([x - 1.0]))
    model = Modely("train_width", inputs=[x], outputs=[triangles, bell])
    target = Fuzzify([0.0, 1.0, 2.0], function=Triangular(radius=1.0), normalize=False)
    model.minimize("train_triangles_fit", triangles, target([x]))
    model.minimize("train_bell_fit", bell, Gaussian(sigma=1.0)([x - 1.0]))
    model.build()
    samples = np.linspace(-1.0, 3.0, 81, dtype=np.float32).reshape(-1, 1)
    data = DataLoader(model, source={"train_width_x": samples})

    model.train(
        data, epochs=200, batch_size=81, optimizer="adam", lr=0.02, printer=None
    )
    np.testing.assert_allclose(np.ravel(radius.value_numpy), [1.0], atol=0.01)  # type: ignore
    np.testing.assert_allclose(np.ravel(sigma.value_numpy), [1.0], atol=0.01)  # type: ignore


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Gradients do not exist")
def test_a_parameter_read_by_rectangles_trains_where_it_is_read_elsewhere():
    # A step has no gradient: centers and a radius read by rectangles alone
    # stay where they are. Read by triangles too, they train there, toward
    # the same targets, and the rectangles follow them.
    def fit(name, shared):
        x = Input(f"{name}_x", sw=1)
        centers = Parameter(f"{name}_centers", value=[0.0, 1.0, 2.0])
        radius = Parameter(f"{name}_radius", value=[0.5])
        boxes = Output(f"{name}_boxes", Fuzzify(centers, function=Rectangular)([x]))
        narrow = Output(f"{name}_narrow", Rectangular(radius=radius)([x - 1.0]))
        model = Modely(name, inputs=[x], outputs=[boxes, narrow])
        targets = Fuzzify([0.0, 1.4, 2.0], function=Rectangular)([x])
        model.minimize(f"{name}_boxes_fit", boxes, targets)
        model.minimize(f"{name}_narrow_fit", narrow, Rectangular(1.0)([x - 1.0]))
        # Something else to train, so that every backend trains.
        gain = Parameter(f"{name}_gain", value=[0.5])
        model.minimize(f"{name}_gain_fit", Output(f"{name}_gain", gain * x), 2.0 * x)
        if shared:
            triangles = Fuzzify(centers, function=Triangular)([x])
            targets = Fuzzify([0.0, 1.4, 2.0], function=Triangular)([x])
            model.minimize(f"{name}_triangles_fit", triangles, targets)
            wide = Triangular(radius=radius)([x - 1.0])
            model.minimize(f"{name}_wide_fit", wide, Triangular(1.0)([x - 1.0]))
        model.build()
        samples = np.linspace(-0.5, 2.5, 31, dtype=np.float32).reshape(-1, 1)
        data = DataLoader(model, source={f"{name}_x": samples})
        model.train(
            data, epochs=100, batch_size=31, optimizer="adam", lr=0.05, printer=None
        )
        return np.ravel(centers.value_numpy), np.ravel(radius.value_numpy)  # type: ignore

    centers, radius = fit("rect_alone", shared=False)
    np.testing.assert_array_equal(centers, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(radius, [0.5])
    centers, radius = fit("rect_shared", shared=True)
    np.testing.assert_allclose(centers, [0.0, 1.4, 2.0], atol=0.01)
    np.testing.assert_allclose(radius, [1.0], atol=0.01)


def test_membership_widths_of_zero_or_below_give_no_nan():
    # A width at zero or below is held at 1e-7: its set collapses, and
    # neither the memberships nor the gradients become NaN. So do centers
    # that meet.
    x = Input("nan_x", sw=1)
    radius = Parameter("nan_radius", value=[0.5])
    sigma = Parameter("nan_sigma", value=[0.5])
    centers = Parameter("nan_centers", value=[0.0, 0.0, 1.0])
    outputs = [
        Output("nan_triangle", Triangular(radius=radius)([x])),
        Output("nan_box", Rectangular(radius=radius)([x])),
        Output("nan_bell", Gaussian(sigma=sigma)([x])),
        Output("nan_fuzzy", Fuzzify(centers, function=Triangular)([x])),
    ]
    model = Modely("nan_model", inputs=[x], outputs=outputs)
    for output in outputs:
        model.minimize(f"{output.name}_fit", output, 0.5)
    model.build()
    samples = np.array([-1.0, 0.0, 0.5], dtype=np.float32).reshape(-1, 1)
    data = DataLoader(model, source={"nan_x": samples})

    for width in (0.0, -1.0):
        radius.param.assign(np.array([width], dtype=np.float32))  # type: ignore
        sigma.param.assign(np.array([width], dtype=np.float32))  # type: ignore
        result = model({"nan_x": samples.reshape(-1, 1, 1)})
        assert all(np.isfinite(to_numpy(value)).all() for value in result.values())
        model.train(data, epochs=1, batch_size=3, optimizer="sgd", printer=None)
        for parameter in (radius, sigma, centers):
            assert np.isfinite(parameter.value_numpy).all()  # type: ignore


def test_fuzzify_takes_the_shape_its_functions_give():
    # The memberships side by side as the functions give them: two features
    # each, or the one sample a Fir makes of a window.
    x = Input("given_shape_x", sw=3)
    features = Fuzzify([0.0, 1.0], function=Linear(features=2), normalize=False)
    filtered = Fuzzify([0.0, 1.0], function=Fir, normalize=False)
    pairs, samples = features([x]), filtered([x])
    assert pairs.shape.dimensions == ((), 3, (4,))
    assert samples.shape.dimensions == ((), 1, (2,))
    model = Modely(
        "given_shape",
        inputs=[x],
        outputs=[Output("given_pairs", pairs), Output("given_samples", samples)],
    ).build()
    result = model({"given_shape_x": np.ones((2, 3, 1), dtype=np.float32)})
    assert tuple(result["given_pairs"].shape) == (2, 3, 4)
    assert tuple(result["given_samples"].shape) == (2, 1, 2)
    # A window beside one sample cannot be put side by side.
    with pytest.raises(ValueError, match="cannot be put side by side"):
        Fuzzify([0.0, 1.0], function=[Triangular, Fir])([x])


def _drawn(ax, count):
    """The signal, the ``count`` memberships drawn on ``ax`` and the centers
    marked after them."""
    lines = ax.get_lines()
    values = np.stack([np.asarray(line.get_ydata()) for line in lines[:count]], axis=1)
    marks = [np.asarray(line.get_xdata())[0] for line in lines[count:]]
    return np.asarray(lines[0].get_xdata()), values, marks


def test_fuzzify_plot_draws_the_memberships_as_they_are_now():
    # plot() draws what the layer computes, its Parameters at their current
    # values: centers moved after the model is built are drawn moved, by the
    # Fuzzify and by the stream it gave alike.
    import matplotlib.pyplot as plt

    def triangles(grid, centers, radius):
        return np.maximum(1 - np.abs(grid[:, None] - np.array(centers)) / radius, 0)

    x = Input("plot_x", sw=1)
    centers = Parameter("plot_centers", value=[0.0, 1.0, 2.0])
    fuzzy = Fuzzify(centers, function=Triangular, normalize=False)
    memberships = fuzzy([x])
    Modely("plot", inputs=[x], outputs=[Output("plot_mu", memberships)]).build()

    grid, values, marks = _drawn(fuzzy.plot(x_range=(-1.0, 3.0), points=9), 3)
    np.testing.assert_allclose(grid, np.linspace(-1.0, 3.0, 9))
    np.testing.assert_allclose(values, triangles(grid, [0, 1, 2], 1.0), atol=1e-6)
    np.testing.assert_allclose(marks, [0.0, 1.0, 2.0])

    # By default the outer gap beyond each end is drawn too.
    centers.param.assign(np.array([0.0, 2.0, 4.0], dtype=np.float32))  # type: ignore
    for layer in (fuzzy, memberships):
        grid, values, marks = _drawn(layer.plot(points=9), 3)
        np.testing.assert_allclose(grid, np.linspace(-2.0, 6.0, 9))
        np.testing.assert_allclose(values, triangles(grid, [0, 2, 4], 2.0), atol=1e-6)
        np.testing.assert_allclose(marks, [0.0, 2.0, 4.0])
    # The stream knows its signal, and names it.
    assert memberships.plot().get_xlabel() == "plot_x"
    plt.close("all")


def test_fuzzify_plot_draws_a_fuzzify_not_applied_yet():
    # Applied to no signal, a Fuzzify draws its functions with the values its
    # Parameters were given; centers computed from the inputs have none.
    import matplotlib.pyplot as plt

    cosines = Fuzzify([0.0, 1.0, 2.0], function=Cos, normalize=False)
    grid, values, _ = _drawn(cosines.plot(points=5), 3)
    np.testing.assert_allclose(grid, np.linspace(-1.0, 3.0, 5))
    expected = np.cos(grid[:, None] - np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(values, expected, atol=1e-6)

    radius = Parameter("plot_radius", value=[0.5])
    narrow = Fuzzify([0.0, 1.0], function=Triangular(radius=radius), normalize=False)
    _, ax = plt.subplots()
    assert narrow.plot(x_range=(0.0, 1.0), points=5, ax=ax) is ax
    _, values, _ = _drawn(ax, 2)
    np.testing.assert_allclose(values[:, 0], [1.0, 0.5, 0.0, 0.0, 0.0], atol=1e-6)

    computed = Fuzzify(Linear(features=3)([Input("plot_u", sw=1)]), function=Triangular)
    with pytest.raises(ValueError, match="is computed, with no value of its own"):
        computed.plot()
    plt.close("all")


def test_local_model_with_user_functions():
    # ------- No input function: the input itself, weighted cell by cell ----
    x = Input("x", dim=1, sw=1)
    k = Input("k", dim=1, sw=1)

    fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")([k])
    local_model = LocalModel(output_function=ReLU(), name="local_model")([x], [fuzzy_k])

    out = Output("out", local_model)
    model = Modely("model_with_local", inputs=[x, k], outputs=[out])
    model.build()

    # ------- Model inference -------
    dummy_input_x = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]], dtype=np.float32)
    dummy_input_k = np.array([[[-7]], [[0.5]], [[0.8]], [[5.0]]], dtype=np.float32)

    result = model({"x": dummy_input_x, "k": dummy_input_k})
    assert "out" in result
    assert result["out"].shape == (4, 1, 1)

    np.testing.assert_allclose(
        to_numpy(result["out"]),
        np.array([[[0.0]], [[2.0]], [[3.0]], [[0.0]]]),
        rtol=1e-5,
        atol=1e-5,
    )


def _fir_factory(kernel, bias, sizes):
    """A factory of Firs whose weights are the Parameters ``kernel[c]`` and
    ``bias[c]``, ``c`` the row-major number of the cell."""

    def factory(index):
        cell = int(np.ravel_multi_index(index, sizes))
        return Fir(
            features=kernel.shape[-1],
            kernel=Parameter(value=kernel[cell]),
            bias=Parameter(value=bias[cell]),
        )

    return factory


def test_local_model_matches_explicit_cells():
    # ------- One Fir per membership, with the weights given as Parameters -------
    x = Input("x_local", dim=1, sw=4)
    k = Input("k_local", dim=1, sw=1)
    centers = [0.0, 1.0, 2.0]
    rng = np.random.default_rng(0)
    kernel = rng.normal(size=(3, 4, 2)).astype(np.float32)  # (cell, time, features)
    bias = rng.normal(size=(3, 2)).astype(np.float32)

    activation = Fuzzify(centers=centers, function=Triangular)([k])
    local = LocalModel(_fir_factory(kernel, bias, (3,)), name="cells_local")(
        [x], [activation]
    )
    cells = [
        Fir(
            features=2,
            kernel=Parameter(value=kernel[i]),
            bias=Parameter(value=bias[i]),
        )([x])
        * Select(idx=i)([activation])
        for i in range(len(centers))
    ]
    model = Modely(
        "cells_local_model",
        inputs=[x, k],
        outputs=[Output("local", local), Output("explicit", Add()(cells))],
    ).build()

    result = model(
        {
            "x_local": np.arange(20, dtype=np.float32).reshape(5, 4, 1) / 10.0,
            "k_local": np.linspace(-0.5, 2.5, 5, dtype=np.float32).reshape(5, 1, 1),
        }
    )
    assert result["local"].shape == (5, 1, 2)
    np.testing.assert_allclose(
        to_numpy(result["local"]), to_numpy(result["explicit"]), rtol=1e-5, atol=1e-5
    )


def _two_fuzzy_inputs(prefix):
    x = Input(f"{prefix}_x", dim=1, sw=5)
    j = Input(f"{prefix}_j", dim=1, sw=1)
    k = Input(f"{prefix}_k", dim=1, sw=1)
    activation_j = Fuzzify(centers=[0.0, 1.0, 2.0, 3.0], function=Triangular)([j])
    activation_k = Fuzzify(centers=[0.0, 1.0, 2.0, 3.0, 4.0], function=Triangular)([k])
    data = {
        f"{prefix}_x": np.arange(30, dtype=np.float32).reshape(6, 5, 1) / 10.0 - 1.0,
        f"{prefix}_j": np.linspace(-0.5, 3.5, 6, dtype=np.float32).reshape(6, 1, 1),
        f"{prefix}_k": np.linspace(0.3, 4.2, 6, dtype=np.float32).reshape(6, 1, 1),
    }
    return x, j, k, activation_j, activation_k, data


def test_local_model_multiplies_fuzzifications_row_major():
    # ------- Two fuzzifications with 4 and 5 centers give 20 cells -------
    x, j, k, activation_j, activation_k, data = _two_fuzzy_inputs("product")
    rng = np.random.default_rng(1)
    kernel = rng.normal(size=(20, 5, 2)).astype(np.float32)
    bias = rng.normal(size=(20, 2)).astype(np.float32)

    local = LocalModel(_fir_factory(kernel, bias, (4, 5)), name="product_local")(
        [x], [activation_j, activation_k]
    )
    # Cell (a, b) is number a * 5 + b and weighs mu_j[a] * mu_k[b].
    cells = [
        Fir(
            features=2,
            kernel=Parameter(value=kernel[a * 5 + b]),
            bias=Parameter(value=bias[a * 5 + b]),
        )([x])
        * Select(idx=a)([activation_j])
        * Select(idx=b)([activation_k])
        for a in range(4)
        for b in range(5)
    ]
    model = Modely(
        "product_local_model",
        inputs=[x, j, k],
        outputs=[Output("local", local), Output("explicit", Add()(cells))],
    ).build()

    result = model(dict(data))
    np.testing.assert_allclose(
        to_numpy(result["local"]), to_numpy(result["explicit"]), rtol=1e-5, atol=1e-5
    )


def test_local_model_per_cell_once_and_vectorized_paths_agree():
    # ------- Per-cell factories and one vectorized call: one model -------
    x, j, k, activation_j, activation_k, data = _two_fuzzy_inputs("paths")
    activations = [activation_j, activation_k]
    rng = np.random.default_rng(2)
    kernel = rng.normal(size=(20, 5, 1)).astype(np.float32)
    bias = rng.normal(size=(20, 1)).astype(np.float32)

    once = LocalModel(
        _fir_factory(kernel, bias, (4, 5)), output_function=Tanh(), name="p_once"
    )([x], activations)
    per_cell = LocalModel(
        _fir_factory(kernel, bias, (4, 5)),
        output_function=lambda index: Tanh(),
        name="p_per_cell",
    )([x], activations)
    # One Fir with a feature per cell: column c is the kernel of cell c.
    vectorized = LocalModel(
        lambda streams: Fir(
            features=20,
            kernel=Parameter(value=kernel[:, :, 0].T),
            bias=Parameter(value=bias[:, 0]),
        )(streams),
        output_function=lambda streams: Tanh()(streams),
        vectorized=True,
        name="p_vectorized",
    )([x], activations)
    # The same Fir given as a layer: with vectorized=True it is called once too.
    vectorized_layer = LocalModel(
        Fir(
            features=20,
            kernel=Parameter(value=kernel[:, :, 0].T),
            bias=Parameter(value=bias[:, 0]),
        ),
        output_function=Tanh(),
        vectorized=True,
        name="p_vectorized_layer",
    )([x], activations)
    model = Modely(
        "paths_model",
        inputs=[x, j, k],
        outputs=[
            Output("once", once),
            Output("per_cell", per_cell),
            Output("vectorized", vectorized),
            Output("vectorized_layer", vectorized_layer),
        ],
    ).build()

    result = model(dict(data))
    assert result["once"].shape == (6, 1, 1)
    for name in ("per_cell", "vectorized", "vectorized_layer"):
        np.testing.assert_allclose(
            to_numpy(result[name]), to_numpy(result["once"]), rtol=1e-5, atol=1e-5
        )


def test_local_model_vectorized_fir_on_vector_cells():
    # ------- Fir(features=N) on a vector gives (*cell, N): the cells -------
    x = Input("vector_cells_x", dim=2, sw=4)
    g, h = Input("vector_cells_g", sw=1), Input("vector_cells_h", sw=1)
    activations = [
        Fuzzify(centers=[0.0, 1.0, 2.0], function=Triangular)([g]),
        Fuzzify(centers=[0.0, 1.0], function=Triangular)([h]),
    ]
    rng = np.random.default_rng(7)
    kernel = rng.normal(size=(2, 4, 6)).astype(np.float32)  # (*dim, window, N)
    bias = rng.normal(size=(2, 6)).astype(np.float32)

    vectorized = LocalModel(
        Fir(
            features=6,
            kernel=Parameter(value=kernel),
            bias=Parameter(value=bias),
        ),
        output_function=Tanh(),
        vectorized=True,
        name="vector_cells_vectorized",
    )([x], activations)
    # Cell (i, j) is number 2 * i + j: its own column of the kernel and bias.
    per_cell = LocalModel(
        lambda index: Fir(
            features=1,
            kernel=Parameter(value=kernel[..., 2 * index[0] + index[1], None]),
            bias=Parameter(value=bias[:, 2 * index[0] + index[1], None]),
        ),
        output_function=Tanh(),
        name="vector_cells_per_cell",
    )([x], activations)
    assert vectorized.shape.dimensions == per_cell.shape.dimensions == ((), 1, (2,))

    model = Modely(
        "vector_cells",
        inputs=[x, g, h],
        outputs=[
            Output("vector_cells_vectorized_out", vectorized),
            Output("vector_cells_per_cell_out", per_cell),
        ],
    ).build()
    result = model(
        {
            "vector_cells_x": rng.normal(size=(5, 4, 2)).astype(np.float32),
            "vector_cells_g": rng.uniform(0.0, 2.0, size=(5, 1, 1)).astype(np.float32),
            "vector_cells_h": rng.uniform(0.0, 1.0, size=(5, 1, 1)).astype(np.float32),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["vector_cells_vectorized_out"]),
        to_numpy(result["vector_cells_per_cell_out"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_model_passes_cell_index():
    # ------- A factory builds each cell from its (i_j, i_k) index -------
    x = Input("index_x", dim=1, sw=1)
    j = Input("index_j", dim=1, sw=1)
    k = Input("index_k", dim=1, sw=1)
    activation_j = Fuzzify(centers=[0.0, 1.0], function=Rectangular)([j])
    activation_k = Fuzzify(centers=[0.0, 1.0, 2.0], function=Rectangular)([k])

    seen = []

    def factory(index):
        seen.append(index)
        return lambda s: s[0] * float(10 * index[0] + index[1])

    local = LocalModel(factory, name="index_local")
    first = local([x], [activation_j, activation_k])
    # A second call applies the cells of the first: the factory is not called again.
    second = local([x], [activation_j, activation_k])
    model = Modely(
        "index_model",
        inputs=[x, j, k],
        outputs=[Output("first", first), Output("second", second)],
    ).build()
    assert seen == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    # Rectangular memberships pick exactly one cell.
    result = model(
        {
            "index_x": np.full((3, 1, 1), 2.0, dtype=np.float32),
            "index_j": np.array([0.0, 1.0, 1.0], dtype=np.float32).reshape(3, 1, 1),
            "index_k": np.array([2.0, 0.0, 1.0], dtype=np.float32).reshape(3, 1, 1),
        }
    )
    expected = np.array([2.0 * 2, 2.0 * 10, 2.0 * 11]).reshape(3, 1, 1)
    for name in ("first", "second"):
        np.testing.assert_allclose(to_numpy(result[name]), expected, atol=1e-5)


def test_local_model_repeated_call_shares_weights():
    x = Input("shared_local_x", sw=3)
    g = Input("shared_local_g", sw=1)
    mu = Fuzzify(centers=[0.0, 1.0], function=Triangular)(g)
    local = LocalModel(lambda index: Fir(features=1), name="shared_local")
    model = Modely(
        "shared_local_model",
        inputs=[x, g],
        outputs=[
            Output("first", local([x], [mu])),
            Output("second", local([x], [mu])),
        ],
    ).build()

    assert model.model is not None
    assert sum(isinstance(layer, FirImpl) for layer in model.model.layers) == 2
    result = model(
        {
            "shared_local_x": np.random.rand(4, 3, 1).astype(np.float32),
            "shared_local_g": np.random.rand(4, 1, 1).astype(np.float32),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["first"]), to_numpy(result["second"]), atol=1e-6
    )


def test_local_model_scalar_input_one_fuzzy_function_by_hand():
    # ------- y = x * sum_i mu_i(g) * gain_i, centers 0, 1, 2 -------
    x = Input("x_hand", sw=1)
    g = Input("g_hand", sw=1)
    mu = Fuzzify(centers=[0.0, 1.0, 2.0], function=Triangular)(g)
    gains = [1.0, 10.0, 100.0]
    y = LocalModel(
        lambda index: Fir(
            features=1, kernel=Parameter(value=[[gains[index[0]]]]), bias=False
        ),
        name="hand_local",
    )([x], [mu])
    model = Modely(
        "local_hand", inputs=[x, g], outputs=[Output("y", y), Output("mu", mu)]
    ).build()

    g_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    result = model(
        {
            "x_hand": np.full((5, 1, 1), 2.0, dtype=np.float32),
            "g_hand": g_values.reshape(-1, 1, 1),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["mu"]).reshape(5, 3),
        [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5], [0, 0, 1]],
        atol=1e-6,
    )
    # e.g. g = 0.5: y = 2 * (0.5 * 1 + 0.5 * 10) = 11
    np.testing.assert_allclose(
        to_numpy(result["y"]).ravel(), [2.0, 11.0, 20.0, 110.0, 200.0], rtol=1e-5
    )


def test_local_model_matrix_input_two_fuzzy_functions_by_hand():
    # ------- 3 x 2 = 6 cells over a 2x2 matrix, cell (i, j) is i * 2 + j -------
    X = Input("X_hand", dim=(2, 2), sw=1)
    a = Input("a_hand", sw=1)
    b = Input("b_hand", sw=1)
    mu_a = Fuzzify(centers=[0.0, 1.0, 2.0], function=Triangular)(a)
    mu_b = Fuzzify(centers=[0.0, 1.0], function=Triangular)(b)
    # Cell c filters every entry of X with the gain c + 1, entry by entry.
    Y = LocalModel(
        lambda index: Fir(
            features=1,
            kernel=Parameter(value=np.full((2, 2, 1, 1), index[0] * 2 + index[1] + 1)),
            bias=False,
        ),
        name="matrix_hand_local",
    )([X], [mu_a, mu_b])
    model = Modely(
        "local_matrix_hand",
        inputs=[X, a, b],
        outputs=[Output("Y", Y), Output("mu_a", mu_a), Output("mu_b", mu_b)],
    ).build()

    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = model(
        {
            "X_hand": values.reshape(1, 1, 2, 2),
            "a_hand": np.array([[[0.5]]], dtype=np.float32),
            "b_hand": np.array([[[0.25]]], dtype=np.float32),
        }
    )
    membership_a = to_numpy(result["mu_a"]).ravel()
    membership_b = to_numpy(result["mu_b"]).ravel()
    np.testing.assert_allclose(membership_a, [0.5, 0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(membership_b, [0.75, 0.25], atol=1e-6)
    np.testing.assert_allclose(
        np.outer(membership_a, membership_b).ravel(),
        [0.375, 0.125, 0.375, 0.125, 0.0, 0.0],
        atol=1e-6,
    )
    # Y = (0.375*1 + 0.125*2 + 0.375*3 + 0.125*4) * X = 2.25 * X
    assert result["Y"].shape == (1, 1, 2, 2)
    np.testing.assert_allclose(
        to_numpy(result["Y"]).reshape(2, 2), 2.25 * values, rtol=1e-5
    )


def test_local_model_matrix_input_without_input_function_by_hand():
    # ------- Y = sum_c sin(mu_c * X), the output applied to the (2, 2, 6) stack ----
    X = Input("X_plain", dim=(2, 2), sw=3)
    a = Input("a_plain", sw=1)
    b = Input("b_plain", sw=1)
    mu_a = Fuzzify(centers=[0.0, 1.0, 2.0], function=Triangular)(a)
    mu_b = Fuzzify(centers=[0.0, 1.0], function=Triangular)(b)
    weighted = LocalModel()([X], [mu_a, mu_b])
    vectorized = LocalModel(
        output_function=lambda streams: Sin()(streams), vectorized=True
    )([X], [mu_a, mu_b])
    model = Modely(
        "local_matrix_plain",
        inputs=[X, a, b],
        outputs=[Output("weighted", weighted), Output("vectorized", vectorized)],
    ).build()

    values = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) / 4.0
    values = values.transpose(0, 3, 1, 2)  # (batch, time, *dim)
    result = model(
        {
            "X_plain": values,
            "a_plain": np.array([[[0.5]]], dtype=np.float32),
            "b_plain": np.array([[[0.25]]], dtype=np.float32),
        }
    )
    mu = np.array([0.375, 0.125, 0.375, 0.125, 0.0, 0.0], dtype=np.float32)
    assert result["vectorized"].shape == (1, 3, 2, 2)
    # The memberships sum to one, so weighting alone gives the input back.
    np.testing.assert_allclose(to_numpy(result["weighted"]), values, rtol=1e-5)
    np.testing.assert_allclose(
        to_numpy(result["vectorized"]),
        np.sin(mu.reshape(6, 1, 1, 1, 1) * values).sum(axis=0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_local_model_memberships_along_the_window_by_hand():
    # ------- y[t] = x[t] * sum_(a, b) mu_a[a, t] * mu_b[b] * (a * 2 + b + 1) -------
    x = Input("x_window", sw=3)
    a = Input("a_window", sw=3)
    b = Input("b_window", sw=1)
    mu_a = Fuzzify(centers=[0.0, 1.0, 2.0], function=Triangular)(
        a
    )  # one membership per sample
    mu_b = Fuzzify(centers=[0.0, 1.0], function=Triangular)(
        b
    )  # the same for every sample
    y = LocalModel(
        lambda index: lambda streams: streams[0] * float(index[0] * 2 + index[1] + 1),
        name="window_local",
    )([x], [mu_a, mu_b])
    model = Modely(
        "local_window",
        inputs=[x, a, b],
        outputs=[Output("y", y), Output("mu_a", mu_a), Output("mu_b", mu_b)],
    ).build()

    x_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = model(
        {
            "x_window": x_values.reshape(1, 3, 1),
            "a_window": np.array([0.0, 0.5, 2.0], dtype=np.float32).reshape(1, 3, 1),
            "b_window": np.array([[[0.25]]], dtype=np.float32),
        }
    )
    membership_a = to_numpy(result["mu_a"]).reshape(3, 3).T  # (center, sample)
    membership_b = to_numpy(result["mu_b"]).ravel()
    gains = np.arange(1.0, 7.0).reshape(3, 2)
    expected = x_values * np.einsum("at,b,ab->t", membership_a, membership_b, gains)
    assert result["y"].shape == (1, 3, 1)
    np.testing.assert_allclose(to_numpy(result["y"]).ravel(), expected, rtol=1e-5)


def test_local_model_rejects_mismatched_inputs():
    x = Input("x_invalid", dim=1, sw=2)
    k = Input("k_invalid", dim=1, sw=3)
    activation = Fuzzify(centers=[0.0, 1.0], function=Triangular)([TimeSelect(-1)(k)])

    with pytest.raises(TypeError, match=r"pass a factory f\(index\)"):
        LocalModel([Fir(features=1)] * 2, name="list_local")

    with pytest.raises(ValueError, match="every cell must have the same shape"):
        LocalModel(lambda index: Fir(features=index[0] + 1), name="uneven_local")(
            [x], [activation]
        )

    with pytest.raises(ValueError, match="every cell must have the same shape"):
        LocalModel(
            output_function=lambda index: Fir(features=index[0] + 1),
            name="uneven_output_local",
        )([x], [activation])

    with pytest.raises(ValueError, match="one entry per cell"):
        LocalModel(
            lambda streams: Fir(features=3)(streams),
            vectorized=True,
            name="vectorized_local",
        )([x], [activation])

    with pytest.raises(ValueError, match="there must be one"):
        LocalModel(name="two_inputs_local")([x, TimeRange(1, 3)(k)], [activation])

    windowed_activation = Fuzzify(centers=[0.0, 1.0], function=Triangular)([k])
    with pytest.raises(ValueError, match="must match, or be 1"):
        LocalModel(name="windowed_local")([x], [windowed_activation])

    sequence = Input("k_invalid_seq", dim=1, seq=-1, sw=1)
    sequence_activation = Fuzzify(centers=[0.0, 1.0], function=Triangular)([sequence])
    with pytest.raises(ValueError, match="activation 0 must have no sequence"):
        LocalModel(name="sequence_local")([x], [sequence_activation])


def test_equation_learner_composes_symbolic_functions_and_multiple_inputs():
    x = Input("equation_x", sw=1)
    y = Input("equation_y", sw=1)
    equation = EquationLearner(
        functions=["identity", (lambda left, right: left * right, 2), Sin],
        linear_in=Linear(
            features=4,
            kernel=Parameter(
                "equation_in_matrix", value=[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
            ),
            bias=False,
            name="equation_linear_in",
        ),
        linear_out=Linear(
            features=1,
            kernel=Parameter("equation_out_weights", value=[2.0, 3.0, 4.0]),
            bias=False,
            name="equation_linear_out",
        ),
        name="equation",
    )
    learned = equation([x, y])
    model = Modely(
        "equation_model",
        inputs=[x, y],
        outputs=[Output("result", learned)],
    ).build()

    inputs = {
        "equation_x": np.array([[[2.0]]], dtype=np.float32),
        "equation_y": np.array([[[0.5]]], dtype=np.float32),
    }
    result = to_numpy(model(inputs)["result"])
    expected = 2.0 * 2.0 + 3.0 * 2.0 * 0.5 + 4.0 * np.sin(0.5)

    assert result.shape == (1, 1, 1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert equation.model is not None
    internal_types = {type(node).__name__ for node in equation.model.order}
    assert {"Linear", "Select", "Multiply", "Sin", "Concatenate"} <= internal_types


def test_equation_learner_supports_layer_classes_and_basis_output():
    x = Input("basis_x", sw=1)
    equation = EquationLearner(
        functions=[Sin, Cos, "add"],
        linear_in=Linear(features=4, kernel="ones", bias=False),
        name="basis_equation",
    )
    basis = equation(x)
    model = Modely(
        "basis_model",
        inputs=[x],
        outputs=[Output("basis", basis)],
    ).build()

    value = np.array([[[0.25]]], dtype=np.float32)
    result = to_numpy(model({"basis_x": value})["basis"])
    expected = np.array(
        [[[np.sin(0.25), np.cos(0.25), 0.5]]],
        dtype=np.float32,
    ).reshape((1, 1, 3))

    assert basis.dim == (3,)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_equation_learner_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="at least one function"):
        EquationLearner([])

    with pytest.raises(ValueError, match="total number of function arguments"):
        EquationLearner(
            [Sin, "add"],
            linear_in=Linear(features=2),
        )

    with pytest.raises(ValueError, match="Unknown EquationLearner function"):
        EquationLearner(["not_a_function"])


def test_integrate():
    # make a simple forward Euler integrator step
    x = Input("x", sw=1)
    x0 = Input("x0", sw=1)
    integrate = IntegrateStep(solver="euler", dt=0.1)(x0, x)
    out = Output("x_hat", integrate)
    model = Modely(name="integrator", inputs=[x, x0], outputs=[out])
    model.build()

    # make a loop that rolls out the integrator along a sequence of rates
    x_seq = Input("x_seq", seq=-1, sw=1)
    x0_loop = Input("x0_loop", sw=1)
    loop = Loop(f=model, callback={x0: out}, init={x0: x0_loop})({x: x_seq})
    out_loop = Output("x_hat_loop", loop)
    model_loop = Modely(
        name="integrator_loop", inputs=[x_seq, x0_loop], outputs=[out_loop]
    ).build()

    dummy = np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)
    start = np.array(0.0).reshape(1, 1, 1)
    by_hand = to_numpy(model_loop({"x_seq": dummy, "x0_loop": start})["x_hat_loop"])
    np.testing.assert_allclose(by_hand, [[[[0.1]], [[0.3]], [[0.6]]]], rtol=1e-6)

    # The block, in a ModelyLoop, closes the same feedback: one rate per step.
    y = Input("y", sw=1)
    y0 = Input("y0", sw=1)
    new_integrate = Integrate(solver="euler", dt=0.1, init=y0)(y)
    out_int = Output("y_hat", new_integrate)
    new_model = ModelyLoop("new_integrator", inputs=[y, y0], outputs=[out_int]).build()

    result = to_numpy(new_model({"y": dummy, "y0": _over_steps(start, 3)})["y_hat"])
    assert result.shape == (1, 3, 1, 1)
    np.testing.assert_allclose(result, by_hand, rtol=1e-6)


# ---------------------------------------------------------------------------
# Integrate: IntegrateStep across the steps of a ModelyLoop
# ---------------------------------------------------------------------------


def _over_steps(value, steps):
    """A value per sample of the batch, laid out over the steps of a rollout:
    a ModelyLoop reads an initial value at its first step."""
    return np.repeat(np.asarray(value, dtype=np.float32)[:, None], steps, axis=1)


def _integrate_model(name, length=None, dim=1, **kwargs):
    rate = Input(f"{name}_rate", dim=dim, sw=1)
    out = Output(f"{name}_out", Integrate(**kwargs)(rate))
    return ModelyLoop(name, inputs=[rate], outputs=[out], length=length).build()


@pytest.mark.parametrize("solver", ["euler", "rectangular", "trapezoidal"])
def test_integrate_matches_integrate_step_over_the_same_samples(solver):
    # Step by step or along a time window, the same rule on the same samples.
    samples = np.array([1.0, -2.0, 4.0, 0.5, 3.0], dtype=np.float32)
    horizon = _integrate_model(f"along_{solver}", solver=solver, dt=0.1, init=0.7)
    w = Input(f"window_{solver}", sw=5)
    window = Modely(
        f"window_{solver}",
        inputs=[w],
        outputs=[
            Output(
                f"window_{solver}_out",
                IntegrateStep(solver=solver, dt=0.1)(0.7, w),
            )
        ],
    ).build()

    along = to_numpy(
        horizon({f"along_{solver}_rate": samples.reshape(1, 5, 1, 1)})[
            f"along_{solver}_out"
        ]
    )
    over = to_numpy(
        window({f"window_{solver}": samples.reshape(1, 5, 1)})[f"window_{solver}_out"]
    )
    np.testing.assert_allclose(along.ravel(), over.ravel(), rtol=1e-5, atol=1e-6)


def test_integrate_follows_a_dynamic_horizon_without_rebuilding():
    model = _integrate_model("dynamic", solver="euler", dt=0.5)
    for steps in (1, 4, 9):
        rate = np.ones((2, steps, 1, 1), dtype=np.float32)
        result = to_numpy(model({"dynamic_rate": rate})["dynamic_out"])
        assert result.shape == (2, steps, 1, 1)
        np.testing.assert_allclose(result[0].ravel(), 0.5 * np.arange(1, steps + 1))


def test_integrate_over_a_fixed_horizon():
    model = _integrate_model("fixed", length=4, solver="euler", dt=1.0)
    result = to_numpy(
        model({"fixed_rate": np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)})[
            "fixed_out"
        ]
    )
    np.testing.assert_allclose(result.ravel(), [1.0, 3.0, 6.0, 10.0])


def test_integrate_starts_from_none_a_number_or_a_stream():
    rate = np.full((1, 3, 1, 1), 2.0, dtype=np.float32)
    zero = _integrate_model("from_zero", dt=0.1)
    number = _integrate_model("from_number", dt=0.1, init=5.0)
    r, r0 = Input("from_stream_rate", sw=1), Input("from_stream_init", sw=1)
    stream = ModelyLoop(
        "from_stream",
        inputs=[r, r0],
        outputs=[Output("from_stream_out", Integrate(dt=0.1, init=r0)(r))],
    ).build()

    np.testing.assert_allclose(
        to_numpy(zero({"from_zero_rate": rate})["from_zero_out"]).ravel(),
        [0.2, 0.4, 0.6],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(number({"from_number_rate": rate})["from_number_out"]).ravel(),
        [5.2, 5.4, 5.6],
        rtol=1e-6,
    )
    # One initial value per sample of the batch.
    result = to_numpy(
        stream(
            {
                "from_stream_rate": np.concatenate([rate, rate]),
                "from_stream_init": _over_steps(
                    np.array([1.0, -1.0]).reshape(2, 1, 1), 3
                ),
            }
        )["from_stream_out"]
    )
    np.testing.assert_allclose(result[0].ravel(), [1.2, 1.4, 1.6], rtol=1e-6)
    np.testing.assert_allclose(result[1].ravel(), [-0.8, -0.6, -0.4], rtol=1e-6)


def test_integrate_each_feature_of_a_vector_rate():
    rate, start = (
        Input("vector_rate", dim=2, sw=1),
        Input("vector_init", dim=2, sw=1),
    )
    scalar_start = Input("vector_scalar_init", sw=1)
    steps = rate
    model = ModelyLoop(
        "vector",
        inputs=[rate, start, scalar_start],
        outputs=[
            Output("vector_out", Integrate(dt=1.0, init=start)(steps)),
            Output(
                "vector_scalar_out",
                Integrate(dt=1.0, init=scalar_start)(steps),
            ),
        ],
    ).build()
    values = np.array([[1.0, 1.0, 1.0], [10.0, 20.0, 30.0]], dtype=np.float32)

    result = model(
        {
            "vector_rate": values.T.reshape(1, 3, 1, 2),
            "vector_init": _over_steps(np.array([0.0, 100.0]).reshape(1, 1, 2), 3),
            "vector_scalar_init": _over_steps(np.array([1.0]).reshape(1, 1, 1), 3),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["vector_out"]).reshape(3, 2).T, [[1, 2, 3], [110, 130, 160]]
    )
    # A scalar initial value starts every feature.
    np.testing.assert_allclose(
        to_numpy(result["vector_scalar_out"]).reshape(3, 2).T,
        [[2, 3, 4], [11, 31, 61]],
    )


def test_integrate_twice_from_acceleration_to_position():
    # Constant acceleration: the trapezoid of a linear velocity is exact.
    dt, steps, a, v0, x0 = 0.1, 20, 2.0, 1.0, 3.0
    acc = Input("acc", sw=1)
    vel = Integrate(solver="trapezoidal", dt=dt, init=v0)(acc)
    pos = Integrate(solver="trapezoidal", dt=dt, init=x0)(vel)
    model = ModelyLoop(
        "kinematics", inputs=[acc], outputs=[Output("vel", vel), Output("pos", pos)]
    ).build()

    result = model({"acc": np.full((1, steps, 1, 1), a, dtype=np.float32)})
    t = dt * np.arange(1, steps + 1)
    np.testing.assert_allclose(to_numpy(result["vel"]).ravel(), v0 + a * t, rtol=1e-5)
    # The first velocity step is a rectangle, so the position trails the exact
    # parabola by the area that rectangle adds: dt/2 * (v(dt) - v0) = a*dt^2/2.
    expected = x0 + v0 * t + a * t**2 / 2 + a * dt**2 / 2
    np.testing.assert_allclose(to_numpy(result["pos"]).ravel(), expected, rtol=1e-5)


@pytest.mark.slow
def test_integrate_is_trained_through():
    # x = Integrate(k * u): the gain of the rate is learnt from the trajectory.
    # (A fixed horizon: one trajectory per simulation.)
    steps = 8
    u = Input("trained_u", sw=1)
    k = Parameter("trained_k", value=0.5)
    x = Output("trained_x", Integrate(dt=0.1)(u * k))
    model = ModelyLoop("trained", inputs=[u], outputs=[x], length=steps)
    model.minimize("fit", x, Input("trained_target", sw=1))
    model.build()
    rng = np.random.default_rng(0)
    rates = rng.uniform(-1.0, 1.0, (64, steps)).astype(np.float32)
    targets = 0.1 * np.cumsum(2.0 * rates, axis=1)
    data = DataLoader(
        model,
        source=[
            {"trained_u": rates[i], "trained_target": targets[i]}
            for i in range(len(rates))
        ],
    )
    assert len(data) == len(rates)  # one trajectory per simulation

    model.train(data, epochs=200, batch_size=16, lr=0.05, printer=None)

    np.testing.assert_allclose(to_numpy(k.param).ravel(), [2.0], atol=1e-2)


def test_integrate_round_trips_through_save_and_the_keras_file(tmp_path):
    import keras

    rate, start = Input("saved_rate", sw=1), Input("saved_init", sw=1)
    model = ModelyLoop(
        "saved_integrator",
        inputs=[rate, start],
        outputs=[
            Output(
                "saved_out",
                Integrate(solver="trapezoidal", dt=0.2, init=start)(rate),
            )
        ],
    ).build()
    data = {
        "saved_rate": np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float32).reshape(
            1, 4, 1, 1
        ),
        "saved_init": _over_steps(np.array([0.5]).reshape(1, 1, 1), 4),
    }
    expected = to_numpy(model(data)["saved_out"])

    model.save(tmp_path / "saved")
    restored = Modely.load(tmp_path / "saved")
    np.testing.assert_allclose(
        to_numpy(restored(data)["saved_out"]), expected, rtol=1e-6
    )

    model.export_keras(tmp_path)
    exported = Modely.import_keras(str(tmp_path / "saved_integrator.keras"))
    assert exported is not None
    np.testing.assert_allclose(
        to_numpy(exported(data)["saved_out"]),  # type: ignore
        expected,
        rtol=1e-6,
    )
    assert isinstance(exported, keras.Model)


@pytest.mark.parametrize("layer", [Fir, Linear])
def test_integrate_a_rate_from_a_layer_with_parameter_weights(layer):
    # rate = 2u + 1 on [1, 2, 3, 4] is [3, 5, 7, 9]; Euler with dt = 1 sums it.
    name = f"prate_{layer.__name__.lower()}"
    u = Input(f"{name}_u", sw=1)
    gain = Parameter(f"{name}_gain", value=[2.0])
    offset = Parameter(f"{name}_offset", value=[1.0])
    rate = layer(features=1, kernel=gain, bias=offset)(u)
    model = ModelyLoop(
        name,
        inputs=[u],
        outputs=[Output(f"{name}_out", Integrate(solver="euler", dt=1.0)(rate))],
        length=4,
    ).build()

    values = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    result = to_numpy(model({f"{name}_u": values})[f"{name}_out"])
    np.testing.assert_allclose(result.ravel(), [3.0, 8.0, 15.0, 24.0])


def test_integrate_feeds_further_relations():
    # The result is the value at the end of the step, a stream to compute with
    # like any other.
    acc = Input("further_acc", sw=1)
    vel = Integrate(dt=0.1)(acc)
    gain = Parameter("further_gain", value=3.0)
    model = ModelyLoop(
        "further",
        inputs=[acc],
        outputs=[Output("further_out", Sin()([vel]) * gain + Integrate(dt=0.1)(vel))],
    ).build()

    result = to_numpy(
        model({"further_acc": np.ones((1, 5, 1, 1), dtype=np.float32)})["further_out"]
    )
    v = 0.1 * np.arange(1, 6)
    np.testing.assert_allclose(
        result.ravel(), 3.0 * np.sin(v) + 0.1 * np.cumsum(v), rtol=1e-5
    )


@pytest.mark.slow
def test_a_dynamic_integrate_trains_on_simulations_of_different_lengths():
    # Every simulation is one trajectory of its own length; the loader pads the
    # shorter ones, and the padded steps are left out of the loss.
    u = Input("dynamic_trained_u", sw=1)
    k = Parameter("dynamic_trained_k", value=0.5)
    x = Output("dynamic_trained_x", Integrate(dt=0.1)(u * k))
    model = ModelyLoop("dynamic_trained", inputs=[u], outputs=[x])
    model.minimize("fit", x, Input("dynamic_trained_target", sw=1))
    model.build()
    rng = np.random.default_rng(1)
    simulations = []
    for length in (4, 9, 6, 12) * 8:
        rates = rng.uniform(-1.0, 1.0, length).astype(np.float32)
        simulations.append(
            {
                "dynamic_trained_u": rates,
                "dynamic_trained_target": 0.1 * np.cumsum(2.0 * rates),
            }
        )
    data = DataLoader(model, source=simulations, seq_length="full")
    assert len(data) == len(simulations)

    model.train(data, epochs=200, batch_size=8, lr=0.05, printer=None)

    np.testing.assert_allclose(to_numpy(k.param).ravel(), [2.0], atol=1e-2)


def test_integrate_rejects_what_it_cannot_integrate():
    # The rate is one step of a ModelyLoop: one sample, no sequence.
    with pytest.raises(ValueError, match="carries a sequence"):
        Integrate(dt=0.1)(Input("with_seq", seq=-1, sw=1))
    with pytest.raises(ValueError, match="one rate sample per step"):
        Integrate(dt=0.1)(Input("windowed", sw=3))
    with pytest.raises(ValueError, match="solver must be one of"):
        Integrate(solver="rk4", dt=0.1)
    with pytest.raises(ValueError, match="dt"):
        Integrate()
    with pytest.raises(TypeError, match="configured first"):
        Integrate(Input("misused", seq=-1, sw=1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integrates one rate"):
        Integrate(dt=0.1)([Input("first", sw=1), Input("second", sw=1)])
    rate = Input("checked_rate", dim=2, sw=1)
    with pytest.raises(ValueError, match="single sample"):
        Integrate(dt=0.1, init=Input("long_init", dim=2, sw=2))(rate)
    with pytest.raises(ValueError, match="init has dim"):
        Integrate(dt=0.1, init=Input("wide_init", dim=3, sw=1))(rate)
    with pytest.raises(ValueError, match="init has the sequence"):
        Integrate(dt=0.1, init=Input("seq_init", dim=2, seq=4, sw=1))(rate)


def test_derivative():
    # A one-step backward difference: the step reads the previous sample.
    x = Input("dstep_x", sw=1)
    previous = Input("dstep_previous", sw=1)
    out = Output("dstep_out", DerivativeStep(dt=0.1)(previous, x))
    model = Modely(
        name="differentiator",
        inputs=[x, previous],
        outputs=[out, Output("dstep_sample", Identity()([x]))],
    ).build()

    # A loop that rolls the step out along a sequence of samples, feeding each
    # sample back as the previous one of the next step.
    x_seq = Input("dstep_x_seq", seq=-1, sw=1)
    x0_loop = Input("dstep_x0_loop", sw=1)
    derivative, _ = Loop(
        f=model, callback={previous: "dstep_sample"}, init={previous: x0_loop}
    )({x: x_seq})
    model_loop = Modely(
        name="differentiator_loop",
        inputs=[x_seq, x0_loop],
        outputs=[Output("dstep_loop", derivative)],
    ).build()

    dummy = np.array([1.0, 3.0, 6.0]).reshape(1, 3, 1, 1)
    start = np.array(0.5).reshape(1, 1, 1)
    by_hand = to_numpy(
        model_loop({"dstep_x_seq": dummy, "dstep_x0_loop": start})["dstep_loop"]
    )
    np.testing.assert_allclose(by_hand.ravel(), [5.0, 20.0, 30.0], rtol=1e-5)

    # The layer, in a ModelyLoop, gives the same derivatives step by step.
    y = Input("dstep_y", sw=1)
    y0 = Input("dstep_y0", sw=1)
    new_model = ModelyLoop(
        "new_differentiator",
        inputs=[y, y0],
        outputs=[Output("dstep_y_hat", Derivative(dt=0.1, init=y0)(y))],
    ).build()

    result = to_numpy(
        new_model({"dstep_y": dummy, "dstep_y0": _over_steps(start, 3)})["dstep_y_hat"]
    )
    assert result.shape == (1, 3, 1, 1)
    np.testing.assert_allclose(result, by_hand, rtol=1e-5)


# ---------------------------------------------------------------------------
# Derivative: DerivativeStep's stencil across the steps of a ModelyLoop
# ---------------------------------------------------------------------------


def _derivative_model(name, length=None, dim=1, **kwargs):
    signal = Input(f"{name}_signal", dim=dim, sw=1)
    out = Output(f"{name}_out", Derivative(**kwargs)(signal))
    return ModelyLoop(name, inputs=[signal], outputs=[out], length=length).build()


@pytest.mark.parametrize(
    "stencil",
    [
        {"order": 1},
        {"order": 2},
        {"order": 1, "window": 5},
        {"order": 1, "window": 3, "poly_order": 2},
    ],
    ids=["first", "second", "smoothing", "bdf2"],
)
def test_derivative_matches_derivative_step_over_the_same_samples(stencil):
    # Step by step or along a time window, the same stencil on the same
    # samples.
    name = "_".join(f"{key}{value}" for key, value in stencil.items())
    samples = np.array([1.0, -2.0, 4.0, 0.5, 3.0, 2.0], dtype=np.float32)
    horizon = _derivative_model(f"dalong_{name}", dt=0.1, init=0.7, **stencil)
    w = Input(f"dwindow_{name}", sw=6)
    window = Modely(
        f"dwindow_{name}",
        inputs=[w],
        outputs=[
            Output(
                f"dwindow_{name}_out",
                DerivativeStep(dt=0.1, **stencil)(0.7, w),
            )
        ],
    ).build()

    along = to_numpy(
        horizon({f"dalong_{name}_signal": samples.reshape(1, 6, 1, 1)})[
            f"dalong_{name}_out"
        ]
    )
    over = to_numpy(
        window({f"dwindow_{name}": samples.reshape(1, 6, 1)})[f"dwindow_{name}_out"]
    )
    np.testing.assert_allclose(along.ravel(), over.ravel(), rtol=1e-5, atol=1e-3)


def test_derivative_follows_a_dynamic_horizon_without_rebuilding():
    # x = k^2: the backward difference is (k^2 - (k - 1)^2) / dt.
    model = _derivative_model("ddynamic", dt=0.5)
    for steps in (1, 4, 9):
        k = np.arange(1, steps + 1, dtype=np.float32)
        signal = np.tile((k**2).reshape(1, steps, 1, 1), (2, 1, 1, 1))
        result = to_numpy(model({"ddynamic_signal": signal})["ddynamic_out"])
        assert result.shape == (2, steps, 1, 1)
        np.testing.assert_allclose(result[1].ravel(), (k**2 - (k - 1) ** 2) / 0.5)


def test_derivative_over_a_fixed_horizon():
    model = _derivative_model("dfixed", length=4, dt=1.0)
    result = to_numpy(
        model({"dfixed_signal": np.array([1.0, 3.0, 6.0, 10.0]).reshape(1, 4, 1, 1)})[
            "dfixed_out"
        ]
    )
    np.testing.assert_allclose(result.ravel(), [1.0, 2.0, 3.0, 4.0])


def test_derivative_starts_from_none_a_number_or_a_stream():
    signal = np.array([2.0, 3.0, 5.0], dtype=np.float32).reshape(1, 3, 1, 1)
    zero = _derivative_model("dfrom_zero", dt=0.1)
    number = _derivative_model("dfrom_number", dt=0.1, init=1.0)
    x, x0 = Input("dfrom_stream_signal", sw=1), Input("dfrom_stream_init", sw=1)
    stream = ModelyLoop(
        "dfrom_stream",
        inputs=[x, x0],
        outputs=[Output("dfrom_stream_out", Derivative(dt=0.1, init=x0)(x))],
    ).build()

    np.testing.assert_allclose(
        to_numpy(zero({"dfrom_zero_signal": signal})["dfrom_zero_out"]).ravel(),
        [20.0, 10.0, 20.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(number({"dfrom_number_signal": signal})["dfrom_number_out"]).ravel(),
        [10.0, 10.0, 20.0],
        rtol=1e-5,
    )
    # One initial value per sample of the batch.
    result = to_numpy(
        stream(
            {
                "dfrom_stream_signal": np.concatenate([signal, signal]),
                "dfrom_stream_init": _over_steps(
                    np.array([1.0, -1.0]).reshape(2, 1, 1), 3
                ),
            }
        )["dfrom_stream_out"]
    )
    np.testing.assert_allclose(result[0].ravel(), [10.0, 10.0, 20.0], rtol=1e-5)
    np.testing.assert_allclose(result[1].ravel(), [30.0, 10.0, 20.0], rtol=1e-5)


def test_derivative_reads_the_history_a_longer_stencil_needs():
    # The second derivative reads the two steps before the first: one init
    # sample stands for both, or the two are given, oldest first. On x = k^2
    # with the true history 0, 1 every second difference is 2.
    x = Input("dhistory_x", sw=1)
    one, two = Input("dhistory_one", sw=1), Input("dhistory_two", sw=2)
    model = ModelyLoop(
        "dhistory",
        inputs=[x, one, two],
        outputs=[
            Output("dhistory_one_out", Derivative(order=2, dt=1.0, init=one)(x)),
            Output("dhistory_two_out", Derivative(order=2, dt=1.0, init=two)(x)),
        ],
    ).build()

    result = model(
        {
            "dhistory_x": np.array([4.0, 9.0, 16.0]).reshape(1, 3, 1, 1),
            "dhistory_one": _over_steps(np.array([1.0]).reshape(1, 1, 1), 3),
            "dhistory_two": _over_steps(np.array([0.0, 1.0]).reshape(1, 2, 1), 3),
        }
    )
    # 1 - 2*1 + 4, 1 - 2*4 + 9, 4 - 2*9 + 16
    np.testing.assert_allclose(
        to_numpy(result["dhistory_one_out"]).ravel(), [3.0, 2.0, 2.0]
    )
    np.testing.assert_allclose(
        to_numpy(result["dhistory_two_out"]).ravel(), [2.0, 2.0, 2.0]
    )


def test_derivative_each_feature_of_a_vector_signal():
    x, start = (
        Input("dvector_signal", dim=2, sw=1),
        Input("dvector_init", dim=2, sw=1),
    )
    scalar_start = Input("dvector_scalar_init", sw=1)
    model = ModelyLoop(
        "dvector",
        inputs=[x, start, scalar_start],
        outputs=[
            Output("dvector_out", Derivative(dt=1.0, init=start)(x)),
            Output("dvector_scalar_out", Derivative(dt=1.0, init=scalar_start)(x)),
        ],
    ).build()
    values = np.array([[1.0, 2.0, 4.0], [10.0, 30.0, 60.0]], dtype=np.float32)

    result = model(
        {
            "dvector_signal": values.T.reshape(1, 3, 1, 2),
            "dvector_init": _over_steps(np.array([0.0, 10.0]).reshape(1, 1, 2), 3),
            "dvector_scalar_init": _over_steps(np.array([1.0]).reshape(1, 1, 1), 3),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["dvector_out"]).reshape(3, 2).T, [[1, 1, 2], [0, 20, 30]]
    )
    # A scalar initial value starts every feature.
    np.testing.assert_allclose(
        to_numpy(result["dvector_scalar_out"]).reshape(3, 2).T,
        [[0, 1, 2], [9, 20, 30]],
    )


def test_integrate_and_derivative_invert_each_other_step_by_step():
    dt = 0.1
    x, x0 = Input("dinverse_x", sw=1), Input("dinverse_init", sw=1)
    model = ModelyLoop(
        "dinverse",
        inputs=[x, x0],
        outputs=[
            Output(
                "dinverse_integral_of_derivative",
                Integrate(dt=dt, init=x0)(Derivative(dt=dt, init=x0)(x)),
            ),
            Output(
                "dinverse_derivative_of_integral",
                Derivative(dt=dt, init=x0)(Integrate(dt=dt, init=x0)(x)),
            ),
        ],
    ).build()
    values = np.random.default_rng(0).normal(size=(2, 7, 1, 1)).astype(np.float32)

    result = model(
        {
            "dinverse_x": values,
            "dinverse_init": _over_steps(np.array([0.3, -2.0]).reshape(2, 1, 1), 7),
        }
    )
    for name in ("dinverse_integral_of_derivative", "dinverse_derivative_of_integral"):
        np.testing.assert_allclose(to_numpy(result[name]), values, atol=1e-4)


def test_derivative_undoes_integrate_step_by_step():
    # The velocity and the acceleration of an integrated position: each
    # Derivative undoes the Integrate it follows, step by step. Over the
    # one-sample window of each step, DerivativeStep reads only init.
    dt, v0, x0 = 0.1, 1.0, 3.0
    acc = Input("dloop_acc", sw=1)
    pos = Integrate(dt=dt, init=x0)(Integrate(dt=dt, init=v0)(acc))
    vel = Derivative(dt=dt, init=x0)(pos)
    model = ModelyLoop(
        "dloop",
        inputs=[acc],
        outputs=[
            Output("dloop_vel", vel),
            Output("dloop_acc_back", Derivative(dt=dt, init=v0)(vel)),
            Output("dloop_window", DerivativeStep(dt=dt)(x0, pos)),
            Output("dloop_pos", pos),
        ],
    ).build()
    a = np.array([2.0, -1.0, 0.5, 3.0, 0.0], dtype=np.float32).reshape(1, 5, 1, 1)

    result = model({"dloop_acc": a})
    np.testing.assert_allclose(
        to_numpy(result["dloop_vel"]).ravel(), v0 + dt * np.cumsum(a), atol=1e-4
    )
    np.testing.assert_allclose(to_numpy(result["dloop_acc_back"]), a, atol=1e-3)
    np.testing.assert_allclose(
        to_numpy(result["dloop_window"]),
        (to_numpy(result["dloop_pos"]) - x0) / dt,
        atol=1e-4,
    )


@pytest.mark.parametrize("layer", [Fir, Linear])
def test_derivative_of_a_signal_from_a_layer_with_parameter_weights(layer):
    # signal = 2u + 1 on [1, 2, 4, 7] is [3, 5, 9, 15]; from zero with dt = 1
    # the differences are [3, 2, 4, 6].
    name = f"dlayer_{layer.__name__.lower()}"
    u = Input(f"{name}_u", sw=1)
    gain = Parameter(f"{name}_gain", value=[2.0])
    offset = Parameter(f"{name}_offset", value=[1.0])
    signal = layer(features=1, kernel=gain, bias=offset)(u)
    model = ModelyLoop(
        name,
        inputs=[u],
        outputs=[Output(f"{name}_out", Derivative(dt=1.0)(signal))],
        length=4,
    ).build()

    values = np.array([1.0, 2.0, 4.0, 7.0]).reshape(1, 4, 1, 1)
    result = to_numpy(model({f"{name}_u": values})[f"{name}_out"])
    np.testing.assert_allclose(result.ravel(), [3.0, 2.0, 4.0, 6.0])


def test_a_derivative_feeds_any_relation():
    # The result is the derivative at the step, a stream to compute with.
    x = Input("ddynamic_further_x", sw=1)
    d = Derivative(dt=0.5)(x)
    gain = Parameter("ddynamic_further_gain", value=3.0)
    model = ModelyLoop(
        "ddynamic_further",
        inputs=[x],
        outputs=[
            Output(
                "ddynamic_further_out",
                Sin()([d]) * gain + d * 2.0 + Integrate(dt=0.5)(d),
            )
        ],
    ).build()

    for steps in (3, 7):
        result = to_numpy(
            model({"ddynamic_further_x": np.ones((2, steps, 1, 1), dtype=np.float32)})[
                "ddynamic_further_out"
            ]
        )
        # From zero, a constant 1 jumps once: d = [2, 0, 0, ...], and its
        # integral is the signal again.
        d_values = np.zeros(steps)
        d_values[0] = 2.0
        assert result.shape == (2, steps, 1, 1)
        np.testing.assert_allclose(
            result[1].ravel(),
            3.0 * np.sin(d_values) + 2.0 * d_values + 1.0,
            rtol=1e-5,
        )


@pytest.mark.slow
def test_derivative_is_trained_through():
    # v = Derivative(k * x): the gain is learnt from the derivative of the
    # trajectory.
    steps = 8
    x = Input("dtrained_x", sw=1)
    k = Parameter("dtrained_k", value=0.5)
    v = Output("dtrained_v", Derivative(dt=1.0)(x * k))
    model = ModelyLoop("dtrained", inputs=[x], outputs=[v], length=steps)
    model.minimize("fit", v, Input("dtrained_target", sw=1))
    model.build()
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.uniform(-1.0, 1.0, (64, steps)), axis=1).astype(
        np.float32
    )
    targets = 2.0 * np.diff(positions, axis=1, prepend=0.0)
    data = DataLoader(
        model,
        source=[
            {"dtrained_x": positions[i], "dtrained_target": targets[i]}
            for i in range(len(positions))
        ],
    )
    assert len(data) == len(positions)  # one trajectory per simulation

    model.train(data, epochs=200, batch_size=16, lr=0.05, printer=None)

    np.testing.assert_allclose(to_numpy(k.param).ravel(), [2.0], atol=1e-2)


def _saved_derivative_model(name, length):
    # A second derivative from a two-sample history: every argument the
    # layers save.
    x, history = Input(f"{name}_x", sw=1), Input(f"{name}_history", sw=2)
    out = Derivative(order=2, dt=0.5, init=history)(x)
    return ModelyLoop(
        name,
        inputs=[x, history],
        outputs=[Output(f"{name}_out", out)],
        length=length,
    ).build()


def _saved_derivative_data(name, steps):
    return {
        f"{name}_x": np.arange(1.0, steps + 1, dtype=np.float32).reshape(1, steps, 1, 1)
        ** 3,
        f"{name}_history": _over_steps(np.array([-1.0, 0.0]).reshape(1, 2, 1), steps),
    }


def test_derivative_round_trips_through_save_and_the_keras_file(tmp_path):
    model = _saved_derivative_model("dsaved", length=None)
    data = _saved_derivative_data("dsaved", 5)
    expected = to_numpy(model(data)["dsaved_out"])

    model.save(tmp_path / "dsaved")
    restored = Modely.load(tmp_path / "dsaved")
    np.testing.assert_allclose(
        to_numpy(restored(data)["dsaved_out"]), expected, rtol=1e-6
    )

    model.export_keras(tmp_path)
    exported = Modely.import_keras(str(tmp_path / "dsaved.keras"))
    assert isinstance(exported, keras.Model)
    np.testing.assert_allclose(
        to_numpy(exported(data)["dsaved_out"]),  # type: ignore
        expected,
        rtol=1e-6,
    )


@requires_onnx_export
def test_derivative_exports_to_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _saved_derivative_model("donnx", length=5)
    data = _saved_derivative_data("donnx", 5)
    model.export_onnx(tmp_path)

    actual = Modely.validate_onnx(tmp_path / "donnx.onnx", data, return_dict=True)
    np.testing.assert_allclose(
        to_numpy(actual["donnx_out"]),  # type: ignore
        to_numpy(model(data)["donnx_out"]),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="only the tensorflow exporter keeps the sequence axis dynamic",
)
@requires_onnx_export
def test_derivative_exports_to_onnx_following_the_horizon(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _saved_derivative_model("donnx_dynamic", length=None)
    model.export_onnx(tmp_path)

    for steps in (2, 6):
        data = _saved_derivative_data("donnx_dynamic", steps)
        actual = Modely.validate_onnx(
            tmp_path / "donnx_dynamic.onnx", data, return_dict=True
        )["donnx_dynamic_out"]  # type: ignore
        expected = to_numpy(model(data)["donnx_dynamic_out"])
        assert expected.shape == (1, steps, 1, 1)
        np.testing.assert_allclose(to_numpy(actual), expected, rtol=1e-5, atol=1e-5)


def test_derivative_rejects_what_it_cannot_differentiate():
    # The signal is one step of a ModelyLoop: one sample, no sequence.
    with pytest.raises(ValueError, match="carries a sequence"):
        Derivative(dt=0.1)(Input("dwith_seq", seq=-1, sw=1))
    with pytest.raises(ValueError, match="has none"):
        Derivative(dt=0.1)(Input("dno_window"))
    with pytest.raises(ValueError, match="one sample per step"):
        Derivative(dt=0.1)(Input("dwindowed", sw=3))
    with pytest.raises(ValueError, match="requires dt"):
        Derivative()
    with pytest.raises(ValueError, match="supports order"):
        Derivative(order=3, dt=0.1)
    with pytest.raises(ValueError, match="reads at least"):
        Derivative(order=2, dt=0.1, window=2)
    signal = Input("dchecked_signal", dim=2, sw=1)
    with pytest.raises(ValueError, match="init must carry"):
        Derivative(dt=0.1, init=Input("dlong_init", dim=2, sw=3))(signal)
    with pytest.raises(ValueError, match="init has dim"):
        Derivative(dt=0.1, init=Input("dwide_init", dim=3, sw=1))(signal)
    with pytest.raises(ValueError, match="init has the sequence"):
        Derivative(dt=0.1, init=Input("dseq_init", dim=2, seq=4, sw=1))(signal)


def test_my_localmodel():
    x = Input("x_local", dim=(2, 2), sw=4)
    g = Input("g_local", dim=1, sw=1)
    k = Input("k_local", dim=1, sw=1)
    mu = Fuzzify(centers=[0.0, 1.0, 2.0], function=Triangular)(g)
    mu_k = Fuzzify(centers=[0.0, 1.0], function=Triangular)(k)
    local = LocalModel(
        Linear(features=6), Fir(features=1), name="local_model", vectorized=True
    )
    local_output = local([x], [mu, mu_k])

    model = Modely(
        "local_model_test",
        inputs=[x, g, k],
        outputs=[Output("local_output", local_output)],
    ).build()

    dummy_x = np.ones([1, 4, 2, 2]).astype(np.float32)
    # g and k are read with last(): one sample each, the latest.
    dummy_g = np.array([1, 1, 2, 3]).reshape([1, 4, 1])[:, -1:].astype(np.float32)
    dummy_k = np.array([0, 1, 0, 1]).reshape([1, 4, 1])[:, -1:].astype(np.float32)
    result = model({"x_local": dummy_x, "g_local": dummy_g, "k_local": dummy_k})
    assert result["local_output"].shape == (1, 1, 1)


def test_fuzzify_rectangles_hold_every_value_once():
    # Each rectangle reaches halfway to the centers next to it, half-open:
    # once, on uneven centers, a rectangle reached halfway to the right on
    # both sides, two sets held 0.25, and the edges 0.5 and 1.5 held none.
    x = Input("boxes_x", sw=1)
    even = Fuzzify([0.0, 1.0, 2.0], function=Rectangular)([x])
    uneven = Fuzzify([0.0, 1.0, 3.0], function=Rectangular)([x])
    moving = Fuzzify(
        Parameter("boxes_centers", value=[0.0, 1.0, 3.0]), function=Rectangular
    )([x])
    model = Modely(
        "boxes",
        inputs=[x],
        outputs=[
            Output("boxes_even", even),
            Output("boxes_uneven", uneven),
            Output("boxes_moving", moving),
        ],
    ).build()

    samples = np.array([-0.5, 0.25, 0.5, 1.5, 1.99, 3.49], dtype=np.float32)
    result = model({"boxes_x": samples.reshape(-1, 1, 1)})
    np.testing.assert_array_equal(
        to_numpy(result["boxes_even"])[:, 0],
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]],
    )
    # Uneven: [-0.5, 0.5), [0.5, 2), [2, 4).
    for name in ("boxes_uneven", "boxes_moving"):
        np.testing.assert_array_equal(
            to_numpy(result[name])[:, 0],
            [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
        )


def test_local_model_refuses_a_layer_class():
    x = Input("class_cells_x", sw=1)
    mu = Fuzzify([0.0, 1.0], function=Triangular)([x])
    with pytest.raises(TypeError, match="pass a layer instance"):
        LocalModel(Tanh)([x], [mu])


def test_equation_learner_called_twice_applies_one_projection():
    # Its default input projection was made anew at every call, so two calls
    # on one stream gave different results.
    x = Input("twice_eql_x", sw=1)
    learner = EquationLearner(["sin", "multiply"], linear_out=Linear(features=1))
    first, second = learner([x]), learner([x])
    model = Modely(
        "twice_eql",
        inputs=[x],
        outputs=[Output("twice_eql_1", first), Output("twice_eql_2", second)],
    ).build()
    assert len(model.parameters) == 4  # one projection in, one out
    result = model({"twice_eql_x": np.full((2, 1, 1), 0.7, dtype=np.float32)})
    np.testing.assert_allclose(
        to_numpy(result["twice_eql_1"]), to_numpy(result["twice_eql_2"])
    )

    # On a dynamic sequence too.
    s = Input("twice_eql_s", seq=-1, sw=1)
    assert learner([s]).shape.dimensions[:2] == ((None,), 1)


def test_local_model_with_an_elementwise_function_on_several_inputs():
    # The function is applied to each input, and a cell is one stream.
    x, y = Input("several_cells_x", sw=1), Input("several_cells_y", sw=1)
    mu = Fuzzify([0.0, 1.0], function=Triangular)([x])
    with pytest.raises(ValueError, match="a cell is one stream"):
        LocalModel(Tanh())([x, y], [mu])


@pytest.mark.parametrize("call_first", [True, False])
def test_a_block_shares_an_input_it_does_not_declare_with_its_caller(call_first):
    # A call once saw only what its caller had inlined before it: reached
    # first, it made an Input of its own, and the model read two Inputs x.
    x, u, r = Input("shared_x", sw=1), Input("shared_u", sw=1), Input("shared_r", sw=1)
    block = Modely("shared_block", inputs=[r], outputs=[Output("shared_bo", x + r)])
    y = block([u]) + x if call_first else x + block([u])
    model = Modely("shared_model", inputs=[x, u], outputs=[Output("shared_y", y)])
    model.build()
    result = model(
        {
            "shared_x": np.ones((1, 1, 1), np.float32),
            "shared_u": np.full((1, 1, 1), 10.0, np.float32),
        }
    )
    np.testing.assert_allclose(to_numpy(result["shared_y"]).ravel(), [12.0])
