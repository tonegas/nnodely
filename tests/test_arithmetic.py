import math

from nnodely import Constant, Gradient, Input, Modely, Output, Parameter
from nnodely.core.layer import Mod
from nnodely.core.shape import Shape
from nnodely.layers.arithmetic import (
    Abs,
    Ceil,
    Clamp,
    Cross,
    Deg2Rad,
    Exp,
    Expm1,
    Floor,
    Log,
    Log1p,
    Log10,
    Log2,
    Max,
    Maximum,
    Mean,
    Min,
    Minimum,
    Norm,
    Rad2Deg,
    Round,
    Sign,
    Sqrt,
    Square,
    Sum,
    SumImpl,
    Negative,
)
from nnodely.layers.trigonometric import Acosh, Asinh, Atanh, Cosh, Sinh
from conftest import to_numpy
import keras
import numpy as np
import pytest


@pytest.mark.parametrize(
    "layer_class, reference, low, high",
    [
        (Exp, np.exp, -2.0, 2.0),
        (Log, np.log, 0.5, 5.0),
        (Log10, np.log10, 0.5, 5.0),
        (Sqrt, np.sqrt, 0.5, 5.0),
        (Abs, np.abs, -3.0, 3.0),
        (Floor, np.floor, -3.0, 3.0),
        (Ceil, np.ceil, -3.0, 3.0),
        (Sign, np.sign, -3.0, 3.0),
        (Negative, np.negative, -3.0, 3.0),
        (Deg2Rad, np.deg2rad, -180.0, 180.0),
        (Expm1, np.expm1, -2.0, 2.0),
        (Log1p, np.log1p, 0.0, 5.0),
        (Log2, np.log2, 0.5, 5.0),
        (Square, np.square, -3.0, 3.0),
        (Round, np.round, -3.0, 3.0),
        (Rad2Deg, np.rad2deg, -np.pi, np.pi),
        (Sinh, np.sinh, -2.0, 2.0),
        (Cosh, np.cosh, -2.0, 2.0),
        (Asinh, np.arcsinh, -3.0, 3.0),
        (Acosh, np.arccosh, 1.0, 5.0),
        (Atanh, np.arctanh, -0.9, 0.9),
    ],
)
def test_arithmetic(layer_class, reference, low, high):
    # ------- Test arithmetic layers -------
    x = Input("x", dim=1, tw=3)
    window = 3
    out = Output("out", layer_class()([x]))
    model = Modely(
        f"arithmetic_{layer_class.__name__.lower()}_model",
        inputs=[x],
        outputs=[out],
    )
    model.build()

    # ------- Model inference -------
    batch_size = 1
    dummy_input_x = np.random.uniform(low, high, size=(batch_size, window, 1))
    result = model({"x": dummy_input_x})
    assert "out" in result
    assert result["out"].shape == (batch_size, window, 1)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        reference(dummy_input_x),
        rtol=1e-5,
        atol=1e-5,
    )


def test_arithmetic_chain():
    # ------- Test chained arithmetic layers -------
    x = Input("x", dim=1, tw=2)
    out = Output("out", Log(name="log")(Abs()(Exp()([x]))))
    model = Modely("arithmetic_chain_model", inputs=[x], outputs=[out])
    model.build()

    dummy_input_x = np.random.uniform(-2.0, 2.0, size=(1, 2, 1))
    np.testing.assert_allclose(
        to_numpy(model({"x": dummy_input_x})["out"]),
        np.log(np.abs(np.exp(dummy_input_x))),
        rtol=1e-5,
        atol=1e-5,
    )


def test_unary_minus():
    # ------- Test -x as Negative -------
    x = Input("x", dim=2, tw=3)
    negated = -x
    assert isinstance(negated, Negative)
    assert negated.dimensions == x.dimensions
    out = Output("out", -(x * 2.0) + x)
    model = Modely("unary_minus_model", inputs=[x], outputs=[out])
    model.build()

    dummy_input_x = np.random.uniform(-2.0, 2.0, size=(1, 3, 2))
    np.testing.assert_allclose(
        to_numpy(model({"x": dummy_input_x})["out"]),
        -dummy_input_x,
        rtol=1e-5,
        atol=1e-5,
    )


def test_stream_operators():
    # ------- Test 2 ** x, x % y, abs(x) and +x -------
    x = Input("x", dim=2, tw=3)
    y = Input("y", dim=2, tw=3)
    assert +x is x
    assert isinstance(abs(x), Abs)
    assert isinstance(x % y, Mod)
    relations = {
        "rpow": 2.0**x,
        "mod": x % y,
        "mod_number": x % 1.5,
        "rmod": 5.0 % y,
        "abs": abs(x),
    }
    model = Modely(
        "arithmetic_operators_model",
        inputs=[x, y],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()

    rng = np.random.default_rng(0)
    xs = rng.uniform(-3.0, 3.0, size=(4, 3, 2)).astype(np.float32)
    ys = (
        rng.uniform(0.5, 2.0, size=(4, 3, 2)) * rng.choice([-1, 1], (4, 3, 2))
    ).astype(np.float32)
    result = model({"x": xs, "y": ys})
    # % takes the sign of the divisor, as Python's does.
    expected = {
        "rpow": 2.0**xs,
        "mod": np.mod(xs, ys),
        "mod_number": np.mod(xs, 1.5),
        "rmod": np.mod(5.0, ys),
        "abs": np.abs(xs),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(
            to_numpy(result[name]), value, rtol=1e-5, atol=1e-5, err_msg=name
        )


def test_maximum_minimum():
    # ------- Test elementwise extremum of streams -------
    x = Input("x", dim=2, tw=3)
    y = Input("y", dim=2)  # no window, broadcast along the window
    bound = Parameter("bound", value=[0.5, -0.5])
    largest = Maximum([x, y, bound])
    smallest = Minimum()([x, bound])
    assert isinstance(largest, Maximum)
    assert largest.dimensions == ((), 3, (2,))
    model = Modely(
        "arithmetic_extremum_model",
        inputs=[x, y],
        outputs=[Output("largest", largest), Output("smallest", smallest)],
    ).build()

    rng = np.random.default_rng(0)
    xs = rng.uniform(-2.0, 2.0, size=(4, 3, 2))
    ys = rng.uniform(-2.0, 2.0, size=(4, 2))
    result = model({"x": xs, "y": ys})
    np.testing.assert_allclose(
        to_numpy(result["largest"]),
        np.maximum(np.maximum(xs, ys[:, None]), [0.5, -0.5]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(result["smallest"]), np.minimum(xs, [0.5, -0.5]), rtol=1e-6
    )

    # One stream is a reduction, not a comparison.
    with pytest.raises(ValueError, match=r"two or more streams.*Max\(\)\(x\)"):
        Maximum(x)
    with pytest.raises(ValueError, match=r"two or more streams.*Min\(\)\(x\)"):
        Minimum()([x])


@pytest.mark.parametrize(
    "layer_class, reference",
    [
        (Mean, np.mean),
        (Max, np.max),
        (Min, np.min),
        (
            Norm,
            lambda v, axis, keepdims: np.sqrt(
                np.sum(v**2, axis=axis, keepdims=keepdims)
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "axis, reference_axis, expected_dim",
    [
        (None, (2, 3), (1, 1)),
        (0, 2, (1, 3)),
        (-1, 3, (2, 1)),
    ],
)
def test_reductions(layer_class, reference, axis, reference_axis, expected_dim):
    # ------- Test reductions over the dim axes -------
    x = Input("x", dim=(2, 3), tw=4)
    out = Output("out", layer_class(axis=axis)([x]))
    model = Modely("arithmetic_reduction_model", inputs=[x], outputs=[out])
    model.build()

    dummy_input_x = np.random.uniform(-2.0, 2.0, size=(5, 4, 2, 3))
    result = model({"x": dummy_input_x})
    assert result["out"].shape == (5, 4, *expected_dim)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        reference(dummy_input_x, axis=reference_axis, keepdims=True),
        rtol=1e-5,
        atol=1e-5,
    )


def test_norm_gradient():
    # ------- Test the gradient of the norm, 0 at the zero vector -------
    x = Input("x", dim=3, tw=1)
    out = Output("out", Gradient(wrt=x)(Norm()(x)))
    model = Modely("arithmetic_norm_gradient_model", inputs=[x], outputs=[out])
    model.build()

    vectors = np.array([[[3.0, 4.0, 0.0]], [[0.0, 0.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        to_numpy(model({"x": vectors})["out"]),
        [[[0.6, 0.8, 0.0]], [[0.0, 0.0, 0.0]]],
        rtol=1e-6,
    )


def test_cross():
    # ------- Test the cross product of 3-vectors -------
    a = Input("a", dim=3, tw=2)
    b = Input("b", dim=3, tw=2)
    omega = Parameter("omega", value=[0.0, 0.0, 2.0])
    m = Input("m", dim=(3, 2), tw=1)  # two vectors, along dim axis 0
    n = Input("n", dim=(3, 2), tw=1)
    relations = {
        "cross": Cross()([a, b]),
        "turned": Cross()([omega, a]),  # omega applies to every sample
        "axis": Cross(axis=0)([m, n]),
    }
    assert relations["turned"].dimensions == ((), 2, (3,))
    model = Modely(
        "arithmetic_cross_model",
        inputs=[a, b, m, n],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()

    rng = np.random.default_rng(0)
    data = {
        "a": rng.normal(size=(4, 2, 3)),
        "b": rng.normal(size=(4, 2, 3)),
        "m": rng.normal(size=(4, 1, 3, 2)),
        "n": rng.normal(size=(4, 1, 3, 2)),
    }
    result = model(data)
    expected = {
        "cross": np.cross(data["a"], data["b"]),
        "turned": np.cross([0.0, 0.0, 2.0], data["a"]),
        "axis": np.cross(data["m"], data["n"], axis=2),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(
            to_numpy(result[name]), value, rtol=1e-5, atol=1e-5, err_msg=name
        )

    with pytest.raises(ValueError, match="3-vectors"):
        Cross()([Input("c", dim=2, tw=2), Input("d", dim=2, tw=2)])
    with pytest.raises(ValueError, match="reads two streams"):
        Cross()([a])
    with pytest.raises(TypeError, match="configured first"):
        Cross([a, b])  # type: ignore


def test_math_layers_save_load_and_export(tmp_path):
    # ------- Test the math layers saved, loaded and exported to Keras -------
    x = Input("x", dim=3, tw=2)
    y = Input("y", dim=3, tw=2)
    bound = Parameter("bound", value=[0.1, 0.2, 0.3])
    relations = {
        "cross": Cross(name="cross_layer")([x, y]),
        "norm": Norm(axis=0, name="norm_layer")([x]),
        "reductions": Mean()([x]) + Max()([y]) - Min()([y]) + Sum()([x]),
        "extremum": Maximum([Minimum([x, y]), bound]),
        "unary": Rad2Deg(Round(Square(Sinh(x) + Cosh(y)))) + Log1p(abs(x)),
        "operators": 2.0**x % 1.5 - y,
    }
    model = Modely(
        "arithmetic_math_save_model",
        inputs=[x, y],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()

    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.uniform(-1.0, 1.0, size=(3, 2, 3)).astype(np.float32),
        "y": rng.uniform(-1.0, 1.0, size=(3, 2, 3)).astype(np.float32),
    }
    expected = model(inputs)

    model.save(tmp_path / "math")
    restored = Modely.load(tmp_path / "math")
    model.export_keras(tmp_path, "math")
    imported = Modely.import_keras(tmp_path / "math.keras")
    for result in (restored(inputs), imported(inputs)):  # type: ignore
        for name, value in expected.items():
            np.testing.assert_allclose(
                to_numpy(result[name]), to_numpy(value), rtol=1e-5, err_msg=name
            )
    nodes = {node.name: node for node in restored.flatten()._order}
    assert nodes["cross_layer"].get_config() == {"name": "cross_layer", "axis": -1}
    assert nodes["norm_layer"].get_config() == {"name": "norm_layer", "axis": 0}


def test_sum_impl_of_2_0_0_keras_files_still_loads():
    # ------- Keras files exported by 2.0.0 name the sum SumImpl -------
    assert keras.saving.get_registered_object("nnodely>SumImpl") is SumImpl
    layer = SumImpl.from_config({"name": "old_sum", "axes": [2]})
    values = np.random.uniform(-1.0, 1.0, size=(2, 3, 4))
    np.testing.assert_allclose(
        to_numpy(layer(values)), values.sum(axis=2, keepdims=True), rtol=1e-5
    )
    assert SumImpl.from_config(layer.get_config()).operation == "sum"


def test_arithmetic_save_load(tmp_path):
    # ------- Test arithmetic layer serialization -------
    x = Input("x", dim=1, tw=3)
    out = Output("out", Sqrt(name="sqrt")([x]))
    model = Modely("arithmetic_save_model", inputs=[x], outputs=[out])
    model.build()

    inputs = {"x": np.random.uniform(0.5, 5.0, size=(1, 3, 1))}
    expected = model(inputs)["out"]

    path = tmp_path / "arithmetic.nnodely"
    model.save(path)
    restored = Modely.load(path)
    np.testing.assert_allclose(
        to_numpy(restored(inputs)["out"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    restored_layer = next(
        node for node in restored.flatten()._order if isinstance(node, Sqrt)
    )
    assert restored_layer.get_config() == {"name": "sqrt"}


@pytest.mark.parametrize(
    "minimum, maximum",
    [(-1.0, 1.0), (-1.0, None), (None, 1.0)],
)
def test_clamp(minimum, maximum):
    # ------- Test clamp bounds -------
    x = Input("x", dim=1, tw=3)
    window = 3
    out = Output("out", Clamp(min=minimum, max=maximum)([x]))
    model = Modely("arithmetic_clamp_model", inputs=[x], outputs=[out])
    model.build()

    dummy_input_x = np.random.uniform(-3.0, 3.0, size=(1, window, 1))
    result = model({"x": dummy_input_x})
    assert result["out"].shape == (1, window, 1)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        np.clip(
            dummy_input_x,
            -np.inf if minimum is None else minimum,
            np.inf if maximum is None else maximum,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_clamp_invalid_bounds():
    # ------- Test clamp bounds validation -------
    with pytest.raises(ValueError):
        Clamp(min=1.0, max=-1.0)


@pytest.mark.parametrize(
    "axis, reference_axis, expected_dim",
    [
        (None, (2, 3), (1, 1)),
        (0, 2, (1, 3)),
        (1, 3, (2, 1)),
        (-1, 3, (2, 1)),
    ],
)
def test_sum(axis, reference_axis, expected_dim):
    # ------- Test sum over the dim axes -------
    x = Input("x", dim=(2, 3), tw=4)
    window = 4
    out = Output("out", Sum(axis=axis)([x]))
    model = Modely("arithmetic_sum_model", inputs=[x], outputs=[out])
    model.build()

    batch_size = 5
    dummy_input_x = np.random.uniform(-2.0, 2.0, size=(batch_size, window, 2, 3))
    result = model({"x": dummy_input_x})
    assert result["out"].shape == (batch_size, window, *expected_dim)
    np.testing.assert_allclose(
        to_numpy(result["out"]),
        dummy_input_x.sum(axis=reference_axis, keepdims=True),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sum_axis_out_of_bounds():
    # ------- Test sum axis validation -------
    x = Input("x", dim=(2, 3), tw=2)
    with pytest.raises(ValueError):
        Sum(axis=2)([x])


def test_clamp_sum_save_load(tmp_path):
    # ------- Test clamp and sum serialization -------
    x = Input("x", dim=(2, 3), tw=3)
    out = Output("out", Sum(name="sum")(Clamp(min=-0.5, max=0.5, name="clamp")([x])))
    model = Modely("arithmetic_clamp_sum_save_model", inputs=[x], outputs=[out])
    model.build()

    inputs = {"x": np.random.uniform(-2.0, 2.0, size=(1, 3, 2, 3))}
    expected = model(inputs)["out"]

    path = tmp_path / "arithmetic_clamp_sum.nnodely"
    model.save(path)
    restored = Modely.load(path)
    np.testing.assert_allclose(
        to_numpy(restored(inputs)["out"]),
        to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    nodes = {node.name: node for node in restored.flatten()._order}
    assert nodes["clamp"].get_config() == {
        "name": "clamp",
        "min": -0.5,
        "max": 0.5,
    }
    assert nodes["sum"].get_config() == {"name": "sum", "axis": None}


@pytest.mark.parametrize(
    "seq1, tw1, dim1, seq2, tw2, dim2, shape",
    [
        (1, 1, 1, 1, 1, 1, (1, 1, 1)),
        (1, 1, (2, 3), 1, 1, 1, (1, 1, 2, 3)),
        (
            1,
            1,
            (2, 3),
            1,
            1,
            2,
            None,
        ),  # dims lined up from the last axis, as Keras: 3 vs 2
        (1, 4, 1, 1, 4, 1, (1, 4, 1)),
        (1, 1, 1, 1, 4, 1, None),
        (1, 2, 1, 1, 4, 1, None),
        (1, 1, 1, 2, 1, 1, None),
        (3, 1, 1, 2, 1, 1, None),
        (3, 1, 1, 3, 1, 1, (3, 1, 1)),
        (1, 1, 1, 2, 3, 1, None),
        (2, 1, 5, 2, 3, 1, None),
        (1, 1, 1, 2, 3, 5, None),
        (1, 3, 1, 2, 3, 5, None),
        (None, 1, 1, None, 1, 1, (None, 1, 1)),
        (None, None, 1, None, 1, 1, (None, 1, 1)),
        (None, None, 3, None, 1, 1, None),
        (None, None, 1, None, 3, 1, (None, 3, 1)),  # broadcast along the window
        (
            None,
            None,
            2,
            None,
            3,
            (2, 3),
            None,
        ),  # dims lined up from the last axis, as Keras: 2 vs 3
        (
            None,
            None,
            2,
            None,
            3,
            (4, 2),
            (None, 3, 4, 2),
        ),  # broadcast along the dim axes
        (
            None,
            3,
            3,
            None,
            3,
            (3, 3),
            (None, 3, 3, 3),
        ),  # the vector along the last axis, as Keras
        (None, 3, (1, 3), None, 3, (3, 3), (None, 3, 3, 3)),  # an axis of 1 stretches
        (
            None,
            3,
            (2, 1),
            None,
            3,
            (1, 3),
            (None, 3, 2, 3),
        ),  # same window: dims broadcast both ways
        (
            None,
            None,
            (2, 1),
            None,
            3,
            (1, 3),
            None,
        ),  # each broadcast, one along the window too
        (3, 2, 1, None, 2, 1, (3, 2, 1)),  # broadcast along the sequence, same tw
        (3, 2, 1, None, 1, 1, None),  # broadcast along the sequence, different tw
        (3, 2, 1, None, None, 1, (3, 2, 1)),  # no tw: along the sequence and the window
        (None, 2, (2, 3), 4, 2, 3, None),  # each broadcast, one along the sequence
    ],
)
# The same policy for every stream: a Constant or a Parameter in place of an Input.
@pytest.mark.parametrize(
    "kind1, kind2",
    [
        ("input", "input"),
        ("constant", "input"),
        ("input", "constant"),
        ("parameter", "input"),
        ("input", "parameter"),
    ],
)
def test_binary_ops(seq1, tw1, dim1, seq2, tw2, dim2, shape, kind1, kind2):
    # ------- Test binary operations with broadcasting -------
    x = _operand(kind1, "x", seq1, tw1, dim1)
    y = _operand(kind2, "y", seq2, tw2, dim2)

    if shape is None:
        with pytest.raises(ValueError):
            relations = {
                "add": x + y,
                "sub": x - y,
                "mul": x * y,
                "div": x / y,
            }
    else:
        relations = {
            "add": x + y,
            "sub": x - y,
            "mul": x * y,
            "div": x / y,
        }
        inputs = [stream for stream in (x, y) if isinstance(stream, Input)]
        model = Modely(
            "arithmetic_binary_ops_model",
            inputs=inputs,
            outputs=[Output(name, relation) for name, relation in relations.items()],
        ).build()

        # Data laid out (batch, *seq, tw, *dim), as each Input declares it.
        data = {
            stream.name: np.random.uniform(-2.0, 2.0, size=(1, *stream.shape.tuple))
            for stream in inputs
        }
        result = model(data)
        # None in shape is an axis the result does not have.
        expected = (1, *(axis for axis in shape if axis is not None))
        for name in relations.keys():
            assert result[name].shape == expected


def _operand(kind, name, seq, tw, dim):
    """An operand of the sizes given: an Input, or a Constant or a Parameter
    holding random values."""
    if kind == "input":
        return Input(name, dim=dim, tw=tw, seq=seq)
    size = math.prod(Shape(dim=dim, tw=tw, seq=seq).tuple)
    value = np.random.uniform(-2.0, 2.0, size=size).tolist()
    value_class = Constant if kind == "constant" else Parameter
    return value_class(name, value=value, dim=dim, tw=tw, seq=seq)


def test_binary_ops_broadcast_the_dims_as_keras(tmp_path):
    # ------- Dims lined up from the last axis, an axis of 1 stretching -------
    x = Input("x", dim=2)  # no window: the same at every sample
    cols = Input("cols", dim=(4, 2), tw=3)
    row = Input("row", dim=3, tw=3)
    square = Input("square", dim=(3, 3), tw=3)
    column = Input("column", dim=(2, 1), tw=3)
    line = Input("line", dim=(1, 3), tw=3)
    matrix = Input("matrix", dim=(2, 3), tw=3)
    relations = {
        "cols": cols - x,
        "square": row * square,
        "both_ways": column + line,
        "cross": Cross()([row, matrix]),
    }
    assert relations["both_ways"].dimensions == ((), 3, (2, 3))
    assert relations["cross"].dimensions == ((), 3, (2, 3))
    model = Modely(
        "arithmetic_keras_dims_model",
        inputs=[x, cols, row, square, column, line, matrix],
        outputs=[Output(name, relation) for name, relation in relations.items()],
    ).build()

    rng = np.random.default_rng(0)
    data = {
        stream.name: rng.normal(size=(2, *stream.shape.tuple))
        for stream in (x, cols, row, square, column, line, matrix)
    }
    expected = {
        "cols": data["cols"] - data["x"][:, None, None, :],
        "square": data["row"][:, :, None, :] * data["square"],
        "both_ways": data["column"] + data["line"],
        "cross": np.cross(data["row"][:, :, None, :], data["matrix"]),
    }
    model.export_keras(tmp_path, "keras_dims")
    imported = Modely.import_keras(tmp_path / "keras_dims.keras")
    for result in (model(data), imported(data)):  # type: ignore
        for name, value in expected.items():
            np.testing.assert_allclose(
                to_numpy(result[name]), value, rtol=1e-5, atol=1e-6, err_msg=name
            )

    # Lined up from the last axis, 2 meets 3.
    with pytest.raises(ValueError, match="the same, or 1"):
        _ = x * matrix
    # Axis 0 of (3, 4) is axis 1 of the product with (3, 3, 4), not axis 0.
    with pytest.raises(ValueError, match="different axes"):
        Cross(axis=0)([Input("a", dim=(3, 4)), Input("b", dim=(3, 3, 4))])
