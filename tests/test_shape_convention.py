"""Every stream is laid out ``(batch, *dim, time, *seq)``.

A Parameter or a Constant is no exception: its value is repeated for every
sample of the batch, so the layers that slice, project or combine streams
read it the way they read an Input.
"""

import numpy as np

from conftest import to_numpy
from nnodely import (
    Concatenate,
    Constant,
    Exp,
    Fir,
    Input,
    Linear,
    Modely,
    Output,
    Parameter,
    Range,
    Select,
    Sum,
    TimeConcatenate,
    TimeRange,
    TimeSelect,
)

BATCH = 3


def _evaluate(name, streams, x):
    """The streams of a model over a batch of three samples of ``x``, which
    gives the model its batch even where a stream does not read it."""
    outputs = [Output(f"{name}_x", x.last())]
    outputs += [Output(key, stream) for key, stream in streams.items()]
    model = Modely(name, inputs=[x], outputs=outputs).build()
    values = np.arange(BATCH, dtype=np.float32).reshape(BATCH, 1, 1)
    result = model({x.name: values})
    return model, {key: to_numpy(result[key]) for key in streams}


def _repeated(sample):
    """``sample`` once per sample of the batch."""
    sample = np.asarray(sample, dtype=np.float32)
    return np.broadcast_to(sample, (BATCH, *sample.shape))


def test_a_parameter_and_a_constant_are_repeated_for_every_sample():
    x = Input("repeat_x")
    parameter = Parameter("repeat_parameter", value=[[1.0], [2.0], [3.0]])
    constant = Constant("repeat_constant", value=[[4.0, 5.0]])

    _, result = _evaluate("repeat", {"parameter": parameter, "constant": constant}, x)

    np.testing.assert_allclose(result["parameter"], _repeated([[1.0], [2.0], [3.0]]))
    np.testing.assert_allclose(result["constant"], _repeated([[4.0, 5.0]]))


def test_dim_slicing_of_a_parameter_selects_along_its_dim_axis():
    x = Input("dim_slice_x")
    parameter = Parameter("dim_slice_parameter", value=[[1.0], [2.0], [3.0]])

    _, result = _evaluate(
        "dim_slice",
        {
            "select": Select(idx=1)([parameter]),
            "range": Range(start=0, end=2)([parameter]),
            "sum": Sum()([parameter]),
        },
        x,
    )

    np.testing.assert_allclose(result["select"], _repeated([[2.0]]))
    np.testing.assert_allclose(result["range"], _repeated([[1.0], [2.0]]))
    np.testing.assert_allclose(result["sum"], _repeated([[6.0]]))


def test_time_slicing_of_a_constant_selects_along_its_time_axis():
    x = Input("time_slice_x")
    constant = Constant("time_slice_constant", value=[[1.0, 2.0, 3.0, 4.0]])

    _, result = _evaluate(
        "time_slice",
        {
            "last": TimeSelect(idx=-1)([constant]),
            "middle": TimeRange(start=1, end=3)([constant]),
        },
        x,
    )

    np.testing.assert_allclose(result["last"], _repeated([[4.0]]))
    np.testing.assert_allclose(result["middle"], _repeated([[2.0, 3.0]]))


def test_concatenation_joins_a_parameter_with_an_input_sample_by_sample():
    x = Input("join_x")
    parameter = Parameter("join_parameter", value=[[1.0], [2.0]])
    window = Constant("join_window", value=[[5.0, 6.0]])

    _, result = _evaluate(
        "join",
        {
            "dim": Concatenate()([x.last(), parameter]),
            "time": TimeConcatenate()([x.last(), window]),
        },
        x,
    )

    samples = np.arange(BATCH, dtype=np.float32)
    np.testing.assert_allclose(
        result["dim"], [[[value], [1.0], [2.0]] for value in samples]
    )
    np.testing.assert_allclose(
        result["time"], [[[value, 5.0, 6.0]] for value in samples]
    )


def test_linear_and_fir_project_a_parameter_like_an_input():
    x = Input("project_x")
    vector = Parameter("project_vector", value=[[1.0], [2.0], [3.0]])
    window = Parameter("project_window", value=[[1.0, 2.0, 3.0, 4.0]])
    linear = Linear(out_features=2, use_bias=False, name="project_linear")([vector])
    fir = Fir(out_features=1, use_bias=False, name="project_fir")([window])
    exp = Exp()([vector])

    model, _ = _evaluate("project", {"linear": linear, "fir": fir, "exp": exp}, x)
    assert linear.kernel is not None and fir.kernel is not None
    linear.kernel.assign(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "float32"))
    fir.kernel.assign(np.ones((4, 1), dtype="float32"))
    values = np.zeros((BATCH, 1, 1), dtype=np.float32)
    result = {
        key: to_numpy(value) for key, value in model({"project_x": values}).items()
    }

    # [1, 2, 3] -> [1 + 3, 2 + 3]; the window sums to 10.
    np.testing.assert_allclose(result["linear"], _repeated([[4.0], [5.0]]))
    np.testing.assert_allclose(result["fir"], _repeated([[10.0]]))
    np.testing.assert_allclose(result["exp"], _repeated(np.exp([[1.0], [2.0], [3.0]])))


def test_division_and_power_keep_the_samples_apart():
    a = Input("apart_a", seq=4)
    b = Input("apart_b")
    quotient = a.last() / b.last()
    power = a.last() ** b.last()
    halved = a.last() / 2.0
    model = Modely(
        "apart",
        inputs=[a, b],
        outputs=[
            Output("quotient", quotient),
            Output("power", power),
            Output("halved", halved),
        ],
    ).build()

    a_values = np.arange(1, 1 + BATCH * 4, dtype=np.float32).reshape(BATCH, 1, 1, 4)
    b_values = np.array([1.0, 2.0, 4.0], dtype=np.float32).reshape(BATCH, 1, 1)
    result = model({"apart_a": a_values, "apart_b": b_values})

    divisor = b_values[..., np.newaxis]  # one per sample, along its sequence
    for name, expected in (
        ("quotient", a_values / divisor),
        ("power", a_values**divisor),
        ("halved", a_values / 2.0),
    ):
        value = to_numpy(result[name])
        assert value.shape == (BATCH, 1, 1, 4), name
        np.testing.assert_allclose(value, expected, rtol=1e-5, err_msg=name)


def test_a_dynamic_sequence_axis_survives_shape_inference():
    # A layer whose shape is probed with dummy zeros - elementwise, arithmetic,
    # a Parameter, a projection - keeps a dynamic sequence axis dynamic.
    x = Input("dynamic_shape_x", dim=2, seq=-1)
    gain = Parameter("dynamic_shape_gain", value=2.0)
    streams = {
        "scaled": x * 3.0,
        "gained": x * gain,
        "squashed": Exp()([x]),
        "projected": Linear(out_features=4)([x]),
    }
    for stream in streams.values():
        assert stream.seq == (None,)
    model = Modely(
        "dynamic_shape",
        inputs=[x],
        outputs=[
            Output(f"dynamic_shape_{name}", stream) for name, stream in streams.items()
        ],
    ).build()

    for steps in (2, 5):
        values = np.full((1, 2, 1, steps), 0.5, dtype=np.float32)
        result = model({"dynamic_shape_x": values})
        np.testing.assert_allclose(
            to_numpy(result["dynamic_shape_scaled"]), 1.5 * np.ones_like(values)
        )
        np.testing.assert_allclose(
            to_numpy(result["dynamic_shape_gained"]), np.ones_like(values)
        )
        np.testing.assert_allclose(
            to_numpy(result["dynamic_shape_squashed"]), np.exp(values), rtol=1e-6
        )
        assert to_numpy(result["dynamic_shape_projected"]).shape == (1, 4, 1, steps)


def test_only_the_dynamic_sequence_axes_become_dynamic_again():
    # A fixed sequence axis next to a dynamic one keeps its size.
    x = Input("mixed_seq_x", seq=(3, -1))
    doubled = x * 2.0
    assert doubled.seq == (3, None)
    fixed = Input("fixed_seq_x", seq=4) * 2.0
    assert fixed.seq == (4,)
