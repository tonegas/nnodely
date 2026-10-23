"""The layout of a stream in a tensor, as ``nnodely.core.shape`` owns it:
``(batch, *seq, time, *dim)`` - the sequence axes outermost first, where a
Loop rolls out, then the window, where temporal operations act, then the
features.

The first tests check what any layout must keep - every axis has one place, a
shape reads back from its tensor, a rollout lands on the outer axis - rather
than where the axes are. The last ones fix how streams of different shapes
combine.
"""

import numpy as np
import pytest

from conftest import to_numpy
from nnodely.core.shape import (
    BATCH_AXIS,
    OUTER_AXIS,
    Shape,
    expand_to_rank,
    outer_view,
    slice_axis,
    transpose,
)
from nnodely.layers.fir import Fir

SHAPES = [
    Shape(),
    Shape(dim=3, time=5),
    Shape(dim=(2, 3), time=4, seq=6),
    Shape(dim=(2,), time=1, seq=(3, 7)),
]


def _tensor(shape, batch=2) -> np.ndarray:
    """A tensor of ``shape`` whose every value is different."""
    sizes = (batch, *shape.tuple)
    return np.arange(np.prod(sizes), dtype=np.float32).reshape(sizes)


@pytest.mark.parametrize("shape", SHAPES, ids=repr)
def test_every_axis_of_a_shape_has_one_place_in_the_tensor(shape):
    axes = [BATCH_AXIS, *shape.seq_axes, shape.time_axis, *shape.dim_axes]
    assert sorted(axes) == list(range(1 + shape.rank))

    sizes = _tensor(shape).shape
    assert tuple(sizes[axis] for axis in shape.seq_axes) == shape.seq
    assert sizes[shape.time_axis] == shape.time
    assert tuple(sizes[axis] for axis in shape.dim_axes) == shape.dim


@pytest.mark.parametrize("shape", SHAPES, ids=repr)
def test_the_window_follows_the_sequence_and_the_outer_axis_comes_first(shape):
    sizes = _tensor(shape).shape
    assert shape.time_axis == 1 + shape.seq_rank
    assert sizes[OUTER_AXIS] == shape.outer
    assert shape.outer == (shape.seq[0] if shape.seq else shape.time)


def test_a_stream_has_a_window_of_at_least_one_sample():
    ## One sample by default, and no sequence
    shape = Shape(dim=2)
    assert shape.dimensions == ((), 1, (2,))
    assert shape.time_axis == 1 and shape.seq_axes == ()
    ## The window is one axis, its length an int: the others are seq
    with pytest.raises(TypeError, match="length of the window"):
        Shape(dim=2, time=(5, 4))
    with pytest.raises(TypeError, match="length of the window"):
        Shape(dim=2, time=0)


def test_seq_is_a_tuple_and_an_int_one_axis():
    assert Shape(dim=2, seq=3).seq == (3,)
    assert Shape(dim=2, seq=(5, 4)).seq == (5, 4)
    assert Shape(dim=2).seq == ()
    assert repr(Shape(dim=4, time=3, seq=5)) == "Shape(S=(5,), T=3, D=(4,))"
    assert repr(Shape()) == "Shape(S=(), T=1, D=(1,))"


@pytest.mark.parametrize("shape", SHAPES, ids=repr)
def test_a_shape_reads_back_from_its_tensor(shape):
    sizes = _tensor(shape).shape[1:]
    by_seq = Shape.from_tensor_shape(sizes, seq_rank=shape.seq_rank)
    by_dim = Shape.from_tensor_shape(sizes, dim_rank=shape.dim_rank)
    assert by_seq.dimensions == by_dim.dimensions == shape.dimensions


def test_an_axis_counts_from_the_end_and_is_checked():
    shape = Shape(dim=(2, 3), time=4, seq=(5, 6))
    assert shape.resolve_dim(-1, "layer") == 1
    assert shape.resolve_seq(-1, "layer") == 1  # the innermost sequence axis
    with pytest.raises(ValueError, match="layer: axis 2 out of bounds for dim rank 2"):
        shape.resolve_dim(2, "layer")
    with pytest.raises(ValueError, match="layer: axis -3 out of bounds for seq rank 2"):
        shape.resolve_seq(-3, "layer")


@pytest.mark.parametrize("shape", SHAPES, ids=repr)
def test_a_rollout_is_a_new_outer_sequence_axis(shape):
    rolled = shape.with_seq(9)
    assert rolled.seq == (9, *shape.seq)
    assert (rolled.time, rolled.dim) == (shape.time, shape.dim)
    assert rolled.outer == 9
    assert _tensor(rolled).shape[OUTER_AXIS] == 9
    assert rolled.without_outer_seq().dimensions == shape.dimensions


def test_without_a_sequence_the_outer_axis_is_the_window():
    shape = Shape(dim=3, time=5)
    assert shape.outer == shape.time == 5
    assert shape.time_axis == OUTER_AXIS


def test_replace_changes_only_the_axes_given():
    shape = Shape(dim=(2, 3), time=4, seq=5)
    assert shape.replace(time=1).dimensions == ((5,), 1, (2, 3))
    assert shape.replace(dim=4).dimensions == ((5,), 4, (4,))
    assert shape.replace(seq=()).dimensions == ((), 4, (2, 3))
    assert shape.dimensions == ((5,), 4, (2, 3))


def test_slice_axis_indexes_one_axis():
    shape = Shape(dim=(2, 3), time=4, seq=5)
    x = _tensor(shape)
    window = slice_axis(x, shape.time_axis, slice(1, 3))
    assert window.shape[shape.time_axis] == 2
    np.testing.assert_array_equal(
        np.take(window, 0, axis=shape.time_axis),
        np.take(x, 1, axis=shape.time_axis),
    )
    # An integer drops the axis.
    assert slice_axis(x, OUTER_AXIS, -1).ndim == x.ndim - 1


def test_a_permutation_and_its_inverse_give_the_tensor_back():
    x = _tensor(Shape(dim=(2, 3), time=4, seq=5))
    permutation = [0, 3, 4, 1, 2]
    back = transpose(to_numpy(transpose(x, permutation)), np.argsort(permutation))
    np.testing.assert_array_equal(to_numpy(back), x)
    assert transpose(x, range(x.ndim)) is x


def test_an_operand_without_rollout_broadcasts_along_it():
    # A value of one step, like a Constant, against a stream rolled out by a Loop.
    step = Shape(dim=(2,), time=3)
    stream = step.with_seq(4)
    value = _tensor(step)
    expanded = expand_to_rank(value, 1 + stream.rank, np.expand_dims)
    tiled = np.broadcast_to(expanded, _tensor(stream).shape)
    for index in range(4):
        np.testing.assert_array_equal(slice_axis(tiled, OUTER_AXIS, index), value)


def test_a_rollout_mask_runs_along_the_outer_axis():
    stream = Shape(dim=(2, 3), time=4).with_seq(5)
    mask = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=bool)
    view = np.broadcast_to(
        outer_view(mask, 1 + stream.rank, np.expand_dims), _tensor(stream).shape
    )
    for step in range(5):
        np.testing.assert_array_equal(
            slice_axis(view, OUTER_AXIS, step).reshape(2, -1).all(axis=1),
            mask[:, step],
        )


def test_a_value_names_its_sequence_and_window_axes():
    from nnodely import Parameter

    value = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    ## Axis 2 is the sequence, axis 1 the window, axis 0 the dim
    parameter = Parameter("p_time", value=value, dim=0, time=1, seq=2)
    assert parameter.shape.dimensions == ((2,), 3, (2,))
    np.testing.assert_array_equal(parameter.value, np.transpose(value, (2, 1, 0)))
    # Without a value, seq and time give the sizes.
    sized = Parameter("q_time", dim=3, time=4, seq=5)
    assert sized.shape.dimensions == ((5,), 4, (3,))
    # Without them, a value is one sample.
    assert Parameter("r_time", value=[1.0, 2.0]).shape.dimensions == ((), 1, (2,))
    # The window is one axis of the value.
    with pytest.raises(TypeError, match="one axis of the value"):
        Parameter("s_time", value=value, time=(2, 1))


def test_sanity_check():
    from nnodely import Input, Output, Modely

    # this is allowed: you can change the time (as it always does)
    x = Input("x", dim=1)
    window = x.sw(3)
    fir = Fir(features=1)(window)
    other_window = x.sw([2, 4])
    fir2 = Fir(features=1)(other_window)
    assert window.shape.tuple == (3, 1)
    assert other_window.shape.tuple == (6, 1)
    assert fir.shape.tuple == fir2.shape.tuple == (1, 1)

    # the sequence belongs to the Input: every window of it keeps it
    x = Input("x", dim=(2, 3), seq=5)
    window = x.sw(3)
    fir = Fir(features=1)(window)
    assert x.shape.tuple == (5, 1, 2, 3)
    assert window.shape.tuple == (5, 3, 2, 3)
    assert fir.shape.tuple == (5, 1, 2, 3)
    assert x.sw(2).shape.tuple == (5, 2, 2, 3)

    out = Output("out", fir)
    model = Modely("sanity", inputs=[x], outputs=[out]).build()
    ## after the build the correct time shape should be collected for every input,
    # otherwise i cannot save the model with the correct dimensions
    assert x.shape.tuple == (5, 3, 2, 3)
    assert model.inputs[0].shape.tuple == (5, 3, 2, 3)

    # the sequence belongs to the Input: every window of it keeps it
    x = Input("x", dim=(2, 3), seq=5)
    window = x.sw(3)
    fir = Fir(features=1)(x)
    assert x.shape.tuple == (5, 1, 2, 3)
    assert window.shape.tuple == (5, 3, 2, 3)
    assert fir.shape.tuple == (5, 1, 2, 3)
    assert x.sw(2).shape.tuple == (5, 2, 2, 3)

    out = Output("out", fir)
    model = Modely("sanity", inputs=[x], outputs=[out]).build()
    ## after the build the correct time shape should be 1 because the sw(3) does not belong to the keras model graph,
    # no operation has sw(3) inside the predecessors.
    assert x.shape.tuple == (5, 1, 2, 3)
    assert model.inputs[0].shape.tuple == (5, 1, 2, 3)


## New test, for the convention [B, S*, T, D*] of the stream tensor
## Test the math with the different shapes and axis
## Test the broadcasting
def _check(allowed, refused):
    """Each allowed pair has the expected shape, whichever comes first; each
    refused pair raises."""
    for left, right, expected in allowed:
        assert (left + right).shape.tuple == expected, (left.name, right.name)
        assert (right + left).shape.tuple == expected, (right.name, left.name)
    for left, right in refused:
        with pytest.raises(ValueError):
            _ = left + right
        with pytest.raises(ValueError):
            _ = right + left


def test_arithmetic_shape():
    from nnodely import Input

    ## Time is always at least 1: read directly, an Input is one sample
    x1 = Input("x1", dim=1)  # shape [B, 1, 1]
    x2 = Input("x2", dim=(4,))  # shape [B, 1, 4]
    x3 = Input("x3", dim=(2, 3))  # shape [B, 1, 2, 3]

    ## .sw() reads a sliding window, which sets the time
    x1t = Input("x1t", dim=1).sw(2)  # shape [B, 2, 1]
    x2t = Input("x2t", dim=(4,)).sw(2)  # shape [B, 2, 4]
    x3t = Input("x3t", dim=(2, 3)).sw(2)  # shape [B, 2, 2, 3]

    x1tm = Input("x1tm", dim=1).sw(3)  # shape [B, 3, 1]
    x2tm = Input("x2tm", dim=(4,)).sw(3)  # shape [B, 3, 4]
    x3tm = Input("x3tm", dim=(2, 3)).sw(3)  # shape [B, 3, 2, 3]

    ## Should be allowed: equal time
    allowed = [
        (x1, x1, (1, 1)),
        (x1, x2, (1, 4)),  # broadcast the scalar to the whole vector
        (x1, x3, (1, 2, 3)),  # broadcast the scalar to the whole matrix
        (x1t, x2t, (2, 4)),  # broadcast the scalar to the whole vector (same time)
        (x1t, x3t, (2, 2, 3)),  # broadcast the scalar to the whole matrix (same time)
        ## if time is 1 it can be broadcast to any time, but not the other way around
        (x1, x1t, (2, 1)),
        (x1, x2t, (2, 4)),  # broadcast the scalar to the whole vector
        (x1, x3t, (2, 2, 3)),  # broadcast the scalar to the whole matrix
        (x2, x2t, (2, 4)),
        (x1, x1tm, (3, 1)),
        (x1, x2tm, (3, 4)),  # broadcast the scalar to the whole vector
        (x1, x3tm, (3, 2, 3)),  # broadcast the scalar to the whole matrix
        (x2, x2tm, (3, 4)),
    ]

    ## should not be allowed
    refused = [
        (x2, x3),  # cannot broadcast the vector to the matrix
        (x2, x3t),  # cannot broadcast the vector to the matrix
        (x3, x2t),  # cannot broadcast the vector to the matrix
        (x2t, x3t),  # cannot broadcast the vector to the matrix
        (x1t, x1tm),  # different time cannot be allowed
        (x1t, x2tm),  # different time cannot be allowed
        (x1t, x3tm),  # different time cannot be allowed
        (x2t, x1tm),  # different time cannot be allowed
        (x2t, x2tm),  # different time cannot be allowed
        (x2t, x3tm),  # different time cannot be allowed
        (x3t, x1tm),  # different time cannot be allowed
        (x3t, x2tm),  # different time cannot be allowed
        (x3t, x3tm),  # different time cannot be allowed
        # merging a temporal quantity with a spatial matrix: x3 would be
        # broadcast along the time of x1t, and x1t along the dim of x3
        (x3, x1t),
        (x3, x1tm),
    ]
    _check(allowed, refused)


def test_only_a_scalar_dim_broadcasts():
    from nnodely import Input

    ## Time is always at least 1
    row = Input("row", dim=3)  # shape [B, 1, 3]
    matrix = Input("matrix", dim=(2, 3))  # shape [B, 1, 2, 3]
    scalar = Input("scalar", dim=(1, 1))  # shape [B, 1, 1, 1]
    row_t = Input("row_t", dim=3).sw(2)  # shape [B, 2, 3]
    matrix_t = Input("matrix_t", dim=(2, 3)).sw(2)  # shape [B, 2, 2, 3]

    # A vector does not broadcast to a matrix, even one whose rows it fits.
    _check(
        allowed=[
            (scalar, matrix, (1, 2, 3)),  # every axis of size 1: a scalar
            (scalar, row, (1, 1, 3)),  # padded with a leading axis, as numpy does
        ],
        refused=[(row, matrix), (row_t, matrix_t)],
    )


def test_arithmetic_shape_sequence():
    from nnodely import Input

    x1st = Input("x1t", dim=1, seq=(5,))  # shape [B, 5, 1, 1]
    x2st = Input("x2t", dim=4, seq=(5,))  # shape [B, 5, 1, 4]
    x3st = Input("x3t", dim=(2, 3), seq=(5,))  # shape [B, 5, 1, 2, 3]

    x1smtm = Input("x1tm", dim=1, seq=(5, 2)).sw(3)  # shape [B, 5, 2, 3, 1]
    x2smtm = Input("x2tm", dim=4, seq=(5, 2)).sw(3)  # shape [B, 5, 2, 3, 4]
    x3smtm = Input("x3tm", dim=(2, 3), seq=(5, 2)).sw(3)  # shape [B, 5, 2, 3, 2, 3]

    ## Should be allowed
    allowed = [
        (x1st, x1st, (5, 1, 1)),
        (x1st, x2st, (5, 1, 4)),  # broadcast the scalar to the spatial vector
        (x1st, x3st, (5, 1, 2, 3)),  # broadcast the scalar to the spatial matrix
        (x1smtm, x1smtm, (5, 2, 3, 1)),
        (x1smtm, x2smtm, (5, 2, 3, 4)),  # broadcast the scalar to the spatial vector
        (x1smtm, x3smtm, (5, 2, 3, 2, 3)),  # broadcast the scalar to the spatial matrix
    ]

    ## should not be allowed
    refused = [
        (x2st, x3st),  # cannot broadcast the vector to the matrix
        (x2smtm, x3smtm),  # cannot broadcast the vector to the matrix
        (x1st, x1smtm),  # different time cannot be allowed
        (x1st, x2smtm),  # different time cannot be allowed
        (x1st, x3smtm),  # different time cannot be allowed
        (x2st, x1smtm),  # different time cannot be allowed
        (x2st, x2smtm),  # different time cannot be allowed
        (x2st, x3smtm),  # different time cannot be allowed
        (x3st, x1smtm),  # different time cannot be allowed
        (x3st, x2smtm),  # different time cannot be allowed
        (x3st, x3smtm),  # different time cannot be allowed
    ]
    _check(allowed, refused)


def test_a_dynamic_time_axis_matches_any_length():
    from nnodely import Input

    dynamic = Input("dyn", dim=2, seq=-1)
    fixed = Input("fixed", dim=2, seq=4)
    assert (dynamic + fixed).shape.dimensions == ((None,), 1, (2,))


def test_broadcast_values_follow_the_shapes():
    # A matrix without time is repeated along the time of the window it is
    # added to, and a scalar window along the dim of the matrix window.
    from nnodely import Input, Modely, Output, Parameter

    x = Input("bx", dim=(2, 3))
    s = Input("bs", dim=1)
    offsets = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    matrix = Parameter("bmatrix", value=offsets)  # S=(), T=1, D=(2, 3)
    out = x.sw(2) + matrix + s.sw(2)
    assert out.shape.tuple == (2, 2, 3)

    model = Modely("broadcast", inputs=[x, s], outputs=[Output("bo", out)]).build()
    xs = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    ss = np.array([[[100.0], [200.0]]], dtype=np.float32)
    result = to_numpy(model({"bx": xs, "bs": ss})["bo"])
    expected = xs + np.array(offsets) + ss[..., np.newaxis]
    np.testing.assert_allclose(result, expected)


def test_fir_filters_the_window_at_every_step_of_the_sequence():
    from nnodely import Input, Modely, Output, Parameter

    x = Input("fir_x", dim=1, seq=2)
    kernel = Parameter("fir_kernel", value=[[1.0], [10.0], [100.0]])  # (window, 1)
    out = Fir(features=1, kernel=kernel, bias=False)(x.sw(3))
    assert out.shape.dimensions == ((2,), 1, (1,))

    model = Modely("fir_inner", inputs=[x], outputs=[Output("fir_out", out)]).build()
    xs = np.array([[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]], dtype=np.float32)
    result = to_numpy(model({"fir_x": xs})["fir_out"])
    # One filtered sample per outer step: 1 + 20 + 300, 4 + 50 + 600.
    np.testing.assert_allclose(result, [[[[321.0]], [[654.0]]]])
