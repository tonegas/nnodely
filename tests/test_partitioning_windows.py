import os

import numpy as np
import pytest

from conftest import to_numpy
from nnodely import (
    DataLoader,
    Input,
    Loop,
    Modely,
    Output,
    Select,
    TimeConcatenate,
    TimeRange,
    TimeReverse,
    TimeSelect,
    TimeStack,
)


def test_partitioning_windows():
    # ------- Model with time window partitioning, Past and Future -------
    # x = Input("x", dim=1, sampling=0.1) ## x is sampled at 0.1Hz, so 1 sample every 10 seconds
    x = Input("x", dim=1)
    y = Input("y", dim=1)
    z = Input("z", dim=1)

    x_stream1 = x.sw(3)  ## 3 samples of x in the past
    y_stream1 = y.sw([0, 3])  ## 3 samples of y in the future
    z_stream1 = z.sw(
        [4, 2]
    )  ## 4 samples of z in the past and 2 samples of z in the future

    x_stream2 = x.last()  ## 1 sample of x in the past
    y_stream2 = y.next()  ## 1 sample of y in the future
    z_stream2 = z.sw(3)  ## 1 sample of z in the past (same as last())

    x_stream3 = x.sw(5)  ## 5 samples of x in the past
    y_stream3 = y.sw(
        [2, 2]
    )  ## 2 samples of y in the past and 2 samples of y in the future
    z_stream3 = z.sw(
        [1, 3]
    )  ## 1 sample of z in the past and 3 samples of z in the future

    add1 = x_stream1 + y_stream1 + z_stream2
    add2 = x_stream2 + y_stream2
    add3 = y_stream3 + z_stream3

    out1 = Output("out1", add1)
    out2 = Output("out2", add2)
    out3 = Output("out3", add3)
    out_x_stream1 = Output("out_x_stream1", x_stream1)
    out_y_stream1 = Output("out_y_stream1", y_stream1)
    out_z_stream1 = Output("out_z_stream1", z_stream1)
    out_x_stream2 = Output("out_x_stream2", x_stream2)
    out_y_stream2 = Output("out_y_stream2", y_stream2)
    out_z_stream2 = Output("out_z_stream2", z_stream2)
    out_x_stream3 = Output("out_x_stream3", x_stream3)
    out_y_stream3 = Output("out_y_stream3", y_stream3)
    out_z_stream3 = Output("out_z_stream3", z_stream3)
    model = Modely(
        "model1",
        inputs=[x, y, z],
        outputs=[
            out1,
            out2,
            out3,
            out_x_stream1,
            out_y_stream1,
            out_z_stream1,
            out_x_stream2,
            out_y_stream2,
            out_z_stream2,
            out_x_stream3,
            out_y_stream3,
            out_z_stream3,
        ],
    )
    model.build()

    ## ------ Load dataset -------
    data_train = DataLoader(
        model,
        format={"x": "data_1", "y": "data_2", "z": "data_3"},
        source=os.path.join(os.path.dirname(__file__), "datasets"),
    )

    assert data_train.dataset is not None
    assert data_train.inputs == ["x", "y", "z"]
    assert data_train.input_specs == {"x": [5, 0], "y": [2, 3], "z": [4, 3]}
    print("Data train dataset:", {k: v.shape for k, v in data_train.dataset.items()})
    assert data_train.dataset["x"].shape == (6, 5, 1)
    assert data_train.dataset["y"].shape == (6, 5, 1)
    assert data_train.dataset["z"].shape == (6, 7, 1)
    # Each sample is (time, dim): the windows read down the time axis.
    assert data_train.dataset["x"][..., 0].tolist() == [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [11.0, 12.0, 13.0, 14.0, 15.0],
        [12.0, 13.0, 14.0, 15.0, 16.0],
        [13.0, 14.0, 15.0, 16.0, 17.0],
    ]
    assert data_train.dataset["y"][..., 0].tolist() == [
        [10.0, 15.0, 21.0, 28.0, 36.0],
        [15.0, 21.0, 28.0, 36.0, 45.0],
        [21.0, 28.0, 36.0, 45.0, 55.0],
        [105.0, 120.0, 136.0, 153.0, 171.0],
        [120.0, 136.0, 153.0, 171.0, 190.0],
        [136.0, 153.0, 171.0, 190.0, 210.0],
    ]
    assert data_train.dataset["z"][..., 0].tolist() == [
        [5.0, 9.0, 14.0, 20.0, 26.0, 33.0, 41.0],
        [9.0, 14.0, 20.0, 26.0, 33.0, 41.0, 50.0],
        [14.0, 20.0, 26.0, 33.0, 41.0, 50.0, 60.0],
        [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    ]

    result = model(data_train.get_samples(1, start=0))
    assert to_numpy(result["out1"])[..., 0].tolist() == [[33, 46, 61]]
    assert to_numpy(result["out2"])[..., 0].tolist() == [[26]]
    assert to_numpy(result["out3"])[..., 0].tolist() == [[30, 41, 54, 69]]
    assert to_numpy(result["out_x_stream1"])[..., 0].tolist() == [[3, 4, 5]]
    assert to_numpy(result["out_x_stream2"])[..., 0].tolist() == [[5]]
    assert to_numpy(result["out_x_stream3"])[..., 0].tolist() == [[1, 2, 3, 4, 5]]
    assert to_numpy(result["out_y_stream1"])[..., 0].tolist() == [[21, 28, 36]]
    assert to_numpy(result["out_y_stream2"])[..., 0].tolist() == [[21]]
    assert to_numpy(result["out_y_stream3"])[..., 0].tolist() == [[10, 15, 21, 28]]
    assert to_numpy(result["out_z_stream1"])[..., 0].tolist() == [
        [5, 9, 14, 20, 26, 33]
    ]
    assert to_numpy(result["out_z_stream2"])[..., 0].tolist() == [[9, 14, 20]]
    assert to_numpy(result["out_z_stream3"])[..., 0].tolist() == [[20, 26, 33, 41]]

    result = model(data_train.get_samples(1, start=1))
    assert to_numpy(result["out1"])[..., 0].tolist() == [[46, 61, 77]]
    assert to_numpy(result["out2"])[..., 0].tolist() == [[34]]
    assert to_numpy(result["out3"])[..., 0].tolist() == [[41, 54, 69, 86]]
    assert to_numpy(result["out_x_stream1"])[..., 0].tolist() == [[4, 5, 6]]
    assert to_numpy(result["out_x_stream2"])[..., 0].tolist() == [[6]]
    assert to_numpy(result["out_x_stream3"])[..., 0].tolist() == [[2, 3, 4, 5, 6]]
    assert to_numpy(result["out_y_stream1"])[..., 0].tolist() == [[28, 36, 45]]
    assert to_numpy(result["out_y_stream2"])[..., 0].tolist() == [[28]]
    assert to_numpy(result["out_y_stream3"])[..., 0].tolist() == [[15, 21, 28, 36]]
    assert to_numpy(result["out_z_stream1"])[..., 0].tolist() == [
        [9, 14, 20, 26, 33, 41]
    ]
    assert to_numpy(result["out_z_stream2"])[..., 0].tolist() == [[14, 20, 26]]
    assert to_numpy(result["out_z_stream3"])[..., 0].tolist() == [[26, 33, 41, 50]]


# ---------------------------------------------------------------------------
# Time axes: TimeSelect, TimeRange, TimeConcatenate, TimeStack, TimeReverse
# ---------------------------------------------------------------------------
def _run(name, inputs, relation, data):
    model = Modely(
        name, inputs=inputs, outputs=[Output(f"{name}_out", relation)]
    ).build()
    return to_numpy(model(data)[f"{name}_out"])


def test_time_select_fixed_drops_the_axis():
    x = Input("seq_select_x", dim=2)
    window = x.sw(2, seq=5)
    third = TimeSelect(2, axis=0, keep_dim=False)([window])
    last = TimeSelect(-1, axis=0, keep_dim=False)([window])

    assert third.shape.dimensions == ((2,), (2,))
    assert last.shape.dimensions == ((2,), (2,))

    values = np.random.rand(3, 5, 2, 2).astype(np.float32)
    np.testing.assert_allclose(
        _run("seq_select_third", [x], third, {"seq_select_x": values}),
        values[:, 2],
    )
    np.testing.assert_allclose(
        _run("seq_select_last", [x], last, {"seq_select_x": values}),
        values[:, -1],
    )


def test_time_select_dynamic_counts_from_the_end_of_the_data():
    x = Input("seq_select_dyn_x")
    last = TimeSelect(-1, axis=0, keep_dim=False)([x.last(seq=-1)])
    assert last.shape.dimensions == ((1,), (1,))

    model = Modely(
        "seq_select_dyn", inputs=[x], outputs=[Output("seq_select_dyn_out", last)]
    ).build()
    for steps in (1, 4, 9):
        values = np.arange(steps, dtype=np.float32).reshape(1, steps, 1, 1)
        result = to_numpy(model({"seq_select_dyn_x": values})["seq_select_dyn_out"])
        np.testing.assert_allclose(result, values[:, -1])


def test_time_select_nested_axis():
    x = Input("seq_select_nested_x")
    window = x.last(seq=(-1, -1))
    outer = TimeSelect(0, axis=1, keep_dim=False)([window])
    inner = TimeSelect(-1, axis=0, keep_dim=False)([window])
    assert outer.shape.time == (None, 1)
    assert inner.shape.time == (None, 1)

    values = np.random.rand(1, 3, 4, 1, 1).astype(np.float32)
    data = {"seq_select_nested_x": values}
    np.testing.assert_allclose(
        _run("seq_select_nested_outer", [x], outer, data), values[:, :, 0]
    )
    np.testing.assert_allclose(
        _run("seq_select_nested_inner", [x], inner, data), values[:, -1]
    )


def test_time_range_fixed():
    x = Input("seq_range_x", dim=2)
    window = x.last(seq=6)
    middle = TimeRange(1, 4, axis=0)([window])
    tail = TimeRange(-3, None, axis=0)([window])
    assert middle.shape.time == (3, 1)
    assert tail.shape.time == (3, 1)

    values = np.random.rand(2, 6, 1, 2).astype(np.float32)
    data = {"seq_range_x": values}
    np.testing.assert_allclose(
        _run("seq_range_middle", [x], middle, data), values[:, 1:4]
    )
    np.testing.assert_allclose(_run("seq_range_tail", [x], tail, data), values[:, -3:])


def test_time_range_dynamic():
    x = Input("seq_range_dyn_x")
    window = x.last(seq=-1)
    head = TimeRange(0, 2, axis=0)([window])
    tail = TimeRange(-3, None, axis=0)([window])
    open_start = TimeRange(None, 2, axis=0)([window])
    # A range with known non-negative bounds has a known length; one counted
    # from the end of the data follows the data.
    assert head.shape.time == (2, 1)
    assert open_start.shape.time == (2, 1)
    assert tail.shape.time == (None, 1)

    model = Modely(
        "seq_range_dyn",
        inputs=[x],
        outputs=[Output("seq_range_head", head), Output("seq_range_tail", tail)],
    ).build()
    for steps in (3, 7):
        values = np.arange(steps, dtype=np.float32).reshape(1, steps, 1, 1)
        result = model({"seq_range_dyn_x": values})
        np.testing.assert_allclose(to_numpy(result["seq_range_head"]), values[:, :2])
        np.testing.assert_allclose(to_numpy(result["seq_range_tail"]), values[:, -3:])


def test_time_concatenate():
    x = Input("seq_concat_x", dim=2)
    y = Input("seq_concat_y", dim=2)
    d = Input("seq_concat_d", dim=2)
    x_window, y_window = x.last(seq=3), y.last(seq=5)
    joined = TimeConcatenate(axis=0)([x_window, y_window])
    dynamic = TimeConcatenate(axis=0)([x_window, d.last(seq=-1)])
    assert joined.shape.time == (8, 1)
    assert dynamic.shape.time == (None, 1)

    xv = np.random.rand(1, 3, 1, 2).astype(np.float32)
    yv = np.random.rand(1, 5, 1, 2).astype(np.float32)
    dv = np.random.rand(1, 4, 1, 2).astype(np.float32)
    np.testing.assert_allclose(
        _run(
            "seq_concat",
            [x, y],
            joined,
            {"seq_concat_x": xv, "seq_concat_y": yv},
        ),
        np.concatenate([xv, yv], axis=1),
    )
    np.testing.assert_allclose(
        _run(
            "seq_concat_dyn",
            [x, d],
            dynamic,
            {"seq_concat_x": xv, "seq_concat_d": dv},
        ),
        np.concatenate([xv, dv], axis=1),
    )


def test_time_concatenate_nested_axis():
    x = Input("seq_concat_nested_x")
    y = Input("seq_concat_nested_y")
    joined = TimeConcatenate(axis=1)([x.last(seq=(-1, 2)), y.last(seq=(-1, 3))])
    assert joined.shape.time == (None, 5, 1)

    xv = np.random.rand(1, 4, 2, 1, 1).astype(np.float32)
    yv = np.random.rand(1, 4, 3, 1, 1).astype(np.float32)
    np.testing.assert_allclose(
        _run(
            "seq_concat_nested",
            [x, y],
            joined,
            {"seq_concat_nested_x": xv, "seq_concat_nested_y": yv},
        ),
        np.concatenate([xv, yv], axis=2),
    )


def test_time_stack():
    a, b, c = (Input(f"seq_stack_{name}", dim=2) for name in "abc")
    stacked = TimeStack(axis=0)([a.last(), b.last(), c.last()])
    assert stacked.shape.dimensions == ((3, 1), (2,))

    values = {
        f"seq_stack_{name}": np.random.rand(2, 1, 2).astype(np.float32)
        for name in "abc"
    }
    np.testing.assert_allclose(
        _run("seq_stack", [a, b, c], stacked, values),
        np.stack(list(values.values()), axis=1),
    )


def test_time_stack_new_outer_axis_of_dynamic_sequences():
    x = Input("seq_stack_dyn_x")
    y = Input("seq_stack_dyn_y")
    stacked = TimeStack(axis=1)([x.last(seq=-1), y.last(seq=-1)])
    assert stacked.shape.time == (None, 2, 1)

    xv = np.random.rand(1, 6, 1, 1).astype(np.float32)
    yv = np.random.rand(1, 6, 1, 1).astype(np.float32)
    np.testing.assert_allclose(
        _run(
            "seq_stack_dyn",
            [x, y],
            stacked,
            {"seq_stack_dyn_x": xv, "seq_stack_dyn_y": yv},
        ),
        np.stack([xv, yv], axis=2),
    )


def test_time_reverse():
    x = Input("seq_reverse_x", dim=2)
    window = x.last(seq=(-1, 3))
    inner = TimeReverse(axis=0)([window])
    outer = TimeReverse(axis=1)([window])
    assert inner.shape.time == outer.shape.time == (None, 3, 1)

    values = np.random.rand(1, 5, 3, 1, 2).astype(np.float32)
    data = {"seq_reverse_x": values}
    np.testing.assert_allclose(
        _run("seq_reverse_inner", [x], inner, data), values[:, ::-1]
    )
    np.testing.assert_allclose(
        _run("seq_reverse_outer", [x], outer, data), values[:, :, ::-1]
    )


def test_time_ops_reject_invalid_declarations():
    x = Input("seq_invalid_x").last(seq=4)
    y = Input("seq_invalid_y").last(seq=(2, 4))
    plain = Input("seq_invalid_plain")

    with pytest.raises(ValueError, match="out of bounds for time axis"):
        TimeSelect(4, axis=0)([x])
    with pytest.raises(ValueError, match="out of bounds for time rank 0"):
        TimeSelect(-1, axis=0)([plain])
    with pytest.raises(ValueError, match="out of bounds for time rank 2"):
        TimeReverse(axis=2)([x])
    with pytest.raises(ValueError, match="invalid range"):
        TimeRange(3, 2, axis=0)([x])
    with pytest.raises(ValueError, match="invalid range"):
        TimeRange(0, 5, axis=0)([x])
    with pytest.raises(ValueError, match="matching dim and time"):
        TimeConcatenate(axis=0)([x, y])
    with pytest.raises(ValueError, match="matching dim and time"):
        TimeStack(axis=0)([x, y])
    with pytest.raises(ValueError, match="at least two inputs"):
        TimeConcatenate(axis=0)([x])


def test_select_drops_its_axis_unless_kept():
    x = Input("select_keep_x", dim=(4, 3))
    v = Input("select_keep_v", dim=4)
    dropped = Select(1)([x.last()])
    kept = Select(-1, axis=1, keep_dim=True)([x.last()])
    # A stream always has a dim: the only axis of a vector stays, as a scalar.
    scalar = Select(2)([v.last()])
    assert dropped.shape.dimensions == ((1,), (3,))
    assert kept.shape.dimensions == ((1,), (4, 1))
    assert scalar.shape.dimensions == ((1,), (1,))

    xv = np.random.rand(2, 1, 4, 3).astype(np.float32)
    vv = np.random.rand(2, 1, 4).astype(np.float32)
    data = {"select_keep_x": xv, "select_keep_v": vv}
    np.testing.assert_allclose(
        _run("select_dropped", [x], dropped, data), xv[..., 1, :]
    )
    np.testing.assert_allclose(_run("select_kept", [x], kept, data), xv[..., -1:])
    np.testing.assert_allclose(_run("select_scalar", [v], scalar, data), vv[..., 2:3])


def test_time_select_keeps_its_axis_by_default():
    x = Input("time_keep_x", dim=2)
    window = x.sw(3, seq=-1)
    last_step = TimeSelect(-1, axis=0)([window])
    last_sample = TimeSelect(-1)([window])
    assert last_step.shape.time == (1, 3)
    assert last_sample.shape.time == (None, 1)

    model = Modely(
        "time_keep",
        inputs=[x],
        outputs=[
            Output("time_keep_step", last_step),
            Output("time_keep_sample", last_sample),
        ],
    ).build()
    for steps in (1, 4):
        values = np.random.rand(1, steps, 3, 2).astype(np.float32)
        result = model({"time_keep_x": values})
        # -1 is the last step of a dynamic axis, of any length.
        np.testing.assert_allclose(to_numpy(result["time_keep_step"]), values[:, -1:])
        np.testing.assert_allclose(
            to_numpy(result["time_keep_sample"]), values[:, :, -1:]
        )


def test_time_ops_act_on_the_window_by_default():
    # The inner time; the outer time, a sequence, is kept.
    x = Input("inner_ops_x")
    y = Input("inner_ops_y")
    x_window, y_window = x.sw(2, seq=4), y.sw(3, seq=4)
    newest = TimeSelect(-1)([x_window])
    middle = TimeRange(1, 3)([y_window])
    joined = TimeConcatenate()([x_window, y_window])
    stacked = TimeStack()([x_window, x_window])
    reversed_ = TimeReverse()([y_window])
    assert newest.shape.time == (4, 1)
    assert middle.shape.time == (4, 2)
    assert joined.shape.time == (4, 5)
    assert stacked.shape.time == (4, 2, 2)
    assert reversed_.shape.time == (4, 3)

    xv = np.random.rand(2, 4, 2, 1).astype(np.float32)
    yv = np.random.rand(2, 4, 3, 1).astype(np.float32)
    data = {"inner_ops_x": xv, "inner_ops_y": yv}
    np.testing.assert_allclose(_run("inner_newest", [x], newest, data), xv[:, :, -1:])
    np.testing.assert_allclose(_run("inner_middle", [y], middle, data), yv[:, :, 1:3])
    np.testing.assert_allclose(
        _run("inner_joined", [x, y], joined, data), np.concatenate([xv, yv], axis=2)
    )
    np.testing.assert_allclose(
        _run("inner_stacked", [x], stacked, data), np.stack([xv, xv], axis=3)
    )
    np.testing.assert_allclose(
        _run("inner_reversed", [y], reversed_, data), yv[:, :, ::-1]
    )


def test_a_selected_sample_broadcasts_along_the_window():
    # Without time, the newest sample combines with every sample of the window.
    x = Input("relative_x")
    window = x.sw(4)
    relative = window - TimeSelect(-1, keep_dim=False)([window])
    assert relative.shape.dimensions == ((4,), (1,))

    values = np.random.rand(2, 4, 1).astype(np.float32)
    np.testing.assert_allclose(
        _run("relative", [x], relative, {"relative_x": values}),
        values - values[:, -1:],
        rtol=1e-6,
    )


def test_time_select_closes_a_loop_chained_over_chunks():
    # #209: an outer Loop chains the chunks of an inner one, each chunk seeded
    # by the last position and velocity of the one before.
    vx0, px0, ax = Input("chunk_vx0"), Input("chunk_px0"), Input("chunk_ax")
    outvx = Output("chunk_outvx", vx0.last() + ax.last())
    outpx = Output("chunk_outpx", px0.last() + vx0.last())
    step = Modely("chunk_step", inputs=[ax, vx0, px0], outputs=[outvx, outpx]).build()

    ax_sq = Input("chunk_ax_sq")
    px0_init, vx0_init = Input("chunk_px0_init"), Input("chunk_vx0_init")
    velocity, position = Loop(
        f=step,
        callback={vx0: outvx, px0: outpx},
        init={vx0: vx0_init.last(), px0: px0_init.last()},
    )({ax: ax_sq.last(seq=-1)})
    chunk = Modely(
        "chunk",
        inputs=[ax_sq, px0_init, vx0_init],
        outputs=[
            Output("chunk_traj", position),
            Output("chunk_last_px", TimeSelect(-1, axis=0, keep_dim=False)([position])),
            Output("chunk_last_vx", TimeSelect(-1, axis=0, keep_dim=False)([velocity])),
        ],
    ).build()

    ax_sq_sq = Input("chunk_ax_sq_sq")
    px0_seed, vx0_seed = Input("chunk_px0_seed"), Input("chunk_vx0_seed")
    trajectory, _, _ = Loop(
        f=chunk,
        callback={"chunk_px0_init": "chunk_last_px", "chunk_vx0_init": "chunk_last_vx"},
        init={"chunk_px0_init": px0_seed.last(), "chunk_vx0_init": vx0_seed.last()},
    )({"chunk_ax_sq": ax_sq_sq.last(seq=(-1, -1))})
    assert trajectory.shape.time == (None, None, 1)
    model = Modely(
        "chunked",
        inputs=[ax_sq_sq, px0_seed, vx0_seed],
        outputs=[Output("chunked_traj", trajectory)],
    ).build()

    steps, chunks = 5, 3
    accelerations = np.random.rand(1, chunks, steps, 1, 1).astype(np.float32)
    result = to_numpy(
        model(
            {
                "chunk_ax_sq_sq": accelerations,
                "chunk_px0_seed": np.full((1, 1, 1), 0.5, dtype=np.float32),
                "chunk_vx0_seed": np.full((1, 1, 1), 0.2, dtype=np.float32),
            }
        )["chunked_traj"]
    )

    # The chunks chained are one rollout over the whole acceleration sequence.
    position_, velocity_, expected = 0.5, 0.2, []
    for acceleration in accelerations[0, :, :, 0, 0].ravel():
        position_, velocity_ = position_ + velocity_, velocity_ + acceleration
        expected.append(position_)
    np.testing.assert_allclose(
        result[0, :, :, 0, 0], np.reshape(expected, (chunks, steps)), rtol=1e-5
    )


def test_one_time_op_applied_to_streams_of_two_shapes():
    # The second application once reused the Keras layer built for the first,
    # with the tensor axis of the first stream: TimeReverse reversed the wrong
    # axis of the second. Counted from the inner time, -2 is the time axis
    # right outside the window of both.
    reverse = TimeReverse(axis=-2)
    c = Input("two_shapes_c")
    d = Input("two_shapes_d", dim=(2, 2))
    model = Modely(
        "two_shapes",
        inputs=[c, d],
        outputs=[
            Output("two_shapes_rc", reverse(c.last(seq=(3, 4)))),
            Output("two_shapes_rd", reverse(d.last(seq=3))),
        ],
    ).build()
    cs = np.arange(12, dtype=np.float32).reshape(1, 3, 4, 1, 1)
    ds = np.arange(12, dtype=np.float32).reshape(1, 3, 1, 2, 2)
    result = model({"two_shapes_c": cs, "two_shapes_d": ds})
    np.testing.assert_array_equal(to_numpy(result["two_shapes_rc"]), cs[:, :, ::-1])
    np.testing.assert_array_equal(to_numpy(result["two_shapes_rd"]), ds[:, ::-1])


def test_unbounded_temporal_window():
    from nnodely import Fir

    x = Input("unbounded_x")
    print(x.shape)
    # this is wrong, x_window will never be used inside the graph but it will still modify the input shape of x
    x_window = x.sw(5)  # noqa: F841 - declared, never used
    print(x.shape)
    x_window_3 = x.sw(3)
    print(x.shape)

    fir = Fir(features=1)(x_window_3)

    out1 = Output("out", fir)

    model = Modely(
        "unbounded_model",
        inputs=[x],
        outputs=[out1],
    ).build()
    print(x.shape)

    assert x.shape.time == (3,)  ## it will give (5) which is wrong

    data_train = DataLoader(
        model,
        format={"unbounded_x": "data_1"},
        source=os.path.join(os.path.dirname(__file__), "datasets"),
    )

    assert data_train.dataset is not None
    sample = data_train.get_samples(1, start=0)
    assert (
        sample["unbounded_x"].shape == (1, 3, 1)
    )  # I expect this to be (1, 3, 1) but it's (1, 5, 1) because of the x_window = x.sw(5) line above


def test_sequence_allowed_only_once():
    x = Input("seq_once_x")
    x.sw(3, seq=3)
    with pytest.raises(ValueError, match="only once"):
        x.sw(2, seq=2)


def test_each_model_reads_the_windows_of_its_own_graph():
    # Requesting a window leaves the Input as it is: build() finds the windows
    # of the model's graph, so two models of one Input read their own.
    from nnodely import Fir

    x = Input("own_windows_x")
    narrow, wide = x.sw(2), x.sw(4)
    assert x.shape.time == ()  # nothing built yet
    a = Modely(
        "own_a", inputs=[x], outputs=[Output("own_a_out", Fir(features=1)(narrow))]
    ).build()
    b = Modely(
        "own_b", inputs=[x], outputs=[Output("own_b_out", Fir(features=1)(wide))]
    ).build()
    assert x.shape.time == (4,)  # the last model built
    source = {"own_windows_x": np.arange(10.0).reshape(-1, 1)}
    assert DataLoader(a, source=source).get_samples(1)["own_windows_x"].shape == (
        1,
        2,
        1,
    )
    assert DataLoader(b, source=source).get_samples(1)["own_windows_x"].shape == (
        1,
        4,
        1,
    )
    # a, built first, still reads its own window.
    assert a.train_inputs[0].shape.time == (2,)


def test_a_window_only_a_minimizer_reads_widens_the_training_input():
    from nnodely import Fir

    x = Input("trained_window_x")
    out = Output("trained_window_out", Fir(features=1)(x.sw(2)))
    model = Modely("trained_window", inputs=[x], outputs=[out])
    model.minimize("fit", out, x.sw([1, 1]))  # one sample ahead
    model.build()
    assert (x.past, x.future) == (2, 1)


def test_an_input_read_whole_and_through_windows_is_refused():
    from nnodely import Fir

    x = Input("whole_and_window_x")
    model = Modely(
        "whole_and_window",
        inputs=[x],
        outputs=[
            Output("whole", x * 2.0),
            Output("window", Fir(features=1)(x.sw(3))),
        ],
    )
    with pytest.raises(ValueError, match="reads the Input 'whole_and_window_x' whole"):
        model.build()


def test_the_sequence_is_set_by_the_first_window_alone():
    x = Input("first_seq_x")
    first = x.sw(3, seq=-1)
    later = x.sw(2)  # inherits it
    assert first.time == (None, 3) and later.time == (None, 2)
    # Set again, even to the same value, it is refused: a window made before
    # would not have it.
    with pytest.raises(ValueError, match="only once"):
        x.last(seq=-1)
    # A first window without one sets "no sequence".
    y = Input("first_seq_y")
    y.last()
    with pytest.raises(ValueError, match="only once"):
        y.sw(2, seq=4)
