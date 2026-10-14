import numpy as np
import pytest

from conftest import to_numpy
from nnodely import (
    Input,
    Loop,
    Modely,
    Output,
    SeqConcatenate,
    SeqRange,
    SeqReverse,
    SeqSelect,
    SeqStack,
)


def _run(name, inputs, relation, data):
    model = Modely(
        name, inputs=inputs, outputs=[Output(f"{name}_out", relation)]
    ).build()
    return to_numpy(model(data)[f"{name}_out"])


def test_seq_select_fixed_drops_the_axis():
    x = Input("seq_select_x", dim=2, seq=5)
    window = x.sw(2)
    third, last = SeqSelect(2)([window]), SeqSelect(-1)([window])

    assert third.shape.dimensions == ((2,), 2, ())
    assert last.shape.dimensions == ((2,), 2, ())

    values = np.random.rand(3, 2, 2, 5).astype(np.float32)
    np.testing.assert_allclose(
        _run("seq_select_third", [x], third, {"seq_select_x": values}),
        values[..., 2],
    )
    np.testing.assert_allclose(
        _run("seq_select_last", [x], last, {"seq_select_x": values}),
        values[..., -1],
    )


def test_seq_select_dynamic_counts_from_the_end_of_the_data():
    x = Input("seq_select_dyn_x", seq=-1)
    last = SeqSelect(-1)([x])
    assert last.shape.dimensions == ((1,), 1, ())

    model = Modely(
        "seq_select_dyn", inputs=[x], outputs=[Output("seq_select_dyn_out", last)]
    ).build()
    for steps in (1, 4, 9):
        values = np.arange(steps, dtype=np.float32).reshape(1, 1, 1, steps)
        result = to_numpy(model({"seq_select_dyn_x": values})["seq_select_dyn_out"])
        np.testing.assert_allclose(result, values[..., -1])


def test_seq_select_nested_axis():
    x = Input("seq_select_nested_x", seq=(-1, -1))
    outer = SeqSelect(0, axis=0)([x])
    inner = SeqSelect(-1)([x])
    assert outer.seq == (None,)
    assert inner.seq == (None,)

    values = np.random.rand(1, 1, 1, 4, 3).astype(np.float32)
    data = {"seq_select_nested_x": values}
    np.testing.assert_allclose(
        _run("seq_select_nested_outer", [x], outer, data), values[..., 0, :]
    )
    np.testing.assert_allclose(
        _run("seq_select_nested_inner", [x], inner, data), values[..., -1]
    )


def test_seq_range_fixed():
    x = Input("seq_range_x", dim=2, seq=6)
    middle = SeqRange(1, 4)([x])
    tail = SeqRange(-3, None)([x])
    assert middle.seq == (3,)
    assert tail.seq == (3,)

    values = np.random.rand(2, 2, 1, 6).astype(np.float32)
    data = {"seq_range_x": values}
    np.testing.assert_allclose(
        _run("seq_range_middle", [x], middle, data), values[..., 1:4]
    )
    np.testing.assert_allclose(
        _run("seq_range_tail", [x], tail, data), values[..., -3:]
    )


def test_seq_range_dynamic():
    x = Input("seq_range_dyn_x", seq=-1)
    head = SeqRange(0, 2)([x])
    tail = SeqRange(-3, None)([x])
    open_start = SeqRange(None, 2)([x])
    # A range with known non-negative bounds has a known length; one counted
    # from the end of the data follows the data.
    assert head.seq == (2,)
    assert open_start.seq == (2,)
    assert tail.seq == (None,)

    model = Modely(
        "seq_range_dyn",
        inputs=[x],
        outputs=[Output("seq_range_head", head), Output("seq_range_tail", tail)],
    ).build()
    for steps in (3, 7):
        values = np.arange(steps, dtype=np.float32).reshape(1, 1, 1, steps)
        result = model({"seq_range_dyn_x": values})
        np.testing.assert_allclose(to_numpy(result["seq_range_head"]), values[..., :2])
        np.testing.assert_allclose(to_numpy(result["seq_range_tail"]), values[..., -3:])


def test_seq_concatenate():
    x = Input("seq_concat_x", dim=2, seq=3)
    y = Input("seq_concat_y", dim=2, seq=5)
    d = Input("seq_concat_d", dim=2, seq=-1)
    joined = SeqConcatenate()([x, y])
    dynamic = SeqConcatenate()([x, d])
    assert joined.seq == (8,)
    assert dynamic.seq == (None,)

    xv = np.random.rand(1, 2, 1, 3).astype(np.float32)
    yv = np.random.rand(1, 2, 1, 5).astype(np.float32)
    dv = np.random.rand(1, 2, 1, 4).astype(np.float32)
    np.testing.assert_allclose(
        _run(
            "seq_concat",
            [x, y],
            joined,
            {"seq_concat_x": xv, "seq_concat_y": yv},
        ),
        np.concatenate([xv, yv], axis=-1),
    )
    np.testing.assert_allclose(
        _run(
            "seq_concat_dyn",
            [x, d],
            dynamic,
            {"seq_concat_x": xv, "seq_concat_d": dv},
        ),
        np.concatenate([xv, dv], axis=-1),
    )


def test_seq_concatenate_nested_axis():
    x = Input("seq_concat_nested_x", seq=(2, -1))
    y = Input("seq_concat_nested_y", seq=(3, -1))
    joined = SeqConcatenate(axis=0)([x, y])
    assert joined.seq == (5, None)

    xv = np.random.rand(1, 1, 1, 2, 4).astype(np.float32)
    yv = np.random.rand(1, 1, 1, 3, 4).astype(np.float32)
    np.testing.assert_allclose(
        _run(
            "seq_concat_nested",
            [x, y],
            joined,
            {"seq_concat_nested_x": xv, "seq_concat_nested_y": yv},
        ),
        np.concatenate([xv, yv], axis=-2),
    )


def test_seq_stack():
    a, b, c = (Input(f"seq_stack_{name}", dim=2) for name in "abc")
    stacked = SeqStack()([a.last(), b.last(), c.last()])
    assert stacked.shape.dimensions == ((2,), 1, (3,))

    values = {
        f"seq_stack_{name}": np.random.rand(2, 2, 1).astype(np.float32)
        for name in "abc"
    }
    np.testing.assert_allclose(
        _run("seq_stack", [a, b, c], stacked, values),
        np.stack(list(values.values()), axis=-1),
    )


def test_seq_stack_new_outer_axis_of_dynamic_sequences():
    x = Input("seq_stack_dyn_x", seq=-1)
    y = Input("seq_stack_dyn_y", seq=-1)
    stacked = SeqStack(axis=0)([x, y])
    assert stacked.seq == (2, None)

    xv = np.random.rand(1, 1, 1, 6).astype(np.float32)
    yv = np.random.rand(1, 1, 1, 6).astype(np.float32)
    np.testing.assert_allclose(
        _run(
            "seq_stack_dyn",
            [x, y],
            stacked,
            {"seq_stack_dyn_x": xv, "seq_stack_dyn_y": yv},
        ),
        np.stack([xv, yv], axis=-2),
    )


def test_seq_reverse():
    x = Input("seq_reverse_x", dim=2, seq=(3, -1))
    inner = SeqReverse()([x])
    outer = SeqReverse(axis=0)([x])
    assert inner.seq == outer.seq == (3, None)

    values = np.random.rand(1, 2, 1, 3, 5).astype(np.float32)
    data = {"seq_reverse_x": values}
    np.testing.assert_allclose(
        _run("seq_reverse_inner", [x], inner, data), values[..., ::-1]
    )
    np.testing.assert_allclose(
        _run("seq_reverse_outer", [x], outer, data), values[..., ::-1, :]
    )


def test_seq_ops_reject_invalid_declarations():
    x = Input("seq_invalid_x", seq=4)
    y = Input("seq_invalid_y", seq=(4, 2))
    plain = Input("seq_invalid_plain")

    with pytest.raises(ValueError, match="out of bounds for seq axis"):
        SeqSelect(4)([x])
    with pytest.raises(ValueError, match="out of bounds for seq rank 0"):
        SeqSelect(-1)([plain.last()])
    with pytest.raises(ValueError, match="out of bounds for seq rank 1"):
        SeqReverse(axis=1)([x])
    with pytest.raises(ValueError, match="invalid range"):
        SeqRange(3, 2)([x])
    with pytest.raises(ValueError, match="invalid range"):
        SeqRange(0, 5)([x])
    with pytest.raises(ValueError, match="matching dim, time and seq"):
        SeqConcatenate()([x, y])
    with pytest.raises(ValueError, match="matching dim, time and seq"):
        SeqStack()([x, y])
    with pytest.raises(ValueError, match="at least two inputs"):
        SeqConcatenate()([x])


def test_seq_select_closes_a_loop_chained_over_chunks():
    # #209: an outer Loop chains the chunks of an inner one, each chunk seeded
    # by the last position and velocity of the one before.
    vx0, px0, ax = Input("chunk_vx0"), Input("chunk_px0"), Input("chunk_ax")
    outvx = Output("chunk_outvx", vx0.last() + ax.last())
    outpx = Output("chunk_outpx", px0.last() + vx0.last())
    step = Modely("chunk_step", inputs=[ax, vx0, px0], outputs=[outvx, outpx]).build()

    ax_sq = Input("chunk_ax_sq", seq=-1)
    px0_init, vx0_init = Input("chunk_px0_init"), Input("chunk_vx0_init")
    velocity, position = Loop(
        f=step,
        callback={vx0: outvx, px0: outpx},
        init={vx0: vx0_init, px0: px0_init},
    )({ax: ax_sq})
    chunk = Modely(
        "chunk",
        inputs=[ax_sq, px0_init, vx0_init],
        outputs=[
            Output("chunk_traj", position),
            Output("chunk_last_px", SeqSelect(-1)([position])),
            Output("chunk_last_vx", SeqSelect(-1)([velocity])),
        ],
    ).build()

    ax_sq_sq = Input("chunk_ax_sq_sq", seq=(-1, -1))
    px0_seed, vx0_seed = Input("chunk_px0_seed"), Input("chunk_vx0_seed")
    trajectory, _, _ = Loop(
        f=chunk,
        callback={"chunk_px0_init": "chunk_last_px", "chunk_vx0_init": "chunk_last_vx"},
        init={"chunk_px0_init": px0_seed, "chunk_vx0_init": vx0_seed},
    )({"chunk_ax_sq": ax_sq_sq})
    assert trajectory.seq == (None, None)
    model = Modely(
        "chunked",
        inputs=[ax_sq_sq, px0_seed, vx0_seed],
        outputs=[Output("chunked_traj", trajectory)],
    ).build()

    steps, chunks = 5, 3
    accelerations = np.random.rand(1, 1, 1, steps, chunks).astype(np.float32)
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
    for acceleration in accelerations[0, 0, 0].T.ravel():
        position_, velocity_ = position_ + velocity_, velocity_ + acceleration
        expected.append(position_)
    np.testing.assert_allclose(
        result[0, 0, 0], np.reshape(expected, (chunks, steps)).T, rtol=1e-5
    )
