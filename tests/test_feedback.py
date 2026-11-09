"""One integrator, y[k] = y[k-1] + dt*u[k], written every way a step can be
rolled out: each declares its feedback and initial value by the same rules."""

import warnings

import numpy as np
import pytest

from conftest import to_numpy
from nnodely import (
    Buffer,
    Fir,
    Input,
    Integrate,
    IntegrateStep,
    Loop,
    Modely,
    ModelyLoop,
    Output,
)

STEPS = 5
FROM_ZERO = [1.0, 2.0, 3.0, 4.0, 5.0]
FROM_TEN = [11.0, 12.0, 13.0, 14.0, 15.0]


def _u():
    return np.ones((1, STEPS, 1, 1), np.float32)


def _y0(steps=None):
    """10, one sample, or the first of ``steps`` along a horizon."""
    shape = (1, 1, 1) if steps is None else (1, steps, 1, 1)
    return np.full(shape, 10.0, np.float32)


def _rolled(model, data, output):
    return to_numpy(model(data)[output]).ravel()


def _integrate_step_loop(tag, seeded):
    # The state is no fed-back Input of the graph: the Loop is told.
    u, y = Input(f"{tag}_u", tw=1), Input(f"{tag}_y", tw=1)
    step = Modely(
        f"{tag}_step",
        inputs=[u, y],
        outputs=[Output(f"{tag}_next", IntegrateStep(dt=1)(y, u))],
    )
    u_seq, y0 = Input(f"{tag}_u_seq", seq=-1, tw=1), Input(f"{tag}_y0", tw=1)
    init = {y: y0} if seeded else None
    trajectory = Loop(f=step, feedback={y: f"{tag}_next"}, init=init)(
        {f"{tag}_u": u_seq}
    )
    return Modely(
        tag,
        inputs=[u_seq, y0] if seeded else [u_seq],
        outputs=[Output(f"{tag}_traj", trajectory)],
    )


def _feedback_loop(tag, seeded):
    # The graph feeds y back: the Loop finds it.
    u, y = Input(f"{tag}_u", tw=1), Input(f"{tag}_y", tw=1)
    y_next = y.feedback(IntegrateStep(dt=1)(y, u))
    step = Modely(f"{tag}_step", inputs=[u, y], outputs=[Output(f"{tag}_next", y_next)])
    u_seq, y0 = Input(f"{tag}_u_seq", seq=-1, tw=1), Input(f"{tag}_y0", tw=1)
    init = {y: y0} if seeded else None
    trajectory = Loop(f=step, init=init)({f"{tag}_u": u_seq})
    return Modely(
        tag,
        inputs=[u_seq, y0] if seeded else [u_seq],
        outputs=[Output(f"{tag}_traj", trajectory)],
    )


def _integrate_loop(tag, seeded):
    # The state of Integrate is hidden: the Loop finds it, and it starts from
    # the init of Integrate, an input of the step bound like any other.
    u, y0 = Input(f"{tag}_u", tw=1), Input(f"{tag}_y0", tw=1)
    y_next = Integrate(dt=1, init=y0 if seeded else None)(u)
    step = Modely(
        f"{tag}_step",
        inputs=[u, y0] if seeded else [u],
        outputs=[Output(f"{tag}_next", y_next)],
    )
    u_seq, y0_init = Input(f"{tag}_u_seq", seq=-1, tw=1), Input(f"{tag}_y0i", tw=1)
    inputs = {f"{tag}_u": u_seq, **({f"{tag}_y0": y0_init} if seeded else {})}
    trajectory = Loop(f=step)(inputs)
    return Modely(
        tag,
        inputs=[u_seq, y0_init] if seeded else [u_seq],
        outputs=[Output(f"{tag}_traj", trajectory)],
    )


def _integrate_modelyloop(tag, seeded):
    u, y0 = Input(f"{tag}_u", tw=1), Input(f"{tag}_y0", tw=1)
    y_next = Integrate(dt=1, init=y0 if seeded else None)(u)
    return ModelyLoop(
        tag,
        inputs=[u, y0] if seeded else [u],
        outputs=[Output(f"{tag}_traj", y_next)],
    )


def _feedback_modelyloop(tag, seeded):
    # Among the inputs, y starts from its data.
    u, y = Input(f"{tag}_u", tw=1), Input(f"{tag}_y", tw=1)
    y_next = y.feedback(IntegrateStep(dt=1)(y, u))
    return ModelyLoop(
        tag,
        inputs=[u, y] if seeded else [u],
        outputs=[Output(f"{tag}_traj", y_next)],
    )


# What builds each, the Input of its signal and of its initial value, and
# whether that carries the horizon: a ModelyLoop reads every Input step by step.
CONSTRUCTIONS = {
    "integrate_step_loop": (_integrate_step_loop, "u_seq", "y0", None),
    "feedback_loop": (_feedback_loop, "u_seq", "y0", None),
    "integrate_loop": (_integrate_loop, "u_seq", "y0i", None),
    "integrate_modelyloop": (_integrate_modelyloop, "u", "y0", STEPS),
    "feedback_modelyloop": (_feedback_modelyloop, "u", "y", STEPS),
}


@pytest.mark.parametrize("seeded", [False, True], ids=["from_zero", "from_y0"])
@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_every_way_of_writing_the_integrator_rolls_out_the_same(
    construction, seeded, tmp_path
):
    build, signal, start, steps = CONSTRUCTIONS[construction]
    tag = f"fb_{construction}"
    model = build(tag, seeded)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.build()
    data = {f"{tag}_{signal}": _u()}
    if seeded:
        data[f"{tag}_{start}"] = _y0(steps)  # type: ignore
    expected = FROM_TEN if seeded else FROM_ZERO
    np.testing.assert_allclose(_rolled(model, data, f"{tag}_traj"), expected)

    model.save(tmp_path / tag)
    loaded = Modely.load(tmp_path / tag)
    np.testing.assert_allclose(_rolled(loaded, data, f"{tag}_traj"), expected)


def test_a_loop_finds_the_feedback_of_a_loaded_step(tmp_path):
    # Saved with the step, the feedback is found again once it is loaded.
    u, y, y0 = Input("fb_ld_u", tw=1), Input("fb_ld_y", tw=1), Input("fb_ld_y0", tw=1)
    y_next = y.feedback(y + Integrate(dt=1, init=y0)(u))
    step = Modely(
        "fb_ld_step", inputs=[u, y, y0], outputs=[Output("fb_ld_next", y_next)]
    ).build()
    step.save(tmp_path / "fb_ld_step")
    loaded = Modely.load(tmp_path / "fb_ld_step")

    def rolled(body, tag):
        u_seq = Input(f"{tag}_u", seq=-1, tw=1)
        y_init, y0_init = Input(f"{tag}_y", tw=1), Input(f"{tag}_y0", tw=1)
        trajectory = Loop(f=body, init={"fb_ld_y": y_init})(
            {"fb_ld_u": u_seq, "fb_ld_y0": y0_init}
        )
        model = Modely(
            tag,
            inputs=[u_seq, y_init, y0_init],
            outputs=[Output(f"{tag}_traj", trajectory)],
        ).build()
        data = {f"{tag}_u": _u(), f"{tag}_y": _y0(), f"{tag}_y0": _y0() * 0.5}
        return _rolled(model, data, f"{tag}_traj")

    # y = 10, then y + the integral, which starts from 5 and grows by one.
    expected = np.cumsum([16.0, 7.0, 8.0, 9.0, 10.0])
    np.testing.assert_allclose(rolled(step, "fb_ld_orig"), expected)
    np.testing.assert_allclose(rolled(loaded, "fb_ld_back"), expected)


def test_two_integrates_that_start_from_one_input_bind_it_once():
    u, y0 = Input("fb_two_u", tw=1), Input("fb_two_y0", tw=1)
    step = Modely(
        "fb_two_step",
        inputs=[u, y0],
        outputs=[
            Output("fb_two_slow", Integrate(dt=1, init=y0)(u)),
            Output("fb_two_fast", Integrate(dt=2, init=y0)(u)),
        ],
    )
    loop = Loop(f=step)
    assert [node.name for node in loop._seed_inputs] == ["fb_two_y0"]
    u_seq, y0_init = Input("fb_two_u_seq", seq=-1, tw=1), Input("fb_two_y0i", tw=1)
    slow, fast = loop({"fb_two_u": u_seq, "fb_two_y0": y0_init})
    model = Modely(
        "fb_two",
        inputs=[u_seq, y0_init],
        outputs=[Output("fb_two_s", slow), Output("fb_two_f", fast)],
    ).build()
    result = model({"fb_two_u_seq": _u(), "fb_two_y0i": _y0()})
    np.testing.assert_allclose(to_numpy(result["fb_two_s"]).ravel(), FROM_TEN)
    np.testing.assert_allclose(
        to_numpy(result["fb_two_f"]).ravel(), [12.0, 14.0, 16.0, 18.0, 20.0]
    )


def test_feedback_given_to_the_loop_adds_to_the_one_the_graph_declares():
    # y is fed back by the graph, x by the Loop; each reads the other.
    u = Input("fb_mix_u", tw=1)
    x, y = Input("fb_mix_x", tw=1), Input("fb_mix_y", tw=1)
    y_next = y.feedback(y + x)
    x_next = Output("fb_mix_x_next", x + u)
    step = Modely(
        "fb_mix_step",
        inputs=[u, x, y],
        outputs=[x_next, Output("fb_mix_y_next", y_next)],
    )
    u_seq, x0 = Input("fb_mix_u_seq", seq=-1, tw=1), Input("fb_mix_x0", tw=1)
    loop = Loop(f=step, feedback={x: x_next}, init={x: x0})({u: u_seq})
    xs, ys = loop
    model = Modely(
        "fb_mix",
        inputs=[u_seq, x0],
        outputs=[Output("fb_mix_xs", xs), Output("fb_mix_ys", ys)],
    ).build()
    result = model({"fb_mix_u_seq": _u(), "fb_mix_x0": _y0()})
    np.testing.assert_allclose(to_numpy(result["fb_mix_xs"]).ravel(), FROM_TEN)
    # y reads x before the step: 10, 11, ... added up.
    np.testing.assert_allclose(
        to_numpy(result["fb_mix_ys"]).ravel(), np.cumsum([10.0, 11, 12, 13, 14])
    )


def test_a_loop_returns_the_outputs_of_the_step_not_the_ones_it_adds(tmp_path):
    # The update of y is no output: the body gains one for it, which the Loop
    # does not return, so it is the one output of the step, a stream.
    u, y = Input("fb_ret_u", tw=1), Input("fb_ret_y", tw=1)
    y.feedback(y + u)
    step = Modely("fb_ret_step", inputs=[u], outputs=[Output("fb_ret_twice", y * 2.0)])
    loop = Loop(f=step)
    assert [output.name for output in loop._f.outputs] == [
        "fb_ret_twice",
        "fb_ret_y_next",
    ]
    u_seq = Input("fb_ret_u_seq", seq=-1, tw=1)
    model = Modely(
        "fb_ret",
        inputs=[u_seq],
        outputs=[Output("fb_ret_traj", loop({"fb_ret_u": u_seq}))],
    ).build()
    data = {"fb_ret_u_seq": _u()}
    # y is read before its update: 0, 1, 2, ... doubled.
    expected = [0.0, 2.0, 4.0, 6.0, 8.0]
    np.testing.assert_allclose(_rolled(model, data, "fb_ret_traj"), expected)
    model.save(tmp_path / "fb_ret")
    loaded = Modely.load(tmp_path / "fb_ret")
    np.testing.assert_allclose(_rolled(loaded, data, "fb_ret_traj"), expected)


def test_a_loop_refuses_a_feedback_or_an_initial_value_it_cannot_apply():
    u, s0 = Input("fb_err_u", tw=1), Input("fb_err_s0", tw=1)
    u_seq = Input("fb_err_u_seq", seq=-1, tw=1)

    plain = Modely("fb_err_plain", inputs=[u], outputs=[Output("fb_err_po", u * 2.0)])
    with pytest.raises(ValueError, match="'fb_err_plain' has no fed-back Input"):
        Loop(f=plain)

    y = Input("fb_err_y", tw=1)
    y_next = y.feedback(y + u)
    step = Modely(
        "fb_err_step",
        inputs=[u, y],
        outputs=[Output("fb_err_next", y_next), Output("fb_err_twice", y * 2.0)],
    )
    with pytest.raises(ValueError, match="fed back with one stream"):
        Loop(f=step, feedback={y: "fb_err_twice"})
    Loop(f=step, feedback={y: "fb_err_next"})  # the same feedback: redundant
    with pytest.raises(ValueError, match="'fb_err_y' is fed back: give its initial"):
        Loop(f=step)({"fb_err_u": u_seq, "fb_err_y": s0})

    # The init of Integrate is an input of the step, not a state; its output
    # is no input; and an init the step does not declare is refused.
    y0 = Input("fb_err_y0", tw=1)
    integral = Integrate(dt=1, init=y0)(u)
    step = Modely(
        "fb_err_int", inputs=[u, y0], outputs=[Output("fb_err_int_o", integral)]
    )
    with pytest.raises(ValueError, match="'fb_err_y0' is not fed back: pass it"):
        Loop(f=step, init={y0: s0})
    with pytest.raises(ValueError, match="'fb_err_int_o' is not an input of"):
        Loop(f=step, init={"fb_err_int_o": s0})
    undeclared = Modely(
        "fb_err_und", inputs=[u], outputs=[Output("fb_err_und_o", integral)]
    )
    with pytest.raises(ValueError, match=r"reads the Inputs \['fb_err_y0'\]"):
        Loop(f=undeclared)

    # An Input the step reads, which nothing feeds back, is declared.
    w, z = Input("fb_err_w", tw=1), Input("fb_err_z", tw=1)
    z.feedback(z + w)
    reads = Modely("fb_err_reads", inputs=[z], outputs=[Output("fb_err_ro", z + u)])
    with pytest.raises(
        ValueError, match=r"reads the Inputs \['fb_err_u', 'fb_err_w'\]"
    ):
        Loop(f=reads)


def test_a_loop_with_several_outputs_returns_them_all(tmp_path):
    u, y = Input("fb_all_u", tw=1), Input("fb_all_y", tw=1)
    y_next = y.feedback(y + u)
    step = Modely(
        "fb_all_step",
        inputs=[u, y],
        outputs=[Output("fb_all_next", y_next), Output("fb_all_twice", y_next * 2.0)],
    )
    u_seq = Input("fb_all_u_seq", seq=-1, tw=1)
    loop = Loop(f=step)({u: u_seq})

    # Read as one stream, it raises: an Output, an operator, a layer.
    for read in (
        lambda: Output("fb_all_one", loop),
        lambda: loop + 1.0,
        lambda: Fir()(loop),
    ):
        with pytest.raises(ValueError, match=r"returns the 2 outputs .* unpack"):
            read()

    assert "fb_all_twice" in repr(loop)
    y_traj, twice = loop
    assert loop[1] is twice
    model = Modely(
        "fb_all",
        inputs=[u_seq],
        outputs=[Output("fb_all_y_traj", y_traj), Output("fb_all_2y", twice)],
    ).build()
    data = {"fb_all_u_seq": _u()}
    expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_allclose(_rolled(model, data, "fb_all_y_traj"), expected)
    np.testing.assert_allclose(
        _rolled(model, data, "fb_all_2y"), [2 * v for v in expected]
    )
    model.save(tmp_path / "fb_all")
    loaded = Modely.load(tmp_path / "fb_all")
    np.testing.assert_allclose(
        _rolled(loaded, data, "fb_all_2y"), [2 * v for v in expected]
    )


def test_a_state_starts_from_zero_along_a_dynamic_sequence(tmp_path):
    # A state with a dynamic sequence axis that starts from zero, or from a
    # number, takes the length of that axis from the value fed back.
    x = Input("fb_dyn_x", tw=1)
    inner = ModelyLoop(
        "fb_dyn_inner",
        inputs=[x],
        outputs=[Output("fb_dyn_xx", Integrate(dt=1)(x))],
    ).build()
    k = Input("fb_dyn_k", tw=1, seq=-1)
    s = Input("fb_dyn_s", tw=1, seq=-1)
    model = ModelyLoop(
        "fb_dyn",
        inputs=[k],
        outputs=[
            # The inner loop rolls out along the sequence of k, the outer one
            # integrates what it returns.
            Output("fb_dyn_kk", Integrate(dt=1)(inner([k]))),
            Output("fb_dyn_from_two", Integrate(dt=1, init=2.0)(k)),
            Output("fb_dyn_buf", Buffer(tw=2)(k)),
            Output("fb_dyn_sum", s.feedback(s + k)),
        ],
    ).build()

    # Three steps of a sequence of four, 1 to 12.
    data = {"fb_dyn_k": np.arange(1, 13, dtype=np.float32).reshape(1, 3, 4, 1, 1)}
    rows = data["fb_dyn_k"].reshape(3, 4)
    expected = {
        "fb_dyn_kk": np.cumsum(np.cumsum(rows, axis=1), axis=0),
        "fb_dyn_from_two": 2.0 + np.cumsum(rows, axis=0),
        "fb_dyn_buf": np.stack([np.vstack([np.zeros(4), rows[:-1]]), rows], -1),
        "fb_dyn_sum": np.cumsum(rows, axis=0),
    }

    def check(rolled):
        result = rolled(data)
        for name, values in expected.items():
            np.testing.assert_allclose(
                to_numpy(result[name]).reshape(values.shape), values
            )

    check(model)
    model.save(tmp_path / "fb_dyn")
    check(Modely.load(tmp_path / "fb_dyn"))


def test_a_dynamic_initial_value_feeds_a_state_of_a_static_sequence():
    # A Loop pinned to three steps returns a sequence of 3; the init of the
    # Integrate on it is read from data of a dynamic sequence, which matches.
    x = Input("fb_pin_x", tw=1)
    step = Modely(
        "fb_pin_step", inputs=[x], outputs=[Output("fb_pin_xx", Integrate(dt=1)(x))]
    ).build()
    k_seq = Input("fb_pin_k_seq", tw=1, seq=-1)
    inner = Modely(
        "fb_pin_inner",
        inputs=[k_seq],
        outputs=[Output("fb_pin_rows", Loop(f=step, length=3)({x: k_seq}))],
    ).build()
    k, k0 = Input("fb_pin_k", tw=1, seq=-1), Input("fb_pin_k0", tw=1, seq=-1)
    rate = inner([k])
    assert tuple(rate.seq) == (3,)
    model = ModelyLoop(
        "fb_pin",
        inputs=[k, k0],
        outputs=[Output("fb_pin_kk", Integrate(dt=1, init=k0)(rate))],
    ).build()

    rows = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    data = {
        "fb_pin_k": rows.reshape(1, 4, 3, 1, 1),
        "fb_pin_k0": np.full((1, 4, 3, 1, 1), 100.0, np.float32),
    }
    expected = 100.0 + np.cumsum(np.cumsum(rows, axis=1), axis=0)
    np.testing.assert_allclose(
        to_numpy(model(data)["fb_pin_kk"]).reshape(4, 3), expected
    )


def test_a_loop_is_called_with_a_dict_a_list_or_its_one_stream():
    # The list skips the fed-back y, and takes y0, the init of the Integrate.
    u, y, y0 = (
        Input("fb_call_u", tw=1),
        Input("fb_call_y", tw=1),
        Input("fb_call_y0", tw=1),
    )
    y_next = y.feedback(y * 0.0 + Integrate(dt=1, init=y0)(u))
    step = Modely(
        "fb_call_step", inputs=[u, y, y0], outputs=[Output("fb_call_next", y_next)]
    )
    loop = Loop(f=step)
    u_seq, y0_init = Input("fb_call_u_seq", seq=-1, tw=1), Input("fb_call_y0i", tw=1)
    data = {"fb_call_u_seq": _u(), "fb_call_y0i": _y0()}

    def rolled(called, tag):
        model = Modely(
            tag, inputs=[u_seq, y0_init], outputs=[Output(f"{tag}_traj", called)]
        ).build()
        return _rolled(model, data, f"{tag}_traj")

    by_dict = rolled(loop({u: u_seq, "fb_call_y0": y0_init}), "fb_call_dict")
    np.testing.assert_allclose(by_dict, FROM_TEN)
    np.testing.assert_allclose(rolled(loop([u_seq, y0_init]), "fb_call_list"), by_dict)
    np.testing.assert_allclose(rolled(loop((u_seq, y0_init)), "fb_call_tup"), by_dict)

    with pytest.raises(ValueError, match=r"has 2 inputs \['fb_call_u', 'fb_call_y0'\]"):
        loop(u_seq)
    with pytest.raises(ValueError, match="one stream for each, in that order"):
        loop([u_seq])
    with pytest.raises(TypeError, match="a list of streams, one per input"):
        loop(3.0)  # type: ignore[arg-type]

    # With one input, its stream alone.
    x = Input("fb_call_x", tw=1)
    single = Modely(
        "fb_call_single",
        inputs=[x],
        outputs=[Output("fb_call_xx", Integrate(dt=1)(x))],
    )
    model = Modely(
        "fb_call_one",
        inputs=[u_seq],
        outputs=[Output("fb_call_one_traj", Loop(f=single)(u_seq))],
    ).build()
    np.testing.assert_allclose(
        _rolled(model, {"fb_call_u_seq": _u()}, "fb_call_one_traj"), FROM_ZERO
    )


def test_close_and_callback_are_gone():
    x = Input("fb_gone_x", tw=1)
    assert not hasattr(x, "close")
    step = Modely("fb_gone", inputs=[x], outputs=[Output("fb_gone_o", x + 1.0)])
    with pytest.raises(TypeError, match="callback"):
        Loop(f=step, callback={x: "fb_gone_o"})  # type: ignore[call-arg]
