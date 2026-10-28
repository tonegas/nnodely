import warnings

import numpy as np
import pytest

from conftest import to_numpy
from nnodely import (
    Buffer,
    DataLoader,
    Derivative,
    Fir,
    Input,
    Integrate,
    IntegrateStep,
    Modely,
    ModelyLoop,
    Output,
    Parameter,
)
from nnodely.layers.loop import LoopImpl


def _ramp(steps, batch=1):
    """1, 2, ..., steps along the horizon of an Input of one sample."""
    return np.tile(np.arange(1, steps + 1, dtype=np.float32), (batch, 1)).reshape(
        batch, steps, 1, 1
    )


def _loops(model):
    return sum(isinstance(layer, LoopImpl) for layer in model.inference_model.layers)


def test_chained_closed_inputs_are_one_loop_as_chained_integrates():
    dt = 0.1
    acc = Input("oc_chain_acc", sw=1)
    vel, pos = Input("oc_chain_v", sw=1), Input("oc_chain_x", sw=1)
    vel_next = vel.close(IntegrateStep(dt=dt)(vel, acc))
    pos_next = pos.close(IntegrateStep(dt=dt)(pos, vel_next))
    model = ModelyLoop(
        "oc_chain",
        inputs=[acc],
        outputs=[Output("oc_chain_vel", vel_next), Output("oc_chain_pos", pos_next)],
    ).build()
    # Integrate closes the same feedbacks by itself.
    blocks = ModelyLoop(
        "oc_chain_blocks",
        inputs=[acc],
        outputs=[
            Output("oc_chain_blocks_pos", Integrate(dt=dt)(Integrate(dt=dt)(acc)))
        ],
    ).build()

    data = np.random.default_rng(0).standard_normal((3, 9, 1, 1)).astype(np.float32)
    expected = dt * np.cumsum(dt * np.cumsum(data, axis=1), axis=1)
    result = to_numpy(model({"oc_chain_acc": data})["oc_chain_pos"])
    assert result.shape == (3, 9, 1, 1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    result = to_numpy(blocks({"oc_chain_acc": data})["oc_chain_blocks_pos"])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    assert _loops(model) == 1 and _loops(blocks) == 1


def test_coupled_states_simulate_a_mass_spring_damper():
    # Each closed Input reads the other: a cycle no chain of Loops can express.
    dt, k, c, m = 0.05, 4.0, 0.5, 2.0
    force = Input("oc_msd_force", sw=1)
    x, v = Input("oc_msd_x", sw=1), Input("oc_msd_v", sw=1)
    acceleration = (force - x * k - v * c) / m
    v_next = v.close(IntegrateStep(dt=dt)(v, acceleration))
    x.close(IntegrateStep(dt=dt)(x, v_next))
    model = ModelyLoop(
        "oc_msd",
        inputs=[force, x, v],
        outputs=[Output("oc_msd_pos", x), Output("oc_msd_vel", v)],
    ).build()

    steps = 25
    forces = np.random.default_rng(1).uniform(-1, 1, (1, steps, 1, 1))
    # Listed among the inputs, x and v start from the first step of their data.
    result = model(
        {
            "oc_msd_force": forces.astype(np.float32),
            "oc_msd_x": np.full((1, steps, 1, 1), 0.3, np.float32),
            "oc_msd_v": np.full((1, steps, 1, 1), -1.0, np.float32),
        }
    )
    position, velocity, expected = 0.3, -1.0, []
    for step in range(steps):
        # x is the value at the start of the step.
        expected.append(position)
        velocity += dt * (forces[0, step, 0, 0] - k * position - c * velocity) / m
        position += dt * velocity
    np.testing.assert_allclose(
        to_numpy(result["oc_msd_pos"]).ravel(), expected, rtol=1e-4, atol=1e-6
    )
    assert to_numpy(result["oc_msd_vel"]).ravel()[0] == pytest.approx(-1.0)


def test_buffer_holds_the_current_sample_and_shifts_after_the_step():
    u = Input("oc_buf_u", sw=1)
    window = Buffer(sw=3)(u)
    counter = Input("oc_buf_counter", sw=1)
    counter.close(counter + 1.0)
    model = ModelyLoop(
        "oc_buf",
        inputs=[u],
        outputs=[Output("oc_buf_window", window), Output("oc_buf_count", counter)],
    ).build()

    result = to_numpy(model({"oc_buf_u": _ramp(4)})["oc_buf_window"])
    assert result.shape == (1, 4, 3, 1)
    np.testing.assert_allclose(
        result.squeeze(), [[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]
    )


def test_buffer_of_a_signal_without_a_window_starts_from_init():
    u = Input("oc_buf_init_u", sw=1)
    count = Input("oc_buf_init_count")
    signal = count.close(count + 1.0)  # 1, 2, 3, ... with no window
    window = Buffer(sw=3, init=-1.0)(signal)
    model = ModelyLoop(
        "oc_buf_init",
        inputs=[u],
        outputs=[Output("oc_buf_init_window", window + u * 0.0)],
    ).build()

    result = to_numpy(model({"oc_buf_init_u": _ramp(3)})["oc_buf_init_window"])
    np.testing.assert_allclose(result.squeeze(), [[-1, -1, 1], [-1, 1, 2], [1, 2, 3]])


def test_buffer_feeds_a_plant_that_reads_a_window_of_the_command():
    # The controller computes u one sample per step; the plant reads three.
    reference = Input("oc_ctrl_ref", sw=1)
    y = Input("oc_ctrl_y", sw=1)
    u = (reference - y) * 0.5
    u_in = Input("oc_ctrl_u_in", sw=3)
    plant = Modely(
        "oc_ctrl_plant",
        inputs=[u_in],
        outputs=[
            Output(
                "oc_ctrl_plant_y",
                Fir(
                    kernel=Parameter("oc_ctrl_kernel", value=[[0.2], [0.3], [0.5]]),
                    bias=False,
                )([u_in]),
            )
        ],
    )
    y_next = y.close(plant([Buffer(sw=3)(u)]))
    model = ModelyLoop(
        "oc_ctrl", inputs=[reference], outputs=[Output("oc_ctrl_out", y_next)]
    ).build()

    steps = 6
    result = model({"oc_ctrl_ref": np.ones((1, steps, 1, 1), np.float32)})
    commands, outputs, last = [0.0, 0.0], [], 0.0
    for _ in range(steps):
        commands.append(0.5 * (1.0 - last))
        last = 0.2 * commands[-3] + 0.3 * commands[-2] + 0.5 * commands[-1]
        outputs.append(last)
    np.testing.assert_allclose(
        to_numpy(result["oc_ctrl_out"]).ravel(), outputs, rtol=1e-5
    )
    # The plant's output closes y, which its own argument reads: still a
    # single call of the plant, one buffer.
    closed = [node for node in model.body.inputs if node.update is not None]
    assert len(closed) == 2


def test_a_model_called_twice_has_a_feedback_per_call():
    inflow = Input("oc_tank_q", sw=1)
    level = Input("oc_tank_h", sw=1)
    tank = Modely(
        "oc_tank",
        inputs=[inflow],
        outputs=[Output("oc_tank_level", level.close(level * 0.5 + inflow))],
    )
    source = Input("oc_tank_source", sw=1)
    first = tank([source])
    second = tank([first])
    model = ModelyLoop(
        "oc_cascade",
        inputs=[source],
        outputs=[Output("oc_cascade_1", first), Output("oc_cascade_2", second)],
    ).build()

    result = model({"oc_tank_source": np.ones((1, 4, 1, 1), np.float32)})
    np.testing.assert_allclose(
        to_numpy(result["oc_cascade_1"]).ravel(), [1, 1.5, 1.75, 1.875]
    )
    np.testing.assert_allclose(
        to_numpy(result["oc_cascade_2"]).ravel(), [1, 2, 2.75, 3.25]
    )
    assert [node.name for node in model.body.inputs] == [
        "oc_tank_source",
        "oc_tank_h",
        "oc_tank_h_2",
    ]


def test_integrate_and_derivative_close_inputs_of_their_own():
    dt, steps = 0.1, 8
    data = np.random.default_rng(2).standard_normal((2, steps, 1, 1)).astype(np.float32)
    acc, v0 = Input("oc_hl_acc", sw=1), Input("oc_hl_v0", sw=1)
    model = ModelyLoop(
        "oc_hl",
        inputs=[acc, v0],
        outputs=[
            Output("oc_hl_pos", Integrate(dt=dt)(Integrate(dt=dt, init=v0)(acc))),
            Output("oc_hl_trap", Integrate(solver="trapezoidal", dt=dt)(acc)),
            Output("oc_hl_der", Derivative(dt=dt)(acc)),
        ],
    ).build()

    result = model({"oc_hl_acc": data, "oc_hl_v0": np.full_like(data, 0.5)})
    a = data[..., 0, 0]
    # The trapezoid's first step has no previous rate: it keeps the rectangle.
    previous = np.concatenate([a[:, :1], a[:, :-1]], axis=1)
    expected = {
        "pos": dt * np.cumsum(0.5 + dt * np.cumsum(a, axis=1), axis=1),
        "trap": dt * np.cumsum((previous + a) / 2, axis=1),
        "der": np.diff(a, axis=1, prepend=0.0) / dt,
    }
    for name, values in expected.items():
        np.testing.assert_allclose(
            to_numpy(result[f"oc_hl_{name}"])[..., 0, 0], values, rtol=1e-5, atol=1e-4
        )
    # Four blocks, one loop: none of them rolls out a Loop of its own.
    assert _loops(model) == 1


def test_a_call_gives_a_declared_closed_input_its_data():
    rate = Input("oc_decl_rate", sw=1)
    level = Input("oc_decl_level", sw=1)
    level_next = level.close(level * 0.5 + rate, init=5.0)
    # The tank declares its level: built as a Modely, the one-step model.
    tank = Modely(
        "oc_decl_tank", inputs=[level, rate], outputs=[Output("oc_decl", level_next)]
    )
    start, source = Input("oc_decl_start", sw=1), Input("oc_decl_source", sw=1)

    # In a ModelyLoop the level keeps its feedback, and the stream the call
    # gives it is its initial value, in place of its init: only the first
    # step of it is read.
    loop = ModelyLoop(
        "oc_decl_loop",
        inputs=[start, source],
        outputs=[Output("oc_decl_loop_out", tank([start, source]))],
    ).build()
    starts = np.full((2, 4, 1, 1), 100.0, np.float32)
    starts[:, 0, 0, 0] = [1.0, -2.0]
    result = loop({"oc_decl_start": starts, "oc_decl_source": np.ones_like(starts)})
    np.testing.assert_allclose(
        to_numpy(result["oc_decl_loop_out"])[..., 0, 0],
        [[1.5, 1.75, 1.875, 1.9375], [0.0, 1.0, 1.5, 1.75]],
    )
    row = next(
        row for row in loop.all_inputs.rows if row["input"].name == "oc_decl_level"
    )
    assert row["starts from"] == "oc_decl_start, first step"

    # In a Modely it reads the stream at every step, as it would its data.
    plain = Modely(
        "oc_decl_plain",
        inputs=[start, source],
        outputs=[Output("oc_decl_plain_out", tank([start, source]))],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no closed Input left to read
        plain.build()
    result = plain(
        {
            "oc_decl_start": np.array([1.0, -2.0], np.float32).reshape(2, 1, 1),
            "oc_decl_source": np.ones((2, 1, 1), np.float32),
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["oc_decl_plain_out"]).ravel(), [1.5, 0.0]
    )

    # An initial value is read once, before the first step: not computed.
    with pytest.raises(ValueError, match="computed from the Inputs"):
        ModelyLoop(
            "oc_decl_expr",
            inputs=[start, source],
            outputs=[Output("oc_decl_expr_out", tank([start * 2.0, source]))],
        )


def test_modely_warns_about_closed_inputs_and_reads_them_from_data():
    rate = Input("oc_open_rate", sw=1)
    state = Input("oc_open_state", sw=1)
    one_step = Output("oc_open_next", state.close(state + rate))
    model = Modely("oc_open", inputs=[rate], outputs=[one_step])
    with pytest.warns(
        UserWarning, match=r"reads the closed Inputs \['oc_open_state'\]"
    ):
        model.build()
    result = model(
        {
            "oc_open_rate": np.full((2, 1, 1), 2.0, np.float32),
            "oc_open_state": np.full((2, 1, 1), 5.0, np.float32),
        }
    )
    np.testing.assert_allclose(to_numpy(result["oc_open_next"]).ravel(), [7.0, 7.0])

    # Declared among the inputs, the Input is read from data on purpose.
    declared = Modely("oc_open_declared", inputs=[rate, state], outputs=[one_step])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        declared.build()


def test_all_inputs_tells_how_each_input_is_fed():
    rate = Input("oc_tab_rate", sw=1)
    level = Input("oc_tab_level", sw=1)
    level_next = level.close(level * 0.5 + rate)
    tank = Modely(
        "oc_tab_tank", inputs=[rate], outputs=[Output("oc_tab_out", level_next)]
    )
    tank.minimize("fit", "oc_tab_out", Input("oc_tab_target", sw=1))

    rows = {row["input"].name: row for row in tank.all_inputs.rows}
    assert [node.name for node in tank.all_inputs] == list(rows)
    assert rows.keys() == {"oc_tab_level", "oc_tab_rate", "oc_tab_target"}
    assert rows["oc_tab_level"]["declared"] is False
    assert rows["oc_tab_level"]["closed with"] == level_next.name
    assert rows["oc_tab_level"]["data"] == "required"  # a Modely reads it from data
    assert (
        rows["oc_tab_rate"]["declared"] and rows["oc_tab_rate"]["closed with"] is None
    )
    assert rows["oc_tab_target"]["data"] == "target"

    # In a ModelyLoop: listed, the tank's level starts from its data; the
    # state of an Integrate starts from its init, read at the first step.
    source, x0 = Input("oc_tab_source", sw=1), Input("oc_tab_x0", sw=1)
    loop = ModelyLoop(
        "oc_tab_loop",
        inputs=[source, level, x0],
        outputs=[
            Output("oc_tab_loop_out", tank([source])),
            Output("oc_tab_integral", Integrate(dt=0.1, init=x0)(source)),
        ],
    )
    rows = {row["input"].name: row for row in loop.all_inputs.rows}
    assert (
        rows["oc_tab_level"]["declared"] and rows["oc_tab_level"]["data"] == "optional"
    )
    assert rows["oc_tab_level"]["starts from"] == "its data, else 0"
    assert rows["oc_tab_source"]["data"] == "required"
    (state,) = [row for name, row in rows.items() if name.endswith("_state")]
    assert state["data"] is None and state["starts from"] == "oc_tab_x0, first step"
    text = str(loop.all_inputs)
    assert text.startswith("Inputs of oc_tab_loop") and "4 inputs, 2 closed" in text


def test_training_identifies_a_spring_from_a_measured_trajectory(tmp_path):
    dt, k_true, steps = 0.05, 4.0, 300
    rng = np.random.default_rng(0)
    forces = rng.uniform(-1, 1, steps).astype(np.float32)
    positions = np.zeros(steps, np.float32)
    velocities = np.zeros(steps, np.float32)
    position = velocity = 0.0
    for step in range(steps):
        positions[step], velocities[step] = position, velocity
        velocity += dt * (forces[step] - k_true * position)
        position += dt * velocity

    force = Input("oc_fit_force", sw=1)
    x_meas, v_meas = Input("oc_fit_x", sw=1), Input("oc_fit_v", sw=1)
    k = Parameter("oc_fit_k", value=1.0)
    x, v = Input("oc_fit_x_state", sw=1), Input("oc_fit_v_state", sw=1)
    v_next = v.close(IntegrateStep(dt=dt)(v, force - k * x), init=v_meas)
    x.close(IntegrateStep(dt=dt)(x, v_next), init=x_meas)
    x_hat = Output("oc_fit_x_hat", x)
    model = ModelyLoop("oc_fit", inputs=[force, x_meas, v_meas], outputs=[x_hat])
    # Step streams: the output and the measured Input, compared as trajectories.
    model.minimize("fit", x_hat, x_meas)
    model.build()
    data = DataLoader(
        model,
        source={"oc_fit_force": forces, "oc_fit_x": positions, "oc_fit_v": velocities},
        seq_length=20,
    )
    model.train(data, epochs=100, batch_size=32, lr=0.05, printer=None)
    np.testing.assert_allclose(to_numpy(k.param).ravel(), [k_true], rtol=1e-2)

    model.save(tmp_path / "oc_fit")
    loaded = Modely.load(tmp_path / "oc_fit")
    batch = data.get_samples(3)
    np.testing.assert_allclose(
        to_numpy(loaded(batch)["oc_fit_x_hat"]),
        to_numpy(model(batch)["oc_fit_x_hat"]),
        rtol=1e-6,
    )


def test_a_dynamic_horizon_follows_the_data_and_length_pins_it():
    rate = Input("oc_len_rate", sw=1)
    total = Input("oc_len_total", sw=1)
    total_next = total.close(total + rate)
    dynamic = ModelyLoop(
        "oc_len_dynamic", inputs=[rate], outputs=[Output("oc_len_out", total_next)]
    ).build()
    for steps in (2, 7):
        result = to_numpy(dynamic({"oc_len_rate": _ramp(steps)})["oc_len_out"])
        np.testing.assert_allclose(result.ravel(), np.cumsum(np.arange(1, steps + 1)))

    fixed = ModelyLoop(
        "oc_len_fixed",
        inputs=[rate],
        outputs=[Output("oc_len_fixed_out", total_next)],
        length=5,
    )
    assert fixed.inputs[0].seq == (5,)


def test_a_closed_input_starts_from_its_data_if_listed_else_from_zero(tmp_path):
    rate = Input("oc_seed_rate", sw=1)
    total = Input("oc_seed_total", sw=1)
    total_next = total.close(total + rate)
    ones = np.ones((1, 3, 1, 1), np.float32)
    tens = np.full((1, 3, 1, 1), 10.0, np.float32)

    listed = ModelyLoop(
        "oc_seed_listed",
        inputs=[rate, total],
        outputs=[Output("oc_seed_out", total_next)],
    ).build()
    run = listed({"oc_seed_rate": ones, "oc_seed_total": tens})["oc_seed_out"]
    np.testing.assert_allclose(to_numpy(run).ravel(), [11, 12, 13])
    # Listed but left out of the data: zero, in a call and in a DataLoader.
    run = listed({"oc_seed_rate": ones})["oc_seed_out"]
    np.testing.assert_allclose(to_numpy(run).ravel(), [1, 2, 3])
    data = DataLoader(listed, source={"oc_seed_rate": np.ones(5)}, seq_length=3)
    np.testing.assert_array_equal(data.get_input("oc_seed_total"), 0.0)

    listed.save(tmp_path / "oc_seed")
    loaded = Modely.load(tmp_path / "oc_seed")
    run = loaded({"oc_seed_rate": ones})["oc_seed_out"]
    np.testing.assert_allclose(to_numpy(run).ravel(), [1, 2, 3])

    # Not listed: zero, whatever the data holds.
    unlisted = ModelyLoop(
        "oc_seed_unlisted", inputs=[rate], outputs=[Output("oc_seed_out2", total_next)]
    ).build()
    run = unlisted({"oc_seed_rate": ones, "oc_seed_total": tens})["oc_seed_out2"]
    np.testing.assert_allclose(to_numpy(run).ravel(), [1, 2, 3])


def test_close_and_modelyloop_reject_what_they_cannot_feed_back():
    rate = Input("oc_err_rate", sw=1)
    state = Input("oc_err_state", sw=1)
    with pytest.raises(ValueError, match="closed with the value of one step"):
        state.close(Input("oc_err_window", sw=3))
    with pytest.raises(ValueError, match="no future samples"):
        Input("oc_err_future", sw=(1, 1)).close(rate)
    with pytest.raises(TypeError, match=r"Buffer\(sw=3\)\(u\)"):
        Buffer(rate)  # type: ignore[arg-type]

    state.close(state + rate)
    with pytest.raises(ValueError, match="already closed"):
        state.close(rate)
    with pytest.raises(ValueError, match="nothing closes and are not among its inputs"):
        ModelyLoop("oc_err_unlisted", inputs=[], outputs=[Output("oc_err_o1", state)])
    with pytest.raises(ValueError, match="reads no closed Input"):
        ModelyLoop("oc_err_none", inputs=[rate], outputs=[Output("oc_err_o3", rate)])

    seeded = Input("oc_err_seeded", sw=1)
    seeded.close(seeded + rate, init=Input("oc_err_init", sw=1) * 2.0)
    with pytest.raises(ValueError, match="computed from the Inputs"):
        ModelyLoop(
            "oc_err_init_expr", inputs=[rate], outputs=[Output("oc_err_o4", seeded)]
        )
    started = Input("oc_err_started", sw=1)
    started.close(started + rate, init=1.0)
    with pytest.raises(ValueError, match="has an initial value"):
        ModelyLoop(
            "oc_err_both",
            inputs=[rate, started],
            outputs=[Output("oc_err_o6", started)],
        )

    inner = Input("oc_err_inner", sw=1)
    inner_next = inner.close(inner + rate)
    model = ModelyLoop(
        "oc_err_minimize", inputs=[rate], outputs=[Output("oc_err_o5", inner_next)]
    )
    with pytest.raises(ValueError, match="computed inside the step"):
        model.minimize("bad", inner_next * 2.0, rate)
