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
    Loop,
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


def test_chained_fed_back_inputs_are_one_loop_as_chained_integrates():
    dt = 0.1
    acc = Input("oc_chain_acc", tw=1)
    vel, pos = Input("oc_chain_v", tw=1), Input("oc_chain_x", tw=1)
    vel_next = vel.feedback(IntegrateStep(dt=dt)(vel, acc))
    pos_next = pos.feedback(IntegrateStep(dt=dt)(pos, vel_next))
    model = ModelyLoop(
        "oc_chain",
        inputs=[acc],
        outputs=[Output("oc_chain_vel", vel_next), Output("oc_chain_pos", pos_next)],
    ).build()
    # Integrate feeds back the same states by itself.
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
    # Each fed-back Input reads the other: a cycle no chain of Loops can
    # express.
    dt, k, c, m = 0.05, 4.0, 0.5, 2.0
    force = Input("oc_msd_force", tw=1)
    x, v = Input("oc_msd_x", tw=1), Input("oc_msd_v", tw=1)
    acceleration = (force - x * k - v * c) / m
    v_next = v.feedback(IntegrateStep(dt=dt)(v, acceleration))
    x.feedback(IntegrateStep(dt=dt)(x, v_next))
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
    u = Input("oc_buf_u", tw=1)
    window = Buffer(tw=3)(u)
    counter = Input("oc_buf_counter", tw=1)
    counter.feedback(counter + 1.0)
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
    u = Input("oc_buf_init_u", tw=1)
    count = Input("oc_buf_init_count")
    signal = count.feedback(count + 1.0)  # 1, 2, 3, ... with no window
    window = Buffer(tw=3, init=-1.0)(signal)
    model = ModelyLoop(
        "oc_buf_init",
        inputs=[u],
        outputs=[Output("oc_buf_init_window", window), Output("oc_buf_init_u_out", u)],
    ).build()

    result = to_numpy(model({"oc_buf_init_u": _ramp(3)})["oc_buf_init_window"])
    np.testing.assert_allclose(result.squeeze(), [[-1, -1, 1], [-1, 1, 2], [1, 2, 3]])


def test_buffer_feeds_a_plant_that_reads_a_window_of_the_command():
    # The controller computes u one sample per step; the plant reads three.
    reference = Input("oc_ctrl_ref", tw=1)
    y = Input("oc_ctrl_y", tw=1)
    u = (reference - y) * 0.5
    u_in = Input("oc_ctrl_u_in", tw=3)
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
    y_next = y.feedback(plant([Buffer(tw=3)(u)]))
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
    # The plant's output feeds y back, which its own argument reads: still a
    # single call of the plant, one buffer.
    fed_back = [node for node in model.body._inputs if node.update is not None]
    assert len(fed_back) == 2


def test_a_model_called_twice_has_a_feedback_per_call():
    inflow = Input("oc_tank_q", tw=1)
    level = Input("oc_tank_h", tw=1)
    tank = Modely(
        "oc_tank",
        inputs=[inflow],
        outputs=[Output("oc_tank_level", level.feedback(level * 0.5 + inflow))],
    )
    source = Input("oc_tank_source", tw=1)
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
    assert [node.name for node in model.body._inputs] == [
        "oc_tank_source",
        "oc_tank_h",
        "oc_tank_h_2",
    ]


def test_integrate_and_derivative_feed_back_inputs_of_their_own():
    dt, steps = 0.1, 8
    data = np.random.default_rng(2).standard_normal((2, steps, 1, 1)).astype(np.float32)
    acc, v0 = Input("oc_hl_acc", tw=1), Input("oc_hl_v0", tw=1)
    model = ModelyLoop(
        "oc_hl",
        inputs=[acc, v0],
        outputs=[
            Output("oc_hl_pos", Integrate(dt=dt)(Integrate(dt=dt, init=v0)(acc))),
            Output("oc_hl_trap", Integrate(method="trapezoidal", dt=dt)(acc)),
            Output("oc_hl_der", Derivative(dt=dt)(acc)),
        ],
    ).build()

    result = model({"oc_hl_acc": data, "oc_hl_v0": np.full_like(data, 0.5)})
    a = data[..., 0, 0]
    # The trapezoid's first step averages the rate with the one before it,
    # zero without init_rate.
    previous = np.concatenate([np.zeros_like(a[:, :1]), a[:, :-1]], axis=1)
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


def test_a_modely_starts_integrate_derivative_and_buffer_from_their_init(tmp_path):
    # Built as a Modely, nothing feeds back the state Integrate, Derivative and
    # Buffer keep: the step starts from their init, zero without one, as the
    # first step of a ModelyLoop does, and no Input of their own is read from
    # data.
    rate, v0 = Input("oc_one_rate", tw=1), Input("oc_one_v0", tw=1)
    past0 = Input("oc_one_past0", tw=2)
    trapezoidal = Integrate(method="trapezoidal", dt=0.5, init=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = Modely(
            "oc_one",
            inputs=[rate, v0, past0],
            outputs=[
                Output("oc_one_int", Integrate(dt=0.5, init=v0)(rate)),
                Output("oc_one_int_zero", Integrate(dt=0.5)(rate)),
                Output("oc_one_trap", trapezoidal(rate)),
                Output(
                    "oc_one_trap_rate",
                    Integrate(method="trapezoidal", dt=0.5, init=1.0, init_rate=v0)(
                        rate
                    ),
                ),
                Output("oc_one_der", Derivative(dt=0.5, init=v0)(rate)),
                Output("oc_one_der_zero", Derivative(dt=0.5)(Derivative(dt=0.5)(rate))),
                Output("oc_one_buf", Buffer(tw=3, init=v0)(rate)),
                Output("oc_one_buf_zero", Buffer(tw=3)(rate)),
                Output("oc_one_buf_past", Buffer(tw=3, init=past0)(rate)),
            ],
        ).build()
    names = ["oc_one_past0", "oc_one_rate", "oc_one_v0"]
    assert sorted(node.name for node in model.inference_inputs) == names
    assert sorted(node.name for node in model.inputs) == names

    data = {
        "oc_one_rate": np.full((2, 1, 1), 4.0, np.float32),
        "oc_one_v0": np.array([1.0, -2.0], np.float32).reshape(2, 1, 1),
        "oc_one_past0": np.array([5.0, 6.0, 7.0, 8.0], np.float32).reshape(2, 2, 1),
    }
    expected = {
        "int": [3.0, 0.0],
        "int_zero": [2.0, 2.0],
        "trap": [2.0, 2.0],
        "trap_rate": [2.25, 1.5],
        "der": [6.0, 12.0],
        "der_zero": [16.0, 16.0],
        "buf": [1.0, 1.0, 4.0, -2.0, -2.0, 4.0],
        "buf_zero": [0.0, 0.0, 4.0, 0.0, 0.0, 4.0],
        "buf_past": [5.0, 6.0, 4.0, 7.0, 8.0, 4.0],
    }
    model.save(tmp_path / "oc_one")
    loaded = Modely.load(tmp_path / "oc_one")
    for built in (model, loaded):
        result = built(data)
        for name, values in expected.items():
            np.testing.assert_allclose(
                to_numpy(result[f"oc_one_{name}"]).ravel(), values, rtol=1e-6
            )


def test_a_call_gives_a_declared_fed_back_input_its_data():
    rate = Input("oc_decl_rate", tw=1)
    level = Input("oc_decl_level", tw=1)
    level_next = level.feedback(level * 0.5 + rate)
    # The tank declares its level: built as a Modely, the one-step model.
    tank = Modely(
        "oc_decl_tank", inputs=[level, rate], outputs=[Output("oc_decl", level_next)]
    )
    start, source = Input("oc_decl_start", tw=1), Input("oc_decl_source", tw=1)

    # In a ModelyLoop the level keeps its feedback, and the stream the call
    # gives it is its initial value: only the first step of it is read.
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
    row = next(row for row in loop.inputs._rows if row["input"].name == "oc_decl_level")
    assert row["starts from"] == "oc_decl_start, first step"

    # In a Modely it reads the stream at every step, as it would its data.
    plain = Modely(
        "oc_decl_plain",
        inputs=[start, source],
        outputs=[Output("oc_decl_plain_out", tank([start, source]))],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no fed-back Input left to read
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
    # Listed too, the level would start from its data as well.
    with pytest.raises(ValueError, match="'oc_decl_level' has an initial value"):
        ModelyLoop(
            "oc_decl_both",
            inputs=[start, source, level],
            outputs=[Output("oc_decl_both_out", tank([start, source]))],
        )


def test_modely_warns_about_fed_back_inputs_and_reads_them_from_data():
    rate = Input("oc_open_rate", tw=1)
    state = Input("oc_open_state", tw=1)
    one_step = Output("oc_open_next", state.feedback(state + rate))
    model = Modely("oc_open", inputs=[rate], outputs=[one_step])
    with pytest.warns(
        UserWarning, match=r"reads the fed-back Inputs \['oc_open_state'\]"
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


def test_inputs_tells_how_each_input_is_fed():
    rate = Input("oc_tab_rate", tw=1)
    level = Input("oc_tab_level", tw=1)
    level_next = level.feedback(level * 0.5 + rate)
    tank = Modely(
        "oc_tab_tank", inputs=[rate], outputs=[Output("oc_tab_out", level_next)]
    )
    tank.minimize("fit", "oc_tab_out", Input("oc_tab_target", tw=1))

    rows = {row["input"].name: row for row in tank.inputs._rows}
    assert [node.name for node in tank.inputs] == list(rows)
    assert rows.keys() == {"oc_tab_level", "oc_tab_rate", "oc_tab_target"}
    assert rows["oc_tab_level"]["declared"] is False
    assert rows["oc_tab_level"]["fed back with"] == level_next.name
    assert rows["oc_tab_level"]["data"] == "required"  # a Modely reads it from data
    assert (
        rows["oc_tab_rate"]["declared"] and rows["oc_tab_rate"]["fed back with"] is None
    )
    assert rows["oc_tab_target"]["data"] == "target"

    # In a ModelyLoop: listed, the tank's level starts from its data; the
    # state of an Integrate starts from its init, read at the first step.
    source, x0 = Input("oc_tab_source", tw=1), Input("oc_tab_x0", tw=1)
    loop = ModelyLoop(
        "oc_tab_loop",
        inputs=[source, level, x0],
        outputs=[
            Output("oc_tab_loop_out", tank([source])),
            Output("oc_tab_integral", Integrate(dt=0.1, init=x0)(source)),
        ],
    )
    rows = {row["input"].name: row for row in loop.inputs._rows}
    assert (
        rows["oc_tab_level"]["declared"] and rows["oc_tab_level"]["data"] == "optional"
    )
    assert rows["oc_tab_level"]["starts from"] == "its data, first step, else 0"
    assert rows["oc_tab_source"]["data"] == "required"
    (state,) = [row for name, row in rows.items() if name.endswith("_oc_tab_source")]
    assert state["data"] is None and state["starts from"] == "oc_tab_x0, first step"
    # The table alone: the header, named after what it lists, and a row per
    # Input.
    lines = str(loop.inputs).splitlines()
    assert lines[0].startswith("┏") and lines[1].startswith("┃ index ┃ input ")
    assert len(lines) == 3 + 4 + 1 and lines[-1].startswith("└")


def test_outputs_tells_what_each_output_is():
    rate = Input("oc_outs_rate", tw=1)
    level = Input("oc_outs_level", tw=1)
    level_next = level.feedback(level * 0.5 + rate)
    doubled = Output("oc_outs_doubled", rate * 2)
    tank = Modely(
        "oc_outs_tank",
        inputs=[rate, level],
        outputs=[Output("oc_outs_level_next", level_next), doubled],
    )
    tank.minimize("fit", "oc_outs_level_next", Input("oc_outs_target", tw=1))
    tank.minimize("reg", doubled, 0.0)

    # A list of the declared Outputs, in their order.
    assert [output.name for output in tank.outputs] == [
        "oc_outs_level_next",
        "oc_outs_doubled",
    ]
    assert tank.outputs[1] is doubled
    rows = {row["output"].name: row for row in tank.outputs._rows}
    assert rows["oc_outs_level_next"]["stream"] == level_next.name
    assert rows["oc_outs_level_next"]["feeds back"] == ["oc_outs_level"]
    assert rows["oc_outs_level_next"]["minimized by"] == ["fit (source)"]
    # Minimized through the stream it names.
    assert rows["oc_outs_doubled"]["feeds back"] == []
    assert rows["oc_outs_doubled"]["minimized by"] == ["reg (source)"]
    text = str(tank.outputs)
    lines = text.splitlines()
    assert lines[0].startswith("┏") and lines[1].startswith("┃ index ┃ output ")
    assert len(lines) == 3 + 2 + 1 and lines[-1].startswith("└")
    assert repr(tank.outputs) == text

    # In a ModelyLoop each output is a trajectory: the row names the stream
    # of the step it collects, the one the Inputs table feeds back.
    loop = ModelyLoop(
        "oc_outs_loop",
        inputs=[rate, level],
        outputs=[Output("oc_outs_loop_level", level_next)],
        length=4,
    )
    (row,) = loop.outputs._rows
    assert row["stream"] == level_next.name
    assert row["feeds back"] == ["oc_outs_level"]
    assert row["output"].seq == (4,)
    assert "(4,)" in str(loop.outputs)


def test_inputs_of_a_loop_are_those_of_its_modelyloop():
    # A ModelyLoop is the Loop over its step, its Inputs with the horizon: a
    # Modely rolling out the same Loop prints the same rows.
    def rows(model):
        return [
            (
                row["input"].name,
                row["seq"],
                row["declared"],
                row["fed back with"],
                row["data"],
                row["starts from"],
            )
            for row in model.inputs._rows
        ]

    u, y0 = Input("oc_same_u", tw=1), Input("oc_same_y0", tw=1)
    integrate = Integrate(dt=1, init=y0, name="oc_same_int")
    shortcut = ModelyLoop(
        "oc_same_shortcut",
        inputs=[u, y0],
        outputs=[Output("oc_same_traj", integrate(u))],
    )
    u, y0 = Input("oc_same_u", tw=1), Input("oc_same_y0", tw=1)
    step = Modely(
        "oc_same_step",
        inputs=[u, y0],
        outputs=[
            Output("oc_same_next", Integrate(dt=1, init=y0, name="oc_same_int")(u))
        ],
    )
    u_seq, y0_seq = (
        Input("oc_same_u", seq=-1, tw=1),
        Input("oc_same_y0", seq=-1, tw=1),
    )
    traj = Loop(f=step)({u: u_seq, y0: y0_seq})
    manual = Modely(
        "oc_same_manual",
        inputs=[u_seq, y0_seq],
        outputs=[Output("oc_same_traj", traj)],
    )
    state = ("oc_same_int_oc_same_u", (), False, "oc_same_int", None)
    assert rows(shortcut) == [
        ("oc_same_y0", (None,), True, None, "required", None),
        ("oc_same_u", (None,), True, None, "required", None),
        (*state, "oc_same_y0, first step"),
    ]
    assert rows(manual) == rows(shortcut)

    # A state its own data starts, and one a stream without the horizon does.
    x = Input("oc_same_x", tw=1)
    x_next = x.feedback(IntegrateStep(dt=1)(x, u))
    shortcut = ModelyLoop(
        "oc_same_x_shortcut", inputs=[u, x], outputs=[Output("oc_same_x_traj", x_next)]
    )
    assert rows(shortcut)[0] == (
        "oc_same_x",
        (None,),
        True,
        x_next.name,
        "optional",
        "its data, first step, else 0",
    )
    step = Modely(
        "oc_same_x_step", inputs=[u, x], outputs=[Output("oc_same_x_next", x_next)]
    )
    x0 = Input("oc_same_x0", tw=1)
    manual = Modely(
        "oc_same_x_manual",
        inputs=[u_seq, x0],
        outputs=[Output("oc_same_x_traj", Loop(f=step, init={x: x0})({u: u_seq}))],
    )
    assert rows(manual)[-1] == ("oc_same_x", (), False, x_next.name, None, "oc_same_x0")


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

    force = Input("oc_fit_force", tw=1)
    k = Parameter("oc_fit_k", value=1.0)
    # Among the inputs, the states start from the first step of their data.
    x, v = Input("oc_fit_x", tw=1), Input("oc_fit_v", tw=1)
    v_next = v.feedback(IntegrateStep(dt=dt)(v, force - k * x))
    x.feedback(IntegrateStep(dt=dt)(x, v_next))
    x_hat = Output("oc_fit_x_hat", x)
    x_meas = Input("oc_fit_x_meas", tw=1)
    model = ModelyLoop("oc_fit", inputs=[force, x, v], outputs=[x_hat])
    # Step streams: the output and the measured Input, compared as trajectories.
    model.minimize("fit", x_hat, x_meas)
    model.build()
    data = DataLoader(
        model,
        source={
            "oc_fit_force": forces,
            "oc_fit_x": positions,
            "oc_fit_v": velocities,
            "oc_fit_x_meas": positions,
        },
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
    rate = Input("oc_len_rate", tw=1)
    total = Input("oc_len_total", tw=1)
    total_next = total.feedback(total + rate)
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
    assert fixed._inputs[0].seq == (5,)


def test_a_fed_back_input_starts_from_its_data_if_listed_else_from_zero(tmp_path):
    rate = Input("oc_seed_rate", tw=1)
    total = Input("oc_seed_total", tw=1)
    total_next = total.feedback(total + rate)
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


def test_feedback_and_modelyloop_reject_what_they_cannot_feed_back():
    rate = Input("oc_err_rate", tw=1)
    state = Input("oc_err_state", tw=1)
    with pytest.raises(ValueError, match="fed back with the value of one step"):
        state.feedback(Input("oc_err_window", tw=3))
    with pytest.raises(ValueError, match="no future samples"):
        Input("oc_err_future", tw=(1, 1)).feedback(rate)
    with pytest.raises(TypeError, match=r"Buffer\(tw=3\)\(u\)"):
        Buffer(rate)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1 sample"):
        Buffer(tw=3, init=Input("oc_err_long", tw=4))(rate)
    with pytest.raises(ValueError, match="neither a scalar"):
        Buffer(tw=3, init=Input("oc_err_wide", dim=2, tw=1))(rate)

    state.feedback(state + rate)
    with pytest.raises(ValueError, match="already fed back"):
        state.feedback(rate)
    with pytest.raises(
        ValueError, match="without declaring them: list them among its inputs"
    ):
        ModelyLoop("oc_err_unlisted", inputs=[], outputs=[Output("oc_err_o1", state)])
    with pytest.raises(ValueError, match="reads no fed-back Input"):
        ModelyLoop("oc_err_none", inputs=[rate], outputs=[Output("oc_err_o3", rate)])

    seeded = Integrate(dt=1.0, init=Input("oc_err_init", tw=1) * 2.0)(rate)
    with pytest.raises(ValueError, match="computed from the Inputs"):
        ModelyLoop(
            "oc_err_init_expr", inputs=[rate], outputs=[Output("oc_err_o4", seeded)]
        )

    inner = Input("oc_err_inner", tw=1)
    inner_next = inner.feedback(inner + rate)
    model = ModelyLoop(
        "oc_err_minimize", inputs=[rate], outputs=[Output("oc_err_o5", inner_next)]
    )
    with pytest.raises(ValueError, match="computed inside the step"):
        model.minimize("bad", inner_next * 2.0, rate)


def test_a_saved_plant_keeps_its_feedback_in_a_modelyloop(tmp_path):
    # The one-step plant of examples/open_callbacks/03: saved, the window of
    # positions loaded as a plain Input, and the plant ran open-loop.
    x, force = Input("oc_plant_x", tw=3), Input("oc_plant_F", tw=3)
    x_kernel = Parameter("oc_plant_kx", value=np.full((3, 1), 0.3))
    f_kernel = Parameter("oc_plant_kf", value=np.full((3, 1), 0.1))
    x_n = Output(
        "oc_plant_x_n",
        Fir(kernel=x_kernel, bias=False)([x])
        + Fir(kernel=f_kernel, bias=False)([force]),
    )
    x.feedback(x_n)
    plant = Modely("oc_plant", inputs=[x, force], outputs=[x_n]).build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plant.save(tmp_path / "oc_plant")
    loaded = Modely.load(tmp_path / "oc_plant")
    loaded_x = next(node for node in loaded._inputs if node.name == "oc_plant_x")
    assert loaded_x.update is not None and loaded_x.update.name == "oc_plant_x_n"

    def looped(model, tag):
        x0, u = Input(f"{tag}_x0", tw=3), Input(f"{tag}_u", tw=1)
        position = model([x0, Buffer(tw=3)(u)])
        return ModelyLoop(
            tag, inputs=[x0, u], outputs=[Output(f"{tag}_out", position)]
        ).build()

    def run(model, tag):
        data = {
            f"{tag}_x0": np.ones((1, 6, 3, 1), np.float32),
            f"{tag}_u": np.ones((1, 6, 1, 1), np.float32),
        }
        return to_numpy(looped(model, tag)(data)[f"{tag}_out"])

    np.testing.assert_allclose(
        run(loaded, "oc_plant_back"), run(plant, "oc_plant_orig"), rtol=1e-6
    )


def test_a_saved_model_keeps_what_feeds_its_inputs_back_and_where_they_start(
    tmp_path,
):
    # Fed back with a stream that is no output, and by Integrates that start
    # from a number, a Parameter and an Input: once loaded, none of them was
    # fed back.
    error = Input("oc_keep_e", tw=1)
    level, count0 = Input("oc_keep_level", tw=1), Input("oc_keep_count0", tw=1)
    gain = Parameter("oc_keep_gain", value=0.5)
    level.feedback(level * gain + error)
    control = Output(
        "oc_keep_u",
        level
        + Integrate(dt=0.1, init=2.0)(error)
        + Integrate(dt=0.1, init=Parameter("oc_keep_start", value=3.0))(error)
        + Integrate(dt=0.1, init=count0)(error),
    )
    with pytest.warns(UserWarning, match="fed-back Inputs"):
        controller = Modely("oc_keep", inputs=[error], outputs=[control]).build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        controller.save(tmp_path / "oc_keep")
    with pytest.warns(UserWarning, match="fed-back Inputs"):
        loaded = Modely.load(tmp_path / "oc_keep")
    # The same block, called by a model that is saved in turn.
    drive = Input("oc_keep_drive", tw=1)
    outer = Modely(
        "oc_keep_outer",
        inputs=[drive],
        outputs=[Output("oc_keep_v", controller([drive]))],
    )
    with pytest.warns(UserWarning, match="fed-back Inputs"):
        outer.build()
    outer.save(tmp_path / "oc_keep_outer")
    with pytest.warns(UserWarning, match="fed-back Inputs"):
        loaded_outer = Modely.load(tmp_path / "oc_keep_outer")

    def run(model, tag):
        # The Input the block starts its Integrate from, by its name: another
        # Input of that name is another Input.
        e, start = Input(f"{tag}_e", tw=1), model.inputs["oc_keep_count0"]
        looped = ModelyLoop(
            tag, inputs=[e, start], outputs=[Output(f"{tag}_out", model([e]))]
        ).build()
        data = {f"{tag}_e": _ramp(5), "oc_keep_count0": np.full((1, 5, 1, 1), 10.0)}
        return to_numpy(looped(data)[f"{tag}_out"])

    expected = run(controller, "oc_keep_orig")
    np.testing.assert_allclose(run(loaded, "oc_keep_back"), expected, rtol=1e-6)
    np.testing.assert_allclose(
        run(loaded_outer, "oc_keep_outer_back"), expected, rtol=1e-6
    )


def test_saving_warns_of_feedback_the_loaded_model_would_lose(tmp_path):
    # A weight only an update reads: the model does not compute it, so the
    # weights file does not hold its current value.
    gain = Parameter("oc_warn_gain", value=0.5)
    u, s = Input("oc_warn_u", tw=1), Input("oc_warn_s", tw=1)
    s.feedback(s * gain + u)
    y = Output("oc_warn_y", s * 2.0)
    ModelyLoop("oc_warn_loop", inputs=[u], outputs=[y]).build()
    with pytest.warns(UserWarning, match="fed-back Inputs"):
        plain = Modely("oc_warn_plain", inputs=[u], outputs=[y]).build()
    with pytest.warns(UserWarning, match=r"without the weights of \['oc_warn_gain'\]"):
        plain.save(tmp_path / "oc_warn_plain")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plain.save(tmp_path / "oc_warn_plain_arch", weights=False)

    # A call giving a stream for a fed-back Input its model declares: the
    # called model is saved as it is, in a folder of its own, so the loaded
    # model keeps the feedback, and saving has nothing to warn about.
    state, drive = Input("oc_warn_state", tw=1), Input("oc_warn_drive", tw=1)
    following = Output("oc_warn_next", state * 0.5 + drive)
    state.feedback(following)
    block = Modely("oc_warn_block", inputs=[state, drive], outputs=[following])
    a, b = Input("oc_warn_a", tw=1), Input("oc_warn_b", tw=1)
    outer = Modely(
        "oc_warn_outer", inputs=[a, b], outputs=[Output("oc_warn_o", block([a, b]))]
    ).build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        outer.save(tmp_path / "oc_warn_outer")
    loaded = Modely.load(tmp_path / "oc_warn_outer")
    rollouts = []
    for tag, model in (("original", outer), ("loaded", loaded)):
        start, push = Input(f"oc_warn_{tag}_a", tw=1), Input(f"oc_warn_{tag}_b", tw=1)
        out = Output(f"oc_warn_{tag}_o", model([start, push]))
        loop = ModelyLoop(f"oc_warn_{tag}", inputs=[start, push], outputs=[out])
        data = {
            f"oc_warn_{tag}_a": np.zeros((1, 3, 1, 1), np.float32),
            f"oc_warn_{tag}_b": np.ones((1, 3, 1, 1), np.float32),
        }
        rollouts.append(to_numpy(loop.build()(data)[out.name]).ravel())
    # From 0, fed back at every step: 1, 1 * 0.5 + 1, 1.5 * 0.5 + 1. Read
    # from the data instead it would stay 0 * 0.5 + 1.
    np.testing.assert_allclose(rollouts[0], [1.0, 1.5, 1.75])
    np.testing.assert_allclose(rollouts[1], rollouts[0])


@pytest.mark.parametrize("call_first", [True, False])
def test_a_fed_back_input_a_call_and_its_caller_both_read_is_refused(call_first):
    # One state per call: the caller's x and the call's were two, and which
    # one the data started depended on which was reached first.
    u, r = Input("oc_cross_u", tw=1), Input("oc_cross_r", tw=1)
    x = Input("oc_cross_x", tw=1)
    x.feedback(x + u)
    block = Modely("oc_cross_block", inputs=[r], outputs=[Output("oc_cross_bo", x + r)])
    y = block([u]) + x if call_first else x + block([u])
    with pytest.raises(
        ValueError,
        match="fed-back Input 'oc_cross_x' both in the graph that calls "
        "'oc_cross_block' and inside it",
    ):
        ModelyLoop("oc_cross", inputs=[u, x], outputs=[Output("oc_cross_y", y)])

    # As the message says: the block declares an Input for the state, and
    # the caller gives it x.
    s = Input("oc_cross_s", tw=1)
    block = Modely(
        "oc_cross_given", inputs=[r, s], outputs=[Output("oc_cross_go", s + r)]
    )
    y = block([u, x]) + x if call_first else x + block([u, x])
    model = ModelyLoop(
        "oc_cross_ok",
        inputs=[u, x],
        outputs=[Output("oc_cross_y", y), Output("oc_cross_xo", x)],
    ).build()
    result = model({"oc_cross_u": _ramp(3), "oc_cross_x": np.full((1, 3, 1, 1), 100.0)})
    np.testing.assert_allclose(to_numpy(result["oc_cross_xo"]).ravel(), [100, 101, 103])
    np.testing.assert_allclose(to_numpy(result["oc_cross_y"]).ravel(), [201, 204, 209])


def test_a_fed_back_input_inside_a_block_called_twice_is_no_output_of_the_caller():
    # Which call's level would it be? It was refused as reading the inflow
    # of the tank, which nothing feeds back.
    inflow, level = Input("oc_tank_q", tw=1), Input("oc_tank_h", tw=1)
    tank = Modely(
        "oc_tank",
        inputs=[inflow],
        outputs=[Output("oc_tank_level", level.feedback(level * 0.5 + inflow))],
    )
    source = Input("oc_tank_source", tw=1)
    first = tank([source])
    with pytest.raises(
        ValueError, match="fed-back Input 'oc_tank_h' both in the graph that calls"
    ):
        ModelyLoop(
            "oc_tank_cascade",
            inputs=[source],
            outputs=[
                Output("oc_tank_h_out", level),
                Output("oc_tank_1", first),
                Output("oc_tank_2", tank([first])),
            ],
        )
