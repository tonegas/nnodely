"""The port-Hamiltonian block earns its place through the properties it cannot
lose, so the tests check those rather than any particular fitted value: with
random weights and random states the power balance has to hold every time."""

import numpy as np
import pytest
import keras

from conftest import to_numpy
from nnodely import DataLoader, Derivative, Input, Modely, Ode, Output
from nnodely.layers.porthamiltonian import PortHamiltonian

STATE_DIM, INPUT_DIM = 4, 1


def _probe(field, suffix):
    """A model exposing dx, y and dH/dx, which is all the balance needs."""
    x = Input(f"probe_x_{suffix}", dim=field.state_dim)
    u = Input(f"probe_u_{suffix}", dim=field.input_dim)
    dx, y = field(x.last(), u.last())
    gradient = Derivative(order=1, respect_to=x)(field.energy(x.last()))
    return Modely(
        f"probe_{suffix}",
        inputs=[x, u],
        outputs=[
            Output(f"dx_{suffix}", dx),
            Output(f"y_{suffix}", y),
            Output(f"grad_{suffix}", gradient),
        ],
    ).build()


def _evaluate(model, suffix, states, drives):
    result = model(
        {
            f"probe_x_{suffix}": states[:, :, None],
            f"probe_u_{suffix}": drives[:, :, None],
        }
    )
    return tuple(
        to_numpy(result[f"{key}_{suffix}"])[..., 0] for key in ("dx", "y", "grad")
    )


def _zero(parameter):
    parameter.param.assign(keras.ops.zeros(parameter.param.shape))


def _random(rng, samples=32):
    return (
        rng.normal(size=(samples, STATE_DIM)).astype(np.float32),
        rng.normal(size=(samples, INPUT_DIM)).astype(np.float32),
    )


def test_shapes_and_one_set_of_weights():
    """dx has the shape of the state, y the shape of the input, and H, J, R, G
    each appear once however many times the block is called."""
    field = PortHamiltonian(STATE_DIM, INPUT_DIM, hamiltonian=[8], name="shape_ph")
    model = _probe(field, "shape")

    rng = np.random.default_rng(0)
    states, drives = _random(rng, samples=5)
    dx, y, gradient = _evaluate(model, "shape", states, drives)

    assert dx.shape == (5, STATE_DIM)
    assert y.shape == (5, INPUT_DIM)
    assert gradient.shape == (5, STATE_DIM)

    paths = [weight.path for weight in model.model.weights]
    assert sum(path.endswith("J/value") for path in paths) == 1
    assert sum(path.endswith("R/value") for path in paths) == 1
    assert sum(path.endswith("G/value") for path in paths) == 1


def test_skew_structure_makes_the_conservative_part_energy_neutral():
    """With R and G zeroed, dx is J dH/dx alone and dH/dt must vanish exactly:
    that is the antisymmetry of J, and it holds for random weights."""
    field = PortHamiltonian(STATE_DIM, INPUT_DIM, hamiltonian=[16], name="skew_ph")
    model = _probe(field, "skew")
    _zero(field.R)
    _zero(field.G)

    states, drives = _random(np.random.default_rng(1))
    dx, _, gradient = _evaluate(model, "skew", states, drives)

    np.testing.assert_allclose(
        np.sum(gradient * dx, axis=1), np.zeros(len(states)), atol=1e-5
    )


def test_dissipation_can_only_remove_energy():
    """With J and G zeroed, dx is -R dH/dx and dH/dt must be non-positive:
    that is R = L L^T, and no parameter value can make it a source."""
    field = PortHamiltonian(STATE_DIM, INPUT_DIM, hamiltonian=[16], name="psd_ph")
    model = _probe(field, "psd")
    _zero(field.J)
    _zero(field.G)

    states, drives = _random(np.random.default_rng(2))
    dx, _, gradient = _evaluate(model, "psd", states, drives)

    assert np.all(np.sum(gradient * dx, axis=1) <= 1e-6)


def test_power_balance_holds_for_random_weights():
    """dH/dt <= y^T u, the property the whole structure exists to guarantee.

    Nothing here is trained: the weights are as initialised, which is the point.
    """
    field = PortHamiltonian(STATE_DIM, INPUT_DIM, hamiltonian=[16, 16], name="power_ph")
    model = _probe(field, "power")

    states, drives = _random(np.random.default_rng(3), samples=128)
    dx, y, gradient = _evaluate(model, "power", states, drives)

    energy_rate = np.sum(gradient * dx, axis=1)
    supplied = np.sum(y * drives, axis=1)
    assert np.all(energy_rate <= supplied + 1e-5)


def test_power_balance_holds_with_state_dependent_matrices():
    """The same balance, with J, R and G produced by MLPs of the state."""
    field = PortHamiltonian(
        STATE_DIM,
        INPUT_DIM,
        hamiltonian=[16],
        J=[8],
        R=[8],
        G=[8],
        name="varying_ph",
    )
    model = _probe(field, "varying")

    states, drives = _random(np.random.default_rng(4), samples=128)
    dx, y, gradient = _evaluate(model, "varying", states, drives)

    energy_rate = np.sum(gradient * dx, axis=1)
    supplied = np.sum(y * drives, axis=1)
    assert np.all(energy_rate <= supplied + 1e-5)


def test_repeated_calls_share_weights():
    """An integrator calls the block once per stage, so the field it evaluates
    at each of them has to be the same field."""
    field = PortHamiltonian(STATE_DIM, INPUT_DIM, hamiltonian=[8], name="shared_ph")
    x = Input("shared_x", dim=STATE_DIM)
    u = Input("shared_u", dim=INPUT_DIM)
    first, _ = field(x.last(), u.last())
    second, _ = field(x.last(), u.last())
    model = Modely(
        "shared_model",
        inputs=[x, u],
        outputs=[Output("first", first), Output("second", second)],
    ).build()

    rng = np.random.default_rng(5)
    states, drives = _random(rng, samples=4)
    result = model({"shared_x": states[:, :, None], "shared_u": drives[:, :, None]})

    np.testing.assert_allclose(
        to_numpy(result["first"]), to_numpy(result["second"]), rtol=1e-6
    )
    paths = [weight.path for weight in model.model.weights]
    assert len(paths) == len(set(paths))


def test_composes_with_ode_and_trains():
    """The block only produces the field; Ode integrates it and the loss is an
    ordinary prediction error, so nothing about the rollout lives in the block.

    The loss is on the integrated state, which is the only way every part of the
    structure - H, J, R and G alike - is on the path to it. Euler is the method
    that evaluates the field at the state input itself, where dH/dx is defined.
    """
    dt = 0.1
    field = PortHamiltonian(STATE_DIM, INPUT_DIM, hamiltonian=[16], name="ode_ph")
    x = Input("ode_x", dim=STATE_DIM)
    u = Input("ode_u", dim=INPUT_DIM)
    target = Input("ode_target", dim=STATE_DIM)

    step = Ode(
        lambda state, drive: field(state, drive)[0],
        x.last(),
        dt,
        method="euler",
        args=(u.last(),),
    )
    model = Modely(
        "ode_model", inputs=[x, u], outputs=[Output("x_next", step)]
    )
    model.minimize("state_error", model.outputs[0], target.last(), loss="mse")
    model.build()

    rng = np.random.default_rng(6)
    states = rng.normal(size=(256, STATE_DIM)).astype(np.float32)
    drives = rng.normal(size=(256, INPUT_DIM)).astype(np.float32)
    data = DataLoader(
        model=model,
        source={"ode_x": states, "ode_u": drives, "ode_target": 0.9 * states},
    )
    history = model.train(train_data=data, epochs=20, batch_size=32)
    assert history["loss"][-1] < history["loss"][0]

    # Every parameter of the structure took part: a variable off the loss path
    # would have made Keras warn instead.
    trained = {weight.path for weight in model.model.weights}
    assert any(path.endswith("ode_ph_J/value") for path in trained)
    assert any(path.endswith("ode_ph_R/value") for path in trained)
    assert any(path.endswith("ode_ph_G/value") for path in trained)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="at least two states"):
        PortHamiltonian(1, 1)
    with pytest.raises(ValueError, match="at least one hidden layer"):
        PortHamiltonian(2, 1, hamiltonian=[])
    with pytest.raises(ValueError, match="'constant' or a list"):
        PortHamiltonian(2, 1, J="skew")

    field = PortHamiltonian(STATE_DIM, INPUT_DIM, hamiltonian=[4], name="bad_ph")
    with pytest.raises(ValueError, match=r"x must have dim"):
        field(Input("bad_x", dim=2).last(), Input("bad_u", dim=INPUT_DIM).last())

    # A stage of rk4 is computed from the state rather than being the state
    # input, so there is nothing to differentiate H against.
    x = Input("stage_x", dim=STATE_DIM)
    u = Input("stage_u", dim=INPUT_DIM)
    with pytest.raises(TypeError, match=r"current sample, x.last\(\)"):
        Ode(lambda state, drive: field(state, drive)[0], x.last(), 0.1, args=(u.last(),))
    with pytest.raises(ValueError, match="window of 3 samples"):
        windowed = Input("windowed_x", dim=STATE_DIM)
        windowed.sw(3)
        field(windowed.last(), u.last())


def test_save_and_load_keep_the_field(tmp_path):
    field = PortHamiltonian(STATE_DIM, INPUT_DIM, hamiltonian=[8], R=[4], name="io_ph")
    model = _probe(field, "io")
    model.save(tmp_path / "io_ph")
    restored = Modely.load(tmp_path / "io_ph")

    states, drives = _random(np.random.default_rng(7), samples=8)
    for loaded, original in zip(
        _evaluate(restored, "io", states, drives),
        _evaluate(model, "io", states, drives),
    ):
        np.testing.assert_allclose(loaded, original, rtol=1e-5, atol=1e-6)
