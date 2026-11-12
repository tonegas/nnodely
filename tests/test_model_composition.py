import numpy as np
import pytest

from nnodely import (
    BatchNorm,
    Constant,
    DataLoader,
    Fir,
    Input,
    Linear,
    Modely,
    Output,
    Parameter,
    PReLU,
)
from conftest import to_numpy
from nnodely import TimeSelect


def test_model_composition():
    x = Input("x", dim=1, tw=10)
    y = Input("y", dim=1, tw=10)

    # Pinned: the body's weights have to survive being inlined, which is the
    # whole point of calling a built model. The biases start at zero.
    result_taps = Parameter("result_taps", value=np.ones((10, 2)))
    result_fir = Fir(features=2, kernel=result_taps)([x + y])
    x_out = Output("x_pred", result_fir)

    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    z = Input("z", dim=1, tw=10)
    z_taps = Parameter("z_taps", value=np.ones(2))
    z_fir = Fir(features=1, kernel=z_taps)([model1([z, z])])
    z_out = Output("z_pred", z_fir)

    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    dummy_z = np.ones((3, 10, 1), dtype=np.float32)
    result = model2({"z": dummy_z})

    assert "z_pred" in result
    assert result["z_pred"].shape == (3, 1, 2)
    # The body sums 10 samples of (z + z) into each of its 2 features, and the
    # outer Fir passes each feature on alone: 10 * 2 = 20 per feature.
    np.testing.assert_allclose(
        to_numpy(result["z_pred"]),
        np.full((3, 1, 2), 20.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_layer_reuse_shares_weights():
    # Applying one layer twice must share its weights, as it does in Keras: the
    # two applications are separate graph nodes that carry the same name.
    x = Input("reuse_x", dim=1, tw=1)
    linear = Linear(features=1, kernel="ones", bias=False)
    first, second = linear(x), linear(x * 2.0)
    model = Modely(
        "reuse", inputs=[x], outputs=[Output("reuse_out", first + second)]
    ).build()
    assert model.model is not None
    kernels = [
        variable
        for variable in model.model.trainable_variables
        if "Linear" in variable.path
    ]
    assert len(kernels) == 1
    kernels[0].assign(np.full((1, 1), 3.0, dtype=np.float32))

    result = model({"reuse_x": np.ones((1, 1, 1), dtype=np.float32)})
    # lin(x) + lin(2x) = 3*k*x = 9 for k = 3, x = 1.
    assert float(to_numpy(result["reuse_out"]).ravel()[0]) == pytest.approx(9.0)


def test_model_called_twice_keeps_calls_independent():
    # The body doubles its input, so the two calls have to disagree.
    u = Input("twice_u", dim=1, tw=1)
    body = Modely(
        "twice_body", inputs=[u], outputs=[Output("twice_doubled", u * 2.0)]
    ).build()

    x = Input("twice_x", dim=1, tw=1)
    first = body([x])  # 2 * 1 = 2
    second = body([x + 1.0])  # 2 * (1 + 1) = 4
    model = Modely(
        "twice_caller", inputs=[x], outputs=[Output("twice_out", first + second)]
    ).build()

    result = to_numpy(
        model({"twice_x": np.ones((1, 1, 1), dtype=np.float32)})["twice_out"]
    )
    # 2 + 4 = 6. Before the body was inlined once per call this returned 4.0,
    # which is 2 * first: the second call re-read the first call's subgraph.
    print("Result of model called twice:", result.ravel()[0])
    assert float(result.ravel()[0]) == pytest.approx(6.0)


def test_model_called_twice_shares_weights():
    # Each call inlines its own copy of the body, and those copies carry the
    # body's layer names, so one set of weights has to serve every call.
    u = Input("shared_u", dim=1, tw=1)
    scaled = Linear(features=1, kernel="ones", bias=False)(u)
    body = Modely(
        "shared_body", inputs=[u], outputs=[Output("shared_scaled", scaled)]
    ).build()

    x = Input("shared_x", dim=1, tw=1)
    first = body([x])
    second = body([x * 2.0])
    model = Modely(
        "shared_caller", inputs=[x], outputs=[Output("shared_out", first + second)]
    ).build()
    assert model.model is not None

    kernels = [
        variable
        for variable in model.model.trainable_variables
        if "Linear" in variable.path
    ]
    assert len(kernels) == 1
    kernels[0].assign(np.full((1, 1), 3.0, dtype=np.float32))

    result = to_numpy(
        model({"shared_x": np.ones((1, 1, 1), dtype=np.float32)})["shared_out"]
    )
    # k*x + k*2x = 3*k*x = 9 for k = 3, x = 1.
    assert float(result.ravel()[0]) == pytest.approx(9.0)


def test_nested_model_called_twice():
    # An inner model called by a body that is itself called twice: the scopes
    # have to nest, and the inner parameter still has to be a single weight.
    gain = Parameter(name="nested_gain", value=[2.0])
    v = Input("nested_v", dim=1, tw=1)
    inner = Modely(
        "nested_inner",
        inputs=[v],
        outputs=[Output("nested_inner_out", gain * v)],
    ).build()

    w = Input("nested_w", dim=1, tw=1)
    outer = Modely(
        "nested_outer",
        inputs=[w],
        outputs=[Output("nested_outer_out", inner([w]) + 1.0)],
    ).build()

    x = Input("nested_x", dim=1, tw=1)
    first = outer([x])
    second = outer([x + 1.0])
    model = Modely(
        "nested_caller", inputs=[x], outputs=[Output("nested_out", first + second)]
    ).build()
    assert model.model is not None
    assert len(model.model.trainable_variables) == 1

    result = to_numpy(
        model({"nested_x": np.ones((1, 1, 1), dtype=np.float32)})["nested_out"]
    )
    # outer(u) = 2u + 1, so outer(1) + outer(2) = 3 + 5 = 8.
    assert float(result.ravel()[0]) == pytest.approx(8.0)


def test_model_called_twice_rejects_colliding_output_names():
    # Both calls return outputs carrying the body's output name, so exposing
    # them directly must fail loudly instead of dropping one.
    u = Input("collide_u", dim=1, tw=1)
    body = Modely(
        "collide_body", inputs=[u], outputs=[Output("collide_out", u * 2.0)]
    ).build()

    x = Input("collide_x", dim=1, tw=1)
    first = body([x])
    second = body([x + 1.0])
    with pytest.raises(ValueError, match="two different outputs named"):
        Modely("collide_caller", inputs=[x], outputs=[first, second]).build()


def test_model_saving_loading_and_compose(tmp_path):
    x = Input("x", dim=1, tw=3)
    y = Input("y", dim=1, tw=3)

    result_taps = Parameter("result_taps", value=np.ones(3))
    result_fir = Fir(features=1, kernel=result_taps, bias=False)([x + y])
    x_out = Output("x_pred", result_fir)

    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    dummy_x = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    dummy_y = np.array([[[4.0], [5.0], [6.0]]], dtype=np.float32)
    result_model1 = model1({"x": dummy_x, "y": dummy_y})
    np.testing.assert_allclose(
        to_numpy(result_model1["x_pred"]),
        np.array([[[5.0 + 7.0 + 9.0]]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    nnodely_path = tmp_path / "concatenate.nnodely"
    model1.save(nnodely_path)
    model_loaded = Modely.load(nnodely_path)

    assert model_loaded.name == model1.name
    result_model_loaded = model_loaded({"x": dummy_x, "y": dummy_y})
    np.testing.assert_allclose(
        to_numpy(result_model1["x_pred"]),
        to_numpy(result_model_loaded["x_pred"]),
        rtol=1e-5,
        atol=1e-5,
    )

    z = Input("z", dim=1, tw=3)
    z_gain = Parameter("z_gain", value=[1.0])
    z_offset = Parameter("z_offset", value=[5.0])
    z_fir = Fir(features=1, kernel=z_gain, bias=z_offset)([model_loaded([z, z])])
    z_out = Output("z_pred", z_fir)

    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    dummy_z = np.array([[[10.0], [20.0], [30.0]]], dtype=np.float32)
    result_model2 = model2({"z": dummy_z})
    np.testing.assert_allclose(
        to_numpy(result_model2["z_pred"]),
        np.array([[[20.0 + 40.0 + 60.0 + 5.0]]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    # Save and load the composed model
    model2.save(tmp_path / "composed_model")
    loaded_model2 = Modely.load(tmp_path / "composed_model")
    assert loaded_model2.name == model2.name

    result_loaded_model2 = loaded_model2({"z": dummy_z})
    np.testing.assert_allclose(
        to_numpy(result_model2["z_pred"]),
        to_numpy(result_loaded_model2["z_pred"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_composition_shares_the_body_weights():
    """A model used as a block keeps its parameters: composing shares them
    rather than re-initializing, so weights set - or loaded - before the
    composition are the ones the composed graph runs with."""
    u = Input("share_u", dim=1, tw=2)
    body_fir = Fir(features=1, bias=False)([u])
    body = Modely(
        "share_body", inputs=[u], outputs=[Output("share_out", body_fir)]
    ).build()
    body_fir.kernel.assign(np.array([[1.0], [10.0]], dtype=np.float32))

    z = Input("share_z", dim=1, tw=2)
    composed = Modely(
        "share_composed", inputs=[z], outputs=[Output("o", body([z]))]
    ).build()

    values = np.array([[[3.0], [4.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        to_numpy(composed({"share_z": values})["o"]),
        np.full((1, 1, 1), 3.0 * 1.0 + 4.0 * 10.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    # Not a copy of the values - the same variable, so the two graphs cannot
    # drift apart.
    assert body.model is not None and composed.model is not None
    assert body.model.trainable_variables[0] is composed.model.trainable_variables[0]
    # And the block's own node still reaches it, so it stays inspectable.
    assert body_fir.kernel is composed.model.trainable_variables[0]


def test_training_a_composed_model_trains_the_block():
    """The consequence of sharing: fitting the composed graph moves the
    block's own weights, and the block evaluates with them afterwards."""
    u = Input("train_u", dim=1, tw=1)
    gain = Parameter("train_gain", value=[0.5])
    body_fir = Fir(features=1, kernel=gain, bias=False)([u])
    body = Modely(
        "train_body", inputs=[u], outputs=[Output("train_out", body_fir)]
    ).build()

    z = Input("train_z", dim=1, tw=1)
    composed_out = Output("composed_out", body([z]))
    composed = Modely("train_composed", inputs=[z], outputs=[composed_out])
    composed.minimize("error", source=composed_out, target=3.0, loss="mse")
    composed.build()

    data = DataLoader(composed, source={"train_z": np.ones((8, 1), dtype=np.float32)})
    # No early stopping: at this learning rate the loss oscillates on its way
    # down, and stopping at a low point of the swing leaves the weight off 3.
    composed.train(train_data=data, epochs=40, batch_size=4, lr=0.2, printer=None)

    trained = float(to_numpy(body_fir.kernel).ravel()[0])
    assert trained == pytest.approx(3.0, abs=0.1)
    # The block, evaluated on its own, now carries what the composition learnt.
    np.testing.assert_allclose(
        to_numpy(body({"train_u": np.ones((1, 1, 1), dtype=np.float32)})["train_out"]),
        np.full((1, 1, 1), trained, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_composition_shares_parameters_and_constants():
    """Every kind of stored value travels with the block, and the symbolic
    nodes keep pointing at the layer the composed graph runs."""
    gain = Parameter(name="pc_gain", value=[4.0])
    offset = Constant(name="pc_offset", value=[1.0])
    u = Input("pc_u", dim=1, tw=1)
    body = Modely(
        "pc_body",
        inputs=[u],
        outputs=[Output("pc_out", u * gain + offset)],
    ).build()

    z = Input("pc_z", dim=1, tw=1)
    composed = Modely(
        "pc_composed", inputs=[z], outputs=[Output("o", body([z]))]
    ).build()

    np.testing.assert_allclose(
        to_numpy(composed({"pc_z": np.full((1, 1, 1), 2.0, dtype=np.float32)})["o"]),
        np.full((1, 1, 1), 2.0 * 4.0 + 1.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    assert gain.param is not None and offset.constant is not None
    gain.param.assign(np.full((1,), 10.0, dtype=np.float32))
    np.testing.assert_allclose(
        to_numpy(composed({"pc_z": np.full((1, 1, 1), 2.0, dtype=np.float32)})["o"]),
        np.full((1, 1, 1), 2.0 * 10.0 + 1.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_composition_rebuilds_layers_that_read_their_predecessor():
    """Only stored state travels. A layer configured from its predecessor is
    rebuilt where it now sits: a block that reads the last sample of its input
    reads the last sample of the stream it is called with - which has the
    shape the block reads, a wider window being refused."""
    u = Input("ctx_u", dim=1, tw=1)
    body = Modely("ctx_body", inputs=[u], outputs=[Output("ctx_out", u * 1.0)]).build()

    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([Input("ctx_w3", dim=1, tw=3)])
    w = Input("ctx_w", dim=1, tw=1)
    composed = Modely(
        "ctx_composed", inputs=[w], outputs=[Output("o", body([w]))]
    ).build()

    values = np.array([[[10.0]], [[30.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        to_numpy(composed({"ctx_w": values})["o"]),
        values,
        rtol=1e-5,
        atol=1e-5,
    )


def test_rebuilding_a_model_keeps_its_weights():
    """build() is idempotent for the values: a second build reuses the layers
    it already made instead of re-initializing them."""
    u = Input("rebuild_u", dim=1, tw=1)
    fir = Fir(features=1, bias=False)([u])
    model = Modely("rebuild_model", inputs=[u], outputs=[Output("o", fir)]).build()
    fir.kernel.assign(np.full((1, 1), 7.0, dtype=np.float32))

    model.build()

    np.testing.assert_allclose(
        to_numpy(model({"rebuild_u": np.ones((1, 1, 1), dtype=np.float32)})["o"]),
        np.full((1, 1, 1), 7.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_building_without_keeping_the_weights_initializes_every_one():
    """build(keep_weights=False) gives every weight its initial value again:
    a Parameter the value it was declared with, a layer's weight a new draw,
    the statistics of a BatchNorm mean 0 and variance 1 - frozen ones too."""
    u = Input("reinit_u", dim=2, tw=3)
    gain = Parameter("reinit_gain", value=[2.0]).freeze()
    filtered = Fir(features=1, name="reinit_fir")([u]) * gain
    normalized = BatchNorm(name="reinit_norm")([PReLU(name="reinit_prelu")([filtered])])
    model = Modely(
        "reinit", inputs=[u], outputs=[Output("reinit_out", normalized)]
    ).build()
    drawn = to_numpy(model.parameters["reinit_fir_kernel"].param).copy()
    for parameter in model.parameters:
        parameter.param.assign(np.full(parameter.param.shape, 3.0, np.float32))
    # Training mode moves the statistics.
    assert model.model is not None
    model.model({"reinit_u": np.random.rand(4, 3, 2).astype(np.float32)}, training=True)
    assert np.any(to_numpy(normalized._layer.moving_mean) != 0.0)

    model.build(keep_weights=False)

    parameters = model.parameters
    np.testing.assert_allclose(to_numpy(parameters["reinit_gain"].param), [2.0])
    assert not parameters["reinit_gain"].trainable
    kernel = to_numpy(parameters["reinit_fir_kernel"].param)
    assert not np.allclose(kernel, 3.0) and not np.allclose(kernel, drawn)
    for name, value in (
        ("reinit_fir_bias", 0.0),
        ("reinit_prelu_alpha", 0.0),
        ("reinit_norm_gamma", 1.0),
        ("reinit_norm_beta", 0.0),
    ):
        np.testing.assert_allclose(to_numpy(parameters[name].param), value)
    np.testing.assert_allclose(to_numpy(normalized._layer.moving_mean), 0.0)
    np.testing.assert_allclose(to_numpy(normalized._layer.moving_variance), 1.0)


@pytest.mark.slow
def test_building_without_keeping_the_weights_reaches_blocks_bodies_and_keras_layers():
    """The weights of a block, of a Loop body and of a KerasFunction's Keras
    layer are initialized again too, in place: the block built on its own
    computes with the same values as the larger model."""
    import keras

    from nnodely import KerasFunction, Loop

    s = Input("reach_s", tw=1)
    block_gain = Parameter("reach_block_gain", value=[2.0])
    block = Modely(
        "reach_block", inputs=[s], outputs=[Output("reach_block_out", s * block_gain)]
    ).build()

    state, step = Input("reach_state", tw=1), Input("reach_step", tw=1)
    body_gain = Parameter("reach_body_gain", value=[0.5])
    next_state = Output("reach_next", state * body_gain + step)
    body = Modely("reach_body", inputs=[state, step], outputs=[next_state]).build()
    steps = Input("reach_steps", seq=3, tw=1)
    rollout = Loop(f=body, feedback={state: next_state})({step: steps})

    dense = keras.layers.Dense(1, kernel_initializer="ones")
    u = Input("reach_u", tw=1)
    model = Modely(
        "reach",
        inputs=[u, steps],
        outputs=[
            Output("reach_out", KerasFunction(dense)([block([u])])),
            Output("reach_rollout", rollout),
        ],
    ).build()
    block_gain.param.assign([5.0])  # type: ignore
    body_gain.param.assign([5.0])  # type: ignore
    dense.weights[0].assign([[5.0]])

    model.build(keep_weights=False)

    np.testing.assert_allclose(to_numpy(block_gain.param), [2.0])
    np.testing.assert_allclose(to_numpy(body_gain.param), [0.5])
    np.testing.assert_allclose(to_numpy(dense.kernel), [[1.0]])
    one = np.ones((1, 1, 1), np.float32)
    np.testing.assert_allclose(
        to_numpy(block({"reach_s": one})["reach_block_out"]), [[[2.0]]]
    )


@pytest.mark.parametrize("layer", [PReLU, BatchNorm])
def test_a_block_on_a_dynamic_sequence_keeps_its_weights_on_a_fixed_one(layer):
    """A layer holding Keras weights of its own was made again where the
    block met a fixed sequence: the larger model started from new weights,
    and the block went on computing with its own."""
    name = f"dynamic_{layer.__name__}"
    s = Input(f"{name}_s", seq=-1, tw=1)
    block = Modely(
        f"{name}_block",
        inputs=[s],
        outputs=[Output(f"{name}_out", layer(name=f"{name}_layer")([s]))],
    ).build()
    for parameter in block.parameters:
        parameter.param.assign(np.full(parameter.param.shape, 0.5, np.float32))
    r = Input(f"{name}_r", seq=5, tw=1)
    model = Modely(
        f"{name}_model", inputs=[r], outputs=[Output(f"{name}_y", block([r]))]
    ).build()

    assert model.model is not None and block.model is not None
    assert {id(w) for w in model.model.weights} == {id(w) for w in block.model.weights}
    values = -np.ones((1, 5, 1, 1), np.float32)
    np.testing.assert_allclose(
        to_numpy(model({f"{name}_r": values})[f"{name}_y"]),
        to_numpy(block({f"{name}_s": values})[f"{name}_out"]),
    )


def test_composition_rejects_a_block_whose_input_shape_changes():
    """Weights fit the shape they were made for, and they are Parameters of
    the block's graph: a block called with an input of a different shape
    cannot use them - and its own weights are left alone."""
    u = Input("fit_u", dim=1, tw=1)
    body_fir = Fir(features=1, bias=False)([u])
    body = Modely("fit_body", inputs=[u], outputs=[Output("fit_out", body_fir)]).build()
    body_fir.kernel.assign(np.full((1, 1), 5.0, dtype=np.float32))

    # Three features where the block was built for one: the call is refused.
    wide = Input("fit_wide", dim=3, tw=1)
    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([wide])

    # The block still evaluates with the weights it had.
    np.testing.assert_allclose(
        to_numpy(body({"fit_u": np.ones((1, 1, 1), dtype=np.float32)})["fit_out"]),
        np.full((1, 1, 1), 5.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_a_model_used_as_a_block_takes_one_stream_per_input():
    a = Input("block_a", tw=1)
    b = Input("block_b", tw=1)
    body = Modely(
        "two_inputs", inputs=[a, b], outputs=[Output("block_sum", a + b)]
    ).build()
    x = Input("block_x", tw=1)
    y = Input("block_y", tw=1)
    z = Input("block_z", tw=1)

    # One stream short: b was once promoted to an input of the outer graph,
    # and a stream too many was dropped without a word.
    with pytest.raises(ValueError, match="2 inputs"):
        body([x])
    with pytest.raises(ValueError, match="2 inputs"):
        body([x, y, z])
    with pytest.raises(ValueError, match="2 inputs .* called with a value alone"):
        body(x)
    with pytest.raises(TypeError, match="list of streams"):
        body(3.0)

    outer = Modely(
        "uses_block",
        inputs=[x, y],
        outputs=[Output("outer_sum", body([x, 10.0 * y]))],
    ).build()
    result = outer(
        {
            "block_x": np.full((1, 1, 1), 1.0, dtype=np.float32),
            "block_y": np.full((1, 1, 1), 2.0, dtype=np.float32),
        }
    )
    assert [node.name for node in outer.inference_inputs] == ["block_x", "block_y"]
    np.testing.assert_allclose(to_numpy(result["outer_sum"]), [[[21.0]]])


def test_a_block_takes_its_streams_as_a_model_takes_its_arrays():
    a = Input("keyed_a", tw=1)
    b = Input("keyed_b", tw=1)
    # Not built: a block is composed from its graph alone.
    body = Modely("keyed_body", inputs=[a, b], outputs=[Output("keyed_diff", a - b)])
    x = Input("keyed_x", tw=1)
    y = Input("keyed_y", tw=1)

    # Keyed by Input or by name, in any order, or listed in the order of the
    # inputs: one block. A key of no input of the block is ignored.
    calls = [
        body([x, y]),
        body({a: x, b: y}),
        body({"keyed_b": y, "keyed_a": x}),
        body({a: x, "keyed_b": y, "keyed_unknown": x}),
    ]
    outer = Modely(
        "keyed_outer",
        inputs=[x, y],
        outputs=[Output(f"keyed_out{i}", call) for i, call in enumerate(calls)],
    ).build()
    result = outer(
        {
            "keyed_x": np.full((1, 1, 1), 5.0, dtype=np.float32),
            "keyed_y": np.full((1, 1, 1), 2.0, dtype=np.float32),
        }
    )
    for i in range(len(calls)):
        np.testing.assert_allclose(to_numpy(result[f"keyed_out{i}"]), [[[3.0]]])

    # Every declared input needs a stream, given once; streams and arrays
    # are not mixed.
    with pytest.raises(ValueError, match=r"\['keyed_b'\] have none"):
        body({a: x})
    with pytest.raises(ValueError, match="two values for its input 'keyed_a'"):
        body({a: x, "keyed_a": x, b: y})
    with pytest.raises(TypeError, match="streams and arrays together"):
        body({a: x, b: np.ones((1, 1, 1), dtype=np.float32)})


def test_a_block_with_one_input_takes_its_stream_or_array_alone():
    a = Input("alone_a", tw=1)
    body = Modely("alone_body", inputs=[a], outputs=[Output("alone_twice", a * 2.0)])
    x = Input("alone_x", tw=1)
    outer = Modely(
        "alone_outer",
        inputs=[x],
        outputs=[
            Output("alone_bare", body(x)),
            Output("alone_listed", body([x])),
        ],
    ).build()
    data = np.full((1, 1, 1), 5.0, dtype=np.float32)
    # Called on its array alone as well.
    result = outer(data)
    np.testing.assert_allclose(to_numpy(result["alone_bare"]), [[[10.0]]])
    np.testing.assert_allclose(to_numpy(result["alone_listed"]), [[[10.0]]])

    b = Input("alone_b", tw=1)
    pair = Modely("alone_pair", inputs=[a, b], outputs=[Output("alone_sum", a + b)])
    with pytest.raises(ValueError, match="2 inputs .* called with a value alone"):
        pair(x)


def test_layers_created_apart_have_their_own_weights_whatever_their_names(tmp_path):
    # Once shared by name: two Fir named alike were one layer.
    x = Input("twins_x", tw=2)
    first = Fir(features=1, bias=False, name="twin")([x])
    second = Fir(features=1, bias=False, name="twin")([x])
    shared = Fir(features=1, bias=False, name="shared_fir")
    model = Modely(
        "twins",
        inputs=[x],
        outputs=[
            Output("twins_first", first),
            Output("twins_second", second),
            Output("twins_shared", shared([x]) + shared([x])),
        ],
    ).build()
    assert first.kernel is not None and second.kernel is not None
    first.kernel.assign(np.full((2, 1), 1.0, dtype=np.float32))
    second.kernel.assign(np.full((2, 1), 3.0, dtype=np.float32))

    assert model.inference_model is not None
    keras_names = [layer.name for layer in model.inference_model.layers]
    assert {"twin", "twin_2"} <= set(keras_names)
    assert len(keras_names) == len(set(keras_names))
    # Three layers: the two twins, and the one applied twice.
    assert len(model.inference_model.weights) == 3
    data = {"twins_x": np.ones((1, 2, 1), dtype=np.float32)}
    result = model(data)
    np.testing.assert_allclose(to_numpy(result["twins_first"]), [[[2.0]]])
    np.testing.assert_allclose(to_numpy(result["twins_second"]), [[[6.0]]])

    # The weights file is keyed by the layers' classes and order, not by
    # these names: saved and loaded, each twin keeps its own weights.
    model.save(tmp_path / "twins")
    result = Modely.load(tmp_path / "twins")(data)
    np.testing.assert_allclose(to_numpy(result["twins_first"]), [[[2.0]]])
    np.testing.assert_allclose(to_numpy(result["twins_second"]), [[[6.0]]])


def test_a_block_takes_the_shapes_it_was_declared_with():
    x = Input("shaped_x", tw=1)
    body = Modely(
        "shaped_body", inputs=[x], outputs=[Output("shaped_out", x * 2.0)]
    ).build()

    # A rollout of 3 steps where the block reads one sample was declared as
    # one sample and computed on three; now it is refused, as is a stream of
    # another window or dim.
    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([Input("shaped_a", seq=3, tw=1)])
    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([Input("shaped_b", tw=2)])
    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([Input("shaped_wide", dim=2, tw=1)])

    # An Input given whole is read through the block's own window.
    c = Input("shaped_c", tw=1)
    outer = Modely(
        "shaped_outer", inputs=[c], outputs=[Output("shaped_whole", body([c]))]
    ).build()
    assert c.shape.dimensions[:2] == ((), 1)
    result = outer({"shaped_c": np.full((2, 1, 1), 3.0, dtype=np.float32)})
    np.testing.assert_allclose(to_numpy(result["shaped_whole"]).ravel(), [6.0, 6.0])

    # A block on a dynamic sequence takes one of any length; an Input given
    # whole needs that sequence.
    s = Input("shaped_s", seq=-1, tw=1)
    seq_body = Modely(
        "shaped_seq_body",
        inputs=[s],
        outputs=[Output("shaped_seq_out", s * 2.0)],
    ).build()
    assert seq_body([Input("shaped_d", seq=5, tw=1)]).shape.dimensions[:2] == (
        (None,),
        1,
    )
    with pytest.raises(ValueError, match="sequence"):
        seq_body([Input("shaped_e", tw=1)])


def test_a_block_reads_the_newest_sample_of_a_wider_stream_it_is_bound_to():
    ## The block reads x directly and through a window of 3. Bound to a
    ## computed window of 3, its direct read is the newest sample of it
    x = Input("newest_x", tw=3)
    taps = Parameter("newest_taps", value=[[1.0], [1.0], [1.0]])
    window_sum = Fir(features=1, kernel=taps, bias=False)([x])
    body = Modely(
        "newest_body",
        inputs=[x],
        outputs=[Output("newest_out", TimeSelect(-1)(x) * 10.0 + window_sum)],
    ).build()

    u = Input("newest_u", tw=3)
    outer = Modely(
        "newest_outer",
        inputs=[u],
        outputs=[Output("newest_result", body([u * 1.0]))],
    ).build()

    values = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    ## 3 * 10, the newest sample, plus 1 + 2 + 3, the window
    np.testing.assert_allclose(
        to_numpy(outer({"newest_u": values})["newest_result"]), [[[36.0]]]
    )


def test_model_augmentation(tmp_path):
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1, 1, 400)
    ys_true = np.zeros_like(xs)
    ys_true[1:] = 0.5 * xs[:-1] + 2.0 * xs[1:] - 1.0

    x_tr, y_tr = Input("tr_x", dim=1, tw=2), Input("tr_y", dim=1, tw=1)
    kernel1 = Parameter("tr_kernel", value=[[0.0], [0.0]])
    bias1 = Parameter("tr_bias", value=[0.0])
    filtered = Fir(features=1, kernel=kernel1, bias=bias1)([x_tr])
    prediction = Output("tr_pred", filtered)
    train_model = Modely("tr_model", inputs=[x_tr], outputs=[prediction])
    train_model.minimize("fit", prediction, y_tr)
    train_model.build()

    ## Train the model
    train_data = DataLoader(train_model, source={"tr_x": xs, "tr_y": ys_true})
    train_model.train(train_data, epochs=60, batch_size=32, lr=0.05, printer=None)
    np.testing.assert_allclose(to_numpy(filtered.kernel), [[0.5], [2.0]], atol=1e-3)
    np.testing.assert_allclose(to_numpy(filtered.bias), [-1.0], atol=1e-3)

    ## Save the nnodely model with trained weights
    train_model.save(path=tmp_path / "augmented_model.nnodely")

    ## Delete the python model object to simulate a fresh start
    del train_model
    del x_tr
    del y_tr
    del filtered
    del prediction

    ## Load the model with trained weights
    loaded_model = Modely.load(path=tmp_path / "augmented_model.nnodely")

    ## Add some layers to the loaded model
    extra_input = Input("extra_input", dim=1, tw=2)
    extra_param = Parameter("extra_param", value=[[1.0], [2.0]])
    fir2 = Fir(features=1, kernel=extra_param, bias=False)([extra_input])
    extra_output = Output("extra_output", fir2 + loaded_model([extra_input]))
    updated_model = Modely(
        "updated_model", inputs=[extra_input], outputs=[extra_output]
    )
    updated_model.build()  ## Here I don't want to re-initialize the weights of the loaded model, only the extra layers. The loaded model's weights should remain intact.

    ## The weights of a model, by name: .param is the variable the built model
    ## computes with (.value is the value a Parameter was declared with)
    parameters = updated_model.parameters
    assert [parameter.name for parameter in parameters] == [
        "extra_param",
        "tr_kernel",
        "tr_bias",
    ]
    np.testing.assert_allclose(
        to_numpy(parameters["tr_kernel"].param), [[0.5], [2.0]], atol=1e-3
    )
    np.testing.assert_allclose(to_numpy(parameters["tr_bias"].param), [-1.0], atol=1e-3)
    np.testing.assert_allclose(
        to_numpy(parameters["extra_param"].param), [[1.0], [2.0]], atol=1e-6
    )

    ## The augmented model computes with them: fir2 + the trained filter
    window = np.array([[[0.2], [-0.4]]], dtype=np.float32)
    result = to_numpy(updated_model({"extra_input": window})["extra_output"])
    expected = (1.0 * 0.2 + 2.0 * -0.4) + (0.5 * 0.2 + 2.0 * -0.4 - 1.0)
    np.testing.assert_allclose(result, [[[expected]]], atol=1e-3)


def test_models_gives_the_called_model_itself():
    """A model called as a block is reached by name, or by position: the
    object itself, sharing its weights with the caller and run on its own
    inputs."""
    x = Input("sub_x", tw=1)
    gain = Parameter("sub_gain", value=[2.0])
    model1 = Modely(
        "sub_model1", inputs=[x], outputs=[Output("sub_out", x * gain)]
    ).build()
    z = Input("sub_z", tw=1)
    model2 = Modely(
        "sub_model2", inputs=[z], outputs=[Output("sub_o", model1([z]) + 1.0)]
    ).build()

    sub_model1 = model2.models["sub_model1"]
    assert sub_model1 is model1
    assert model2.models[0] is model1
    assert sub_model1.parameters["sub_gain"] is model2.parameters["sub_gain"]

    # Not a copy: a weight changed shows through both.
    gain.param.assign([5.0])  # type: ignore
    one = np.ones((1, 1, 1), dtype=np.float32)
    np.testing.assert_allclose(
        to_numpy(sub_model1({"sub_x": one})["sub_out"]), [[[5.0]]]
    )
    np.testing.assert_allclose(to_numpy(model2({"sub_z": one})["sub_o"]), [[[6.0]]])

    assert str(model2.models) == "\n".join(
        [
            "┏━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓",
            "┃ idx ┃ model      ┃ inputs ┃ outputs ┃ parameters ┃ built ┃",
            "┡━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩",
            "│ 0   │ sub_model1 │ sub_x  │ sub_out │ 1          │ yes   │",
            "└─────┴────────────┴────────┴─────────┴────────────┴───────┘",
        ]
    )
    assert repr(model2.models) == str(model2.models)


def test_models_reaches_every_depth_each_model_once():
    """The models called by the models called, and the bodies of Loops, are
    listed too; one called twice is listed once. Unbuilt ones are listed as
    they are."""
    from nnodely import Loop

    v = Input("deep_v", tw=1)
    inner = Modely("deep_inner", inputs=[v], outputs=[Output("deep_in", v * 2.0)])
    w = Input("deep_w", tw=1)
    outer = Modely(
        "deep_outer",
        inputs=[w],
        outputs=[Output("deep_out", inner([w]) + inner([w * 3.0]))],
    )
    state, step = Input("deep_state", tw=1), Input("deep_step", tw=1)
    next_state = Output("deep_next", state * 0.5 + step)
    body = Modely("deep_body", inputs=[state, step], outputs=[next_state]).build()
    steps = Input("deep_steps", seq=3, tw=1)
    rollout = Loop(f=body, feedback={state: next_state})({step: steps})
    x = Input("deep_x", tw=1)
    caller = Modely(
        "deep_caller",
        inputs=[x, steps],
        outputs=[Output("deep_o", outer([x])), Output("deep_r", rollout)],
    )

    assert list(caller.models) == [outer, body, inner]
    assert caller.models["deep_inner"] is inner
    assert caller.models["deep_outer"].models["deep_inner"] is inner
    assert caller.models["deep_body"] is body
    assert not caller.models["deep_outer"].built


def test_models_refuses_a_name_two_models_carry():
    u = Input("amb_u", tw=1)
    first = Modely("amb_twin", inputs=[u], outputs=[Output("amb_one", u * 1.0)])
    second = Modely("amb_twin", inputs=[u], outputs=[Output("amb_three", u * 3.0)])
    x = Input("amb_x", tw=1)
    caller = Modely(
        "amb_caller",
        inputs=[x],
        outputs=[Output("amb_o1", first([x])), Output("amb_o2", second([x]))],
    )

    with pytest.raises(ValueError, match=r"'amb_twin' is ambiguous.*\[0, 1\]"):
        caller.models["amb_twin"]
    # By position, each is still reached.
    assert caller.models[0] is first and caller.models[1] is second
    with pytest.raises(KeyError, match="no model named 'amb_missing'"):
        caller.models["amb_missing"]


def test_a_loaded_model_lists_the_models_it_was_saved_with(tmp_path):
    """A model called as a block is saved as a model of its own, as the body
    of a Loop is, so the loaded model calls it again."""
    from nnodely import Loop

    u = Input("saved_u", tw=1)
    block = Modely("saved_block", inputs=[u], outputs=[Output("saved_b", u * 2.0)])
    state, step = Input("saved_state", tw=1), Input("saved_step", tw=1)
    next_state = Output("saved_next", state * 0.5 + step)
    body = Modely("saved_body", inputs=[state, step], outputs=[next_state]).build()
    steps = Input("saved_steps", seq=3, tw=1)
    rollout = Loop(f=body, feedback={state: next_state})({step: steps})
    x = Input("saved_x", tw=1)
    model = Modely(
        "saved_model",
        inputs=[x, steps],
        outputs=[Output("saved_o", block([x])), Output("saved_r", rollout)],
    ).build()
    assert [m.name for m in model.models] == ["saved_block", "saved_body"]

    model.save(tmp_path / "saved_model")
    loaded = Modely.load(tmp_path / "saved_model")

    assert [m.name for m in loaded.models] == ["saved_block", "saved_body"]
    assert loaded.models["saved_block"].name == "saved_block"
