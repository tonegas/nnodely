import numpy as np
import pytest

from nnodely import (
    Constant,
    DataLoader,
    Fir,
    Input,
    Linear,
    Modely,
    Output,
    Parameter,
)
from conftest import to_numpy
from nnodely import TimeSelect


def test_model_composition():
    x = Input("x", dim=1, sw=10)
    y = Input("y", dim=1, sw=10)

    # Pinned: the body's weights have to survive being inlined, which is the
    # whole point of calling a built model. The biases start at zero.
    result_taps = Parameter("result_taps", value=np.ones((10, 2)))
    result_fir = Fir(features=2, kernel=result_taps)([x + y])
    x_out = Output("x_pred", result_fir)

    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    z = Input("z", dim=1, sw=10)
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
    x = Input("reuse_x", dim=1, sw=1)
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
    u = Input("twice_u", dim=1, sw=1)
    body = Modely(
        "twice_body", inputs=[u], outputs=[Output("twice_doubled", u * 2.0)]
    ).build()

    x = Input("twice_x", dim=1, sw=1)
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
    u = Input("shared_u", dim=1, sw=1)
    scaled = Linear(features=1, kernel="ones", bias=False)(u)
    body = Modely(
        "shared_body", inputs=[u], outputs=[Output("shared_scaled", scaled)]
    ).build()

    x = Input("shared_x", dim=1, sw=1)
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
    v = Input("nested_v", dim=1, sw=1)
    inner = Modely(
        "nested_inner",
        inputs=[v],
        outputs=[Output("nested_inner_out", gain * v)],
    ).build()

    w = Input("nested_w", dim=1, sw=1)
    outer = Modely(
        "nested_outer",
        inputs=[w],
        outputs=[Output("nested_outer_out", inner([w]) + 1.0)],
    ).build()

    x = Input("nested_x", dim=1, sw=1)
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
    u = Input("collide_u", dim=1, sw=1)
    body = Modely(
        "collide_body", inputs=[u], outputs=[Output("collide_out", u * 2.0)]
    ).build()

    x = Input("collide_x", dim=1, sw=1)
    first = body([x])
    second = body([x + 1.0])
    with pytest.raises(ValueError, match="two different outputs named"):
        Modely("collide_caller", inputs=[x], outputs=[first, second]).build()


def test_model_saving_loading_and_compose(tmp_path):
    x = Input("x", dim=1, sw=3)
    y = Input("y", dim=1, sw=3)

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

    z = Input("z", dim=1, sw=3)
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
    u = Input("share_u", dim=1, sw=2)
    body_fir = Fir(features=1, bias=False)([u])
    body = Modely(
        "share_body", inputs=[u], outputs=[Output("share_out", body_fir)]
    ).build()
    body_fir.kernel.assign(np.array([[1.0], [10.0]], dtype=np.float32))

    z = Input("share_z", dim=1, sw=2)
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
    u = Input("train_u", dim=1, sw=1)
    gain = Parameter("train_gain", value=[0.5])
    body_fir = Fir(features=1, kernel=gain, bias=False)([u])
    body = Modely(
        "train_body", inputs=[u], outputs=[Output("train_out", body_fir)]
    ).build()

    z = Input("train_z", dim=1, sw=1)
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
    u = Input("pc_u", dim=1, sw=1)
    body = Modely(
        "pc_body",
        inputs=[u],
        outputs=[Output("pc_out", u * gain + offset)],
    ).build()

    z = Input("pc_z", dim=1, sw=1)
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
    u = Input("ctx_u", dim=1, sw=1)
    body = Modely("ctx_body", inputs=[u], outputs=[Output("ctx_out", u * 1.0)]).build()

    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([Input("ctx_w3", dim=1, sw=3)])
    w = Input("ctx_w", dim=1, sw=1)
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
    u = Input("rebuild_u", dim=1, sw=1)
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


def test_composition_rejects_a_block_whose_input_shape_changes():
    """Weights fit the shape they were made for, and they are Parameters of
    the block's graph: a block called with an input of a different shape
    cannot use them - and its own weights are left alone."""
    u = Input("fit_u", dim=1, sw=1)
    body_fir = Fir(features=1, bias=False)([u])
    body = Modely("fit_body", inputs=[u], outputs=[Output("fit_out", body_fir)]).build()
    body_fir.kernel.assign(np.full((1, 1), 5.0, dtype=np.float32))

    # Three features where the block was built for one: the call is refused.
    wide = Input("fit_wide", dim=3, sw=1)
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
    a = Input("block_a", sw=1)
    b = Input("block_b", sw=1)
    body = Modely(
        "two_inputs", inputs=[a, b], outputs=[Output("block_sum", a + b)]
    ).build()
    x = Input("block_x", sw=1)
    y = Input("block_y", sw=1)
    z = Input("block_z", sw=1)

    # One stream short: b was once promoted to an input of the outer graph,
    # and a stream too many was dropped without a word.
    with pytest.raises(ValueError, match="2 inputs"):
        body([x])
    with pytest.raises(ValueError, match="2 inputs"):
        body([x, y, z])
    with pytest.raises(TypeError, match="list of streams"):
        body(x)

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


def test_layers_created_apart_have_their_own_weights_whatever_their_names():
    # Once shared by name: two Fir named alike were one layer.
    x = Input("twins_x", sw=2)
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
    assert {"twin", "twin_1"} <= set(keras_names)
    assert len(keras_names) == len(set(keras_names))
    # Three layers: the two twins, and the one applied twice.
    assert len(model.inference_model.weights) == 3
    result = model({"twins_x": np.ones((1, 2, 1), dtype=np.float32)})
    np.testing.assert_allclose(to_numpy(result["twins_first"]), [[[2.0]]])
    np.testing.assert_allclose(to_numpy(result["twins_second"]), [[[6.0]]])


def test_a_block_takes_the_shapes_it_was_declared_with():
    x = Input("shaped_x", sw=1)
    body = Modely(
        "shaped_body", inputs=[x], outputs=[Output("shaped_out", x * 2.0)]
    ).build()

    # A rollout of 3 steps where the block reads one sample was declared as
    # one sample and computed on three; now it is refused, as is a stream of
    # another window or dim.
    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([Input("shaped_a", seq=3, sw=1)])
    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([Input("shaped_b", sw=2)])
    with pytest.raises(ValueError, match="takes a stream of that shape"):
        body([Input("shaped_wide", dim=2, sw=1)])

    # An Input given whole is read through the block's own window.
    c = Input("shaped_c", sw=1)
    outer = Modely(
        "shaped_outer", inputs=[c], outputs=[Output("shaped_whole", body([c]))]
    ).build()
    assert c.shape.dimensions[:2] == ((), 1)
    result = outer({"shaped_c": np.full((2, 1, 1), 3.0, dtype=np.float32)})
    np.testing.assert_allclose(to_numpy(result["shaped_whole"]).ravel(), [6.0, 6.0])

    # A block on a dynamic sequence takes one of any length; an Input given
    # whole needs that sequence.
    s = Input("shaped_s", seq=-1, sw=1)
    seq_body = Modely(
        "shaped_seq_body",
        inputs=[s],
        outputs=[Output("shaped_seq_out", s * 2.0)],
    ).build()
    assert seq_body([Input("shaped_d", seq=5, sw=1)]).shape.dimensions[:2] == (
        (None,),
        1,
    )
    with pytest.raises(ValueError, match="sequence"):
        seq_body([Input("shaped_e", sw=1)])


def test_a_block_reads_the_newest_sample_of_a_wider_stream_it_is_bound_to():
    ## The block reads x directly and through a window of 3. Bound to a
    ## computed window of 3, its direct read is the newest sample of it
    x = Input("newest_x", sw=3)
    taps = Parameter("newest_taps", value=[[1.0], [1.0], [1.0]])
    window_sum = Fir(features=1, kernel=taps, bias=False)([x])
    body = Modely(
        "newest_body",
        inputs=[x],
        outputs=[Output("newest_out", TimeSelect(-1)(x) * 10.0 + window_sum)],
    ).build()

    u = Input("newest_u", sw=3)
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

    x_tr, y_tr = Input("tr_x", dim=1, sw=2), Input("tr_y", dim=1, sw=1)
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
    extra_input = Input("extra_input", dim=1, sw=2)
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
