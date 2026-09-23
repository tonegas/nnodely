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


def test_model_composition():
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    result_fir = Fir(out_features=2)([x.sw(10) + y.sw(10)])
    x_out = Output("x_pred", result_fir)

    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()
    # Pinned before composing: the body's weights have to survive being
    # inlined, which is the whole point of calling a built model.
    result_fir.kernel.assign(np.ones((10, 2), dtype=np.float32))
    result_fir.bias.assign(np.zeros((2,), dtype=np.float32))

    z = Input("z", dim=1)
    z_fir = Fir(out_features=1)([model1([z.sw(10), z.sw(10)])])
    z_out = Output("z_pred", z_fir)

    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()
    z_fir.kernel.assign(np.ones((2, 1), dtype=np.float32))
    z_fir.bias.assign(np.zeros((1,), dtype=np.float32))

    dummy_z = np.ones((3, 1, 10), dtype=np.float32)
    result = model2([dummy_z])

    assert "z_pred" in result
    assert result["z_pred"].shape == (3, 1, 1)
    # The body sums 10 samples of (z + z) into each of its 2 features, and the
    # outer Fir sums those: 10 * 2 = 20 per feature, 40 in total.
    np.testing.assert_allclose(
        to_numpy(result["z_pred"]),
        np.full((3, 1, 1), 40.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_layer_reuse_shares_weights():
    # Applying one layer twice must share its weights, as it does in Keras: the
    # two applications are separate graph nodes that carry the same name.
    x = Input("reuse_x", dim=1)
    linear = Linear(out_features=1, use_bias=False, initializer="ones")
    first, second = linear(x.last()), linear(x.last() * 2.0)
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
    u = Input("twice_u", dim=1)
    body = Modely(
        "twice_body", inputs=[u], outputs=[Output("twice_doubled", u.last() * 2.0)]
    ).build()

    x = Input("twice_x", dim=1)
    first = body([x.last()])  # 2 * 1 = 2
    second = body([x.last() + 1.0])  # 2 * (1 + 1) = 4
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


if __name__ == "__main__":
    test_model_called_twice_keeps_calls_independent()


def test_model_called_twice_shares_weights():
    # Each call inlines its own copy of the body, and those copies carry the
    # body's layer names, so one set of weights has to serve every call.
    u = Input("shared_u", dim=1)
    scaled = Linear(out_features=1, use_bias=False, initializer="ones")(u.last())
    body = Modely(
        "shared_body", inputs=[u], outputs=[Output("shared_scaled", scaled)]
    ).build()

    x = Input("shared_x", dim=1)
    first = body([x.last()])
    second = body([x.last() * 2.0])
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
    v = Input("nested_v", dim=1)
    inner = Modely(
        "nested_inner",
        inputs=[v],
        outputs=[Output("nested_inner_out", gain * v.last())],
    ).build()

    w = Input("nested_w", dim=1)
    outer = Modely(
        "nested_outer",
        inputs=[w],
        outputs=[Output("nested_outer_out", inner([w.last()]) + 1.0)],
    ).build()

    x = Input("nested_x", dim=1)
    first = outer([x.last()])
    second = outer([x.last() + 1.0])
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
    u = Input("collide_u", dim=1)
    body = Modely(
        "collide_body", inputs=[u], outputs=[Output("collide_out", u.last() * 2.0)]
    ).build()

    x = Input("collide_x", dim=1)
    first = body([x.last()])
    second = body([x.last() + 1.0])
    with pytest.raises(ValueError, match="two different outputs named"):
        Modely("collide_caller", inputs=[x], outputs=[first, second]).build()


def test_model_saving_loading_and_compose(tmp_path):
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    result_fir = Fir(out_features=1, use_bias=False)([x.sw(3) + y.sw(3)])
    x_out = Output("x_pred", result_fir)

    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()
    result_fir.kernel.assign(np.ones((3, 1), dtype=np.float32))

    dummy_x = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    dummy_y = np.array([[[4.0, 5.0, 6.0]]], dtype=np.float32)
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

    z = Input("z", dim=1)
    z_fir = Fir(out_features=1, use_bias=True)([model_loaded([z.sw(3), z.sw(3)])])
    z_out = Output("z_pred", z_fir)

    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()
    z_fir.kernel.assign(np.ones((1, 1), dtype=np.float32))
    z_fir.bias.assign(np.array(5.0, dtype=np.float32).reshape((1,)))

    dummy_z = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32)
    result_model2 = model2({"z": dummy_z})
    np.testing.assert_allclose(
        to_numpy(result_model2["z_pred"]),
        np.array([[[20.0 + 40.0 + 60.0 + 5.0]]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    # Save and load the composed model
    model2.save("composed_model")
    loaded_model2 = Modely.load("composed_model")
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
    u = Input("share_u", dim=1)
    body_fir = Fir(out_features=1, use_bias=False)([u.sw(2)])
    body = Modely(
        "share_body", inputs=[u], outputs=[Output("share_out", body_fir)]
    ).build()
    body_fir.kernel.assign(np.array([[1.0], [10.0]], dtype=np.float32))

    z = Input("share_z", dim=1)
    composed = Modely(
        "share_composed", inputs=[z], outputs=[Output("o", body([z.sw(2)]))]
    ).build()

    values = np.array([[[3.0, 4.0]]], dtype=np.float32)
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
    u = Input("train_u", dim=1)
    body_fir = Fir(out_features=1, use_bias=False)([u.last()])
    body = Modely(
        "train_body", inputs=[u], outputs=[Output("train_out", body_fir)]
    ).build()
    body_fir.kernel.assign(np.full((1, 1), 0.5, dtype=np.float32))

    z = Input("train_z", dim=1)
    composed_out = Output("composed_out", body([z.last()]))
    composed = Modely("train_composed", inputs=[z], outputs=[composed_out])
    composed.minimize("error", source=composed_out, target=3.0, loss="mse")
    composed.build()

    data = DataLoader(composed, source={"train_z": np.ones((8, 1), dtype=np.float32)})
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
    gain = Parameter(name="pc_gain", value=[[4.0]])
    offset = Constant(name="pc_offset", value=[1.0])
    u = Input("pc_u", dim=1)
    body = Modely(
        "pc_body",
        inputs=[u],
        outputs=[Output("pc_out", u.last() * gain + offset)],
    ).build()

    z = Input("pc_z", dim=1)
    composed = Modely(
        "pc_composed", inputs=[z], outputs=[Output("o", body([z.last()]))]
    ).build()

    np.testing.assert_allclose(
        to_numpy(composed({"pc_z": np.full((1, 1, 1), 2.0, dtype=np.float32)})["o"]),
        np.full((1, 1, 1), 2.0 * 4.0 + 1.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    assert gain.param is not None and offset.constant is not None
    gain.param.assign(np.full((1, 1), 10.0, dtype=np.float32))
    np.testing.assert_allclose(
        to_numpy(composed({"pc_z": np.full((1, 1, 1), 2.0, dtype=np.float32)})["o"]),
        np.full((1, 1, 1), 2.0 * 10.0 + 1.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_composition_rebuilds_layers_that_read_their_predecessor():
    """Only stored state travels. A layer configured from its predecessor is
    rebuilt where it now sits: a block that reads the last sample of its input
    must read the last sample of the stream it is called with, not the oldest
    one it happened to be built against."""
    u = Input("ctx_u", dim=1)
    body = Modely(
        "ctx_body", inputs=[u], outputs=[Output("ctx_out", u.last() * 1.0)]
    ).build()

    w = Input("ctx_w", dim=1)
    composed = Modely(
        "ctx_composed", inputs=[w], outputs=[Output("o", body([w.sw(3)]))]
    ).build()

    values = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        to_numpy(composed({"ctx_w": values})["o"]),
        np.full((1, 1, 1), 30.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rebuilding_a_model_keeps_its_weights():
    """build() is idempotent for the values: a second build reuses the layers
    it already made instead of re-initializing them."""
    u = Input("rebuild_u", dim=1)
    fir = Fir(out_features=1, use_bias=False)([u.last()])
    model = Modely("rebuild_model", inputs=[u], outputs=[Output("o", fir)]).build()
    fir.kernel.assign(np.full((1, 1), 7.0, dtype=np.float32))

    model.build()

    np.testing.assert_allclose(
        to_numpy(model({"rebuild_u": np.ones((1, 1, 1), dtype=np.float32)})["o"]),
        np.full((1, 1, 1), 7.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_composition_rebuilds_a_block_whose_input_shape_changes():
    """Weights fit the shape they were made for. A block called with an input
    of a different shape cannot share them, so it is rebuilt where it lands -
    and the block's own weights are left alone."""
    u = Input("fit_u", dim=1)
    body_fir = Fir(out_features=1, use_bias=False)([u.last()])
    body = Modely("fit_body", inputs=[u], outputs=[Output("fit_out", body_fir)]).build()
    body_fir.kernel.assign(np.full((1, 1), 5.0, dtype=np.float32))

    # Three features where the block was built for one: its kernel does not fit.
    wide = Input("fit_wide", dim=3)
    composed = Modely(
        "fit_composed", inputs=[wide], outputs=[Output("o", body([wide.last()]))]
    ).build()

    result = composed({"fit_wide": np.ones((1, 3, 1), dtype=np.float32)})
    assert to_numpy(result["o"]).shape == (1, 1, 1)

    # The block still evaluates with the weights it had.
    np.testing.assert_allclose(
        to_numpy(body({"fit_u": np.ones((1, 1, 1), dtype=np.float32)})["fit_out"]),
        np.full((1, 1, 1), 5.0, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
