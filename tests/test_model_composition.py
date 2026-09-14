import numpy as np
import pytest

from nnodely import (
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

    z = Input("z", dim=1)
    z_fir = Fir(out_features=1)([model1([z.sw(10), z.sw(10)])])
    z_out = Output("z_pred", z_fir)

    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    dummy_z = np.ones((3, 1, 10), dtype=np.float32)
    result = model2([dummy_z])

    assert "z_pred" in result
    assert result["z_pred"].shape == (3, 1, 1)


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
