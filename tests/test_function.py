import keras
import numpy as np
import pytest
from conftest import requires_onnx_export, to_numpy

from nnodely import DataLoader, Input, KerasFunction, Linear, Modely, Output


# Custom functions as a user writes them: native Keras layers, not registered.
class Power(keras.layers.Layer):
    def __init__(self, exponent=2.0, **kwargs):
        super().__init__(**kwargs)
        self.exponent = exponent

    def call(self, x):
        return keras.ops.power(x, self.exponent)

    def get_config(self):
        return {**super().get_config(), "exponent": self.exponent}


class Shifted(keras.layers.Layer):
    def call(self, x, y):
        return x * x + y


class Gain(keras.layers.Layer):
    def build(self, input_shape):
        self.gain = self.add_weight(shape=(), initializer="ones", name="gain")

    def call(self, x):
        return self.gain * x


class Norm(keras.layers.Layer):
    """The length of the feature vector of every sample: dim (n,) to (1,)."""

    def call(self, x):
        return keras.ops.sqrt(keras.ops.sum(x * x, axis=-1, keepdims=True))


def test_keras_function_applies_a_keras_layer_to_a_stream():
    x = Input("kf_x")
    cubed = KerasFunction(Power(exponent=3.0))([x.sw(3)])  # an instance
    squared = KerasFunction(Power)([x.sw(3)])  # a class, made with no arguments
    model = Modely(
        "kf_model",
        inputs=[x],
        outputs=[Output("kf_cubed", cubed), Output("kf_squared", squared)],
    ).build()
    assert cubed.shape.dimensions == ((), 3, (1,))

    values = np.array([[[1.0], [2.0], [-3.0]]], dtype=np.float32)
    result = model({"kf_x": values})
    np.testing.assert_allclose(to_numpy(result["kf_cubed"]), values**3, rtol=1e-6)
    np.testing.assert_allclose(to_numpy(result["kf_squared"]), values**2, rtol=1e-6)


def test_keras_function_passes_its_streams_as_separate_arguments():
    a = Input("kf_a")
    b = Input("kf_b")
    shifted = KerasFunction(Shifted)([a, b])
    model = Modely(
        "kf_two_inputs", inputs=[a, b], outputs=[Output("kf_shifted", shifted)]
    ).build()

    result = model(
        {
            "kf_a": np.array([[[3.0]], [[-2.0]]], dtype=np.float32),
            "kf_b": np.array([[[1.0]], [[0.5]]], dtype=np.float32),
        }
    )
    np.testing.assert_allclose(to_numpy(result["kf_shifted"]).ravel(), [10.0, 4.5])


def test_keras_function_reads_its_output_shape_off_the_layer():
    v = Input("kf_v", dim=3, seq=-1)
    length = KerasFunction(Norm)([v.sw(2)])
    model = Modely("kf_norm", inputs=[v], outputs=[Output("kf_length", length)]).build()
    # The dim is reduced, the time window and the dynamic sequence kept.
    assert length.shape.dimensions == ((None,), 2, (1,))

    for steps in (3, 5):
        values = np.random.default_rng(steps).normal(size=(1, steps, 2, 3))
        result = to_numpy(model({"kf_v": values.astype(np.float32)})["kf_length"])
        np.testing.assert_allclose(
            result, np.linalg.norm(values, axis=-1, keepdims=True), rtol=1e-5
        )


def test_keras_function_shares_an_instance_and_makes_a_class_anew():
    x = Input("kf_share_x")
    gain = Gain()
    shared = [KerasFunction(gain)([x]) for _ in range(2)]
    apart = [KerasFunction(Gain)([x]) for _ in range(2)]
    model = Modely(
        "kf_share",
        inputs=[x],
        outputs=[
            Output(f"kf_{kind}_{index}", node)
            for kind, nodes in (("shared", shared), ("apart", apart))
            for index, node in enumerate(nodes)
        ],
    ).build()

    # One gain for the instance, used twice; one for each made class.
    assert model.model is not None and len(model.model.trainable_weights) == 3
    gain.gain.assign(4.0)
    result = model({"kf_share_x": np.ones((1, 1, 1), dtype=np.float32)})
    for name, expected in (
        ("kf_shared_0", 4.0),
        ("kf_shared_1", 4.0),
        ("kf_apart_0", 1.0),
        ("kf_apart_1", 1.0),
    ):
        np.testing.assert_allclose(to_numpy(result[name]).ravel(), [expected])


@pytest.mark.slow
def test_keras_function_trains_the_weights_of_its_layer():
    # gain * x against 3 on x = 1: one SGD step on (g - 3)^2 at lr 0.25 moves
    # the gain from 1 to 1 - 0.25 * 2 * (1 - 3) = 2.
    x = Input("kf_train_x")
    gain = Gain()
    out = Output("kf_train_out", KerasFunction(gain)([x]))
    model = Modely("kf_train", inputs=[x], outputs=[out])
    model.minimize("kf_fit", out, 3.0)
    model.build()

    data = DataLoader(model, source={"kf_train_x": np.ones((1, 1), dtype=np.float32)})
    model.train(data, epochs=1, batch_size=1, optimizer="sgd", lr=0.25, printer=None)

    np.testing.assert_allclose(to_numpy(gain.gain), 2.0, rtol=1e-5)


def test_keras_function_rejects_what_is_not_a_keras_layer():
    with pytest.raises(TypeError, match="cannot be made without arguments"):

        class NeedsArguments(keras.layers.Layer):
            def __init__(self, size):
                super().__init__()

        KerasFunction(NeedsArguments)
    with pytest.raises(TypeError, match="keras.layers.Layer subclass or instance"):
        KerasFunction(lambda x: x * 2)
    # An nnodely layer is not a Keras layer: it is used as it is.
    with pytest.raises(TypeError, match="keras.layers.Layer subclass or instance"):
        KerasFunction(Linear(features=1))


def _custom_function_model():
    # A class defined here, not at module level, nor registered by hand.
    class Offset(keras.layers.Layer):
        def call(self, x):
            return x + 10.0

    x = Input("kf_saved_x")
    y = Input("kf_saved_y")
    gain = Gain()
    model = Modely(
        "kf_saved",
        inputs=[x, y],
        outputs=[
            Output("kf_saved_cubed", KerasFunction(Power(exponent=3.0))([x])),
            Output("kf_saved_shifted", KerasFunction(Shifted)([x, y])),
            Output("kf_saved_gain", KerasFunction(gain)([x])),
            Output("kf_saved_offset", KerasFunction(Offset)([x])),
        ],
    ).build()
    gain.gain.assign(2.5)  # as if trained
    return model


_SAVED_OUTPUTS = (
    "kf_saved_cubed",
    "kf_saved_shifted",
    "kf_saved_gain",
    "kf_saved_offset",
)
_SAVED_DATA = {
    "kf_saved_x": np.array([[[1.0]], [[-2.0]], [[0.5]]], dtype=np.float32),
    "kf_saved_y": np.array([[[3.0]], [[1.0]], [[-1.0]]], dtype=np.float32),
}


def test_keras_functions_save_load_and_export_to_keras(tmp_path):
    model = _custom_function_model()
    before = model(_SAVED_DATA)
    x = _SAVED_DATA["kf_saved_x"]
    np.testing.assert_allclose(to_numpy(before["kf_saved_cubed"]), x**3, rtol=1e-6)
    np.testing.assert_allclose(to_numpy(before["kf_saved_gain"]), 2.5 * x, rtol=1e-6)

    model.save(tmp_path / "kf_saved")
    restored = Modely.load(tmp_path / "kf_saved")
    model.export_keras(tmp_path, "kf_saved")
    imported = Modely.import_keras(tmp_path / "kf_saved.keras")
    for result in (restored(_SAVED_DATA), imported(_SAVED_DATA)):  # type: ignore
        for name in _SAVED_OUTPUTS:
            np.testing.assert_allclose(
                to_numpy(result[name]), to_numpy(before[name]), rtol=1e-6
            )

    # Saved as the architecture alone: the gain is drawn again, the exponent
    # of the configured Power is kept.
    model.save(tmp_path / "kf_architecture", weights=False)
    fresh = Modely.load(tmp_path / "kf_architecture")(_SAVED_DATA)
    np.testing.assert_allclose(to_numpy(fresh["kf_saved_gain"]), x, rtol=1e-6)
    np.testing.assert_allclose(to_numpy(fresh["kf_saved_cubed"]), x**3, rtol=1e-6)


@pytest.mark.slow
@requires_onnx_export
def test_keras_functions_export_to_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _custom_function_model()
    data = {name: values[:1] for name, values in _SAVED_DATA.items()}
    before = model(data)

    model.export_onnx(tmp_path, "kf_saved")
    result = Modely.validate_onnx(tmp_path / "kf_saved.onnx", data, return_dict=True)

    assert isinstance(result, dict)
    for name in _SAVED_OUTPUTS:
        np.testing.assert_allclose(
            to_numpy(result[name]), to_numpy(before[name]), rtol=1e-5, atol=1e-6
        )


def test_one_keras_layer_wrapped_twice_saves_and_loads_as_one(tmp_path):
    # Two wrappers of one instance once saved its weights once and loaded two
    # layers: "expected 1 variables, but received 0".
    x = Input("kf_twice_x")
    gain = Gain()
    model = Modely(
        "kf_twice",
        inputs=[x],
        outputs=[
            Output("kf_twice_a", KerasFunction(gain)([x])),
            Output("kf_twice_b", KerasFunction(gain)([x])),
        ],
    ).build()
    gain.gain.assign(3.0)
    data = {"kf_twice_x": np.full((1, 1, 1), 2.0, dtype=np.float32)}

    model.save(tmp_path / "kf_twice")
    restored = Modely.load(tmp_path / "kf_twice")
    assert restored.model is not None and len(restored.model.trainable_weights) == 1
    for name in ("kf_twice_a", "kf_twice_b"):
        np.testing.assert_allclose(to_numpy(restored(data)[name]).ravel(), [6.0])
