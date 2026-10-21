import numpy as np
import pytest

from nnodely import Conv2D, DataLoader, Flatten, Input, Linear, Modely, Output, Reshape
from conftest import to_numpy


def _set_weights(conv, channels, filters, kernel_size):
    rng = np.random.default_rng(0)
    kernel = rng.normal(size=(*kernel_size, channels, filters)).astype(np.float32)
    bias = rng.normal(size=filters).astype(np.float32)
    conv.kernel.assign(kernel)
    conv.bias.assign(bias)
    return kernel, bias


def _convolve(images, kernel, bias, strides, padding):
    """Reference convolution of images laid out [samples, H, W, C]."""
    (kh, kw), (sh, sw) = kernel.shape[:2], strides
    if padding == "same":
        # Padded as Keras does: the odd pixel goes after.
        pads = []
        for length, size, stride in zip(images.shape[1:3], (kh, kw), (sh, sw)):
            total = max((-(-length // stride) - 1) * stride + size - length, 0)
            pads.append((total // 2, total - total // 2))
        images = np.pad(images, [(0, 0), *pads, (0, 0)])
    height = (images.shape[1] - kh) // sh + 1
    width = (images.shape[2] - kw) // sw + 1
    out = np.empty((len(images), height, width, kernel.shape[3]), np.float32)
    for i in range(height):
        for j in range(width):
            patch = images[:, i * sh : i * sh + kh, j * sw : j * sw + kw, :]
            out[:, i, j, :] = np.einsum("nhwc,hwcf->nf", patch, kernel) + bias
    return out


# ---------------------------------------------------------------------------
# Conv2D convolves every image of a stream with the same kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strides, padding, size",
    [(1, "valid", (3, 4)), (2, "valid", (2, 2)), (1, "same", (5, 6)), (2, "same", (3, 3))],
)
def test_conv2d_convolves_every_image_of_the_window(strides, padding, size):
    x = Input("conv_x", dim=(5, 6, 2))
    conv = Conv2D(4, 3, strides=strides, padding=padding, name="conv")(x.sw(3))
    model = Modely("conv_model", inputs=[x], outputs=[Output("conv_out", conv)]).build()
    assert conv.shape.dimensions == ((3,), (*size, 4))

    kernel, bias = _set_weights(conv, 2, 4, (3, 3))
    values = np.random.default_rng(1).normal(size=(2, 3, 5, 6, 2)).astype(np.float32)
    result = to_numpy(model({"conv_x": values})["conv_out"])

    assert result.shape == (2, 3, *size, 4)
    for step in range(3):
        expected = _convolve(values[:, step], kernel, bias, (strides, strides), padding)
        np.testing.assert_allclose(result[:, step], expected, rtol=1e-4, atol=1e-5)


def test_conv2d_keeps_the_outer_time_axes():
    x = Input("conv_seq_x", dim=(4, 5, 1))
    conv = Conv2D(2, (2, 3), name="conv_seq")(x.sw(2, seq=3))
    model = Modely(
        "conv_seq_model", inputs=[x], outputs=[Output("conv_seq_out", conv)]
    ).build()
    assert conv.shape.dimensions == ((3, 2), (3, 3, 2))

    kernel, bias = _set_weights(conv, 1, 2, (2, 3))
    values = np.random.default_rng(2).normal(size=(2, 3, 2, 4, 5, 1)).astype(np.float32)
    result = to_numpy(model({"conv_seq_x": values})["conv_seq_out"])

    assert result.shape == (2, 3, 2, 3, 3, 2)
    for index in range(3):
        for step in range(2):
            expected = _convolve(values[:, index, step], kernel, bias, (1, 1), "valid")
            np.testing.assert_allclose(
                result[:, index, step], expected, rtol=1e-4, atol=1e-5
            )


def test_conv2d_follows_a_dynamic_sequence_length():
    x = Input("conv_dynamic_x", dim=(3, 3, 1))
    conv = Conv2D(2, 2, name="conv_dynamic")(x.last(seq=-1))
    model = Modely(
        "conv_dynamic_model", inputs=[x], outputs=[Output("conv_dynamic_out", conv)]
    ).build()
    assert conv.shape.dimensions == ((None, 1), (2, 2, 2))

    for length in (2, 5):
        values = np.ones((1, length, 1, 3, 3, 1), dtype=np.float32)
        result = to_numpy(model({"conv_dynamic_x": values})["conv_dynamic_out"])
        assert result.shape == (1, length, 1, 2, 2, 2)


@pytest.mark.parametrize(
    "dim, kernel_size, message",
    [(16, 3, "height, width, channels"), ((4, 4), 3, "height, width, channels"),
     ((2, 8, 1), 3, "does not fit")],
    ids=["flat", "no_channels", "kernel_too_large"],
)  # fmt: skip
def test_conv2d_rejects_a_stream_it_cannot_convolve(dim, kernel_size, message):
    x = Input("conv_bad_x", dim=dim)
    with pytest.raises(ValueError, match=message):
        Conv2D(1, kernel_size)(x.last())


def test_conv2d_rejects_an_unknown_padding():
    with pytest.raises(ValueError, match="padding"):
        Conv2D(1, 3, padding="causal")


# ---------------------------------------------------------------------------
# Flatten merges the dim axes, so that Linear sees every feature
# ---------------------------------------------------------------------------


def test_flatten_merges_the_dim_axes_in_c_order():
    x = Input("flat_x", dim=(2, 3, 4))
    flat = Flatten(x.sw(3, seq=2))
    model = Modely("flat_model", inputs=[x], outputs=[Output("flat_out", flat)]).build()
    assert flat.shape.dimensions == ((2, 3), (24,))

    values = np.arange(2 * 2 * 3 * 24, dtype=np.float32).reshape(2, 2, 3, 2, 3, 4)
    result = to_numpy(model({"flat_x": values})["flat_out"])
    np.testing.assert_array_equal(result, values.reshape(2, 2, 3, 24))


def test_flatten_follows_a_dynamic_sequence_length():
    x = Input("flat_dynamic_x", dim=(2, 2))
    flat = Flatten(x.last(seq=-1))
    model = Modely(
        "flat_dynamic_model", inputs=[x], outputs=[Output("flat_dynamic_out", flat)]
    ).build()
    assert flat.shape.dimensions == ((None, 1), (4,))

    for length in (2, 5):
        values = np.random.default_rng(length).normal(size=(1, length, 1, 2, 2))
        result = to_numpy(
            model({"flat_dynamic_x": values.astype(np.float32)})["flat_dynamic_out"]
        )
        np.testing.assert_allclose(result, values.reshape(1, length, 1, 4), rtol=1e-6)


# ---------------------------------------------------------------------------
# Reshape lays the dim axes out again, undoing Flatten
# ---------------------------------------------------------------------------


def test_reshape_undoes_flatten():
    x = Input("reshape_x", dim=(2, 3, 4))
    image = Reshape((2, 3, 4))(Flatten(x.sw(3, seq=2)))
    model = Modely(
        "reshape_model", inputs=[x], outputs=[Output("reshape_out", image)]
    ).build()
    assert image.shape.dimensions == ((2, 3), (2, 3, 4))

    values = np.arange(2 * 2 * 3 * 24, dtype=np.float32).reshape(2, 2, 3, 2, 3, 4)
    result = to_numpy(model({"reshape_x": values})["reshape_out"])
    np.testing.assert_array_equal(result, values)


def test_reshape_follows_a_dynamic_sequence_length():
    x = Input("reshape_dynamic_x", dim=6)
    image = Reshape((2, 3))(x.last(seq=-1))
    model = Modely(
        "reshape_dynamic_model",
        inputs=[x],
        outputs=[Output("reshape_dynamic_out", image)],
    ).build()
    assert image.shape.dimensions == ((None, 1), (2, 3))

    for length in (2, 5):
        values = np.random.default_rng(length).normal(size=(1, length, 1, 6))
        result = to_numpy(
            model({"reshape_dynamic_x": values.astype(np.float32)})["reshape_dynamic_out"]
        )
        np.testing.assert_allclose(result, values.reshape(1, length, 1, 2, 3), rtol=1e-6)


def test_reshape_rejects_a_shape_of_another_size():
    x = Input("reshape_bad_x", dim=12)
    with pytest.raises(ValueError, match="cannot reshape"):
        Reshape((1, 3, 5))(x.last())


def test_reshape_round_trips_through_save(tmp_path):
    x = Input("saved_reshape_x", dim=6)
    out = Output("saved_reshape_out", Reshape((1, 2, 3))(x.sw(2)))
    model = Modely("saved_reshape", inputs=[x], outputs=[out]).build()
    values = np.arange(12, dtype=np.float32).reshape(1, 2, 6)

    model.save(tmp_path / "saved_reshape")
    restored = Modely.load(tmp_path / "saved_reshape")
    np.testing.assert_array_equal(
        to_numpy(restored({"saved_reshape_x": values})["saved_reshape_out"]),
        values.reshape(1, 2, 1, 2, 3),
    )


# ---------------------------------------------------------------------------
# A convolutional encoder trains, saves and exports like any other model
# ---------------------------------------------------------------------------


def _encoder(name):
    x = Input(f"{name}_x", dim=(6, 6, 1))
    y = Input(f"{name}_y")
    features = Conv2D(3, 3, strides=2, padding="same", name=f"{name}_conv")(x.last())
    out = Output(f"{name}_out", Linear(1, name=f"{name}_linear")(Flatten(features)))
    model = Modely(name, inputs=[x, y], outputs=[out])
    model.minimize(f"{name}_error", source=out, target=y.last())
    return model.build()


def test_a_convolutional_encoder_learns_where_the_bright_pixel_is():
    model = _encoder("bright")
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 6, size=256)
    images = np.zeros((256, 6, 6, 1), np.float32)
    images[np.arange(256), rows, rng.integers(0, 6, size=256), 0] = 1.0
    data = {"bright_x": images, "bright_y": rows.astype(np.float32) / 5}

    def error():
        result = to_numpy(model({"bright_x": images[:, None]})["bright_out"])
        return float(np.mean((result.reshape(-1) - data["bright_y"]) ** 2))

    before = error()
    model.train(DataLoader(model, source=data), epochs=200, batch_size=32, lr=1e-2, optimizer="adam")
    assert error() < 0.1 * before


def test_a_convolutional_encoder_round_trips_through_save_and_export(tmp_path):
    model = _encoder("saved_conv")
    values = np.random.default_rng(4).normal(size=(3, 1, 6, 6, 1)).astype(np.float32)
    expected = to_numpy(model({"saved_conv_x": values})["saved_conv_out"])

    model.save(tmp_path / "saved_conv")
    restored = Modely.load(tmp_path / "saved_conv")
    np.testing.assert_allclose(
        to_numpy(restored({"saved_conv_x": values})["saved_conv_out"]), expected, rtol=1e-5
    )
    conv = {node.name: node for node in restored.order}["saved_conv_conv"]
    assert isinstance(conv, Conv2D)
    assert (conv.filters, conv.kernel_size, conv.strides, conv.padding) == (
        3, (3, 3), (2, 2), "same",
    )  # fmt: skip

    model.export_keras(tmp_path, "saved_conv")
    keras_model = Modely.import_keras(tmp_path / "saved_conv.keras")
    keras_result = keras_model({"saved_conv_x": values}, training=False)  # type: ignore
    np.testing.assert_allclose(
        to_numpy(keras_result["saved_conv_out"]), expected, rtol=1e-5
    )
