from nnodely import Input, Output, Modely, DataLoader, Loop
from conftest import to_numpy
import os
import warnings
import numpy as np
import pytest


def test_dataset_creation_and_iteration():
    x = Input("x", dim=1)
    y = Input("y", dim=1)
    z = Input("z", dim=1)

    x_stream1 = x.sw(3)  ## 3 samples of x in the past
    y_stream1 = y.sw([0, 3])  ## 3 samples of y in the future
    z_stream1 = z.sw(
        [4, 2]
    )  ## 4 samples of z in the past and 2 samples of z in the future

    x_stream2 = x.last()  ## 1 sample of x in the past
    y_stream2 = y.next()  ## 1 sample of y in the future
    z_stream2 = z.sw(3)  ## 3 samples of z in the past

    x_stream3 = x.sw(5)  ## 5 samples of x in the past
    y_stream3 = y.sw(
        [2, 2]
    )  ## 2 samples of y in the past and 2 samples of y in the future
    z_stream3 = z.sw(
        [1, 3]
    )  ## 1 sample of z in the past and 3 samples of z in the future

    add1 = x_stream1 + y_stream1 + z_stream2
    add2 = x_stream2 + y_stream2
    add3 = y_stream3 + z_stream3

    out1 = Output("out1", add1)
    out2 = Output("out2", add2)
    out3 = Output("out3", add3)
    out_x_stream1 = Output("out_x_stream1", x_stream1)
    out_y_stream1 = Output("out_y_stream1", y_stream1)
    out_z_stream1 = Output("out_z_stream1", z_stream1)
    out_x_stream2 = Output("out_x_stream2", x_stream2)
    out_y_stream2 = Output("out_y_stream2", y_stream2)
    out_z_stream2 = Output("out_z_stream2", z_stream2)
    out_x_stream3 = Output("out_x_stream3", x_stream3)
    out_y_stream3 = Output("out_y_stream3", y_stream3)
    out_z_stream3 = Output("out_z_stream3", z_stream3)
    model = Modely(
        "model1",
        inputs=[x, y, z],
        outputs=[
            out1,
            out2,
            out3,
            out_x_stream1,
            out_y_stream1,
            out_z_stream1,
            out_x_stream2,
            out_y_stream2,
            out_z_stream2,
            out_x_stream3,
            out_y_stream3,
            out_z_stream3,
        ],
    )
    model.build()

    ## ------ Load dataset -------
    data_train = DataLoader(
        model,
        format={"x": "data_1", "y": "data_2", "z": "data_3"},
        source=os.path.join("tests", "datasets"),
    )

    ## ------ Iterate through the dataset -------
    for batch in data_train:
        assert "x" in batch and "y" in batch and "z" in batch
        assert batch["x"].shape[0] == 1  # Check if x has the correct dimension
        assert batch["y"].shape[0] == 1  # Check if y has the correct dimension
        assert batch["z"].shape[0] == 1  # Check if z has the correct dimension
        assert batch["x"].shape[1] == 5  # Check if x has the correct dimension
        assert batch["y"].shape[1] == 5  # Check if y has the correct dimension
        assert batch["z"].shape[1] == 7  # Check if z has the correct dimension


def test_sequence_windows_are_created_on_temporal_windows():
    x = Input("x", dim=1, seq=3)
    target = Input("target", dim=1)
    x_window = x.sw(5)
    output = Output("out", x_window)
    model = Modely("sequence_dataset", inputs=[x], outputs=[output])
    model.minimize("error", source=output, target=target.last(), loss="mse")
    model.build()

    loader = DataLoader(
        model,
        source={
            "x": np.arange(9, dtype=np.float32),
            "target": np.arange(9, dtype=np.float32),
        },
    )

    assert loader.dataset["x"].shape == (3, 1, 5, 3)
    assert loader.dataset["target"].shape == (3, 1, 1)
    np.testing.assert_array_equal(
        loader.dataset["x"][0, 0],
        np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        loader.dataset["target"][:, 0, 0],
        np.array([6, 7, 8], dtype=np.float32),
    )


def test_explicit_dataloader_normalization_and_denormalization():
    x = Input("normalization_x", dim=2)
    target = Input("normalization_target")
    relation = x.sw(2)
    output = Output("normalization_output", relation)
    model = Modely("normalization_model", inputs=[x], outputs=[output])
    model.minimize("error", output, target.last(), loss="mse")
    model.build()

    loader = DataLoader(
        model,
        source={
            "normalization_x": np.array(
                [[0.0, 100.0], [2.0, 200.0], [4.0, 300.0], [6.0, 400.0]],
                dtype=np.float32,
            ),
            "normalization_target": np.array(
                [10.0, 20.0, 30.0, 40.0], dtype=np.float32
            ),
        },
    )
    original = {
        name: np.array(values, copy=True) for name, values in loader.as_dict().items()
    }

    loader.normalize(method="minmax")

    normalized_x = loader.dataset["normalization_x"]
    np.testing.assert_array_equal(np.min(normalized_x, axis=(0, 2)), [-1.0, -1.0])
    np.testing.assert_array_equal(np.max(normalized_x, axis=(0, 2)), [1.0, 1.0])
    assert np.min(loader.dataset["normalization_target"]) == -1.0
    assert np.max(loader.dataset["normalization_target"]) == 1.0

    restored = loader.denormalize(loader.as_dict())
    for name in original:
        if isinstance(restored, dict):
            np.testing.assert_allclose(
                to_numpy(restored[name]), to_numpy(original[name]), atol=1e-6
            )
        else:
            np.testing.assert_allclose(
                to_numpy(restored), to_numpy(original[name]), atol=1e-6
            )

    normalized_prediction = loader.dataset["normalization_target"][:1]
    restored_prediction = loader.denormalize(
        {"normalization_output": normalized_prediction}
    )
    if isinstance(restored_prediction, dict):
        np.testing.assert_allclose(
            to_numpy(restored_prediction["normalization_output"]),
            to_numpy(original["normalization_target"][:1]),
            atol=1e-6,
        )
    else:
        np.testing.assert_allclose(
            to_numpy(restored_prediction),
            to_numpy(original["normalization_target"][:1]),
            atol=1e-6,
        )

    loader.denormalize()
    for name in original:
        np.testing.assert_array_equal(loader.dataset[name], original[name])


def test_standard_normalization_handles_constant_inputs():
    x = Input("constant_normalization_x")
    model = Modely(
        "constant_normalization_model",
        inputs=[x],
        outputs=[Output("constant_normalization_output", x.last())],
    ).build()
    loader = DataLoader(
        model,
        source={"constant_normalization_x": np.full(4, 5.0, dtype=np.float32)},
    )

    loader.normalize(method="standard")
    np.testing.assert_array_equal(loader.dataset["constant_normalization_x"], 0.0)
    restored = loader.denormalize(
        loader.dataset["constant_normalization_x"],
        name="constant_normalization_x",
    )
    np.testing.assert_array_equal(restored, 5.0)


def test_step_subsamples_sample_windows():
    x = Input("step_x", dim=1)
    model = Modely(
        "step_windows_model",
        inputs=[x],
        outputs=[Output("step_out", x.sw(3))],
    ).build()

    loader = DataLoader(
        model,
        source={"step_x": np.array([1, 2, 3, 4, 5], dtype=np.float32)},
        step=2,
    )

    ## [1, 2, 3, 4, 5] with sw=3 and step=2 -> [[1, 2, 3], [3, 4, 5]]
    assert loader.dataset["step_x"].shape == (2, 1, 3)
    np.testing.assert_array_equal(
        loader.dataset["step_x"][:, 0],
        np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32),
    )


def test_step_keeps_multiple_inputs_aligned():
    x = Input("step_align_x", dim=1)
    y = Input("step_align_y", dim=1)
    model = Modely(
        "step_align_model",
        inputs=[x, y],
        outputs=[Output("step_align_out", x.sw(3) + y.last())],
    ).build()

    loader = DataLoader(
        model,
        source={
            "step_align_x": np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "step_align_y": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        },
        step=2,
    )

    np.testing.assert_array_equal(
        loader.dataset["step_align_x"][:, 0],
        np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32),
    )
    ## Every window is aligned to its last temporal sample
    np.testing.assert_array_equal(
        loader.dataset["step_align_y"][:, 0],
        np.array([[3], [5]], dtype=np.float32),
    )


def test_step_subsamples_sequences():
    x = Input("step_seq_x", dim=1, seq=3)
    model = Modely(
        "step_sequence_model",
        inputs=[x],
        outputs=[Output("step_seq_out", x.sw(2))],
    ).build()

    loader = DataLoader(
        model,
        source={"step_seq_x": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)},
        step=2,
    )

    ## [1, ..., 6] with sw=2, seq=3 and step=2 ->
    ## [[[1, 2], [2, 3], [3, 4]], [[3, 4], [4, 5], [5, 6]]]
    assert loader.dataset["step_seq_x"].shape == (2, 1, 2, 3)
    np.testing.assert_array_equal(
        loader.dataset["step_seq_x"][0, 0].T,
        np.array([[1, 2], [2, 3], [3, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loader.dataset["step_seq_x"][1, 0].T,
        np.array([[3, 4], [4, 5], [5, 6]], dtype=np.float32),
    )


def test_step_subsamples_sequences_with_seq_length():
    x = Input("step_seq_length_x", dim=1, seq=(None,))
    model = Modely(
        "step_seq_length_model",
        inputs=[x],
        outputs=[Output("step_seq_length_out", x.sw(2))],
    ).build()

    loader = DataLoader(
        model,
        source={"step_seq_length_x": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)},
        seq_length=3,
        step=2,
    )

    assert loader.dataset["step_seq_length_x"].shape == (2, 1, 2, 3)
    np.testing.assert_array_equal(
        loader.dataset["step_seq_length_x"][0, 0].T,
        np.array([[1, 2], [2, 3], [3, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loader.dataset["step_seq_length_x"][1, 0].T,
        np.array([[3, 4], [4, 5], [5, 6]], dtype=np.float32),
    )


def test_step_does_not_mix_csv_files():
    x = Input("step_csv_x", dim=1)
    model = Modely(
        "step_csv_model",
        inputs=[x],
        outputs=[Output("step_csv_out", x.sw(2))],
    ).build()

    stepped = DataLoader(
        model,
        format={"step_csv_x": "data_1"},
        source=os.path.join("tests", "datasets"),
        step=3,
    )

    ## Each file is windowed independently, so the step restarts on every file
    expected = np.concatenate(
        [
            DataLoader(
                model,
                format={"step_csv_x": "data_1"},
                source=os.path.join("tests", "datasets"),
                csv_glob=csv_name,
                step=3,
            ).dataset["step_csv_x"]
            for csv_name in ("test.csv", "test2.csv")
        ]
    )
    np.testing.assert_array_equal(stepped.dataset["step_csv_x"], expected)


def test_invalid_step_raises():
    x = Input("step_invalid_x", dim=1)
    model = Modely(
        "step_invalid_model",
        inputs=[x],
        outputs=[Output("step_invalid_out", x.last())],
    ).build()

    with np.testing.assert_raises(ValueError):
        DataLoader(
            model, source={"step_invalid_x": np.arange(4, dtype=np.float32)}, step=0
        )


def test_format_maps_multiple_columns_to_one_input():
    x = Input("multi_column_x", dim=3)
    model = Modely(
        "multi_column_model",
        inputs=[x],
        outputs=[Output("multi_column_out", x.sw(2))],
    ).build()

    loader = DataLoader(
        model,
        format={"multi_column_x": ["a", "b", "c"]},
        source=os.path.join("tests", "datasets", "test.csv"),
    )

    ## One window per pair of consecutive rows, features in the declared order
    assert loader.dataset["multi_column_x"].shape == (9, 3, 2)
    np.testing.assert_array_equal(
        loader.dataset["multi_column_x"][0],
        np.array([[2, 6], [3, 7], [4, 8]], dtype=np.float32),
    )

    ## Positional indices select the same three columns
    by_index = DataLoader(
        model,
        format={"multi_column_x": [3, 4, 5]},
        source=os.path.join("tests", "datasets", "test.csv"),
    )
    np.testing.assert_array_equal(
        by_index.dataset["multi_column_x"], loader.dataset["multi_column_x"]
    )


def test_format_rejects_a_column_count_that_does_not_match_dim():
    x = Input("wrong_width_x", dim=3)
    model = Modely(
        "wrong_width_model",
        inputs=[x],
        outputs=[Output("wrong_width_out", x.sw(2))],
    ).build()

    with np.testing.assert_raises(ValueError):
        DataLoader(
            model,
            format={"wrong_width_x": ["a", "b"]},
            source=os.path.join("tests", "datasets", "test.csv"),
        )


def test_on_short_skips_simulations_that_are_too_short():
    x = Input("short_x", dim=1)
    model = Modely(
        "short_model",
        inputs=[x],
        outputs=[Output("short_out", x.sw(4))],
    ).build()

    simulations = [
        {"short_x": np.arange(1, 6, dtype=np.float32)},  ## 5 samples -> 2 windows
        {"short_x": np.arange(10, 13, dtype=np.float32)},  ## 3 samples -> too short
    ]

    with np.testing.assert_raises(ValueError):
        DataLoader(model, source=simulations)

    loader = DataLoader(model, source=simulations, on_short="skip")
    assert loader.dataset["short_x"].shape == (2, 1, 4)
    np.testing.assert_array_equal(
        loader.dataset["short_x"][:, 0],
        np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32),
    )

    ## Skipping every simulation leaves nothing to train on
    with np.testing.assert_raises(ValueError):
        DataLoader(model, source=simulations[1:], on_short="skip")


def test_simulations_are_windowed_independently():
    x = Input("independent_x", dim=1)
    model = Modely(
        "independent_model",
        inputs=[x],
        outputs=[Output("independent_out", x.sw(2))],
    ).build()

    loader = DataLoader(
        model,
        source=[
            {"independent_x": np.array([1, 2, 3], dtype=np.float32)},
            {"independent_x": np.array([10, 11], dtype=np.float32)},
        ],
    )

    ## No window spans two simulations
    np.testing.assert_array_equal(
        loader.dataset["independent_x"][:, 0],
        np.array([[1, 2], [2, 3], [10, 11]], dtype=np.float32),
    )


def test_multiple_dynamic_sequence_lengths_are_rejected():
    x = Input("two_dynamic_x", dim=1, seq=(None, None))
    model = Modely(
        "two_dynamic_model",
        inputs=[x],
        outputs=[Output("two_dynamic_out", x.sw(2))],
    ).build()

    with np.testing.assert_raises(ValueError):
        DataLoader(
            model,
            source={"two_dynamic_x": np.arange(20, dtype=np.float32)},
            seq_length=3,
        )


def test_full_sequence_spans_each_simulation_and_pads():
    x = Input("full_x", dim=1, seq=(None,))
    model = Modely(
        "full_model",
        inputs=[x],
        outputs=[Output("full_out", x.sw(1))],
    ).build()

    loader = DataLoader(
        model,
        source=[
            {"full_x": np.arange(1, 6, dtype=np.float32)},
            {"full_x": np.arange(10, 13, dtype=np.float32)},
        ],
        seq_length="full",
    )

    ## One sample per simulation, padded on the rollout axis to the longest one
    assert len(loader) == 2
    assert loader.dataset["full_x"].shape == (2, 1, 1, 5)
    np.testing.assert_array_equal(
        loader.dataset["full_x"][:, 0, 0],
        np.array([[1, 2, 3, 4, 5], [10, 11, 12, 12, 12]], dtype=np.float32),
    )

    ## The mask marks the real steps of every simulation
    assert loader.padded_inputs == {"full_x"}
    np.testing.assert_array_equal(
        loader.mask,
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=bool),
    )


def test_full_sequence_leaves_the_inner_levels_alone():
    x = Input("full_nested_x", dim=1, seq=(2, None))
    model = Modely(
        "full_nested_model",
        inputs=[x],
        outputs=[Output("full_nested_out", x.sw(1))],
    ).build()

    loader = DataLoader(
        model,
        source=[
            {"full_nested_x": np.arange(6, dtype=np.float32)},
            {"full_nested_x": np.arange(4, dtype=np.float32)},
        ],
        seq_length="full",
    )

    ## The inner level still spans 2, so the outermost one covers what is left
    assert loader.dataset["full_nested_x"].shape == (2, 1, 1, 2, 5)
    np.testing.assert_array_equal(
        loader.mask,
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=bool),
    )


def test_full_sequence_of_equal_simulations_needs_no_mask():
    x = Input("uniform_x", dim=1, seq=(None,))
    model = Modely(
        "uniform_model",
        inputs=[x],
        outputs=[Output("uniform_out", x.sw(1))],
    ).build()

    loader = DataLoader(
        model,
        source=[
            {"uniform_x": np.arange(4, dtype=np.float32)},
            {"uniform_x": np.arange(10, 14, dtype=np.float32)},
        ],
        seq_length="full",
    )

    assert loader.dataset["uniform_x"].shape == (2, 1, 1, 4)
    assert loader.mask is None
    assert loader.padded_inputs == set()


def test_full_sequence_rejects_step_and_inner_dynamic_sequences():
    x = Input("full_step_x", dim=1, seq=(None,))
    model = Modely(
        "full_step_model",
        inputs=[x],
        outputs=[Output("full_step_out", x.sw(1))],
    ).build()

    ## The sequence already spans the simulation, so there is nothing to step over
    with np.testing.assert_raises(ValueError):
        DataLoader(
            model,
            source={"full_step_x": np.arange(6, dtype=np.float32)},
            seq_length="full",
            step=2,
        )

    inner = Input("full_inner_x", dim=1, seq=(None, 2))
    inner_model = Modely(
        "full_inner_model",
        inputs=[inner],
        outputs=[Output("full_inner_out", inner.sw(1))],
    ).build()

    ## 'full' resolves the outermost sequence only
    with np.testing.assert_raises(ValueError):
        DataLoader(
            inner_model,
            source={"full_inner_x": np.arange(6, dtype=np.float32)},
            seq_length="full",
        )


def test_normalization_ignores_padded_steps():
    x = Input("padded_norm_x", dim=1, seq=(None,))
    model = Modely(
        "padded_norm_model",
        inputs=[x],
        outputs=[Output("padded_norm_out", x.sw(1))],
    ).build()

    loader = DataLoader(
        model,
        source=[
            {"padded_norm_x": np.arange(1, 6, dtype=np.float32)},
            {"padded_norm_x": np.array([10, 11], dtype=np.float32)},
        ],
        seq_length="full",
    )
    loader.normalize(method="standard")

    ## The three repeated steps of the short simulation must not move the mean
    observed = np.array([1, 2, 3, 4, 5, 10, 11], dtype=np.float64)
    assert loader.normalization_stats["padded_norm_x"]["offset"].ravel()[
        0
    ] == pytest.approx(observed.mean())
    assert loader.normalization_stats["padded_norm_x"]["scale"].ravel()[
        0
    ] == pytest.approx(observed.std())


def test_uncollected_loop_warns_on_simulations_of_different_lengths():
    body_x = Input("warn_body_x", dim=1)
    body_out = Output("warn_body_out", body_x.last())
    body = Modely("warn_body", inputs=[body_x], outputs=[body_out]).build()

    seed = Input("warn_x", dim=1, seq=(None,))
    loop = Loop(
        f=body,
        callback={body_x: body_out},
        initial={body_x: seed},
        length=4,
        collect=False,
        name="warn_loop",
    )
    model = Modely(
        "warn_model", inputs=[seed], outputs=[Output("warn_out", loop)]
    ).build()

    ## The last rollout step of a short simulation lies past the end of its data
    with pytest.warns(UserWarning, match="collect=False"):
        DataLoader(
            model,
            source=[
                {"warn_x": np.arange(4, dtype=np.float32)},
                {"warn_x": np.arange(2, dtype=np.float32)},
            ],
            seq_length="full",
        )

    ## Equal lengths need no padding, so there is nothing to warn about
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        DataLoader(
            model,
            source=[
                {"warn_x": np.arange(4, dtype=np.float32)},
                {"warn_x": np.arange(10, 14, dtype=np.float32)},
            ],
            seq_length="full",
        )
