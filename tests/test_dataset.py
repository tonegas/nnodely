from nnodely import Input, Output, Modely, DataLoader, Loop
from conftest import CONVERGED, to_numpy
import os
import warnings
import numpy as np
import pytest
from nnodely import TimeRange, TimeSelect


def test_dataset_creation_and_iteration():
    x = Input("x", dim=1, tw=5)
    y = Input("y", dim=1, tw=(2, 3))
    z = Input("z", dim=1, tw=(4, 3))

    x_stream1 = TimeRange(2, 5)(x)  ## 3 samples of x in the past
    y_stream1 = TimeRange(2, 5)(y)  ## 3 samples of y in the future
    z_stream1 = TimeRange(0, 6)(
        z
    )  ## 4 samples of z in the past and 2 samples of z in the future

    x_stream2 = TimeSelect(-1)(x)  ## 1 sample of x in the past
    y_stream2 = TimeSelect(-3)(y)  ## 1 sample of y in the future
    z_stream2 = TimeRange(1, 4)(z)  ## 3 samples of z in the past

    x_stream3 = x  ## 5 samples of x in the past
    y_stream3 = TimeRange(0, 4)(
        y
    )  ## 2 samples of y in the past and 2 samples of y in the future
    z_stream3 = TimeRange(3, 7)(
        z
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
        source=os.path.join(os.path.dirname(__file__), "datasets"),
    )

    ## ------ Iterate through the dataset -------
    for batch in data_train:
        assert "x" in batch and "y" in batch and "z" in batch
        assert batch["x"].shape[1] == 1  # Check if x has the correct dimension
        assert batch["y"].shape[1] == 1  # Check if y has the correct dimension
        assert batch["z"].shape[1] == 1  # Check if z has the correct dimension
        assert batch["x"].shape[0] == 5  # Check if x has the correct dimension
        assert batch["y"].shape[0] == 5  # Check if y has the correct dimension
        assert batch["z"].shape[0] == 7  # Check if z has the correct dimension


def test_sequence_windows_are_created_on_temporal_windows():
    x = Input("x", dim=1, seq=3, tw=5)
    target = Input("target", dim=1, tw=1)
    x_window = x
    output = Output("out", x_window)
    model = Modely("sequence_dataset", inputs=[x], outputs=[output])
    model.minimize("error", source=output, target=target, loss="mse")
    model.build()

    loader = DataLoader(
        model,
        source={
            "x": np.arange(9, dtype=np.float32),
            "target": np.arange(9, dtype=np.float32),
        },
    )

    assert loader._dataset["x"].shape == (3, 3, 5, 1)
    assert loader._dataset["target"].shape == (3, 1, 1)
    np.testing.assert_array_equal(
        loader._dataset["x"][0, :, :, 0].T,
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
        loader._dataset["target"][:, 0, 0],
        np.array([6, 7, 8], dtype=np.float32),
    )


def test_step_subsamples_sample_windows():
    x = Input("step_x", dim=1, tw=3)
    model = Modely(
        "step_windows_model",
        inputs=[x],
        outputs=[Output("step_out", x)],
    ).build()

    loader = DataLoader(
        model,
        source={"step_x": np.array([1, 2, 3, 4, 5], dtype=np.float32)},
        step=2,
    )

    ## [1, 2, 3, 4, 5] with tw=3 and step=2 -> [[1, 2, 3], [3, 4, 5]]
    assert loader._dataset["step_x"].shape == (2, 3, 1)
    np.testing.assert_array_equal(
        loader._dataset["step_x"][:, :, 0],
        np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32),
    )


def test_step_keeps_multiple_inputs_aligned():
    x = Input("step_align_x", dim=1, tw=3)
    y = Input("step_align_y", dim=1, tw=1)
    model = Modely(
        "step_align_model",
        inputs=[x, y],
        # One output per window: a window of three and one of one sample have
        # different times, which are never added together.
        outputs=[
            Output("step_align_out", x),
            Output("step_align_y_out", y),
        ],
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
        loader._dataset["step_align_x"][:, :, 0],
        np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32),
    )
    ## Every window is aligned to its last temporal sample
    np.testing.assert_array_equal(
        loader._dataset["step_align_y"][:, :, 0],
        np.array([[3], [5]], dtype=np.float32),
    )


def test_step_subsamples_sequences():
    x = Input("step_seq_x", dim=1, seq=3, tw=2)
    model = Modely(
        "step_sequence_model",
        inputs=[x],
        outputs=[Output("step_seq_out", x)],
    ).build()

    loader = DataLoader(
        model,
        source={"step_seq_x": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)},
        step=2,
    )

    ## [1, ..., 6] with tw=2, seq=3 and step=2 ->
    ## [[[1, 2], [2, 3], [3, 4]], [[3, 4], [4, 5], [5, 6]]]
    assert loader._dataset["step_seq_x"].shape == (2, 3, 2, 1)
    np.testing.assert_array_equal(
        loader._dataset["step_seq_x"][0, :, :, 0],
        np.array([[1, 2], [2, 3], [3, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loader._dataset["step_seq_x"][1, :, :, 0],
        np.array([[3, 4], [4, 5], [5, 6]], dtype=np.float32),
    )


def test_step_subsamples_sequences_with_seq_length():
    x = Input("step_seq_length_x", dim=1, seq=-1, tw=2)
    model = Modely(
        "step_seq_length_model",
        inputs=[x],
        outputs=[Output("step_seq_length_out", x)],
    ).build()

    loader = DataLoader(
        model,
        source={"step_seq_length_x": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)},
        seq_length=3,
        step=2,
    )

    assert loader._dataset["step_seq_length_x"].shape == (2, 3, 2, 1)
    np.testing.assert_array_equal(
        loader._dataset["step_seq_length_x"][0, :, :, 0],
        np.array([[1, 2], [2, 3], [3, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loader._dataset["step_seq_length_x"][1, :, :, 0],
        np.array([[3, 4], [4, 5], [5, 6]], dtype=np.float32),
    )


def test_step_does_not_mix_csv_files():
    x = Input("step_csv_x", dim=1, tw=2)
    model = Modely(
        "step_csv_model",
        inputs=[x],
        outputs=[Output("step_csv_out", x)],
    ).build()

    stepped = DataLoader(
        model,
        format={"step_csv_x": "data_1"},
        source=os.path.join(os.path.dirname(__file__), "datasets"),
        step=3,
    )

    ## Each file is windowed independently, so the step restarts on every file
    expected = np.concatenate(
        [
            DataLoader(
                model,
                format={"step_csv_x": "data_1"},
                source=os.path.join(os.path.dirname(__file__), "datasets"),
                csv_glob=csv_name,
                step=3,
            )._dataset["step_csv_x"]
            for csv_name in ("test.csv", "test2.csv")
        ]
    )
    np.testing.assert_array_equal(stepped._dataset["step_csv_x"], expected)


def test_invalid_step_raises():
    x = Input("step_invalid_x", dim=1, tw=1)
    model = Modely(
        "step_invalid_model",
        inputs=[x],
        outputs=[Output("step_invalid_out", x)],
    ).build()

    with np.testing.assert_raises(ValueError):
        DataLoader(
            model, source={"step_invalid_x": np.arange(4, dtype=np.float32)}, step=0
        )


def test_format_maps_multiple_columns_to_one_input():
    x = Input("multi_column_x", dim=3, tw=2)
    model = Modely(
        "multi_column_model",
        inputs=[x],
        outputs=[Output("multi_column_out", x)],
    ).build()

    loader = DataLoader(
        model,
        format={"multi_column_x": ["a", "b", "c"]},
        source=os.path.join(os.path.dirname(__file__), "datasets", "test.csv"),
    )

    ## One window per pair of consecutive rows, features in the declared order
    assert loader._dataset["multi_column_x"].shape == (9, 2, 3)
    np.testing.assert_array_equal(
        loader._dataset["multi_column_x"][0].T,
        np.array([[2, 6], [3, 7], [4, 8]], dtype=np.float32),
    )

    ## Positional indices select the same three columns
    by_index = DataLoader(
        model,
        format={"multi_column_x": [3, 4, 5]},
        source=os.path.join(os.path.dirname(__file__), "datasets", "test.csv"),
    )
    np.testing.assert_array_equal(
        by_index._dataset["multi_column_x"], loader._dataset["multi_column_x"]
    )


def test_format_rejects_a_column_count_that_does_not_match_dim():
    x = Input("wrong_width_x", dim=3, tw=2)
    model = Modely(
        "wrong_width_model",
        inputs=[x],
        outputs=[Output("wrong_width_out", x)],
    ).build()

    with np.testing.assert_raises(ValueError):
        DataLoader(
            model,
            format={"wrong_width_x": ["a", "b"]},
            source=os.path.join(os.path.dirname(__file__), "datasets", "test.csv"),
        )


def test_a_matrix_input_reads_its_data_as_dim_or_flat_never_reordered():
    # Only the number of values was checked before reshaping: (N, 3, 2) fed to
    # dim=(2, 3) was taken, every matrix scrambled.
    x = Input("matrix_x", dim=(2, 3), tw=1)
    model = Modely(
        "matrix_model", inputs=[x], outputs=[Output("matrix_out", x)]
    ).build()
    matrices = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    for given in (matrices, matrices.reshape(4, 6), matrices.reshape(4, 1, 2, 3)):
        loader = DataLoader(model, source={"matrix_x": given})
        np.testing.assert_array_equal(loader.get_input("matrix_x")[:, 0], matrices)
    with pytest.raises(
        ValueError, match=r"laid out \(3, 2\) per timestep, but its dim=\(2, 3\)"
    ):
        DataLoader(model, source={"matrix_x": matrices.reshape(4, 3, 2)})


def test_on_short_skips_simulations_that_are_too_short():
    x = Input("short_x", dim=1, tw=4)
    model = Modely(
        "short_model",
        inputs=[x],
        outputs=[Output("short_out", x)],
    ).build()

    simulations = [
        {"short_x": np.arange(1, 6, dtype=np.float32)},  ## 5 samples -> 2 windows
        {"short_x": np.arange(10, 13, dtype=np.float32)},  ## 3 samples -> too short
    ]

    with np.testing.assert_raises(ValueError):
        DataLoader(model, source=simulations)

    loader = DataLoader(model, source=simulations, on_short="skip")
    assert loader._dataset["short_x"].shape == (2, 4, 1)
    np.testing.assert_array_equal(
        loader._dataset["short_x"][:, :, 0],
        np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32),
    )

    ## Skipping every simulation leaves nothing to train on
    with np.testing.assert_raises(ValueError):
        DataLoader(model, source=simulations[1:], on_short="skip")


def test_simulations_are_windowed_independently():
    x = Input("independent_x", dim=1, tw=2)
    model = Modely(
        "independent_model",
        inputs=[x],
        outputs=[Output("independent_out", x)],
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
        loader._dataset["independent_x"][:, :, 0],
        np.array([[1, 2], [2, 3], [10, 11]], dtype=np.float32),
    )


def test_multiple_dynamic_sequence_lengths_are_rejected():
    x = Input("two_dynamic_x", dim=1, seq=(-1, -1), tw=2)
    model = Modely(
        "two_dynamic_model",
        inputs=[x],
        outputs=[Output("two_dynamic_out", x)],
    ).build()

    with np.testing.assert_raises(ValueError):
        DataLoader(
            model,
            source={"two_dynamic_x": np.arange(20, dtype=np.float32)},
            seq_length=3,
        )


def test_full_sequence_spans_each_simulation_and_pads():
    x = Input("full_x", dim=1, seq=-1, tw=1)
    model = Modely(
        "full_model",
        inputs=[x],
        outputs=[Output("full_out", x)],
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
    assert loader._dataset["full_x"].shape == (2, 5, 1, 1)
    np.testing.assert_array_equal(
        loader._dataset["full_x"][:, :, 0, 0],
        np.array([[1, 2, 3, 4, 5], [10, 11, 12, 12, 12]], dtype=np.float32),
    )

    ## The mask marks the real steps of every simulation
    np.testing.assert_array_equal(
        loader.mask,
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=bool),
    )


def test_full_sequence_leaves_the_inner_levels_alone():
    x = Input("full_nested_x", dim=1, seq=(-1, 2), tw=1)
    model = Modely(
        "full_nested_model",
        inputs=[x],
        outputs=[Output("full_nested_out", x)],
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
    assert loader._dataset["full_nested_x"].shape == (2, 5, 2, 1, 1)
    np.testing.assert_array_equal(
        loader.mask,
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=bool),
    )


def test_full_sequence_of_equal_simulations_needs_no_mask():
    x = Input("uniform_x", dim=1, seq=-1, tw=1)
    model = Modely(
        "uniform_model",
        inputs=[x],
        outputs=[Output("uniform_out", x)],
    ).build()

    loader = DataLoader(
        model,
        source=[
            {"uniform_x": np.arange(4, dtype=np.float32)},
            {"uniform_x": np.arange(10, 14, dtype=np.float32)},
        ],
        seq_length="full",
    )

    assert loader._dataset["uniform_x"].shape == (2, 4, 1, 1)
    assert loader.mask is None


def test_full_sequence_rejects_step_and_inner_dynamic_sequences():
    x = Input("full_step_x", dim=1, seq=-1, tw=1)
    model = Modely(
        "full_step_model",
        inputs=[x],
        outputs=[Output("full_step_out", x)],
    ).build()

    ## The sequence already spans the simulation, so there is nothing to step over
    with np.testing.assert_raises(ValueError):
        DataLoader(
            model,
            source={"full_step_x": np.arange(6, dtype=np.float32)},
            seq_length="full",
            step=2,
        )

    inner = Input("full_inner_x", dim=1, seq=(2, -1), tw=1)
    inner_model = Modely(
        "full_inner_model",
        inputs=[inner],
        outputs=[Output("full_inner_out", inner)],
    ).build()

    ## 'full' resolves the outermost sequence only
    with np.testing.assert_raises(ValueError):
        DataLoader(
            inner_model,
            source={"full_inner_x": np.arange(6, dtype=np.float32)},
            seq_length="full",
        )


def test_uncollected_loop_warns_on_simulations_of_different_lengths():
    body_x = Input("warn_body_x", dim=1, tw=1)
    body_out = Output("warn_body_out", body_x)
    body = Modely("warn_body", inputs=[body_x], outputs=[body_out]).build()

    seed = Input("warn_x", dim=1, seq=-1, tw=1)
    loop = Loop(
        f=body,
        feedback={body_x: body_out},
        seq=4,
        collect=False,
        name="warn_loop",
        init={body_x: seed},
    )()
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


# ------------------------------------------------------------------
# get_samples: consecutive samples of one simulation
# ------------------------------------------------------------------

_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "datasets")
_SAMPLES_FORMAT = {"samples_x": "data_1", "samples_y": "data_3"}


def _samples_model():
    """pred = 2 * x[t] from a 3-sample window of x, against y[t + 1]."""
    from nnodely import Parameter

    x = Input("samples_x", tw=3)
    y = Input("samples_y", tw=(0, 1))
    window = x
    pred = Output("samples_pred", TimeSelect(-1)(x) * Parameter(value=[2.0]))
    model = Modely(
        "samples_model", inputs=[x, y], outputs=[pred, Output("samples_window", window)]
    )
    model.minimize("error", pred, y)
    return model.build()


def _raw(name):
    import pandas as pd

    return pd.read_csv(os.path.join(_SAMPLES_DIR, name))


def test_get_samples_returns_consecutive_windows_of_one_simulation():
    data = DataLoader(_samples_model(), source=_SAMPLES_DIR, format=_SAMPLES_FORMAT)
    raw = _raw("test2.csv")

    samples = data.get_samples(4, start=2, simulation="test2.csv")

    assert set(samples) == set(data.as_dict())
    assert samples["samples_x"].shape == (4, 3, 1)
    assert samples["samples_y"].shape == (4, 1, 1)
    for i in range(4):
        ## sample i ends on row start + i + 2; its next() is the row after
        end = 2 + i + 2
        np.testing.assert_array_equal(
            samples["samples_x"][i].ravel(), raw["data_1"][end - 2 : end + 1]
        )
        assert samples["samples_y"][i].ravel()[0] == raw["data_3"][end + 1]


def test_get_samples_picks_a_simulation_by_index_or_label():
    data = DataLoader(_samples_model(), source=_SAMPLES_DIR, format=_SAMPLES_FORMAT)

    by_index = data.get_samples(3, simulation=1)
    by_label = data.get_samples(3, simulation="test2.csv")
    first = data.get_samples(3)

    for name in by_index:
        np.testing.assert_array_equal(by_index[name], by_label[name])
    ## simulation 0 is the first file read, and starts the concatenated dataset
    for name, values in data.as_dict().items():
        np.testing.assert_array_equal(first[name], values[:3])


def test_get_samples_never_crosses_into_the_next_simulation():
    data = DataLoader(_samples_model(), source=_SAMPLES_DIR, format=_SAMPLES_FORMAT)
    ## 10 rows, a 3-sample past window and one future sample leave 7 per file
    assert len(data) == 14

    assert data.get_samples(7)["samples_x"].shape[0] == 7
    with pytest.raises(ValueError, match="has 7 samples"):
        data.get_samples(8)
    with pytest.raises(ValueError, match="has 7 samples"):
        data.get_samples(3, start=5)


def test_get_samples_rejects_invalid_requests():
    data = DataLoader(_samples_model(), source=_SAMPLES_DIR, format=_SAMPLES_FORMAT)

    with pytest.raises(ValueError, match="positive integer"):
        data.get_samples(0)
    with pytest.raises(ValueError, match="missing.csv"):
        data.get_samples(2, simulation="missing.csv")
    with pytest.raises(IndexError, match="out of range"):
        data.get_samples(2, simulation=2)
    with pytest.raises(ValueError, match="starting at -1"):
        data.get_samples(2, start=-1)


def test_get_samples_follows_the_loader_step():
    data = DataLoader(
        _samples_model(), source=_SAMPLES_DIR, format=_SAMPLES_FORMAT, step=2
    )
    raw = _raw("test.csv")

    samples = data.get_samples(3)

    ## step=2 keeps every other window: they end on rows 2, 4 and 6
    np.testing.assert_array_equal(
        samples["samples_x"][:, -1, 0], raw["data_1"][[2, 4, 6]]
    )


def test_inference_on_get_samples_is_a_time_series():
    model = _samples_model()
    data = DataLoader(model, source=_SAMPLES_DIR, format=_SAMPLES_FORMAT)
    raw = _raw("test.csv")

    prediction = to_numpy(model(data.get_samples(5, start=1))["samples_pred"])

    ## one prediction per sample, in time order: 2 * x on rows 3 .. 7
    assert prediction.shape == (5, 1, 1)
    np.testing.assert_allclose(
        prediction.ravel(), to_numpy(2.0 * raw["data_1"][3:8]), rtol=1e-6
    )


# ------------------------------------------------------------------
# CSV options go to pandas as they are
# ------------------------------------------------------------------


def _csv_model():
    x = Input("csv_x", tw=1)
    y = Input("csv_y", tw=1)
    return Modely(
        "csv_model", inputs=[x, y], outputs=[Output("csv_out", x + y)]
    ).build()


def _column_values(data, name):
    return data.as_dict()[name].ravel().tolist()


def test_csv_delimiter_is_the_pandas_separator(tmp_path):
    (tmp_path / "semicolon.csv").write_text(
        "csv_x;csv_y\n1;10\n2;20\n3;30\n", encoding="utf-8"
    )

    data = DataLoader(_csv_model(), source=tmp_path / "semicolon.csv", delimiter=";")

    assert _column_values(data, "csv_x") == [1.0, 2.0, 3.0]
    assert _column_values(data, "csv_y") == [10.0, 20.0, 30.0]


def test_csv_without_a_header_row_keeps_its_first_row(tmp_path):
    # Once read with the default header: the first row named the columns.
    (tmp_path / "bare.csv").write_text("1,10\n2,20\n3,30\n", encoding="utf-8")

    data = DataLoader(
        _csv_model(),
        source=tmp_path / "bare.csv",
        header=None,
        format={"csv_x": 0, "csv_y": 1},
    )

    assert _column_values(data, "csv_x") == [1.0, 2.0, 3.0]
    assert _column_values(data, "csv_y") == [10.0, 20.0, 30.0]


def test_csv_header_can_follow_a_title_line(tmp_path):
    (tmp_path / "titled.csv").write_text(
        "bench run 3\ncsv_x,csv_y\n1,10\n2,20\n", encoding="utf-8"
    )

    data = DataLoader(_csv_model(), source=tmp_path / "titled.csv", header=1)

    assert _column_values(data, "csv_x") == [1.0, 2.0]
    assert _column_values(data, "csv_y") == [10.0, 20.0]


def test_na_cells_are_kept_as_nan_with_a_warning(tmp_path):
    (tmp_path / "holes.csv").write_text(
        "csv_x,csv_y\n1,10\nn/a,20\n3,\n", encoding="utf-8"
    )

    with pytest.warns(UserWarning) as caught:
        data = DataLoader(_csv_model(), source=tmp_path / "holes.csv")

    messages = {str(warning.message) for warning in caught}
    assert "'holes.csv': input 'csv_x' has 1 n/a values, kept as NaN." in messages
    assert "'holes.csv': input 'csv_y' has 1 n/a values, kept as NaN." in messages
    assert np.isnan(_column_values(data, "csv_x")[1])
    assert np.isnan(_column_values(data, "csv_y")[2])


# ------------------------------------------------------------------
# windowed=True: data already laid out as the model's tensors
# ------------------------------------------------------------------


def _window_model():
    x = Input("win_x", tw=10)
    return Modely("win_model", inputs=[x], outputs=[Output("win_out", x)]).build()


def test_windowed_data_is_used_as_given():
    windows = np.random.rand(100, 10, 1).astype(np.float32)  # (B, T, D)

    loader = DataLoader(_window_model(), source={"win_x": windows}, windowed=True)

    ## No window is built and no sample is dropped
    assert len(loader) == 100
    np.testing.assert_array_equal(loader.as_dict()["win_x"], windows)

    ## A single feature may leave its axis out, and nested lists are arrays
    for given in (windows[..., 0], windows.tolist()):
        same = DataLoader(_window_model(), source={"win_x": given}, windowed=True)
        np.testing.assert_array_equal(same.as_dict()["win_x"], windows)

    ## The dataset is a copy: the caller's array changing later does not reach it
    windows[0] = -1.0
    assert (loader.as_dict()["win_x"][0] != -1.0).all()


def test_windowed_data_round_trips_what_the_loader_builds():
    x = Input("trip_x", dim=2, tw=5)
    y = Input("trip_y", tw=(2, 3))
    s = Input("trip_s", seq=3, tw=2)
    model = Modely(
        "trip_model",
        inputs=[x, y, s],
        outputs=[Output("trip_out", TimeSelect(-1)(x) + TimeSelect(-1)(y))],
    )
    model.minimize("trip", TimeSelect(-1)(s), TimeSelect(-1)(y))
    model.build()
    signals = {
        "trip_x": np.random.rand(40, 2),
        "trip_y": np.random.rand(40),
        "trip_s": np.random.rand(40),
    }

    built = DataLoader(model, source=signals).as_dict()
    given = DataLoader(model, source=built, windowed=True).as_dict()

    assert given.keys() == built.keys()
    for name in built:
        np.testing.assert_array_equal(given[name], built[name])


def test_windowed_target_is_not_shifted_again():
    x = Input("next_x", tw=1)
    y = Input("next_y", tw=(0, 1))  # the next sample
    model = Modely("next_model", inputs=[x], outputs=[Output("next_out", x)])
    model.minimize("next", "next_out", y)
    model.build()

    ## The pairs are already made: y[i] is the sample after x[i]
    xs = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
    ys = xs + 1

    ## Raw data is aligned by the loader, which reads y one sample ahead...
    raw = DataLoader(model, source={"next_x": xs, "next_y": ys})
    np.testing.assert_array_equal(raw.as_dict()["next_y"] - raw.as_dict()["next_x"], 2)

    ## ...windowed data keeps the pairs it is given
    windowed = DataLoader(
        model, source={"next_x": xs, "next_y": ys}, windowed=True
    ).as_dict()
    np.testing.assert_array_equal(windowed["next_y"] - windowed["next_x"], 1)


def test_windowed_data_reads_several_dynamic_lengths_from_its_shape():
    from nnodely import Parameter

    x = Input("dyn2_x", seq=(-1, -1), tw=1)
    y = Input("dyn2_y", seq=(-1, -1), tw=1)
    gain = Parameter("dyn2_gain", value=1.0)
    model = Modely("dyn2_model", inputs=[x], outputs=[Output("dyn2_out", x * gain)])
    model.minimize("dyn2", "dyn2_out", y)
    model.build()

    xs = np.random.rand(16, 4, 6, 1, 1).astype(np.float32)

    ## Building the windows resolves a single dynamic length
    with pytest.raises(ValueError, match="windowed=True"):
        DataLoader(model, source={"dyn2_x": xs[:, 0, 0, 0], "dyn2_y": xs[:, 0, 0, 0]})

    loader = DataLoader(model, source={"dyn2_x": xs, "dyn2_y": 3 * xs}, windowed=True)
    assert loader.as_dict()["dyn2_x"].shape == (16, 4, 6, 1, 1)
    assert loader.mask is None

    model.train(loader, epochs=200, batch_size=16, lr=5e-2, printer=None, **CONVERGED)
    np.testing.assert_allclose(np.ravel(gain.value_numpy), [3.0], atol=1e-2)  # type: ignore


def test_windowed_rollouts_of_different_lengths_are_padded_and_masked():
    x = Input("roll_x", seq=-1, tw=1)
    model = Modely("roll_model", inputs=[x], outputs=[Output("roll_out", x)]).build()
    long = np.arange(15, dtype=np.float32).reshape(3, 5, 1, 1)
    short = np.arange(100, 106, dtype=np.float32).reshape(2, 3, 1, 1)

    loader = DataLoader(
        model, source=[{"roll_x": long}, {"roll_x": short}], windowed=True
    )

    ## The shorter rollout repeats its last step up to the longest one
    assert loader.as_dict()["roll_x"].shape == (5, 5, 1, 1)
    np.testing.assert_array_equal(
        loader.as_dict()["roll_x"][3:, :, 0, 0],
        [[100, 101, 102, 102, 102], [103, 104, 105, 105, 105]],
    )

    ## One mask row per sample, each with the rollout of its simulation
    np.testing.assert_array_equal(
        loader.mask,
        [[1, 1, 1, 1, 1]] * 3 + [[1, 1, 1, 0, 0]] * 2,
    )

    ## The simulations are kept apart
    np.testing.assert_array_equal(
        loader.get_samples(2, simulation=1)["roll_x"][:, :3, 0, 0],
        short[:, :, 0, 0],
    )


def test_windowed_lengths_that_cannot_be_padded_are_rejected():
    x = Input("pad_x", seq=(-1, -1), tw=1)
    u = Input("pad_u", seq=-1, tw=1)
    model = Modely(
        "pad_model",
        inputs=[x, u],
        outputs=[Output("pad_out", x), Output("pad_u_out", u)],
    ).build()

    def simulation(rollout, inner, rollout_u=None):
        return {
            "pad_x": np.zeros((2, rollout, inner, 1, 1)),
            "pad_u": np.zeros((2, rollout if rollout_u is None else rollout_u, 1, 1)),
        }

    ## Only the rollout is padded, so an inner dynamic length must match
    with pytest.raises(ValueError, match="only the outermost dynamic"):
        DataLoader(model, source=[simulation(5, 2), simulation(3, 4)], windowed=True)

    ## Every input whose outermost axis is dynamic shares one rollout
    with pytest.raises(ValueError, match="share one rollout"):
        DataLoader(model, source=simulation(5, 2, rollout_u=4), windowed=True)

    ## The rollout itself may differ
    loader = DataLoader(
        model, source=[simulation(5, 2), simulation(3, 2)], windowed=True
    )
    assert loader.as_dict()["pad_x"].shape == (4, 5, 2, 1, 1)
    assert loader.mask is not None and loader.mask.shape == (4, 5)


def test_windowed_modelyloop_matches_the_windows_it_would_build():
    from nnodely import IntegrateStep, ModelyLoop, Parameter

    force, x, v = Input("wl_F", tw=1), Input("wl_x", tw=1), Input("wl_v", tw=1)
    k = Parameter("wl_k", value=1.0)
    v_next = v.feedback(IntegrateStep(dt=0.1)(v, force - k * x))
    x_next = x.feedback(IntegrateStep(dt=0.1)(x, v_next))
    model = ModelyLoop(
        "wl_loop", inputs=[force, x, v], outputs=[Output("wl_x_next", x_next)]
    )
    model.minimize("wl", "wl_x_next", Input("wl_x_meas", tw=1))
    model.build()
    signals = {
        name: np.random.rand(60) for name in ("wl_F", "wl_x", "wl_v", "wl_x_meas")
    }

    built = DataLoader(model, source=signals, seq_length=8, step=4).as_dict()
    given = DataLoader(model, source=built, windowed=True).as_dict()
    for name in built:
        np.testing.assert_array_equal(given[name], built[name])

    ## An optional input left out reads zeros over the shared rollout
    without_v = {name: values for name, values in built.items() if name != "wl_v"}
    given = DataLoader(model, source=without_v, windowed=True).as_dict()
    assert given["wl_v"].shape == built["wl_v"].shape
    assert not given["wl_v"].any()


def test_windowed_rejects_a_wrong_layout_and_options_it_does_not_use():
    model = _window_model()
    windows = np.random.rand(100, 10, 1)

    with pytest.raises(ValueError, match=r"laid out as its tensor, \(N, 10, 1\)"):
        DataLoader(model, source={"win_x": windows[:, :9]}, windowed=True)
    with pytest.raises(ValueError, match="cannot be read as one array"):
        DataLoader(model, source={"win_x": [[1.0] * 10, [1.0] * 9]}, windowed=True)
    with pytest.raises(TypeError, match="windowed=True reads a dict"):
        DataLoader(model, source=_SAMPLES_DIR, windowed=True)
    with pytest.raises(ValueError, match="seq_length"):
        DataLoader(model, source={"win_x": windows}, windowed=True, seq_length=3)
    with pytest.raises(ValueError, match="step"):
        DataLoader(model, source={"win_x": windows}, windowed=True, step=2)
    with pytest.raises(TypeError, match="windowed is True or False"):
        DataLoader(model, source={"win_x": windows}, windowed="yes")  # type: ignore[arg-type]


def test_windowed_inputs_hold_the_same_number_of_samples():
    a, b = Input("count_a", tw=2), Input("count_b", tw=2)
    model = Modely("count_model", inputs=[a, b], outputs=[Output("count_out", a + b)])
    model.build()

    with pytest.raises(ValueError, match="different numbers of samples"):
        DataLoader(
            model,
            source={"count_a": np.zeros((10, 2, 1)), "count_b": np.zeros((9, 2, 1))},
            windowed=True,
        )


def test_raw_data_laid_out_as_windows_points_to_windowed():
    with pytest.raises(ValueError, match="pass windowed=True"):
        DataLoader(_window_model(), source={"win_x": np.random.rand(100, 10, 1)})
