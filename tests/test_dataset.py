from nnodely import Input, Output, Modely, DataLoader

import os
import numpy as np


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
    z_stream2 = z.sw(3)  ## 1 sample of z in the past (same as last())

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
