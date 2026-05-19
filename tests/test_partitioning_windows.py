# from nnodely import Input, Output, Modely, DataLoader

# import os


# def test_partitioning_windows():
#     # ------- Model with time window partitioning, Past and Future -------
#     # x = Input("x", dim=1, sampling=0.1) ## x is sampled at 0.1Hz, so 1 sample every 10 seconds
#     x = Input("x", dim=1)
#     y = Input("y", dim=1)
#     z = Input("z", dim=1)

#     x_stream1 = x.sw(3)  ## 3 samples of x in the past
#     y_stream1 = y.sw([0, 3])  ## 3 samples of y in the future
#     z_stream1 = z.sw(
#         [4, 2]
#     )  ## 4 samples of z in the past and 2 samples of z in the future

#     x_stream2 = x.last()  ## 1 sample of x in the past
#     y_stream2 = y.next()  ## 1 sample of y in the future
#     z_stream2 = z.sw(3)  ## 1 sample of z in the past (same as last())

#     x_stream3 = x.sw(5)  ## 5 samples of x in the past
#     y_stream3 = y.sw(
#         [2, 2]
#     )  ## 2 samples of y in the past and 2 samples of y in the future
#     z_stream3 = z.sw(
#         [1, 3]
#     )  ## 1 sample of z in the past and 3 samples of z in the future

#     add1 = x_stream1 + y_stream1 + z_stream2
#     add2 = x_stream2 + y_stream2
#     add3 = y_stream3 + z_stream3

#     out1 = Output("out1", add1)
#     out2 = Output("out2", add2)
#     out3 = Output("out3", add3)
#     out_x_stream1 = Output("out_x_stream1", x_stream1)
#     out_y_stream1 = Output("out_y_stream1", y_stream1)
#     out_z_stream1 = Output("out_z_stream1", z_stream1)
#     out_x_stream2 = Output("out_x_stream2", x_stream2)
#     out_y_stream2 = Output("out_y_stream2", y_stream2)
#     out_z_stream2 = Output("out_z_stream2", z_stream2)
#     out_x_stream3 = Output("out_x_stream3", x_stream3)
#     out_y_stream3 = Output("out_y_stream3", y_stream3)
#     out_z_stream3 = Output("out_z_stream3", z_stream3)
#     model = Modely(
#         "model1",
#         inputs=[x, y, z],
#         outputs=[
#             out1,
#             out2,
#             out3,
#             out_x_stream1,
#             out_y_stream1,
#             out_z_stream1,
#             out_x_stream2,
#             out_y_stream2,
#             out_z_stream2,
#             out_x_stream3,
#             out_y_stream3,
#             out_z_stream3,
#         ],
#     )
#     model.build()

#     ## ------ Load dataset -------
#     data_train = DataLoader(
#         model,
#         format={"x": "data_1", "y": "data_2", "z": "data_3"},
#         source=os.path.join("tests", "data"),
#     )

#     assert data_train.dataset is not None
#     assert data_train.inputs == ["x", "y", "z"]
#     assert data_train.input_specs == {"x": [5, 0], "y": [2, 3], "z": [4, 3]}
#     assert data_train.dataset["x"].shape == (6, 5)
#     assert data_train.dataset["y"].shape == (6, 5)
#     assert data_train.dataset["z"].shape == (6, 7)
#     assert data_train.dataset["x"].tolist() == [
#         [1, 2, 3, 4, 5],
#         [2, 3, 4, 5, 6],
#         [3, 4, 5, 6, 7],
#         [11.0, 12.0, 13.0, 14.0, 15.0],
#         [12.0, 13.0, 14.0, 15.0, 16.0],
#         [13.0, 14.0, 15.0, 16.0, 17.0],
#     ]
#     assert data_train.dataset["y"].tolist() == [
#         [10.0, 15.0, 21.0, 28.0, 36.0],
#         [15.0, 21.0, 28.0, 36.0, 45.0],
#         [21.0, 28.0, 36.0, 45.0, 55.0],
#         [105.0, 120.0, 136.0, 153.0, 171.0],
#         [120.0, 136.0, 153.0, 171.0, 190.0],
#         [136.0, 153.0, 171.0, 190.0, 210.0],
#     ]
#     assert data_train.dataset["z"].tolist() == [
#         [5.0, 9.0, 14.0, 20.0, 26.0, 33.0, 41.0],
#         [9.0, 14.0, 20.0, 26.0, 33.0, 41.0, 50.0],
#         [14.0, 20.0, 26.0, 33.0, 41.0, 50.0, 60.0],
#         [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
#         [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
#         [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
#     ]

#     result = model([data_train[0]])
#     assert result["out1"].cpu().numpy().tolist() == [[[33], [46], [61]]]
#     assert result["out2"].cpu().numpy().tolist() == [[[26]]]
#     assert result["out3"].cpu().numpy().tolist() == [[[30], [41], [54], [69]]]
#     assert result["out_x_stream1"].cpu().numpy().tolist() == [[[3], [4], [5]]]
#     assert result["out_x_stream2"].cpu().numpy().tolist() == [[[5]]]
#     assert result["out_x_stream3"].cpu().numpy().tolist() == [[[1], [2], [3], [4], [5]]]
#     assert result["out_y_stream1"].cpu().numpy().tolist() == [[[21], [28], [36]]]
#     assert result["out_y_stream2"].cpu().numpy().tolist() == [[[21]]]
#     assert result["out_y_stream3"].cpu().numpy().tolist() == [[[10], [15], [21], [28]]]
#     assert result["out_z_stream1"].cpu().numpy().tolist() == [
#         [[5], [9], [14], [20], [26], [33]]
#     ]
#     assert result["out_z_stream2"].cpu().numpy().tolist() == [[[9], [14], [20]]]
#     assert result["out_z_stream3"].cpu().numpy().tolist() == [[[20], [26], [33], [41]]]

#     result = model([data_train[1]])
#     assert result["out1"].cpu().numpy().tolist() == [[[46], [61], [77]]]
#     assert result["out2"].cpu().numpy().tolist() == [[[34]]]
#     assert result["out3"].cpu().numpy().tolist() == [[[41], [54], [69], [86]]]
#     assert result["out_x_stream1"].cpu().numpy().tolist() == [[[4], [5], [6]]]
#     assert result["out_x_stream2"].cpu().numpy().tolist() == [[[6]]]
#     assert result["out_x_stream3"].cpu().numpy().tolist() == [[[2], [3], [4], [5], [6]]]
#     assert result["out_y_stream1"].cpu().numpy().tolist() == [[[28], [36], [45]]]
#     assert result["out_y_stream2"].cpu().numpy().tolist() == [[[28]]]
#     assert result["out_y_stream3"].cpu().numpy().tolist() == [[[15], [21], [28], [36]]]
#     assert result["out_z_stream1"].cpu().numpy().tolist() == [
#         [[9], [14], [20], [26], [33], [41]]
#     ]
#     assert result["out_z_stream2"].cpu().numpy().tolist() == [[[14], [20], [26]]]
#     assert result["out_z_stream3"].cpu().numpy().tolist() == [[[26], [33], [41], [50]]]
