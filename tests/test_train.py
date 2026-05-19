# from nnodely import Input, Output, Fir, Modely, DataLoader, Parameter, Constant
# import numpy as np

# import pytest
# import os


# @pytest.mark.slow
# def test_train_basic():
#     # ------- Model definition and training -------
#     x = Input("x", dim=1)
#     y = Input("y", dim=1)
#     x_fir = Fir(out_features=1)([x.sw(5)])
#     y_fir = Fir(out_features=1)([y.sw(5)])

#     x_out = Output("fir_pred", x_fir + y_fir)
#     model1 = Modely("linear_fit", inputs=[x, y], outputs=[x_out])

#     # ------- Define loss and minimizer -------
#     model1.minimize(
#         "error", source=x_out, target=Input("x_target", dim=1).sw(1), loss="mse"
#     )
#     model1.build()

#     # ------ Load dataset -------
#     data_train = DataLoader(
#         model1,
#         format={"x": "data_1", "y": "data_2", "x_target": "data_3"},
#         source=os.path.join("tests", "data"),
#     )

#     # ------ Train the model -------
#     model1.train(train_data=data_train, epochs=60, batch_size=4)

#     # ------ Remove minimizer and retrain with multi loss -------
#     model1.remove_minimizer("error")
#     model1.minimize(
#         "error_fir_x",
#         source=x_fir,
#         target=Input("x_target", dim=1).sw(1),
#         loss="mse",
#     )
#     model1.minimize(
#         "error_fir_y", source=y_fir, target=None, loss="mse"
#     )  ## with target=None, the loss will be minimized to zero

#     model1.build()  # Rebuild the model after removing the minimizer otherwise the training loop will still try to compute the loss and update the model based on it, even if it's not used for training anymore
#     model1.train(train_data=data_train, epochs=60, batch_size=4)

#     # ------ Remove one minimizer and retrain with a constant value -------
#     model1.remove_minimizer("error_fir_y")
#     model1.minimize(
#         "error_fir_y", source=y_fir, target=3.0, loss="mse"
#     )  ## this will minimize the difference between y_fir and the constant value 3.0, effectively training the model to make y_fir close to 3.0
#     model1.build()
#     model1.train(train_data=data_train, epochs=60, batch_size=4)


# @pytest.mark.slow
# def test_train_with_parameters():
#     # ------- Model definition with Parameters and Constants -------
#     x = Input("x", dim=1)
#     param = Parameter("param1", value=[1.0])
#     const = Constant("const1", value=[1.0])
#     x_param = x.sw(1) * param + const
#     x_out = Output("x_out", x_param)
#     model = Modely("model", inputs=[x], outputs=[x_out])
#     model.minimize(
#         "error", source=x_out, target=Input("x_target", dim=1).sw(1), loss="mse"
#     )
#     model.build()

#     dummy_input_x = np.ones((4, 1, 1), dtype=np.float32)

#     # ------ Create a simple dataset and train the model -------
#     true_param = 3.5  # The true parameter value we want to learn
#     dataframe = {
#         "x": np.ones((100, 1, 1), dtype=np.float32),
#         "x_target": np.ones((100, 1, 1), dtype=np.float32) * true_param,
#     }
#     data_train = DataLoader(model, source=dataframe)

#     model.train(train_data=data_train, epochs=50, batch_size=16)
#     assert np.isclose(
#         a=np.array(param.value_numpy),
#         b=np.array(true_param - const.value_numpy),
#         atol=0.1,
#     )
#     assert np.isclose(const.value_numpy, 1.0, atol=0.1)

#     # ------ Inference after training -------
#     result_after_training = model({"x": dummy_input_x})
#     assert np.isclose(result_after_training["x_out"].mean(), true_param, atol=0.1)
