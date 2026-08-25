import os

from nnodely import Input, Output, Fir, Modely, DataLoader, Parameter, Constant, Linear

import pytest
import numpy as np
import keras
from conftest import to_numpy
from nnodely.utils.utils import _resolve_loss, _resolve_optimizer


@keras.saving.register_keras_serializable(package="nnodely_test")
class SimpleCustomOptimizer(keras.optimizers.Optimizer):
    """Minimal SGD-like optimizer used to verify custom optimizer support."""

    def update_step(self, gradient, variable, learning_rate):
        gradient = keras.ops.cast(gradient, variable.dtype)
        learning_rate = keras.ops.cast(learning_rate, variable.dtype)
        self.assign_sub(variable, learning_rate * gradient)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="nnodely_test")
def custom_quartic_loss(y_true, y_pred):
    """Fourth-power error used to verify custom callable loss support."""
    error = y_pred - y_true
    return keras.ops.mean(keras.ops.square(keras.ops.square(error)), axis=-1)


@pytest.mark.slow
def test_train_basic():
    # ------- Model definition and training -------
    x = Input("x", dim=1)
    y = Input("y", dim=1)
    x_fir = Fir(out_features=1)([x.sw(5)])
    y_fir = Fir(out_features=1)([y.sw(5)])

    x_out = Output("fir_pred", x_fir + y_fir)
    model1 = Modely("linear_fit", inputs=[x, y], outputs=[x_out])

    # ------- Define loss and minimizer -------
    model1.minimize(
        "error", source=x_out, target=Input("x_target", dim=1).sw(1), loss="mse"
    )
    model1.build()

    dummy = {
        "x": np.ones((1, 1, 5), dtype=np.float32),
        "y": np.ones((1, 1, 5), dtype=np.float32),
        "x_target": np.ones((1, 1, 1), dtype=np.float32),
    }
    result = model1(dummy)
    assert result["fir_pred"].shape == (1, 1, 1)  # Check if the output shape is correct

    # ------ Load dataset -------
    data_train = DataLoader(
        model1,
        format={"x": "data_1", "y": "data_2", "x_target": "data_3"},
        source=os.path.join("tests", "datasets"),
    )

    # ------ Train the model -------
    model1.train(train_data=data_train, epochs=60, batch_size=4)

    dummy = {
        "x": np.ones((1, 1, 5), dtype=np.float32),
        "y": np.ones((1, 1, 5), dtype=np.float32),
        "x_target": np.ones((1, 1, 1), dtype=np.float32),
    }
    result_trained = model1(dummy)
    assert result_trained["fir_pred"].shape == (
        1,
        1,
        1,
    )  # Check if the output shape is correct after training
    assert (
        result["fir_pred"] != result_trained["fir_pred"]
    )  # Check if the model output has changed after training, indicating that training had an effect

    # ------ Remove minimizer and retrain with multi loss -------
    model1.remove_minimizer("error")
    model1.minimize(
        "error_fir_x",
        source=x_fir,
        target=Input("x_target", dim=1).sw(1),
        loss="mse",
    )
    model1.minimize(
        "error_fir_y", source=y_fir, target=None, loss="mse"
    )  ## with target=None, the loss will be minimized to zero

    model1.build()  # Rebuild the model after removing the minimizer otherwise the training loop will still try to compute the loss and update the model based on it, even if it's not used for training anymore
    model1.train(train_data=data_train, epochs=60, batch_size=4)

    # ------ Remove one minimizer and retrain with a constant value -------
    model1.remove_minimizer("error_fir_y")
    model1.minimize(
        "error_fir_y", source=y_fir, target=3.0, loss="mse"
    )  ## this will minimize the difference between y_fir and the constant value 3.0, effectively training the model to make y_fir close to 3.0
    model1.build()
    model1.train(train_data=data_train, epochs=60, batch_size=4)


@pytest.mark.slow
def test_train_with_parameters():
    # ------- Model definition with Parameters and Constants -------
    x = Input("x", dim=1)
    param = Parameter("param1", value=[1.0])
    const = Constant("const1", value=[1.0])
    x_param = x.sw(1) * param + const
    x_out = Output("x_out", x_param)
    model = Modely("model", inputs=[x], outputs=[x_out])
    model.minimize(
        "error", source=x_out, target=Input("x_target", dim=1).sw(1), loss="mse"
    )
    model.build()

    dummy_input_x = np.ones((1, 1, 1), dtype=np.float32)

    # ------ Create a simple dataset and train the model -------
    true_param = np.array([3.5])  # The true parameter value we want to learn
    dataframe = {
        "x": np.ones((100, 1, 1), dtype=np.float32),
        "x_target": np.ones((100, 1, 1), dtype=np.float32) * true_param,
    }
    data_train = DataLoader(model, source=dataframe)

    model.train(train_data=data_train, epochs=100, batch_size=16, lr=0.01)
    assert np.isclose(
        a=np.array(param.value_numpy),
        b=np.array(true_param - const.value_numpy),
        atol=0.01,
    )
    assert np.isclose(const.value_numpy, np.array([1.0]), atol=0.01)  # type: ignore

    # ------ Inference after training -------
    result_after_training = model(
        {
            "x": dummy_input_x,
            "x_target": np.ones((1, 1, 1), dtype=np.float32) * true_param,
        }
    )
    np.testing.assert_allclose(
        to_numpy(result_after_training["x_out"]),
        np.ones((1, 1, 1), dtype=np.float32) * true_param,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.slow
def test_training_values_linear():
    input1 = Input("in1")
    target = Input("out1").last()
    linear_out = Linear(initializer="ones", bias_initializer="ones")(input1.last())
    output1 = Output("out", linear_out)

    model = Modely("test_model", inputs=[input1], outputs=[output1])
    model.minimize("error", source=output1, target=target, loss="mse")
    model.build()

    dataset = {"in1": [1], "out1": [3]}
    data_train = DataLoader(model, source=dataset)
    model.train(
        train_data=data_train,
        epochs=1,
        batch_size=1,
        optimizer="sgd",
        lr=1.0,
    )
    assert linear_out.kernel is not None
    assert linear_out.bias is not None

    np.testing.assert_allclose(
        to_numpy(linear_out.kernel), [[3.0]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(to_numpy(linear_out.bias), [3.0], rtol=1e-5, atol=1e-5)

    model.build()
    model.train(
        train_data=data_train,
        epochs=1,
        batch_size=1,
        optimizer="adam",
        lr=1.0,
    )

    np.testing.assert_allclose(
        to_numpy(linear_out.kernel), [[2.0]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(to_numpy(linear_out.bias), [2.0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "optimizer_name",
    [
        "sgd",
        "rmsprop",
        "adam",
        "adamw",
        "adadelta",
        "adagrad",
        "adamax",
        "adafactor",
        "nadam",
        "ftrl",
        "lion",
    ],
)
def test_resolve_builtin_optimizer(optimizer_name):
    optimizer = _resolve_optimizer(
        optimizer_name,
        learning_rate=0.123,
        optimizer_kwargs=None,
    )

    assert isinstance(optimizer, keras.optimizers.Optimizer)
    np.testing.assert_allclose(to_numpy(optimizer.learning_rate), 0.123)


def test_resolve_optimizer_instance_and_config():
    optimizer_with_kwargs = _resolve_optimizer(
        "sgd",
        learning_rate=0.1,
        optimizer_kwargs={"momentum": 0.9, "nesterov": True},
    )
    assert isinstance(optimizer_with_kwargs, keras.optimizers.SGD)
    assert optimizer_with_kwargs.momentum == 0.9
    assert optimizer_with_kwargs.nesterov is True

    optimizer_instance = keras.optimizers.SGD(learning_rate=0.25, momentum=0.5)
    assert _resolve_optimizer(optimizer_instance, 1.0, None) is optimizer_instance

    config = keras.optimizers.serialize(optimizer_instance)
    assert type(config) is dict
    optimizer_from_config = _resolve_optimizer(config, 1.0, None)
    assert isinstance(optimizer_from_config, keras.optimizers.SGD)
    np.testing.assert_allclose(to_numpy(optimizer_from_config.learning_rate), 0.25)
    assert optimizer_from_config.momentum == 0.5


def test_train_with_custom_optimizer():
    input_node = Input("custom_optimizer_input")
    target = Input("custom_optimizer_target").last()
    linear = Linear(initializer="ones", bias_initializer="ones")(input_node.last())

    model = Modely(
        "custom_optimizer_model",
        inputs=[input_node],
        outputs=[Output("custom_optimizer_output", linear)],
    )
    model.minimize("error", source=linear, target=target, loss="mse")
    model.build()

    data = DataLoader(
        model,
        source={"custom_optimizer_input": [1], "custom_optimizer_target": [3]},
    )
    optimizer = SimpleCustomOptimizer(learning_rate=0.25)
    model.train(
        train_data=data,
        epochs=1,
        batch_size=1,
        optimizer=optimizer,
    )

    assert model.model is not None
    assert model.model.optimizer is optimizer
    assert linear.kernel is not None
    assert linear.bias is not None
    # Initial prediction is 1 * 1 + 1 = 2. MSE gives a gradient of -2 for
    # both variables, so one custom update adds 0.25 * 2 = 0.5.
    np.testing.assert_allclose(to_numpy(linear.kernel), [[1.5]], atol=1e-5)
    np.testing.assert_allclose(to_numpy(linear.bias), [1.5], atol=1e-5)


@pytest.mark.parametrize(
    "loss_name",
    [
        "binary_crossentropy",
        "binary_focal_crossentropy",
        "categorical_crossentropy",
        "categorical_focal_crossentropy",
        "sparse_categorical_crossentropy",
        "poisson",
        "ctc",
        "kl_divergence",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_logarithmic_error",
        "cosine_similarity",
        "huber",
        "log_cosh",
        "tversky",
        "dice",
        "hinge",
        "squared_hinge",
        "categorical_hinge",
        "circle",
    ],
)
def test_resolve_builtin_loss(loss_name):
    assert callable(_resolve_loss(loss_name))


def test_resolve_loss_instance_and_config():
    loss_instance = keras.losses.MeanSquaredError(reduction="sum")
    assert _resolve_loss(loss_instance) is loss_instance

    config = keras.losses.serialize(loss_instance)
    assert type(config) is dict
    loss_from_config = _resolve_loss(config)

    assert isinstance(loss_from_config, keras.losses.MeanSquaredError)
    np.testing.assert_allclose(
        to_numpy(
            loss_from_config(
                keras.ops.array([[0.0], [0.0]]),
                keras.ops.array([[1.0], [1.0]]),
            )
        ),
        2.0,
    )


@pytest.mark.parametrize("loss_name", ["mse", "mae", "huber", "log_cosh"])
def test_train_with_named_loss(loss_name):
    input_node = Input(f"{loss_name}_input")
    target = Input(f"{loss_name}_target").last()
    linear = Linear(initializer="ones", bias_initializer="zeros")(input_node.last())
    output = Output(f"{loss_name}_output", linear)
    model = Modely(f"{loss_name}_model", inputs=[input_node], outputs=[output])
    model.minimize(f"{loss_name}_error", output, target, loss=loss_name)
    model.build()

    data = DataLoader(
        model,
        source={f"{loss_name}_input": [1], f"{loss_name}_target": [3]},
    )
    history = model.train(
        train_data=data,
        epochs=1,
        batch_size=1,
        optimizer="sgd",
        lr=0.1,
    )

    assert np.isfinite(history["loss"][-1])
    assert linear.kernel is not None
    assert not np.allclose(to_numpy(linear.kernel), [[1.0]])


def test_train_with_multiple_minimizer_losses():
    input_node = Input("multi_loss_input")
    mse_target = Input("mse_target").last()
    mae_target = Input("mae_target").last()
    mse_linear = Linear(initializer="ones", bias_initializer="zeros")(input_node.last())
    mae_linear = Linear(initializer="ones", bias_initializer="zeros")(input_node.last())
    mse_output = Output("mse_output", mse_linear)
    mae_output = Output("mae_output", mae_linear)

    model = Modely(
        "multiple_loss_model",
        inputs=[input_node],
        outputs=[mse_output, mae_output],
    )
    model.minimize("mse_error", mse_output, mse_target, loss="mse")
    model.minimize("mae_error", mae_output, mae_target, loss="mae")
    model.build()

    data = DataLoader(
        model,
        source={"multi_loss_input": [1], "mse_target": [3], "mae_target": [2]},
    )
    history = model.train(
        train_data=data,
        epochs=1,
        batch_size=1,
        optimizer="sgd",
        lr=0.1,
    )

    assert np.isfinite(history["loss"][-1])
    assert mse_linear.kernel is not None
    assert mae_linear.kernel is not None
    assert not np.allclose(to_numpy(mse_linear.kernel), [[1.0]])
    assert not np.allclose(to_numpy(mae_linear.kernel), [[1.0]])


def test_train_with_custom_loss_function():
    input_node = Input("custom_loss_input")
    target = Input("custom_loss_target").last()
    linear = Linear(initializer="ones", bias_initializer="ones")(input_node.last())
    output = Output("custom_loss_output", linear)
    model = Modely("custom_loss_model", inputs=[input_node], outputs=[output])
    model.minimize("custom_error", output, target, loss=custom_quartic_loss)
    model.build()

    data = DataLoader(
        model,
        source={"custom_loss_input": [1], "custom_loss_target": [3]},
    )
    model.train(
        train_data=data,
        epochs=1,
        batch_size=1,
        optimizer="sgd",
        lr=0.1,
    )

    assert linear.kernel is not None
    assert linear.bias is not None
    # Initial prediction is 2 and d((prediction - 3)^4)/d prediction is -4.
    # With x=1 and SGD(lr=0.1), both variables increase by 0.4.
    np.testing.assert_allclose(to_numpy(linear.kernel), [[1.4]], atol=1e-5)
    np.testing.assert_allclose(to_numpy(linear.bias), [1.4], atol=1e-5)
