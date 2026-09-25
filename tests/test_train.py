import os

from nnodely import (
    Input,
    Output,
    Fir,
    Modely,
    DataLoader,
    Parameter,
    Constant,
    EquationLearner,
    Linear,
    Loop,
    Roll,
)

import pytest
import numpy as np
import keras
from conftest import to_numpy
from nnodely.utils.utils import MaskedLoss, _resolve_loss, _resolve_optimizer


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

    model1.validate(data_train)

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

    model1.build()  # Rebuild the model after removing the minimizer otherwise the training Roll will still try to compute the loss and update the model based on it, even if it's not used for training anymore
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


def test_training_values_fir_linear():
    input1 = Input("in1")
    target = Input("target1").last()

    fir_out = Fir(out_features=1, use_bias=False)(input1.last())
    linear_out = Linear(out_features=1, initializer="ones", bias_initializer="ones")(
        fir_out
    )

    output1 = Output("out1", fir_out)
    output2 = Output("out2", linear_out)

    model = Modely("test_model", inputs=[input1], outputs=[output1, output2])
    model.minimize("error", source=output2, target=target, loss="mse")
    model.build()

    assert fir_out.kernel is not None
    assert linear_out.kernel is not None
    assert linear_out.bias is not None

    def reset_weights():
        fir_out.kernel.assign([[1.0]])
        linear_out.kernel.assign([[1.0]])
        linear_out.bias.assign([1.0])

    def assert_weights(fir_kernel, kernel, bias):
        np.testing.assert_allclose(
            to_numpy(fir_out.kernel), fir_kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            to_numpy(linear_out.kernel), kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            to_numpy(linear_out.bias), bias, rtol=1e-5, atol=1e-5
        )

    reset_weights()
    result = model(
        {
            "in1": np.ones((1, 1, 1), dtype=np.float32),
            "target1": np.ones((1, 1, 1), dtype=np.float32) * 3,
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["out1"]), [[[1.0]]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(result["out2"]), [[[2.0]]], rtol=1e-5, atol=1e-5
    )

    # ------- One sample, one epoch at a time -------
    dataset = {"in1": [1], "target1": [3]}
    data_train = DataLoader(model, source=dataset)

    assert_weights([[1.0]], [[1.0]], [1.0])
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[3.0]], [[3.0]], [3.0])
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[-51.0]], [[-51.0]], [-15.0])

    # ------- Same result training both epochs in a single call -------
    reset_weights()
    model.train(train_data=data_train, epochs=2, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[-51.0]], [[-51.0]], [-15.0])

    # ------- Two identical samples in one batch: the mean error gives the same step -------
    data_train2 = DataLoader(model, source={"in1": [1, 1], "target1": [3, 3]})

    reset_weights()
    model.train(train_data=data_train2, epochs=1, batch_size=2, optimizer="sgd", lr=1.0)
    assert_weights([[3.0]], [[3.0]], [3.0])

    reset_weights()
    model.train(train_data=data_train2, epochs=2, batch_size=2, optimizer="sgd", lr=1.0)
    assert_weights([[-51.0]], [[-51.0]], [-15.0])


def test_training_values_fir_linear_only_model():
    ## The old test trained one sub-model at a time (`trainModel(models=...)`);
    ## the new API has a single model, so the other block is frozen instead
    input1 = Input("in1")
    target = Input("target1").last()

    fir_out = Fir(out_features=1, use_bias=False)(input1.last())
    linear_out = Linear(out_features=1, initializer="ones", bias_initializer="ones")(
        fir_out
    )

    output1 = Output("out1", fir_out)
    output2 = Output("out2", linear_out)

    model = Modely("test_model", inputs=[input1], outputs=[output1, output2])
    model.minimize("error", source=output2, target=target, loss="mse")
    model.build()

    assert fir_out.kernel is not None
    assert linear_out.kernel is not None
    assert linear_out.bias is not None

    def reset_weights():
        fir_out.kernel.assign([[1.0]])
        linear_out.kernel.assign([[1.0]])
        linear_out.bias.assign([1.0])

    def assert_weights(fir_kernel, kernel, bias):
        np.testing.assert_allclose(
            to_numpy(fir_out.kernel), fir_kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            to_numpy(linear_out.kernel), kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            to_numpy(linear_out.bias), bias, rtol=1e-5, atol=1e-5
        )

    reset_weights()
    result = model(
        {
            "in1": np.ones((1, 1, 1), dtype=np.float32),
            "target1": np.ones((1, 1, 1), dtype=np.float32) * 3,
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["out1"]), [[[1.0]]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(result["out2"]), [[[2.0]]], rtol=1e-5, atol=1e-5
    )

    dataset = {"in1": [1], "target1": [3]}
    data_train = DataLoader(model, source=dataset)

    # ------- Both blocks trainable -------
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[3.0]], [[3.0]], [3.0])
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[-51.0]], [[-51.0]], [-15.0])

    # ------- Only the Fir block trainable -------
    reset_weights()
    linear_out._layer.trainable = False
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[3.0]], [[1.0]], [1.0])
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[1.0]], [[1.0]], [1.0])

    # ------- Only the Linear block trainable -------
    reset_weights()
    linear_out._layer.trainable = True
    fir_out._layer.trainable = False
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[1.0]], [[3.0]], [3.0])
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[1.0]], [[-3.0]], [-3.0])


def test_training_values_fir_linear_more_samples():
    input1 = Input("in1")
    target = Input("out1").last()

    fir_out = Fir(out_features=1, use_bias=False)(input1.last())
    linear_out = Linear(out_features=1, initializer="ones", bias_initializer="ones")(
        fir_out
    )

    output1 = Output("out1-net", fir_out)
    output2 = Output("out2-net", linear_out)

    model = Modely("test_model", inputs=[input1], outputs=[output1, output2])
    model.minimize("error", source=output2, target=target, loss="mse")
    model.build()

    assert fir_out.kernel is not None
    assert linear_out.kernel is not None
    assert linear_out.bias is not None

    def reset_weights():
        fir_out.kernel.assign([[1.0]])
        linear_out.kernel.assign([[1.0]])
        linear_out.bias.assign([1.0])

    def assert_weights(fir_kernel, kernel, bias):
        np.testing.assert_allclose(
            to_numpy(fir_out.kernel), fir_kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            to_numpy(linear_out.kernel), kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            to_numpy(linear_out.bias), bias, rtol=1e-5, atol=1e-5
        )

    reset_weights()
    result = model(
        {
            "in1": np.ones((1, 1, 1), dtype=np.float32),
            "out1": np.ones((1, 1, 1), dtype=np.float32) * 3,
        }
    )
    np.testing.assert_allclose(
        to_numpy(result["out1-net"]), [[[1.0]]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(result["out2-net"]), [[[2.0]]], rtol=1e-5, atol=1e-5
    )

    # ------- Four samples in a single batch -------
    dataset = {"in1": [0, 2, 7, 1], "out1": [3, 4, 5, 1]}
    data_train = DataLoader(model, source=dataset)
    assert len(data_train) == 4

    model.train(train_data=data_train, epochs=1, batch_size=4, optimizer="sgd", lr=1.0)
    assert_weights([[-9.0]], [[-9.0]], [0.5])

    # ------- The old prediction_samples=3 grouped four consecutive samples into a
    # block, and train_batch_size=2 put two overlapping blocks in the same batch -------
    in1 = [0, 2, 7, 1, 5, 0, 2]
    out1 = [1, 4, 8, 2, 6, 1, 1]
    order = [index for start in range(4) for index in range(start, start + 4)]
    data_train2 = DataLoader(
        model,
        source={"in1": [in1[i] for i in order], "out1": [out1[i] for i in order]},
    )
    assert len(data_train2) == 16

    reset_weights()
    model.train(
        train_data=data_train2,
        epochs=1,
        batch_size=8,
        optimizer="sgd",
        lr=1.0,
        shuffle=False,
    )
    assert_weights([[-162.75]], [[-162.75]], [-15.75])


def test_training_values_linear_fir_window():
    input1 = Input("in1", dim=2)
    target = Input("target").last()

    lin_out = Linear(out_features=1)(input1.sw(2))
    fir_out = Fir(out_features=1, use_bias=False)(lin_out)

    output1 = Output("out1", lin_out)
    output2 = Output("out2", fir_out)

    model = Modely("test_model", inputs=[input1], outputs=[output1, output2])
    model.minimize("error2", source=output2, target=target, loss="mse")
    model.build()

    assert lin_out.kernel is not None
    assert lin_out.bias is not None
    assert fir_out.kernel is not None

    def reset_weights():
        lin_out.kernel.assign([[-1.0], [-5.0]])
        lin_out.bias.assign([1.0])
        fir_out.kernel.assign([[4.0], [5.0]])

    def assert_weights(kernel, bias, fir_kernel):
        np.testing.assert_allclose(
            to_numpy(lin_out.kernel), kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(to_numpy(lin_out.bias), bias, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            to_numpy(fir_out.kernel), fir_kernel, rtol=1e-5, atol=1e-5
        )

    reset_weights()
    dataset = {
        "in1": [[0, 1], [2, 3], [7, 4], [1, 3], [4, 2]],
        "target": [3, 4, 5, 1, 3],
    }
    data = DataLoader(model, source=dataset)
    assert len(data) == 4
    result = model(data.as_dict())
    np.testing.assert_allclose(
        to_numpy(result["out1"])[:, 0],
        [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0]],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["out2"]).reshape(-1),
        [-96.0, -194.0, -179.0, -125.0],
        rtol=1e-5,
        atol=1e-5,
    )

    # ------- One sample -------
    # The old state carried a window of five, so only the last window was left
    data_train = DataLoader(model, source={"in1": [[1, 3], [4, 2]], "target": [1, 3]})
    assert len(data_train) == 1

    assert_weights([[-1.0], [-5.0]], [1.0], [[4.0], [5.0]])
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[6143.0], [5627.0]], [2305.0], [[-3836.0], [-3323.0]])

    # ------- Four samples in a single batch -------
    dataset2 = {
        "in1": [[1, 3], [4, 2], [6, 5], [4, 5], [0, 0]],
        "target": [1, 3, 0, 1, 0],
    }
    data_train2 = DataLoader(model, source=dataset2)
    assert len(data_train2) == 4

    reset_weights()
    model.train(train_data=data_train2, epochs=1, batch_size=4, optimizer="sgd", lr=1.0)
    assert_weights([[12779.0], [11678.5]], [3142.0], [[-7682.0], [-7457.5]])


def test_training_values_fir_and_linear_closed_loop():
    input1 = Input("in1")
    input2 = Input("in2")
    target1 = Input("target1").last()
    target2 = Input("target2").last()

    fir_out = Fir(out_features=1, use_bias=False)(input1.last())
    lin_out = Linear(out_features=1, initializer="ones", bias_initializer="ones")(
        input2.last()
    )
    output1 = Output("out1", fir_out)
    output2 = Output("out2", lin_out)

    ## Each relation is closed on its own input, so each one is its own Loop body
    body1 = Modely("body1", inputs=[input1], outputs=[output1]).build()
    body2 = Modely("body2", inputs=[input2], outputs=[output2]).build()

    assert fir_out.kernel is not None
    assert lin_out.kernel is not None
    assert lin_out.bias is not None

    def reset_weights():
        fir_out.kernel.assign([[1.0]])
        lin_out.kernel.assign([[1.0]])
        lin_out.bias.assign([1.0])

    def assert_weights(fir_kernel, kernel, bias):
        np.testing.assert_allclose(
            to_numpy(fir_out.kernel), fir_kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            to_numpy(lin_out.kernel), kernel, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(to_numpy(lin_out.bias), bias, rtol=1e-5, atol=1e-5)

    reset_weights()

    # ------- Six closed loop steps from a zero state -------
    seed1 = Input("seed1", seq=6)
    seed2 = Input("seed2", seq=6)
    loop1 = Loop(
        f=body1, callback={input1: output1}, initial={input1: seed1}, name="loop1"
    )
    loop2 = Loop(
        f=body2, callback={input2: output2}, initial={input2: seed2}, name="loop2"
    )
    rollout = Modely(
        "rollout",
        inputs=[seed1, seed2],
        outputs=[Output("rollout1", loop1), Output("rollout2", loop2)],
    ).build()

    zeros = np.zeros((1, 1, 1, 6), dtype=np.float32)
    result = rollout({"seed1": zeros, "seed2": zeros})
    np.testing.assert_allclose(
        to_numpy(result["rollout1"]).reshape(-1), [0.0] * 6, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(result["rollout2"]).reshape(-1),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        rtol=1e-5,
        atol=1e-5,
    )

    # ------- Training is a single step, so the loops are not rolled -------
    model = Modely("test_model", inputs=[input1, input2], outputs=[output1, output2])
    model.minimize("error1", source=output1, target=target1, loss="mse")
    model.minimize("error2", source=output2, target=target2, loss="mse")
    model.build()

    ones = np.ones((1, 1, 1), dtype=np.float32)
    reset_weights()  ## build() creates fresh layers, so the weights are set again
    result = model({"in1": ones, "in2": ones, "target1": ones * 3, "target2": ones * 3})
    np.testing.assert_allclose(
        to_numpy(result["out1"]), [[[1.0]]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(result["out2"]), [[[2.0]]], rtol=1e-5, atol=1e-5
    )

    dataset = {"in1": [1], "in2": [1.0], "target1": [3], "target2": [3]}
    data_train = DataLoader(model, source=dataset)

    assert_weights([[1.0]], [[1.0]], [1.0])
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[5.0]], [[3.0]], [3.0])
    model.train(train_data=data_train, epochs=1, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[1.0]], [[-3.0]], [-3.0])

    reset_weights()
    model.train(train_data=data_train, epochs=2, batch_size=1, optimizer="sgd", lr=1.0)
    assert_weights([[1.0]], [[-3.0]], [-3.0])

    # ------- Two identical samples in one batch -------
    dataset2 = {
        "in1": [1.0, 1.0],
        "in2": [1.0, 1.0],
        "target1": [3.0, 3.0],
        "target2": [3.0, 3.0],
    }
    data_train2 = DataLoader(model, source=dataset2)

    reset_weights()
    model.train(train_data=data_train2, epochs=1, batch_size=2, optimizer="sgd", lr=1.0)
    assert_weights([[5.0]], [[3.0]], [3.0])

    reset_weights()
    model.train(train_data=data_train2, epochs=2, batch_size=2, optimizer="sgd", lr=1.0)
    assert_weights([[1.0]], [[-3.0]], [-3.0])


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


def test_train_equation_learner_updates_symbolic_coefficients():
    input_node = Input("equation_train_input")
    target = Input("equation_train_target").last()
    equation = EquationLearner(
        functions=["identity"],
        linear_in=Linear(
            out_features=1,
            use_bias=False,
            initializer="ones",
        ),
        linear_out=Linear(
            out_features=1,
            use_bias=False,
            initializer="ones",
        ),
        name="train_equation",
    )
    prediction = equation(input_node.last())
    output = Output("equation_train_output", prediction)
    model = Modely("equation_train_model", inputs=[input_node], outputs=[output])
    model.minimize("error", source=output, target=target, loss="mse")
    model.build()

    data = DataLoader(
        model,
        source={"equation_train_input": [1.0], "equation_train_target": [3.0]},
    )
    history = model.train(
        train_data=data,
        epochs=1,
        batch_size=1,
        optimizer="sgd",
        lr=0.1,
    )

    assert equation.linear_in is not None
    assert equation.linear_out is not None
    assert equation.linear_in.kernel is not None
    assert equation.linear_out.kernel is not None
    assert np.isfinite(history["loss"][-1])
    np.testing.assert_allclose(
        to_numpy(equation.linear_in.kernel), [[1.4]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(equation.linear_out.kernel), [[1.4]], rtol=1e-5, atol=1e-5
    )


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


def test_train_with_loop():
    state = Input("state")
    x = Input("x", seq=5)
    target = Input("target")
    relation = Linear(
        out_features=1,
        use_bias=True,
        initializer="ones",
        bias_initializer="zeros",
    )(state.last())
    output = Output("out", relation)
    body = Modely("body", inputs=[state], outputs=[output])
    body.build()
    # x carries the five rollout steps and seeds the state with x[0]
    loop = Loop(
        f=body,
        callback={state: output},
        initial={state: x},
        name="loop",
        collect=False,
    )
    out_loop = Output("out_loop", loop)
    model = Modely("model", inputs=[x], outputs=[out_loop])
    model.minimize("error", source=out_loop, target=target.last(), loss="mse")
    model.build()

    dataset = {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "target": [
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
        ],
    }
    data = DataLoader(model, source=dataset)

    assert data.dataset["x"].shape == (16, 1, 1, 5)
    assert data.dataset["target"].shape == (16, 1, 1)
    np.testing.assert_array_equal(data.dataset["x"][0, 0, 0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(data.dataset["x"][-1, 0, 0], [16, 17, 18, 19, 20])
    np.testing.assert_array_equal(data.dataset["target"][:, 0, 0], np.arange(25, 41))

    initial_prediction = model(data.as_dict())
    initial_error = np.mean(
        np.square(to_numpy(initial_prediction["out_loop"]) - data.dataset["target"])
    )
    history = model.train(
        train_data=data,
        epochs=10,
        batch_size=4,
        optimizer="adam",
        lr=0.01,
    )
    final_prediction = model(data.as_dict())
    final_error = np.mean(
        np.square(to_numpy(final_prediction["out_loop"]) - data.dataset["target"])
    )

    assert np.isfinite(final_error)
    assert final_error < initial_error
    assert history["loss"][-1] < history["loss"][0]


def test_train_with_roll():
    x = Input("x")
    target = Input("target")
    relation = Linear(
        out_features=1,
        use_bias=True,
        initializer="ones",
        bias_initializer="zeros",
    )(x.last())
    output = Output("out", relation)
    body = Modely("body", inputs=[x], outputs=[output])
    body.build()
    roll = Roll(f=body, callback={x: output}, steps=3, name="roll")
    out_scan = Output("out_scan", roll)
    model = Modely("model", inputs=[x], outputs=[out_scan])
    model.minimize("error", source=out_scan, target=target.last(), loss="mse")
    model.build()

    dataset = {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "target": [
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
        ],
    }
    data = DataLoader(model, source=dataset)

    assert data.dataset["x"].shape == (20, 1, 1)
    assert data.dataset["target"].shape == (20, 1, 1)
    np.testing.assert_array_equal(data.dataset["x"][:, 0, 0], np.arange(1, 21))
    np.testing.assert_array_equal(data.dataset["target"][:, 0, 0], np.arange(21, 41))

    initial_prediction = model(data.as_dict())
    initial_error = np.mean(
        np.square(to_numpy(initial_prediction["out_scan"]) - data.dataset["target"])
    )
    history = model.train(
        train_data=data,
        epochs=10,
        batch_size=4,
        optimizer="adam",
        lr=0.01,
    )
    final_prediction = model(data.as_dict())
    final_error = np.mean(
        np.square(to_numpy(final_prediction["out_scan"]) - data.dataset["target"])
    )

    assert np.isfinite(final_error)
    assert final_error < initial_error
    assert history["loss"][-1] < history["loss"][0]


def test_train_with_model_rollback_uses_final_value_only():
    x = Input("closed_train_x")
    target = Input("closed_train_target")
    relation = Linear(
        out_features=1,
        use_bias=False,
        initializer="ones",
        name="closed_train_linear",
    )(x.last())
    output = Output("closed_train_output", relation)
    model = Modely("closed_train_model", inputs=[x], outputs=[output])
    model.rollback({x: output}, steps=3)
    model.minimize("closed_train_error", output, target.last(), loss="mse")
    model.build()

    data = DataLoader(
        model,
        source={"closed_train_x": [2.0], "closed_train_target": [4.0]},
    )
    optimizer = keras.optimizers.SGD(learning_rate=0.01)

    before = model(data.as_dict())
    assert before["closed_train_output"].shape == (1, 1, 1)
    np.testing.assert_allclose(to_numpy(before["closed_train_output"]), [[[2.0]]])

    history = model.train(
        train_data=data,
        epochs=1,
        batch_size=1,
        optimizer=optimizer,
    )

    # The final rollout value is x * w**3. At w=1, MSE(2, 4)=4 and
    # d(loss)/dw=-24, therefore one SGD update gives w=1.24.
    assert relation.kernel is not None
    np.testing.assert_allclose(history["loss"], [4.0], atol=1e-5)
    np.testing.assert_allclose(to_numpy(relation.kernel), [[1.24]], atol=1e-5)
    assert int(to_numpy(optimizer.iterations)) == 1


def test_masked_loss_ignores_padded_target_steps():
    loss = MaskedLoss(_resolve_loss("mse"))
    y_true = np.array([[1.0, 2.0, np.nan, np.nan]], dtype=np.float32)
    far = np.array([[1.5, 2.5, 100.0, -100.0]], dtype=np.float32)
    near = np.array([[1.5, 2.5, 0.0, 7.0]], dtype=np.float32)

    ## Whatever the model predicts on a padded step cannot change the loss
    np.testing.assert_allclose(
        to_numpy(loss(y_true, far)), to_numpy(loss(y_true, near)), rtol=1e-6
    )
    ## Only the two real steps contribute, averaged over the padded width
    np.testing.assert_allclose(
        to_numpy(loss(y_true, far)), [(0.25 + 0.25) / 4], rtol=1e-6
    )


@pytest.mark.slow
def test_train_on_simulations_of_different_lengths(tmp_path):
    ## x[t + 1] = w * x[t], rolled out over the whole simulation
    ratio = 0.8
    body_x = Input("pad_body_x", dim=1)
    relation = Linear(
        out_features=1,
        use_bias=False,
        initializer="ones",
        name="pad_body_linear",
    )(body_x.last())
    body_out = Output("pad_body_out", relation)
    body = Modely("pad_body", inputs=[body_x], outputs=[body_out]).build()

    seed = Input("pad_x", dim=1, seq=(None,))
    loop = Loop(
        f=body,
        callback={body_x: body_out},
        initial={body_x: seed},
        length=6,
        name="pad_loop",
    )
    output = Output("pad_out", loop)
    model = Modely("pad_loop_model", inputs=[seed], outputs=[output])
    model.minimize(
        "pad_error", output, Input("pad_target", dim=1, seq=(None,)), loss="mse"
    )
    model.build()

    ## Three simulations of different lengths, the longest fixing the rollout
    signals = [
        start * ratio ** np.arange(length, dtype=np.float32)
        for start, length in ((1.0, 6), (2.0, 3), (-1.5, 5))
    ]
    data = DataLoader(
        model,
        source=[{"pad_x": signal, "pad_target": ratio * signal} for signal in signals],
        seq_length="full",
    )

    assert data.dataset["pad_x"].shape == (3, 1, 1, 6)
    assert data.mask is not None
    np.testing.assert_array_equal(data.mask.sum(axis=1), [6, 3, 5])

    history = model.train(
        train_data=data, epochs=300, batch_size=3, lr=0.02, optimizer="adam"
    )

    assert np.isfinite(history["loss"][-1])
    assert history["loss"][-1] < history["loss"][0]
    assert relation.kernel is not None
    np.testing.assert_allclose(to_numpy(relation.kernel), [[ratio]], atol=1e-2)

    ## The export is the model as declared, without its minimizers: it reads the
    ## simulations only, not the target the loss compared them against
    export_path = os.path.join(tmp_path, "padded_loop_model.keras")
    model.export_keras(export_path)
    reloaded = Modely.import_keras(export_path)
    assert [tensor.name for tensor in reloaded.inputs] == ["pad_x"]  # type: ignore
    inputs = {"pad_x": data.as_dict()["pad_x"]}
    np.testing.assert_allclose(
        to_numpy(reloaded(inputs)["pad_out"]),  # type: ignore
        to_numpy(model(inputs)["pad_out"]),
        atol=1e-5,
    )


def test_masked_loss_survives_a_diverging_padded_rollout():
    ## Past the end of its simulation a rollout is driven by the model alone and
    ## can overflow. Those steps must still cost nothing, not poison the loss.
    loss = MaskedLoss(_resolve_loss("mse"))
    y_true = np.array([[1.0, 2.0, np.nan, np.nan]], dtype=np.float32)
    exploded = np.array([[1.5, 2.5, np.inf, -np.inf]], dtype=np.float32)

    np.testing.assert_allclose(
        to_numpy(loss(y_true, exploded)), [(0.25 + 0.25) / 4], rtol=1e-6
    )
