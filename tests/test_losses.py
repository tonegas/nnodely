import unittest, os, sys
import numpy as np
import torch

from nnodely import *
from nnodely.basic.relation import NeuObj
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 3 Tests
# Test the looses comparison between closed loop and states
# Test the looses comparison between connect and states

data_folder = os.path.join(os.path.dirname(__file__), "_data/")


class ModelyTrainingTest(unittest.TestCase):
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert (
            np.asarray(data1, dtype=np.float32).ndim
            == np.asarray(data2, dtype=np.float32).ndim
        ), (
            f"Inputs must have the same dimension! Received {type(data1)} and {type(data2)}"
        )
        if type(data1) == type(data2) == list:
            self.assertEqual(len(data1), len(data2))
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_losses_compare(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        target1 = Input("out1")
        target2 = Input("out2")
        a = Parameter("a", sw=1, values=[[1]])
        output1 = Output("out", Fir(W=a)(input1.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel("model", output1)
        test.addMinimize("error1", target1.last(), output1)
        test.addMinimize("error2", target2.last(), output1)
        test.neuralizeModel()

        dataset = {
            "in1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "out1": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "out2": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        }
        test.loadData(name="dataset", source=dataset)
        test.trainAndAnalyze(
            optimizer="SGD", num_of_epochs=5, lr=0.5, splits=[70, 20, 10]
        )
        self.TestAlmostEqual(
            [[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]],
            test.prediction["dataset_train"]["error1"]["A"],
        )
        self.TestAlmostEqual(
            [[[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]]],
            test.prediction["dataset_train"]["error1"]["B"],
        )
        self.TestAlmostEqual(
            [[[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]]],
            test.prediction["dataset_train"]["error2"]["A"],
        )
        self.TestAlmostEqual(
            [1.0, 16.0, 1.0, 16.0, 1.0], test._training["error1"]["train"]
        )
        self.TestAlmostEqual(
            [16.0, 1.0, 16.0, 1.0, 16.0], test._training["error1"]["val"]
        )
        self.TestAlmostEqual(
            [16.0, 1.0, 16.0, 1.0, 16.0], test._training["error2"]["train"]
        )
        self.TestAlmostEqual(
            [1.0, 16.0, 1.0, 16.0, 1.0], test._training["error2"]["val"]
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["total"]["mean_error"],
            (test._training["error1"]["val"][-1] + test._training["error2"]["val"][-1])
            / 2.0,
        )
        self.TestAlmostEqual(
            test.performance["dataset_test"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_test"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_test"]["total"]["mean_error"],
            (test._training["error1"]["val"][-1] + test._training["error2"]["val"][-1])
            / 2.0,
        )

        test.neuralizeModel(clear_model=True)
        test.trainAndAnalyze(
            optimizer="SGD",
            splits=[60, 20, 20],
            num_of_epochs=5,
            lr=0.5,
            train_batch_size=2,
        )
        self.TestAlmostEqual(
            [[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]],
            test.prediction["dataset_train"]["error1"]["A"],
        )
        self.TestAlmostEqual(
            [[[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]]],
            test.prediction["dataset_train"]["error1"]["B"],
        )
        self.TestAlmostEqual(
            [[[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]]],
            test.prediction["dataset_train"]["error2"]["A"],
        )
        self.TestAlmostEqual(
            [6.0, 11.0, 6.0, 11.0, 6.0], test._training["error1"]["train"]
        )
        self.TestAlmostEqual(
            [16.0, 1.0, 16.0, 1.0, 16.0], test._training["error1"]["val"]
        )
        self.TestAlmostEqual(
            [11.0, 6.0, 11.0, 6.0, 11.0], test._training["error2"]["train"]
        )
        self.TestAlmostEqual(
            [1.0, 16.0, 1.0, 16.0, 1.0], test._training["error2"]["val"]
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_test"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_test"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )

    def test_losses_compare_closed_loop_state(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        target1 = Input("out1")
        target2 = Input("out2")
        a = Parameter("a", sw=1, values=[[1]])
        relation = Fir(W=a)(input1.last())
        relation.closedLoop(input1)
        output1 = Output("out", relation)

        test = Modely(visualizer=None, seed=42)
        test.addModel("model", output1)
        test.addMinimize("error1", target1.last(), output1)
        test.addMinimize("error2", target2.last(), output1)
        test.neuralizeModel()

        dataset = {
            "in1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "out1": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "out2": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        }
        test.loadData(name="dataset", source=dataset)
        test.trainAndAnalyze(
            optimizer="SGD",
            num_of_epochs=5,
            lr=0.5,
            shuffle_data=False,
            splits=[70, 20, 10],
        )
        self.TestAlmostEqual(
            [[[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]]],
            test.prediction["dataset_train"]["error1"]["A"],
        )
        self.TestAlmostEqual(
            [[[[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]]]],
            test.prediction["dataset_train"]["error1"]["B"],
        )
        self.TestAlmostEqual(
            [[[[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]]]],
            test.prediction["dataset_train"]["error2"]["A"],
        )
        self.TestAlmostEqual(
            [1.0, 16.0, 1.0, 16.0, 1.0], test._training["error1"]["train"]
        )
        self.TestAlmostEqual(
            [16.0, 1.0, 16.0, 1.0, 16.0], test._training["error1"]["val"]
        )
        self.TestAlmostEqual(
            [16.0, 1.0, 16.0, 1.0, 16.0], test._training["error2"]["train"]
        )
        self.TestAlmostEqual(
            [1.0, 16.0, 1.0, 16.0, 1.0], test._training["error2"]["val"]
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["total"]["mean_error"],
            (test._training["error1"]["val"][-1] + test._training["error2"]["val"][-1])
            / 2.0,
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["total"]["mean_error"],
            (test._training["error1"]["val"][-1] + test._training["error2"]["val"][-1])
            / 2.0,
        )

        test.neuralizeModel(clear_model=True)
        test.trainAndAnalyze(
            optimizer="SGD",
            splits=[60, 20, 20],
            num_of_epochs=5,
            lr=0.5,
            train_batch_size=2,
        )
        self.TestAlmostEqual(
            [[[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]]],
            test.prediction["dataset_train"]["error1"]["A"],
        )
        self.TestAlmostEqual(
            [[[[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]]]],
            test.prediction["dataset_train"]["error1"]["B"],
        )
        self.TestAlmostEqual(
            [[[[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]]]],
            test.prediction["dataset_train"]["error2"]["A"],
        )
        self.TestAlmostEqual(
            [6.0, 11.0, 6.0, 11.0, 6.0], test._training["error1"]["train"]
        )
        self.TestAlmostEqual(
            [16.0, 1.0, 16.0, 1.0, 16.0], test._training["error1"]["val"]
        )
        self.TestAlmostEqual(
            [11.0, 6.0, 11.0, 6.0, 11.0], test._training["error2"]["train"]
        )
        self.TestAlmostEqual(
            [1.0, 16.0, 1.0, 16.0, 1.0], test._training["error2"]["val"]
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_test"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_test"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )

        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainAndAnalyze(
                optimizer="SGD",
                splits=[60, 20, 20],
                num_of_epochs=5,
                lr=0.5,
                train_batch_size=2,
                prediction_samples=4,
            )

        test.neuralizeModel(clear_model=True)
        test.trainAndAnalyze(
            optimizer="SGD",
            splits=[50, 50, 0],
            num_of_epochs=5,
            lr=0.001,
            train_batch_size=2,
            prediction_samples=3,
        )

        self.TestAlmostEqual(
            [
                [[[2.0]], [[2.0]]],
                [[[2.0]], [[2.0]]],
                [[[2.0]], [[2.0]]],
                [[[2.0]], [[2.0]]],
            ],
            test.prediction["dataset_train"]["error1"]["A"],
        )
        self.TestAlmostEqual(
            [
                [[[1.1285]], [[1.1285]]],
                [[[1.2735]], [[1.2735]]],
                [[[1.4371]], [[1.4371]]],
                [[[1.6217]], [[1.6217]]],
            ],
            test.prediction["dataset_train"]["error1"]["B"],
        )
        self.TestAlmostEqual(
            [
                [[[5.0]], [[5.0]]],
                [[[5.0]], [[5.0]]],
                [[[5.0]], [[5.0]]],
                [[[5.0]], [[5.0]]],
            ],
            test.prediction["dataset_train"]["error2"]["A"],
        )
        self.TestAlmostEqual(
            [1.0, 0.8768, 0.75615, 0.64057, 0.533], test._training["error1"]["train"]
        )
        self.TestAlmostEqual(
            [0.8768, 0.75615, 0.64057, 0.533, 0.4368], test._training["error1"]["val"]
        )
        self.TestAlmostEqual(
            [16.0, 15.4923, 14.9602, 14.4059, 13.8328],
            test._training["error2"]["train"],
        )
        self.TestAlmostEqual(
            [15.4923, 14.9602, 14.4059, 13.8328, 13.2457],
            test._training["error2"]["val"],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )

    def test_losses_compare_closed_loop(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        target1 = Input("out1")
        target2 = Input("out2")
        a = Parameter("a", sw=1, values=[[1]])
        output1 = Output("out", Fir(W=a)(input1.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel("model", output1)
        test.addMinimize("error1", target1.last(), output1)
        test.addMinimize("error2", target2.last(), output1)
        test.neuralizeModel()

        dataset = {
            "in1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "out1": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "out2": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        }
        test.loadData(name="dataset", source=dataset)
        test.trainAndAnalyze(
            optimizer="SGD",
            splits=[50, 50, 0],
            num_of_epochs=5,
            lr=0.001,
            train_batch_size=2,
            prediction_samples=3,
            closed_loop={"in1": "out"},
        )

        self.TestAlmostEqual(
            [
                [[[2.0]], [[2.0]]],
                [[[2.0]], [[2.0]]],
                [[[2.0]], [[2.0]]],
                [[[2.0]], [[2.0]]],
            ],
            test.prediction["dataset_train"]["error1"]["A"],
        )
        self.TestAlmostEqual(
            [
                [[[1.1285]], [[1.1285]]],
                [[[1.2735]], [[1.2735]]],
                [[[1.4371]], [[1.4371]]],
                [[[1.6217]], [[1.6217]]],
            ],
            test.prediction["dataset_train"]["error1"]["B"],
        )
        self.TestAlmostEqual(
            [
                [[[5.0]], [[5.0]]],
                [[[5.0]], [[5.0]]],
                [[[5.0]], [[5.0]]],
                [[[5.0]], [[5.0]]],
            ],
            test.prediction["dataset_train"]["error2"]["A"],
        )
        self.TestAlmostEqual(
            [1.0, 0.8768, 0.75615, 0.64057, 0.533], test._training["error1"]["train"]
        )
        self.TestAlmostEqual(
            [0.8768, 0.75615, 0.64057, 0.533, 0.4368], test._training["error1"]["val"]
        )
        self.TestAlmostEqual(
            [16.0, 15.4923, 14.9602, 14.4059, 13.8328],
            test._training["error2"]["train"],
        )
        self.TestAlmostEqual(
            [15.4923, 14.9602, 14.4059, 13.8328, 13.2457],
            test._training["error2"]["val"],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error1"]["mse"],
            test._training["error1"]["val"][-1],
        )
        self.TestAlmostEqual(
            test.performance["dataset_val"]["error2"]["mse"],
            test._training["error2"]["val"][-1],
        )

    def test_categorical_crossentropy(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        target1 = Input("out1")
        target2 = Input("out2", dimensions=5)

        k = Parameter("k", values=[[0.1, 0.1, 0.1, 0.1, 0.6]])
        linear = Linear(output_dimension=5, W=k, b=False)(input1.last())
        output = Output("out", linear)

        test = Modely(visualizer=None, seed=42, log_internal=True)
        test.addModel("model", output)
        test.addMinimize(
            "error1", output, target1.last(), loss_function="cross_entropy"
        )
        test.addMinimize(
            "error2", output, target2.last(), loss_function="cross_entropy"
        )
        test.neuralizeModel()

        dataset = {"in1": [1], "out1": [4], "out2": [[0.0, 0.0, 0.0, 0.0, 1.0]]}
        test.loadData(name="dataset", source=dataset)
        test.trainAndAnalyze(
            optimizer="SGD",
            train_dataset="dataset",
            train_batch_size=1,
            num_of_epochs=1,
            lr=0.0,
        )
        loss = torch.nn.CrossEntropyLoss()
        self.assertAlmostEqual(
            1.2314292192459106,
            loss(
                torch.tensor(test.prediction["dataset"]["error1"]["A"]).squeeze(),
                torch.tensor(
                    test.prediction["dataset"]["error1"]["B"], dtype=torch.long
                ).squeeze(),
            ).item(),
        )
        self.assertAlmostEqual(
            1.2314292192459106,
            loss(
                torch.tensor(test.prediction["dataset"]["error2"]["A"]).squeeze(),
                torch.tensor(
                    test.prediction["dataset"]["error2"]["B"], dtype=torch.float32
                ).squeeze(),
            ).item(),
        )
