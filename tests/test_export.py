import os
import unittest
import torch
import shutil
import numpy as np

from nnodely import *
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

# 11 Tests
# Test of export and import the network to a file in different format


class ModelyExportTest(unittest.TestCase):
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

    def __init__(self, *args, **kwargs):
        super(ModelyExportTest, self).__init__(*args, **kwargs)
        clearNames()

        self.result_path = "./results"
        self.test = Modely(visualizer=None, seed=42, workspace=self.result_path)

        x = Input("x")
        y = Input("y")
        z = Input("z")

        ## create the relations
        def myFun(K1, p1, p2):
            return K1 * p1 * p2

        K_x = Parameter(
            "k_x", dimensions=1, tw=1, init="init_constant", init_params={"value": 1}
        )
        K_y = Parameter("k_y", dimensions=1, tw=1)
        w = Parameter(
            "w", dimensions=1, tw=1, init="init_constant", init_params={"value": 1}
        )
        t = Parameter("t", dimensions=1, tw=1)
        c_v = Constant("c_v", tw=1, values=[[1], [2]])
        c = 5
        w_5 = Parameter("w_5", dimensions=1, tw=5)
        t_5 = Parameter("t_5", dimensions=1, tw=5)
        c_5 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        c_5_2 = Constant("c_5_2", tw=5, values=c_5)
        parfun_x = ParamFun(myFun, parameters_and_constants=[K_x, c_v])
        parfun_y = ParamFun(myFun, parameters_and_constants=[K_y])
        parfun_z = ParamFun(myFun)
        fir_w = Fir(W=w_5)(x.tw(5))
        fir_t = Fir(W=t_5)(y.tw(5))
        time_part = TimePart(x.tw(5), i=1, j=3)
        sample_select = SampleSelect(x.sw(5), i=1)

        def fuzzyfun(x):
            return torch.tan(x)

        fuzzy = Fuzzify(output_dimension=4, range=[0, 4], functions=fuzzyfun)(x.tw(1))
        fuzzyTriang = Fuzzify(centers=[1, 2, 3, 7])(x.tw(1))

        out = Output("out", Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v)))
        out2 = Output("out2", Add(w, x.tw(1)) + Add(t, y.tw(1)) + Add(w, c))
        out3 = Output("out3", Add(fir_w, fir_t))
        out4 = Output("out4", Linear(output_dimension=1)(fuzzy + fuzzyTriang))
        out5 = Output("out5", Fir(time_part) + Fir(sample_select))
        out6 = Output("out6", LocalModel(output_function=Fir())(x.tw(1), fuzzy))
        with self.assertRaises(TypeError):
            parfun_z(x.tw(5), t_5, c_5)
        out7 = Output(
            "out7",
            Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v))
            + Fir(parfun_z(x.tw(5), t_5, c_5_2)),
        )

        self.test.addModel("modelA", out)
        self.test.addModel("modelB", [out2, out3, out4])
        self.test.addModel("modelC", [out4, out5, out6])
        self.test.addModel("modelD", [out7])
        self.test.addMinimize("error1", x.last(), out)
        self.test.addMinimize("error2", y.last(), out3, loss_function="rmse")
        self.test.addMinimize("error3", z.last(), out6, loss_function="rmse")
        self.test.addMinimize("error4", z.last(), out7, loss_function="rmse")

    def test_export_pt(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export torch file .pt
        # Save torch model and load it
        self.test.neuralizeModel(0.5)
        old_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.saveTorchModel()
        self.test.neuralizeModel(clear_model=True)
        # The new_out is different from the old_out because the model is cleared
        new_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        # The new_out_after_load is the same as the old_out because the model is loaded with the same parameters
        self.test.loadTorchModel()
        new_out_after_load = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )

        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out)
        self.assertEqual(old_out, new_out_after_load)

        with self.assertRaises(RuntimeError):
            test2 = Modely(visualizer=None, workspace=self.result_path)
            # You need not neuralized model to load a torch model
            test2.loadTorchModel()

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_json_not_neuralized(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export json of nnodely model before neuralize
        # Save a not neuralized nnodely json model and load it
        self.test.saveModel()  # Save a model without parameter values and samples values
        with self.assertRaises(RuntimeError):
            self.test(
                {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                }
            )
        self.test.loadModel()  # Load the nnodely model without parameter values
        with self.assertRaises(RuntimeError):
            self.test(
                {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                }
            )
        test2 = Modely(visualizer=None, workspace=self.test.getWorkspace())
        test2.loadModel()  # Load the nnodely model with parameter values
        with self.assertRaises(AttributeError):
            test2.neuralized = True
        with self.assertRaises(AttributeError):
            test2.traced = True
        self.assertEqual(test2.json, self.test.json)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_json_untrained(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export json of nnodely model
        # Save a untrained nnodely json model and load it
        # the new_out and new_out_after_load are different because the model saved model is not trained
        self.test.neuralizeModel(0.5)
        old_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.saveModel()  # Save a model without parameter values
        self.test.neuralizeModel(clear_model=True)  # Create a new torch model
        new_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.loadModel()  # Load the nnodely model without parameter values
        # Use the preloaded torch model for inference
        with self.assertRaises(RuntimeError):
            self.test(
                {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                }
            )
        self.test.neuralizeModel(0.5)
        new_out_after_load = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out)
        with self.assertRaises(AssertionError):
            self.assertEqual(new_out, new_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_json_trained(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export json of nnodely model with parameter valuess
        # The old_out is the same as the new_out_after_load because the model is loaded with the same parameters
        self.test.neuralizeModel(0.5)
        old_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.saveModel()  # Save the model with and without parameter values
        self.test.neuralizeModel(clear_model=True)  # Create a new torch model
        new_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.loadModel()  # Load the nnodely model with parameter values
        with self.assertRaises(RuntimeError):
            self.test(
                {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                }
            )
        self.test.neuralizeModel()
        new_out_after_load = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out)
        with self.assertRaises(AssertionError):
            self.assertEqual(new_out, new_out_after_load)
        self.assertEqual(old_out, new_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_import_json_new_object(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Import nnodely json model in a new object
        self.test.neuralizeModel(0.5)
        old_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.neuralizeModel()
        self.test.saveModel()  # Save the model with and without parameter values
        test2 = Modely(visualizer=None, workspace=self.test.getWorkspace())
        test2.loadModel()  # Load the nnodely model with parameter values
        with self.assertRaises(RuntimeError):
            test2(
                {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                }
            )
        test2.neuralizeModel()
        new_model_out_after_load = test2(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.assertEqual(old_out, new_model_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_torch_script(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Export and import of a torch script .py
        # The old_out is the same as the new_out_after_load because the model is loaded with the same parameters
        with self.assertRaises(RuntimeError):
            self.test.exportPythonModel()  # The model is not neuralized yet
        self.test.neuralizeModel(0.5)
        old_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.exportPythonModel()  # Export the trace model
        self.test.neuralizeModel(clear_model=True)  # Create a new torch model
        new_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.importPythonModel()  # Import the tracer model
        with self.assertRaises(RuntimeError):
            self.test.exportPythonModel()  # The model is traced
        # Perform inference with the imported tracer model
        new_out_after_load = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out)
        self.assertEqual(old_out, new_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_torch_script_new_object(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Import of a torch script .py
        self.test.neuralizeModel(0.5, clear_model=True)
        old_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.exportPythonModel()  # Export the trace model
        self.test.neuralizeModel(clear_model=True)
        test2 = Modely(visualizer=None, workspace=self.test.getWorkspace())
        test2.importPythonModel()  # Load the nnodely model with parameter values
        with self.assertRaises(RuntimeError):
            test2.exportPythonModel()  # The model is traced
        new_out_after_load = test2(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.assertEqual(old_out, new_out_after_load)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_trained_torch_script(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Perform training on an imported tracer model
        data_x = np.arange(0.0, 1, 0.1)
        data_y = np.arange(0.0, 1, 0.1)
        a, b = -1.0, 2.0
        dataset = {"x": data_x, "y": data_y, "z": a * data_x + b * data_y}
        params = {"num_of_epochs": 1, "lr": 0.01}
        self.test.neuralizeModel(0.5, clear_model=True)
        old_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.exportPythonModel()  # Export the trace model
        self.test.loadData(name="dataset", source=dataset)  # Create the dataset
        self.test.trainModel(
            optimizer="SGD", training_params=params
        )  # Train the traced model
        new_out_after_train = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out_after_train)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_torch_script_new_object_train(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)
        # Perform training on an imported new tracer model
        self.test.neuralizeModel(0.5, clear_model=True)
        old_out = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        self.test.exportPythonModel()  # Export the trace model
        data_x = np.arange(0.0, 1, 0.1)
        data_y = np.arange(0.0, 1, 0.1)
        a, b = -1.0, 2.0
        dataset = {"x": data_x, "y": data_y, "z": a * data_x + b * data_y}
        params = {"num_of_epochs": 1, "lr": 0.01}
        self.test.loadData(name="dataset", source=dataset)  # Create the dataset
        self.test.trainModel(
            optimizer="SGD", training_params=params
        )  # Train the traced model
        old_out_after_train = self.test(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, old_out_after_train)
        test2 = Modely(visualizer=None, workspace=self.test.getWorkspace())
        test2.importPythonModel()  # Load the nnodely model with parameter values
        test2.loadData(name="dataset", source=dataset)  # Create the dataset
        test2.trainModel(
            optimizer="SGD", training_params=params
        )  # Train the traced model
        new_out_after_train = test2(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(old_out, new_out_after_train)
        self.assertEqual(old_out_after_train, new_out_after_train)

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_onnx(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)

        self.test.neuralizeModel(0.5, clear_model=True)
        # Export all network with minimize
        self.test.exportONNX(
            inputs_order=["x", "y", "z"],
            outputs_order=["out", "out2", "out3", "out4", "out5", "out6", "out7"],
        )  # Export the onnx model
        # Export the all models in onnx format
        self.test.exportONNX(
            models=["modelA", "modelB", "modelC", "modelD"],
            inputs_order=["x", "y"],
            outputs_order=["out", "out2", "out3", "out4", "out5", "out6"],
        )  # Export the onnx model
        # Export only the modelB in onnx format
        self.test.exportONNX(
            inputs_order=["x", "y"],
            outputs_order=["out3", "out4", "out2"],
            models=["modelB"],
        )  # Export the onnx model
        self.assertTrue(
            os.path.exists(os.path.join(self.test.getWorkspace(), "onnx", "net.onnx"))
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.test.getWorkspace(),
                    "onnx",
                    "net_modelA_modelB_modelC_modelD.onnx",
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test.getWorkspace(), "onnx", "net_modelB.onnx")
            )
        )

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())

    def test_export_report(self):
        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
        os.makedirs(self.result_path, exist_ok=True)

        self.test.resetSeed(42)
        self.test.neuralizeModel(0.5, clear_model=True)
        data_x = np.arange(0.0, 10, 0.1)
        data_y = np.arange(0.0, 10, 0.1)
        a, b = -1.0, 2.0
        dataset = {"x": data_x, "y": data_y, "z": a * data_x + b * data_y}
        params = {"num_of_epochs": 20, "lr": 0.0005}
        self.test.loadData(name="dataset", source=dataset)  # Create the dataset
        self.test.trainModel(
            optimizer="SGD", training_params=params
        )  # Train the traced model
        self.test.exportReport()

        self.test.loadData(name="dataset2", source=dataset)  # Create the dataset
        self.test.trainAndAnalyze(
            optimizer="SGD",
            train_dataset="dataset",
            validation_dataset="dataset2",
            training_params=params,
        )  # Train the traced model
        self.test.exportReport()

        self.test.removeMinimize(["error1", "error2", "error3", "error4"])
        self.test.exportReport()

        if os.path.exists(self.test.getWorkspace()):
            shutil.rmtree(self.test.getWorkspace())
