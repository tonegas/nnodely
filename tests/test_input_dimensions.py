import unittest, sys, os, torch
import numpy as np

from nnodely import *
from nnodely.support.logger import logging, nnLogger
from nnodely.basic.relation import NeuObj

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 14 Tests
# This file tests the dimensions of the inputs in particular:
# The dimensions for each input
# input_tw_backward, input_tw_forward
# test.json['Inputs'][KEY]['ns'], and test.json['Inputs'][KEY]['ntot']
# The total maximum dimensions:
# json['Info']['ns'][0], json['Info']['ns'][1], and json['Info']['ntot']
# And finally the dimensions for each relation
# relation_samples


class ModelyNetworkBuildingTest(unittest.TestCase):
    def test_network_building_very_simple(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        rel1 = Fir(input1.last())
        fun = Output("out", rel1)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0,test.input_tw_backward['in1'])
        # self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(1, test.json["Inputs"]["in1"]["ns"][0])
        self.assertEqual(0, test.json["Inputs"]["in1"]["ns"][1])
        self.assertEqual(1, test.json["Inputs"]["in1"]["ntot"])

        self.assertEqual(1, test.json["Info"]["ns"][0])
        self.assertEqual(0, test.json["Info"]["ns"][1])
        self.assertEqual(1, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_simple(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output("out", rel1 + rel2)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0.05,test.input_tw_backward['in1'])
        # self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(5, test.json["Inputs"]["in1"]["ns"][0])
        self.assertEqual(0, test.json["Inputs"]["in1"]["ns"][1])
        self.assertEqual(5, test.json["Inputs"]["in1"]["ntot"])

        self.assertEqual(5, test.json["Info"]["ns"][0])
        self.assertEqual(0, test.json["Info"]["ns"][1])
        self.assertEqual(5, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_tw(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        input2 = Input("in2")
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02, 0.02]))
        fun = Output("out", rel1 + rel2 + rel3 + rel4)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual({'in1': 0.05, 'in2': 0.05}, test.input_tw_backward)
        # self.assertEqual({'in1': 0, 'in2': 0.02},test.input_tw_forward)
        # self.assertEqual({'in1': 5, 'in2': 5},test.input_ns_backward)
        # self.assertEqual({'in1': 0, 'in2': 2},test.input_ns_forward)
        # self.assertEqual({'in1': 5, 'in2': 7},test.input_n_samples)
        self.assertEqual([5, 0], test.json["Inputs"]["in1"]["ns"])
        self.assertEqual([5, 2], test.json["Inputs"]["in2"]["ns"])
        self.assertEqual(5, test.json["Inputs"]["in1"]["ntot"])
        self.assertEqual(7, test.json["Inputs"]["in2"]["ntot"])

        self.assertEqual(5, test.json["Info"]["ns"][0])
        self.assertEqual(2, test.json["Info"]["ns"][1])
        self.assertEqual(
            7, test.json["Info"]["ntot"]
        )  # 5 samples + 2 samples of the horizon

    def test_network_building_tw2(self):
        NeuObj.clearNames()
        input2 = Input("in2")
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02, 0.02]))
        rel5 = Fir(input2.tw([-0.03, 0.03]))
        rel6 = Fir(input2.tw([-0.03, 0]))
        rel7 = Fir(input2.tw(0.03))
        fun = Output("out", rel3 + rel4 + rel5 + rel6 + rel7)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0.05,test.input_tw_backward['in2'])
        # self.assertEqual(0.03,test.input_tw_forward['in2'])
        self.assertEqual(5, test.json["Inputs"]["in2"]["ns"][0])
        self.assertEqual(3, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(
            8, test.json["Inputs"]["in2"]["ntot"]
        )  # 5 samples + 3 samples of the horizon

        self.assertEqual(5, test.json["Info"]["ns"][0])
        self.assertEqual(3, test.json["Info"]["ns"][1])
        self.assertEqual(8, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_tw3(self):
        NeuObj.clearNames()
        input2 = Input("in2")
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.01, 0.03]))
        rel5 = Fir(input2.tw([-0.04, 0.01]))
        fun = Output("out", rel3 + rel4 + rel5)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0.05, test.input_tw_backward['in2'])
        # self.assertEqual(0.03, test.input_tw_forward['in2'])
        self.assertEqual(
            5,
            test.json["Inputs"]["in2"]["ns"][0],
        )
        self.assertEqual(3, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(
            8, test.json["Inputs"]["in2"]["ntot"]
        )  # 5 samples + 3 samples of the horizon

        self.assertEqual(5, test.json["Info"]["ns"][0])
        self.assertEqual(3, test.json["Info"]["ns"][1])
        self.assertEqual(8, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_tw_with_offest(self):
        NeuObj.clearNames()
        input2 = Input("in2")
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.04, 0.02]))
        rel5 = Fir(input2.tw([-0.04, 0.02], offset=-0.04))
        rel6 = Fir(input2.tw([-0.04, 0.02], offset=-0.01))
        rel7 = Fir(input2.tw([-0.04, 0.02], offset=0.01))
        fun = Output("out", rel3 + rel4 + rel5 + rel6 + rel7)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0.05, test.input_tw_backward['in2'])
        # self.assertEqual(0.02, test.input_tw_forward['in2'])
        self.assertEqual(5, test.json["Inputs"]["in2"]["ns"][0])
        self.assertEqual(2, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(
            7, test.json["Inputs"]["in2"]["ntot"]
        )  # 5 samples + 2 samples of the horizon

        self.assertEqual(5, test.json["Info"]["ns"][0])
        self.assertEqual(2, test.json["Info"]["ns"][1])
        self.assertEqual(7, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_tw_negative(self):
        NeuObj.clearNames()
        input2 = Input("in2")
        rel1 = Fir(input2.tw([-0.05, -0.01]))
        rel2 = Fir(input2.tw([-0.06, -0.03]))
        fun = Output("out", rel1 + rel2)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0.06,test.input_tw_backward['in2'])
        # self.assertEqual( -0.01, test.input_tw_forward['in2'])
        self.assertEqual(6, test.json["Inputs"]["in2"]["ns"][0])
        self.assertEqual(-1, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(
            5, test.json["Inputs"]["in2"]["ntot"]
        )  # 6 samples - 1 samples of the horizon

        self.assertEqual(6, test.json["Info"]["ns"][0])
        self.assertEqual(-1, test.json["Info"]["ns"][1])
        self.assertEqual(5, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_tw_negative_with_offset(self):
        NeuObj.clearNames()
        input2 = Input("in2")
        rel1 = Fir(input2.tw([-0.05, -0.01], offset=-0.05))
        rel2 = Fir(input2.tw([-0.02, -0.01], offset=-0.02))
        rel3 = Fir(input2.tw([-0.06, -0.03], offset=-0.06))
        rel4 = Fir(input2.tw([-0.06, -0.03], offset=-0.05))
        with self.assertRaises(ValueError):
            input2.tw([-0.01, -0.01], offset=-0.02)
        with self.assertRaises(IndexError):
            input2.tw([-0.06, -0.03], offset=-0.07)
        with self.assertRaises(IndexError):
            input2.tw([-0.06, -0.01], offset=-0.01)
        fun = Output("out", rel1 + rel2 + rel3 + rel4)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0.06,test.input_tw_backward['in2'])
        # self.assertEqual( -0.01, test.input_tw_forward['in2'])
        self.assertEqual(6, test.json["Inputs"]["in2"]["ns"][0])
        self.assertEqual(-1, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(
            5, test.json["Inputs"]["in2"]["ntot"]
        )  # 6 samples - 1 samples of the horizon

        self.assertEqual(6, test.json["Info"]["ns"][0])
        self.assertEqual(-1, test.json["Info"]["ns"][1])
        self.assertEqual(5, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_tw_positive(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        rel = Fir(input1.tw([0.03, 0.04]))
        fun = Output("out1", rel)
        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        input2 = Input("in2")
        rel1 = Fir(input2.tw([0.01, 0.04]))
        rel2 = Fir(input2.tw([0.03, 0.07]))
        fun = Output("out2", rel1 + rel2)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(-0.01, test.input_tw_backward['in2'])
        # self.assertEqual(0.07, test.input_tw_forward['in2'])
        self.assertEqual(-1, test.json["Inputs"]["in2"]["ns"][0])
        self.assertEqual(7, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(
            6, test.json["Inputs"]["in2"]["ntot"]
        )  # -1 samples + 6 samples of the horizon

        self.assertEqual(-1, test.json["Info"]["ns"][0])
        self.assertEqual(7, test.json["Info"]["ns"][1])
        self.assertEqual(6, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_tw_positive_with_offset(self):
        NeuObj.clearNames()
        input2 = Input("in2")
        rel1 = Fir(input2.tw([0.01, 0.04], offset=0.02))
        rel2 = Fir(input2.tw([0.03, 0.07], offset=0.04))
        with self.assertRaises(ValueError):
            input2.tw([0.03, 0.02])
        with self.assertRaises(IndexError):
            input2.tw([0.03, 0.07], offset=0.08)
        with self.assertRaises(IndexError):
            input2.tw([0.03, 0.07], offset=0)

        fun = Output("out", rel1 + rel2)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(-0.01,test.input_tw_backward['in2'])
        # self.assertEqual( 0.07, test.input_tw_forward['in2'])
        self.assertEqual(-1, test.json["Inputs"]["in2"]["ns"][0])
        self.assertEqual(7, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(
            6, test.json["Inputs"]["in2"]["ntot"]
        )  # 6 samples - 1 samples of the horizon

        self.assertEqual(-1, test.json["Info"]["ns"][0])
        self.assertEqual(7, test.json["Info"]["ns"][1])
        self.assertEqual(6, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_sw(self):
        NeuObj.clearNames()
        input1 = Input("in1")
        rel3 = Fir(input1.sw(2))
        rel4 = Fir(input1.sw([-2, 2]))
        rel5 = Fir(input1.sw([-3, 3]))
        rel6 = Fir(input1.sw([-3, 0]))
        rel7 = Fir(input1.sw(3))
        fun = Output("out", rel3 + rel4 + rel5 + rel6 + rel7)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0,test.input_tw_backward['in1'])
        # self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(3, test.json["Inputs"]["in1"]["ns"][0])
        self.assertEqual(3, test.json["Inputs"]["in1"]["ns"][1])
        self.assertEqual(
            6, test.json["Inputs"]["in1"]["ntot"]
        )  # 6 samples - 1 samples of the horizon

        self.assertEqual(3, test.json["Info"]["ns"][0])
        self.assertEqual(3, test.json["Info"]["ns"][1])
        self.assertEqual(6, test.json["Info"]["ntot"])  # 5 samples

    def test_network_building_sw_with_offset(self):
        NeuObj.clearNames()
        input2 = Input("in2")
        rel3 = Fir(input2.sw(5))
        rel4 = Fir(input2.sw([-4, 2]))
        rel5 = Fir(input2.sw([-4, 2], offset=0))
        rel6 = Fir(input2.sw([-4, 2], offset=1))
        rel7 = Fir(input2.sw([-2, 2], offset=1))
        rel8 = Fir(input2.sw([-4, 2], offset=-3))
        fun = Output("out", rel3 + rel4 + rel5 + rel6 + rel7 + rel8)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0, test.input_tw_backward['in2'])
        # self.assertEqual(0, test.input_tw_forward['in2'])
        self.assertEqual(5, test.json["Inputs"]["in2"]["ns"][0])
        self.assertEqual(2, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(7, test.json["Inputs"]["in2"]["ntot"])

        self.assertEqual(5, test.json["Info"]["ns"][0])
        self.assertEqual(2, test.json["Info"]["ns"][1])
        self.assertEqual(7, test.json["Info"]["ntot"])

    def test_network_building_sw_and_tw(self):
        NeuObj.clearNames()
        input2 = Input("in2")
        with self.assertRaises(TypeError):
            input2.sw(5) + input2.tw(0.05)

        rel1 = Fir(input2.sw([-4, 2])) + Fir(input2.tw([-0.01, 0]))
        fun = Output("out", rel1)

        test = Modely(visualizer=None)
        test.addModel("fun", fun)
        test.neuralizeModel(0.01)

        # self.assertEqual(0.01,test.input_tw_backward['in2'])
        # self.assertEqual(0,test.input_tw_forward['in2'])
        self.assertEqual(4, test.json["Inputs"]["in2"]["ns"][0])
        self.assertEqual(2, test.json["Inputs"]["in2"]["ns"][1])
        self.assertEqual(6, test.json["Inputs"]["in2"]["ntot"])

        self.assertEqual(4, test.json["Info"]["ns"][0])
        self.assertEqual(2, test.json["Info"]["ns"][1])
        self.assertEqual(6, test.json["Info"]["ntot"])

    def test_example_parametric_different_dim_input(self):
        NeuObj.clearNames()
        test = Modely(visualizer=None, seed=42)
        x = Input("x")
        y = Input("y")
        z = Input("z")

        ## create the relations
        def myFun(K1, p1, p2):
            return K1 * p1 * p2

        K_x = Parameter("k_x", dimensions=1, tw=1)
        K_y = Parameter("k_y", dimensions=1, tw=1)
        w = Parameter("w", dimensions=1, tw=1)
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

        out = Output("out", Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v)))
        out2 = Output("out2", Add(w, x.tw(1)) + Add(t, y.tw(1)) + Add(w, c))
        out3 = Output("out3", Add(fir_w, fir_t))
        out4 = Output("out4", Linear(output_dimension=1)(fuzzy))
        out5 = Output("out5", Fir(time_part) + Fir(sample_select))
        out6 = Output("out6", LocalModel(output_function=Fir())(x.tw(1), fuzzy))
        with self.assertRaises(TypeError):
            parfun_z(x.tw(5), t_5, c_5)
        out7 = Output(
            "out7",
            Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v))
            + Fir(parfun_z(x.tw(5), t_5, c_5_2)),
        )

        # parfun = ParamFun(myFun, map_over_batch=True)
        # p = Constant('co', values=[[2]])
        # with self.assertRaises(TypeError):
        #     Output('out-12', parfun(p, x.sw(4)))

        test.addModel("modelA", out)
        test.addModel("modelB", [out2, out3, out4])
        test.addModel("modelC", [out4, out5, out6])
        test.addModel("modelD", [out7])
        test.addMinimize("error1", x.last(), out)
        test.addMinimize("error2", y.last(), out3, loss_function="rmse")
        test.addMinimize("error3", z.last(), out6, loss_function="rmse")
        test.neuralizeModel(0.5)

        self.assertEqual([10, 0], test.json["Inputs"]["x"]["ns"])
        self.assertEqual([10, 0], test.json["Inputs"]["y"]["ns"])
        self.assertEqual([1, 0], test.json["Inputs"]["z"]["ns"])
        #
        # self.assertEqual(4,test.json['Info']['ns'][0])
        # self.assertEqual(2,test.json['Info']['ns'][1])
        # self.assertEqual(6,test.json['Info']['ntot'])

    def test_batch_size_and_step(self):
        NeuObj.clearNames()
        test = Modely(visualizer=None, seed=42, log_internal=True)
        x = Input("x")
        y = Input("y")

        rel_out = Fir(x.last()) + Fir(y.last())
        rel_out.closedLoop(y)
        out = Output("out", rel_out)

        test.addModel("modelA", out)
        test.addMinimize("error1", out, x.next())
        test.neuralizeModel()

        data_x = np.random.rand(101, 1)
        data_y = np.random.rand(101, 1)
        dataset = {"x": data_x, "y": data_y}
        test.loadData(name="dataset", source=dataset)

        ## 100 // (step+batch) = 2
        test.trainModel(
            train_dataset="dataset",
            num_of_epochs=1,
            train_batch_size=10,
            step=30,
            prediction_samples=20,
            shuffle_data=False,
        )
        self.assertEqual(2 * 21, len(test.internals.keys()))

        ## Clip the step to the maximum number of samples (100 - prediction_samples - batch) = 70
        test.trainModel(
            train_dataset="dataset",
            num_of_epochs=1,
            train_batch_size=10,
            step=200,
            prediction_samples=20,
            shuffle_data=True,
        )
        self.assertEqual(1 * 21, len(test.internals.keys()))

        ## Clip the step to 0
        test.trainModel(
            train_dataset="dataset",
            num_of_epochs=1,
            train_batch_size=10,
            step=-4,
            prediction_samples=20,
            shuffle_data=True,
        )
        self.assertEqual(8 * 21, len(test.internals.keys()))
