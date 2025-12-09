import unittest, sys, os, torch

import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj, Stream
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 11 Tests
# This file tests the dimensions and the of the element created in the pytorch environment

class ModelyNetworkBuildingTest(unittest.TestCase):

    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:
            self.assertEqual(len(data1),len(data2))
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_network_building_very_simple(self):
        NeuObj.clearNames()
        input1 = Input('in1').last()
        rel1 = Fir(input1)
        fun = Output('out', rel1)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[1, 1], [1, 1]]
        for ind, (key, value) in enumerate({k: v for k, v in test._model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))
      
    def test_network_building_simple(self):
        NeuObj.clearNames()
        Stream.resetCount()
        input1 = Input('in1')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output('out',rel1+rel2)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)
        
        list_of_dimensions = {'Fir2':[5,1],'Fir5':[1,1]}
        for key, value in {k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw(self):
        NeuObj.clearNames()
        Stream.resetCount()
        input1 = Input('in1')
        input2 = Input('in2')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        fun = Output('out',rel1+rel2+rel3+rel4)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)
        
        list_of_dimensions = {'Fir2':[5,1], 'Fir5':[1,1], 'Fir8':[5,1], 'Fir11':[4,1]}
        for key, value in {k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))
    
    def test_network_building_tw2(self):
        Stream.resetCount()
        NeuObj.clearNames()
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        rel5 = Fir(input2.tw([-0.03,0.03]))
        rel6 = Fir(input2.tw([-0.03, 0]))
        rel7 = Fir(input2.tw(0.03))
        fun = Output('out',rel3+rel4+rel5+rel6+rel7)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test._max_n_samples, 8) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 8} , test._input_n_samples)
        
        list_of_dimensions = {'Fir2':[5,1], 'Fir5':[4,1], 'Fir8':[6,1], 'Fir11':[3,1], 'Fir14':[3,1]}
        for  key, value in {k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw3(self):
        NeuObj.clearNames()
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.01,0.03]))
        rel5 = Fir(input2.tw([-0.04,0.01]))
        fun = Output('out',rel3+rel4+rel5)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[5,1], [4,1], [5,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_tw_with_offest(self):
        NeuObj.clearNames()
        Stream.resetCount()
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.04,0.02]))
        rel5 = Fir(input2.tw([-0.04,0.02],offset=0))
        rel6 = Fir(input2.tw([-0.04,0.02],offset=0.01))
        fun = Output('out',rel3+rel4+rel5+rel6)


        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = {'Fir2':[5,1], 'Fir5':[6,1], 'Fir8':[6,1], 'Fir11':[6,1]}
        for key, value in {k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw_negative(self):
        NeuObj.clearNames()
        input2 = Input('in2')
        rel1 = Fir(input2.tw([-0.04,-0.01]))
        rel2 = Fir(input2.tw([-0.06,-0.03]))
        fun = Output('out',rel1+rel2)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_tw_positive(self):
        NeuObj.clearNames()
        input2 = Input('in2')
        rel1 = Fir(input2.tw([0.01,0.04]))
        rel2 = Fir(input2.tw([0.03,0.06]))
        fun = Output('out',rel1+rel2)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_sw_with_offset(self):
        Stream.resetCount()
        NeuObj.clearNames()
        input2 = Input('in2')
        rel3 = Fir(input2.sw(5))
        rel4 = Fir(input2.sw([-4,2]))
        rel5 = Fir(input2.sw([-4,2],offset=0))
        rel6 = Fir(input2.sw([-4,2],offset=1))
        fun = Output('out',rel3+rel4+rel5+rel6)

        test = Modely(visualizer=None, seed=1)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = {'Fir2':[5,1], 'Fir5':[6,1], 'Fir8':[6,1], 'Fir11':[6,1]}
        for key, value in {k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_sw_and_tw(self):
        NeuObj.clearNames()
        input2 = Input('in2')
        with self.assertRaises(TypeError):
            input2.sw(5)+input2.tw(0.05)

        rel1 = Fir(input2.sw([-4,2]))+Fir(input2.tw([-0.01,0]))
        fun = Output('out',rel1)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[6,1], [1,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test._model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_linear(self):
        NeuObj.clearNames()
        input = Input('in1')
        rel1 = Linear(input.sw([-4,2]))
        rel2 = Linear(5)(input.sw([-1, 2]))
        fun1 = Output('out1',rel1)
        fun2 = Output('out2', rel2)

        input5 = Input('in5', dimensions=3)
        rel15 = Linear(input5.sw([-4,2]))
        rel25 = Linear(5)(input5.last())
        fun15 = Output('out51',rel15)
        fun25 = Output('out52', rel25)

        test = Modely(seed =1, visualizer=None)
        test.addModel('fun',[fun1,fun2,fun15,fun25])
        test.neuralizeModel(0.01)

        list_of_dimensions = [[1,1],[1,5],[3,1],[3,5]]
        for ind, (key, value) in enumerate({k:v for k,v in test._model.relation_forward.items() if 'Linear' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_linear_interpolation_train(self):
        NeuObj.clearNames()
        x = Input('x')
        param = Parameter(name='a', sw=1)
        rel1 = Fir(W=param)(Interpolation(x_points=[1.0, 2.0, 3.0, 4.0],y_points=[2.0, 4.0, 6.0, 8.0], mode='linear')(x.last()))
        out = Output('out',rel1)

        test = Modely(seed = 1, visualizer=None)
        test.addModel('fun',[out])
        test.addMinimize('error', out, x.last())
        test.neuralizeModel(0.01)

        dataset = {'x':np.random.uniform(1,4,100)}
        test.loadData(name='dataset', source=dataset)
        test.trainModel(num_of_epochs=100, train_batch_size=10)
        self.assertAlmostEqual(test.parameters['a'][0][0], 0.5, places=2)

    def test_network_linear_interpolation(self):
        NeuObj.clearNames()

        x = Input('x')
        rel1 = Interpolation(x_points=[1.0, 2.0, 3.0, 4.0],y_points=[1.0, 4.0, 9.0, 16.0], mode='linear')(x.last())
        out = Output('out',rel1)

        test = Modely(seed=1,visualizer=None)
        test.addModel('fun',[out])
        test.neuralizeModel(0.01)

        inference = test(inputs={'x':[1.5,2.5,3.5]})
        self.assertEqual(inference['out'],[2.5,6.5,12.5])

        x1 = Input('x1')
        rel1 = Interpolation(x_points=[1.0, 4.0, 3.0, 2.0],y_points=[1.0, 16.0, 9.0, 4.0], mode='linear')(x1.last())
        out = Output('out1',rel1)

        test = Modely(visualizer=None)
        test.addModel('fun',[out])
        test.neuralizeModel(0.01)

        inference = test(inputs={'x1':[1.5,2.5,3.5]})
        self.assertEqual(inference['out1'],[2.5,6.5,12.5])

    def test_softmax_and_sigmoid(self):
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y', dimensions=3)
        softmax = Softmax(y.last())
        sigmoid = Sigmoid(x.last())
        out = Output('softmax',softmax)
        out2 = Output('sigmoid',sigmoid)

        test = Modely(visualizer=None)
        test.addModel('model',[out,out2])
        test.neuralizeModel(0.01)

        inference = test(inputs={'x':[-1000.0, 0.0, 1000.0], 'y':[[-1.0,0.0,1.0],[-1000.0,0.0,1000.0],[1.0,2.0,3.0]]})
        self.assertEqual(inference['sigmoid'],[0.0, 0.5, 1.0])
        self.assertEqual(inference['softmax'],[[[0.09003057330846786, 0.2447284758090973, 0.6652409434318542]],
                                               [[0.0, 0.0, 1.0]],
                                               [[0.09003057330846786, 0.2447284758090973, 0.6652409434318542]]])

    def test_sech_cosh_function(self):
        torch.manual_seed(1)
        input = Input('in1')
        sech_rel = Sech(input.last())
        sech_rel_2 = Sech(input.sw(2))
        cosh_rel = Cosh(input.last())
        cosh_rel_2 = Cosh(input.sw(2))

        input5 = Input('in5', dimensions=5)
        sech_rel_5 = Sech(input5.last())
        cosh_rel_5 = Cosh(input5.last())

        out1 = Output('sech_out_1', sech_rel)
        out2 = Output('sech_out_2', sech_rel_2)
        out3 = Output('sech_out_3', sech_rel_5)
        out4 = Output('cosh_out_1', cosh_rel)
        out5 = Output('cosh_out_2', cosh_rel_2)
        out6 = Output('cosh_out_3', cosh_rel_5)

        test = Modely(visualizer=None)
        test.addModel('model',[out1,out2,out3,out4,out5,out6])
        test.neuralizeModel(0.01)

        result = test(inputs={'in1':[[3.0],[-2.0]], 'in5':[[4.0,1.0,0.0,-6.0,2.0]]})
        self.TestAlmostEqual([0.2658022344112396], result['sech_out_1'])
        self.TestAlmostEqual([[0.0993279218673706, 0.2658022344112396]], result['sech_out_2'])
        self.TestAlmostEqual([[[0.03661899268627167, 0.6480542421340942, 1.0, 0.004957473836839199, 0.2658022344112396]]], result['sech_out_3'])
        self.TestAlmostEqual([3.762195587158203], result['cosh_out_1'])
        self.TestAlmostEqual([[10.067662239074707, 3.762195587158203]], result['cosh_out_2'])
        self.TestAlmostEqual([[[27.3082332611084, 1.5430806875228882, 1.0, 201.71563720703125, 3.762195587158203]]], result['cosh_out_3'])

    def test_concatenate_time_concatenate(self):
        NeuObj.clearNames()
        input = Input('in1')
        input2 = Input('in2')
        concatenate_rel = Concatenate(input.last(),input2.last())
        timeconcatenate_rel = TimeConcatenate(input.last(),input2.last())
        concatenate_tw_rel = Concatenate(input.tw(3),input2.tw(3))
        timeconcatenate_tw_rel = TimeConcatenate(input.tw(3),input2.tw(3))

        input3 = Input('in3', dimensions=5)
        input4 = Input('in4', dimensions=5)

        concatenate_rel_5 = Concatenate(input3.last(),input4.last())
        timeconcatenate_rel_5 = TimeConcatenate(input3.last(),input4.last())
        concatenate_tw_rel_5 = Concatenate(input3.tw(3),input4.tw(3))
        timeconcatenate_tw_rel_5 = TimeConcatenate(input3.tw(3),input4.tw(3))

        out1 = Output('concatenate', concatenate_rel)
        out2 = Output('time_concatenate', timeconcatenate_rel)
        out3 = Output('concatenate_tw', concatenate_tw_rel)
        out4 = Output('time_concatenate_tw', timeconcatenate_tw_rel)
        out5 = Output('concatenate_5', concatenate_rel_5)
        out6 = Output('time_concatenate_5', timeconcatenate_rel_5)
        out7 = Output('concatenate_tw_5', concatenate_tw_rel_5)
        out8 = Output('time_concatenate_tw_5', timeconcatenate_tw_rel_5)

        test = Modely(seed=1,visualizer=None)
        test.addModel('model',[out1,out2,out3,out4,out5,out6,out7,out8])
        test.neuralizeModel(1)

        result = test(inputs={'in1':[[1.0],[2.0],[3.0]], 'in2':[[4.0],[5.0],[6.0]],
                              'in3':[[7.0,8.0,9.0,10.0,11.0],[12.0,13.0,14.0,15.0,16.0],[17.0,18.0,19.0,20.0,21.0]],
                              'in4':[[22.0,23.0,24.0,25.0,26.0],[27.0,28.0,29.0,30.0,31.0],[32.0,33.0,34.0,35.0,36.0]]})
        self.assertEqual((1,1,2), np.array(result['concatenate']).shape)
        self.assertEqual([[[3.0, 6.0]]], result['concatenate'])
        self.assertEqual((1,2), np.array(result['time_concatenate']).shape)
        self.assertEqual([[3.0, 6.0]], result['time_concatenate'])
        self.assertEqual((1,3,2), np.array(result['concatenate_tw']).shape)
        self.assertEqual([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]], result['concatenate_tw'])
        self.assertEqual((1,6), np.array(result['time_concatenate_tw']).shape)
        self.assertEqual([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], result['time_concatenate_tw'])
        self.assertEqual((1,1,10), np.array(result['concatenate_5']).shape)
        self.assertEqual([[[17.0, 18.0, 19.0, 20.0, 21.0, 32.0, 33.0, 34.0, 35.0, 36.0]]], result['concatenate_5'])
        self.assertEqual((1,2,5), np.array(result['time_concatenate_5']).shape)
        self.assertEqual([[[17.0, 18.0, 19.0, 20.0, 21.0], [32.0, 33.0, 34.0, 35.0, 36.0]]], result['time_concatenate_5'])
        self.assertEqual((1,3,10), np.array(result['concatenate_tw_5']).shape)
        self.assertEqual([[[7.0, 8.0, 9.0, 10.0, 11.0, 22.0, 23.0, 24.0, 25.0, 26.0],
                           [12.0, 13.0, 14.0, 15.0, 16.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                           [17.0, 18.0, 19.0, 20.0, 21.0, 32.0, 33.0, 34.0, 35.0, 36.0]]], result['concatenate_tw_5'])
        self.assertEqual((1,6,5), np.array(result['time_concatenate_tw_5']).shape)
        self.assertEqual([[[7.0, 8.0, 9.0, 10.0, 11.0],
                           [12.0, 13.0, 14.0, 15.0, 16.0],
                           [17.0, 18.0, 19.0, 20.0, 21.0],
                           [22.0, 23.0, 24.0, 25.0, 26.0],
                           [27.0, 28.0, 29.0, 30.0, 31.0],
                           [32.0, 33.0, 34.0, 35.0, 36.0]]], result['time_concatenate_tw_5'])

    def test_equation_learner(self):
        NeuObj.clearNames()
        x = Input('x')
        F = Input('F')

        def myFun(p1,p2,k1,k2):
            return k1*p1+k2*p2

        K1 = Parameter('k1', dimensions =  1, sw = 1,values=[[2.0]])
        K2 = Parameter('k2', dimensions =  1, sw = 1,values=[[3.0]])
        parfun = ParamFun(myFun, parameters_and_constants=[K1,K2])
        parfun2 = ParamFun(myFun, parameters_and_constants=[K1,K2])
        parfun3 = ParamFun(myFun, parameters_and_constants=[K1,K2])
        fuzzi = Fuzzify(centers=[0,1,2,3])

        linear_layer_in = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)
        linear_layer_in_dim_2 = Linear(output_dimension=3, W_init=init_constant, W_init_params={'value':0}, b_init=init_constant, b_init_params={'value':0}, b=False)
        linear_layer_out = Linear(output_dimension=1, W_init=init_constant, W_init_params={'value':1}, b_init=init_constant, b_init_params={'value':0}, b=False)

        equation_learner = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer_in)(x.last())
        equation_learner_out = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer_in, linear_out=linear_layer_out)(x.last())
        equation_learner_tw = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer_in)(x.sw(2))
        equation_learner_multi_tw = EquationLearner(functions=[Tan, Sin, Cos], linear_in=linear_layer_in_dim_2)((x.sw(2),F.sw(2)))

        linear_layer_in_2 = Linear(output_dimension=5, W_init=init_constant, W_init_params={'value':1})
        linear_layer_in_2_dim_2 = Linear(output_dimension=5, W_init=init_constant, W_init_params={'value':1})

        equation_learner_2 = EquationLearner(functions=[Add, Mul, Identity], linear_in=linear_layer_in_2)(x.last())
        equation_learner_2_tw = EquationLearner(functions=[Add, Mul, Identity], linear_in=linear_layer_in_2)(x.sw(2))
        equation_learner_2_multi_tw = EquationLearner(functions=[Add, Mul, Identity], linear_in=linear_layer_in_2_dim_2)((x.sw(2),F.sw(2)))

        linear_layer_in_3 = Linear(output_dimension=5, W_init=init_constant, W_init_params={'value':1}, b_init=init_constant, b_init_params={'value':0}, b=False)
        linear_layer_in_3_dim_2 = Linear(output_dimension=5, W_init=init_constant, W_init_params={'value':1}, b_init=init_constant, b_init_params={'value':0}, b=False)

        equation_learner_3 = EquationLearner(functions=[parfun, Add, fuzzi], linear_in=linear_layer_in_3)(x.last())
        equation_learner_3_tw = EquationLearner(functions=[parfun2, Add, fuzzi], linear_in=linear_layer_in_3)(x.sw(2))
        equation_learner_3_multi_tw = EquationLearner(functions=[parfun3, Add, fuzzi], linear_in=linear_layer_in_3_dim_2)((x.sw(2),F.sw(2)))

        out = Output('el',equation_learner)
        out2 = Output('el_out',equation_learner_out)
        out3 = Output('el_tw',equation_learner_tw)
        out4 = Output('el_multi_tw',equation_learner_multi_tw)
        out5 = Output('el2',equation_learner_2)
        out6 = Output('el2_tw',equation_learner_2_tw)
        out7 = Output('el2_multi_tw',equation_learner_2_multi_tw)
        out8 = Output('el3',equation_learner_3)
        out9 = Output('el3_tw',equation_learner_3_tw)
        out10 = Output('el3_multi_tw',equation_learner_3_multi_tw)
        example = Modely(visualizer=None)
        example.addModel('model',[out,out2,out3,out4,out5,out6,out7,out8,out9,out10])
        example.neuralizeModel()

        result = example({'x':[[1.0],[2.0]], 'F':[[3.0],[4.0]]})
        self.assertEqual(result['el'], [[[0.0, 0.0, 1.0]]])
        self.assertEqual(result['el_out'], [1.0])
        self.assertEqual(result['el_tw'], [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        self.assertEqual(result['el_multi_tw'], [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        self.assertEqual(result['el2'], [[[4.0, 4.0, 2.0]]])
        self.assertEqual(result['el2_tw'], [[[2.0, 1.0, 1.0], [4.0, 4.0, 2.0]]])
        self.assertEqual(result['el2_multi_tw'], [[[8.0, 16.0, 4.0], [12.0, 36.0, 6.0]]])
        self.assertEqual(result['el3'], [[[10.0, 4.0, 0.0, 0.0, 1.0, 0.0]]])
        self.assertEqual(result['el3_tw'], [[[5.0, 2.0, 0.0, 1.0, 0.0, 0.0], [10.0, 4.0, 0.0, 0.0, 1.0, 0.0]]])
        self.assertEqual(result['el3_multi_tw'], [[[20.0, 8.0, 0.0, 0.0, 0.0, 1.0], [30.0, 12.0, 0.0, 0.0, 0.0, 1.0]]])

    def test_localmodel(self):
        NeuObj.clearNames()
        x = Input('x')
        activationA = Fuzzify(2, [0, 1], functions='Triangular')(x.last())
        loc = LocalModel(input_function=Fir())(x.tw(1), activationA)
        out = Output('out', loc)
        example = Modely(visualizer=None,seed=5)
        example.addModel('out', out)
        example.neuralizeModel(0.25)
        # The output is 2 samples
        self.assertEqual({'out': [1.7170718908309937, 1.9410502910614014]}, example({'x': [-1, 0, 1, 2, 0]}))
        self.assertEqual({'out': [1.7170718908309937, 1.9410502910614014]}, example({'x': [[-1, 0, 1, 2], [0, 1, 2, 0]]}, sampled=True))

    def test_arithmetic(self):
        NeuObj.clearNames()
        y = Input('y', dimensions=10)

        k = Parameter('k', dimensions=1)
        rel1 = y.last() + (5 * y.last())
        rel2 = y.last() - (5 * y.last())
        rel3 = y.last() + (k * y.last())
        rel4 = y.last() - (k * y.last())
        rel5 = y.last() * (5 * y.last())
        rel6 = y.last() / (5 * y.last())
        rel7 = y.last() * (k * y.last())
        rel8 = y.last() / (k * y.last())
        rel9 = Sum(y.last())

        out = Output('out', rel1)
        out2 = Output('out2', rel2)
        out3 = Output('out3', rel3)
        out4 = Output('out4', rel4)
        out5 = Output('out5', rel5)
        out6 = Output('out6', rel6)
        out7 = Output('out7', rel7)
        out8 = Output('out8', rel8)
        out9 = Output('out9', rel9)
        example = Modely(visualizer=None, seed=42)
        example.addModel('out', [out,out2,out3,out4,out5,out6,out7,out8,out9])
        example.neuralizeModel(0.25)

        self.assertEqual(rel1.dim['dim'], 10)
        self.assertEqual(rel2.dim['dim'], 10)
        self.assertEqual(rel3.dim['dim'], 10)
        self.assertEqual(rel4.dim['dim'], 10)
        self.assertEqual(rel5.dim['dim'], 10)
        self.assertEqual(rel6.dim['dim'], 10)
        self.assertEqual(rel7.dim['dim'], 10)
        self.assertEqual(rel8.dim['dim'], 10)
        self.assertEqual(rel9.dim['dim'], 1)

    def test_rungekutta(self):
        NeuObj.clearNames()
        x = Input('x')

        def fun(x):
            return x

        fe_rel = ForwardEuler(f=fun)(x.last())
        rk2_rel = RK2(f=fun)(x.last())
        rk4_rel = RK4(f=fun)(x.last())

        out_fe = Output('fe', fe_rel)
        out_rk2 = Output('rk2', rk2_rel)
        out_rk4 = Output('rk4', rk4_rel)

        model = Modely(visualizer=None)
        model.addModel('model', [out_fe, out_rk2, out_rk4])
        model.neuralizeModel(1)

        inputs = {'x': [[1.0], [2.0]]}
        result = model(inputs=inputs)
        # expected: fe -> [2.0, 4.0], rk2 -> [2.5, 5.0]
        self.TestAlmostEqual([2.0, 4.0], result['fe'])
        self.TestAlmostEqual([2.5, 5.0], result['rk2'])
        self.TestAlmostEqual([2.708333, 5.41666], result['rk4'])

        # now test with step h = 0.1
        model.neuralizeModel(0.1, clear_model=True)
        result = model(inputs=inputs)
        self.TestAlmostEqual([1.1, 2.2], result['fe'])
        self.TestAlmostEqual([1.105, 2.21], result['rk2'])
        self.TestAlmostEqual([1.10517, 2.21034], result['rk4'])