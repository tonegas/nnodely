import unittest, os, sys
import numpy as np
from pygments.unistring import xid_start

from nnodely import *
from nnodely.basic.relation import NeuObj
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 15 Tests
# Test the value of the weight after the recurrent training

# Linear function
def linear_fun(x,a,b):
    return x*a+b

data_x = np.random.rand(500)*20-10
data_a = 2
data_b = -3
dataset = {'in1': data_x, 'out': linear_fun(data_x,data_a,data_b)}
data_folder = '/tests/_data/'

class ModelyTrainingTest(unittest.TestCase):
    def assertAlmostEqual(self, data1, data2, precision=3):
        if type(data1) == type(data2) == list:
            assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2,dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
            self.assertEqual(len(data1), len(data2))
            for pred, label in zip(data1, data2):
                self.assertAlmostEqual(pred, label, precision=precision)
        elif type(data1) == type(data2) == dict:
            self.assertEqual(len(data1.items()), len(data2.items()))
            for (pred_key,pred_value), (label_key,label_value) in zip(data1.items(), data2.items()):
                self.assertAlmostEqual(pred_value, label_value, precision=precision)
        else:
            super().assertAlmostEqual(data1, data2, places=precision)

    def test_recurrent_shuffle(self):
        NeuObj.clearNames()
        target = Input('target')
        x = Input('x')
        relation = Fir(x.last())
        relation.closedLoop(x)
        output = Output('out', relation)

        test = Modely(visualizer=None, seed=42, log_internal=True)
        test.addModel('model', output)
        test.addMinimize('out', target.next(), 'out')
        test.neuralizeModel(0.01)

        dataset = {'x': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 'target': [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}
        test.loadData(name='dataset', source=dataset)

        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=0.01, num_of_epochs=1, train_batch_size=4, prediction_samples=1, step=1, shuffle_data=True)
        self.assertListEqual([[[4.0]], [[1.0]], [[9.0]], [[18.0]]], test.internals['inout_0_0']['XY']['x'])
        self.assertListEqual([[[25.0]], [[22.0]], [[30.0]], [[39.0]]], test.internals['inout_0_0']['XY']['target'])
        self.assertListEqual([[[26.0]], [[23.0]], [[31.0]], [[40.0]]], test.internals['inout_0_1']['XY']['target'])
        self.assertListEqual([[[5.0]], [[16.0]], [[3.0]], [[13.0]]], test.internals['inout_1_0']['XY']['x'])
        self.assertListEqual([[[26.0]], [[37.0]], [[24.0]], [[34.0]]], test.internals['inout_1_0']['XY']['target'])
        self.assertListEqual([[[27.0]], [[38.0]], [[25.0]], [[35.0]]], test.internals['inout_1_1']['XY']['target'])
        self.assertListEqual([[[15.0]], [[2.0]], [[17.0]], [[14.0]]], test.internals['inout_2_0']['XY']['x'])
        self.assertListEqual([[[36.0]], [[23.0]], [[38.0]], [[35.0]]], test.internals['inout_2_0']['XY']['target'])
        self.assertListEqual([[[37.0]], [[24.0]], [[39.0]], [[36.0]]], test.internals['inout_2_1']['XY']['target'])

        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=0.01, num_of_epochs=1, train_batch_size=2,
                       prediction_samples=2, step=0, shuffle_data=True)
        # ( number_samples - window_size - prediction_samples )// (batch_size + step=0) * (predictoin_samples+1)
        self.assertEqual((20-1-2)//2*3, len(test.internals.keys()))
        with self.assertRaises(ValueError):
            test.trainModel(dataset='dataset', splits=[40,30,30], optimizer='SGD', lr=0.01, num_of_epochs=1, train_batch_size=2,
                            prediction_samples=50, step=0, shuffle_data=True)
        with self.assertRaises(ValueError):
            test.trainModel(dataset='dataset', splits=[40,10,40], optimizer='SGD', lr=0.01, num_of_epochs=1, train_batch_size=2,
                            prediction_samples=50, step=0, shuffle_data=True)

        from nnodely.support.earlystopping import early_stop_patience, select_best_model
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=0.01, num_of_epochs=15,
                        train_batch_size=2,  early_stopping=early_stop_patience, early_stopping_params={'patience':2}, select_model=select_best_model,
                        prediction_samples=2, step=0, shuffle_data=True)

    def test_train_multifiles(self):
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y')
        relation = Fir()(x.tw(0.05))+Fir(y.sw([-2,2]))
        relation.closedLoop(x)
        output = Output('out', relation)

        test = Modely(visualizer=None, log_internal=True)
        test.addModel('model', output)
        test.addMinimize('error', 'out', x.next())
        test.neuralizeModel(0.01)

        ## The folder contains 3 files with 10, 20 and 30 samples respectively
        data_struct = ['x', 'y']
        data_folder = os.path.join(os.path.dirname(__file__), 'multifile/')
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1)
        self.assertEqual(len(test._data['dataset']['x']), 42)
        self.assertEqual(len(test._data['dataset']['y']), 42)

        test.trainModel(splits=[70, 20, 10], train_batch_size = 3, num_of_epochs=1, prediction_samples=2)
        self.assertEqual(len(list(test.internals.keys())), 3*7)
        self.assertEqual(list(np.mean(np.array(test.internals['inout_0_0']['XY']['y']),axis=1)),
                         list(np.mean(np.array(test.internals['inout_0_1']['XY']['y']),axis=1)))
        self.assertEqual(list(np.mean(np.array(test.internals['inout_0_0']['XY']['y']), axis=1)),
                         list(np.mean(np.array(test.internals['inout_0_2']['XY']['y']), axis=1)))
        self.assertEqual(list(np.mean(np.array(test.internals['inout_1_0']['XY']['y']),axis=1)),
                         list(np.mean(np.array(test.internals['inout_1_1']['XY']['y']),axis=1)))
        self.assertEqual(list(np.mean(np.array(test.internals['inout_1_0']['XY']['y']), axis=1)),
                         list(np.mean(np.array(test.internals['inout_1_2']['XY']['y']), axis=1)))
        self.assertEqual(list(np.mean(np.array(test.internals['inout_2_0']['XY']['y']),axis=1)),
                         list(np.mean(np.array(test.internals['inout_2_1']['XY']['y']),axis=1)))
        self.assertEqual(list(np.mean(np.array(test.internals['inout_2_0']['XY']['y']), axis=1)),
                         list(np.mean(np.array(test.internals['inout_2_2']['XY']['y']), axis=1)))
        self.assertEqual(list(np.mean(np.array(test.internals['inout_6_0']['XY']['y']),axis=1)),
                         list(np.mean(np.array(test.internals['inout_6_1']['XY']['y']),axis=1)))
        self.assertEqual(list(np.mean(np.array(test.internals['inout_6_0']['XY']['y']), axis=1)),
                         list(np.mean(np.array(test.internals['inout_6_2']['XY']['y']), axis=1)))

    def test_training_values_fir_connect_linear(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target = Input('target1')
        a = Parameter('a', sw=1, values=[[1]])
        relation = Fir(W=a)(input1.last())

        inout = Input('inout')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=[1])

        relation.connect(inout)
        output1 = Output('out1', relation)
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}))

        dataset = {'in1': [1], 'target1': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertListEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertListEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2)
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertListEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

        dataset = {'in1': [1,1], 'target1': [3,3]}
        test.loadData(name='dataset2', source=dataset)
        test.neuralizeModel(clear_model=True)
        # the out is 3.0 due the mean of the error is not the same of two epochs
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2)
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertListEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

    def test_training_values_fir_train_connect_linear(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', sw=1, values=[[1]])
        output1 = Output('out1-net',Fir(W=a)(input1.last()))

        inout = Input('inout')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=1)
        output2 = Output('out2-net', Linear(W=W,b=b)(inout.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1-net': [1.0], 'out2-net': [2.0]}, test({'in1': [1]}, connect={'inout': 'out1-net'}))

        dataset = {'in1': [1], 'out1': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1-net'})
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertEqual([3.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1-net'})
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1-net'})
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertEqual([3.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1-net'})
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2, connect={'inout': 'out1-net'})
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

        with self.assertRaises(KeyError):
            test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2)

        dataset = {'in1': [1,1], 'out1': [3,3]}
        test.loadData(name='dataset2', source=dataset)
        test.neuralizeModel(clear_model=True)
        # the out is 3.0 due the mean of the error is not the same of two epochs
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, connect={'inout': 'out1-net'})
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertEqual([3.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2, connect={'inout': 'out1-net'})
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

    def test_training_values_fir_connect_linear_only_model(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target = Input('target1')
        a = Parameter('a', sw=1, values=[[1]])
        output1 = Output('out1',Fir(W=a)(input1.last()))

        inout = Input('inout')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=[1])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addModel('model2', output2)
        test.addMinimize('error', target.last(), output2)
        test.addConnect(output1, inout)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}))

        dataset = {'in1': [1], 'target1': [3]}
        test.loadData(name='dataset', source=dataset)

        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertListEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        test.trainModel(models='model1', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.trainModel(models='model1', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        test.trainModel(models='model2', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(models='model2', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertListEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

    def test_training_values_fir_train_connect_linear_only_model(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target = Input('target1')
        a = Parameter('a', sw=1, values=[[1]])
        output1 = Output('out1',Fir(W=a)(input1.last()))

        inout = Input('inout')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=[1])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addModel('model2', output2)
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [1.0]}, test({'in1': [1]}))
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}, connect={'inout': 'out1'}))

        dataset = {'in1': [1], 'target1': [3]}
        test.loadData(name='dataset', source=dataset)

        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[-51.0]], test.parameters['W'])
        self.assertListEqual([-15.0], test.parameters['b'])
        self.assertListEqual([[-51.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        test.trainModel(models='model1', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[3.0]], test.parameters['a'])
        test.trainModel(models='model1', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        test.trainModel(models='model2', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(models='model2', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertListEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

    def test_training_values_fir_connect_linear_more_prediction(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', sw=1, values=[[1]])
        relation =Fir(W=a)(input1.last())

        inout = Input('inout')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=1)

        relation.connect(inout)

        output1 = Output('out1-net', relation)
        output2 = Output('out2-net', Linear(W=W,b=b)(inout.last()))

        test = Modely(visualizer=None,seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1-net': [1.0], 'out2-net': [2.0]}, test({'in1': [1]}))

        dataset = {'in1': [0,2,7,1], 'out1': [3,4,5,1], 'inout': [1,1,2,2]}
        test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[-9.0]], test.parameters['W'])
        self.assertEqual([0.5], test.parameters['b'])
        self.assertListEqual([[-9.0]], test.parameters['a'])
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[-9.0]], test.parameters['W'])
        self.assertEqual([0.5], test.parameters['b'])
        self.assertListEqual([[-9.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        #TODO add this test for check prediction_Sample -1 for connect
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=-1)
        # self.assertListEqual([[-9.0]], test.parameters['W']) ?
        # self.assertEqual([0.5], test.parameters['b']) ?
        # self.assertListEqual([[-9.0]], test.parameters['a']) ?

        dataset = {'in1': [0, 2, 7, 1, 5, 0, 2], 'out1': [1, 4, 8, 2, 6, 1, 1]}
        test.loadData(name='dataset3', source=dataset)

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset3', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=2, prediction_samples=3)
        self.assertListEqual([[-162.75]], test.parameters['W'])
        self.assertEqual([-15.75], test.parameters['b'])
        self.assertListEqual([[-162.75]], test.parameters['a'])

        # Because is a connect and the window is 1 the initialization of the state is overwritten by the out1
        test.loadData(name='dataset4', source=dataset|{'inout': [0,0,0,0,0,0,0]})
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset4', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=2, prediction_samples=3)
        self.assertListEqual([[-162.75]], test.parameters['W'])
        self.assertEqual([-15.75], test.parameters['b'])
        self.assertListEqual([[-162.75]], test.parameters['a'])

    def test_training_values_fir_train_connect_linear_more_prediction(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target = Input('target1')
        a = Parameter('a', sw=1, values=[[1]])
        output1 = Output('out1',Fir(W=a)(input1.last()))

        inout = Input('inout')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=[1])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]},connect={'inout': 'out1'}))

        dataset = {'in1': [0,2,7,1], 'target1': [3,4,5,1]}
        test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, connect={'inout': 'out1'})
        self.assertListEqual([[-9.0]], test.parameters['W'])
        self.assertListEqual([0.5], test.parameters['b'])
        self.assertListEqual([[-9.0]], test.parameters['a'])
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10, connect={'inout': 'out1'})
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, connect={'inout': 'out1'}) # TODO add this test
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[-9.0]], test.parameters['W'])
        self.assertListEqual([0.5], test.parameters['b'])
        self.assertListEqual([[-9.0]], test.parameters['a'])

        # Because is a connect and the window is 1 the initialization of the state is overwritten by the out1
        dataset = {'in1': [0,2,7,1], 'target1': [3,4,5,1], 'inout':[1,1,2,2]}
        test.loadData(name='dataset3', source=dataset)
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[-9.0]], test.parameters['W'])
        self.assertListEqual([0.5], test.parameters['b'])
        self.assertListEqual([[-9.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[-273.0]], test.parameters['W'])
        self.assertListEqual([-137.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=-1)
        self.assertListEqual([[-273.0]], test.parameters['W'])
        self.assertListEqual([-137.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=1)
        self.assertListEqual([[-273.0]], test.parameters['W'])
        self.assertListEqual([-137.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        dataset = {'in1': [0, 2, 7, 1, 5, 0, 2], 'target1': [1, 4, 8, 2, 6, 1, 1]}
        test.loadData(name='dataset4', source=dataset)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset4', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1,
                        train_batch_size=2, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[-162.75]], test.parameters['W'])
        self.assertListEqual([-15.75], test.parameters['b'])
        self.assertListEqual([[-162.75]], test.parameters['a'])

        # Because is a connect and the window is 1 the initialization of the state is overwritten by the out1
        test.loadData(name='dataset5', source=dataset | {'inout': [0, 0, 0, 0, 0, 0, 0]})
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset5', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1,
                        train_batch_size=2, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[-162.75]], test.parameters['W'])
        self.assertListEqual([-15.75], test.parameters['b'])
        self.assertListEqual([[-162.75]], test.parameters['a'])

    def test_training_values_fir_connect_linear_more_window(self):
        NeuObj.clearNames()
        input1 = Input('in1', dimensions=2)
        W = Parameter('W', values=[[-1], [-5]])
        b = Parameter('b', values=1)
        lin_out = Linear(W=W, b=b)(input1.sw(2))

        inout = Input('inout')
        a = Parameter('a', sw=2, values=[[4], [5]])
        a_big = Parameter('ab', sw=5, values=[[1], [2], [3], [4], [5]])

        lin_out.connect(inout)

        output1 = Output('out1', lin_out)
        output2 = Output('out2', Fir(W=a)(inout.sw(2)))
        output3 = Output('out3', Fir(W=a_big)(inout.sw(5)))
        output4 = Output('out4', Fir(W=a)(lin_out))

        target = Input('target')

        test = Modely(visualizer=None, seed=42, log_internal=True)
        test.addModel('model', [output1, output2, output3, output4])
        test.addMinimize('error2', target.last(), output2)
        #test.addMinimize('error3', target.last(), output3)
        #test.addMinimize('error4', target.last(), output4)
        test.neuralizeModel()

        # Dataset with only one sample
        dataset = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2]], 'target': [3,4,5,1,3]}
        self.assertEqual({'out1': [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0]],
                               'out2': [-96.0, -194.0, -179.0, -125.0],
                               'out3': [-96.0, -206.0, -235.0, -239.0],
                               'out4': [-96.0, -194.0, -179.0, -125.0]
                          }, test(dataset))
        test.loadData(name='dataset', source=dataset)
        # TODO add and error
        # dataset = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2]], 'target': [3,4,5,1]}
        # test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[-1], [-5]], test.parameters['W'])
        self.assertEqual([1 ], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[6143], [5627]], test.parameters['W'])
        self.assertEqual([2305], test.parameters['b'])
        self.assertListEqual([[-3836], [-3323]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)

        # Data set with more samples
        test.neuralizeModel(clear_model=True)
        dataset2 = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2],[6,5],[4,5],[0,0]], 'target': [3,4,5,1,3,0,1,0]}
        test.loadData(name='dataset2', source=dataset2)
        self.maxDiff = None
        self.assertEqual({'out1': [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0], [-13.0, -30.0], [-30.0, -28.0], [-28.0, 1.0]],
                          'out2': [-96.0, -194.0, -179.0, -125.0, -202.0, -260.0, -107.0],
                          'out3': [-96.0, -206.0, -235.0, -239.0, -315.0, -355.0, -238.0],
                          'out4': [-96.0, -194.0, -179.0, -125.0, -202.0, -260.0, -107.0]
                          }, test(dataset2))

        # Use a train_batch_size of 4
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1) # TODO add this test
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertEqual([3142], test.parameters['b'])
        self.assertListEqual([[-7682], [-7457.5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4)
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertEqual([3142], test.parameters['b'])
        self.assertListEqual([[-7682], [-7457.5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])

        # Use a small batch but with a prediction sample of 3 = to 4 samples
        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=4)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertEqual([3142], test.parameters['b'])
        self.assertListEqual([[-7682], [-7457.5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])

        # Different minimize
        test.removeMinimize('error2')
        test.addMinimize('error3', target.last(), output3)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[-1], [-5]], test.parameters['W'])
        self.assertEqual([1], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])

        # Use a train_batch_size of 4
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertEqual([3142], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.parameters['ab'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4)
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertEqual([3142], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.parameters['ab'])

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, shuffle_data=False, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[1.0, 3.0], [4.0, 2.0]]], test.internals['inout_0_0']['XY']['in1'])
        self.assertListEqual([[[4.0, 2.0], [6.0, 5.0]]], test.internals['inout_0_1']['XY']['in1'])
        self.assertListEqual([[[6.0, 5.0], [4.0, 5.0]]], test.internals['inout_0_2']['XY']['in1'])
        self.assertListEqual([[[4.0, 5.0], [0.0, 0.0]]], test.internals['inout_0_3']['XY']['in1'])
        self.assertListEqual([[[3.0]]], test.internals['inout_0_0']['XY']['target'])
        self.assertListEqual([[[0.0]]], test.internals['inout_0_1']['XY']['target'])
        self.assertListEqual([[[1.0]]], test.internals['inout_0_2']['XY']['target'])
        self.assertListEqual([[[0.0]]], test.internals['inout_0_3']['XY']['target'])
        self.assertDictEqual({'out1': [[[-15.0], [-13.0]]], 'out2': [[[-125.0]]], 'out3': [[[-125.0]]], 'out4': [[[-125.0]]]}, test.internals['inout_0_0']['out'])
        self.assertDictEqual({'out1': [[[-13.0], [-30.0]]], 'out2': [[[-202.0]]], 'out3': [[[-247.0]]], 'out4': [[[-202.0]]]}, test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[4*-1+2*-5+1.0], [6*-1+5*-5+1.0]]],
                                  'out2': [[[(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]],
                                  'out3': [[[(-15)*3.0+(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]],
                                  'out4': [[[(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]]}, test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[6*-1+5*-5+1.0], [4*-1+5*-5+1.0]]],
                                   'out2': [[[(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]],
                                   'out3': [[[(-15)*2.0+(4*-1+2*-5+1.0)*3.0+(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]],
                                   'out4': [[[(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]]}, test.internals['inout_0_2']['out'])
        self.assertDictEqual({'out1': [[[4*-1+5*-5+1.0], [0*-1+0*-5+1.0]]],
                                  'out2': [[[(4*-1+5*-5+1.0)*4.0+(0*-1+0*-5+1.0)*5.0]]],
                                  'out3': [[[(-15)*1.0+(4*-1+2*-5+1.0)*2.0+(6*-1+5*-5+1.0)*3.0+(4*-1+5*-5+1.0)*4.0+(0*-1+0*-5+1.0)*5.0]]],
                                  'out4': [[[(4*-1+5*-5+1.0)*4.0+(0*-1+0*-5+1.0)*5.0]]]}, test.internals['inout_0_3']['out'])
        self.assertDictEqual({'inout': [[[0.0],[0.0], [0.0], [-15.0], [-13.0]]]}, test.internals['inout_0_0']['state'])
        self.assertDictEqual({'inout': [[[0.0],[0.0], [-15.0], [-13.0], [-30.0]]]}, test.internals['inout_0_1']['state'])
        self.assertDictEqual({'inout': [[[0.0], [-15.0], [-13.0], [-30.0], [-28.0]]]}, test.internals['inout_0_2']['state'])
        self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [1.0]]]}, test.internals['inout_0_3']['state'])
        # Replace instead of roll
        # self.assertDictEqual({'inout': [[[0.0], [0.0], [-15.0], [-13.0], [0.0]]]}, test.internals['inout_0_0']['state'])
        # self.assertDictEqual({'inout': [[[0.0], [-15.0], [-13.0], [-30.0], [0.0]]]}, test.internals['inout_0_1']['state'])
        # self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [0.0]]]}, test.internals['inout_0_2']['state'])
        # self.assertDictEqual({'inout': [[[-13.0], [-30.0], [-28.0], [1.0], [-15.0]]]}, test.internals['inout_0_3']['state'])
        self.assertListEqual([[22273.5], [20993.0]], test.parameters['W'])
        self.assertEqual([6154.0], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[-1784.0], [-4020.0], [-7564.5], [-10843.5], [-9033.0]], test.parameters['ab'])
        with self.assertRaises(KeyError):
            test.internals['inout_1_0']

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=0.01, shuffle_data=False,  num_of_epochs=1, train_batch_size=1,
                         prediction_samples=2)
        self.assertListEqual([[[1.0, 3.0], [4.0, 2.0]]], test.internals['inout_0_0']['XY']['in1'])
        self.assertListEqual([[[4.0, 2.0], [6.0, 5.0]]], test.internals['inout_0_1']['XY']['in1'])
        self.assertListEqual([[[6.0, 5.0], [4.0, 5.0]]], test.internals['inout_0_2']['XY']['in1'])
        self.assertListEqual([[[3.0]]], test.internals['inout_0_0']['XY']['target'])
        self.assertListEqual([[[0.0]]], test.internals['inout_0_1']['XY']['target'])
        self.assertListEqual([[[1.0]]], test.internals['inout_0_2']['XY']['target'])
        self.assertDictEqual({'out1': [[[-15.0], [-13.0]]], 'out2': [[[-125.0]]], 'out3': [[[-125.0]]], 'out4': [[[-125.0]]]}, test.internals['inout_0_0']['out'])
        self.assertDictEqual({'out1': [[[-13.0], [-30.0]]], 'out2': [[[-202.0]]], 'out3': [[[-247.0]]], 'out4': [[[-202.0]]]}, test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[4*-1+2*-5+1.0], [6*-1+5*-5+1.0]]],
                              'out2': [[[(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]],
                              'out3': [[[(-15)*3.0+(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]],
                              'out4': [[[(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]]}, test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[6*-1+5*-5+1.0], [4*-1+5*-5+1.0]]],
                                   'out2': [[[(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]],
                                   'out3': [[[(-15)*2.0+(4*-1+2*-5+1.0)*3.0+(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]],
                                   'out4': [[[(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]]}, test.internals['inout_0_2']['out'])
        self.assertDictEqual({'inout': [[[0.0], [0.0], [0.0], [-15.0], [-13.0]]]}, test.internals['inout_0_0']['state'])
        self.assertDictEqual({'inout': [[[0.0], [0.0], [-15.0], [-13.0], [-30.0]]]}, test.internals['inout_0_1']['state'])
        self.assertDictEqual({'inout': [[[0.0], [-15.0], [-13.0], [-30.0], [-28.0]]]}, test.internals['inout_0_2']['state'])
        # Replace instead of rolling
        # self.assertDictEqual({'inout': [[[0.0], [0.0], [-15.0], [-13.0], [0.0]]]}, test.internals['inout_0_0']['state'])
        # self.assertDictEqual({'inout': [[[0.0], [-15.0], [-13.0], [-30.0], [0.0]]]}, test.internals['inout_0_1']['state'])
        # self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [0.0]]]}, test.internals['inout_0_2']['state'])
        W = test.internals['inout_1_0']['param']['W']
        b = test.internals['inout_1_0']['param']['b']
        self.assertListEqual([[[4.0, 2.0], [6.0, 5.0]]], test.internals['inout_1_0']['XY']['in1'])
        self.assertListEqual([[[6.0, 5.0], [4.0, 5.0]]], test.internals['inout_1_1']['XY']['in1'])
        self.assertListEqual([[[4.0, 5.0], [0.0, 0.0]]], test.internals['inout_1_2']['XY']['in1'])
        self.assertListEqual([[[0.0]]], test.internals['inout_1_0']['XY']['target'])
        self.assertListEqual([[[1.0]]], test.internals['inout_1_1']['XY']['target'])
        self.assertListEqual([[[0.0]]], test.internals['inout_1_2']['XY']['target'])
        self.assertAlmostEqual({'inout': [[[0.0], [0.0], [0.0], [W[0][0]*4.0+W[1][0]*2.0+b[0]], [W[0][0]*6.0+W[1][0]*5.0+b[0]]]]}, test.internals['inout_1_0']['state'])
        self.assertAlmostEqual({'inout': [[[0.0], [0.0], [W[0][0]*4.0+W[1][0]*2.0+b[0]], [W[0][0]*6.0+W[1][0]*5.0+b[0]], [W[0][0]*4.0+W[1][0]*5.0+b[0]]]]}, test.internals['inout_1_1']['state'])
        self.assertAlmostEqual({'inout': [[[0.0], [W[0][0]*4.0+W[1][0]*2.0+b[0]], [W[0][0]*6.0+W[1][0]*5.0+b[0]], [W[0][0]*4.0+W[1][0]*5.0+b[0]],  [W[0][0]*0.0+W[1][0]*0.0+b[0]]]]}, test.internals['inout_1_2']['state'])
        # Replace instead of rolling
        # self.assertAlmostEqual({'inout': [[[0.0], [0.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
        #                                    [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]], [0.0]]]},
        #                        test.internals['inout_1_0']['state'])
        # self.assertAlmostEqual({'inout': [
        #     [[0.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #      [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [0.0]]]}, test.internals['inout_1_1']['state'])
        # self.assertAlmostEqual({'inout': [
        #     [[W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #      [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [W[0][0][0] * 0.0 + W[0][1][0] * 0.0 + b[0][0]], [0.0]]]},
        #                        test.internals['inout_1_2']['state'])

        with self.assertRaises(KeyError):
            test.internals['inout_2_0']

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1, train_batch_size=2, prediction_samples=1)
        self.assertListEqual([[[1.0, 3.0], [4.0, 2.0]], [[4.0, 2.0], [6.0, 5.0]]], test.internals['inout_0_0']['XY']['in1'])
        self.assertListEqual([[[4.0, 2.0], [6.0, 5.0]], [[6.0, 5.0], [4.0, 5.0]]], test.internals['inout_0_1']['XY']['in1'])
        self.assertListEqual([[[3.0]],[[0.0]]], test.internals['inout_0_0']['XY']['target'])
        self.assertListEqual([[[0.0]],[[1.0]]], test.internals['inout_0_1']['XY']['target'])
        with self.assertRaises(KeyError):
            test.internals['inout_1_0']

        test.neuralizeModel(clear_model=True)
        dataset3 = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2],[6,5],[4,5],[0,0]], 'target': [3,4,5,1,3,0,1,0], 'inout':[9,8,7,6,5,4,3,2]}
        test.loadData(name='dataset3', source=dataset3)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1,
                        train_batch_size=1,
                        prediction_samples=2)
        self.assertDictEqual({'inout': [[[8.0], [7.0], [6.0], [-15.0], [-13.0]]]}, test.internals['inout_0_0']['state'])
        self.assertDictEqual({'inout': [[[7.0], [6.0], [-15.0], [-13.0], [-30.0]]]}, test.internals['inout_0_1']['state'])
        self.assertDictEqual({'inout': [[[6.0], [-15.0], [-13.0], [-30.0], [-28.0]]]}, test.internals['inout_0_2']['state'])
        # Replace insead of rolling
        # self.assertDictEqual({'inout': [[[8.0], [7.0], [-15.0], [-13.0], [9.0]]]}, test.internals['inout_0_0']['state'])
        # self.assertDictEqual({'inout': [[[7.0], [-15.0], [-13.0], [-30.0], [8.0]]]}, test.internals['inout_0_1']['state'])
        # self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [7.0]]]}, test.internals['inout_0_2']['state'])
        W = test.internals['inout_1_0']['param']['W']
        b = test.internals['inout_1_0']['param']['b']
        self.assertAlmostEqual({'inout': [[[7.0], [6.0], [5.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
                                           [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]]]]},
                               test.internals['inout_1_0']['state'])
        self.assertAlmostEqual({'inout': [[[6.0], [5.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
                                           [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
                                           [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]]]]},
                               test.internals['inout_1_1']['state'])
        self.assertAlmostEqual({'inout': [
            [[5.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]], [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
             [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]], [W[0][0] * 0.0 + W[1][0] * 0.0 + b[0]]]]},
                               test.internals['inout_1_2']['state'])
        # replace insead of rolling
        # self.assertAlmostEqual({'inout': [[[7.0], [6.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
        #                                    [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]], [8.0]]]},
        #                        test.internals['inout_1_0']['state'])
        # self.assertAlmostEqual({'inout': [[[6.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
        #                                    [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #                                    [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [7.0]]]},
        #                        test.internals['inout_1_1']['state'])
        # self.assertAlmostEqual({'inout': [
        #     [[W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #      [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [W[0][0][0] * 0.0 + W[0][1][0] * 0.0 + b[0][0]], [6.0]]]},
        #                        test.internals['inout_1_2']['state'])

    def test_training_values_fir_connect_train_linear_more_window(self):
        NeuObj.clearNames()
        input1 = Input('in1', dimensions=2)
        W = Parameter('W', values=[[-1], [-5]])
        b = Parameter('b', values=[1])
        lin_out = Linear(W=W, b=b)(input1.sw(2))
        output1 = Output('out1', lin_out)

        inout = Input('inout')
        a = Parameter('a', sw=2, values=[[4], [5]])
        a_big = Parameter('ab', sw=5, values=[[1], [2], [3], [4], [5]])
        output2 = Output('out2', Fir(W=a)(inout.sw(2)))
        output3 = Output('out3', Fir(W=a_big)(inout.sw(5)))
        output4 = Output('out4', Fir(W=a)(lin_out))

        target = Input('target')

        test = Modely(visualizer=None, seed=42, log_internal=True)
        test.addModel('model', [output1, output2, output3, output4])
        test.addMinimize('error2', target.last(), output2)
        test.neuralizeModel()

        # Dataset with only one sample
        dataset = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2]], 'target': [3,4,5,1,3]}
        self.assertEqual({'out1': [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0]],
                               'out2': [-96.0, -194.0, -179.0, -125.0],
                               'out3': [-96.0, -206.0, -235.0, -239.0],
                               'out4': [-96.0, -194.0, -179.0, -125.0]
                          }, test(dataset, connect={'inout':'out1'}))
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[-1], [-5]], test.parameters['W'])
        self.assertListEqual([1], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, connect={'inout':'out1'})
        self.assertListEqual([[6143], [5627]], test.parameters['W'])
        self.assertListEqual([2305], test.parameters['b'])
        self.assertListEqual([[-3836], [-3323]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])
        with self.assertRaises(KeyError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10, connect={'inout':'out1'})
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, connect={'inout':'out1'})

        # Data set with more samples
        test.neuralizeModel(clear_model=True)
        dataset2 = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2],[6,5],[4,5],[0,0]], 'target': [3,4,5,1,3,0,1,0]}
        test.loadData(name='dataset2', source=dataset2)
        self.maxDiff = None
        self.assertEqual({'out1': [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0], [-13.0, -30.0], [-30.0, -28.0], [-28.0, 1.0]],
                          'out2': [-96.0, -194.0, -179.0, -125.0, -202.0, -260.0, -107.0],
                          'out3': [-96.0, -206.0, -235.0, -239.0, -315.0, -355.0, -238.0],
                          'out4': [-96.0, -194.0, -179.0, -125.0, -202.0, -260.0, -107.0]
                          }, test(dataset2, connect={'inout':'out1'}))

        # Use a train_batch_size of 4
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, connect={'inout':'out1'}) TODO add this test
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, connect={'inout':'out1'})
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertListEqual([3142], test.parameters['b'])
        self.assertListEqual([[-7682], [-7457.5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4, connect={'inout':'out1'})
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertListEqual([3142], test.parameters['b'])
        self.assertListEqual([[-7682], [-7457.5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])

        # Use a small batch but with a prediction sample of 3 = to 4 samples
        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=4, connect={'inout':'out1'})
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout':'out1'})
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertListEqual([3142], test.parameters['b'])
        self.assertListEqual([[-7682], [-7457.5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])

        # Different minimize
        test.removeMinimize('error2')
        test.addMinimize('error3', target.last(), output3)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[-1], [-5]], test.parameters['W'])
        self.assertListEqual([1], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [4], [5]], test.parameters['ab'])

        # Use a train_batch_size of 4
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, connect={'inout':'out1'})
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertListEqual([3142], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.parameters['ab'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4, connect={'inout':'out1'})
        self.assertListEqual([[12779], [11678.5]], test.parameters['W'])
        self.assertListEqual([3142], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.parameters['ab'])

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout':'out1'})
        self.assertListEqual([[[1.0, 3.0], [4.0, 2.0]]],test.internals['inout_0_0']['XY']['in1'])
        self.assertListEqual([[[4.0, 2.0], [6.0, 5.0]]],test.internals['inout_0_1']['XY']['in1'])
        self.assertListEqual([[[6.0, 5.0], [4.0, 5.0]]],test.internals['inout_0_2']['XY']['in1'])
        self.assertListEqual([[[4.0, 5.0], [0.0, 0.0]]],test.internals['inout_0_3']['XY']['in1'])
        self.assertListEqual([[[3.0]]],test.internals['inout_0_0']['XY']['target'])
        self.assertListEqual([[[0.0]]],test.internals['inout_0_1']['XY']['target'])
        self.assertListEqual([[[1.0]]],test.internals['inout_0_2']['XY']['target'])
        self.assertListEqual([[[0.0]]],test.internals['inout_0_3']['XY']['target'])
        self.assertDictEqual(
            {'out1': [[[-15.0], [-13.0]]], 'out2': [[[-125.0]]], 'out3': [[[-125.0]]], 'out4': [[[-125.0]]]},
            test.internals['inout_0_0']['out'])
        self.assertDictEqual(
            {'out1': [[[-13.0], [-30.0]]], 'out2': [[[-202.0]]], 'out3': [[[-247.0]]], 'out4': [[[-202.0]]]},
            test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[4 * -1 + 2 * -5 + 1.0], [6 * -1 + 5 * -5 + 1.0]]],
                              'out2': [[[(4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]],
                              'out3': [[[(-15) * 3.0 + (4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]],
                              'out4': [[[(4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]]},
                             test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[6 * -1 + 5 * -5 + 1.0], [4 * -1 + 5 * -5 + 1.0]]],
                              'out2': [[[(6 * -1 + 5 * -5 + 1.0) * 4.0 + (4 * -1 + 5 * -5 + 1.0) * 5.0]]],
                              'out3': [[[(-15) * 2.0 + (4 * -1 + 2 * -5 + 1.0) * 3.0 + (6 * -1 + 5 * -5 + 1.0) * 4.0 + (
                                          4 * -1 + 5 * -5 + 1.0) * 5.0]]],
                              'out4': [[[(6 * -1 + 5 * -5 + 1.0) * 4.0 + (4 * -1 + 5 * -5 + 1.0) * 5.0]]]},
                             test.internals['inout_0_2']['out'])
        self.assertDictEqual({'out1': [[[4 * -1 + 5 * -5 + 1.0], [0 * -1 + 0 * -5 + 1.0]]],
                              'out2': [[[(4 * -1 + 5 * -5 + 1.0) * 4.0 + (0 * -1 + 0 * -5 + 1.0) * 5.0]]],
                              'out3': [[[(-15) * 1.0 + (4 * -1 + 2 * -5 + 1.0) * 2.0 + (6 * -1 + 5 * -5 + 1.0) * 3.0 + (
                                          4 * -1 + 5 * -5 + 1.0) * 4.0 + (0 * -1 + 0 * -5 + 1.0) * 5.0]]],
                              'out4': [[[(4 * -1 + 5 * -5 + 1.0) * 4.0 + (0 * -1 + 0 * -5 + 1.0) * 5.0]]]},
                             test.internals['inout_0_3']['out'])
        self.assertListEqual([[22273.5], [20993.0]], test.parameters['W'])
        self.assertListEqual([6154.0], test.parameters['b'])
        self.assertListEqual([[4], [5]], test.parameters['a'])
        self.assertListEqual([[-1784.0], [-4020.0], [-7564.5], [-10843.5], [-9033.0]],
                             test.parameters['ab'])
        with self.assertRaises(KeyError):
            test.internals['inout_1_0']

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1,
                        train_batch_size=1,
                        prediction_samples=2, connect={'inout':'out1'})
        self.assertListEqual([[[1.0, 3.0], [4.0, 2.0]]],test.internals['inout_0_0']['XY']['in1'])
        self.assertListEqual([[[4.0, 2.0], [6.0, 5.0]]],test.internals['inout_0_1']['XY']['in1'])
        self.assertListEqual([[[6.0, 5.0], [4.0, 5.0]]],test.internals['inout_0_2']['XY']['in1'])
        self.assertListEqual([[[3.0]]],test.internals['inout_0_0']['XY']['target'])
        self.assertListEqual([[[0.0]]],test.internals['inout_0_1']['XY']['target'])
        self.assertListEqual([[[1.0]]],test.internals['inout_0_2']['XY']['target'])
        self.assertDictEqual(
            {'out1': [[[-15.0], [-13.0]]], 'out2': [[[-125.0]]], 'out3': [[[-125.0]]], 'out4': [[[-125.0]]]},
            test.internals['inout_0_0']['out'])
        self.assertDictEqual(
            {'out1': [[[-13.0], [-30.0]]], 'out2': [[[-202.0]]], 'out3': [[[-247.0]]], 'out4': [[[-202.0]]]},
            test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[4 * -1 + 2 * -5 + 1.0], [6 * -1 + 5 * -5 + 1.0]]],
                              'out2': [[[(4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]],
                              'out3': [[[(-15) * 3.0 + (4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]],
                              'out4': [[[(4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]]},
                             test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[6 * -1 + 5 * -5 + 1.0], [4 * -1 + 5 * -5 + 1.0]]],
                              'out2': [[[(6 * -1 + 5 * -5 + 1.0) * 4.0 + (4 * -1 + 5 * -5 + 1.0) * 5.0]]],
                              'out3': [[[(-15) * 2.0 + (4 * -1 + 2 * -5 + 1.0) * 3.0 + (6 * -1 + 5 * -5 + 1.0) * 4.0 + (
                                          4 * -1 + 5 * -5 + 1.0) * 5.0]]],
                              'out4': [[[(6 * -1 + 5 * -5 + 1.0) * 4.0 + (4 * -1 + 5 * -5 + 1.0) * 5.0]]]},
                             test.internals['inout_0_2']['out'])
        W = test.internals['inout_1_0']['param']['W']
        b = test.internals['inout_1_0']['param']['b']
        self.assertListEqual([[[4.0, 2.0], [6.0, 5.0]]],test.internals['inout_1_0']['XY']['in1'])
        self.assertListEqual([[[6.0, 5.0], [4.0, 5.0]]],test.internals['inout_1_1']['XY']['in1'])
        self.assertListEqual([[[4.0, 5.0], [0.0, 0.0]]],test.internals['inout_1_2']['XY']['in1'])
        self.assertListEqual([[[0.0]]],test.internals['inout_1_0']['XY']['target'])
        self.assertListEqual([[[1.0]]],test.internals['inout_1_1']['XY']['target'])
        self.assertListEqual([[[0.0]]],test.internals['inout_1_2']['XY']['target'])
        # self.assertAlmostEqual({'inout': [[[0.0], [0.0], [0.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
        #                                  [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]]]]},
        #                      test.internals['inout_1_0']['state'])
        # self.assertAlmostEqual({'inout': [[[0.0], [0.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
        #                                  [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
        #                                  [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]]]]},
        #                      test.internals['inout_1_1']['state'])
        # self.assertAlmostEqual({'inout': [
        #     [[0.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]], [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
        #      [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]], [W[0][0] * 0.0 + W[1][0] * 0.0 + b[0]]]]},
        #                      test.internals['inout_1_2']['state'])
        self.assertAlmostEqual({'inout': [[[0.0], [0.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
                                         [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]], [float('inf')]]]},
                             test.internals['inout_1_0']['state'])
        self.assertAlmostEqual({'inout': [[[0.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
                                         [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
                                         [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]], [float('inf')]]]},
                             test.internals['inout_1_1']['state'])
        self.assertAlmostEqual({'inout': [
            [[W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]], [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
             [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]], [W[0][0] * 0.0 + W[1][0] * 0.0 + b[0]], [float('inf')]]]},
                             test.internals['inout_1_2']['state'])
        with self.assertRaises(KeyError):
            test.internals['inout_2_0']

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1,
                        train_batch_size=2, prediction_samples=1, connect={'inout':'out1'})
        self.assertListEqual([[[1.0, 3.0], [4.0, 2.0]], [[4.0, 2.0], [6.0, 5.0]]],test.internals['inout_0_0']['XY']['in1'])
        self.assertListEqual([[[4.0, 2.0], [6.0, 5.0]], [[6.0, 5.0], [4.0, 5.0]]],test.internals['inout_0_1']['XY']['in1'])
        self.assertListEqual([[[3.0]], [[0.0]]],test.internals['inout_0_0']['XY']['target'])
        self.assertListEqual([[[0.0]], [[1.0]]],test.internals['inout_0_1']['XY']['target'])
        with self.assertRaises(KeyError):
            test.internals['inout_1_0']

        test.neuralizeModel(clear_model=True)
        dataset3 = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2],[6,5],[4,5],[0,0]], 'target': [3,4,5,1,3,0,1,0], 'inout':[9,8,7,6,5,4,3,2]}
        test.loadData(name='dataset3', source=dataset3)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1,
                        train_batch_size=1,
                        prediction_samples=2, connect={'inout':'out1'})
        # self.assertDictEqual({'inout': [[[8.0], [7.0], [6.0], [-15.0], [-13.0]]]}, test.internals['inout_0_0']['state'])
        # self.assertDictEqual({'inout': [[[7.0], [6.0],  [-15.0], [-13.0], [-30.0]]]}, test.internals['inout_0_1']['state'])
        # self.assertDictEqual({'inout': [[[6.0], [-15.0], [-13.0], [-30.0], [-28.0]]]}, test.internals['inout_0_2']['state'])
        self.assertDictEqual({'inout': [[[8.0], [7.0], [-15.0], [-13.0], [float('inf')]]]}, test.internals['inout_0_0']['state'])
        self.assertDictEqual({'inout': [[[7.0], [-15.0], [-13.0], [-30.0], [float('inf')]]]}, test.internals['inout_0_1']['state'])
        self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [float('inf')]]]}, test.internals['inout_0_2']['state'])
        # Replace instead of rolling
        # self.assertDictEqual({'inout': [[[8.0], [7.0], [-15.0], [-13.0], [9.0]]]}, test.internals['inout_0_0']['connect'])
        # self.assertDictEqual({'inout': [[[7.0], [-15.0], [-13.0], [-30.0], [8.0]]]}, test.internals['inout_0_1']['connect'])
        # self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [7.0]]]}, test.internals['inout_0_2']['connect'])
        W = test.internals['inout_1_0']['param']['W']
        b = test.internals['inout_1_0']['param']['b']
        # self.assertAlmostEqual({'inout': [[[7.0], [6.0], [5.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
        #                                    [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]]]]},
        #                        test.internals['inout_1_0']['state'])
        # self.assertAlmostEqual({'inout': [[[6.0], [5.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
        #                                    [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
        #                                    [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]]]]},
        #                        test.internals['inout_1_1']['state'])
        # self.assertAlmostEqual({'inout':
        #     [[[5.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]], [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
        #      [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]], [W[0][0] * 0.0 + W[1][0] * 0.0 + b[0]]]]},
        #                        test.internals['inout_1_2']['state'])
        # Replace insead of rolling
        self.assertAlmostEqual({'inout': [[[7.0], [6.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
                                           [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]], [float('inf')]]]},
                               test.internals['inout_1_0']['state'])
        self.assertAlmostEqual({'inout': [[[6.0], [W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]],
                                           [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
                                           [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]], [float('inf')]]]},
                               test.internals['inout_1_1']['state'])
        self.assertAlmostEqual({'inout': [
            [[W[0][0] * 4.0 + W[1][0] * 2.0 + b[0]], [W[0][0] * 6.0 + W[1][0] * 5.0 + b[0]],
             [W[0][0] * 4.0 + W[1][0] * 5.0 + b[0]], [W[0][0] * 0.0 + W[1][0] * 0.0 + b[0]], [float('inf')]]]},
                               test.internals['inout_1_2']['state'])

    def test_training_values_fir_and_liner_closed_loop(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target_out1 = Input('target1')
        a = Parameter('a', sw=1, values=[[1]])
        relation1 = Fir(W=a)(input1.last())
        relation1.closedLoop(input1)
        output1 = Output('out1', relation1)

        input2 = Input('in2')
        target_out2 = Input('target2')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=[1])
        relation2 = Linear(W=W,b=b)(input2.last())
        relation2.closedLoop(input2)
        output2 = Output('out2', relation2)

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', target_out1.last(), output1)
        test.addMinimize('error2', target_out2.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [0.0], 'out2': [1.0]}, test())
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1.0],'in2': [1.0]}))
        self.assertEqual({'out1': [1.0], 'out2': [3.0]}, test())
        test.resetStates()
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2':  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, test(prediction_samples=5, num_of_samples=6))

        dataset = {'in1': [1], 'in2': [1.0], 'target1': [3], 'target2': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[5.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertListEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[5.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertListEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2)
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertListEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        dataset = {'in1': [1.0,1.0], 'in2': [1.0,1.0], 'target1': [3.0,3.0], 'target2': [3.0,3.0]}
        test.loadData(name='dataset2', source=dataset)
        test.neuralizeModel(clear_model=True)
        # the out is 3.0 due the mean of the error is not the same of two epochs
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertListEqual([3.0], test.parameters['b'])
        self.assertListEqual([[5.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2)
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertListEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

    def test_training_values_fir_and_liner_train_closed_loop(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target_out1 = Input('target1')
        a = Parameter('a', sw=1, values=[[1]])
        output1 = Output('out1',Fir(W=a)(input1.last()))

        input2 = Input('in2')
        target_out2 = Input('target2')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=1)
        output2 = Output('out2', Linear(W=W,b=b)(input2.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', target_out1.last(), output1)
        test.addMinimize('error2', target_out2.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [0.0], 'out2': [1.0]}, test(closed_loop={'in1':'out1','in2':'out2'}))
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1.0],'in2': [1.0]},closed_loop={'in1':'out1','in2':'out2'}))
        # # The memory is reset for each call
        self.assertEqual({'out1': [0.0], 'out2': [1.0]}, test(closed_loop={'in1':'out1', 'in2':'out2'}))

        dataset = {'in1': [1], 'in2': [1.0], 'target1': [3], 'target2': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertEqual([3.0], test.parameters['b'])
        self.assertListEqual([[5.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertEqual([3.0], test.parameters['b'])
        self.assertListEqual([[5.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertEqual([3.0], test.parameters['b'])
        self.assertListEqual([[5.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        dataset = {'in1': [1.0,1.0], 'in2': [1.0,1.0], 'target1': [3.0,3.0], 'target2': [3.0,3.0]}
        test.loadData(name='dataset2', source=dataset)
        test.neuralizeModel(clear_model=True)
        # the out is 3.0 due the mean of the error is not the same of two epochs
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[3.0]], test.parameters['W'])
        self.assertEqual([3.0], test.parameters['b'])
        self.assertListEqual([[5.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[-3.0]], test.parameters['W'])
        self.assertEqual([-3.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

    def test_training_values_fir_and_linear_closed_loop_more_prediction(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target_out1 = Input('target1')
        a = Parameter('a', sw=1, values=[[1]])
        relation1 = Fir(W=a)(input1.last())
        relation1.closedLoop(input1)
        output1 = Output('out1',relation1)

        input2 = Input('in2')
        target_out2 = Input('target2')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=[1])
        relation2=Linear(W=W,b=b)(input2.last())
        relation2.closedLoop(input2)
        output2 = Output('out2', relation2)

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', target_out1.last(), output1)
        test.addMinimize('error2', target_out2.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                         test(prediction_samples=5, num_of_samples=6))
        #self.assertEqual({'out1': [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 'out2': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]},
        #                 test({'in1':[1.0,2.0]},prediction_samples=5))
        self.assertEqual({'out1': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0]},
                          test({'in1': [1.0, 2.0]}, prediction_samples=5, num_of_samples=7))
        #self.assertEqual({'out1': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 'out2': [0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]},
        #                 test({'in2':[-1.0,-2.0,-3.0]},prediction_samples=5))
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]},
                          test({'in2':[-1.0,-2.0,-3.0]}, prediction_samples=5, num_of_samples=8))

        dataset = {'in1': [0,2,7,1], 'in2': [-1,0,-3,7], 'target1': [3,4,5,1], 'target2': [-3,-4,-5,-1]}
        test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[-24.5]], test.parameters['W'])
        self.assertListEqual([-9.0], test.parameters['b'])
        self.assertListEqual([[-4.0]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1)
        self.assertListEqual([[-24.5]], test.parameters['W'])
        self.assertListEqual([-9.0], test.parameters['b'])
        self.assertListEqual([[-4.0]], test.parameters['a'])

        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        # test.neuralizeModel(clear_model=True)
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)
        # self.assertListEqual([[[1.0]]], test.parameters['W'])
        # self.assertListEqual([[1.0]], test.parameters['b'])
        # self.assertListEqual([[1.0]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([-24.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        # test.neuralizeModel(clear_model=True)
        # self.assertListEqual([[[1.0]]],test.parameters['W'])
        # self.assertListEqual([[1.0]], test.parameters['b'])
        # self.assertListEqual([[1.0]], test.parameters['a'])
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2, train_batch_size=2, prediction_samples=2)
        # self.assertListEqual([[[1.0]]], test.parameters['W'])
        # self.assertListEqual([[-24.0]], test.parameters['b'])
        # self.assertListEqual([[1.0]], test.parameters['a'])

    def test_training_values_fir_and_linear_train_closed_loop_more_prediction(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target_out1 = Input('target1')
        a = Parameter('a', sw=1, values=[[1]])
        output1 = Output('out1',Fir(W=a)(input1.last()))

        input2 = Input('in2')
        target_out2 = Input('target2')
        W = Parameter('W', values=[[1]])
        b = Parameter('b', values=[1])
        output2 = Output('out2', Linear(W=W,b=b)(input2.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', target_out1.last(), output1)
        test.addMinimize('error2', target_out2.last(), output2)
        test.neuralizeModel()
        # self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        #                  test(prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}, _num_of_samples=6))
        self.assertEqual({'out1': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0]},
                         test({'in1':[1.0, 2.0]}, prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}, num_of_samples=7))
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]},
                         test({'in2':[-1.0,-2.0,-3.0]}, prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}, num_of_samples=8))

        dataset = {'in1': [0,2,7,1], 'in2': [-1,0,-3,7], 'target1': [3,4,5,1], 'target2': [-3,-4,-5,-1]}
        test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, closed_loop={'in2':'out2','in1':'out1'})
        self.assertListEqual([[-24.5]], test.parameters['W'])
        self.assertListEqual([-9.0], test.parameters['b'])
        self.assertListEqual([[-4.0]], test.parameters['a'])
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10, closed_loop={'in2':'out2','in1':'out1'})
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1,  closed_loop={'in2':'out2','in1':'out1'}) # TODO add this test
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3,  closed_loop={'in2':'out2','in1':'out1'})
        self.assertListEqual([[1.0]], test.parameters['W'])
        self.assertListEqual([-24.0], test.parameters['b'])
        self.assertListEqual([[1.0]], test.parameters['a'])

        # test.neuralizeModel(clear_model=True) # TODO add this test
        # self.assertListEqual([[[1.0]]],test.parameters['W'])
        # self.assertListEqual([[1.0]], test.parameters['b'])
        # self.assertListEqual([[1.0]], test.parameters['a'])
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2, train_batch_size=2, prediction_samples=2,  closed_loop={'in2':'out2','in1':'out1'})
        # self.assertListEqual([[[1.0]]], test.parameters['W'])
        # self.assertListEqual([[-24.0]], test.parameters['b'])
        # self.assertListEqual([[1.0]], test.parameters['a'])

    def test_training_values_fir_and_liner_closed_loop_bigger_window(self):
        NeuObj.clearNames()
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[-1],[-5]])
        b = Parameter('b', values=[1])
        relation1 = Linear(W=W, b=b)(input1.sw(2))

        input2 = Input('in2')
        a = Parameter('a', sw=4, values=[[1,3],[2,4],[3,5],[4,6]])
        relation2 = Fir(output_dimension=2,W=a)(input2.sw(4))

        relation2.closedLoop(input1)
        relation1.closedLoop(input2)

        output1 = Output('out1', relation1)
        output2 = Output('out2', relation2)

        target1 = Input('target1')
        target2 = Input('target2', dimensions=2)

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', output1, target1.sw(2))
        test.addMinimize('error2', output2, target2.last())
        test.neuralizeModel()

        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0]], 'out2': [[[-49.0, -107.0]], [[-12.0, -46.0]], [[13.0, 15.0]]]},
                          test({'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0]], 'in2': [-10, -16, -5, 2, 2, 2]}))

        dataset = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0]], 'in2': [-10, -16, -5, 2],
                   'target1': [-11, -17, -12, -20],
                   'target2': [[-34.0, -86.0], [-31.0, -90.0], [-32.0, -86.0], [-33.0, -84.0]]}
        test.loadData(name='dataset', source=dataset)
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0]],
                          'out2': [[[-49.0, -107.0]], [[-120.0, -214.0]], [[-205.0, -333.0]]]},
                         test(dataset))

        self.assertListEqual([[-1],[-5]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1,3],[2,4],[3,5],[4,6]], test.parameters['a'])
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[83.0],[105.0]], test.parameters['W'])
        self.assertListEqual([6.0], test.parameters['b'])
        self.assertListEqual([[-159.0, -227.0], [-254., -364.], [-77., -110.], [36., 52.]], test.parameters['a'])
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, minimize_gain={'error1':0})
        self.assertListEqual([[-1],[-5]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[-159.0, -227.0], [-254., -364.], [-77., -110.], [36., 52.]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, minimize_gain={'error2':0})
        self.assertListEqual([[83.0],[105.0]], test.parameters['W'])
        self.assertListEqual([6.0], test.parameters['b'])
        self.assertListEqual([[1,3],[2,4],[3,5],[4,6]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        dataset2 = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 'in2': [-10, -16, -5, 2, 3, -3, 5],
                   'target1':[-11,-17,-12,-20,5,1,0],
                    'target2':[[-34.0, -86.0],[-31.0, -90.0],[-32.0, -86.0],[-33.0, -84.0],[-31.0, -84.0],[0.0, -84.0],[-31.0, 0.0]]}
        test.loadData(name='dataset2', source=dataset2)
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0],[-4.0,1.0],[1.0,-4.0],[-4.0,0.0]],
                          'out2': [[[-49, -107]], [[-8, -40]], [[-4, -10]], [[19, 33]], [[-11, -17]], [[-24, -44]]]},
                         test(dataset2))

        # Use a train_batch_size of 4
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1) # TODO Add this test
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[20.0], [21.0]], test.parameters['W'])
        self.assertListEqual([2.75], test.parameters['b'])
        self.assertListEqual([[23., 197.5], [-68.75, -94.75], [12., -76.5], [-70.75, -1.25]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4)
        self.assertListEqual([[20.0], [21.0]], test.parameters['W'])
        self.assertListEqual([2.75], test.parameters['b'])
        self.assertListEqual([[23., 197.5], [-68.75, -94.75], [12., -76.5], [-70.75, -1.25]], test.parameters['a'])

        # Use a small batch but with a prediction sample of 3 = to 4 samples
        dataset3 = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 'in2': [-10, -16, -5, 2, 3, -3, 5],
                   'target1':[-11, -17, -30, -2, 582, 1421, -18975],
                    'target2':[[-34.0, -86.0],[-31.0, -90.0],[-32.0, -86.0],[-48, -106], [-140, -256], [2254, 3341], [7420, 11374]]}
        test.loadData(name='dataset3', source=dataset3)
        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=4)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[-3010.5],[-5211.25]], test.parameters['W'])
        self.assertListEqual([199.5], test.parameters['b'])
        self.assertListEqual([[252.75, 1172.5], [420.5, 1998.25], [-228.5, 627.5], [-626.75, 3036.5]], test.parameters['a'])

        # test.neuralizeModel(clear_model=True)
        # test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=0.01, num_of_epochs=2, train_batch_size=2, prediction_samples=2)
        # self.assertListEqual([[[-3010.5],[-5211.25]]],test.parameters['W'])
        # self.assertListEqual([[199.5]], test.parameters['b'])
        # self.assertListEqual([[252.75, 1172.5], [420.5, 1998.25], [-228.5, 627.5], [-626.75, 3036.5]], test.parameters['a'])

    def test_training_values_fir_and_liner_train_closed_loop_bigger_window(self):
        NeuObj.clearNames()
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[-1],[-5]])
        b = Parameter('b', values=[1])
        output1 = Output('out1', Linear(W=W,b=b)(input1.sw(2)))

        input2 = Input('in2')
        a = Parameter('a', sw=4, values=[[1,3],[2,4],[3,5],[4,6]])
        output2 = Output('out2', Fir(output_dimension=2,W=a)(input2.sw(4)))

        target1 = Input('target1')
        target2 = Input('target2', dimensions=2)

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', output1, target1.sw(2))
        test.addMinimize('error2', output2, target2.last())
        test.neuralizeModel()

        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0]], 'out2': [[[-49.0, -107.0]], [[-12.0, -46.0]], [[13.0, 15.0]]]},
                          test({'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0]], 'in2': [-10, -16, -5, 2, 2, 2]},closed_loop={'in1':'out2','in2':'out1'}))

        dataset = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0]], 'in2': [-10, -16, -5, 2],
                   'target1': [-11, -17, -12, -20],
                   'target2': [[-34.0, -86.0], [-31.0, -90.0], [-32.0, -86.0], [-33.0, -84.0]]}
        test.loadData(name='dataset', source=dataset)
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0]],
                          'out2': [[[-49.0, -107.0]], [[-120.0, -214.0]], [[-205.0, -333.0]]]},
                         test(dataset,closed_loop={'in1':'out2','in2':'out1'}))

        self.assertListEqual([[-1],[-5]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[1,3],[2,4],[3,5],[4,6]], test.parameters['a'])
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0,closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[83.0],[105.0]], test.parameters['W'])
        self.assertListEqual([6.0], test.parameters['b'])
        self.assertListEqual([[-159.0, -227.0], [-254., -364.], [-77., -110.], [36., 52.]], test.parameters['a'])
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10,closed_loop={'in1':'out2','in2':'out1'})
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1,closed_loop={'in1':'out2','in2':'out1'})
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, minimize_gain={'error1':0},closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[-1],[-5]], test.parameters['W'])
        self.assertListEqual([1.0], test.parameters['b'])
        self.assertListEqual([[-159.0, -227.0], [-254., -364.], [-77., -110.], [36., 52.]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, minimize_gain={'error2':0},closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[83.0],[105.0]], test.parameters['W'])
        self.assertListEqual([6.0], test.parameters['b'])
        self.assertListEqual([[1,3],[2,4],[3,5],[4,6]], test.parameters['a'])

        test.neuralizeModel(clear_model=True)
        dataset2 = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 'in2': [-10, -16, -5, 2, 3, -3, 5],
                   'target1':[-11,-17,-12,-20,5,1,0],
                    'target2':[[-34.0, -86.0],[-31.0, -90.0],[-32.0, -86.0],[-33.0, -84.0],[-31.0, -84.0],[0.0, -84.0],[-31.0, 0.0]]}
        test.loadData(name='dataset2', source=dataset2)
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0],[-4.0,1.0],[1.0,-4.0],[-4.0,0.0]],
                          'out2': [[[-49, -107]], [[-8, -40]], [[-4, -10]], [[19, 33]], [[-11, -17]], [[-24, -44]]]},
                         test(dataset2,closed_loop={'in1':'out2','in2':'out1'}))

        # Use a train_batch_size of 4
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, closed_loop={'in1':'out2','in2':'out1'}) # TODO Add this test
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[20.0], [21.0]], test.parameters['W'])
        self.assertListEqual([2.75], test.parameters['b'])
        self.assertListEqual([[23., 197.5], [-68.75, -94.75], [12., -76.5], [-70.75, -1.25]], test.parameters['a'])
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4,closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[20.0], [21.0]], test.parameters['W'])
        self.assertListEqual([2.75], test.parameters['b'])
        self.assertListEqual([[23., 197.5], [-68.75, -94.75], [12., -76.5], [-70.75, -1.25]], test.parameters['a'])

        # Use a small batch but with a prediction sample of 3 = to 4 samples
        dataset3 = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 'in2': [-10, -16, -5, 2, 3, -3, 5],
                   'target1':[-11, -17, -30, -2, 582, 1421, -18975],
                    'target2':[[-34.0, -86.0],[-31.0, -90.0],[-32.0, -86.0],[-48, -106], [-140, -256], [2254, 3341], [7420, 11374]]}
        test.loadData(name='dataset3', source=dataset3)
        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=4,closed_loop={'in1':'out2','in2':'out1'})
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3,closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[-3010.5],[-5211.25]], test.parameters['W'])
        self.assertListEqual([199.5], test.parameters['b'])
        self.assertListEqual([[252.75, 1172.5], [420.5, 1998.25], [-228.5, 627.5], [-626.75, 3036.5]], test.parameters['a'])

        # test.neuralizeModel(clear_model=True)
        # test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=1, num_of_epochs=2, train_batch_size=2, prediction_samples=2,closed_loop={'in1':'out2','in2':'out1'})
        # self.assertListEqual([[[-3010.5],[-5211.25]]],test.parameters['W'])
        # self.assertListEqual([[199.5]], test.parameters['b'])
        # self.assertListEqual([[252.75, 1172.5], [420.5, 1998.25], [-228.5, 627.5], [-626.75, 3036.5]], test.parameters['a'])

    def test_train_compare_state_and_closed_loop(self):
        NeuObj.clearNames()
        dataset = {'control': [-1, -1, -5, 2,-1, -1, -5, 2,-1, -1, -5, 2,-1, -1, -5, 2,-1, -1, -5, 2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2],
                   'target1': [-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2],
                   'target2': [[-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0]
                               ]}

        feed = Input('control')
        input1 = Input('in1', dimensions=2)
        W = Parameter('W', values=[[0.1],[0.1]])
        b = Parameter('b', values=[0.1])
        output1 = Output('out1', feed.sw(2)+Linear(W=W, b=b)(input1.sw(2)))

        input2 = Input('in2')
        a = Parameter('a', sw=4, values=[[0.1,0.3],[0.2,0.4],[0.3,0.5],[0.4,0.6]])
        output2 = Output('out2', Fir(output_dimension=2,W=a)(input2.sw(4)))

        target1 = Input('target1')
        target2 = Input('target2', dimensions=2)

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', output1, target1.sw(2))
        test.addMinimize('error2', output2, target2.last())
        test.neuralizeModel()
        test.loadData(name='dataset', source=dataset)
        test.trainModel(splits=[60,40,0], optimizer='SGD', lr=0.001, num_of_epochs=1, prediction_samples=10,
                        closed_loop={'in1': 'out2', 'in2': 'out1'})

        NeuObj.clearNames()
        feed = Input('control')
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[0.1],[0.1]])
        b = Parameter('b', values=[0.1])
        relation1 = feed.sw(2) + Linear(W=W, b=b)(input1.sw(2))

        input2 = Input('in2')
        a = Parameter('a', sw=4, values=[[0.1,0.3],[0.2,0.4],[0.3,0.5],[0.4,0.6]])
        relation2 = Fir(output_dimension=2, W=a)(input2.sw(4))

        relation1.closedLoop(input2)
        relation2.closedLoop(input1)

        output1 = Output('out1', relation1)
        output2 = Output('out2', relation2)

        target1 = Input('target1')
        target2 = Input('target2', dimensions=2)

        test2 = Modely(visualizer=None, seed=42)
        test2.addModel('model', [output1, output2])
        test2.addMinimize('error1', output1, target1.sw(2))
        test2.addMinimize('error2', output2, target2.last())
        test2.neuralizeModel()
        test2.loadData(name='dataset', source=dataset)
        test2.trainModel(splits=[60,40,0], optimizer='SGD', lr=0.001, num_of_epochs=1, prediction_samples=10)

        self.assertListEqual(test2.parameters['W'], test.parameters['W'])
        self.assertListEqual(test2.parameters['a'], test.parameters['a'])
        self.assertListEqual(test2.parameters['b'], test.parameters['b'])
        self.assertListEqual(test2._training['error1']['train'] , test._training['error1']['train'])
        self.assertListEqual(test2._training['error1']['val'], test._training['error1']['val'])
        self.assertListEqual(test2._training['error2']['train'] , test._training['error2']['train'])
        self.assertListEqual(test2._training['error2']['val'], test._training['error2']['val'])

        test2 = Modely(visualizer=None, seed=42, log_internal=True)
        test2.addModel('model', [output1, output2])
        test2.addMinimize('error1', output1, target1.sw(2))
        test2.addMinimize('error2', output2, target2.last())
        test2.neuralizeModel()
        test2.loadData(name='dataset', source=dataset)
        test2.trainModel(splits=[100,0,0], train_batch_size=1, step=10, optimizer='SGD', lr=0.001, num_of_epochs=1, prediction_samples=3)

        self.assertEqual(len(test2.internals.keys()), (3+1)*((26+10)//(10+1)))
        #self.assertEqual(test2.internals)

    def test_train_derivate_wrt_input_closed_loop(self):
        NeuObj.clearNames()
        x = Input('x')
        x_target = Input('x_target')
        y = Input('y')
        x_last = x.last()
        y_last = y.last()

        p=Parameter('fir',sw=1,values=[[-0.5]])

        fun = Sin(x_last) + Fir(W=p)(x_last) + Cos(y_last)
        out_der = Differentiate(fun, x_last) + Differentiate(fun, y_last)
        out_der.closedLoop(x)
        out = Output('out', out_der)

        m = Modely(visualizer=None, seed=7)
        m.addModel('model', [out])
        m.addMinimize('error', out, x_target.next())
        m.neuralizeModel()

        K = 0

        def fun_data(x, y, K):
            return K + np.cos(x) - np.sin(y)

        x_data, y_data = [], []
        x = -0.2
        y = 0.5
        for i in range(100):
            x = y = fun_data(x, y, K)
            x_data.append(x)
            y_data.append(y)

        dataset = {'x': x_data, 'y': y_data, 'x_target': x_data}
        m.loadData('dataset', dataset)
        m.trainModel(lr=0.4, num_of_epochs=200, closed_loop={'y': 'out'}, prediction_samples=9)
        result = m({'x': [-0.2], 'y': [0.5]}, closed_loop={'y':'out'}, num_of_samples=10, prediction_samples=10)
        self.assertAlmostEqual([a.tolist() for a in x_data[0:10]],result['out'])

    def test_train_derivate_wrt_input_connect(self):
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y')
        x_last = x.last()
        y_last = y.last()
        p1 = Parameter('p1', sw=1, values=[[0]])
        fun = Sin(x_last) + Fir(W=p1)(x_last) + Cos(y_last)
        out_der = Differentiate(fun, x_last) + Differentiate(fun, y_last)

        x2 = Input('x2')
        y2 = Input('y2')
        x2_last = x2.last()
        y2_last = y2.last()
        p2 = Parameter('p2', sw=1, values=[[0]])
        fun2 = Sin(x2_last) + Fir(W=p2)(x2_last) + Cos(y2_last)
        out_der2 = Differentiate(fun2, x2_last) + Differentiate(fun2, y2_last)
        out_der.connect(x2)

        out1 = Output('out1', out_der)
        out2 = Output('out2', out_der2)

        target = Input('target')

        m = Modely(visualizer=None, seed=5)
        m.addModel('model', [out1,out2])
        m.addMinimize('error', out_der2, target.last())
        m.neuralizeModel()

        K1 = -0.5
        K2 = 3

        def fun_data(x, y, K):
            return K + np.cos(x) - np.sin(y)

        def fun_data2(x, y, K1, K2):
            return K2 + np.cos(fun_data(x,y,K1)) - np.sin(fun_data(x,y,K1))

        target = []
        import numpy as np
        x = np.random.rand(100)
        y = np.random.rand(100)
        for (xi,yi) in zip(x,y):
            r = fun_data2(xi, yi, K1, K2)
            target.append(r)

        dataset = {'x': x.tolist(), 'y': y.tolist(), 'target': target}
        m.loadData('dataset', dataset)
        m.trainModel(lr=0.3, num_of_epochs=200, splits=[70,20,10], connect={'y2': 'out1'}, prediction_samples=9)

        result = m({'x': x.tolist(), 'y': y.tolist()}, connect={'y2':'out1'}, num_of_samples=10, prediction_samples=10)
        self.assertAlmostEqual([a.tolist() for a in target[0:10]],result['out2'])

    def test_training_values_fir_connect_linear_more_window(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        W = Parameter('W', sw=4, values=[[1], [1], [1], [1]])
        b = Parameter('b', values=0)
        lin_out = Fir(W=W, b=b)(input1.sw(4))

        inout = Input('inout')

        lin_out.connect(inout)

        W1 = Parameter('W1', sw=4, values=[[1], [1], [1], [1]])
        b1 = Parameter('b1', values=0)
        fir_out = Fir(W=W1, b=b1)(inout.sw(4))
        output = Output('out', inout.sw(4))
        output1 = Output('out1', lin_out)
        output2 = Output('out2', fir_out)

        target = Input('target')

        test = Modely(visualizer=None, seed=42, log_internal=True)
        test.addModel('model', [output, output1, output2])
        test.addMinimize('error2', target.last(), output2)
        test.neuralizeModel()

        # Dataset with only one sample
        dataset = {'in1': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], 
                   'inout':[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], 
                   'target': [10.0, 22.0, 34.0, 46.0, 58.0, 70.0, 82.0, 94.0, 106.0]}
        test.loadData(name='dataset', source=dataset)


        # Use a small batch but with a prediction sample of 3 = to 4 samples
        self.assertListEqual([[1.0], [1.0], [1.0], [1.0]], test.parameters['W'])
        self.assertListEqual([[1.0], [1.0], [1.0], [1.0]], test.parameters['W1'])
        self.assertListEqual([0.0], test.parameters['b'])
        self.assertListEqual([0.0], test.parameters['b1'])
        inference = test(inputs=dataset, prediction_samples=3)
        self.assertDictEqual({'out': [[10.0, 20.0, 30.0, 12.0], [20.0, 30.0, 12.0, 20.0], [30.0, 12.0, 20.0, 28.0], [12.0, 20.0, 28.0, 36.0], [50.0, 60.0, 70.0, 44.0], [60.0, 70.0, 44.0, 52.0]], 
                              'out1': [12.0, 20.0, 28.0, 36.0, 44.0, 52.0], 
                              'out2': [72.0, 82.0, 90.0, 96.0, 224.0, 226.0]}, 
                              inference)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=0.0001, train_batch_size=1, num_of_epochs=1, shuffle_data=False, prediction_samples=3)
        for i, (k,v) in enumerate(test.internals.items()):
            if i > 3:
                break
            self.assertEqual(v['out']['out'][0][-1][0], v['out']['out1'][0][0][0])
            self.assertEqual(sum([x[0] for x in v['out']['out'][0]]), v['out']['out2'][0][0][0])
        self.assertDictEqual({'inout': [[[20.0], [30.0], [12.0], [float('inf')]]]}, test.internals['inout_0_0']['state'])
        self.assertDictEqual({'inout': [[[30.0], [12.0], [20.0], [float('inf')]]]}, test.internals['inout_0_1']['state'])
        self.assertDictEqual({'inout': [[[12.0], [20.0], [28.0], [float('inf')]]]}, test.internals['inout_0_2']['state'])
        self.assertDictEqual({'inout': [[[20.0], [28.0], [36.0], [float('inf')]]]}, test.internals['inout_0_3']['state'])


    # def test_state_initialization(self):
    #     NeuObj.clearNames()
    #     x = Input('x')
    #     y = Input('y')
    #     target = Input('target')
    #
    #     p_1 = Parameter('p1', sw=1, values=[[1]])
    #     p_2 = Parameter('p2', sw=1, values=[[2]])
    #     fir1 = Fir(W=p_1, b=False)(x.last())
    #     fir2 = Fir(W=p_2, b=False)(y.last())
    #     relation = fir1 + fir2
    #     #relation = ClosedLoop(relation, y, init=fir1)
    #     relation.closedLoop(y, init=fir1)
    #     out = Output('out', relation)
    #
    #     model = Modely(visualizer=None, seed=42, log_internal=True)
    #     model.addModel('model', out)
    #     model.addMinimize('error', out, target.last())
    #     model.neuralizeModel(1)
    #
    #     def fun(x, a, b):
    #         return a*x + b
    #
    #     x_data = np.linspace(1, 10, 10)
    #     target_data = fun(x_data, 2, 3)
    #
    #     dataset = {'x': x_data, 'target': target_data}
    #     model.loadData('dataset', dataset)
    #     model.trainModel(lr=0.0, train_dataset='dataset', num_of_epochs=1, prediction_samples=5, train_batch_size=1, shuffle_data=False)
    #     self.assertEqual(model.internals['inout_0_0']['out']['out'], [[[3.0]]])
    #     self.assertEqual(model.internals['inout_0_1']['out']['out'], [[[8.0]]])
    #     self.assertEqual(model.internals['inout_0_2']['out']['out'], [[[19.0]]])
    #     self.assertEqual(model.internals['inout_0_3']['out']['out'], [[[42.0]]])
    #     self.assertEqual(model.internals['inout_0_4']['out']['out'], [[[89.0]]])
    #     self.assertEqual(model.internals['inout_0_5']['out']['out'], [[[184.0]]])
    #     self.assertEqual(model.internals['inout_1_0']['out']['out'], [[[6.0]]])