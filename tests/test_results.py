import unittest, os, sys
import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 3 Tests
# Test the value of the mse performance of the model
# in closed loop and states cases
# in connect and states cases

data_folder = os.path.join(os.path.dirname(__file__), '_data/')

class ModelyTrainingTest(unittest.TestCase):
    def test_analysis_results(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', sw=1, values=[[1]])
        output1 = Output('out', Fir(W=a)(input1.last()))

        test = Modely(visualizer=None,seed=42)
        test.addModel('model', output1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1,1,1,1,1,1,1,1,1,1], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset', source=dataset)

        # Test prediction
        test.analyzeModel('dataset')
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]]]},
                         test.prediction['dataset']['error1'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((2.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((1+4)/2.0, test.performance['dataset']['total']['mean_error'])

        test.analyzeModel('dataset', batch_size=5)
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]]]},
                         test.prediction['dataset']['error1'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((2.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((1+4)/2.0, test.performance['dataset']['total']['mean_error'])

        test.analyzeModel('dataset', batch_size=6)
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]]]},
                         test.prediction['dataset']['error1'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((2.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((1+4)/2.0, test.performance['dataset']['total']['mean_error'])

        dataset = {'in1': [1,1,1,1,1,1,2,2,3,3], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset2', source=dataset)

        test.analyzeModel('dataset2')
        self.assertAlmostEqual((1.0 ** 2.0) * 8.0 / 10.0, test.performance['dataset2']['error1']['mse'], places=6)
        self.assertAlmostEqual(((2.0 ** 2) * 6.0 + (1.0 ** 2) * 2.0) / 10.0, test.performance['dataset2']['error2']['mse'], places=6)
        self.assertAlmostEqual(((1.0 ** 2) * 8.0 / 10.0 + ((2.0 ** 2) * 6.0 + (1.0 ** 2) * 2.0) / 10.0 )/2.0, test.performance['dataset2']['total']['mean_error'], places=6)

        test.analyzeModel('dataset2', batch_size=5)
        self.assertAlmostEqual((1.0 ** 2.0) * 8.0 / 10.0, test.performance['dataset2']['error1']['mse'], places=6)
        self.assertAlmostEqual(((2.0 ** 2) * 6.0 + (1.0 ** 2) * 2.0) / 10.0, test.performance['dataset2']['error2']['mse'], places=6)
        self.assertAlmostEqual(((1.0 ** 2) * 8.0 / 10.0 + ((2.0 ** 2) * 6.0 + (1.0 ** 2) * 2.0) / 10.0 )/2.0, test.performance['dataset2']['total']['mean_error'], places=6)

        test.analyzeModel('dataset2', batch_size=6)
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]]]},
                         test.prediction['dataset2']['error1'])
        self.assertEqual((1.0 ** 2) * 6.0 / 6.0, test.performance['dataset2']['error1']['mse'])
        self.assertEqual((2.0 ** 2) * 6.0 / 6.0, test.performance['dataset2']['error2']['mse'])
        self.assertEqual((1+4)/2.0, test.performance['dataset2']['total']['mean_error'])

        test.analyzeModel('dataset2', minimize_gain={'error1': 0.5, 'error2': 0.0})
        self.assertAlmostEqual((1.0 ** 2.0) * 8.0 / 10.0 * 0.5, test.performance['dataset2']['error1']['mse'], places=6)
        self.assertAlmostEqual(0.0, test.performance['dataset2']['error2']['mse'], places=6)
        self.assertAlmostEqual(((1.0 ** 2) * 8.0 / 10.0 * 0.5 + 0.0)/2.0, test.performance['dataset2']['total']['mean_error'], places=6)

    def test_analysis_results_closed_loop_state(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', sw=1, values=[[2]])
        relation = Fir(W=a)(input1.last())
        relation.closedLoop(input1)
        output1 = Output('out', relation)

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', output1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        #Prediction samples = None
        dataset = {'in1': [1,1,1,1,1,1,1,1,1,1], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset', source=dataset)

        # Test prediction
        test.analyzeModel('dataset',prediction_samples=-1)
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]]},
                         test.prediction['dataset']['error1'])
        self.assertEqual((0.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((0+1)/2.0, test.performance['dataset']['total']['mean_error'])

        dataset = {'in1': [1,1,1,1,1,1,2,2,3,3], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset2', source=dataset)

        test.analyzeModel('dataset2', batch_size=5)
        self.assertEqual((0.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((0+1)/2.0, test.performance['dataset']['total']['mean_error'])

        # Prediction samples = 5
        dataset = {'in1': [1,2,3,4,5,6,7,8,9,10], 'out1': [11,12,13,14,15,16,17,18,19,20], 'out2': [10,20,30,40,50,60,70,80,90,100]}
        test.loadData(name='dataset3', source=dataset)

        test.analyzeModel('dataset3', prediction_samples=5, batch_size=2)
        A =[[[[11.0]], [[12.0]], [[13.0]], [[14.0]]],
                                   [[[12.0]], [[13.0]], [[14.0]], [[15.0]]],
                                   [[[13.0]], [[14.0]], [[15.0]], [[16.0]]],
                                   [[[14.0]], [[15.0]], [[16.0]], [[17.0]]],
                                   [[[15.0]], [[16.0]], [[17.0]], [[18.0]]],
                                   [[[16.0]], [[17.0]], [[18.0]], [[19.0]]]]
        B =[[[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
                                   [[[4.0]], [[8.0]], [[12.0]], [[16.0]]],
                                   [[[8.0]], [[16.0]], [[24.0]], [[32.0]]],
                                   [[[16.0]], [[32.0]], [[48.0]], [[64.0]]],
                                   [[[32.0]], [[64.0]], [[96.0]], [[128.0]]],
                                   [[[64.0]], [[128.0]], [[192.0]], [[256.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
                                   [[[20.0]], [[30.0]], [[40.0]], [[50.0]]],
                                   [[[30.0]], [[40.0]], [[50.0]], [[60.0]]],
                                   [[[40.0]], [[50.0]], [[60.0]], [[70.0]]],
                                   [[[50.0]], [[60.0]], [[70.0]], [[80.0]]],
                                   [[[60.0]], [[70.0]], [[80.0]], [[90.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset3']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset3']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/24.0, test.performance['dataset3']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/24.0, test.performance['dataset3']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/24.0+np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/24.0)/2.0, test.performance['dataset3']['total']['mean_error'], places=3)

        test.analyzeModel('dataset3', prediction_samples=5, batch_size=4)
        A =[[[[11.0]], [[12.0]], [[13.0]], [[14.0]]],
                                   [[[12.0]], [[13.0]], [[14.0]], [[15.0]]],
                                   [[[13.0]], [[14.0]], [[15.0]], [[16.0]]],
                                   [[[14.0]], [[15.0]], [[16.0]], [[17.0]]],
                                   [[[15.0]], [[16.0]], [[17.0]], [[18.0]]],
                                   [[[16.0]], [[17.0]], [[18.0]], [[19.0]]]]
        B =[[[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
                                   [[[4.0]], [[8.0]], [[12.0]], [[16.0]]],
                                   [[[8.0]], [[16.0]], [[24.0]], [[32.0]]],
                                   [[[16.0]], [[32.0]], [[48.0]], [[64.0]]],
                                   [[[32.0]], [[64.0]], [[96.0]], [[128.0]]],
                                   [[[64.0]], [[128.0]], [[192.0]], [[256.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
                                   [[[20.0]], [[30.0]], [[40.0]], [[50.0]]],
                                   [[[30.0]], [[40.0]], [[50.0]], [[60.0]]],
                                   [[[40.0]], [[50.0]], [[60.0]], [[70.0]]],
                                   [[[50.0]], [[60.0]], [[70.0]], [[80.0]]],
                                   [[[60.0]], [[70.0]], [[80.0]], [[90.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset3']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset3']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/24.0, test.performance['dataset3']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/24.0, test.performance['dataset3']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/24.0+np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/24.0)/2.0, test.performance['dataset3']['total']['mean_error'], places=3)

        test.analyzeModel('dataset3', prediction_samples=4, batch_size=6)
        A =[[[[11.0]], [[12.0]], [[13.0]], [[14.0]], [[15.0]], [[16.0]]],
                                   [[[12.0]], [[13.0]], [[14.0]], [[15.0]], [[16.0]], [[17.0]]],
                                   [[[13.0]], [[14.0]], [[15.0]], [[16.0]], [[17.0]], [[18.0]]],
                                   [[[14.0]], [[15.0]], [[16.0]], [[17.0]], [[18.0]], [[19.0]]],
                                   [[[15.0]], [[16.0]], [[17.0]], [[18.0]], [[19.0]], [[20.0]]]]
        B =[[[[2.0]], [[4.0]], [[6.0]], [[8.0]], [[10.0]], [[12.0]]],
                                   [[[4.0]], [[8.0]], [[12.0]], [[16.0]], [[20.0]], [[24.0]]],
                                   [[[8.0]], [[16.0]], [[24.0]], [[32.0]], [[40.0]], [[48.0]]],
                                   [[[16.0]], [[32.0]], [[48.0]], [[64.0]], [[80.0]], [[96.0]]],
                                   [[[32.0]], [[64.0]], [[96.0]], [[128.0]], [[160.0]], [[192.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]]],
                                   [[[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]], [[70.0]]],
                                   [[[30.0]], [[40.0]], [[50.0]], [[60.0]], [[70.0]], [[80.0]]],
                                   [[[40.0]], [[50.0]], [[60.0]], [[70.0]], [[80.0]], [[90.0]]],
                                   [[[50.0]], [[60.0]], [[70.0]], [[80.0]], [[90.0]], [[100.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset3']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset3']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/30.0, test.performance['dataset3']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/30.0, test.performance['dataset3']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/30.0+np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/30.0)/2.0, test.performance['dataset3']['total']['mean_error'], places=3)

        dataset = {'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset4', source=dataset)
        test.trainModel(dataset='dataset4', prediction_samples=-1)  #TODO FIX
        test.analyzeModel('dataset4', prediction_samples=-1) #TODO FIX


    def test_analysis_results_closed_loop(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', sw=1, values=[[2]])
        output1 = Output('out', Fir(W=a)(input1.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model', output1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'out1': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   'out2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        test.loadData(name='dataset', source=dataset)

        test.analyzeModel('dataset', closed_loop={'in1': 'out'}, prediction_samples=5, batch_size=2)
        A = [[[[11.0]], [[12.0]], [[13.0]], [[14.0]]],
             [[[12.0]], [[13.0]], [[14.0]], [[15.0]]],
             [[[13.0]], [[14.0]], [[15.0]], [[16.0]]],
             [[[14.0]], [[15.0]], [[16.0]], [[17.0]]],
             [[[15.0]], [[16.0]], [[17.0]], [[18.0]]],
             [[[16.0]], [[17.0]], [[18.0]], [[19.0]]]]
        B = [[[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
             [[[4.0]], [[8.0]], [[12.0]], [[16.0]]],
             [[[8.0]], [[16.0]], [[24.0]], [[32.0]]],
             [[[16.0]], [[32.0]], [[48.0]], [[64.0]]],
             [[[32.0]], [[64.0]], [[96.0]], [[128.0]]],
             [[[64.0]], [[128.0]], [[192.0]], [[256.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
             [[[20.0]], [[30.0]], [[40.0]], [[50.0]]],
             [[[30.0]], [[40.0]], [[50.0]], [[60.0]]],
             [[[40.0]], [[50.0]], [[60.0]], [[70.0]]],
             [[[50.0]], [[60.0]], [[70.0]], [[80.0]]],
             [[[60.0]], [[70.0]], [[80.0]], [[90.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 24.0,
                               test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten() - np.array(B).flatten()) ** 2) / 24.0,
                               test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 24.0 + np.sum(
            (np.array(C).flatten() - np.array(B).flatten()) ** 2) / 24.0) / 2.0,
                               test.performance['dataset']['total']['mean_error'], places=3)

        test.analyzeModel('dataset', closed_loop={'in1': 'out'}, prediction_samples=5, batch_size=4)
        A = [[[[11.0]], [[12.0]], [[13.0]], [[14.0]]],
             [[[12.0]], [[13.0]], [[14.0]], [[15.0]]],
             [[[13.0]], [[14.0]], [[15.0]], [[16.0]]],
             [[[14.0]], [[15.0]], [[16.0]], [[17.0]]],
             [[[15.0]], [[16.0]], [[17.0]], [[18.0]]],
             [[[16.0]], [[17.0]], [[18.0]], [[19.0]]]]
        B = [[[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
             [[[4.0]], [[8.0]], [[12.0]], [[16.0]]],
             [[[8.0]], [[16.0]], [[24.0]], [[32.0]]],
             [[[16.0]], [[32.0]], [[48.0]], [[64.0]]],
             [[[32.0]], [[64.0]], [[96.0]], [[128.0]]],
             [[[64.0]], [[128.0]], [[192.0]], [[256.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
             [[[20.0]], [[30.0]], [[40.0]], [[50.0]]],
             [[[30.0]], [[40.0]], [[50.0]], [[60.0]]],
             [[[40.0]], [[50.0]], [[60.0]], [[70.0]]],
             [[[50.0]], [[60.0]], [[70.0]], [[80.0]]],
             [[[60.0]], [[70.0]], [[80.0]], [[90.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 24.0,
                               test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten() - np.array(B).flatten()) ** 2) / 24.0,
                               test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 24.0 + np.sum(
            (np.array(C).flatten() - np.array(B).flatten()) ** 2) / 24.0) / 2.0,
                               test.performance['dataset']['total']['mean_error'], places=3)

        test.analyzeModel('dataset', closed_loop={'in1': 'out'}, prediction_samples=4, batch_size=6)
        A = [[[[11.0]], [[12.0]], [[13.0]], [[14.0]], [[15.0]], [[16.0]]],
             [[[12.0]], [[13.0]], [[14.0]], [[15.0]], [[16.0]], [[17.0]]],
             [[[13.0]], [[14.0]], [[15.0]], [[16.0]], [[17.0]], [[18.0]]],
             [[[14.0]], [[15.0]], [[16.0]], [[17.0]], [[18.0]], [[19.0]]],
             [[[15.0]], [[16.0]], [[17.0]], [[18.0]], [[19.0]], [[20.0]]]]
        B = [[[[2.0]], [[4.0]], [[6.0]], [[8.0]], [[10.0]], [[12.0]]],
             [[[4.0]], [[8.0]], [[12.0]], [[16.0]], [[20.0]], [[24.0]]],
             [[[8.0]], [[16.0]], [[24.0]], [[32.0]], [[40.0]], [[48.0]]],
             [[[16.0]], [[32.0]], [[48.0]], [[64.0]], [[80.0]], [[96.0]]],
             [[[32.0]], [[64.0]], [[96.0]], [[128.0]], [[160.0]], [[192.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]]],
             [[[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]], [[70.0]]],
             [[[30.0]], [[40.0]], [[50.0]], [[60.0]], [[70.0]], [[80.0]]],
             [[[40.0]], [[50.0]], [[60.0]], [[70.0]], [[80.0]], [[90.0]]],
             [[[50.0]], [[60.0]], [[70.0]], [[80.0]], [[90.0]], [[100.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 30.0,
                               test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten() - np.array(B).flatten()) ** 2) / 30.0,
                               test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 30.0 + np.sum(
            (np.array(C).flatten() - np.array(B).flatten()) ** 2) / 30.0) / 2.0,
                               test.performance['dataset']['total']['mean_error'], places=3)
